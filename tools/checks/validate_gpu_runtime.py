#!/usr/bin/env python3
"""Validate trained GPU motion through the real server and client transport."""

import argparse
import csv
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import shutil
import signal
import socket
import subprocess
import sys
import threading
import time
from urllib.error import URLError
from urllib.request import ProxyHandler, build_opener

ROOT = Path(__file__).resolve().parents[2]
ASSET_KEYS = (
    "MOTION_BASE_CHECKPOINT",
    "MOTION_CHECKPOINT",
    "MOTION_CONTROLNET_DATA_ROOT",
    "MOTION_AUDIO_DATA_ROOT",
    "MOTION_CLIP_MODEL",
    "PROACT_DATA_PIPE",
    "PROACT_OUTPUT_SCALER",
    "PROACT_TEXT_DATA_PIPE",
    "PROACT_TEXT_OUTPUT_SCALER",
)
RELEASE_CHECKSUMS = {
    "MOTION_BASE_CHECKPOINT": "7fe87ab26e09c3f1f227aeee783666d01576790724c41a15bb3dc81d156646a6",
    "MOTION_CHECKPOINT": "cbef5122b44f4d37003005a94fa23344c1540743a16de9d475b7f9f3278bdcb8",
    "PROACT_TEXT_DATA_PIPE": "93bc1dff82362598584b1bae9ff332584017a24f936d62e26e7194bc676f05d0",
    "PROACT_TEXT_OUTPUT_SCALER": "f5c1aad36ff2a4bde4cb96ea0a53d5f2a865f39b0adbdd666b1b7d16bd58959b",
    "PROACT_DATA_PIPE": "3ec0ea067e5c96cb84ba775f815f1d39d6edc60a9f72919f8d92ce1540befd0d",
    "PROACT_OUTPUT_SCALER": "eb023637181a504f97024ffa35fb7d9ac6e2d98b9924f0e76b08bc559a3eb90f",
}


def write_json(destination, value):
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2), encoding="utf-8")
    temporary.replace(destination)


def checksum(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def available_port():
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        return listener.getsockname()[1]


def worker(arguments):
    import torch
    import torch.distributed as distributed
    import uvicorn

    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("Select exactly one available CUDA GPU for validation")
    torch.cuda.reset_peak_memory_stats()
    sys.argv = [
        "motion_server.py", "--port", str(arguments.port), "--enable_motion",
        "--mode", "controlnet", "--output_save_every_second", "1",
        "--turn_complete_padding", "2",
    ]
    specification = importlib.util.spec_from_file_location(
        "proact_gpu_validation_server", ROOT / "src/server" / "motion_server.py"
    )
    server = importlib.util.module_from_spec(specification)
    sys.modules[specification.name] = server
    specification.loader.exec_module(server)
    original_generation = server.server_generate_diffusion_with_recurrent_text_instruction
    measurements = []
    measurement_lock = threading.Lock()
    information = {
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "device": torch.cuda.get_device_name(0),
        "visible_gpu": os.environ["CUDA_VISIBLE_DEVICES"],
        "sample_steps": server.trainer.config.inference.sample_steps,
        "measurements": measurements,
        "shutdown_complete": False,
    }

    def measured_generation(**parameters):
        torch.cuda.synchronize()
        started = time.perf_counter()
        result = original_generation(**parameters)
        torch.cuda.synchronize()
        motion = result[1]
        frames = int(motion.shape[-2])
        measurement = {
            "have_text": bool(parameters.get("have_text")),
            "seconds": time.perf_counter() - started,
            "shape": list(motion.shape),
            "finite": bool(torch.isfinite(motion).all().item()),
            "motion_seconds": frames / server.trainer.config.data.framerate,
            "sample_steps": parameters["sample_steps"],
        }
        with measurement_lock:
            measurements.append(measurement)
            information["peak_gpu_memory_mib"] = torch.cuda.max_memory_allocated() / 1024 ** 2
            write_json(arguments.output / "gpu-worker.json", information)
        return result

    server.server_generate_diffusion_with_recurrent_text_instruction = measured_generation
    write_json(arguments.output / "gpu-worker.json", information)
    try:
        uvicorn.run(server.app, host="127.0.0.1", port=arguments.port, log_level="info")
    finally:
        information["shutdown_complete"] = True
        information["peak_gpu_memory_mib"] = torch.cuda.max_memory_allocated() / 1024 ** 2
        write_json(arguments.output / "gpu-worker.json", information)
        if distributed.is_initialized():
            distributed.destroy_process_group()


def inspect_artifacts(directory):
    import numpy as np

    arrays = []
    for path in sorted(directory.rglob("motion_*.npy")):
        motion = np.load(path, allow_pickle=False)
        finite = bool(np.isfinite(motion).all())
        if motion.ndim != 2 or motion.shape[1] != 31 or not finite:
            raise RuntimeError(f"Invalid decoded motion: {path.name}")
        arrays.append({"file": str(path.relative_to(directory)), "shape": list(motion.shape), "finite": finite})
    csv_files = []
    for path in sorted(directory.rglob("raw_motion_*.csv")):
        with path.open(newline="") as stream:
            rows = csv.reader(stream)
            header = next(rows)
            row_count = 0
            for row in rows:
                if not row:
                    continue
                if len(row) != len(header) or not all(math.isfinite(float(value)) for value in row):
                    raise RuntimeError(f"Invalid inverse-transform CSV: {path.name}")
                row_count += 1
        if not row_count:
            raise RuntimeError(f"Empty inverse-transform CSV: {path.name}")
        csv_files.append({"file": str(path.relative_to(directory)), "rows": row_count, "columns": len(header), "finite": True})
    if not arrays or not csv_files:
        raise RuntimeError("The client did not retain decoded motion and inverse-transform CSV")
    return {"motion_arrays": arrays, "inverse_transform_csv": csv_files}


def validate(arguments):
    environment = dict(os.environ)
    if arguments.asset_config:
        configured = json.loads(arguments.asset_config.read_text())
        environment.update({key: str(configured[key]) for key in ASSET_KEYS if key in configured})
    environment.setdefault("MOTION_AUDIO_DATA_ROOT", str(ROOT / "src/assets" / "preprocessing" / "audio"))
    audio_root = Path(environment["MOTION_AUDIO_DATA_ROOT"])
    environment.setdefault("PROACT_DATA_PIPE", str(audio_root / "data_pipe_30fps.sav"))
    environment.setdefault("PROACT_OUTPUT_SCALER", str(audio_root / "output_scaler.sav"))
    missing = [key for key in ASSET_KEYS if not environment.get(key) or not Path(environment[key]).exists()]
    if missing:
        raise RuntimeError("Missing local assets: " + ", ".join(missing))
    if not arguments.gpu or "," in arguments.gpu:
        raise RuntimeError("Pass --gpu with one idle device index or GPU UUID")
    arguments.output.mkdir(parents=True, exist_ok=True)
    temporary_root = arguments.output / "tmp"
    temporary_root.mkdir(exist_ok=True)
    assets = {}
    for key in ASSET_KEYS:
        path = Path(environment[key])
        if path.is_file():
            digest = checksum(path)
            if key in RELEASE_CHECKSUMS and digest != RELEASE_CHECKSUMS[key]:
                raise RuntimeError(f"{key} differs from the published trained release")
            assets[key] = {"sha256": digest, "bytes": path.stat().st_size}
    text_scaler = Path(environment["MOTION_CONTROLNET_DATA_ROOT"]) / "input_scaler.sav"
    assets["TEXT_INPUT_SCALER"] = {"sha256": checksum(text_scaler), "bytes": text_scaler.stat().st_size}
    if assets["TEXT_INPUT_SCALER"]["sha256"] != "d9bc0807edcd97591bf174229e3b2b2211dde12644116c543324b06bdfb61e76":
        raise RuntimeError("The text input scaler differs from the published data export")
    audio_scaler = Path(environment["MOTION_AUDIO_DATA_ROOT"]) / "input_scaler.sav"
    assets["AUDIO_INPUT_SCALER"] = {"sha256": checksum(audio_scaler), "bytes": audio_scaler.stat().st_size}
    if assets["AUDIO_INPUT_SCALER"]["sha256"] != "d14543752439bfea7fdee205f82a695ed29000c1fb847d5528f8302e8a7b9d14":
        raise RuntimeError("The audio input scaler differs from the bundled trained release")
    port = available_port()
    environment.update({
        "CUDA_VISIBLE_DEVICES": arguments.gpu,
        "RANK": "0", "LOCAL_RANK": "0", "WORLD_SIZE": "1",
        "MASTER_ADDR": "127.0.0.1", "MASTER_PORT": str(available_port()),
        "OPENAI_API_KEY": "offline-test",
        "MOTION_OUTPUT_DIR": str(arguments.output / "generated"),
        "MOTION_EXPERIMENT_DIR": str(arguments.output / "experiments"),
        "PROACT_MEDIA_DIR": str(arguments.output / "client-media"),
        "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
        "PYTHONDONTWRITEBYTECODE": "1", "PYTHONUNBUFFERED": "1",
        "OMP_NUM_THREADS": "2", "MKL_NUM_THREADS": "2", "TMPDIR": str(temporary_root),
    })
    opener = build_opener(ProxyHandler({}))
    server_origin = f"http://127.0.0.1:{port}"
    result = {
        "status": "running",
        "trained_release_checksums_verified": True,
        "assets": assets,
        "model_repository": "UsanoCoCr/ProAct",
        "model_revision": "b246d61db9f41502b6ed00b142e85461b2a2938c",
        "with_text": arguments.with_text,
        "live_model_api_exercised": False,
        "robot_hardware_exercised": False,
        "python": sys.version.split()[0],
    }
    write_json(arguments.output / "acceptance.json", result)
    print("Trained checkpoint hashes verified; starting the dedicated GPU worker.", flush=True)
    with (arguments.output / "server.log").open("w") as server_log:
        process = subprocess.Popen(
            [sys.executable, str(Path(__file__).resolve()), "--worker", "--port", str(port),
             "--output", str(arguments.output)],
            cwd=ROOT, env=environment, stdout=server_log, stderr=subprocess.STDOUT,
        )
        forced_stop = False
        try:
            deadline = time.monotonic() + arguments.startup_timeout
            while True:
                if process.poll() is not None:
                    raise RuntimeError(f"GPU server exited during startup; see {arguments.output / 'server.log'}")
                try:
                    with opener.open(server_origin + "/api/v1/health", timeout=2) as response:
                        if json.load(response).get("status") == "OK":
                            break
                except (URLError, TimeoutError):
                    pass
                if time.monotonic() >= deadline:
                    raise TimeoutError("The GPU server did not become ready before the startup timeout")
                time.sleep(0.25)
            print("GPU worker ready; exercising audio, text conditioning, interruption, and reset.", flush=True)
            command = [sys.executable, str(ROOT / "tools/checks" / "smoke_runtime.py"),
                       "--server", server_origin, "--expect-motion", "--timeout", str(arguments.timeout)]
            if arguments.with_text:
                command.append("--with-text")
            smoke = subprocess.run(command, cwd=ROOT, env=environment, capture_output=True,
                                   text=True, timeout=arguments.timeout * 3 + 30)
            (arguments.output / "client.log").write_text(smoke.stdout + "\n" + smoke.stderr)
            print(smoke.stdout, end="", flush=True)
            if smoke.returncode:
                raise RuntimeError(f"GPU client smoke failed; see {arguments.output / 'client.log'}")
            artifact_lines = [line for line in smoke.stdout.splitlines() if line.startswith("Client artifacts: ")]
            if not artifact_lines:
                raise RuntimeError("The smoke did not report retained client artifacts")
            artifact_root = Path(artifact_lines[-1].removeprefix("Client artifacts: ")).resolve()
            if not artifact_root.is_relative_to(temporary_root.resolve()):
                raise RuntimeError("Unexpected client artifact directory")
            retained = arguments.output / "client-artifacts"
            shutil.copytree(artifact_root, retained, dirs_exist_ok=True)
            result["artifacts"] = inspect_artifacts(retained)
            result["status"] = "passed"
        except Exception as error:
            result["status"] = "failed"
            result["error"] = str(error)
            raise
        finally:
            if process.poll() is None:
                process.send_signal(signal.SIGINT)
                try:
                    process.wait(timeout=25)
                except subprocess.TimeoutExpired:
                    forced_stop = True
                    process.kill()
                    process.wait(timeout=10)
            result["server_process_stopped"] = process.poll() is not None
            result["server_forced_stop"] = forced_stop
            worker_report = arguments.output / "gpu-worker.json"
            if worker_report.exists():
                result["gpu"] = json.loads(worker_report.read_text())
            write_json(arguments.output / "acceptance.json", result)
    gpu = result.get("gpu", {})
    measurements = gpu.get("measurements", [])
    if (
        forced_stop
        or not gpu.get("shutdown_complete")
        or not measurements
        or any(not measurement["finite"] or measurement["sample_steps"] != 50 for measurement in measurements)
        or not any(not measurement["have_text"] for measurement in measurements)
        or (arguments.with_text and not any(measurement["have_text"] for measurement in measurements))
    ):
        result["status"] = "failed"
        result["error"] = "GPU measurements or graceful shutdown did not satisfy the acceptance criteria"
        write_json(arguments.output / "acceptance.json", result)
        raise RuntimeError(result["error"])
    print(f"GPU acceptance passed: {arguments.output / 'acceptance.json'}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--asset-config", type=Path)
    parser.add_argument("--gpu", default=os.getenv("CUDA_VISIBLE_DEVICES"))
    parser.add_argument("--output", type=Path, default=ROOT / "runtime" / "gpu-validation")
    parser.add_argument("--timeout", type=float, default=120)
    parser.add_argument("--startup-timeout", type=float, default=180)
    parser.add_argument("--with-text", action="store_true")
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--port", type=int, default=0, help=argparse.SUPPRESS)
    arguments = parser.parse_args()
    arguments.output = arguments.output.resolve()
    if arguments.worker:
        worker(arguments)
    else:
        validate(arguments)


if __name__ == "__main__":
    main()

