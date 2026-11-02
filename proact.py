#!/usr/bin/env python3
"""Run ProAct inference, training, data preparation and benchmark commands."""

import argparse
import os
from pathlib import Path
import shlex
import subprocess
import sys

ROOT = Path(__file__).resolve().parent


def commands_for(route, processes):
    python = sys.executable
    distributed = [python, "-m", "torch.distributed.run", "--standalone", f"--nproc_per_node={processes}"]
    direct = {
        "interact": [python, str(ROOT / "src/client/realtime_client.py")],
        "prepare-seg": [python, str(ROOT / "src/motion/data_process/prepare_seg_data.py"),
                        str(ROOT / "configs/preprocessing/seg.yaml")],
        "prepare-audio": [python, str(ROOT / "src/motion/data_process/prepare_mfcc_data.py"),
                          str(ROOT / "configs/preprocessing/audio.yaml")],
        "benchmark-validate": [python, str(ROOT / "tools/proactbench/scripts/validate_benchmark.py")],
        "benchmark-evaluate": [python, str(ROOT / "tools/proactbench/scripts/evaluate_predictions.py")],
        "benchmark-protocol": [python, str(ROOT / "tools/proactbench/scripts/run_protocol_smoke.py")],
        "validate-gpu": [python, str(ROOT / "tools/checks/validate_gpu_runtime.py")],
    }
    if route == "serve":
        return [(distributed + [str(ROOT / "src/server/motion_server.py"), "--enable_motion",
                                "--mode", "controlnet", "--output_save_every_second", "1",
                                "--turn_complete_padding", "2"], ROOT)]
    if route in {"train-backbone", "train-controlnet"}:
        configuration = route.removeprefix("train-")
        command = distributed + [str(ROOT / "src/motion/train.py"), "--train_args_yaml",
                                 str(ROOT / f"configs/training/{configuration}.yaml")]
        if configuration == "controlnet":
            command.append("--controlnet")
        return [(command, ROOT)]
    if route == "retarget":
        return [([python, "-m", "retarget"], ROOT)]
    if route == "check":
        return [
            ([python, "-m", "pytest", "tools/tests", "-q"], ROOT),
            ([python, "-m", "unittest", "discover", "-s", "tools/proactbench/tests", "-q"], ROOT),
            (direct["benchmark-validate"], ROOT),
        ]
    return [(direct[route], ROOT)]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true", help="Print commands without starting them")
    parser.add_argument("--processes", type=int, default=1, help="GPU processes for serve/train; default: 1")
    subcommands = parser.add_subparsers(dest="command", required=True)
    for name, description in {
        "serve": "Start the trained motion server",
        "interact": "Start the realtime audio/video client",
        "retarget": "Convert BVH to G1 in the retargeting environment",
        "check": "Run source, runtime and benchmark offline checks",
        "validate-gpu": "Validate trained GPU inference and client transport",
    }.items():
        subcommands.add_parser(name, help=description, add_help=name == "check").set_defaults(route=name)
    groups = {
        "prepare": {"seg": "Prepare per-clip SeG features", "audio": "Prepare audio/motion features"},
        "train": {"backbone": "Train the audio backbone", "controlnet": "Train the semantic ControlNet"},
        "benchmark": {"validate": "Validate the 40-session dataset",
                      "evaluate": "Score saved cognitive predictions", "protocol": "Check canonical-server transport"},
    }
    for group, entries in groups.items():
        group_parser = subcommands.add_parser(group)
        variants = group_parser.add_subparsers(dest="target", required=True)
        for name, description in entries.items():
            variant = variants.add_parser(name, help=description, add_help=group == "prepare")
            variant.set_defaults(route=f"{group}-{name}")
            if group == "prepare":
                variant.add_argument("--config", type=Path, help="Use another preparation YAML")
    arguments, forwarded = parser.parse_known_args()
    if arguments.processes < 1:
        parser.error("--processes must be positive")
    if (arguments.route == "check" or arguments.route.startswith("prepare-")) and forwarded:
        parser.error("Unexpected additional arguments for this command")
    environment = dict(os.environ)
    environment["PYTHONPATH"] = os.pathsep.join(filter(None, [
        str(ROOT / "src"), str(ROOT / "src/motion"), str(ROOT / "tools/retargeting"), environment.get("PYTHONPATH"),
    ]))
    for command, directory in commands_for(arguments.route, arguments.processes):
        command = command + forwarded
        if getattr(arguments, "config", None):
            command[-1] = str(arguments.config.resolve())
        if arguments.dry_run:
            print(f"[{directory.relative_to(ROOT) or '.'}] {shlex.join(command)}")
            continue
        result = subprocess.run(command, cwd=directory, env=environment)
        if result.returncode:
            return result.returncode
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
