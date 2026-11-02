"""Exercise the ProAct motion server transport using synthetic silence."""

import argparse
import asyncio
import base64
from contextlib import contextmanager
import io
import inspect
import json
from pathlib import Path
import socket
import subprocess
import sys
import tempfile
import time
from urllib.error import URLError
from urllib.parse import urlsplit, urlunsplit, urlencode
from urllib.request import Request, urlopen
import uuid
import wave

from benchmark import DEFAULT_DATA, load, transcripts, write_json


def request(server, route, uid=None, body=None, method="GET"):
    headers = {"Content-Type": "application/json"}
    if uid:
        headers["uid"] = uid
    data = json.dumps(body).encode() if body is not None else None
    with urlopen(
        Request(server.rstrip("/") + route, data=data, headers=headers, method=method),
        timeout=15,
    ) as response:
        return json.load(response)


def silence(seconds):
    stream = io.BytesIO()
    with wave.open(stream, "wb") as output:
        output.setnchannels(1)
        output.setsampwidth(2)
        output.setframerate(15360)
        output.writeframes(b"\x00\x00" * round(15360 * seconds))
    return base64.b64encode(stream.getvalue()).decode("ascii")


def websocket_url(server, uid):
    parsed = urlsplit(server)
    if (
        parsed.scheme not in ("http", "https")
        or parsed.path not in ("", "/")
        or parsed.query
        or parsed.fragment
        or parsed.username
        or parsed.password
    ):
        raise ValueError(
            "--server must be an HTTP(S) origin without credentials, paths, or query strings"
        )
    return urlunsplit(
        (
            "wss" if parsed.scheme == "https" else "ws",
            parsed.netloc,
            "/ws/motion",
            urlencode({"uid": uid}),
            "",
        )
    )


async def consume_until(ws, expected_type, uid, turn=None, sequence=None):
    deadline = asyncio.get_running_loop().time() + 30
    motions = 0
    while True:
        remaining = deadline - asyncio.get_running_loop().time()
        if remaining <= 0:
            raise TimeoutError(f"No {expected_type} acknowledgement")
        packet = json.loads(await asyncio.wait_for(ws.recv(), remaining))
        if packet.get("type") == "error":
            raise RuntimeError("Inference server reported a motion generation error")
        payload = packet.get("data", {})
        if payload.get("id") != uid:
            raise ValueError("WebSocket session ID mismatch")
        if packet.get("type") == "motion":
            # The release sends a base64-encoded NumPy container, not raw floats.
            encoded = base64.b64decode(payload["motion_b64"], validate=True)
            if not encoded.startswith(b"\x93NUMPY"):
                raise ValueError("Invalid motion container")
            motions += 1
        if packet.get("type") == expected_type:
            if turn is not None and payload.get("turn") != turn:
                raise ValueError("Turn acknowledgement mismatch")
            event = payload.get("interruption", payload)
            if sequence is not None and event.get("sequence") != sequence:
                raise ValueError("Interruption acknowledgement mismatch")
            return motions


async def run(server, dataset, seconds):
    # Imported lazily so dataset validation and offline scoring stay dependency-free.
    import websockets

    # Validate before making any request; auth material never belongs in this URL.
    websocket_url(server, "validation")
    health = await asyncio.to_thread(request, server, "/api/v1/health")
    if health.get("status") != "OK":
        raise ValueError("Server is not ready")
    wav = silence(seconds)
    results = []
    for session in transcripts(dataset):
        uid = "bench-" + uuid.uuid4().hex
        turns = motions = 0
        try:
            await asyncio.to_thread(
                request, server, "/api/v1/set_zero_time", uid, None, "POST"
            )
            websocket_options = {"max_size": 16 * 1024 * 1024, "open_timeout": 10}
            if "proxy" in inspect.signature(websockets.connect).parameters:
                websocket_options["proxy"] = None
            async with websockets.connect(websocket_url(server, uid), **websocket_options) as ws:
                for index, turn in enumerate(session["transcript"], 1):
                    speaker = "assistant" if turn["speaker"] == "agent" else "user"
                    body = {
                        "messages": [
                            {
                                "role": speaker,
                                "content": [
                                    {
                                        "type": "input_audio",
                                        "input_audio": {
                                            "data": wav,
                                            "format": "wav",
                                            "speaker": speaker,
                                            "sample_rate": 15360,
                                            "index": index,
                                            "turn": index,
                                            "offset_ms": round((index - 1) * seconds * 1000),
                                        },
                                    },
                                    {
                                        "type": "turn_complete",
                                        "turn_complete": {"turn": index},
                                    },
                                ],
                            }
                        ]
                    }
                    ack = await asyncio.to_thread(
                        request, server, "/api/v1/stream", uid, body, "POST"
                    )
                    if (
                        ack.get("id") != uid
                        or ack.get("choices", {}).get("content") != "success"
                        or ack.get("choices", {}).get("finish_reason") != f"turn_complete:{index}"
                    ):
                        raise ValueError("Unexpected HTTP acknowledgement")
                    turns += 1
                if health.get("interruption_support"):
                    await asyncio.to_thread(
                        request,
                        server,
                        "/api/v1/interrupt",
                        uid,
                        {
                            "type": "manual",
                            "source_turn": turns,
                            "target_turn": turns + 1,
                            "audio_position_ms": 0,
                            "timestamp_ms": int(time.time() * 1000),
                            "current_audio_id": turns,
                            "sequence": 1,
                        },
                        "POST",
                    )
                    motions += await consume_until(ws, "interruption", uid, sequence=1)
            results.append(
                {
                    "session_id": session["session_id"],
                    "audio_packets": turns,
                    "turn_acknowledgements": turns,
                    "motion_packets": motions,
                    "status": "passed",
                }
            )
        finally:
            await asyncio.to_thread(
                request, server, "/api/v1/set_zero_time", uid, None, "POST"
            )
    return {
        "kind": "protocol_smoke",
        "schema_version": "1.0.0",
        "dataset_id": dataset["dataset_id"],
        "dataset_source_sha256": dataset["source"]["sha256"],
        "health": health,
        "session_mode": "sequential_single_session",
        "turn_acknowledgement": "http_finish_reason",
        "synthetic_audio_seconds_per_turn": seconds,
        "sessions": results,
        "scope": "Transport checks using silence. No speech synthesis, visual replay, cognitive predictions, motion quality score, or benchmark accuracy.",
    }


@contextmanager
def local_server(root, python):
    root = Path(root).resolve()
    entrypoint = root / "src/server/motion_server.py"
    if not entrypoint.is_file():
        raise ValueError("--runtime-root must contain src/server/motion_server.py")
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    server = f"http://127.0.0.1:{port}"
    with tempfile.TemporaryFile(mode="w+b") as log:
        process = subprocess.Popen(
            [python, str(entrypoint), "--host", "127.0.0.1", "--port", str(port)],
            cwd=root,
            stdout=log,
            stderr=subprocess.STDOUT,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
        try:
            for _ in range(120):
                if process.poll() is not None:
                    # Logs can contain user-provided configuration; keep them out of reports.
                    raise RuntimeError(
                        f"Local server exited with status {process.returncode}; check inference dependencies"
                    )
                try:
                    if request(server, "/api/v1/health").get("status") == "OK":
                        break
                except (URLError, TimeoutError):
                    time.sleep(0.25)
            else:
                raise TimeoutError("Local inference server did not become ready")
            yield server
        finally:
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    target = parser.add_mutually_exclusive_group()
    target.add_argument(
        "--server", help="Existing server origin, e.g. http://127.0.0.1:32558"
    )
    target.add_argument(
        "--runtime-root",
        type=Path,
        help="Launch a temporary protocol-only server from this checkout",
    )
    parser.add_argument(
        "--python",
        default=sys.executable,
        help="Python executable for the temporary server",
    )
    parser.add_argument("--data", default=DEFAULT_DATA)
    parser.add_argument("--audio-seconds", type=float, default=0.5)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if not 0.5 <= args.audio_seconds <= 4:
        parser.error("--audio-seconds must be between 0.5 and 4")
    dataset = load(args.data)
    if not args.runtime_root and not args.server:
        candidate = Path(__file__).resolve().parents[3]
        if not (candidate / "src/server/motion_server.py").is_file():
            parser.error("Provide --runtime-root or --server")
        args.runtime_root = candidate
    if args.runtime_root:
        with local_server(args.runtime_root, args.python) as server:
            report = asyncio.run(run(server, dataset, args.audio_seconds))
    else:
        report = asyncio.run(run(args.server, dataset, args.audio_seconds))
    write_json(args.output, report)
    print(f"Passed {len(report['sessions'])} sessions; report type: protocol_smoke")


if __name__ == "__main__":
    main()
