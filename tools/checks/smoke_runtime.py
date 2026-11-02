#!/usr/bin/env python3
"""Exercise the real client transport against a dedicated running server.

This resets the server session. It uses generated WAV input and needs no API
key, microphone, speaker, or robot. --expect-motion additionally requires a
server started with --enable_motion and matching client preprocessing exports.
"""
import argparse
import asyncio
import base64
import csv
import io
import json
import math
import os
from pathlib import Path
import sys
import tempfile
import time


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--server", default="http://localhost:32558")
    parser.add_argument("--expect-motion", action="store_true")
    parser.add_argument(
        "--with-text",
        action="store_true",
        help="After the audio check, wait for a have_text=true motion (requires --expect-motion).",
    )
    parser.add_argument("--timeout", type=float, default=120)
    args = parser.parse_args()
    if args.with_text and not args.expect_motion:
        parser.error("--with-text requires --expect-motion")
    return args


def _assert_finite_raw_csv(path: Path) -> None:
    """Check every numeric cell in a client raw-motion CSV for finiteness."""
    with path.open("r", newline="") as csv_file:
        rows = csv.reader(csv_file)
        try:
            header = next(rows)
        except StopIteration as exc:
            raise AssertionError(f"Raw motion CSV is empty: {path}") from exc
        if not header:
            raise AssertionError(f"Raw motion CSV has no columns: {path}")

        data_rows = 0
        for row_number, row in enumerate(rows, start=2):
            if not row:
                continue
            if len(row) != len(header):
                raise AssertionError(
                    f"Raw motion CSV row {row_number} has {len(row)} columns; expected {len(header)}: {path}"
                )
            for column, value in zip(header, row):
                try:
                    numeric_value = float(value)
                except ValueError as exc:
                    raise AssertionError(
                        f"Raw motion CSV contains non-numeric {column!r} at row {row_number}: {path}"
                    ) from exc
                if not math.isfinite(numeric_value):
                    raise AssertionError(
                        f"Raw motion CSV contains non-finite {column!r} at row {row_number}: {path}"
                    )
            data_rows += 1
        if data_rows == 0:
            raise AssertionError(f"Raw motion CSV has no data rows: {path}")


def _assert_motion_array(audio_id, motion) -> None:
    """Validate the 31-channel motion emitted by the server and client."""
    assert motion.ndim == 2 and motion.shape[1] == 31, (audio_id, motion.shape)
    assert np_isfinite(motion), audio_id


def np_isfinite(array) -> bool:
    """Keep NumPy confined to the smoke's runtime helpers."""
    import numpy as np

    return bool(np.isfinite(array).all())


async def _wait_for_motion_artifacts(client, audio_id, deadline) -> None:
    """Wait for the real client listener to inverse-transform one motion id."""
    import numpy as np

    motion_path = Path(client.local_motion_dir) / f"motion_{audio_id}.npy"
    raw_path = Path(client.local_motion_dir) / f"raw_motion_{audio_id}.csv"
    while True:
        if motion_path.is_file() and raw_path.is_file():
            motion = np.load(motion_path, allow_pickle=False)
            _assert_motion_array(audio_id, motion)
            _assert_finite_raw_csv(raw_path)
            return
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError(
                f"Client did not inverse-transform motion audio_id={audio_id} before the smoke timeout"
            )
        await asyncio.sleep(min(0.1, remaining))


async def _wait_for_motion_packet(
    ws,
    client,
    captured_ids,
    deadline,
    min_audio_id=None,
    allow_text_before_min=False,
):
    """Read motion packets until one at/after ``min_audio_id`` is available."""
    import json

    while True:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError("No motion packet arrived before the smoke timeout")
        raw_packet = await asyncio.wait_for(ws.recv(), timeout=remaining)
        packet = json.loads(raw_packet)
        if packet.get("type") != "motion":
            continue
        data = packet.get("data") or {}
        if data.get("id") != client.UID:
            continue
        motion_b64 = data.get("motion_b64")
        audio_id = data.get("audio_id")
        if not motion_b64 or audio_id is None:
            continue
        motion = client._decode_motion_b64(motion_b64)
        _assert_motion_array(audio_id, motion)
        audio_id = int(audio_id)
        captured_ids.add(audio_id)
        if (
            min_audio_id is None
            or audio_id >= min_audio_id
            or (allow_text_before_min and data.get("have_text") is True)
        ):
            return data


async def _wait_for_interruption(ws, client, captured_ids, deadline) -> None:
    """Drain motion packets while waiting for the interruption broadcast."""
    while True:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError("No interruption broadcast arrived before the smoke timeout")
        raw_packet = await asyncio.wait_for(ws.recv(), timeout=remaining)
        packet = json.loads(raw_packet)
        if packet.get("type") == "motion":
            data = packet.get("data") or {}
            if data.get("id") != client.UID or not data.get("motion_b64"):
                continue
            audio_id = data.get("audio_id")
            if audio_id is None:
                continue
            motion = client._decode_motion_b64(data["motion_b64"])
            _assert_motion_array(audio_id, motion)
            captured_ids.add(int(audio_id))
            continue
        if packet.get("type") != "interruption":
            continue
        data = packet.get("data") or {}
        if data.get("id") != client.UID:
            continue
        interruption = data.get("interruption") or {}
        assert interruption.get("sequence") == 1, interruption
        return


def _assert_all_raw_csv_finite(client) -> None:
    """Validate every raw CSV emitted in this smoke's client session."""
    raw_files = sorted(Path(client.local_motion_dir).glob("raw_motion_*.csv"))
    assert raw_files, "No raw motion CSV files were emitted"
    for path in raw_files:
        _assert_finite_raw_csv(path)


async def smoke(args, client):
    import numpy as np
    import soundfile as sf
    import websockets

    responses = []

    def check_response(response, **kwargs):
        responses.append(response)
        response.raise_for_status()

    client._local_session.hooks["response"].append(check_response)
    health = client._local_session.get(f"{client.SERVER}/api/v1/health", timeout=5)
    assert health.json()["status"] == "OK"
    assert client.synchronize_time_with_server(), "Time synchronization failed"
    client.prepare_local_dirs()
    ws_url = client._build_ws_url(client.SERVER, "/ws/motion", client.UID)
    listener = None
    try:
        async with websockets.connect(ws_url, max_size=None, proxy=None) as ws:
            captured_motion_ids = set()
            if args.expect_motion:
                client._load_motion_assets()
                listener = asyncio.create_task(client._motion_ws_client(ws_url, client.UID))
                # Both subscribers must be connected before audio reaches the server.
                await asyncio.sleep(0.5)

            audio_b64 = {}
            for rate in (16000, 24000):
                buffer = io.BytesIO()
                samples = np.zeros(rate * 2, dtype=np.float32)
                sf.write(buffer, samples, rate, format="WAV", subtype="PCM_16")
                audio_b64[rate] = base64.b64encode(buffer.getvalue()).decode()

            def send_audio_pair(first_index, turn):
                for index, (speaker, rate) in enumerate((
                    ("user", 16000),
                    ("assistant", 24000),
                ), first_index):
                    client.send_audio_chunk(
                        audio_b64[rate], speaker, index, turn, rate, (index - 1) * 2000
                    )

            client.send_turn_complete(99)  # An empty turn must be harmless.
            send_audio_pair(1, 1)
            client.send_turn_complete(1)
            latest_audio_index = 2
            if args.expect_motion:
                deadline = time.monotonic() + args.timeout
                packet = await _wait_for_motion_packet(
                    ws, client, captured_motion_ids, deadline
                )
                audio_id = int(packet["audio_id"])
                await _wait_for_motion_artifacts(client, audio_id, deadline)
                print(f"Motion received and inverse-transformed: audio_id={audio_id}")

            if args.with_text:
                slow_thinking_payload = {
                    "mode": "reactive",
                    "analysis": None,
                    "clip_texts": {
                        "intent": "wave hello",
                        "pose": "raise both hands",
                        "motion": "perform a friendly wave",
                    },
                    "motivation_score": 4,
                    "intention_score": 4,
                    "proactive_level": "L2_proactive",
                    "proactive_trigger": "runtime smoke",
                    "locomotion": [],
                    "emotion": "supportive",
                    "should_interrupt_user": False,
                    "speech_prefill": "",
                    "confidence": "high",
                    "motion_length": 4,
                    "timestamp_ms": int(time.time() * 1000),
                }
                text_request = {
                    "messages": [{
                        "role": "system",
                        "content": [{
                            "type": "slow_thinking",
                            "slow_thinking": slow_thinking_payload,
                        }],
                    }],
                }
                text_response = client._local_session.post(
                    f"{client.SERVER}/api/v1/stream",
                    headers=client.HEADERS,
                    json=text_request,
                    timeout=10,
                )
                text_response.raise_for_status()
                assert text_response.json()["choices"]["finish_reason"] == "slow_thinking"

                # There is no public CLIP-ready endpoint or text-ready WS
                # event.  Replay one complete audio window at a time and use
                # the server's have_text flag as the readiness signal.  Each
                # retry waits for its motion result before sending another
                # window, so this does not race or queue an unbounded backlog.
                text_deadline = time.monotonic() + args.timeout
                text_index = 3
                while True:
                    send_audio_pair(text_index, 2)
                    latest_audio_index = text_index + 1
                    packet = await _wait_for_motion_packet(
                        ws,
                        client,
                        captured_motion_ids,
                        text_deadline,
                        min_audio_id=text_index,
                        # The initial pair's second window may finish after
                        # the text request and consume the newly ready CLIP
                        # result; accept that packet as the text proof.
                        allow_text_before_min=True,
                    )
                    if packet.get("have_text") is True:
                        text_audio_id = int(packet["audio_id"])
                        await _wait_for_motion_artifacts(client, text_audio_id, text_deadline)
                        print(f"Text-conditioned motion received: audio_id={text_audio_id}")
                        break
                    print(
                        f"Motion audio_id={packet.get('audio_id')} was not text-conditioned yet; "
                        "waiting for the next CLIP poll window."
                    )
                    text_index += 2
                    if time.monotonic() >= text_deadline:
                        raise TimeoutError("No have_text=true motion arrived before the text smoke timeout")
                client.send_turn_complete(2)

            event = dict(type="manual", source_turn=1, target_turn=2,
                         audio_position_ms=0, timestamp_ms=int(time.time() * 1000),
                         sequence=1, current_audio_id=latest_audio_index)
            response = client._local_session.post(f"{client.SERVER}/api/v1/interrupt",
                                                  headers=client.HEADERS, json=event, timeout=5)
            assert response.json()["status"] == "ok", response.text
            await _wait_for_interruption(
                ws, client, captured_motion_ids, time.monotonic() + args.timeout
            )
            reset_response = client._local_session.post(
                f"{client.SERVER}/api/v1/set_zero_time", timeout=5
            )
            reset_response.raise_for_status()
            response = client._local_session.post(f"{client.SERVER}/api/v1/interrupt",
                                                  headers=client.HEADERS, json=event, timeout=5)
            assert response.json()["status"] == "ok", response.text
            if args.expect_motion:
                _assert_all_raw_csv_finite(client)
        assert responses and all(r.ok for r in responses), [(r.url, r.status_code) for r in responses]
        print("Client/server transport passed: clock, WAV upload, empty/completed turns, interruption broadcast and session restart.")
        print(f"Client artifacts: {client.local_session_dir}")
    finally:
        client.motion_stop_event.set()
        if listener:
            listener.cancel()
            await asyncio.gather(listener, return_exceptions=True)
        client._local_session.close()


def main():
    args = parse_args()
    os.environ["PROACT_SERVER_URL"] = args.server
    os.environ["PROACT_CLIENT_ID"] = "runtime-smoke"
    os.environ["PROACT_MEDIA_DIR"] = tempfile.mkdtemp(prefix="proact-smoke-")
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src/client"))
    import realtime_client
    asyncio.run(smoke(args, realtime_client))


if __name__ == "__main__":
    main()
