"""Exercise service boundaries without weights, CUDA, or network access."""
import base64
import io
import importlib.util
from pathlib import Path
import sys
import threading
import time
import wave
from types import SimpleNamespace

import numpy as np
import pytest
from sklearn.preprocessing import StandardScaler
from fastapi.testclient import TestClient


@pytest.fixture(scope="module")
def server():
    from transformers import CLIPTextModel, CLIPTokenizer
    patcher = pytest.MonkeyPatch()
    def no_download(*args, **kwargs):
        raise AssertionError("A no-motion service must not load CLIP")
    patcher.setattr(CLIPTokenizer, "from_pretrained", no_download)
    patcher.setattr(CLIPTextModel, "from_pretrained", no_download)
    patcher.setattr(sys, "argv", ["motion_server.py"])
    path = Path(__file__).resolve().parents[2] / "src/server" / "motion_server.py"
    spec = importlib.util.spec_from_file_location("proact_test_server", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    try:
        spec.loader.exec_module(module)
        with TestClient(module.app) as client:
            yield module, client
    finally:
        patcher.undo()
        sys.modules.pop(spec.name, None)


def test_health_and_session_reset(server):
    module, client = server
    assert client.get("/api/v1/health").json()["status"] == "OK"
    assert client.get("/api/v1/time").json()["server_time"] > 0
    assert client.post("/api/v1/set_zero_time").json()["zero_point"] > 0
    assert module.trainer is None
    assert module._clip_text_encoder is None


@pytest.mark.parametrize("mode", ["controlnet", "mix"])
def test_warm_motion_pipeline_uses_audio_scaler_for_mode(server, tmp_path, monkeypatch, mode):
    module, _ = server
    data_root = tmp_path / "data"
    audio_root = tmp_path / "audio"
    data_root.mkdir()
    audio_root.mkdir()

    rng = np.random.default_rng(0)
    text_scaler = StandardScaler().fit(rng.normal(size=(8, 512)))
    audio_scaler = StandardScaler().fit(rng.normal(size=(8, 181)))
    mix_audio_scaler = StandardScaler().fit(rng.normal(size=(8, 362)))
    module.jl.dump(text_scaler, data_root / "input_scaler.sav")
    module.jl.dump(audio_scaler, audio_root / "input_scaler.sav")
    module.jl.dump(mix_audio_scaler, data_root / "input_scaler_audio.sav")

    fake_trainer = SimpleNamespace(config=SimpleNamespace(
        data=SimpleNamespace(data_root=str(data_root), framerate=30),
        audio=SimpleNamespace(audio_data_root=str(audio_root)),
        inference=SimpleNamespace(
            inference_sovler="fake",
            sample_steps=1,
            guidance_weight=1.0,
            eta=0.0,
        ),
    ))
    monkeypatch.setattr(module.args, "enable_motion", True)
    monkeypatch.setattr(module.args, "mode", mode)
    monkeypatch.setattr(module, "trainer", fake_trainer)
    monkeypatch.setattr(
        module,
        "_compute_channel_feats",
        lambda path, fps: np.ones((4, 181), dtype=np.float32),
    )

    captured = []

    def fake_generation(**kwargs):
        captured.append(kwargs)
        return None

    if mode == "controlnet":
        monkeypatch.setattr(module, "server_generate_diffusion_with_recurrent_text_instruction", fake_generation)
    else:
        monkeypatch.setattr(module, "server_generate_diffusion_with_text_instruction", fake_generation)

    module._warm_motion_pipeline()

    assert len(captured) == 1
    audio_features = captured[0]["audio_feats"]
    assert audio_features.shape == (1, 4, 362)
    assert np.isfinite(audio_features).all()


def _wav_b64(sample_rate=16000, frames=160):
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(b"\0\0" * frames)
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def test_empty_turn_complete_is_a_noop(server):
    module, client = server
    client.post("/api/v1/set_zero_time")
    response = client.post(
        "/api/v1/stream",
        headers={"uid": "empty-turn"},
        json={
            "messages": [{
                "role": "system",
                "content": [{"type": "turn_complete", "turn_complete": {"turn": 1}}],
            }],
        },
    )
    assert response.status_code == 200
    assert response.json()["choices"]["finish_reason"] == "turn_complete:1"
    assert module.stream_manager.completed_turns == {1}
    assert module.stream_manager.audio_id == 0


@pytest.mark.parametrize("first_index", [1, 7])
def test_new_session_restarts_audio_ids_and_start_time(server, first_index):
    module, client = server
    headers = {"uid": "session-reset"}
    payload = {
        "messages": [{
            "role": "user",
            "content": [{
                "type": "input_audio",
                "input_audio": {
                    "data": _wav_b64(),
                    "format": "wav",
                    "speaker": "user",
                    "index": first_index,
                    "turn": 1,
                    "sample_rate": 16000,
                },
            }],
        }],
    }

    first = client.post("/api/v1/stream", headers=headers, json=payload)
    assert first.status_code == 200
    assert first.json()["choices"]["finish_reason"] == f"audio:user:{first_index}"
    first_start_time = module.stream_manager.start_time
    assert module.stream_manager.audio_id == first_index
    assert first_start_time is not None

    time.sleep(0.001)
    assert client.post("/api/v1/set_zero_time").status_code == 200
    assert module.stream_manager.audio_id == 0
    assert module.stream_manager.start_time is None

    second = client.post("/api/v1/stream", headers=headers, json=payload)
    assert second.status_code == 200
    assert second.json()["choices"]["finish_reason"] == f"audio:user:{first_index}"
    assert module.stream_manager.audio_id == first_index
    assert module.stream_manager.start_time is not None
    assert module.stream_manager.start_time > first_start_time


def test_session_reset_clears_interruption_state(server):
    module, client = server
    headers = {"uid": "interruption-reset"}
    event = dict(
        type="manual",
        source_turn=1,
        target_turn=1,
        audio_position_ms=0,
        timestamp_ms=1,
        sequence=1,
        current_audio_id=1,
    )

    client.post("/api/v1/set_zero_time")
    assert client.post("/api/v1/interrupt", headers=headers, json=event).json()["status"] == "ok"
    assert module.interruption_handler.is_audio_interrupted(1)
    assert client.get("/api/v1/interruption_stats").json() == {
        "current_sequence": 1,
        "interrupted_audio_count": 2,
    }

    assert client.post("/api/v1/set_zero_time").status_code == 200
    assert client.get("/api/v1/interruption_stats").json() == {
        "current_sequence": 0,
        "interrupted_audio_count": 0,
    }
    assert not module.interruption_handler.is_audio_interrupted(1)
    assert client.post("/api/v1/interrupt", headers=headers, json=event).json()["status"] == "ok"


def test_interruption_before_first_audio_survives_session_initialization(server):
    module, client = server
    client.post("/api/v1/set_zero_time")
    headers = {"uid": "pre-audio-interruption"}
    event = dict(
        type="manual",
        source_turn=1,
        target_turn=1,
        audio_position_ms=0,
        timestamp_ms=1,
        sequence=1,
    )

    assert client.post("/api/v1/interrupt", headers=headers, json=event).json()["status"] == "ok"
    payload = {
        "messages": [{
            "role": "user",
            "content": [{
                "type": "input_audio",
                "input_audio": {
                    "data": _wav_b64(),
                    "format": "wav",
                    "speaker": "user",
                    "index": 1,
                    "turn": 1,
                    "sample_rate": 16000,
                },
            }],
        }],
    }
    response = client.post("/api/v1/stream", headers=headers, json=payload)
    assert response.status_code == 200
    assert response.json()["choices"]["finish_reason"] == "audio:user:1"
    assert module.interruption_handler.get_current_sequence() == 1


def test_session_reset_invalidates_queued_motion_task(server, tmp_path, monkeypatch):
    module, _ = server

    class CaptureExecutor:
        def __init__(self):
            self.calls = []

        def submit(self, function, *args, **kwargs):
            self.calls.append((function, args, kwargs))
            return object()

        def shutdown(self, **kwargs):
            return None

    handler = module.InterruptionHandler(module.logger)
    manager = module.MotionStreamManagerWithInterruption(
        enable_motion=True,
        port=module.args.port,
        window_seconds=1.0,
        interruption_handler=handler,
    )
    cache_dir = tmp_path / "audio_cache"
    cache_dir.mkdir()
    manager.uid = "stale-task"
    manager.session_dir = str(tmp_path)
    manager.audio_id = 1
    manager.start_audio_id = 1
    manager.window_buffers["user"]["data"] = np.zeros(1600, dtype=np.float32)
    manager.window_buffers["assistant"]["data"] = np.zeros(2400, dtype=np.float32)
    worker = CaptureExecutor()
    manager.motion_worker = worker
    manager._emit_windowed_motion_segments("stale-task")
    assert len(worker.calls) == 1

    published = []
    monkeypatch.setattr(module.motion_hub, "publish", lambda *args, **kwargs: published.append((args, kwargs)))
    function, call_args, call_kwargs = worker.calls[0]
    manager.reset_session()
    function(*call_args, **call_kwargs)
    assert published == []
    manager.motion_worker.shutdown(wait=False, cancel_futures=True)


def test_session_reset_discards_running_motion_result(server, tmp_path, monkeypatch):
    module, _ = server
    handler = module.InterruptionHandler(module.logger)
    manager = module.MotionStreamManagerWithInterruption(
        enable_motion=True,
        port=module.args.port,
        window_seconds=1.0,
        interruption_handler=handler,
    )
    manager.uid = "running-stale"
    manager.session_dir = str(tmp_path)

    user_path = tmp_path / "user.wav"
    agent_path = tmp_path / "agent.wav"
    module.soundfile.write(user_path, np.zeros(1536, dtype=np.float32), 15360)
    module.soundfile.write(agent_path, np.zeros(1536, dtype=np.float32), 15360)

    class FakeExtractor:
        def process_chunk(self, audio, return_overlap, drop_last_frame):
            features = np.ones((2, 1), dtype=np.float32)
            overlap = np.ones((1, 1), dtype=np.float32)
            return features, overlap

    class IdentityScaler:
        def transform(self, values):
            return values

    manager.streaming_extractor_user = FakeExtractor()
    manager.streaming_extractor_agent = FakeExtractor()
    manager.audio_input_scaler = IdentityScaler()
    manager.text_input_scaler = None
    fake_trainer = SimpleNamespace(config=SimpleNamespace(
        data=SimpleNamespace(data_root=str(tmp_path), framerate=30),
        audio=SimpleNamespace(audio_data_root=str(tmp_path)),
        inference=SimpleNamespace(
            inference_sovler="fake",
            sample_steps=1,
            guidance_weight=1.0,
            eta=0.0,
        ),
    ))
    monkeypatch.setattr(module, "trainer", fake_trainer)
    monkeypatch.setattr(module.jl, "load", lambda path: IdentityScaler())
    # This test exercises cancellation, independent of librosa's cold JIT cache.
    monkeypatch.setattr(module.librosa, "load", lambda *args, **kwargs: (np.zeros(1536, dtype=np.float32), 15360))

    generation_started = threading.Event()
    release_generation = threading.Event()

    def fake_generation(**kwargs):
        generation_started.set()
        assert release_generation.wait(timeout=5.0)
        return (
            module.torch.zeros((1, 1, 1)),
            module.torch.zeros((1, 1)),
            module.torch.zeros((1, 1, 1)),
        )

    monkeypatch.setattr(module, "server_generate_diffusion_with_recurrent_text_instruction", fake_generation)
    published = []
    monkeypatch.setattr(module.motion_hub, "publish", lambda *args, **kwargs: published.append((args, kwargs)))

    generation = threading.Thread(
        target=manager._generate_and_publish_motion,
        args=("running-stale", 1, str(user_path), str(agent_path), 1, "user", manager._session_generation),
    )
    generation.start()
    try:
        assert generation_started.wait(timeout=5.0)
        manager.reset_session()
    finally:
        release_generation.set()
        generation.join(timeout=5.0)
        manager.motion_worker.shutdown(wait=False, cancel_futures=True)
    assert not generation.is_alive()
    assert manager.cached_motion is None
    assert manager.cached_audio_feats is None
    assert manager.total_motion is None
    assert published == []


def test_websocket_disconnect_removes_connection(server):
    module, client = server
    with client.websocket_connect("/ws/motion?uid=disconnect-test"):
        assert "disconnect-test" in module.motion_hub._conns
    assert "disconnect-test" not in module.motion_hub._conns


def test_interruption_sequence_and_request_errors(server):
    _, client = server
    headers = {"uid": "offline-test"}
    event = dict(type="manual", source_turn=1, target_turn=2,
                 audio_position_ms=0, timestamp_ms=1, sequence=100)
    assert client.post("/api/v1/interrupt", json=event).status_code == 400
    assert client.post("/api/v1/interrupt", headers=headers, json=event).json()["status"] == "ok"
    assert client.post("/api/v1/interrupt", headers=headers, json=event).json()["status"] == "ignored"
    assert client.get("/api/v1/interruption_stats").json()["current_sequence"] == 100
    assert client.post("/api/v1/stream", headers=headers, content="invalid-json").status_code == 400


def test_text_condition_uses_motivation_without_extra_intention_gate(server):
    module, _ = server
    assert module._build_clip_text_payload({"motivation_score": 3, "clip_texts": {"intent": "wave"}})[0] is None
    texts, _ = module._build_clip_text_payload({
        "motivation_score": 4, "intention_score": 1, "clip_texts": {"intent": "wave"},
    })
    assert texts["intent"] == "wave"
    assert module._build_clip_text_payload({"motivation_score": 5, "clip_texts": {"intent": ""}})[0] is None
