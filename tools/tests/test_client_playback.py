"""Offline behavior tests for the realtime client's local audio sink."""

import asyncio
import importlib
import logging
import queue
import threading
from types import SimpleNamespace

import numpy as np
import pytest
import requests


@pytest.fixture
def realtime_client(monkeypatch):
    monkeypatch.syspath_prepend("src/client")
    return importlib.import_module("realtime_client")


class FakeOutputStream:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.started = False
        self.stopped = False
        self.closed = False
        self.samples = []
        self._lock = threading.Lock()

    def start(self):
        self.started = True

    def write(self, data):
        with self._lock:
            self.samples.append(np.array(data, copy=True))

    def stop(self):
        self.stopped = True

    def close(self):
        self.closed = True


class FakeResponse:
    def __init__(self, status_code=200, server_time=None):
        self.status_code = status_code
        self.server_time = server_time
        self.raise_calls = 0

    def raise_for_status(self):
        self.raise_calls += 1
        if self.status_code >= 400:
            raise requests.HTTPError(f"HTTP {self.status_code}")

    def json(self):
        return {"server_time": self.server_time}


def test_playback_worker_writes_pcm_and_closes_stream(realtime_client, monkeypatch):
    client = realtime_client
    stream = FakeOutputStream()

    def output_stream_factory(**kwargs):
        stream.kwargs = kwargs
        return stream

    monkeypatch.setattr(client.sd, "OutputStream", output_stream_factory)
    playback_queue = queue.Queue()
    monkeypatch.setattr(client, "playback_queue", playback_queue)

    started_stream = client._start_output_stream()
    first = np.array([0.0, 0.25, -0.5], dtype=np.float32)
    second = np.array([0.75, -1.0], dtype=np.float32)
    playback_queue.put(first)
    playback_queue.put(second)
    playback_queue.put(None)

    worker = threading.Thread(target=client._playback_worker, args=(started_stream,))
    worker.start()
    worker.join(timeout=1.0)

    assert not worker.is_alive()
    assert stream.started
    assert len(stream.samples) == 2
    np.testing.assert_array_equal(stream.samples[0], first)
    np.testing.assert_array_equal(stream.samples[1], second)
    assert stream.stopped
    assert stream.closed
    assert stream.kwargs == {
        "samplerate": client.OUTPUT_SAMPLE_RATE,
        "channels": client.OUTPUT_CHANNELS,
        "dtype": "float32",
    }


def test_interruption_drains_pending_local_audio(realtime_client, monkeypatch):
    client = realtime_client
    playback_queue = queue.Queue()
    monkeypatch.setattr(client, "playback_queue", playback_queue)
    playback_queue.put(np.array([0.1], dtype=np.float32))
    playback_queue.put(np.array([0.2], dtype=np.float32))

    assert client._clear_playback_queue() == 2
    assert playback_queue.empty()


def test_playback_loop_outputs_realtime_pcm_without_device(realtime_client, monkeypatch):
    client = realtime_client
    stream = FakeOutputStream()
    output_queue = queue.Queue()
    playback_queue = queue.Queue()
    monkeypatch.setattr(client, "gpt_output_queue", output_queue)
    monkeypatch.setattr(client, "playback_queue", playback_queue)
    monkeypatch.setattr(client, "local_audio_playback_enabled", True)
    monkeypatch.setattr(client, "_start_output_stream", lambda: (stream.start() or stream))

    class Accumulator:
        def add_chunk(self, data):
            pass

        def flush_if_idle(self):
            pass

        def flush(self):
            pass

        def clear(self):
            pass

    class Tracker:
        def add_audio_bytes(self, count):
            pass

        def mark_generation_complete(self):
            pass

        def end_turn(self):
            pass

    monkeypatch.setattr(client, "agent_accumulator", Accumulator())
    monkeypatch.setattr(client, "agent_playback_tracker", Tracker())
    monkeypatch.setattr(
        client,
        "gpt_session",
        SimpleNamespace(
            is_stopped=lambda: True,
            clear_interrupted_items=lambda: None,
        ),
    )
    monkeypatch.setattr(client, "interruption_manager", None)
    monkeypatch.setattr(client, "dialogue_prefill_manager", None)
    monkeypatch.setattr(client, "_cancel_manual_interrupt_timer", lambda: None)
    monkeypatch.setattr(client, "mark_turn_complete", lambda: None)
    monkeypatch.setattr(client, "get_current_turn", lambda: 1)
    monkeypatch.setattr(client, "is_receiving_response", False)
    monkeypatch.setattr(client, "manual_interrupt_active", False)

    pcm = np.array([16384, -16384, 0], dtype=np.int16).tobytes()
    output_queue.put(pcm)
    output_queue.put(None)

    client.playback_loop()

    assert stream.started
    assert len(stream.samples) == 1
    np.testing.assert_allclose(stream.samples[0], np.array([0.5, -0.5, 0.0], dtype=np.float32))
    assert stream.stopped
    assert stream.closed
    assert playback_queue.empty()


def test_motion_websocket_bypasses_environment_proxy(realtime_client, monkeypatch):
    client = realtime_client
    stop_event = threading.Event()
    calls = []

    class FakeConnection:
        async def __aenter__(self):
            stop_event.set()
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return False

    class FakeWebsockets:
        @staticmethod
        def connect(url, **kwargs):
            calls.append((url, kwargs))
            return FakeConnection()

    monkeypatch.setattr(client, "websockets", FakeWebsockets)
    monkeypatch.setattr(client, "motion_stop_event", stop_event)

    asyncio.run(client._motion_ws_client("ws://localhost:32558/ws/motion?uid=test", "test"))

    assert calls == [
        (
            "ws://localhost:32558/ws/motion?uid=test",
            {"max_size": None, "proxy": None},
        )
    ]


@pytest.mark.parametrize(
    "sender_name, sender_args, expected_log",
    [
        (
            "send_audio_chunk",
            ("encoded", "assistant", 1, 2, 24000, 10),
            "Failed to send assistant audio idx=1",
        ),
        (
            "send_turn_complete",
            (2,),
            "Failed to send turn_complete for turn=2",
        ),
        (
            "send_slow_thinking_update",
            (
                SimpleNamespace(
                    should_interrupt_user=False,
                    analysis=None,
                    clip_texts=SimpleNamespace(model_dump=lambda: {}),
                    motivation_score=3,
                    intention_score=3,
                    proactive_level="L1_reactive",
                    proactive_trigger="none",
                    locomotion=[],
                    emotion="neutral",
                    dialogue_instruction=SimpleNamespace(prefill_text=""),
                    confidence="low",
                ),
            ),
            "Failed to send slow thinking update",
        ),
    ],
)
def test_stream_posts_log_http_errors(realtime_client, monkeypatch, caplog,
                                      sender_name, sender_args, expected_log):
    client = realtime_client
    response = FakeResponse(status_code=500)
    monkeypatch.setattr(client._local_session, "post", lambda *args, **kwargs: response)

    with caplog.at_level(logging.ERROR, logger=client.logger.name):
        getattr(client, sender_name)(*sender_args)

    assert response.raise_calls == 1
    assert expected_log in caplog.text


def test_time_sync_reset_http_error_returns_false(realtime_client, monkeypatch, caplog):
    client = realtime_client
    get_response = FakeResponse(server_time=1)
    post_response = FakeResponse(status_code=500)
    monkeypatch.setattr(client._local_session, "get", lambda *args, **kwargs: get_response)
    monkeypatch.setattr(client._local_session, "post", lambda *args, **kwargs: post_response)

    with caplog.at_level(logging.ERROR, logger=client.logger.name):
        assert client.synchronize_time_with_server() is False

    assert get_response.raise_calls == 1
    assert post_response.raise_calls == 1
    assert "Failed to notify server to set zero time" in caplog.text
    assert "Time synchronized and zero point set" not in caplog.text
