#!/usr/bin/env python3
"""Realtime social-robot client with interruption, streaming, and rendering."""

import argparse
import asyncio
import base64
import io
import json
import logging
import os
import queue
import threading
import time
import wave
from typing import Any, Dict, Optional, List, Tuple, Sequence
import multiprocessing
import subprocess
import sys
from pathlib import Path

# Preprocessing exports reference data_process.pymo classes when unpickled.
_MOTION_CODE_DIR = str(Path(__file__).resolve().parents[1] / "motion")
if _MOTION_CODE_DIR not in sys.path:
    sys.path.append(_MOTION_CODE_DIR)
# Older motion exports were created with ``data_process`` itself on
# ``sys.path`` and therefore refer to the package as top-level ``pymo``.
# Keep that package name importable alongside the cleaned ``data_process.pymo``
# layout so joblib can unpickle both generations of assets.
_MOTION_DATA_PROCESS_DIR = str(Path(_MOTION_CODE_DIR) / "data_process")
if _MOTION_DATA_PROCESS_DIR not in sys.path:
    sys.path.append(_MOTION_DATA_PROCESS_DIR)
import queue as pyqueue
import joblib as jl
import pandas as pd
from scipy.signal import resample_poly
import numpy as np
import soxr

import cv2
import numpy as np
import requests
import sounddevice as sd
import soundfile as sf
from asr_utils import DEFAULT_SHERPA_MODEL_DIR, LiveTranscriptStore, SherpaASRWorker
from agents.realtime import (
    RealtimeAgent,
    RealtimePlaybackTracker,
    RealtimeRunner,
)
from agents.realtime.config import RealtimeSessionModelSettings
from agents.realtime.model import RealtimeModelConfig
from agents.realtime.model_inputs import RealtimeModelSendRawMessage
from logging.handlers import RotatingFileHandler

from interruption_core import (
    InterruptionManager,
    InterruptionType,
    InterruptionState,
    PlaybackController,
    MotionSyncController,
)

try:
    from g1_csv_sim_mujoco import main_mujoco
except ImportError:
    main_mujoco = None

try:
    from g1_csv_arm7_real import main_real
except ImportError:
    main_real = None

# try:
#     from g1_csv_arm7_pid import main_pid as main_real
# except ImportError:
#     main_real = None

try:
    from blender_stream_client import (
        BlenderStreamClient,
        BLENDER_STREAM_HOST,
        BLENDER_STREAM_PORT,
        BLENDER_FRAME_INTERVAL,
        BLENDER_AUDIO_CHUNK_SECONDS,
    )
except ImportError:
    BlenderStreamClient = None

from cognition import (
    CURRENT_SCENARIO_SLOW,
    SlowThinkingSystem,
    SlowThinkingConfig,
    SlowThinkingResult,
    DialogueInstruction,
)
from slow_thinking_visual import run_visualizer_process

class PulseAudioInputStream:
    def __init__(self, samplerate, blocksize, channels, dtype, callback):
        self.samplerate, self.blocksize, self.channels, self.dtype, self.callback = samplerate, blocksize, channels, dtype, callback
        self.process, self._stop_event, self._thread = None, threading.Event(), None
    def __enter__(self):
        cmd = ['parec', '--latency-msec=20', f'--format=s16le', f'--channels={self.channels}', f'--rate={self.samplerate}', '--raw']
        self.process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        return self
    def _run(self):
        bytes_to_read = self.blocksize * 2 * self.channels
        while not self._stop_event.is_set() and self.process.poll() is None:
            raw_data = self.process.stdout.read(bytes_to_read)
            if raw_data:
                indata = np.frombuffer(raw_data, dtype=np.int16).reshape(-1, self.channels)
                self.callback(indata, len(indata), None, None)
    def __exit__(self, et, ev, tb):
        self._stop_event.set()
        if self.process: self.process.terminate()

try:
    import websockets
except ImportError:
    websockets = None

try:
    from pynput import keyboard as pynput_keyboard
except ImportError:
    pynput_keyboard = None

# Configuration is supplied through environment variables so deployments can
# select their own server, scenario, credentials, and robot identity.
_REPO_ROOT = str(Path(__file__).resolve().parents[2])
CURRENT_SCENARIO = CURRENT_SCENARIO_SLOW
SERVER = os.getenv("PROACT_SERVER_URL", "http://localhost:32558").rstrip("/")
_local_session = requests.Session()
_local_session.trust_env = False
UID = os.getenv("PROACT_CLIENT_ID", "mic_user_001")
HEADERS = {"uid": UID}

REALTIME_MODEL = os.getenv("OPENAI_REALTIME_MODEL", "gpt-realtime")
AGENT_VOICE = os.getenv("OPENAI_REALTIME_VOICE", "echo")

def get_system_instructions(scenario: str) -> str:
    return scenario

SYSTEM_INSTRUCTIONS = get_system_instructions(CURRENT_SCENARIO)

# Build Realtime settings for the requested interruption mode.
def _get_realtime_settings(allow_interrupt: bool = False) -> RealtimeSessionModelSettings:
    """Return Realtime settings with optional user interruption support."""
    return {
        "model_name": REALTIME_MODEL,
        "instructions": get_system_instructions(CURRENT_SCENARIO),
        "modalities": ["audio"],
        "voice": AGENT_VOICE,
        "input_audio_format": "pcm16",
        "output_audio_format": "pcm16",
        "turn_detection": {
            "type": "semantic_vad",
            "eagerness": "auto",
            "create_response": True,
            "interrupt_response": allow_interrupt,
        },
        "input_audio_transcription": {
            "model": "gpt-4o-mini-transcribe",
        },
        "speed": 1.0,
        "tracing": {
            "enabled": True,
            "level": "basic"
        }
    }

# Defaults are rebuilt in main after parsing command-line options.
REALTIME_SETTINGS: RealtimeSessionModelSettings = _get_realtime_settings(allow_interrupt=False)
REALTIME_MODEL_CONFIG: RealtimeModelConfig = {
    "initial_model_settings": REALTIME_SETTINGS,
}
playback_tracker = RealtimePlaybackTracker()
agent = RealtimeAgent(
    name="ProAct",
    instructions=SYSTEM_INSTRUCTIONS,
)

def parse_cli_args(argv=None):
    parser = argparse.ArgumentParser(description="Social robot client with interruption support")
    parser.add_argument(
        "--need_slow_thinking",
        action="store_true",
        help="Enable slow-thinking planning loop",
    )
    parser.add_argument(
        "--render_target",
        choices=["mujoco", "blender", "none"],
        default="mujoco",
        help="Where to play motions/audio: mujoco or blender (or none).",
    )
    parser.add_argument(
        "--real",
        action="store_true",
        help="Enable real robot control (main_real)",
    )
    parser.add_argument(
        "--asr",
        action="store_true",
        help="Enable ASR transcriber",
    )
    parser.add_argument(
        "--allow-interrupt",
        action="store_true",
        help="Enable interruption support (user can interrupt agent speech)",
    )
    return parser.parse_args(argv)

def _truncate_str(s: str, max_length: int) -> str:
    if len(s) > max_length:
        return s[:max_length] + "..."
    return s

# Audio parameters
INPUT_RAW_SAMPLE_RATE = 48000
INPUT_SAMPLE_RATE = 24000
OUTPUT_SAMPLE_RATE = 24000
INPUT_CHANNELS = 1
OUTPUT_CHANNELS = 1
SAMPLE_WIDTH = 2  # bytes for int16

SESSION_INITIAL_OFFSET_SEC = 1.0
MIN_AUDIO_DURATION_SEC = 0.1
MIN_USER_SAMPLES = int(INPUT_SAMPLE_RATE * MIN_AUDIO_DURATION_SEC)
MIN_ASSIST_PCM_BYTES = int(OUTPUT_SAMPLE_RATE * MIN_AUDIO_DURATION_SEC) * SAMPLE_WIDTH

SEGMENT_DURATION = 1.0
SEGMENT_SIZE = int(INPUT_SAMPLE_RATE * SEGMENT_DURATION)

IMG_CAPTURE_INTERVAL = float(os.getenv("PROACT_IMAGE_CAPTURE_INTERVAL", "1.0"))
IMG_SEND_INTERVAL = float(os.getenv("PROACT_IMAGE_SEND_INTERVAL", "5.0"))
IMAGE_JPEG_QUALITY = 50

# Keep generated session data outside the source tree by default. Override it
# with PROACT_MEDIA_DIR when a deployment needs a different storage location.
start_session_label = time.strftime("%Y%m%d_%H%M%S")
LOCAL_MEDIA_BASE = os.path.abspath(
    os.getenv("PROACT_MEDIA_DIR", os.path.join(_REPO_ROOT, "runtime", "client_media"))
)
local_session_dir: Optional[str] = None
local_user_audio_dir: Optional[str] = None
local_assistant_audio_dir: Optional[str] = None
local_image_dir: Optional[str] = None
local_motion_dir: Optional[str] = None
os.makedirs(LOCAL_MEDIA_BASE, exist_ok=True)

# === Global state ===
input_audio_queue: "queue.Queue[Optional[np.ndarray]]" = queue.Queue()
gpt_input_queue: "queue.Queue[Optional[Dict[str, Any]]]" = queue.Queue(maxsize=128)
gpt_output_queue: "queue.Queue[Optional[bytes]]" = queue.Queue()

locomotion_queue: "Optional[multiprocessing.Queue]" = None
image_queue: "Optional[multiprocessing.Queue]" = None
viz_queue: "Optional[multiprocessing.Queue]" = None

is_robot_playing: Any = None
last_robot_play_end_time: Any = None
last_response_create_time: Any = None
user_reply_threshold = float(os.getenv("PROACT_REPLY_COOLDOWN_SEC", "16.0"))
is_mujoco_playing: Any = None

is_receiving_lock = threading.Lock()
is_receiving_response = False
manual_interrupt_active = False
MANUAL_INTERRUPT_FAILSAFE_SEC = 10.0
_manual_interrupt_timer: Optional[threading.Timer] = None

segment_index_lock = threading.Lock()
segment_index = 0
session_start_time = time.time()
session_time_lock = threading.Lock()
first_local_save_done = False

turn_lock = threading.Lock()
current_turn = 1
turn_user_audio_pending = False

segment_meta_lock = threading.Lock()
last_segment_speaker: Optional[str] = None
last_segment_turn: Optional[int] = None

# Realtime response timeout diagnostics
_last_user_audio_sent_time = 0.0
_last_gpt_response_time = 0.0
GPT_RESPONSE_TIMEOUT_SEC = float(os.getenv("PROACT_RESPONSE_TIMEOUT_SEC", "15.0"))

user_buffer_lock = threading.Lock()
user_buffer = np.empty((0,), dtype='float32')
last_user_voice_time = 0.0
USER_SPEECH_GRACE_SEC = float(os.getenv("PROACT_USER_SPEECH_GRACE_SEC", "1.5"))
USER_SPEECH_RMS_THRESHOLD = float(os.getenv("PROACT_USER_SPEECH_RMS_THRESHOLD", "350.0"))

time_offset_ns = 0
zero_point_ns = 0

logger = logging.getLogger("robot_gesture_client")
asr_worker: Optional[SherpaASRWorker] = None
live_transcripts = LiveTranscriptStore()
last_user_turn_for_asr: Optional[int] = None
last_assistant_turn_for_asr: Optional[int] = None
_last_asr_key: Optional[Tuple[str, int]] = None
_last_asr_text: str = ""
viz_queue: Optional[multiprocessing.Queue] = None

STALE_TRANSCRIPT_FINALISE_SEC = 1.2

_DEFAULT_DATA_DIR = os.path.join(_REPO_ROOT, "src", "assets", "preprocessing", "audio")
_DEFAULT_TEXT_DATA_DIR = os.path.join(_REPO_ROOT, "data", "ProAct-Data", "processed_seg")
data_pipe_path = os.getenv(
    "PROACT_DATA_PIPE", os.path.join(_DEFAULT_DATA_DIR, "data_pipe_30fps.sav")
)
body_scaler_path = os.getenv(
    "PROACT_OUTPUT_SCALER", os.path.join(_DEFAULT_DATA_DIR, "output_scaler.sav")
)
text_data_pipe_path = os.getenv(
    "PROACT_TEXT_DATA_PIPE", os.path.join(_DEFAULT_TEXT_DATA_DIR, "data_pipe_30fps.sav")
)
text_body_scaler_path = os.getenv(
    "PROACT_TEXT_OUTPUT_SCALER", os.path.join(_DEFAULT_TEXT_DATA_DIR, "output_scaler.sav")
)
data_pipe = None
body_scaler = None
text_data_pipe = None
text_body_scaler = None
text_data_dir = os.path.dirname(text_data_pipe_path) if text_data_pipe_path else None


def _load_motion_assets() -> None:
    """Load external motion preprocessing assets on first motion response."""
    global data_pipe, body_scaler, text_data_pipe, text_body_scaler
    if all(asset is not None for asset in (data_pipe, body_scaler, text_data_pipe, text_body_scaler)):
        return
    paths = {
        "PROACT_DATA_PIPE": data_pipe_path,
        "PROACT_OUTPUT_SCALER": body_scaler_path,
        "PROACT_TEXT_DATA_PIPE": text_data_pipe_path,
        "PROACT_TEXT_OUTPUT_SCALER": text_body_scaler_path,
    }
    missing = [f"{name}={path}" for name, path in paths.items() if not path or not os.path.isfile(path)]
    if missing:
        raise RuntimeError(
            "Motion assets are unavailable. Configure the four motion asset "
            "environment variables: " + ", ".join(missing)
        )
    try:
        data_pipe = jl.load(data_pipe_path)
        body_scaler = jl.load(body_scaler_path)
        text_data_pipe = jl.load(text_data_pipe_path)
        text_body_scaler = jl.load(text_body_scaler_path)
    except Exception as exc:
        raise RuntimeError("Failed to load configured motion assets.") from exc


def inv_standardize(data, scaler):
    shape = data.shape
    flat = data.copy().reshape((shape[0] * shape[1], shape[2]))
    scaled = scaler.inverse_transform(flat).reshape(shape)
    return scaled


class ConversationTracker:
    """Track user/agent transcripts for slow-thinking context and logging."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._history: List[Tuple[str, str, float]] = []
        self._current_user: str = ""
        self._current_agent: str = ""
        self._last_user_finish_text: str = ""
        self._last_agent_finish_text: str = ""
        self._last_user_finish_ts: float = 0.0
        self._last_agent_finish_ts: float = 0.0
        self._current_user_ts: float = 0.0
        self._current_agent_ts: float = 0.0
        self._current_user_start_ts: float = 0.0
        self._current_agent_start_ts: float = 0.0

    def reset(self) -> None:
        with self._lock:
            self._history.clear()
            self._current_user = ""
            self._current_agent = ""
            self._last_user_finish_text = ""
            self._last_agent_finish_text = ""
            self._last_user_finish_ts = 0.0
            self._last_agent_finish_ts = 0.0
            self._current_user_ts = 0.0
            self._current_agent_ts = 0.0
            self._current_user_start_ts = 0.0
            self._current_agent_start_ts = 0.0

    def update_user_transcription(
        self,
        text: Optional[str],
    ) -> Tuple[bool, Optional[str]]:
        """Update user transcript and finalise agent turn if a switch occurred."""
        if not text:
            return False, None
        normalized = text.strip()
        if not normalized:
            return False, None
        now = time.time()
        finalized_agent_text: Optional[str] = None
        with self._lock:
            self._finalize_stale_locked(now)
            finalized_agent_text = None
            if not self._current_user and self._current_agent:
                finalized_agent_text = self._commit_agent(now)
            if not self._current_user:
                self._current_user = normalized
                self._current_user_start_ts = now
            else:
                self._current_user = normalized
            self._current_user_ts = now
        return finalized_agent_text is not None, finalized_agent_text

    def update_agent_transcription(
        self,
        text: Optional[str],
    ) -> Tuple[bool, Optional[str]]:
        """Update agent transcript and finalise user turn if a switch occurred."""
        if not text:
            return False, None
        normalized = text.strip()
        if not normalized:
            return False, None
        now = time.time()
        finalized_user_text: Optional[str] = None
        with self._lock:
            self._finalize_stale_locked(now)
            finalized_user_text = None
            if not self._current_agent and self._current_user:
                finalized_user_text = self._commit_user(now)
            if not self._current_agent:
                self._current_agent = normalized
                self._current_agent_start_ts = now
            else:
                self._current_agent = normalized
            self._current_agent_ts = now
        return finalized_user_text is not None, finalized_user_text

    def get_transcript_text(self) -> str:
        with self._lock:
            self._finalize_stale_locked(time.time())
            lines = [
                f"USER: {text}" if speaker == "user" else f"AGENT: {text}"
                for speaker, text, _ in self._history
            ]
        return "\n".join(lines).strip()

    def _finalize_stale_locked(self, now: float) -> None:
        if self._current_user and (now - self._current_user_ts) > STALE_TRANSCRIPT_FINALISE_SEC:
            self._commit_user(now)
        if self._current_agent and (now - self._current_agent_ts) > STALE_TRANSCRIPT_FINALISE_SEC:
            self._commit_agent(now)

    def _commit_user(self, ts: float) -> Optional[str]:
        if not self._current_user:
            return None
        text = self._current_user.strip()
        if not text:
            self._clear_user()
            return None
        if text == self._last_user_finish_text and (ts - self._last_user_finish_ts) < 0.35:
            self._clear_user()
            return None
        self._history.append(("user", text, ts))
        self._last_user_finish_text = text
        self._last_user_finish_ts = ts
        self._clear_user()
        if len(self._history) > 200:
            self._history = self._history[-200:]
        return text

    def _commit_agent(self, ts: float) -> Optional[str]:
        if not self._current_agent:
            return None
        text = self._current_agent.strip()
        if not text:
            self._clear_agent()
            return None
        if text == self._last_agent_finish_text and (ts - self._last_agent_finish_ts) < 0.35:
            self._clear_agent()
            return None
        self._history.append(("assistant", text, ts))
        self._last_agent_finish_text = text
        self._last_agent_finish_ts = ts
        self._clear_agent()
        if len(self._history) > 200:
            self._history = self._history[-200:]
        return text

    def _clear_user(self) -> None:
        self._current_user = ""
        self._current_user_ts = 0.0
        self._current_user_start_ts = 0.0

    def _clear_agent(self) -> None:
        self._current_agent = ""
        self._current_agent_ts = 0.0
        self._current_agent_start_ts = 0.0

    def _merge_text(self, current: str, incoming: str) -> str:
        if not current:
            return incoming
        if not incoming:
            return current
        if incoming == current:
            return current
        if incoming.startswith(current):
            return incoming
        if current.startswith(incoming):
            return current
        max_overlap = min(len(current), len(incoming))
        for i in range(max_overlap, 0, -1):
            if current[-i:] == incoming[:i]:
                return current + incoming[i:]
        if current.endswith(" ") or incoming.startswith(" "):
            return current + incoming
        return current + " " + incoming


class AgentPlaybackTracker:
    """
    Track agent audio playback and expose time-based transcript progress.
    """

    def __init__(self):
        self._lock = threading.Lock()
        self._is_active = False
        self._generation_complete = False
        self._playback_start_time = 0.0
        self._total_audio_bytes = 0
        self._full_transcript = ""
        self._current_turn = 0

    def add_audio_bytes(self, byte_count: int) -> None:
        """Record newly received audio bytes."""
        with self._lock:
            if not self._is_active:
                self._is_active = True
                self._generation_complete = False
                self._playback_start_time = time.time()
                self._total_audio_bytes = 0
            self._total_audio_bytes += byte_count

    def mark_generation_complete(self) -> None:
        """Mark audio generation as complete."""
        with self._lock:
            self._generation_complete = True

    def update_transcript(self, text: str, turn: Optional[int] = None) -> None:
        """Update the current response transcript."""
        with self._lock:
            self._full_transcript = text
            if turn is not None:
                self._current_turn = turn

    def end_turn(self) -> None:
        """Reset playback state at the end of a turn."""
        with self._lock:
            self._is_active = False
            self._generation_complete = False
            self._full_transcript = ""
            self._total_audio_bytes = 0
            self._playback_start_time = 0.0

    def get_state(self) -> dict:
        """Return current playback state and transcript progress."""
        with self._lock:
            if not self._is_active or not self._full_transcript:
                return {
                    "is_playing": False,
                    "generation_complete": True,
                    "played_transcript": self._full_transcript,
                    "pending_transcript": "",
                    "progress_ratio": 1.0,
                    "elapsed_sec": 0.0,
                    "remaining_sec": 0.0,
                    "total_duration_sec": 0.0,
                    "turn": self._current_turn,
                }

            total_duration = self._total_audio_bytes / (SAMPLE_WIDTH * OUTPUT_SAMPLE_RATE)
            elapsed = time.time() - self._playback_start_time

            if not self._generation_complete:
                return {
                    "is_playing": True,
                    "generation_complete": False,
                    "played_transcript": "",
                    "pending_transcript": self._full_transcript,
                    "progress_ratio": 0.0,
                    "elapsed_sec": elapsed,
                    "remaining_sec": -1.0,
                    "total_duration_sec": -1.0,
                    "turn": self._current_turn,
                }

            if total_duration <= 0:
                progress = 1.0
            else:
                progress = min(1.0, elapsed / total_duration)

            total_chars = len(self._full_transcript)
            played_chars = int(total_chars * progress)

            played_text = self._full_transcript[:played_chars]
            pending_text = self._full_transcript[played_chars:]

            if pending_text and played_text and not pending_text[0].isspace():
                last_space = played_text.rfind(' ')
                if last_space > 0:
                    pending_text = played_text[last_space + 1:] + pending_text
                    played_text = played_text[:last_space + 1]

            return {
                "is_playing": True,
                "generation_complete": True,
                "played_transcript": played_text.strip(),
                "pending_transcript": pending_text.strip(),
                "progress_ratio": progress,
                "elapsed_sec": elapsed,
                "remaining_sec": max(0, total_duration - elapsed),
                "total_duration_sec": total_duration,
                "turn": self._current_turn,
            }


class DialoguePrefillManager:
    """Stage dialogue instructions and dispatch at turn boundaries."""

    def __init__(self, session: "GPTSession") -> None:
        self._session = session
        self._lock = threading.Lock()
        self._pending: Dict[str, Optional[DialogueInstruction]] = {
            "before_agent_reply": None,
            "before_user_reply": None,
        }

    def queue_instruction(self, instruction: DialogueInstruction) -> None:
        with self._lock:
            self._pending[instruction.insertion_point] = instruction

    def dispatch_before_agent(self) -> None:
        instruction = self._pop("before_agent_reply")
        if instruction:
            self._send(instruction)

    def dispatch_before_user(self) -> None:
        instruction = self._pop("before_user_reply")
        if instruction:
            self._send(instruction)

    def _pop(self, key: str) -> Optional[DialogueInstruction]:
        with self._lock:
            instruction = self._pending.get(key)
            self._pending[key] = None
        return instruction

    def _send(self, instruction: DialogueInstruction) -> None:
        logger.info(
            "Dispatching dialogue instruction (%s): %s",
            instruction.insertion_point,
            instruction.prefill_text.replace("\n", " ")[:200],
        )
        if viz_queue:
            viz_queue.put({
                "type": "slow_thinking",
                "data": {
                    "dialogue": {
                        'speak_now_score': 0,
                        "prefill_text": instruction.prefill_text,
                        "insertion_point": instruction.insertion_point,
                    }
                },
            })

        self._session.enqueue_prefill(
            instruction.prefill_text,
        )


class SlowThinkingImageSampler:
    """Track saved frames and return evenly spaced samples per cycle."""

    def __init__(self, max_images: int = 3) -> None:
        self._max_images = max(1, max_images)
        self._lock = threading.Lock()
        self._last_mtime = 0.0

    def reset(self) -> None:
        with self._lock:
            self._last_mtime = 0.0

    def sample(self, directory: Optional[str]) -> List[str]:
        if not directory or not os.path.isdir(directory):
            return []
        try:
            entries = []
            for name in os.listdir(directory):
                if not name.lower().endswith(".jpg"):
                    continue
                path = os.path.join(directory, name)
                try:
                    mtime = os.path.getmtime(path)
                except OSError:
                    continue
                entries.append((mtime, path))
        except OSError:
            return []

        if not entries:
            return []

        with self._lock:
            new_entries = [item for item in entries if item[0] > self._last_mtime]
            if not new_entries:
                return []
            new_entries.sort()
            selected = self._uniform_sample(new_entries, self._max_images)
            self._last_mtime = max(ts for ts, _ in new_entries)
            return selected

    def _uniform_sample(
        self,
        items: Sequence[Tuple[float, str]],
        limit: int,
    ) -> List[str]:
        if len(items) <= limit:
            return [path for _, path in items]
        sampled: List[str] = []
        for i in range(limit):
            position = (i + 0.5) * len(items) / limit
            idx = min(int(position), len(items) - 1)
            sampled.append(items[idx][1])
        return sampled


conversation_tracker = ConversationTracker()
image_sampler = SlowThinkingImageSampler()
dialogue_prefill_manager: Optional[DialoguePrefillManager] = None
slow_thinking_system: Optional[SlowThinkingSystem] = None
agent_playback_tracker = AgentPlaybackTracker()


interruption_manager: Optional[InterruptionManager] = None
motion_sync_controller: Optional[MotionSyncController] = None


def slow_thinking_context_provider() -> Optional[Tuple[str, Sequence[str], int]]:
    """
    Retrieves the latest context for slow thinking.
    """
    snap = live_transcripts.snapshot()
    playback_state = agent_playback_tracker.get_state()

    def _parse_key(key: str) -> Tuple[Optional[str], Optional[int]]:
        if "-turn" not in key:
            return None, None
        speaker, turn_part = key.split("-turn", 1)
        try:
            return speaker, int(turn_part)
        except Exception:
            return speaker, None

    def _format_line(speaker: str, turn: Optional[int], text: str) -> str:
        role = "User" if speaker == "user" else "Agent"
        turn_suffix = f"-turn_{turn}" if turn is not None else ""
        return f"{role}{turn_suffix}: {text.strip()}"

    final_dict = snap.get("final", {})
    partial_dict = snap.get("partial", {})
    recent_dict = snap.get("recent", {})
    current_turn_idx = get_current_turn()

    def _sort_key(key: str) -> int:
        _, turn = _parse_key(key)
        return turn if turn is not None else 99999

    partial_current_turn = {
        k: v for k, v in partial_dict.items()
        if _parse_key(k)[1] == current_turn_idx
    }
    recent_current_turn = {
        k: v for k, v in recent_dict.items()
        if _parse_key(k)[1] == current_turn_idx
    }

    current_key: Optional[str] = None
    if partial_current_turn:
        current_key = sorted(partial_current_turn.keys(), key=_sort_key)[-1]

    current_full = ""
    past_lines: List[str] = []

    if current_key:
        cur_speaker, cur_turn = _parse_key(current_key)
        if current_key in final_dict:
            current_full = _format_line(cur_speaker, cur_turn, final_dict[current_key])
            for k, v in final_dict.items():
                if k == current_key or not v:
                    continue
                sp, tn = _parse_key(k)
                past_lines.append(_format_line(sp, tn, v))
        else:
            current_full = _format_line(cur_speaker, cur_turn, partial_dict[current_key])
            for k, v in final_dict.items():
                if not v:
                    continue
                sp, tn = _parse_key(k)
                past_lines.append(_format_line(sp, tn, v))
    else:
        for k, v in final_dict.items():
            if not v:
                continue
            sp, tn = _parse_key(k)
            past_lines.append(_format_line(sp, tn, v))

    history_block = "\n".join(past_lines).strip() or "[No past transcript yet]"
    if current_full.strip():
        current_full_block = current_full.strip()
    else:
        current_full_block = _format_line("user", current_turn_idx, "[No current full transcript]")

    recent_block = ""
    if current_key and current_key in recent_current_turn:
        recent_block = recent_current_turn[current_key].strip()
    elif recent_current_turn:
        last_recent_key = sorted(recent_current_turn.keys(), key=_sort_key)[-1]
        recent_block = recent_current_turn[last_recent_key].strip()
    if not recent_block:
        recent_block = "[No recent words]"

    agent_playback_block = ""
    if playback_state["is_playing"]:
        if playback_state["generation_complete"]:
            progress_pct = playback_state["progress_ratio"] * 100
            remaining_sec = playback_state["remaining_sec"]
            played_text = playback_state["played_transcript"]
            pending_text = playback_state["pending_transcript"]

            agent_playback_block = (
                f"## Agent Speaking (Progress: {progress_pct:.0f}%, Remaining: {remaining_sec:.1f}s)\n"
                f"Spoken: {played_text if played_text else '[Just started]'}\n"
                f"Pending: {pending_text if pending_text else '[Almost done]'}"
            )
        else:
            elapsed_sec = playback_state["elapsed_sec"]
            pending_text = playback_state["pending_transcript"]

            agent_playback_block = (
                f"## Agent Speaking (Generating, Elapsed: {elapsed_sec:.1f}s)\n"
                f"Content: {pending_text if pending_text else '[Generating...]'}"
            )

    transcript_parts = [
        f"## History transcripts\n{history_block}",
    ]

    if agent_playback_block:
        transcript_parts.append(agent_playback_block)
    else:
        transcript_parts.append(f"## Current Agent Sentence\n{current_full_block}")
        transcript_parts.append(f"## Current Agent Words\n{recent_block}")

    transcript = "\n\n".join(transcript_parts)

    sampled_images = image_sampler.sample(local_image_dir)

    if not transcript.strip() and not sampled_images:
        return None
    return transcript, sampled_images, get_current_segment_index()


def send_slow_thinking_update(result: SlowThinkingResult) -> None:
    """Sends the structured slow-thinking result to the server."""
    current_mode = "proactive" if result.should_interrupt_user else "reactive"

    slow_thinking_payload = {
        "mode": current_mode,
        "analysis": result.analysis.model_dump() if result.analysis else None,
        "clip_texts": result.clip_texts.model_dump(),
        "motivation_score": result.motivation_score,
        "intention_score": result.intention_score,
        "proactive_level": result.proactive_level,
        "proactive_trigger": result.proactive_trigger,
        "locomotion": [action.model_dump() for action in result.locomotion],
        "emotion": result.emotion,
        "should_interrupt_user": result.should_interrupt_user,
        "speech_prefill": result.dialogue_instruction.prefill_text,
        "confidence": result.confidence,
        "timestamp_ms": int(time.time() * 1000),
    }

    payload = {
        "messages": [
            {
                "role": "system",
                "content": [
                    {
                        "type": "slow_thinking",
                        "slow_thinking": slow_thinking_payload,
                    }
                ],
            }
        ]
    }

    t0 = time.time()
    try:
        res = _local_session.post(f"{SERVER}/api/v1/stream", headers=HEADERS, json=payload, timeout=10)
        res.raise_for_status()
        dt = time.time() - t0
        status = getattr(res, "status_code", "n/a")
        logger.debug("Sent slow thinking update status=%s duration=%.2fms", status, dt * 1000)
    except Exception as exc:
        logger.error("Failed to send slow thinking update: %s", exc)


def slow_thinking_result_handler(result: SlowThinkingResult) -> None:
    """Handles the result locally."""
    current_mode = "PROACTIVE" if result.should_interrupt_user else "REACTIVE"
    loco_str = "None"
    if result.locomotion:
        valid_actions = [f"{a.action}({a.parameter})" for a in result.locomotion if a.action != "none"]
        if valid_actions:
            loco_str = ", ".join(valid_actions)


    for a in result.locomotion:
        print('EXECUTING LOCOMOTION INSTRUCTION FROM SLOW THINKING...')
        print('result.locomotion_instruction:', a.action)
        print('result.locomotion_parameter:', a.parameter)

        if locomotion_queue is not None:
            try:
                locomotion_queue.put((a.action, a.parameter))
                logger.info("Queued locomotion instruction: %s with parameter: %s",
                            a.action, a.parameter)
            except Exception as e:
                logger.error("Failed to queue locomotion instruction: %s", e)
        else:
            logger.warning("Locomotion queue is not initialized, skipping instruction.")

    if result.analysis:
        analysis_summary = {
            "my_intention": result.analysis.my_intention,
            "engagement_level": result.analysis.engagement_level,
            "social_distance": result.analysis.social_distance,
            "conversation_phase": result.analysis.conversation_phase,
            "atmosphere": result.analysis.atmosphere,
            "empathy_requirement": result.analysis.empathy_requirement,
            "my_intention": result.dialogue_instruction.prefill_text
        }
    else:
        analysis_summary = {"my_intention": result.dialogue_instruction.prefill_text}
    logger.info(f"SlowThinking Analysis: {json.dumps(analysis_summary, ensure_ascii=False)}")

    logger.info(
        "SlowThinking [%s|%s] -> Intent: %s | Emo: %s | Loco: [%s] | Prefill: %s",
        current_mode,
        result.confidence.upper(),
        # str(result.intention_score),
        result.clip_texts.intent,
        result.emotion,
        loco_str,
        _truncate_str(result.dialogue_instruction.prefill_text.replace("\n", " "), 60),
    )

    # if result.timings:
    #     timings = result.timings
    #     logger.info(
    #         "SlowThinking run_id=%s timing total=%.2fs text=%.2fs memory=%.2fs proact=%.2fs",
    #         result.run_id,
    #         timings.get("total_s", 0.0),
    #         timings.get("scene2text_s", 0.0),
    #         timings.get("scene2memory_s", 0.0),
    #         timings.get("scene2proact_s", 0.0),
    #     )

    send_slow_thinking_update(result)
    result.dialogue_instruction.insertion_point = "before_agent_reply"
    if dialogue_prefill_manager:
        dialogue_prefill_manager.queue_instruction(result.dialogue_instruction)

    if result.dialogue_instruction.should_interrupt_user:
        # Keep the prefill queued, but wait until the user has stopped speaking
        # before interrupting the current agent response.
        can_interrupt = not user_is_speaking()
        if not can_interrupt:
            logger.info("Queued proactive prefill; delaying interruption while user speech is active.")
        if last_robot_play_end_time is not None and last_response_create_time is not None:
            reference_time = max(last_robot_play_end_time.value, last_response_create_time.value)
            time_since_last_play = time.time() - reference_time
            if time_since_last_play < user_reply_threshold:
                logger.info(f"Slow thinking requested interrupt, but suppressed: time_since_last_play={time_since_last_play:.2f}s < threshold={user_reply_threshold}s (ref_time source: {'play_end' if last_robot_play_end_time.value > last_response_create_time.value else 'response.create'})")
                can_interrupt = False

        if can_interrupt:
            if request_agent_interrupt("slow_thinking"):
                logger.info("Slow thinking requested an agent interrupt.")


# === Logging ===
def setup_logging(level: int = logging.DEBUG):
    os.makedirs(LOCAL_MEDIA_BASE, exist_ok=True)
    base = os.path.join(LOCAL_MEDIA_BASE, start_session_label)
    os.makedirs(base, exist_ok=True)
    log_file = os.path.join(base, f"client_{start_session_label}.log")

    logger.setLevel(level)
    logger.propagate = False

    if not logger.handlers:
        fmt = logging.Formatter("%(asctime)s - %(levelname)s - %(threadName)s - %(message)s")

        fh = RotatingFileHandler(log_file, maxBytes=5 * 1024 * 1024, backupCount=5, encoding="utf-8")
        fh.setFormatter(fmt)
        fh.setLevel(level)
        logger.addHandler(fh)

        console = logging.StreamHandler()
        console.setFormatter(fmt)
        console.setLevel(level)
        logger.addHandler(console)

    blender_logger = logging.getLogger("blender_stream_client")
    blender_logger.setLevel(level)
    blender_logger.propagate = False
    if not blender_logger.handlers:
        for h in logger.handlers:
            blender_logger.addHandler(h)

    logger.info("Logging initialised: %s", log_file)


def clean_old_files(directories):
    """Remove old files from the cache directories."""
    for directory in directories:
        for filename in os.listdir(directory):
            path = os.path.join(directory, filename)
            if os.path.isfile(path):
                os.remove(path)

def synchronize_time_with_server():
    """Synchronize the local clock offset with the server."""
    global time_offset_ns, zero_point_ns
    try:
        t1 = time.time_ns()
        response = _local_session.get(f"{SERVER}/api/v1/time", timeout=2)
        response.raise_for_status()
        t4 = time.time_ns()

        server_time_data = response.json()
        t2_t3 = server_time_data.get("server_time")
        if not t2_t3:
            logger.error("[Time] Server time not in response.")
            return False

        round_trip_time = t4 - t1
        time_offset_ns = t2_t3 - (t1 + round_trip_time // 2)
        zero_point_ns = time.time_ns() + time_offset_ns

        try:
            response = _local_session.post(f"{SERVER}/api/v1/set_zero_time", timeout=1)
            response.raise_for_status()
        except requests.exceptions.RequestException as e:
            logger.error(f"[Time] Failed to notify server to set zero time: {e}")
            return False

        logger.info("[Time] Time synchronized and zero point set.")
        logger.info(f"[Time] RTT: {round_trip_time / 1_000_000:.2f} ms")
        logger.info(f"[Time] Clock offset: {time_offset_ns / 1_000_000:.2f} ms")
        logger.info(f"[Time] Zero point set to: {zero_point_ns}")
        return True

    except requests.exceptions.RequestException as e:
        logger.error(f"[Time] Failed to synchronize time with server: {e}")
        return False

def time_logging(event: str, audio_id: str):
    """Logs an event with a timestamp relative to the synchronized zero point."""
    try:
        if zero_point_ns == 0:
            return

        synced_time_ns = time.time_ns() + time_offset_ns
        relative_time_ms = (synced_time_ns - zero_point_ns) // 1_000_000

        log_filename = f"log_client.csv"
        log_filepath = os.path.join(local_session_dir or LOCAL_MEDIA_BASE, log_filename)

        if not os.path.exists(log_filepath):
            with open(log_filepath, "w") as log_file:
                log_file.write("timestamp_ms,event,audio_id\n")

        with open(log_filepath, "a") as log_file:
            log_file.write(f"{relative_time_ms},{event},{audio_id}\n")

    except Exception as e:
        logger.error(f"[Time] Failed to log event: {e}")

# === ASR helper ===
def _record_asr_turn(speaker: str, turn: int) -> None:
    global last_user_turn_for_asr, last_assistant_turn_for_asr
    if speaker == "user":
        last_user_turn_for_asr = turn
    elif speaker == "assistant":
        last_assistant_turn_for_asr = turn


def _resolve_turn_for_speaker(speaker: str) -> int:
    if speaker == "user" and last_user_turn_for_asr is not None:
        return last_user_turn_for_asr
    if speaker == "assistant" and last_assistant_turn_for_asr is not None:
        return last_assistant_turn_for_asr
    return get_current_turn()


def _on_asr_result(**kwargs) -> None:
    """Callback for chunk-level ASR result from ASR worker."""
    global _last_asr_key, _last_asr_text
    speaker = kwargs.get("speaker") or ""
    turn = int(kwargs.get("turn") or 0) or _resolve_turn_for_speaker(speaker)
    index = int(kwargs.get("index") or 0)
    offset_ms = int(kwargs.get("offset_ms") or 0)
    text = kwargs.get("text") or ""
    key = (speaker, turn)
    if _last_asr_key and key != _last_asr_key and _last_asr_text:
        prev_speaker, prev_turn = _last_asr_key
        live_transcripts.finalize(prev_speaker, prev_turn, _last_asr_text)
        _last_asr_key = None
        _last_asr_text = ""
    combined = live_transcripts.add_partial(speaker, turn, index, offset_ms, text)
    if combined:
        logger.debug("[ASR partial] %s turn=%s idx=%s: %s", speaker, turn, index, combined)
        _last_asr_key = key
        _last_asr_text = combined

        if slow_thinking_system:
            slow_thinking_system.request_run_soon(
                reason=f"asr_partial_{speaker}_turn{turn}"
            )


def start_asr_transcriber(log_dir: Optional[str]) -> None:
    """Start the background ASR worker."""
    global asr_worker
    if asr_worker:
        return
    model_path = os.environ.get("SHERPA_MODEL_DIR", DEFAULT_SHERPA_MODEL_DIR)
    try:
        asr_worker = SherpaASRWorker(
            model_dir=model_path,
            log_dir=log_dir or LOCAL_MEDIA_BASE,
            on_result=_on_asr_result,
        )
        asr_worker.start()
        ready = asr_worker.wait_until_ready(timeout=60.0)
        if ready:
            logger.info("Started ASR worker with model at %s", model_path)
        else:
            logger.warning("ASR worker did not signal ready within timeout; proceeding anyway")
    except Exception as exc:
        asr_worker = None
        logger.error("Failed to start ASR worker: %s", exc)


def stop_asr_transcriber() -> None:
    """Stop the background ASR worker."""
    global asr_worker
    try:
        if asr_worker:
            asr_worker.stop()
    finally:
        asr_worker = None


def prepare_local_dirs():
    """Create local cache directories for the current session."""
    global local_session_dir, local_user_audio_dir, local_assistant_audio_dir, local_image_dir, local_motion_dir
    base = os.path.join(LOCAL_MEDIA_BASE, start_session_label)
    local_session_dir = base
    local_user_audio_dir = os.path.join(base, "user_audio")
    local_assistant_audio_dir = os.path.join(base, "assistant_audio")
    local_motion_dir = os.path.join(base, "motion_data")
    local_image_dir = os.path.join(base, "images")
    for path in (base, local_user_audio_dir, local_assistant_audio_dir, local_motion_dir, local_image_dir):
        os.makedirs(path, exist_ok=True)
    logger.info("Local media cache initialised at %s", base)


def audio_to_wav_bytes(audio_np: np.ndarray, sample_rate: int) -> bytes:
    """Encode a NumPy audio array as WAV bytes."""
    buffer = io.BytesIO()
    sf.write(buffer, audio_np, sample_rate, format='WAV')
    return buffer.getvalue()


def pcm16_to_wav_bytes(pcm_bytes: bytes, sample_rate: int) -> bytes:
    """Pack raw PCM16 bytes into a WAV container."""
    buffer = io.BytesIO()
    with wave.open(buffer, 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(SAMPLE_WIDTH)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm_bytes)
    return buffer.getvalue()


def wav_bytes_to_base64(wav_bytes: bytes) -> str:
    """Encode WAV bytes as Base64 text."""
    return base64.b64encode(wav_bytes).decode('utf-8')


def save_audio_bytes_local(wav_bytes: bytes, speaker: str, turn: int, index: int, offset_ms: int, sample_rate: int) -> None:
    """Save one audio segment to the session cache."""
    directory = local_user_audio_dir if speaker == "user" else local_assistant_audio_dir
    if not directory:
        return
    filename = f"{speaker}_turn{turn:02d}_idx{index:03d}_t{offset_ms:06d}.wav"
    path = os.path.join(directory, filename)
    _record_asr_turn(speaker, turn)
    if asr_worker:
        try:
            asr_worker.submit(path, sample_rate, speaker, turn, index, offset_ms, audio_bytes=wav_bytes)
        except Exception as exc:
            logger.error("ASR submit failed for %s chunk idx=%s: %s", speaker, index, exc)

    try:
        with open(path, 'wb') as f:
            f.write(wav_bytes)
        logger.info(
            "Saved %s audio segment: turn=%s idx=%s offset=%sms size=%d bytes",
            speaker,
            turn,
            index,
            offset_ms,
            len(wav_bytes),
        )
        if speaker == 'assistant':
            alternate_filename = f"output_{index}.wav"
            with open(os.path.join(directory, alternate_filename), 'wb') as f:
                f.write(wav_bytes)

        full_filename = f"{speaker}_turn{turn:02d}.wav"
        full_path = os.path.join(directory, full_filename)
        try:
            new_data, sr = sf.read(io.BytesIO(wav_bytes))
            if os.path.exists(full_path):
                existing_data, existing_sr = sf.read(full_path)
                combined_data = np.concatenate((existing_data, new_data))
                sf.write(full_path, combined_data, existing_sr)
            else:
                sf.write(full_path, new_data, sr)
        except Exception as e:
            logger.error("Failed to append to full wav %s: %s", full_filename, e)
    except Exception as exc:
        logger.error("Failed to save local %s audio chunk idx=%s: %s", speaker, index, exc)
        return


def save_image_local(image_bytes: bytes, turn: int, index: int, offset_ms: int) -> None:
    """Save one captured image to the session cache."""
    if not local_image_dir:
        return
    filename = f"image_turn{turn:02d}_idx{index:03d}_t{offset_ms:06d}.jpg"
    path = os.path.join(local_image_dir, filename)
    try:
        with open(path, 'wb') as f:
            f.write(image_bytes)
        logger.info("Saved image id %s at %sms", index, offset_ms)
    except Exception as exc:
        logger.error("Failed to save local image at %sms: %s", offset_ms, exc)



def emit_user_segment(segment: np.ndarray) -> None:
    """Encode and upload one user speech segment."""
    if segment.size < MIN_USER_SAMPLES:
        logger.debug(
            "Skip user segment: too short (%d samples < %d)",
            segment.size,
            MIN_USER_SAMPLES,
        )
        return
    offset_ms = allocate_offset_ms()
    turn = mark_user_audio_activity()
    idx = next_segment_index(offset_ms, "user", turn)
    wav_bytes = audio_to_wav_bytes(segment, INPUT_SAMPLE_RATE)
    audio_b64 = wav_bytes_to_base64(wav_bytes)
    logger.debug("Emitting user segment: turn=%s idx=%s offset=%sms size=%d samples", turn, idx, offset_ms, segment.size)
    save_audio_bytes_local(wav_bytes, "user", turn, idx, offset_ms, INPUT_SAMPLE_RATE)
    send_audio_chunk(
        audio_b64,
        speaker="user",
        index=idx,
        turn=turn,
        sample_rate=INPUT_SAMPLE_RATE,
        offset_ms=offset_ms,
    )


def flush_user_buffer() -> None:
    """Upload any remaining buffered user audio."""
    global user_buffer
    with user_buffer_lock:
        if user_buffer.size == 0:
            return
        segment = user_buffer.copy()
        user_buffer = np.empty((0,), dtype='float32')
    emit_user_segment(segment)


def _cancel_manual_interrupt_timer() -> None:
    global _manual_interrupt_timer
    if _manual_interrupt_timer is not None:
        _manual_interrupt_timer.cancel()
        _manual_interrupt_timer = None


def _manual_interrupt_timeout() -> None:
    global manual_interrupt_active, is_receiving_response
    with is_receiving_lock:
        if manual_interrupt_active and is_receiving_response:
            logger.warning("Manual interrupt timed out; resuming microphone input.")
            is_receiving_response = False
    manual_interrupt_active = False
    _cancel_manual_interrupt_timer()


def _schedule_manual_interrupt_fail_safe() -> None:
    global _manual_interrupt_timer
    _cancel_manual_interrupt_timer()
    timer = threading.Timer(MANUAL_INTERRUPT_FAILSAFE_SEC, _manual_interrupt_timeout)
    timer.daemon = True
    timer.start()
    _manual_interrupt_timer = timer


def request_agent_interrupt(source: str) -> bool:
    """Force the assistant to speak by faking an assist turn."""
    global manual_interrupt_active, is_receiving_response
    if gpt_session.is_stopped():
        logger.warning("Interrupt request (%s) ignored; Realtime session is not active.", source)
        return False
    with is_receiving_lock:
        if is_receiving_response:
            logger.info("Interrupt request (%s) skipped; already receiving response.", source)
            return False
        is_receiving_response = True
        manual_interrupt_active = True
    logger.info("Requesting agent interrupt via %s", source)

    if dialogue_prefill_manager:
        dialogue_prefill_manager.dispatch_before_agent()

    gpt_session.enqueue_interrupt_silence()
    _schedule_manual_interrupt_fail_safe()

    return True


def next_segment_index(cur_relative_time: int, speaker: str, turn: int) -> int:
    """Allocate a global segment index shared by user and agent audio."""
    global segment_index, last_segment_speaker, last_segment_turn
    with segment_index_lock, segment_meta_lock:
        segment_index += 1
        last_segment_speaker = speaker
        last_segment_turn = turn
        return segment_index


def allocate_offset_ms() -> int:
    """Return the session-relative offset, with the initial one-second shift."""
    global session_start_time, first_local_save_done
    with session_time_lock:
        if not first_local_save_done:
            session_start_time = time.time() - SESSION_INITIAL_OFFSET_SEC
            first_local_save_done = True
        return int((time.time() - session_start_time) * 1000)


def send_audio_chunk(audio_b64: str, speaker: str, index: int, turn: int, sample_rate: int, offset_ms: int) -> None:
    timestamp = str(int(time.time() * 1000))
    payload = {
        "messages": [
            {
                "role": speaker,
                "content": [
                    {
                        "type": "input_audio",
                        "input_audio": {
                            "data": audio_b64,
                            "format": "wav",
                            "timestamp": timestamp,
                            "speaker": speaker,
                            "index": index,
                            "offset_ms": offset_ms,
                            "turn": turn,
                            "sample_rate": sample_rate,
                        },
                    }
                ],
            }
        ]
    }
    t0 = time.time()
    try:
        res = _local_session.post(f"{SERVER}/api/v1/stream", headers=HEADERS, json=payload, timeout=10)
        res.raise_for_status()
        dt = time.time() - t0
        logger.debug("Sent %s audio chunk idx=%s status=%s duration=%.2fms, response time=%.2fms",
                 speaker, index, res.status_code, dt * 1000, res.elapsed.total_seconds() * 1000)
        time_logging(f'AUDIO_SENT_{speaker}', index)
    except Exception as exc:
        logger.error("Failed to send %s audio idx=%s: %s", speaker, index, exc)


def extract_usage_tokens(usage_obj: Any) -> Dict[str, Any]:
    """Extract token usage metadata from an SDK response object."""

    def _get_value(obj: Any, keys: List[str]) -> int:
        for key in keys:
            val = getattr(obj, key, None)
            if val is None and isinstance(obj, dict):
                val = obj.get(key)
            if val is not None:
                try:
                    return int(val)
                except (TypeError, ValueError):
                    continue
        return 0

    input_tokens = _get_value(usage_obj, [
        "input_token_count",
        "prompt_token_count",
        "input_tokens",
        "prompt_tokens",
        "image_count"
    ])

    output_tokens = _get_value(usage_obj, [
        "output_token_count",
        "candidates_token_count",
        "candidates_tokens",
        "output_tokens",
        "text_count"
    ])

    total_tokens = _get_value(usage_obj, [
        "total_token_count",
        "total_tokens",
    ])

    if total_tokens == 0:
        total_tokens = input_tokens + output_tokens

    breakdown: List[Tuple[str, int]] = []
    details = getattr(usage_obj, "response_tokens_details", None)
    if details is None and isinstance(usage_obj, dict):
        details = usage_obj.get("response_tokens_details")

    if details:
        try:
            iterator = list(details)
        except TypeError:
            iterator = []
        for detail in iterator:
            modality = getattr(detail, "modality", None)
            count = getattr(detail, "token_count", None)
            if modality is None and isinstance(detail, dict):
                modality = detail.get("modality")
                count = detail.get("token_count")
            if modality is None or count is None:
                continue
            try:
                breakdown.append((str(modality), int(count)))
            except (TypeError, ValueError):
                continue

    return {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": total_tokens,
        "breakdown": breakdown,
    }


def send_turn_complete(turn_index: int) -> None:
    payload = {
        "messages": [
            {
                "role": "system",
                "content": [
                    {
                        "type": "turn_complete",
                        "turn_complete": {
                            "turn": turn_index,
                        },
                    }
                ],
            }
        ]
    }
    t0 = time.time()
    try:
        res = _local_session.post(f"{SERVER}/api/v1/stream", headers=HEADERS, json=payload, timeout=10)
        res.raise_for_status()
        dt = time.time() - t0
        logger.debug("Sent turn_complete for turn=%s status=%s duration=%.2fms", turn_index, res.status_code, dt * 1000)
    except Exception as exc:
        logger.error("Failed to send turn_complete for turn=%s: %s", turn_index, exc)


def get_current_segment_index() -> int:
    with segment_index_lock:
        return segment_index

def get_current_turn() -> int:
    with turn_lock:
        return current_turn

def mark_user_audio_activity() -> int:
    global turn_user_audio_pending
    with turn_lock:
        turn_user_audio_pending = True
        return current_turn


def has_pending_user_audio() -> bool:
    with turn_lock:
        pending_flag = turn_user_audio_pending
    if pending_flag:
        return True
    with user_buffer_lock:
        return user_buffer.size >= MIN_USER_SAMPLES


def user_is_speaking() -> bool:
    """Return whether recent microphone activity indicates active user speech."""
    with user_buffer_lock:
        buffered_audio = user_buffer.size > 0
    recent_voice = (time.time() - last_user_voice_time) < USER_SPEECH_GRACE_SEC
    return buffered_audio or recent_voice


def mark_turn_complete() -> None:
    global current_turn, turn_user_audio_pending, user_buffer, _last_asr_key, _last_asr_text
    with turn_lock:
        turn_index = current_turn

    with user_buffer_lock:
        user_buffer = np.empty((0,), dtype='float32')
    live_transcripts.finalize("user", turn_index)
    live_transcripts.finalize("assistant", turn_index)
    if _last_asr_key and _last_asr_key[1] == turn_index:
        _last_asr_key = None
        _last_asr_text = ""

    with turn_lock:
        current_turn = turn_index + 1
        turn_user_audio_pending = False
    logger.info("Turn %s complete", turn_index)


# === OpenAI Realtime session ===
class GPTSession:
    """Manage the OpenAI Realtime session and its input/output queues."""

    def __init__(self):
        self.queue = gpt_input_queue
        self.output_queue = gpt_output_queue
        self._thread = threading.Thread(target=self._run, name="GPTSession", daemon=True)
        self._stop = threading.Event()
        self._ready = threading.Event()
        self.loop: Optional[asyncio.AbstractEventLoop] = None
        self.session = None
        self._user_item_text: Dict[str, str] = {}
        self._assistant_item_text: Dict[str, str] = {}
        self._remaining_audio_ms: float = 0.0
        self._last_audio_update: float = 0.0
        self._pending_agent_end: bool = False
        self._finalize_task: Optional[asyncio.Task] = None


        self._interrupted_item_ids: set = set()
        self._item_id_lock = threading.Lock()
        self._current_item_id: Optional[str] = None

    def start(self):
        self._thread.start()

    def wait_until_ready(self, timeout: float = 10.0) -> bool:
        return self._ready.wait(timeout)

    def stop(self):
        self._stop.set()
        try:
            self.queue.put_nowait(None)
        except queue.Full:
            pass
        if self.loop and self.session:
            def _close_session():
                try:
                    asyncio.create_task(self.session.close())
                except Exception:
                    pass
            try:
                self.loop.call_soon_threadsafe(_close_session)
            except Exception:
                pass
        if self._finalize_task:
            self._finalize_task.cancel()
            self._finalize_task = None

    def is_stopped(self) -> bool:
        return self._stop.is_set()

    def enqueue_audio(self, pcm_bytes: bytes):
        if self._stop.is_set():
            return
        try:
            self.queue.put_nowait({"type": "audio", "data": pcm_bytes})
        except queue.Full:
            logger.warning("Realtime input queue full; dropping audio chunk")

    def enqueue_media(self, mime_type: str, data: bytes):
        if self._stop.is_set():
            return
        try:
            self.queue.put_nowait({"type": "media", "mime_type": mime_type, "data": data})
        except queue.Full:
            logger.warning("Realtime input queue full; dropping frame")

    def enqueue_prefill(self, text: str) -> None:
        if self._stop.is_set():
            return
        payload = {
            "type": "prefill",
            "text": text.strip(),
        }
        if not payload["text"]:
            return
        try:
            self.queue.put_nowait(payload)
        except queue.Full:
            logger.warning("Realtime input queue full; dropping prefill")

    def enqueue_interrupt_silence(self, duration_ms: int = 1000) -> None:
        if self._stop.is_set():
            return
        try:
            self.queue.put_nowait({"type": "force_response"})
        except queue.Full:
            logger.warning("Realtime input queue full; dropping interrupt silence chunk")

    def enqueue_cancel_response(self) -> None:
        """Send a raw message to the GPT session."""
        if self._stop.is_set():
            return
        try:
            self.queue.put_nowait({"type": "cancel_response"})
            logger.debug("Enqueued cancel_response request")
        except queue.Full:
            logger.warning("Realtime input queue full; dropping cancel_response")

    def mark_item_interrupted(self, item_id: Optional[str]) -> None:
        """Return the playback start time for an item ID."""
        if not item_id:
            return
        with self._item_id_lock:
            self._interrupted_item_ids.add(item_id)
            logger.info("Marked item_id=%s as interrupted, will discard its audio", item_id)

    def is_item_interrupted(self, item_id: Optional[str]) -> bool:
        """item_id"""
        if not item_id:
            return False
        with self._item_id_lock:
            return item_id in self._interrupted_item_ids

    def clear_interrupted_items(self) -> None:
        """Return the turn associated with an item ID."""
        with self._item_id_lock:
            if self._interrupted_item_ids:
                logger.debug("Clearing %d interrupted item_ids", len(self._interrupted_item_ids))
            self._interrupted_item_ids.clear()

    def has_interrupted_items(self) -> bool:
        """item_id"""
        with self._item_id_lock:
            return len(self._interrupted_item_ids) > 0

    def get_current_item_id(self) -> Optional[str]:
        """item_id"""
        with self._item_id_lock:
            return self._current_item_id

    async def _send_image(self, session, mime_type: str, data: bytes) -> None:
        """Send image as conversation item without triggering an auto response."""
        data_url = f"data:{mime_type};base64,{base64.b64encode(data).decode('utf-8')}"
        raw_msg = {
            "type": "conversation.item.create",
            "other_data": {
                "item": {
                    "type": "message",
                    "role": "user",
                    "content": [
                        {
                            "type": "input_image",
                            "image_url": data_url,
                            "detail": "low",
                        }
                    ],
                },
            },
        }
        await session.model.send_event(
            RealtimeModelSendRawMessage(message=raw_msg)
        )

    def _cancel_finalize_task(self) -> None:
        if self._finalize_task:
            self._finalize_task.cancel()
            self._finalize_task = None

    def _decay_remaining(self) -> None:
        now = time.monotonic()
        if self._last_audio_update > 0:
            elapsed_ms = (now - self._last_audio_update) * 1000.0
            if elapsed_ms > 0:
                self._remaining_audio_ms = max(0.0, self._remaining_audio_ms - elapsed_ms)
        self._last_audio_update = now

    def _add_audio_duration(self, byte_len: int) -> None:
        self._decay_remaining()
        duration_ms = (byte_len / float(SAMPLE_WIDTH * OUTPUT_SAMPLE_RATE)) * 1000.0
        self._remaining_audio_ms += duration_ms
        self._pending_agent_end = False
        self._cancel_finalize_task()

    async def _finalize_agent_end(self, session) -> None:
        global is_receiving_response, manual_interrupt_active
        try:
            await session.model.send_event(
                RealtimeModelSendRawMessage(message={"type": "input_audio_buffer.clear"})
            )
        except Exception as exc:
            logger.debug("Failed to clear input audio buffer at turn end: %s", exc)
        logger.info("Predict this turn agent's audio has been completed. So open generation_complete event.")
        self.output_queue.put_nowait(None)
        with is_receiving_lock:
            if is_receiving_response:
                logger.info("Realtime response ended, set is_receiving_response = False")
                is_receiving_response = False
        manual_interrupt_active = False
        _cancel_manual_interrupt_timer()
        self._pending_agent_end = False
        self._remaining_audio_ms = 0.0
        self._last_audio_update = 0.0
        self._cancel_finalize_task()

    async def _wait_and_finalize(self, session) -> None:
        try:
            while not self._stop.is_set():
                self._decay_remaining()
                if self._remaining_audio_ms <= 0.0:
                    break
                await asyncio.sleep(0.05)
            if self._pending_agent_end and not self._stop.is_set():
                await self._finalize_agent_end(session)
        except asyncio.CancelledError:
            pass
        finally:
            self._finalize_task = None

    async def _apply_prefill(self, session, text: str) -> None:
        """Inject a one-shot system message as prefill."""
        payload = (text or "").strip()
        if not payload:
            return
        raw_msg = {
            "type": "conversation.item.create",
            "other_data": {
                "item": {
                    "type": "message",
                    "role": "system",
                    "content": [
                        {
                            "type": "input_text",
                            "text": payload,
                        }
                    ],
                }
            },
        }
        await session.model.send_event(
            RealtimeModelSendRawMessage(message=raw_msg)
        )

    async def _send_loop(self, session):
        loop = asyncio.get_running_loop()
        while not self._stop.is_set():
            item = await loop.run_in_executor(None, self.queue.get)
            if item is None:
                break
            try:
                if item["type"] == "audio":
                    await session.send_audio(item["data"])
                elif item["type"] == "media":
                    await self._send_image(session, item.get("mime_type", "image/jpeg"), item["data"])
                elif item["type"] == "prefill":
                    logger.info("Sending prefill text to Realtime API: %s", item["text"])
                    await self._apply_prefill(session, item["text"])
                elif item["type"] == "force_response":
                    logger.info("Forcing response via response.create (cancelling any active response first)")

                    try:
                        await session.model.send_event(
                            RealtimeModelSendRawMessage(message={"type": "response.cancel"})
                        )
                        logger.debug("Sent response.cancel")
                    except Exception as cancel_exc:
                        logger.debug("response.cancel failed (may be no active response): %s", cancel_exc)


                    try:
                        await session.model.send_event(
                            RealtimeModelSendRawMessage(message={"type": "input_audio_buffer.clear"})
                        )
                        logger.debug("Cleared input audio buffer")
                    except Exception as clear_exc:
                        logger.debug("input_audio_buffer.clear failed: %s", clear_exc)


                    await asyncio.sleep(0.1)


                    await session.model.send_event(
                        RealtimeModelSendRawMessage(message={"type": "response.create"})
                    )
                    logger.info("Sent response.create")
                    if last_response_create_time is not None:
                        last_response_create_time.value = time.time()
                elif item["type"] == "cancel_response":

                    logger.info("Cancelling GPT response due to user interruption")
                    try:
                        await session.model.send_event(
                            RealtimeModelSendRawMessage(message={"type": "response.cancel"})
                        )
                        logger.info("Sent response.cancel to GPT")
                    except Exception as cancel_exc:
                        logger.debug("response.cancel failed (may be no active response): %s", cancel_exc)


                    try:
                        await session.model.send_event(
                            RealtimeModelSendRawMessage(message={"type": "input_audio_buffer.clear"})
                        )
                        logger.debug("Cleared input audio buffer after cancel")
                    except Exception as clear_exc:
                        logger.debug("input_audio_buffer.clear failed: %s", clear_exc)
            except Exception as exc:
                logger.error("Failed sending realtime input: %s", exc)
                break

    def _handle_history_item(self, item) -> None:
        global _last_asr_key, _last_asr_text
        if getattr(item, "role", None) not in ("user", "assistant"):
            return
        texts: List[str] = []
        for entry in getattr(item, "content", []):
            entry_type = getattr(entry, "type", None)
            if entry_type in ("input_text", "text"):
                text_val = getattr(entry, "text", None)
                if text_val:
                    texts.append(text_val)
            elif entry_type in ("input_audio", "audio"):
                transcript = getattr(entry, "transcript", None)
                if transcript:
                    texts.append(transcript)
        if not texts:
            return
        combined = "\n".join([t.strip() for t in texts if t and t.strip()])
        if not combined:
            return
        if item.role == "user":
            prev = self._user_item_text.get(item.item_id)
            if prev == combined:
                return
            self._user_item_text[item.item_id] = combined
            finalized_agent, agent_text = conversation_tracker.update_user_transcription(combined)
            logger.info("[Realtime user transcript] %s %s", combined, agent_text or "")

            if viz_queue:
                try:
                    viz_queue.put({
                        "type": "asr",
                        "data": {"user": combined}
                    })
                except Exception:
                    pass

            turn_for_final = _resolve_turn_for_speaker("user")
            live_transcripts.finalize("user", turn_for_final, combined)
            if _last_asr_key == ("user", turn_for_final):
                _last_asr_key = None
                _last_asr_text = ""
        else:
            prev = self._assistant_item_text.get(item.item_id)
            if prev == combined:
                return
            self._assistant_item_text[item.item_id] = combined
            finalized_user, user_text = conversation_tracker.update_agent_transcription(combined)
            logger.info("[Realtime agent transcript] %s %s", combined, user_text or "")

            if viz_queue:
                try:
                    viz_queue.put({
                        "type": "asr",
                        "data": {"agent": combined}
                    })
                except Exception:
                    pass

            turn_for_final = _resolve_turn_for_speaker("assistant")
            live_transcripts.finalize("assistant", turn_for_final, combined)
            agent_playback_tracker.update_transcript(combined, turn_for_final)
            if _last_asr_key == ("assistant", turn_for_final):
                _last_asr_key = None
                _last_asr_text = ""

    async def _receive_loop(self, session):
        global is_receiving_response, manual_interrupt_active
        while not self._stop.is_set():
            try:
                async for event in session:
                    if event.type == "audio":
                        data = getattr(event.audio, "data", None)

                        item_id = getattr(event, "item_id", None)

                        if data:

                            if item_id and self.is_item_interrupted(item_id):
                                logger.info("[AudioFilter] Discarding audio from interrupted item_id=%s", item_id)
                                continue


                            with self._item_id_lock:
                                if item_id and self._current_item_id != item_id:
                                    logger.info("New audio item started: item_id=%s (prev=%s)",
                                               item_id, self._current_item_id)
                                    self._current_item_id = item_id

                            self._add_audio_duration(len(data))
                            with is_receiving_lock:


                                if not is_receiving_response and not (interruption_manager and interruption_manager.is_interrupted()):
                                    logger.info("Realtime response audio started, set is_receiving_response = True")
                                    is_receiving_response = True

                            self.output_queue.put_nowait({"audio_data": data, "item_id": item_id})
                    elif event.type == "audio_end":
                        logger.info("Realtime receive audio_end event")
                        self.output_queue.put_nowait({"event": "generation_complete"})
                    elif event.type == "audio_interrupted":


                        if interruption_manager:

                            #




                            if not manual_interrupt_active:

                                logger.info(
                                    "Ignoring GPT audio_interrupted event (manual_interrupt_active=False). "
                                    "Use hotkey 'i' to manually interrupt agent."
                                )
                                continue


                            interrupt_type = InterruptionType.AGENT_SELF_INTERRUPT
                            logger.info("Realtime audio_interrupted event received (AGENT_SELF_INTERRUPT)")


                            current_item = self.get_current_item_id()
                            if current_item:
                                self.mark_item_interrupted(current_item)
                                logger.info("Marked current item_id=%s as interrupted", current_item)






                            current_audio_id = get_current_segment_index() or None

                            interruption_manager.trigger_interruption(
                                interrupt_type=interrupt_type,
                                source_turn=get_current_turn(),
                                target_turn=get_current_turn(),
                                audio_position_ms=int(agent_playback_tracker.get_state().get("elapsed_sec", 0) * 1000),
                                current_audio_id=current_audio_id,
                            )


                            self.output_queue.put_nowait({
                                "event": "audio_interrupted",
                                "interrupt_type": interrupt_type.value,
                            })
                        else:

                            logger.info("Realtime audio interrupted")
                            agent_playback_tracker.end_turn()
                    elif event.type in ("history_added", "history_updated"):
                        items = []
                        if event.type == "history_added":
                            items = [event.item]
                        else:
                            items = getattr(event, "history", []) or []
                        for item in items:
                            self._handle_history_item(item)
                    elif event.type == "agent_end":
                        logger.info("Realtime receive agent_end event")
                        self._decay_remaining()
                        self._pending_agent_end = True
                        self._cancel_finalize_task()
                        if self._remaining_audio_ms <= 0.0:
                            await self._finalize_agent_end(session)
                        else:
                            self._finalize_task = asyncio.create_task(self._wait_and_finalize(session))
                        continue
                    elif event.type == "input_audio_timeout_triggered":
                        logger.info("Realtime input audio timeout triggered")
                    elif event.type == "error":
                        logger.error("Realtime error: %s", event.error)
                    elif event.type == "raw_model_event":
                        raw_data = getattr(event, "data", None)
                        raw_type = None
                        if isinstance(raw_data, dict):
                            raw_type = raw_data.get("type")
                        else:
                            raw_type = getattr(raw_data, "type", None)
                        if raw_type == "input_audio_buffer.committed":
                            if dialogue_prefill_manager:
                                dialogue_prefill_manager.dispatch_before_agent()
                        elif raw_type == "response.created":
                            if last_response_create_time is not None:
                                last_response_create_time.value = time.time()
                            logger.info("Realtime server event: response.created")

            except Exception as exc:
                logger.error("Realtime receive loop error: %s", exc)
                break


    async def _main(self):
        retry_count = 0
        while not self._stop.is_set():
            try:
                runner = RealtimeRunner(agent)
                async with await runner.run(model_config={**REALTIME_MODEL_CONFIG, "playback_tracker": playback_tracker}) as session:
                    self.session = session
                    self.loop = asyncio.get_running_loop()
                    self._ready.set()
                    retry_count = 0

                    truncation_update = {
                        "type": "session.update",
                        "session": {
                            "truncation": {
                                "type": "retention_ratio",
                                "retention_ratio": 0.7
                            },
                            "max_response_output_tokens": 4096
                        }
                    }
                    try:
                        await session.model.send_event(
                            RealtimeModelSendRawMessage(message=truncation_update)
                        )
                        logger.info("Sent truncation settings: retention_ratio=0.7, max_response_output_tokens=4096")
                    except Exception as exc:
                        logger.warning("Failed to send truncation settings: %s", exc)
                    await asyncio.gather(
                        self._send_loop(session),
                        self._receive_loop(session),
                    )
            except (ConnectionResetError, ConnectionError, BrokenPipeError) as exc:
                if self._stop.is_set():
                    break
                logger.warning("Realtime session [Errno 104] Connection reset by peer (or similar). Retrying in 2 seconds... (Retry #%d): %s", retry_count + 1, exc)
                retry_count += 1
                self._ready.clear()
                self.session = None
                await asyncio.sleep(2.0)
                continue
            except Exception as exc:
                if self._stop.is_set():
                    break
                logger.error("Realtime session terminated with fatal error: %s", exc)
                break
            finally:
                self._ready.clear()
                self.loop = None
                self.session = None


        self.output_queue.put_nowait(None)
        logger.info("Realtime session _main thread exiting.")

    def _run(self):
        asyncio.run(self._main())


gpt_session = GPTSession()
dialogue_prefill_manager = DialoguePrefillManager(gpt_session)
blender_client = None


_last_audio_enqueue_log_time = 0.0

def audio_callback(indata, frames, time_info, status):
    """Process microphone frames for the Realtime session."""
    global _last_audio_enqueue_log_time, _last_user_audio_sent_time, last_user_voice_time
    if status:
        logger.warning("Audio input status: %s", status)

    # Record recent voice activity before the playback gates below. The
    # interruption guard must still see microphone speech while the agent or
    # robot is speaking, even when those frames are not forwarded upstream.
    if indata.ndim == 2 and indata.shape[1] > 1:
        mono = indata.mean(axis=1)
    else:
        mono = indata[:, 0] if indata.ndim == 2 else indata
    mono_f = mono.astype(np.float32)
    if mono_f.size:
        rms = float(np.sqrt(np.mean(np.square(mono_f))))
        if rms >= USER_SPEECH_RMS_THRESHOLD:
            last_user_voice_time = time.time()

    robot_playing = False
    if is_robot_playing is not None:
        robot_playing = is_robot_playing.value

    mujoco_playing = False
    if is_mujoco_playing is not None:
        mujoco_playing = is_mujoco_playing.value

    with is_receiving_lock:
        if is_receiving_response or robot_playing or mujoco_playing:
            return
    if status:
        print(status)

    # indata shape: (frames, channels), dtype: int16.
    # resampled = resample_poly(mono_f, up=1, down=2)
    resampled = soxr.resample(mono_f, INPUT_RAW_SAMPLE_RATE, INPUT_SAMPLE_RATE, quality='HQ')
    # resampled = np.clip(resampled, -32768, 32767)


    pcm16 = resampled.astype(np.int16).tobytes()

    if gpt_session:
        gpt_session.enqueue_audio(pcm16)
        _last_user_audio_sent_time = time.time()

        now = time.time()
        if now - _last_audio_enqueue_log_time > 5.0:
            _last_audio_enqueue_log_time = now
            logger.info("[AudioDebug] Sending audio to GPT, turn=%s, waiting for GPT response...", get_current_turn())
    resampled = resampled / 32768.0
    # resampled = resampled * 1.5
    input_audio_queue.put(resampled)
    # logger.debug('resampled avg, variance:', np.mean(resampled), np.std(resampled))

def audio_upload_loop():
    """Collect input frames into one-second segments and upload them."""
    global user_buffer
    while True:
        chunk = input_audio_queue.get()
        if chunk is None:
            flush_user_buffer()
            break
        if not isinstance(chunk, np.ndarray) or not len(chunk):
            continue
        segments: list[np.ndarray] = []
        with user_buffer_lock:
            user_buffer = np.concatenate([user_buffer, chunk])
            while user_buffer.size >= SEGMENT_SIZE:
                segment = user_buffer[:SEGMENT_SIZE].copy()
                user_buffer = user_buffer[SEGMENT_SIZE:]
                segments.append(segment)

        robot_playing = False
        if is_robot_playing is not None:
            robot_playing = is_robot_playing.value

        mujoco_playing = False
        if is_mujoco_playing is not None:
            mujoco_playing = is_mujoco_playing.value

        with is_receiving_lock:
            if is_receiving_response or robot_playing or mujoco_playing:
                continue

        for segment in segments:
            emit_user_segment(segment)


class AgentAudioAccumulator:
    """Accumulate agent response audio for local persistence."""
    def __init__(self, sample_rate: int, segment_seconds: float = 1.0, idle_flush_ms: int = 500):
        self.sample_rate = sample_rate
        self.segment_bytes = int(sample_rate * segment_seconds) * SAMPLE_WIDTH
        self.buffer = bytearray()
        self.lock = threading.Lock()
        self.idle_flush_ms = idle_flush_ms
        self._last_feed = time.monotonic()

    def add_chunk(self, chunk: bytes):
        with self.lock:
            self._last_feed = time.monotonic()
            self.buffer.extend(chunk)
            while len(self.buffer) >= self.segment_bytes:
                logger.debug("Cur assistant output cache chunk has above 1s, now is=%s", len(self.buffer))
                slab = bytes(self.buffer[:self.segment_bytes])
                del self.buffer[:self.segment_bytes]
                if len(slab) < MIN_ASSIST_PCM_BYTES:
                    logger.debug(
                        "Skip assistant chunk: %d bytes < %d",
                        len(slab),
                        MIN_ASSIST_PCM_BYTES,
                    )
                    continue
                offset_ms = allocate_offset_ms()
                turn = get_current_turn()
                idx = next_segment_index(offset_ms, "assistant", turn)
                wav_bytes = pcm16_to_wav_bytes(slab, self.sample_rate)
                save_audio_bytes_local(wav_bytes, "assistant", turn, idx, offset_ms, self.sample_rate)
                time_logging('AUDIO_SAVED_assistant', idx)
                audio_b64 = wav_bytes_to_base64(wav_bytes)
                send_audio_chunk(
                    audio_b64,
                    speaker="assistant",
                    index=idx,
                    turn=turn,
                    sample_rate=self.sample_rate,
                    offset_ms=offset_ms,
                )

    def flush(self):
        slab = None
        with self.lock:
            if not self.buffer:
                return
            slab = bytes(self.buffer)
            self.buffer.clear()
        if len(slab) < MIN_ASSIST_PCM_BYTES:
            logger.debug(
                "Skip assistant residual chunk: %d bytes < %d",
                len(slab),
                MIN_ASSIST_PCM_BYTES,
            )
            return

        offset_ms = allocate_offset_ms()
        turn = get_current_turn()
        idx = next_segment_index(offset_ms, "assistant", turn)
        logger.debug("Flushing remaining assistant output cache chunk, size=%s, at idx=%s", len(slab), idx)
        wav_bytes = pcm16_to_wav_bytes(slab, self.sample_rate)
        save_audio_bytes_local(wav_bytes, "assistant", turn, idx, offset_ms, self.sample_rate)
        audio_b64 = wav_bytes_to_base64(wav_bytes)
        send_audio_chunk(
            audio_b64,
            speaker="assistant",
            index=idx,
            turn=turn,
            sample_rate=self.sample_rate,
            offset_ms=offset_ms,
        )
        send_turn_complete(turn)

    def flush_if_idle(self):
        slab = None
        with self.lock:
            if not self.buffer:
                return
            if (time.monotonic() - self._last_feed) * 1000 >= self.idle_flush_ms:
                slab = bytes(self.buffer)
                self.buffer.clear()
        if slab:
            offset_ms = allocate_offset_ms()
            turn = get_current_turn()
            idx = next_segment_index(offset_ms, "assistant", turn)
            logger.debug("Ahead of generation_complete Flushing remaining assistant output cache chunk, size=%s, at idx=%s", len(slab), idx)
            wav_bytes = pcm16_to_wav_bytes(slab, self.sample_rate)
            save_audio_bytes_local(wav_bytes, "assistant", turn, idx, offset_ms, self.sample_rate)
            audio_b64 = wav_bytes_to_base64(wav_bytes)
            send_audio_chunk(audio_b64, "assistant", idx, turn, self.sample_rate, offset_ms)
            send_turn_complete(turn)

    def clear(self):
        """Clear accumulated audio after an interruption."""
        with self.lock:
            self.buffer.clear()
            logger.info("AgentAudioAccumulator buffer cleared due to interruption")


agent_accumulator = AgentAudioAccumulator(OUTPUT_SAMPLE_RATE)
playback_queue: "queue.Queue[Optional[np.ndarray]]" = queue.Queue()
# The local sounddevice sink is used for ``--render_target none`` and as a
# fallback when the requested external renderer cannot be started.  Main
# updates this after renderer startup; keeping the default enabled makes the
# playback loop useful for standalone/offline callers as well.
local_audio_playback_enabled = True


def _start_output_stream() -> Optional[sd.OutputStream]:
    try:
        stream = sd.OutputStream(
            samplerate=OUTPUT_SAMPLE_RATE,
            channels=OUTPUT_CHANNELS,
            dtype="float32",
        )
        stream.start()
        return stream
    except Exception as exc:
        logger.error("Failed to start output stream: %s", exc)
        return None


def _clear_playback_queue() -> int:
    """Drop locally queued samples after an interruption.

    The external renderers have their own interruption handling.  Samples
    waiting in this process must be removed as well, otherwise stale response
    audio would be written after the user has taken the turn back.
    """
    cleared = 0
    while True:
        try:
            playback_queue.get_nowait()
        except pyqueue.Empty:
            return cleared
        cleared += 1


def _playback_worker(stream: Optional[sd.OutputStream]) -> None:
    if stream is None:
        return
    try:
        while True:
            data = playback_queue.get()
            if data is None:
                break
            try:
                stream.write(data)
            except Exception as exc:
                logger.error("Playback worker failed: %s", exc)
                break
    finally:
        try:
            stream.stop()
            stream.close()
        except Exception:
            pass


def playback_loop():
    """Play Realtime audio responses and persist local segments."""
    global is_receiving_response, manual_interrupt_active, _last_gpt_response_time
    stream = _start_output_stream() if local_audio_playback_enabled else None
    playback_thread: Optional[threading.Thread] = None
    if stream is not None:
        playback_thread = threading.Thread(
            target=_playback_worker,
            args=(stream,),
            name="PlaybackWorker",
            daemon=True,
        )
        playback_thread.start()

    idle_timeout = 0.50
    last_timeout_warning_time = 0.0

    try:
        receiving_audio = False
        last_non_interrupted_item_id = None
        while True:
            try:
                chunk = gpt_output_queue.get(timeout=idle_timeout)
            except pyqueue.Empty:
                agent_accumulator.flush_if_idle()

                now = time.time()
                if (_last_user_audio_sent_time > 0 and
                    _last_gpt_response_time < _last_user_audio_sent_time and
                    now - _last_user_audio_sent_time > GPT_RESPONSE_TIMEOUT_SEC and
                    now - last_timeout_warning_time > 10.0):
                    last_timeout_warning_time = now
                    elapsed = now - _last_user_audio_sent_time
                    logger.warning(
                        "[GPT Timeout] No GPT response for %.1fs after user audio. "
                        "Turn=%s, last_user_audio=%.1fs ago, last_gpt_response=%.1fs ago",
                        elapsed,
                        get_current_turn(),
                        now - _last_user_audio_sent_time if _last_user_audio_sent_time > 0 else -1,
                        now - _last_gpt_response_time if _last_gpt_response_time > 0 else -1,
                    )
                continue
            if isinstance(chunk, dict):
                if chunk.get("event") == "generation_complete":
                    logger.debug("[Playback] Received generation_complete event")
                    agent_accumulator.flush()
                    agent_playback_tracker.mark_generation_complete()



                    if interruption_manager is not None:
                        with is_receiving_lock:
                            if is_receiving_response:
                                logger.info("Generation complete (with interruption support), set is_receiving_response = False")
                                is_receiving_response = False
                    continue
                elif chunk.get("event") == "audio_interrupted":

                    interrupt_type_str = chunk.get("interrupt_type", "user_interrupt")
                    is_agent_self = interrupt_type_str == "agent_self"

                    if is_agent_self:
                        logger.info("[Playback] AGENT_SELF_INTERRUPT: clearing queues, new agent response incoming")
                    else:
                        logger.info("[Playback] USER_INTERRUPT: clearing queues, returning control to user")


                    agent_accumulator.clear()
                    _clear_playback_queue()
                    agent_playback_tracker.end_turn()
                    receiving_audio = False





                    if not is_agent_self:
                        old_turn = get_current_turn()
                        mark_turn_complete()
                        logger.info("[Playback] Turn %s marked complete due to USER interruption, new turn=%s",
                                   old_turn, get_current_turn())
                    else:
                        logger.info("[Playback] AGENT_SELF_INTERRUPT: turn NOT incremented (same turn continues)")




                    if is_agent_self:


                        logger.info("[Playback] Keeping is_receiving_response=True for new agent response")
                    else:

                        with is_receiving_lock:
                            is_receiving_response = False
                        logger.info("[Playback] Set is_receiving_response=False, user can speak now")

                    manual_interrupt_active = False
                    _cancel_manual_interrupt_timer()
                    continue
                elif chunk.get("event") == "manual_interrupt_complete":



                    logger.info("[Playback] Manual interrupt complete: resetting receiving_audio to False")
                    _clear_playback_queue()
                    receiving_audio = False



                    continue
                usage = chunk.get("usage")
                if usage:
                    breakdown = usage.get("breakdown", [])
                    breakdown_text = ", ".join(f"{modality}:{count}" for modality, count in breakdown)
                    if breakdown_text:
                        breakdown_text = f"; breakdown: {breakdown_text}"
                    logger.info(
                        "Turn %s token usage -> total=%s, input=%s, output=%s%s",
                        get_current_turn(),
                        usage.get("total_tokens", 0),
                        usage.get("input_tokens", 0),
                        usage.get("output_tokens", 0),
                        breakdown_text,
                    )
                    continue



            audio_data = None
            item_id = None
            if isinstance(chunk, dict) and "audio_data" in chunk:
                audio_data = chunk.get("audio_data")
                item_id = chunk.get("item_id")

                if item_id and gpt_session.is_item_interrupted(item_id):
                    logger.info("[Playback] Discarding audio from interrupted item_id=%s", item_id)
                    continue


                if item_id and item_id != last_non_interrupted_item_id:
                    if gpt_session.has_interrupted_items():
                        logger.info("[Playback] New item_id=%s arrived, clearing interrupted items", item_id)
                        gpt_session.clear_interrupted_items()
                    last_non_interrupted_item_id = item_id
            elif isinstance(chunk, bytes):

                audio_data = chunk

            if chunk is None:
                if receiving_audio:
                    logger.debug("[2] Received realtime output none chunk")
                    mark_turn_complete()
                    agent_playback_tracker.end_turn()

                    gpt_session.clear_interrupted_items()
                    with is_receiving_lock:
                        if is_receiving_response:
                            logger.info("Realtime response audio ended, set is_receiving_response = False")
                            is_receiving_response = False
                    manual_interrupt_active = False
                    _cancel_manual_interrupt_timer()
                    if dialogue_prefill_manager:
                        dialogue_prefill_manager.dispatch_before_user()
                    receiving_audio = False
                if gpt_session.is_stopped():
                    break
                continue
            if not audio_data:
                continue
            logger.debug("[1] Received realtime output chunk: %d bytes, item_id=%s", len(audio_data), item_id)
            _last_gpt_response_time = time.time()
            agent_accumulator.add_chunk(audio_data)
            agent_playback_tracker.add_audio_bytes(len(audio_data))

            data = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0
            if not receiving_audio:
                receiving_audio = True

                if interruption_manager and interruption_manager.is_waiting_for_new_audio():
                    interruption_manager.notify_new_audio_received()
                    logger.info("New agent audio received after interruption")

                with is_receiving_lock:


                    if not is_receiving_response and not (interruption_manager and interruption_manager.is_interrupted()):
                        logger.info("Realtime response audio started, set is_receiving_response = True")
                        is_receiving_response = True

                manual_interrupt_active = False
                _cancel_manual_interrupt_timer()

            if playback_thread and playback_thread.is_alive():
                playback_queue.put(data)
    finally:
        if playback_thread:
            playback_queue.put(None)
        if playback_thread:
            playback_thread.join(timeout=2.0)
            if playback_thread.is_alive():
                logger.warning("Playback worker did not stop within timeout")
        _clear_playback_queue()


# === Motion listener ===
def _build_ws_url(base_http: str, path: str, uid: str) -> str:
    scheme = "wss" if base_http.startswith("https") else "ws"
    host = base_http.split("://", 1)[1].rstrip("/")
    path = path if path.startswith("/") else f"/{path}"
    return f"{scheme}://{host}{path}?uid={uid}"


def _decode_motion_b64(motion_b64: str) -> np.ndarray:
    raw = base64.b64decode(motion_b64)
    bio = io.BytesIO(raw)
    return np.load(bio, allow_pickle=False)


def _ensure_dir(path: str) -> None:
    if path:
        os.makedirs(path, exist_ok=True)


def _motion_filename(audio_id: Optional[int], ts: float) -> str:
    if audio_id is not None:
        return f"motion_{audio_id}.npy"
    return f"motion_{int(ts * 1000)}.npy"

kPi = 3.141592654
kPi_2 = 1.57079632

_root_last_pos = None
_root_last_yaw = 0.0

def to_bvh_diffusion(anim_clip, save_path, data_pipe, framerate):
    global _root_last_pos, _root_last_yaw
    shape = anim_clip.shape
    if len(shape) == 2:
        anim_clip = anim_clip.reshape((1, shape[0], shape[1]))
        shape = anim_clip.shape
    elif len(shape) != 3:
        raise ValueError(f'anim_clip should be 2D or 3D, got {len(shape)}D')
    combine_anim_clip = anim_clip.reshape((shape[0] * shape[1], shape[2]))

    combine_anim_clip = np.expand_dims(combine_anim_clip, axis=0)
    root_step = None
    try:
        root_step = data_pipe.named_steps.get("root")
    except Exception:
        pass

    if root_step is not None:
        if not hasattr(root_step, "start_pos_override"):
            root_step.start_pos_override = None
        if not hasattr(root_step, "start_yaw_override"):
            root_step.start_yaw_override = None
        if not hasattr(root_step, "last_end_pos"):
            root_step.last_end_pos = None
        if not hasattr(root_step, "last_end_yaw"):
            root_step.last_end_yaw = None
        if _root_last_pos is not None:
            root_step.start_pos_override = _root_last_pos
        if _root_last_yaw is not None:
            root_step.start_yaw_override = _root_last_yaw

    inv_data = data_pipe.inverse_transform(combine_anim_clip)

    if root_step is not None:
        if getattr(root_step, "last_end_pos", None) is not None:
            _root_last_pos = root_step.last_end_pos
        if getattr(root_step, "last_end_yaw", None) is not None:
            _root_last_yaw = float(np.arctan2(np.sin(root_step.last_end_yaw), np.cos(root_step.last_end_yaw)))

    return inv_data[0].values, inv_data[0]

def visualize_and_write_no_gt(results, save_dir, audio_id, have_text=False):
    _load_motion_assets()
    if have_text == False or text_data_dir is None:
        body_motion = inv_standardize(np.expand_dims(results, axis=0), body_scaler)
        print(f'[Motion] Received motion of shape {body_motion.shape} at id {audio_id}')
        return to_bvh_diffusion(body_motion, os.path.join(save_dir, f"motion_{audio_id}.bvh"), data_pipe, 30)
    else:
        body_motion = inv_standardize(np.expand_dims(results, axis=0), text_body_scaler)
        print(f'[Motion] Received motion of shape {body_motion.shape} at id {audio_id}')
        return to_bvh_diffusion(body_motion, os.path.join(save_dir, f"motion_{audio_id}.bvh"), text_data_pipe, 30)

def save_to_csv(motion_dataframe, save_dir, audio_id):

    correspondance = [
        ("waist_yaw_Zrotation", "waist_yaw_joint"),
        ("waist_roll_Xrotation", "waist_roll_joint"),
        ("waist_pitch_Yrotation", "waist_pitch_joint"),
        ("left_shoulder_pitch_Yrotation", "left_shoulder_pitch_joint"),
        ("left_shoulder_roll_Xrotation", "left_shoulder_roll_joint"),
        ("left_shoulder_yaw_Zrotation", "left_shoulder_yaw_joint"),
        ("left_elbow_Yrotation", "left_elbow_joint"),
        ("left_wrist_roll_Xrotation", "left_wrist_roll_joint"),
        ("left_wrist_pitch_Yrotation", "left_wrist_pitch_joint"),
        ("left_wrist_yaw_Zrotation", "left_wrist_yaw_joint"),
        ("left_hand_thumb_0_Yrotation", "left_hand_thumb_0_joint"),
        ("left_hand_thumb_1_Zrotation", "left_hand_thumb_1_joint"),
        ("left_hand_thumb_2_Zrotation", "left_hand_thumb_2_joint"),
        ("left_hand_middle_0_Zrotation", "left_hand_middle_0_joint"),
        ("left_hand_middle_1_Zrotation", "left_hand_middle_1_joint"),
        ("left_hand_index_0_Zrotation", "left_hand_index_0_joint"),
        ("left_hand_index_1_Zrotation", "left_hand_index_1_joint"),
        ("right_shoulder_pitch_Yrotation", "right_shoulder_pitch_joint"),
        ("right_shoulder_roll_Xrotation", "right_shoulder_roll_joint"),
        ("right_shoulder_yaw_Zrotation", "right_shoulder_yaw_joint"),
        ("right_elbow_Yrotation", "right_elbow_joint"),
        ("right_wrist_roll_Xrotation", "right_wrist_roll_joint"),
        ("right_wrist_pitch_Yrotation", "right_wrist_pitch_joint"),
        ("right_wrist_yaw_Zrotation", "right_wrist_yaw_joint"),
        ("right_hand_thumb_0_Yrotation", "right_hand_thumb_0_joint"),
        ("right_hand_thumb_1_Zrotation", "right_hand_thumb_1_joint"),
        ("right_hand_thumb_2_Zrotation", "right_hand_thumb_2_joint"),
        ("right_hand_middle_0_Zrotation", "right_hand_middle_0_joint"),
        ("right_hand_middle_1_Zrotation", "right_hand_middle_1_joint"),
        ("right_hand_index_0_Zrotation", "right_hand_index_0_joint"),
        ("right_hand_index_1_Zrotation", "right_hand_index_1_joint"),
    ]

    column_mapping = dict(correspondance)

    motion_dataframe_renamed = motion_dataframe.rename(columns=column_mapping)

    new_column_names = [new_name for old_name, new_name in correspondance]
    columns_to_keep = [col for col in new_column_names if col in motion_dataframe_renamed.columns]
    motion_dataframe_filtered = motion_dataframe_renamed[columns_to_keep]
    motion_dataframe_filtered = motion_dataframe_filtered / 180 * np.pi

    temp_file_path = os.path.join(save_dir, f"processing_{audio_id}.wait")
    final_file_path = os.path.join(save_dir, f"motion_{audio_id}.csv")

    with open(temp_file_path, 'w') as f:
        f.write('')

    motion_dataframe_filtered.to_csv(final_file_path, index=False)

    if os.path.exists(temp_file_path):
        os.remove(temp_file_path)

    return motion_dataframe_filtered

async def _motion_ws_client(ws_url: str, uid: str) -> None:
    if websockets is None:
        return
    save_dir = local_motion_dir
    _ensure_dir(save_dir)
    backoff = 1.0
    while not motion_stop_event.is_set():
        try:
            async with websockets.connect(ws_url, max_size=None, proxy=None) as ws:
                logger.info("[Motion] Connected to %s", ws_url)
                backoff = 1.0
                while not motion_stop_event.is_set():
                    try:
                        msg = await asyncio.wait_for(ws.recv(), timeout=1.0)
                    except asyncio.TimeoutError:
                        continue
                    except Exception:
                        if motion_stop_event.is_set():
                            break
                        raise

                    try:
                        payload = json.loads(msg)
                    except Exception:
                        logger.warning("[Motion] Received invalid JSON packet")
                        continue

                    if not isinstance(payload, dict) or payload.get("type") != "motion":
                        continue
                    data = payload.get("data", {})
                    if not data or data.get("id") != uid:
                        continue
                    motion_b64 = data.get("motion_b64")
                    audio_id = data.get("audio_id")
                    have_text = data.get("have_text")
                    speaker = data.get("speaker", "user")
                    time_logging('MOTION_RECV', str(audio_id) if audio_id is not None else '0')
                    ts = float(data.get("ts", time.time()))
                    if not motion_b64:
                        continue


                    if motion_sync_controller and not motion_sync_controller.should_accept_motion(audio_id):
                        logger.info("[Motion] Dropping stale motion audio_id=%s due to interruption", audio_id)
                        continue

                    motion_np = _decode_motion_b64(motion_b64)
                    filename = _motion_filename(audio_id, ts)
                    np.save(os.path.join(save_dir, filename), motion_np)
                    logger.info("[Motion] Saved %s (shape=%s)", filename, motion_np.shape)
                    try:
                        motion_df, motion_to_bvh = visualize_and_write_no_gt(motion_np, save_dir, audio_id, have_text)

                        try:
                            raw_motion_path = os.path.join(save_dir, f"raw_motion_{audio_id}.csv")
                            if isinstance(motion_df, pd.DataFrame):
                                motion_df.to_csv(raw_motion_path, index=False)
                                motion_df_raw = motion_df
                            else:
                                motion_df_raw = pd.DataFrame(motion_df)
                                motion_df_raw.to_csv(raw_motion_path, index=False)
                        except Exception as exc:
                            logger.warning("[Motion] Failed to save raw CSV for blender: %s", exc)
                            motion_df_raw = motion_df if isinstance(motion_df, pd.DataFrame) else None
                        if main_mujoco:
                            save_to_csv(motion_df, save_dir, audio_id)
                        else:
                            pass
                        if blender_client and blender_client.enabled:
                            try:


                                if speaker == "assistant":
                                    blender_client.stream_motion(
                                        motion_df_raw.iloc[1:] if motion_df_raw is not None else motion_df.iloc[1:],
                                        audio_id,
                                        audio_dir=local_assistant_audio_dir,
                                    )
                                else:

                                    blender_client.stream_motion(
                                        motion_df_raw.iloc[1:] if motion_df_raw is not None else motion_df.iloc[1:],
                                        audio_id=None,
                                        audio_dir=None,
                                    )
                            except Exception as exc:
                                logger.warning("[Blender] Failed to dispatch motion to Blender: %s", exc)
                        logger.info("[Motion] Saved motion for audio_id %s (speaker=%s)", audio_id, speaker)
                        time_logging('MOTION_SAVE', str(audio_id) if audio_id is not None else '0')


                        if interruption_manager and interruption_manager.is_waiting_for_new_motion():
                            interruption_manager.notify_new_motion_received()

                    except Exception as exc:
                        logger.error("[Motion] Failed to decode/save motion: %s", exc)
        except Exception as exc:
            if motion_stop_event.is_set():
                break
            logger.warning("[Motion] WebSocket error: %s (reconnecting in %.1fs)", exc, backoff)
            await asyncio.sleep(backoff)
            backoff = min(backoff * 2, 10.0)

    logger.info("[Motion] Listener stopped")


def motion_listener_loop():
    if websockets is None:
        logger.warning("[Motion] Package 'websockets' not installed; skipping motion listener")
        return
    ws_url = _build_ws_url(SERVER, "/ws/motion", UID)
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        loop.run_until_complete(_motion_ws_client(ws_url, UID))
    finally:
        loop.close()


# === Image capture ===
image_stop_event = threading.Event()
motion_stop_event = threading.Event()


def image_capture_loop():
    cap = cv2.VideoCapture(0)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    for _ in range(3):
        cap.read()
        time.sleep(0.02)

    if not cap.isOpened():
        logger.warning("Camera not available; skipping image capture")
        return
    try:
        next_capture = time.monotonic()
        last_send_time = 0.0
        while not image_stop_event.is_set():
            ret, frame = cap.read()
            if not ret:
                time.sleep(0.1)
                continue

            success, encoded = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), IMAGE_JPEG_QUALITY])
            if success:
                image_bytes = encoded.tobytes()
                offset_ms = allocate_offset_ms()
                turn = get_current_turn()
                idx = get_current_segment_index()
                save_image_local(image_bytes, turn, idx, offset_ms)

                current_time = time.monotonic()
                if current_time - last_send_time >= IMG_SEND_INTERVAL:
                    with is_receiving_lock:
                        if not is_receiving_response:
                            gpt_session.enqueue_media("image/jpeg", image_bytes)
                            last_send_time = current_time

            next_capture += IMG_CAPTURE_INTERVAL
            delay = max(0.0, next_capture - time.monotonic())
            image_stop_event.wait(delay)
    finally:
        cap.release()

def robot_ext_image_capture_loop():
    robot_ssh_host = os.getenv("PROACT_ROBOT_SSH_HOST")
    if not robot_ssh_host:
        logger.info("Remote robot image capture disabled; set PROACT_ROBOT_SSH_HOST to enable it.")
        return
    robot_ssh_user = os.getenv("PROACT_ROBOT_SSH_USER", "unitree")
    try:
        last_send_time = 0.0
        next_capture = time.monotonic()
        while not image_stop_event.is_set():
            try:
                p = subprocess.Popen(
                    ["ssh", f"{robot_ssh_user}@{robot_ssh_host}",
                    "ffmpeg -f v4l2 -input_format mjpeg -video_size 1280x720 \
                    -i /dev/video0 \
                    -frames:v 1 \
                    -f image2pipe -vcodec mjpeg -"],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.DEVNULL
                )
                image_bytes = p.stdout.read()
                offset_ms = allocate_offset_ms()
                turn = get_current_turn()
                idx = get_current_segment_index()
                save_image_local(image_bytes, turn, idx, offset_ms)

                current_time = time.monotonic()
                if current_time - last_send_time >= IMG_SEND_INTERVAL:
                    with is_receiving_lock:
                        if not is_receiving_response:
                            gpt_session.enqueue_media("image/jpeg", image_bytes)
                            last_send_time = current_time

            except pyqueue.Empty:
                continue
            except Exception as e:
                logger.error(f"Error processing robot image: {e}")

            next_capture += IMG_CAPTURE_INTERVAL
            delay = max(0.0, next_capture - time.monotonic())
            image_stop_event.wait(delay)
    finally:
        pass


# === Manual Interrupt (Keyboard Triggered) ===
def request_manual_interrupt() -> bool:
    """
     Agent

    Handle a manual interrupt from the keyboard. This clears the pending response,
    cancels the Realtime response, and preserves the playback boundary.
    """
    global is_receiving_response, manual_interrupt_active, interruption_manager, blender_client, gpt_session

    logger.info("=== MANUAL INTERRUPT TRIGGERED (keyboard) ===")




    manual_interrupt_active = False
    _cancel_manual_interrupt_timer()


    if gpt_session:
        current_item = gpt_session.get_current_item_id()
        if current_item:
            gpt_session.mark_item_interrupted(current_item)
            logger.info("Manual interrupt: Marked item_id=%s as interrupted", current_item)


    current_audio_id = get_current_segment_index() or None


    interrupt_triggered = False
    if interruption_manager:
        interrupt_triggered = interruption_manager.trigger_interruption(
            interrupt_type=InterruptionType.MANUAL_INTERRUPT,
            source_turn=get_current_turn(),
            target_turn=get_current_turn(),
            audio_position_ms=0,
            current_audio_id=current_audio_id,
        )


    with is_receiving_lock:
        if is_receiving_response:
            logger.info("Manual interrupt: set is_receiving_response = False, user can speak now")
            is_receiving_response = False


    mark_turn_complete()
    logger.info("Manual interrupt: Turn incremented to %s", get_current_turn())


    agent_accumulator.clear()
    _clear_playback_queue()
    agent_playback_tracker.end_turn()



    try:
        gpt_output_queue.put_nowait({"event": "manual_interrupt_complete"})
        logger.debug("Manual interrupt: Sent manual_interrupt_complete to playback queue")
    except Exception as e:
        logger.error("Failed to send manual_interrupt_complete: %s", e)



    if blender_client and blender_client.enabled:
        try:
            if current_audio_id is not None:
                blender_client.set_pending_interrupt(current_audio_id)
                logger.info("Manual interrupt: Blender pending interrupt set for audio_id=%s", current_audio_id)
            else:
                blender_client.clear_queue()
                logger.info("Manual interrupt: Blender queue cleared")
        except Exception as e:
            logger.error("Failed to set Blender pending interrupt: %s", e)




    if gpt_session and not gpt_session._stop.is_set():
        try:
            gpt_session.enqueue_cancel_response()
            logger.info("Manual interrupt: Sent cancel_response to GPT")
        except Exception as e:
            logger.error("Manual interrupt: Failed to send cancel_response: %s", e)

    logger.info("Manual interrupt completed. User microphone is now open.")
    return True

# === New Hotkey Handlers ===



# === Hotkey listener ===
class InterruptHotkeyListener:
    """
    Listen for hotkeys using pynput.

    Hotkeys:
    - 'o': Force agent reply ()
    - 'i': Manual interrupt while the agent is speaking.
    """


    DEBOUNCE_INTERVAL = 1.0

    def __init__(self, hotkey_reply: str = "o", hotkey_interrupt: str = "i", allow_interrupt: bool = False) -> None:
        self.hotkey_reply = (hotkey_reply or "o").lower()
        self.hotkey_interrupt = (hotkey_interrupt or "i").lower()
        self.allow_interrupt = allow_interrupt
        self._listener = None
        self._enabled = False
        self._last_key_times: Dict[str, float] = {}
        self._init_key_map()

    def _init_key_map(self):
        """Build the supported manual control key map."""
        self.key_map = {
            self.hotkey_reply: self._handle_reply,
            self.hotkey_interrupt: self._handle_interrupt,
            'q': self._hotkey_q,
            'e': self._hotkey_e,
            'w': self._hotkey_w,
            'a': self._hotkey_a,
            's': self._hotkey_s,
            'd': self._hotkey_d,
        }

    def _hotkey_q(self):
        locomotion_queue.put(('turn_left', 30.0))
        logger.info("Queued locomotion instruction: %s with parameter: %s", 'turn_left', 30.0)

    def _hotkey_e(self):
        locomotion_queue.put(('turn_right', 30.0))
        logger.info("Queued locomotion instruction: %s with parameter: %s", 'turn_right', 30.0)

    def _hotkey_w(self):
        locomotion_queue.put(('move_forward', 0.5))
        logger.info("Queued locomotion instruction: %s with parameter: %s", 'move_forward', 0.5)

    def _hotkey_a(self):
        locomotion_queue.put(('move_left', 0.5))
        logger.info("Queued locomotion instruction: %s with parameter: %s", 'move_left', 0.5)
    def _hotkey_s(self):
        locomotion_queue.put(('move_backward', 0.5))
        logger.info("Queued locomotion instruction: %s with parameter: %s", 'move_backward', 0.5)

    def _hotkey_d(self):
        locomotion_queue.put(('move_right', 0.5))
        logger.info("Queued locomotion instruction: %s with parameter: %s", 'move_right', 0.5)
    def _handle_reply(self):
        logger.info("Hotkey '%s' detected; forcing agent reply.", self.hotkey_reply)
        request_agent_interrupt("hotkey")

    def _handle_interrupt(self):
        if self.allow_interrupt:
                logger.info("Hotkey '%s' detected; triggering manual interrupt.", self.hotkey_interrupt)
                request_manual_interrupt()
        else:
            logger.debug("Hotkey '%s' detected but interrupt disabled (--allow-interrupt not set).", self.hotkey_interrupt)

    def start(self) -> None:
        if pynput_keyboard is None:
            logger.warning("Hotkey listener requires 'pynput'; feature disabled.")
            return
        if self._listener:
            return
        self._enabled = True
        self._listener = pynput_keyboard.Listener(on_press=self._on_key_press)
        self._listener.start()
        if self.allow_interrupt:
            logger.info(
                "Hotkey listener ready. Press '%s' to force agent reply, '%s' to interrupt agent.",
                self.hotkey_reply,
                self.hotkey_interrupt,
            )
        else:
            logger.info(
                "Hotkey listener ready. Press '%s' to force agent reply.",
                self.hotkey_reply,
            )

    def stop(self) -> None:
        self._enabled = False
        if self._listener:
            try:
                self._listener.stop()
            except Exception:
                pass
            self._listener = None

    def _is_debounced(self, key_char: str) -> bool:
        """Return whether a key press should be debounced."""
        now = time.time()
        last_time = self._last_key_times.get(key_char, 0.0)
        if now - last_time < self.DEBOUNCE_INTERVAL:
            return True
        self._last_key_times[key_char] = now
        return False

    def _on_key_press(self, key) -> None:
        if not self._enabled:
            return
        try:
            key_char = key.char
        except AttributeError:
            return
        if not key_char:
            return

        key_lower = key_char.lower()


        if self._is_debounced(key_lower):
            return


        handler = self.key_map.get(key_lower)
        if handler:
            handler()


# === Main orchestration ===
def main(args: Optional[argparse.Namespace] = None):
    """"""
    args = args or parse_cli_args()
    need_slow_thinking = getattr(args, "need_slow_thinking", True)
    render_target = getattr(args, "render_target", "mujoco")
    enable_blender = BlenderStreamClient is not None and render_target == "blender"
    enable_mujoco = main_mujoco is not None and render_target == "mujoco"
    allow_interrupt = getattr(args, "allow_interrupt", False)
    external_audio_renderer_active = False

    global blender_client
    global local_audio_playback_enabled
    global session_start_time, segment_index, current_turn, turn_user_audio_pending, user_buffer, first_local_save_done, last_segment_speaker, last_segment_turn, last_user_voice_time
    global interruption_manager, motion_sync_controller
    global REALTIME_SETTINGS, REALTIME_MODEL_CONFIG


    REALTIME_SETTINGS = _get_realtime_settings(allow_interrupt=allow_interrupt)
    REALTIME_MODEL_CONFIG = {
        "initial_model_settings": REALTIME_SETTINGS,
    }

    with session_time_lock:
        session_start_time = time.time()
        first_local_save_done = False
    with segment_index_lock:
        segment_index = 0
    with segment_meta_lock:
        last_segment_speaker = None
        last_segment_turn = None
    with turn_lock:
        current_turn = 1
        turn_user_audio_pending = False
    with user_buffer_lock:
        user_buffer = np.empty((0,), dtype='float32')
    last_user_voice_time = 0.0

    conversation_tracker.reset()
    live_transcripts.reset()
    global last_user_turn_for_asr, last_assistant_turn_for_asr
    last_user_turn_for_asr = None
    last_assistant_turn_for_asr = None
    global _last_asr_key, _last_asr_text
    _last_asr_key = None
    _last_asr_text = ""
    image_sampler.reset()

    setup_logging()
    if render_target == "blender" and BlenderStreamClient is None:
        logger.warning("Blender streaming requested but blender_stream_client.py not available; skipping Blender output.")
    prepare_local_dirs()


    if allow_interrupt:
        logger.info("Interruption support ENABLED (--allow-interrupt)")

        interruption_manager = InterruptionManager(
            server_url=SERVER,
            uid=UID,
            logger=logger,
            motion_dir=local_motion_dir,
        )


        motion_sync_controller = MotionSyncController(
            interruption_manager=interruption_manager,
            playback_controller=None,
            logger=logger,
        )
    else:
        logger.info("Interruption support DISABLED (use --allow-interrupt to enable)")
        interruption_manager = None
        motion_sync_controller = None


    if allow_interrupt and interruption_manager:
        def on_interruption(state: InterruptionState):
            interrupt_type_name = state.type.value if state else "unknown"
            is_agent_self = state and state.type == InterruptionType.AGENT_SELF_INTERRUPT

            if is_agent_self:
                logger.info("Interruption callback [AGENT_SELF]: clearing queues, new agent response incoming")
            else:
                logger.info("Interruption callback [USER]: clearing queues, returning control to user")


            agent_accumulator.clear()
            _clear_playback_queue()
            agent_playback_tracker.end_turn()


            interrupt_audio_id = interruption_manager.get_interrupt_after_audio_id()



            if blender_client and blender_client.enabled:
                try:
                    if interrupt_audio_id is not None:
                        blender_client.set_pending_interrupt(interrupt_audio_id)
                        logger.info("Interruption callback: Blender pending interrupt set for audio_id=%s", interrupt_audio_id)
                    else:

                        blender_client.clear_queue()
                        logger.info("Interruption callback: Blender queue cleared (no audio_id)")
                except Exception as e:
                    logger.error("Failed to set Blender pending interrupt: %s", e)



            if not is_agent_self:
                try:
                    gpt_session.enqueue_cancel_response()
                    logger.info("Interruption callback: Sent cancel_response to GPT")
                except Exception as e:
                    logger.error("Failed to send cancel_response to GPT: %s", e)

        interruption_manager.register_on_interrupt(on_interruption)

    if args.asr:
        start_asr_transcriber(local_session_dir)
    else:
        logger.info("ASR transcriber disabled (use --asr to enable).")

    blender_client = None
    if enable_blender:
        use_viz = not sys.platform.startswith("darwin")
        blender_client = BlenderStreamClient(
            host=BLENDER_STREAM_HOST,
            port=BLENDER_STREAM_PORT,
            enabled=True,
            frame_interval=BLENDER_FRAME_INTERVAL,
            audio_chunk_seconds=BLENDER_AUDIO_CHUNK_SECONDS,
            audio_dir=local_assistant_audio_dir,
            enable_visualizer=use_viz,
        )
        if blender_client.connect():
            external_audio_renderer_active = True
        else:
            # A configured but unreachable Blender must not suppress local
            # audio. Disable retries for this session so both sinks cannot
            # become active if Blender appears later.
            logger.warning("Blender renderer unavailable; falling back to local audio playback.")
            blender_client.enabled = False

    logger.info("[Main] Attempting to synchronize time with server...")
    if not synchronize_time_with_server():
        logger.warning("[Main] WARNING: Time synchronization failed. Timestamps may be inaccurate.")

    global is_mujoco_playing, last_robot_play_end_time, last_response_create_time
    is_mujoco_playing = multiprocessing.Value('b', False)
    last_robot_play_end_time = multiprocessing.Value('d', 0.0)
    last_response_create_time = multiprocessing.Value('d', 0.0)

    mujoco_process: Optional[multiprocessing.Process] = None
    if enable_mujoco:
        try:
            logger.info("Starting MuJoCo simulation process...")
            mujoco_process = multiprocessing.Process(
                target=main_mujoco,
                args=(local_assistant_audio_dir, local_motion_dir, zero_point_ns, local_session_dir, session_start_time, is_mujoco_playing, last_robot_play_end_time),
                # args=(local_assistant_audio_dir, local_motion_dir, zero_point_ns, local_session_dir, session_start_time, is_mujoco_playing),
                daemon=True
            )
            mujoco_process.start()
            if mujoco_process.is_alive():
                external_audio_renderer_active = True
                logger.info("MuJoCo simulation process started.")
            else:
                logger.warning("MuJoCo simulation process exited during startup; falling back to local audio playback.")
        except Exception as e:
            logger.error(f"Failed to launch MuJoCo process: {e}")

    global locomotion_queue
    locomotion_queue = multiprocessing.Queue()

    global image_queue
    image_queue = multiprocessing.Queue()

    global is_robot_playing
    is_robot_playing = multiprocessing.Value('b', False)

    real_process: Optional[multiprocessing.Process] = None
    if getattr(args, "real", False):
        if main_real is None:
             logger.error("Real robot control requested but main_real module not found.")
        else:
            try:
                logger.info("Starting Real deploy process...")
                real_process = multiprocessing.Process(
                    target=main_real,
                    args=(local_assistant_audio_dir, local_motion_dir, zero_point_ns, local_session_dir, session_start_time, locomotion_queue, image_queue, is_robot_playing, last_robot_play_end_time),
                    daemon=True
                )
                real_process.start()
                logger.info("Real deploy process started.")
                if real_process.is_alive():
                    external_audio_renderer_active = True
            except Exception as e:
                logger.error(f"Failed to launch Real deploy process: {e}")

    local_audio_playback_enabled = (
        not getattr(args, "real", False) and not external_audio_renderer_active
    )
    if local_audio_playback_enabled:
        logger.info("Local sounddevice playback enabled (no external audio renderer).")
    else:
        logger.info("Local sounddevice playback disabled (external renderer or --real mode).")

    global viz_queue
    # viz_queue = multiprocessing.Queue()

    viz_process = multiprocessing.Process(
        target=run_visualizer_process,
        args=(viz_queue,),
        daemon=True,
        name="StateVisualizer"
    )
    # viz_process.start()

    global slow_thinking_system
    if not need_slow_thinking:
        logger.info("Slow thinking disabled via --need_slow_thinking=false; skipping initialization.")
        slow_thinking_system = None
    else:
        slow_api_key = os.getenv("SLOW_THINKING_API_KEY") or os.getenv("OPENAI_API_KEY")
        slow_config = SlowThinkingConfig(
            log_dir=os.path.join(local_session_dir or LOCAL_MEDIA_BASE, "slow_thinking"),
            api_key=slow_api_key,
            base_url=os.getenv("SLOW_THINKING_BASE_URL") or os.getenv("OPENAI_BASE_URL"),
            model=os.getenv("PROACT_COGNITIVE_MODEL", "gpt-4.1-mini-2025-04-14"),
        )
        if not slow_config.api_key:
            logger.warning("SLOW_THINKING_API_KEY or OPENAI_API_KEY is missing; slow-thinking loop disabled.")
            slow_thinking_system = None
        else:
            slow_thinking_system = SlowThinkingSystem(
                context_provider=slow_thinking_context_provider,
                result_callback=slow_thinking_result_handler,
                # interval_seconds=13.0,
                interval_seconds=0.0,
                # interval_seconds=8.0, # for poster
                run_continuously=True,
                config=slow_config,
                logger=logger
            )
            slow_thinking_system.start()

    gpt_session.start()
    if not gpt_session.wait_until_ready():
        logger.warning("Realtime session did not become ready within timeout")

    hotkey_listener = InterruptHotkeyListener(allow_interrupt=allow_interrupt)
    hotkey_listener.start()

    motion_thread: Optional[threading.Thread] = None
    if websockets is not None:
        motion_stop_event.clear()
        motion_thread = threading.Thread(target=motion_listener_loop, name="MotionListener", daemon=True)
        motion_thread.start()
    else:
        logger.warning("[Motion] websockets not available; motion updates disabled")

    uploader_thread = threading.Thread(target=audio_upload_loop, name="UserUpload", daemon=True)
    uploader_thread.start()

    playback_thread = threading.Thread(target=playback_loop, name="Playback", daemon=True)
    playback_thread.start()

    image_thread = threading.Thread(target=image_capture_loop, name="ImageCapture", daemon=True)
    robot_image_thread = threading.Thread(target=robot_ext_image_capture_loop, name="RobotImageCapture", daemon=True)
    if getattr(args, "real", False):
        robot_image_thread.start()
    else:
        image_thread.start()

    try:

                # if sd.default.device[0] == 7:

        # print(sd.query_devices())
        # with sd.InputStream(samplerate=INPUT_RAW_SAMPLE_RATE,
        with PulseAudioInputStream(samplerate=INPUT_RAW_SAMPLE_RATE,
                            blocksize=int(INPUT_RAW_SAMPLE_RATE * 0.1),
                            channels=INPUT_CHANNELS,
                            dtype='int16',
                            callback=audio_callback):
            logger.info("Listening to microphone and streaming to OpenAI Realtime...")
            if allow_interrupt:
                logger.info("=== INTERRUPTION SUPPORT ENABLED ===")
                logger.info("interrupt_response: True - User can interrupt agent speech")
            else:
                logger.info("=== INTERRUPTION SUPPORT DISABLED ===")
                logger.info("interrupt_response: False - running without interruption support")
            while True:
                time.sleep(1)
    except KeyboardInterrupt:
        logger.info("User interrupted; shutting down")
    finally:
        image_stop_event.set()
        motion_stop_event.set()
        gpt_session.stop()
        gpt_input_queue.put(None)
        input_audio_queue.put(None)
        gpt_output_queue.put(None)
        playback_thread.join(timeout=2)
        if hotkey_listener:
            hotkey_listener.stop()
        if motion_thread:
            motion_thread.join(timeout=2)
        if slow_thinking_system:
            slow_thinking_system.stop(join_timeout=2)
        stop_asr_transcriber()
        if blender_client:
            blender_client.close()
        if mujoco_process and mujoco_process.is_alive():
            logger.info("Terminating MuJoCo process...")
            mujoco_process.terminate()
            mujoco_process.join(timeout=2.0)
            logger.info("MuJoCo process terminated.")
        if real_process and real_process.is_alive():
            logger.info("Terminating Real deploy process...")
            real_process.terminate()
            real_process.join(timeout=2.0)
            logger.info("Real deploy process terminated.")


if __name__ == "__main__":
    main()
