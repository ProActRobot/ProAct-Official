# CUDA_VISIBLE_DEVICES=0 torchrun --standalone --nproc_per_node=1 src/server/motion_server.py --port 32558 --enable_motion --output_save_every_second 2

"""Streaming motion server with interruption and text conditioning.

The service accepts interruption events, clears pending motion windows, broadcasts
the event to subscribed clients, and resumes generation from the next audio window.
"""

import base64
import json
import asyncio
import numpy as np
import os
import sys
import io
import threading
import time
import logging
import argparse
import queue
import copy
from typing import Any, Dict, Optional, Set, Tuple
from collections import defaultdict
from dataclasses import dataclass
from enum import Enum
from concurrent.futures import ThreadPoolExecutor, Future
from fractions import Fraction

import librosa
import soundfile
import torch
from easydict import EasyDict
from transformers import CLIPTokenizer, CLIPTextModel
from fastapi import FastAPI, Header, Query, Request, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse
import uvicorn
from scipy.signal import resample_poly
import joblib as jl
import math

cur_path = os.path.dirname(os.path.realpath(__file__))
motion_code_dir = os.path.abspath(os.path.join(cur_path, "..", "motion"))
sys.path.insert(0, motion_code_dir)
# Historical joblib exports stored transformer classes as top-level pymo.*.
sys.path.append(os.path.join(motion_code_dir, "data_process"))
from data_process.utils.utils import inv_standardize, standardize
from utils.streaming_audio_feature import StreamingAudioFeatureExtractor

zero_point_ns = 0

CLIP_MODEL_DIR = os.getenv("MOTION_CLIP_MODEL", "openai/clip-vit-base-patch32")
SLOW_THINKING_KEYS = ["intent", "pose", "motion"]
TEXT_MOGEN_PERIOD = 2

slow_thinking_payload_lock = threading.Lock()
slow_thinking_latest: Optional[Dict[str, Any]] = None
slow_thinking_clip_features: Optional[np.ndarray] = None
slow_thinking_features_used: bool = False

_clip_model_lock = threading.Lock()
_clip_tokenizer: Optional[CLIPTokenizer] = None
_clip_text_encoder: Optional[CLIPTextModel] = None



class InterruptionType(Enum):
    NONE = "none"
    USER_INTERRUPT = "user_interrupt"
    AGENT_SELF_INTERRUPT = "agent_self"
    MANUAL_INTERRUPT = "manual"


@dataclass
class InterruptionEvent:
    """A client interruption event with a monotonic sequence number."""
    type: InterruptionType
    source_turn: int
    target_turn: int
    audio_position_ms: int
    timestamp_ms: int
    sequence: int
    current_audio_id: Optional[int] = None


class InterruptionHandler:
    """Validate interruption events and notify the motion stream."""

    def __init__(self, logger: logging.Logger):
        self.logger = logger
        self._lock = threading.RLock()
        self._current_sequence = 0
        self._last_interruption: Optional[InterruptionEvent] = None
        self._interrupted_audio_ids: Set[int] = set()


        self._on_interrupt_callbacks: list = []

    def register_callback(self, callback: callable) -> None:
        """Register a callback invoked for each accepted interruption."""
        with self._lock:
            self._on_interrupt_callbacks.append(callback)

    def reset(self) -> None:
        """Clear interruption state when a new stream session starts."""
        with self._lock:
            self._current_sequence = 0
            self._last_interruption = None
            self._interrupted_audio_ids.clear()

    def handle_interruption(self, interruption_data: Dict[str, Any]) -> bool:
        """Parse and dispatch an interruption payload."""
        try:
            interrupt_type_str = interruption_data.get("type", "none")
            try:
                interrupt_type = InterruptionType(interrupt_type_str)
            except ValueError:
                interrupt_type = InterruptionType.NONE

            event = InterruptionEvent(
                type=interrupt_type,
                source_turn=interruption_data.get("source_turn", 0),
                target_turn=interruption_data.get("target_turn", 0),
                audio_position_ms=interruption_data.get("audio_position_ms", 0),
                timestamp_ms=interruption_data.get("timestamp_ms", int(time.time() * 1000)),
                sequence=interruption_data.get("sequence", 0),
                current_audio_id=interruption_data.get("current_audio_id"),
            )

            with self._lock:

                if event.sequence <= self._current_sequence:
                    self.logger.warning(
                        "Ignoring stale interruption signal: seq=%s (current=%s)",
                        event.sequence,
                        self._current_sequence,
                    )
                    return False

                self._current_sequence = event.sequence
                self._last_interruption = event

            self.logger.info(
                "=== INTERRUPTION RECEIVED === type=%s, source_turn=%s, target_turn=%s, audio_pos=%sms, current_audio_id=%s",
                event.type.value,
                event.source_turn,
                event.target_turn,
                event.audio_position_ms,
                event.current_audio_id,
            )


            for callback in self._on_interrupt_callbacks:
                try:
                    callback(event)
                except Exception as e:
                    self.logger.error("Interruption callback error: %s", e)

            return True

        except Exception as e:
            self.logger.error("Failed to handle interruption: %s", e)
            return False

    def mark_audio_interrupted(self, audio_id: int) -> None:
        """Mark an audio window so queued motion for it is discarded."""
        with self._lock:
            self._interrupted_audio_ids.add(audio_id)

    def is_audio_interrupted(self, audio_id: int) -> bool:
        """Return whether an audio window was invalidated by an interruption."""
        with self._lock:
            return audio_id in self._interrupted_audio_ids

    def clear_interrupted_audio(self, before_id: int) -> None:
        """Drop invalidation markers older than ``before_id``."""
        with self._lock:
            self._interrupted_audio_ids = {
                aid for aid in self._interrupted_audio_ids if aid >= before_id
            }

    def get_current_sequence(self) -> int:
        """Return the latest accepted interruption sequence."""
        with self._lock:
            return self._current_sequence


def time_logging(event: str, audio_id: str, session_dir: Optional[str]):
    """Logs a timing event to a CSV file in the session directory."""
    try:
        if zero_point_ns == 0 or not session_dir:
            return

        log_path = os.path.join(session_dir, "timing_log.csv")
        relative_time_ms = (time.time_ns() - zero_point_ns) // 1_000_000

        # Write header if file doesn't exist
        if not os.path.exists(log_path):
            with open(log_path, "w") as log_file:
                log_file.write("timestamp_ms,event,audio_id\n")

        with open(log_path, "a") as log_file:
            log_file.write(f"{relative_time_ms},{event},{audio_id}\n")
            log_file.flush()
    except Exception as e:
        logger.error(f"Failed to log time event: {e}")


def setup_logger(session_dir: Optional[str]) -> logging.Logger:
    logger = logging.getLogger("api_logger_interruption_text")
    logger.setLevel(logging.DEBUG)

    formatter = logging.Formatter(
        '%(asctime)s.%(msecs)03d-%(levelname)s-[%(filename)s:%(lineno)d] - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    stdout_handler = logging.StreamHandler(sys.stdout)
    stdout_handler.setLevel(logging.INFO)
    stdout_handler.setFormatter(formatter)
    stdout_handler.addFilter(lambda record: record.levelno <= logging.INFO)

    stderr_handler = logging.StreamHandler(sys.stderr)
    stderr_handler.setLevel(logging.WARNING)
    stderr_handler.setFormatter(formatter)

    log_save_path = os.path.join(session_dir, "server.log")
    os.makedirs(os.path.dirname(log_save_path), exist_ok=True)
    file_handler = logging.FileHandler(log_save_path)
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)

    # Avoid duplicate handlers if setup_logger is called multiple times
    if not logger.handlers:
        logger.addHandler(stdout_handler)
        logger.addHandler(stderr_handler)
        logger.addHandler(file_handler)

    return logger


parser = argparse.ArgumentParser()
parser.add_argument('--host', default='0.0.0.0')
parser.add_argument('--port', type=int, default=32558)
parser.add_argument('--output_save_every_second', type=float, default=4.0, help="Window length (seconds) used when saving intermediate audio")
parser.add_argument('--enable_motion', action='store_true', help="Enable motion generation")
parser.add_argument('--turn_complete_padding', type=float, default=2.0, help="Padding seconds added after each turn completion")
parser.add_argument('--mogen_padding', type=float, default=0.0, help="Padding seconds added after each motion generation window")
parser.add_argument('--mode', type=str, default='controlnet', choices=['controlnet', 'mix'], help="Choose motion generation mode")
parser.add_argument('--config', type=str, default=None, help="Motion model configuration path")
args = parser.parse_args()

timestamp = time.strftime("%Y%m%d_%H%M%S")
base_dir = os.path.join(cur_path, "..", "..", "runtime", "server", str(args.port), f"{timestamp}")

app = FastAPI()
logger = setup_logger(base_dir)

if args.mode == 'controlnet':
    from generate_multimodal_gestures import server_generate_diffusion_with_text_instruction, server_generate_diffusion_with_recurrent_text_instruction, smooth_transition_tensor, _compute_channel_feats, _standardize_2d_lastdim
    from Gesture_trainer import to_bvh, Controlnet_trainer
    trainer: Optional[Controlnet_trainer] = None
    DEFAULT_CFG = os.path.abspath(os.path.join(
        cur_path, "..", "..", "configs", "inference", "controlnet.yaml"
    ))
elif args.mode == 'mix':
    from generate_triplet_gestures import smooth_transition_tensor, _compute_channel_feats, _standardize_2d_lastdim
    from generate_triplet_gestures import server_generate_diffusion_mix_generation as server_generate_diffusion_with_text_instruction
    from Gesture_trainer import to_bvh
    from Gesture_trainer import MixTask_trainer as Controlnet_trainer
    trainer: Optional[Controlnet_trainer] = None
    DEFAULT_CFG = os.path.abspath(os.path.join(
        cur_path, "..", "..", "configs", "legacy", "triplet.yaml"
    ))
else:
    raise ValueError(f"Unsupported mode: {args.mode}")


def _warm_motion_pipeline() -> None:
    """Run a short inference to initialize model kernels and caches."""
    if not args.enable_motion or trainer is None:
        return
    try:
        warm_root = os.path.join(base_dir, "warmup")
        os.makedirs(warm_root, exist_ok=True)
        left_path = os.path.join(warm_root, "warm_left.wav")
        right_path = os.path.join(warm_root, "warm_right.wav")

        sr = 15360
        silence = np.zeros(int(sr * 0.3), dtype=np.float32)
        soundfile.write(left_path, silence, sr)
        soundfile.write(right_path, silence, sr)

        feats_left = _compute_channel_feats(left_path, fps=trainer.config.data.framerate)
        feats_right = _compute_channel_feats(right_path, fps=trainer.config.data.framerate)

        if args.mode == 'controlnet':
            scaler_path = os.path.join(trainer.config.audio.audio_data_root, "input_scaler.sav")
        elif args.mode == 'mix':
            scaler_path = os.path.join(trainer.config.data.data_root, "input_scaler_audio.sav")
        else:
            raise ValueError(f"Unsupported mode: {args.mode}")
        input_scaler = jl.load(scaler_path)

        T_min = min(feats_left.shape[0], feats_right.shape[0])
        feats_left = feats_left[:T_min]
        feats_right = feats_right[:T_min]
        if args.mode == 'controlnet':
            feats_left = _standardize_2d_lastdim(feats_left, input_scaler)
            feats_right = _standardize_2d_lastdim(feats_right, input_scaler)
            audio_features = np.concatenate([feats_left, feats_right], axis=1)[None, :, :].astype(np.float32)
        elif args.mode == 'mix':
            feats_all = np.concatenate([feats_left, feats_right], axis=1)
            audio_features = _standardize_2d_lastdim(feats_all, input_scaler)[None, :, :].astype(np.float32)
        else:
            raise ValueError(f"Unsupported mode: {args.mode}")

        generation_fn = (
            server_generate_diffusion_with_recurrent_text_instruction
            if args.mode == 'controlnet'
            else server_generate_diffusion_with_text_instruction
        )
        generation_fn(
            audio_feats=audio_features,
            trainer=trainer,
            solver=trainer.config.inference.inference_sovler,
            cached_body_latent=None,
            cached_audio_feats=None,
            sample_steps=trainer.config.inference.sample_steps,
            guidance_weight=trainer.config.inference.guidance_weight,
            eta=trainer.config.inference.eta,
            have_text=False,
            text_features=None,
            mogen_padding=args.mogen_padding,
        )
        logger.info("Motion pipeline warmup finished")
    except Exception as exc:
        logger.warning("Motion pipeline warmup skipped: %s", exc)


if args.enable_motion:
    cfg_path = os.path.abspath(args.config or os.getenv("MOTION_CONFIG", DEFAULT_CFG))
    if not os.path.exists(cfg_path):
        logger.error("Config not found: %s", cfg_path)
        raise FileNotFoundError(f"Config file not found: {cfg_path}")
    else:
        trainer = Controlnet_trainer(cfg_path)
        _warm_motion_pipeline()


def _encode_motion_np_b64(motion: np.ndarray) -> str:
    """Encode motion numpy array as base64 string for transport."""
    bio = io.BytesIO()
    np.save(bio, motion)
    return base64.b64encode(bio.getvalue()).decode("utf-8")


def _ensure_clip_models(clip_dir: str) -> None:
    """Lazily load the CLIP tokenizer and text encoder from a model ID or path."""
    global _clip_tokenizer, _clip_text_encoder
    if _clip_tokenizer is not None and _clip_text_encoder is not None:
        return
    with _clip_model_lock:
        if _clip_tokenizer is not None and _clip_text_encoder is not None:
            return
        logger.info("Loading CLIP tokenizer and text encoder from %s", clip_dir)
        _clip_tokenizer = CLIPTokenizer.from_pretrained(clip_dir)
        _clip_text_encoder = CLIPTextModel.from_pretrained(clip_dir)
        for param in _clip_text_encoder.parameters():
            param.requires_grad = False
        _clip_text_encoder.eval()


def _safe_to_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, (list, tuple, set)):
        return ", ".join(str(v) for v in value if v)
    return str(value)


def _build_clip_text_payload(
    payload: Dict[str, Any],
) -> Tuple[Optional[Dict[str, str]], Optional[Any]]:
    """Build the runtime text condition after applying the motivation policy."""
    texts = {key: "" for key in SLOW_THINKING_KEYS}

    motivation_score = payload.get("motivation_score")
    if motivation_score is None:
        # Older clients sent intention_score only. Keep that field as a
        # compatibility fallback while using motivation_score as the policy
        # decision for current clients.
        motivation_score = payload.get("intention_score")
    if motivation_score is None:
        raise ValueError("Missing motivation_score in slow-thinking payload")

    clip_texts = payload.get("clip_texts")
    motion_length = payload.get("motion_length")
    if isinstance(clip_texts, dict):
        for key in SLOW_THINKING_KEYS:
            texts[key] = _safe_to_text(clip_texts.get(key, ""))
    else:
        raise ValueError("clip_texts must be a dictionary")

    if motivation_score < 4 or not texts["intent"]:
        return None, motion_length
    return texts, motion_length


@torch.no_grad()
def encode_texts_with_clip(clip_dir: str, texts: Dict[str, str]) -> np.ndarray:
    """Encode the intent, pose, and motion texts into [3, 77, H]."""
    _ensure_clip_models(clip_dir)
    if _clip_tokenizer is None or _clip_text_encoder is None:
        raise RuntimeError("CLIP models failed to load")

    feats = []
    device = next(_clip_text_encoder.parameters()).device
    for key in SLOW_THINKING_KEYS:
        text_value = texts.get(key, "") or ""
        logger.debug("Encoding slow-thinking text [%s]: %s", key, text_value)
        inputs = _clip_tokenizer(
            text_value,
            return_tensors="pt",
            padding='max_length',
            truncation=True,
            max_length=77,
        ).to(device)
        hidden = _clip_text_encoder(**inputs).last_hidden_state.squeeze(0)
        feats.append(hidden.cpu().numpy())

    return np.stack(feats, axis=0).astype(np.float32)


class SlowThinkingClipWorker:
    """Background worker that encodes new slow-thinking payloads with CLIP."""

    def __init__(self, clip_dir: str) -> None:
        self.clip_dir = clip_dir
        self._queue: "queue.Queue[Optional[Dict[str, Any]]]" = queue.Queue()
        self._thread: Optional[threading.Thread] = None
        self._running = threading.Event()

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._running.set()
        self._thread = threading.Thread(target=self._run, name="SlowThinkingClipWorker", daemon=True)
        self._thread.start()

    def stop(self, timeout: float = 2.0) -> None:
        if not self._running.is_set():
            return
        self._running.clear()
        try:
            self._queue.put_nowait(None)
        except Exception:
            pass
        if self._thread:
            self._thread.join(timeout=timeout)

    def submit(self, payload: Dict[str, Any]) -> None:
        if not payload:
            return
        self.start()
        self._queue.put(copy.deepcopy(payload))

    def _run(self) -> None:
        global slow_thinking_features_used
        while self._running.is_set():
            try:
                payload = self._queue.get(timeout=1.0)
            except queue.Empty:
                continue
            if payload is None:
                break
            try:
                clip_texts, motion_length = _build_clip_text_payload(payload)
                if clip_texts is None:
                    logger.info("Skipping CLIP encoding below the motivation threshold")
                    continue
                feats = encode_texts_with_clip(self.clip_dir, clip_texts)
                # print('[debug] featsshape:', feats.shape)
            except FileNotFoundError as exc:
                logger.warning("CLIP checkpoint missing (%s); stopping feature worker.", exc)
                self._running.clear()
                break
            except Exception as exc:
                logger.exception("Slow-thinking CLIP encoding failed: %s", exc)
                continue

            with slow_thinking_payload_lock:
                global slow_thinking_clip_features, slow_thinking_features_used, slow_thinking_features_length
                slow_thinking_clip_features = feats
                slow_thinking_features_length = motion_length
                slow_thinking_features_used = False


def _register_slow_thinking_payload(payload: Dict[str, Any]) -> None:
    """Cache the latest slow-thinking dict and trigger CLIP feature extraction."""
    global slow_thinking_features_used
    if not isinstance(payload, dict):
        logger.warning("Ignoring slow-thinking payload with invalid type: %s", type(payload))
        return
    payload_copy = copy.deepcopy(payload)
    with slow_thinking_payload_lock:
        global slow_thinking_latest, slow_thinking_features_used
        slow_thinking_latest = payload_copy
        slow_thinking_features_used = False
    slow_thinking_clip_worker.submit(payload_copy)


def get_latest_slow_thinking_payload() -> Optional[Dict[str, Any]]:
    with slow_thinking_payload_lock:
        return copy.deepcopy(slow_thinking_latest) if slow_thinking_latest is not None else None


def get_latest_slow_thinking_features(copy_array: bool = True) -> Optional[np.ndarray]:
    global slow_thinking_features_used, slow_thinking_clip_features, slow_thinking_features_length
    with slow_thinking_payload_lock:
        if slow_thinking_clip_features is None:
            return None, None
        if slow_thinking_features_used == True:
            return None, None
        slow_thinking_features_used = True
        latest_clip_features = np.array(slow_thinking_clip_features, copy=copy_array)
        slow_thinking_clip_features = None  # Clear after use
        latest_motion_length = slow_thinking_features_length
        if latest_motion_length is None:
            return None, None
        return latest_clip_features, latest_motion_length


slow_thinking_clip_worker = SlowThinkingClipWorker(CLIP_MODEL_DIR)
if args.enable_motion:
    slow_thinking_clip_worker.start()



interruption_handler = InterruptionHandler(logger)


class MotionHubWithInterruption:
    """Forward generated motion and interruption events to WebSocket clients."""

    def __init__(self, interruption_handler: InterruptionHandler) -> None:
        self.interruption_handler = interruption_handler
        self.loop: Optional[asyncio.AbstractEventLoop] = None
        self._queue: "queue.Queue[Dict[str, Any]]" = queue.Queue()
        self._conns: Dict[str, Set[WebSocket]] = {}
        self._lock = threading.Lock()
        self._running = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._running.set()
        self._thread = threading.Thread(target=self._run, name="MotionHubThread", daemon=True)
        self._thread.start()
        logger.info("MotionHubWithInterruption started")

    def stop(self) -> None:
        self._running.clear()
        try:
            self._queue.put_nowait(None)  # type: ignore[arg-type]
        except Exception:
            pass

    def publish(self, uid: str, motion_np: np.ndarray, audio_id: int, have_text: bool, speaker: str = "user") -> None:
        """Queue one generated motion segment for a connected client."""
        if not uid:
            return


        if self.interruption_handler.is_audio_interrupted(audio_id):
            logger.info("Dropping motion for interrupted audio_id=%s", audio_id)
            return

        payload = {
            "id": uid,
            "audio_id": audio_id,
            "speaker": speaker,
            "motion_b64": _encode_motion_np_b64(motion_np),
            "shape": motion_np.shape,
            "dtype": str(motion_np.dtype),
            "ts": time.time(),
            "have_text": have_text,
        }
        self._queue.put(payload)

    def broadcast_interruption(self, uid: str, event: InterruptionEvent) -> None:
        """Queue an interruption event for all subscribers of ``uid``."""
        if not uid:
            return
        payload = {
            "type": "interruption",
            "id": uid,
            "interruption": {
                "type": event.type.value,
                "source_turn": event.source_turn,
                "target_turn": event.target_turn,
                "audio_position_ms": event.audio_position_ms,
                "sequence": event.sequence,
            },
            "ts": time.time(),
        }
        self._queue.put(payload)
        logger.info("Broadcasted interruption signal to clients: uid=%s", uid)

    def add_connection(self, uid: str, ws: WebSocket) -> None:
        with self._lock:
            self._conns.setdefault(uid, set()).add(ws)
        logger.info("MotionHub: client subscribed for uid=%s. total=%d", uid, len(self._conns.get(uid, [])))

    def remove_connection(self, uid: str, ws: WebSocket) -> None:
        with self._lock:
            if uid in self._conns and ws in self._conns[uid]:
                self._conns[uid].remove(ws)
                if not self._conns[uid]:
                    del self._conns[uid]
        logger.info("MotionHub: client unsubscribed for uid=%s", uid)

    async def _send(self, ws: WebSocket, data: Dict[str, Any]) -> None:
        msg_type = data.get("type", "motion")
        await ws.send_json({"type": msg_type, "data": data})

    def _run(self) -> None:
        while self._running.is_set():
            item = self._queue.get()
            if item is None:
                break
            uid = item.get("id")
            audio_id = item.get("audio_id")
            item_type = item.get("type", "motion")
            if self.loop is None:
                logger.warning("[MotionHub] Dropping %s audio_id=%s: event loop is None", item_type, audio_id)
                continue
            with self._lock:
                conns = list(self._conns.get(uid, set()))
            if not conns:
                logger.warning("[MotionHub] Dropping %s audio_id=%s for uid=%s: no connected clients (conns=%d)",
                             item_type, audio_id, uid, len(self._conns))
                continue
            for ws in conns:
                try:
                    fut = asyncio.run_coroutine_threadsafe(self._send(ws, item), self.loop)
                    fut.result(timeout=2.0)
                    logger.info("[MotionHub] Sent %s audio_id=%s to client uid=%s", item_type, audio_id, uid)
                except Exception as e:
                    logger.error("[MotionHub] Failed to send %s audio_id=%s: %s", item_type, audio_id, e)
                    try:
                        self.remove_connection(uid, ws)
                    except Exception:
                        pass


motion_hub = MotionHubWithInterruption(interruption_handler)


class MotionStreamManagerWithInterruption:
    """Maintain per-session audio buffers and invoke the motion generator."""

    def __init__(
        self,
        enable_motion: bool,
        port: int,
        window_seconds: float,
        interruption_handler: InterruptionHandler,
    ) -> None:
        self.enable_motion = enable_motion
        self.port = port
        self.window_seconds = max(window_seconds, 1.0)
        self.session_timeout = self.window_seconds * 10.0
        self.last_handshake_time = None
        self.lock = threading.RLock()
        self.cache_lock = threading.Lock()
        self.motion_worker = ThreadPoolExecutor(max_workers=1)
        self.text_mogen_inuse_id = None
        self.text_mogen_dict = None
        self.text_motion_remain_length = None
        self.interruption_handler = interruption_handler


        self._current_task: Optional[Future] = None
        self._task_lock = threading.Lock()
        self._session_generation = 0


        interruption_handler.register_callback(self._on_interruption)

        self.reset_session()

    def _invalidate_motion_tasks(self) -> None:
        """Invalidate queued/running work so it cannot publish into a new state."""
        with self._task_lock:
            # Keep generation invalidation and all guarded task mutations under
            # the same lock.  A running task must not pass its generation check
            # and then have reset_session interleave before it updates caches.
            with self.cache_lock:
                self._session_generation += 1
            if self.motion_worker:
                self.motion_worker.shutdown(wait=False, cancel_futures=True)
                logger.info("Shutdown motion worker, cancelled all pending tasks")
                self.motion_worker = ThreadPoolExecutor(max_workers=1)
                logger.info("Created new motion worker")
            self._current_task = None
            self.text_mogen_inuse_id = None
            self.text_mogen_dict = None
            self.text_motion_remain_length = None

    def _on_interruption(self, event: InterruptionEvent) -> None:
        """Cancel pending work and reset buffers after an interruption."""
        logger.info("MotionStreamManager: handling interruption, current_audio_id=%s", event.current_audio_id)


        with self.lock:
            self._invalidate_motion_tasks()


            client_audio_id = event.current_audio_id
            server_audio_id = getattr(self, 'audio_id', 0)

            if client_audio_id is not None:

                for aid in range(0, client_audio_id + 1):
                    self.interruption_handler.mark_audio_interrupted(aid)
                logger.info("Marked audio_id 0-%s as interrupted (from client)", client_audio_id)
            elif server_audio_id > 0:

                for aid in range(0, server_audio_id + 1):
                    self.interruption_handler.mark_audio_interrupted(aid)
                logger.info("Marked audio_id 0-%s as interrupted (from server)", server_audio_id)


            self.window_buffers = {
                "user": {"data": np.empty((0,), dtype=np.float32), "sr": 16000},
                "assistant": {"data": np.empty((0,), dtype=np.float32), "sr": 24000},
            }
            self.window_elapsed_sec = 0.0


            self.text_mogen_inuse_id = None
            self.text_mogen_dict = None
            self.text_motion_remain_length = None


            if hasattr(self, 'streaming_extractor_user') and self.streaming_extractor_user:
                self.streaming_extractor_user = StreamingAudioFeatureExtractor(
                    sr=15360, fps=trainer.config.data.framerate if trainer else 30, buffer_seconds=3.0,
                    mel_ref_mode='running_max'
                )
            if hasattr(self, 'streaming_extractor_agent') and self.streaming_extractor_agent:
                self.streaming_extractor_agent = StreamingAudioFeatureExtractor(
                    sr=15360, fps=trainer.config.data.framerate if trainer else 30, buffer_seconds=3.0,
                    mel_ref_mode='running_max'
                )

            logger.info("Cleared motion stream buffers due to interruption")


        if hasattr(self, 'uid') and self.uid:
            motion_hub.broadcast_interruption(self.uid, event)

    def reset_session(self) -> None:
        self._invalidate_motion_tasks()
        self.interruption_handler.reset()

        self.uid: Optional[str] = None
        self.session_dir: Optional[str] = None
        self.user_audio_id = 0
        self.agent_audio_id = 0
        self.audio_id = 0
        self.start_audio_id = 1
        self.end_audio_id = 1
        self.cached_audio_feats: Optional[torch.Tensor] = None
        self.cached_motion: Optional[torch.Tensor] = None
        self.total_motion: Optional[torch.Tensor] = None
        self.last_request_time: Optional[float] = None
        self.last_stream_time: Optional[float] = None
        self.start_time: Optional[float] = None
        self.turn_chunks = {
            "user": defaultdict(list),
            "assistant": defaultdict(list),
        }
        self.completed_turns: Set[int] = set()
        self.segment_buffers = {
            "user": defaultdict(lambda: {"data": np.empty((0,), dtype=np.float32), "sr": None}),
            "assistant": defaultdict(lambda: {"data": np.empty((0,), dtype=np.float32), "sr": None}),
        }
        self.motion_segment_counter = 0

        self.window_buffers = {
            "user": {"data": np.empty((0,), dtype=np.float32), "sr": 16000},
            "assistant": {"data": np.empty((0,), dtype=np.float32), "sr": 24000},
        }
        self.window_elapsed_sec = 0.0
        self.last_speaker = None

        if trainer is not None:
            if args.mode == 'controlnet':
                self.text_input_scaler = jl.load(os.path.join(trainer.config.data.data_root, 'input_scaler.sav'))
                self.audio_input_scaler = jl.load(os.path.join(trainer.config.audio.audio_data_root, 'input_scaler.sav'))
            elif args.mode == 'mix':
                self.text_input_scaler = None
                self.audio_input_scaler = jl.load(os.path.join(trainer.config.data.data_root, 'input_scaler_audio.sav'))
            else:
                raise ValueError(f"Unsupported mode: {args.mode}")
            self.output_scaler = jl.load(os.path.join(trainer.config.data.data_root, 'output_scaler.sav'))
            self.data_pipe = jl.load(os.path.join(trainer.config.data.data_root, 'data_pipe_' + str(trainer.config.data.framerate) + 'fps.sav'))

            self.streaming_extractor_user = StreamingAudioFeatureExtractor(
                sr=15360, fps=trainer.config.data.framerate, buffer_seconds=3.0,
                mel_ref_mode='running_max'
            )
            self.streaming_extractor_agent = StreamingAudioFeatureExtractor(
                sr=15360, fps=trainer.config.data.framerate, buffer_seconds=3.0,
                mel_ref_mode='running_max'
            )
        else:
            self.streaming_extractor_user = None
            self.streaming_extractor_agent = None
        self.text_mogen_inuse_id = None
        self.text_mogen_dict = None
        self.text_motion_remain_length = None

    def update_last_request_time(self) -> None:
        self.last_request_time = time.time()

    def update_last_stream_time(self) -> None:
        self.last_stream_time = time.time()

    async def check_activity(self) -> None:
        while True:
            await asyncio.sleep(5)

    def process_payload(self, uid: str, data: Dict[str, Any]) -> str:
        if "messages" not in data:
            raise ValueError("Missing messages field")
        result = "noop"

        now = time.time()

        print("uid: ", uid)
        with self.lock:
            self._ensure_session(uid)
            self.last_handshake_time = now
            for message in data.get("messages", []):
                logger.debug("receive message from local")
                result = self._process_message(uid, message)
        return result

    def _ensure_session(self, uid: str) -> None:
        if self.uid == uid and self.session_dir:
            return

        audio_dir = os.path.join(base_dir, "audio_log")
        audio_chunk_dir = os.path.join(base_dir, "audio_cache")
        os.makedirs(audio_dir, exist_ok=True)
        os.makedirs(audio_chunk_dir, exist_ok=True)
        os.makedirs(os.path.join(base_dir, "feedback_log"), exist_ok=True)
        os.makedirs(os.path.join(base_dir, "all_result"), exist_ok=True)
        os.makedirs(os.path.join(base_dir, "output_bvh_log"), exist_ok=True)

        previous_uid = self.uid
        self.uid = uid
        self.session_dir = base_dir

        self._invalidate_motion_tasks()
        if previous_uid is not None and previous_uid != uid:
            self.interruption_handler.reset()

        self.user_audio_id = 0
        self.agent_audio_id = 0
        self.audio_id = 0
        self.start_audio_id = 1
        self.end_audio_id = 1
        self.cached_audio_feats = None
        self.cached_motion = None
        self.total_motion = None
        self.turn_chunks = {
            "user": defaultdict(list),
            "assistant": defaultdict(list),
        }
        self.completed_turns = set()
        self.segment_buffers = {
            "user": defaultdict(lambda: {"data": np.empty((0,), dtype=np.float32), "sr": None}),
            "assistant": defaultdict(lambda: {"data": np.empty((0,), dtype=np.float32), "sr": None}),
        }
        self.motion_segment_counter = 0
        self.start_time = None

        self.window_buffers = {
            "user": {"data": np.empty((0,), dtype=np.float32), "sr": 16000},
            "assistant": {"data": np.empty((0,), dtype=np.float32), "sr": 24000},
        }
        self.window_elapsed_sec = 0.0
        self.last_speaker = None

        if trainer is not None:
            if args.mode == 'controlnet':
                self.text_input_scaler = jl.load(os.path.join(trainer.config.data.data_root, 'input_scaler.sav'))
                self.audio_input_scaler = jl.load(os.path.join(trainer.config.audio.audio_data_root, 'input_scaler.sav'))
            elif args.mode == 'mix':
                self.text_input_scaler = None
                self.audio_input_scaler = jl.load(os.path.join(trainer.config.data.data_root, 'input_scaler_audio.sav'))
            else:
                raise ValueError(f"Unsupported mode: {args.mode}")
            self.output_scaler = jl.load(os.path.join(trainer.config.data.data_root, 'output_scaler.sav'))
            self.data_pipe = jl.load(os.path.join(trainer.config.data.data_root, 'data_pipe_' + str(trainer.config.data.framerate) + 'fps.sav'))
        self.text_mogen_inuse_id = None
        self.text_mogen_dict = None
        self.text_motion_remain_length = None

    def _process_message(self, uid: str, message: Dict[str, Any]) -> str:
        contents = message.get("content", [])
        finish_reason = "noop"
        for item in contents:
            ctype = item.get("type")
            print(f"Processing content type: {ctype}")

            if ctype == "input_audio":
                audio_info = item.get("input_audio", {})
                data_b64 = audio_info.get("data")
                if not data_b64:
                    continue
                try:
                    audio_bytes = base64.b64decode(data_b64)
                except Exception as exc:
                    logger.error("Failed to decode audio: %s", exc)
                    raise ValueError("Invalid base64 audio data")

                speaker = audio_info.get("speaker", "user").lower()
                print(f"Speaker: {speaker}")
                if self.last_speaker == "user" and speaker == "assistant":
                    speaker_change_to_agent = True
                else:
                    speaker_change_to_agent = False
                self.last_speaker = speaker

                fmt = audio_info.get("format", "wav").lower()
                index_val = audio_info.get("index")
                try:
                    index_int = int(index_val) if index_val is not None else None
                except (TypeError, ValueError):
                    index_int = None
                sample_rate = audio_info.get("sample_rate")
                offset_ms = audio_info.get("offset_ms")
                turn_val = audio_info.get("turn")
                try:
                    turn_int = int(turn_val) if turn_val is not None else None
                except (TypeError, ValueError):
                    turn_int = None

                audio_id = self._handle_audio_chunk(
                    uid,
                    audio_bytes,
                    speaker,
                    fmt,
                    index=index_int,
                    offset_ms=offset_ms,
                    sample_rate=sample_rate,
                    turn=turn_int,
                    speaker_change=speaker_change_to_agent
                )
                finish_reason = f"audio:{speaker}:{audio_id}"

            elif ctype == "slow_thinking":
                slow_payload = item.get("slow_thinking")
                if isinstance(slow_payload, dict):
                    logger.debug("Received slow-thinking payload with ts=%s", slow_payload.get("timestamp_ms"))
                    _register_slow_thinking_payload(slow_payload)
                    finish_reason = "slow_thinking"
                else:
                    logger.warning("slow_thinking content missing dict payload: %s", slow_payload)

            elif ctype == "turn_complete":
                turn_info = item.get("turn_complete", {})
                turn_val = turn_info.get("turn")
                try:
                    turn_idx = int(turn_val)
                    print(f"Finalizing turn: {turn_idx}")
                except (TypeError, ValueError):
                    raise ValueError("Invalid turn index in turn_complete message")
                self._finalize_turn(turn_idx)
                finish_reason = f"turn_complete:{turn_idx}"

            elif ctype == "interruption":

                interruption_info = item.get("interruption", {})
                success = self.interruption_handler.handle_interruption(interruption_info)
                finish_reason = f"interruption:{'ok' if success else 'ignored'}"

            else:
                continue
        return finish_reason

    def _handle_audio_chunk(
        self,
        uid: str,
        audio_bytes: bytes,
        speaker: str,
        fmt: str,
        index: Optional[int] = None,
        offset_ms: Optional[Any] = None,
        sample_rate: Optional[Any] = None,
        turn: Optional[int] = None,
        speaker_change: bool = False,
    ) -> int:
        if fmt != "wav":
            raise ValueError(f"Unsupported audio format: {fmt}")
        if not self.session_dir:
            raise RuntimeError("Session directory not initialised")

        time_logging(f"AUDIO_CHUNK_RECV_{speaker}", str(index), self.session_dir)

        if speaker in {"assistant", "agent"}:
            prefix = "output_audio"
            from_agent = True
            if index is None:
                self.agent_audio_id += 1
                index = self.agent_audio_id
            else:
                self.agent_audio_id = max(self.agent_audio_id, index)
        else:
            prefix = "input_audio"
            from_agent = False
            if index is None:
                self.user_audio_id += 1
                index = self.user_audio_id
            else:
                self.user_audio_id = max(self.user_audio_id, index)

        self.audio_id = index
        audio_path = os.path.join(self.session_dir, "audio_log", f"{prefix}_{self.audio_id}.wav")

        if self.start_time is None:
            self.start_time = time.time()
        print(f"Saving audio chunk: {audio_path}, Current time(ms): {int((time.time() - self.start_time) * 1000)}")

        with open(audio_path, "wb") as f:
            f.write(audio_bytes)

        turn_value = turn if isinstance(turn, int) and turn > 0 else 1
        logger.debug(
            "Saved %s audio chunk to %s (turn=%s, offset_ms=%s, sample_rate=%s)",
            speaker, audio_path, turn_value, offset_ms, sample_rate,
        )

        key = "assistant" if from_agent else "user"
        self.turn_chunks[key][turn_value].append(self.audio_id)

        self._accumulate_motion_chunk(
            uid=uid,
            speaker_key=key,
            turn=turn_value,
            audio_bytes=audio_bytes,
            sample_rate=sample_rate,
            from_agent=from_agent,
            change_id=self.audio_id if speaker_change else -1,
        )
        return self.audio_id

    def _accumulate_motion_chunk(
        self,
        uid: str,
        speaker_key: str,
        turn: int,
        audio_bytes: bytes,
        sample_rate: Optional[Any],
        from_agent: bool,
        change_id: int = -1,
    ) -> None:
        other_speaker_key = "assistant" if speaker_key == "user" else "user"

        try:
            audio_np, detected_sr = soundfile.read(io.BytesIO(audio_bytes), dtype='float32')
        except Exception as exc:
            logger.error("Failed to decode audio bytes for motion accumulation: %s", exc)
            return

        if audio_np.ndim > 1:
            audio_np = np.mean(audio_np, axis=1)

        sr = int(sample_rate) if sample_rate is not None else detected_sr
        if sr <= 0:
            sr = 24000 if from_agent else 16000

        active_buf = self.window_buffers[speaker_key]
        if active_buf["sr"] != sr:
            try:
                frac = Fraction(active_buf["sr"], sr).limit_denominator()
                audio_np = resample_poly(audio_np, frac.numerator, frac.denominator)
                sr = active_buf["sr"]
            except Exception as exc:
                logger.error("Resample error for active buffer: %s", exc)
                return

        active_buf["data"] = np.concatenate([active_buf["data"], audio_np])

        chunk_duration = len(audio_np) / sr

        other_buf = self.window_buffers[other_speaker_key]
        padding_samples = int(chunk_duration * other_buf["sr"])
        padding = np.zeros(padding_samples, dtype=np.float32)
        other_buf["data"] = np.concatenate([other_buf["data"], padding])

        self.window_elapsed_sec += chunk_duration

        self._last_trigger_speaker = speaker_key

        if self.window_elapsed_sec >= self.window_seconds:
            self._emit_windowed_motion_segments(uid, change_id, speaker_key)
            self.window_elapsed_sec = 0.0

    def _emit_windowed_motion_segments(self, uid: str, change_id: int = -1, trigger_speaker: str = "user") -> None:
        if not self.session_dir:
            return

        self.motion_segment_counter += 1

        user_buf = self.window_buffers["user"]
        agent_buf = self.window_buffers["assistant"]

        print(f"User buffer length before generation: {user_buf['data'].shape[0]}, Agent buffer length: {agent_buf['data'].shape[0]}")

        self.end_audio_id = self.audio_id
        if change_id == -1:
            user_audio_path = os.path.join(self.session_dir, "audio_cache", f"input_segment_{self.start_audio_id}_{self.end_audio_id}.wav")
            agent_audio_path = os.path.join(self.session_dir, "audio_cache", f"output_segment_{self.start_audio_id}_{self.end_audio_id}.wav")
        else:
            user_audio_path = os.path.join(self.session_dir, "audio_cache", f"input_segment_{self.start_audio_id}_{self.end_audio_id}_{change_id}.wav")
            agent_audio_path = os.path.join(self.session_dir, "audio_cache", f"output_segment_{self.start_audio_id}_{self.end_audio_id}_{change_id}.wav")

        this_start_audio_id = self.start_audio_id
        self.start_audio_id = self.audio_id + 1


        user_buf["data"] = np.concatenate([user_buf["data"], np.zeros(int(user_buf["sr"] * args.mogen_padding), dtype=np.float32)])
        agent_buf["data"] = np.concatenate([agent_buf["data"], np.zeros(int(agent_buf["sr"] * args.mogen_padding), dtype=np.float32)])

        try:
            soundfile.write(user_audio_path, user_buf["data"], user_buf["sr"])
            soundfile.write(agent_audio_path, agent_buf["data"], agent_buf["sr"])
            logger.info(f"Saved windowed audio segments for motion step {self.motion_segment_counter}")
        except Exception as exc:
            logger.error(f"Failed to write windowed audio segments: {exc}")
            return
        finally:
            print(f"User buffer length after generation: {user_buf['data'].shape[0]}, Agent buffer length: {agent_buf['data'].shape[0]}")
            user_buf["data"] = np.empty((0,), dtype=np.float32)
            agent_buf["data"] = np.empty((0,), dtype=np.float32)

        if self.enable_motion:
            with self._task_lock:
                session_generation = self._session_generation
                self._current_task = self.motion_worker.submit(
                    self._generate_and_publish_motion,
                    uid, this_start_audio_id, user_audio_path, agent_audio_path,
                    self.motion_segment_counter, trigger_speaker, session_generation,
                )

    def _generate_and_publish_motion(
        self,
        uid: str,
        start_id: int,
        user_audio_path: str,
        agent_audio_path: str,
        step_idx: int,
        trigger_speaker: str = "user",
        session_generation: Optional[int] = None,
    ) -> None:

        with self._task_lock:
            if session_generation is not None and session_generation != self._session_generation:
                logger.info("Skipping stale motion task from session generation %s", session_generation)
                return
            # Interruptions replace the streaming extractors.  Keep this task
            # on the extractor instances belonging to its own generation.
            streaming_extractor_agent = self.streaming_extractor_agent
            streaming_extractor_user = self.streaming_extractor_user

        if self.interruption_handler.is_audio_interrupted(start_id):
            logger.info("Skipping motion generation for interrupted audio_id=%s", start_id)
            return

        with self._task_lock:
            if session_generation is not None and session_generation != self._session_generation:
                logger.info("Skipping stale motion task before text state update")
                return
            if self.text_mogen_inuse_id is not None:
                assert self.text_mogen_dict is not None, "text_mogen_dict must be set when text_mogen_inuse_id is not None"
                self.text_motion_remain_length = {"motion_id": self.text_mogen_inuse_id,
                                                  "motion_length": self.text_mogen_dict["motion_length"] * 30 + 1 - 30,
                                                  "motion_feats": self.text_mogen_dict["motion_feats"]}
                print(f"[INFO] text_mogen_inuse_id={self.text_mogen_inuse_id}")
                self.text_mogen_inuse_id = None
                self.text_mogen_dict = None

        t0 = time.time()
        try:
            time_logging('MOTION_GEN_START', str(start_id), self.session_dir)
            print(f"[debug] processing {step_idx} user audio: {user_audio_path}, agent audio: {agent_audio_path}")

            if streaming_extractor_agent is not None and streaming_extractor_user is not None:
                target_sr = 15360
                t_load_start = time.time()
                agent_audio, _ = librosa.load(agent_audio_path, sr=target_sr)
                user_audio, _ = librosa.load(user_audio_path, sr=target_sr)
                logger.info(f"[TIMING] Audio load+resample: {time.time() - t_load_start:.3f}s")

                USE_OVERLAP = True
                DROP_LAST_FRAME = True

                t_feat_start = time.time()
                result_left = streaming_extractor_agent.process_chunk(
                    agent_audio, return_overlap=USE_OVERLAP, drop_last_frame=DROP_LAST_FRAME
                )

                result_right = streaming_extractor_user.process_chunk(
                    user_audio, return_overlap=USE_OVERLAP, drop_last_frame=DROP_LAST_FRAME
                )
                logger.info(f"[TIMING] Feature extraction (streaming): {time.time() - t_feat_start:.3f}s")

                if USE_OVERLAP:
                    feats_left, overlap_left = result_left
                    feats_right, overlap_right = result_right
                else:
                    feats_left, feats_right = result_left, result_right
                    overlap_left, overlap_right = None, None

                if feats_left is None or feats_right is None or feats_left.size == 0 or feats_right.size == 0:
                    logger.warning("Streaming feature extraction returned empty, falling back to direct extraction")
                    feats_left = _compute_channel_feats(agent_audio_path, fps=trainer.config.data.framerate)
                    feats_right = _compute_channel_feats(user_audio_path, fps=trainer.config.data.framerate)
                    overlap_left, overlap_right = None, None
            else:
                feats_left = _compute_channel_feats(agent_audio_path, fps=trainer.config.data.framerate)
                feats_right = _compute_channel_feats(user_audio_path, fps=trainer.config.data.framerate)
                overlap_left, overlap_right = None, None

            text_embeddings = None
            have_text = False

            if session_generation is not None and session_generation != self._session_generation:
                logger.info("Discarding stale motion task before text conditioning")
                return

            with self._task_lock:
                if session_generation is not None and session_generation != self._session_generation:
                    logger.info("Discarding stale motion task before remaining text update")
                    return
                if self.text_motion_remain_length is not None:
                    print(f"[INFO] text_motion id: {self.text_motion_remain_length["motion_id"]}, current start_id: {start_id}")
                    if self.text_motion_remain_length["motion_id"] <= start_id:
                        if self.text_motion_remain_length["motion_length"] >= TEXT_MOGEN_PERIOD * feats_left.shape[0]:
                            print(f"[INFO] Skipping motion generation for step {step_idx}, text motion remains {self.text_motion_remain_length["motion_length"]} frames before this skipping, audio feats length {feats_left.shape[0]}*{TEXT_MOGEN_PERIOD}={feats_left.shape[0] * TEXT_MOGEN_PERIOD}")
                            self.text_motion_remain_length["motion_length"] -= TEXT_MOGEN_PERIOD * feats_left.shape[0]


                            text_embeddings = self.text_motion_remain_length["motion_feats"]
                            have_text = True
                        else:
                            print(f"[INFO] text motion remains {self.text_motion_remain_length["motion_length"]} frames before this skipping, audio feats length {feats_left.shape[0]}, do not skip this generation")
                            self.text_motion_remain_length = None
                    else:

                        return


            if self.interruption_handler.is_audio_interrupted(start_id):
                logger.info("Motion generation cancelled due to interruption: audio_id=%s", start_id)
                return

            print('[debug] feats_left shape:', feats_left.shape, 'feats_right shape:', feats_right.shape)
            T_min = min(feats_left.shape[0], feats_right.shape[0])
            print('[debug] T_min:', T_min)
            if feats_left.shape[0] != feats_right.shape[0]:
                print(f"[INFO] L/R length mismatch -> truncate to min T={T_min} (L={feats_left.shape[0]}, R={feats_right.shape[0]})")
            feats_left = feats_left[:T_min]
            feats_right = feats_right[:T_min]

            if args.mode == 'controlnet':
                feats_left = _standardize_2d_lastdim(feats_left, self.audio_input_scaler)
                feats_right = _standardize_2d_lastdim(feats_right, self.audio_input_scaler)
                audio_features = np.concatenate([feats_left, feats_right], axis=1)[None, :, :].astype(np.float32)
                print('[debug] audio_features shape:', audio_features.shape)
            elif args.mode == 'mix':
                feats_all = np.concatenate([feats_left, feats_right], axis=1)
                audio_features = _standardize_2d_lastdim(feats_all, self.audio_input_scaler)[None, :, :].astype(np.float32)
            else:
                raise ValueError(f"Invalid mode: {args.mode}")



            cached_audio_feats = None
            with self.cache_lock:
                if session_generation is not None and session_generation != self._session_generation:
                    logger.info("Discarding stale motion task before cached feature update")
                    return
                if self.cached_audio_feats is not None:
                    if overlap_left is not None and overlap_right is not None:
                        overlap_T = min(overlap_left.shape[0], overlap_right.shape[0])
                        if args.mode == 'controlnet':
                            overlap_left_std = _standardize_2d_lastdim(overlap_left[:overlap_T], self.audio_input_scaler)
                            overlap_right_std = _standardize_2d_lastdim(overlap_right[:overlap_T], self.audio_input_scaler)
                            overlap_combined = np.concatenate([overlap_left_std, overlap_right_std], axis=1)[None, :, :].astype(np.float32)
                        elif args.mode == 'mix':
                            overlap_all = np.concatenate([overlap_left[:overlap_T], overlap_right[:overlap_T]], axis=1)
                            overlap_combined = _standardize_2d_lastdim(overlap_all, self.audio_input_scaler)[None, :, :].astype(np.float32)
                        else:
                            overlap_combined = None

                        if overlap_combined is not None:
                            overlap_tensor = torch.from_numpy(overlap_combined).to(self.cached_audio_feats.device)

                            cache_len = self.cached_audio_feats.shape[1]
                            if cache_len >= overlap_T:
                                self.cached_audio_feats = torch.cat([
                                    self.cached_audio_feats[:, :-overlap_T, :],
                                    overlap_tensor
                                ], dim=1)
                                logger.debug(f"Replaced last {overlap_T} frames in cached_audio_feats with recomputed overlap")

                    cached_audio_feats = self.cached_audio_feats
                cached_motion = self.cached_motion

            with self._task_lock:
                if session_generation is not None and session_generation != self._session_generation:
                    logger.info("Discarding stale motion task before text feature update")
                    return
                cur_text_features, cur_text_motion_length = get_latest_slow_thinking_features(copy_array=True)
                if cur_text_features is not None:
                    text_embeddings = cur_text_features
                    if self.text_input_scaler is not None:
                        text_embeddings = standardize(text_embeddings, self.text_input_scaler)
                    have_text = True
                    self.text_mogen_inuse_id = step_idx
                    self.text_mogen_dict = {"motion_length": cur_text_motion_length,
                                            "motion_feats": text_embeddings}

            print(f'[debug] motion generation started for id {start_id}; text={have_text}')
            generation_fn = (
                server_generate_diffusion_with_recurrent_text_instruction
                if args.mode == 'controlnet'
                else server_generate_diffusion_with_text_instruction
            )
            generation_kwargs = dict(
                audio_feats=audio_features,
                trainer=trainer,
                solver=trainer.config.inference.inference_sovler,
                cached_body_latent=cached_motion,
                cached_audio_feats=cached_audio_feats,
                sample_steps=trainer.config.inference.sample_steps,
                guidance_weight=trainer.config.inference.guidance_weight,
                eta=trainer.config.inference.eta,
                have_text=have_text,
                text_features=text_embeddings,
                mogen_padding=args.mogen_padding,
            )
            if args.mode == 'controlnet':
                generation_kwargs["text_motion_period"] = TEXT_MOGEN_PERIOD
            current_cached_z_body, data, current_cached_audio_feats = generation_fn(**generation_kwargs)
            print(f'[debug] motion generation finished for id {start_id}; text={have_text}')


            if session_generation is not None and session_generation != self._session_generation:
                logger.info("Discarding stale motion result from session generation %s", session_generation)
                return

            if self.interruption_handler.is_audio_interrupted(start_id):
                logger.info("Motion generation result discarded due to interruption: audio_id=%s", start_id)
                return


            with self._task_lock:
                if session_generation is not None and session_generation != self._session_generation:
                    logger.info("Discarding stale motion result before cache update")
                    return
                with self.cache_lock:
                    if self.cached_motion is None:
                        self.cached_motion = current_cached_z_body
                    else:
                        self.cached_motion = torch.cat((self.cached_motion, current_cached_z_body), dim=1)

                    if self.cached_audio_feats is None:
                        self.cached_audio_feats = current_cached_audio_feats
                    else:
                        self.cached_audio_feats = torch.cat((self.cached_audio_feats, current_cached_audio_feats), dim=1)

                    if self.total_motion is None:
                        self.total_motion = data
                    else:
                        data = smooth_transition_tensor(self.total_motion, data, k=15)
                        self.total_motion = torch.cat((self.total_motion, data), dim=0)

            motion_np = data.cpu().numpy()

            if uid:

                with self._task_lock:
                    if session_generation is not None and session_generation != self._session_generation:
                        logger.info("Discarding stale motion publish from session generation %s", session_generation)
                        return
                    if self.interruption_handler.is_audio_interrupted(start_id):
                        logger.info("Motion discarded at publish time: audio_id=%s", start_id)
                        return
                    motion_hub.publish(uid, motion_np, start_id, have_text, trigger_speaker)

            logger.info("Motion job #%s done in %.2fs", step_idx, time.time() - t0)
            time_logging('MOTION_SAVE', str(start_id), self.session_dir)

        except Exception as exc:
            logger.error(
                "Motion generation failed for window %s: %s",
                self.motion_segment_counter,
                exc,
                exc_info=True
            )

    def _merge_audio_files(self, indices: list[int], prefix: str, out_path: str) -> None:
        if not self.session_dir:
            return
        audio_dir = os.path.join(self.session_dir, "audio_log")
        segments = []
        sample_rate = None
        for idx in sorted(indices):
            file_path = os.path.join(audio_dir, f"{prefix}_{idx}.wav")
            if not os.path.exists(file_path):
                logger.warning("Expected audio chunk missing: %s", file_path)
                continue
            data, sr = soundfile.read(file_path, dtype='float32')
            if sample_rate is None:
                sample_rate = sr
            elif sr != sample_rate:
                logger.warning("Sample rate mismatch in turn merge (%s vs %s) for %s", sample_rate, sr, file_path)
            segments.append(data)
        if segments and sample_rate:
            combined = np.concatenate(segments)
            soundfile.write(out_path, combined, sample_rate)
            logger.info("Merged %d chunks into %s", len(segments), out_path)

    def _finalize_turn(self, turn: int) -> None:
        if turn in self.completed_turns:
            return
        if not self.session_dir:
            return

        user_buf = self.window_buffers["user"]
        agent_buf = self.window_buffers["assistant"]

        user_indices = list(self.turn_chunks["user"].get(turn, []))
        assistant_indices = list(self.turn_chunks["assistant"].get(turn, []))
        if not user_indices and not assistant_indices and not user_buf["data"].size and not agent_buf["data"].size:
            logger.info("No audio chunks to finalize for turn %s", turn)
            self.completed_turns.add(turn)
            return

        user_buf["data"] = np.concatenate([user_buf["data"], np.zeros(int((args.turn_complete_padding - args.mogen_padding) * user_buf["sr"]), dtype=np.float32)])
        agent_buf["data"] = np.concatenate([agent_buf["data"], np.zeros(int((args.turn_complete_padding - args.mogen_padding) * agent_buf["sr"]), dtype=np.float32)])

        if user_buf["data"].size:
            if user_buf["data"].size and user_buf["sr"] == 24000:
                try:
                    resampled = resample_poly(user_buf["data"], 2, 3)
                    user_buf["data"] = resampled.astype(np.float32, copy=False)
                    user_buf["sr"] = 16000
                except Exception as exc:
                    logger.error("Failed to resample user buffer to 16kHz: %s", exc)

            print(f"user remaining length: {user_buf['data'].shape[0]}, agent remaining length: {agent_buf['data'].shape[0]}")
            # self.text_mogen_inuse_id = None
            final_speaker = "assistant" if agent_buf["data"].size > 0 else "user"
            self._emit_windowed_motion_segments(self.uid or "", change_id=-1, trigger_speaker=final_speaker)
            self.window_elapsed_sec = 0.0

        audio_dir = os.path.join(self.session_dir, "audio_log")
        user_indices = list(self.turn_chunks["user"].pop(turn, []))
        assistant_indices = list(self.turn_chunks["assistant"].pop(turn, []))
        if not user_indices and not assistant_indices:
            logger.info("No audio chunks to merge for turn %s", turn)
            self.completed_turns.add(turn)
            return
        if user_indices:
            out_path = os.path.join(audio_dir, f"input_turn_{turn}.wav")
            self._merge_audio_files(user_indices, "input_audio", out_path)
        if assistant_indices:
            out_path = os.path.join(audio_dir, f"output_turn_{turn}.wav")
            self._merge_audio_files(assistant_indices, "output_audio", out_path)
        self.completed_turns.add(turn)
        logger.info("Turn %s finalised (user_chunks=%d, assistant_chunks=%d)", turn, len(user_indices), len(assistant_indices))


stream_manager = MotionStreamManagerWithInterruption(
    enable_motion=args.enable_motion,
    port=args.port,
    window_seconds=args.output_save_every_second,
    interruption_handler=interruption_handler,
)


@app.on_event("startup")
async def startup_event() -> None:
    logger.info("Starting application with interruption and text support")
    asyncio.create_task(stream_manager.check_activity())
    motion_hub.loop = asyncio.get_running_loop()
    motion_hub.start()


@app.on_event("shutdown")
async def shutdown_event() -> None:
    logger.info("Shutting down application")
    motion_hub.stop()
    slow_thinking_clip_worker.stop()


@app.post("/stream")
@app.post("/api/v1/stream")
async def stream(request: Request, uid: Optional[str] = Header(None)):
    if not uid:
        raise HTTPException(status_code=400, detail="Missing uid in headers")

    stream_manager.update_last_request_time()
    try:
        data = await request.json()
    except json.JSONDecodeError as exc:
        raise HTTPException(status_code=400, detail=f"Invalid JSON: {exc}")

    try:
        finish_reason = stream_manager.process_payload(uid, data)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        logger.error("Unexpected error in /stream: %s", exc)
        raise HTTPException(status_code=500, detail="Internal server error")

    stream_manager.update_last_stream_time()
    response = {
        "id": uid,
        "choices": {
            "role": "assistant",
            "content": "success",
            "finish_reason": finish_reason,
        },
    }
    return JSONResponse(content=response, status_code=200)


@app.post("/api/v1/interrupt")
async def interrupt(request: Request, uid: Optional[str] = Header(None)):
    """Accept an interruption event and return the current event sequence."""
    if not uid:
        raise HTTPException(status_code=400, detail="Missing uid in headers")

    try:
        data = await request.json()
    except json.JSONDecodeError as exc:
        raise HTTPException(status_code=400, detail=f"Invalid JSON: {exc}")

    interruption_data = data.get("interruption", data)
    success = interruption_handler.handle_interruption(interruption_data)

    return JSONResponse(
        content={
            "id": uid,
            "status": "ok" if success else "ignored",
            "sequence": interruption_handler.get_current_sequence(),
        },
        status_code=200
    )


@app.websocket("/ws/motion")
async def websocket_motion(websocket: WebSocket, uid: Optional[str] = Query(None)):
    if not uid:
        await websocket.close(code=4000, reason="Missing uid in request")
        return
    await websocket.accept()
    motion_hub.add_connection(uid, websocket)
    try:
        while True:
            # Reading from the socket lets Starlette surface a client close;
            # an indefinite sleep leaves the connection task alive forever
            # and prevents clean application shutdown.
            message = await websocket.receive()
            if message.get("type") == "websocket.disconnect":
                break
    except WebSocketDisconnect:
        pass
    except Exception as exc:
        logger.error("/ws/motion error: %s", exc)
    finally:
        motion_hub.remove_connection(uid, websocket)


@app.get("/api/v1/time")
async def get_server_time():
    return {"server_time": time.time_ns()}


@app.post("/api/v1/set_zero_time")
async def set_zero_time():
    global zero_point_ns
    zero_point_ns = time.time_ns()
    logger.info(f"Server zero point set to: {zero_point_ns}")
    stream_manager.reset_session()
    logger.info("Session reset after setting zero point")
    return {"status": "ok", "zero_point": zero_point_ns}


@app.get('/health')
@app.get('/api/v1/health')
async def health_check():
    return {"status": "OK", "interruption_support": True, "text_support": True}


@app.get('/api/v1/interruption_stats')
async def interruption_stats():
    """Return the current interruption sequence and invalidated window count."""
    return {
        "current_sequence": interruption_handler.get_current_sequence(),
        "interrupted_audio_count": len(interruption_handler._interrupted_audio_ids),
    }


if __name__ == "__main__":
    uvicorn.run(app, host=args.host, port=args.port)
