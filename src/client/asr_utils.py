import io
import json
import logging
import math
import os
import queue
import re
import threading
import time
import wave
from typing import Any, Dict, List, Optional, Tuple
from collections import deque

import numpy as np

try:
    import sherpa_onnx
except ImportError:  # pragma: no cover - optional dependency
    sherpa_onnx = None

# The model is optional and can be supplied through SHERPA_MODEL_DIR.
DEFAULT_SHERPA_MODEL_DIR = os.path.join(os.path.dirname(__file__), "data", "asr_models")
TARGET_SAMPLE_RATE = 24000
SAMPLE_WIDTH = 2  # bytes for int16
CONF_THRESHOLD = 0.6


class LiveTranscriptStore:
    """Tracks partial (chunk-level) and finalized transcripts per speaker/turn."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._partial: Dict[Tuple[str, int], List[Tuple[int, int, str]]] = {}
        self._final: Dict[Tuple[str, int], str] = {}
        # The two most recent ASR segments, shown for the current turn.
        self._recent: Dict[Tuple[str, int], deque[str]] = {}

    def add_partial(
        self,
        speaker: str,
        turn: int,
        index: int,
        offset_ms: int,
        text: str,
    ) -> str:
        if not text or not text.strip():
            norm = ""
        else:
            norm = text.strip()
        key = (speaker, int(turn))
        with self._lock:
            if norm:
                buf = self._partial.setdefault(key, [])
                buf.append((offset_ms, index, norm))
                buf.sort(key=lambda x: (x[0], x[1]))
                recent_buf = self._recent.setdefault(key, deque(maxlen=2))
                recent_buf.append(norm)
                return " ".join(seg for _, _, seg in buf)
            # Write an empty recent placeholder so consumers can detect an empty turn.
            recent_buf = self._recent.setdefault(key, deque(maxlen=2))
            recent_buf.append("")
            return ""

    def finalize(self, speaker: str, turn: int, text: Optional[str] = None) -> str:
        key = (speaker, int(turn))
        with self._lock:
            final_text = (text or "").strip()
            if final_text:
                self._final[key] = final_text
            # self._partial.pop(key, None)
            return final_text

    def snapshot(self) -> Dict[str, Dict[str, str]]:
        with self._lock:
            partial = {
                f"{spk}-turn{turn}": " ".join(seg for _, _, seg in entries)
                for (spk, turn), entries in self._partial.items()
                if entries
            }
            final = {
                f"{spk}-turn{turn}": text
                for (spk, turn), text in self._final.items()
                if text
            }
            recent = {
                f"{spk}-turn{turn}": " ".join(list(deq))
                for (spk, turn), deq in self._recent.items()
                if deq
            }
            return {"partial": partial, "final": final, "recent": recent}

    def format_partial_lines(self) -> str:
        """Return partial lines with role/turn tagging (User/Agent-turn_X)."""
        snap = self.snapshot()
        lines = []

        def _parse_key(key: str) -> tuple[Optional[str], Optional[int]]:
            if "-turn" not in key:
                return None, None
            speaker, turn_part = key.split("-turn", 1)
            try:
                turn_num = int(turn_part)
            except Exception:
                return speaker, None
            return speaker, turn_num

        for key, text in snap["partial"].items():
            if not text:
                continue
            speaker, turn_num = _parse_key(key)
            role = "User" if speaker == "user" else "Agent"
            turn_suffix = f"-turn_{turn_num}" if turn_num is not None else ""
            lines.append(f"{role}{turn_suffix} (partial): {text}")
        return "\n".join(lines).strip()

    def reset(self) -> None:
        with self._lock:
            self._partial.clear()
            self._final.clear()
            self._recent.clear()


def _linear_resample(x: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    if orig_sr == target_sr:
        return x
    duration = len(x) / orig_sr
    target_len = int(duration * target_sr)
    old_idx = np.linspace(0, len(x) - 1, num=len(x))
    new_idx = np.linspace(0, len(x) - 1, num=target_len)
    return np.interp(new_idx, old_idx, x).astype(np.float32)


def _wav_bytes_to_float32(audio_bytes: bytes) -> Tuple[Optional[np.ndarray], Optional[int]]:
    try:
        with wave.open(io.BytesIO(audio_bytes), "rb") as wf:
            channels = wf.getnchannels()
            sampwidth = wf.getsampwidth()
            sr = wf.getframerate()
            if channels != 1 or sampwidth != SAMPLE_WIDTH:
                return None, None
            frames = wf.readframes(wf.getnframes())
            samples = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
            return samples, sr
    except Exception:
        return None, None


def _tokens_to_words(tokens: List[str], timestamps: List[float]) -> List[Dict[str, Any]]:
    """Convert sentencepiece tokens + timestamps to word spans."""
    if not tokens or not timestamps or len(tokens) != len(timestamps):
        return []
    words: List[Dict[str, Any]] = []
    cur_word = ""
    cur_start: Optional[float] = None
    last_ts: Optional[float] = None

    def _flush():
        nonlocal cur_word, cur_start, last_ts
        if cur_word and cur_start is not None:
            words.append(
                {
                    "word": cur_word.strip(),
                    "start": cur_start,
                    "end": last_ts if last_ts is not None else cur_start,
                }
            )
        cur_word = ""
        cur_start = None

    for tok, ts in zip(tokens, timestamps):
        try:
            t = float(ts)
        except Exception:
            continue
        last_ts = t
        piece = tok
        is_new = piece.startswith("▁")
        if is_new:
            _flush()
            piece = piece.lstrip("▁")
            cur_start = t
            cur_word = piece
        else:
            if cur_start is None:
                cur_start = t
            cur_word += piece
    _flush()
    return [w for w in words if w.get("word")]

def compute_confidence(result) -> Optional[float]:
    """
    Rough confidence estimate using ys_log_probs:
    - ys_log_probs is log(prob) per token; closer to 0 means more confident.
    - exp back to probs and take the average.
    """
    ys_log_probs = getattr(result, "ys_log_probs", None)
    if not ys_log_probs:
        return None
    if isinstance(ys_log_probs[0], (list, tuple)):
        flat = [p for row in ys_log_probs for p in row]
    else:
        flat = ys_log_probs
    if not flat:
        return None
    probs = [math.exp(lp) for lp in flat]
    return float(sum(probs) / len(probs))


def _normalize_for_compare(text: str) -> str:
    return re.sub(r"\s+", "", text or "").lower()


def _token_words_match_text(words: List[Dict[str, Any]], text: str) -> bool:
    if not words or not text:
        return False
    joined = "".join(w.get("word", "") for w in words)
    return _normalize_for_compare(joined) == _normalize_for_compare(text)


def _is_cjk(char: str) -> bool:
    code = ord(char)
    return (
        0x4E00 <= code <= 0x9FFF
        or 0x3400 <= code <= 0x4DBF
        or 0x20000 <= code <= 0x2A6DF
        or 0x2A700 <= code <= 0x2B73F
        or 0x2B740 <= code <= 0x2B81F
        or 0x2B820 <= code <= 0x2CEAF
        or 0xF900 <= code <= 0xFAFF
        or 0x2F800 <= code <= 0x2FA1F
    )


def _split_bilingual_text(text: str) -> List[str]:
    if not text:
        return []
    segments: List[str] = []
    buffer = ""
    mode: Optional[str] = None

    def _flush() -> None:
        nonlocal buffer, mode
        if buffer:
            segments.append(buffer.lower() if mode == "latin" else buffer)
        buffer = ""
        mode = None

    for ch in text:
        if ch.isspace():
            _flush()
            continue
        if _is_cjk(ch):
            _flush()
            segments.append(ch)
            continue
        if ch.isalnum() or ch in {"'", "’", "-"}:
            if mode != "latin":
                _flush()
                mode = "latin"
            buffer += ch
            continue
        _flush()
    _flush()
    return [seg for seg in segments if seg]


def _text_to_words(text: str, base_offset: float) -> List[Dict[str, Any]]:
    segments = _split_bilingual_text(text)
    if not segments:
        return []
    unit = 0.01  # keep ordering stable without real timestamps
    words: List[Dict[str, Any]] = []
    for idx, seg in enumerate(segments):
        start = base_offset + idx * unit
        words.append({"word": seg, "start": start, "end": start})
    return words


def _extract_words(result, offset_sec: float) -> List[Dict[str, Any]]:
    tokens = getattr(result, "tokens", None)
    timestamps = getattr(result, "timestamps", None)
    words = _tokens_to_words(tokens or [], timestamps or [])
    text_val = (getattr(result, "text", None) or "").strip()
    if not _token_words_match_text(words, text_val):
        # fallback: use full text to build pseudo timestamps
        words = _text_to_words(text_val, base_offset=offset_sec)
    if offset_sec > 0.0 and words:
        for w in words:
            w["start"] = float(w.get("start", 0.0)) - offset_sec
            w["end"] = float(w.get("end", w.get("start", 0.0))) - offset_sec
    return words


def _tail_from_words(words: List[Dict[str, Any]], cutoff: float = 0.0) -> str:
    return " ".join(w["word"] for w in words if w.get("end", 0.0) > cutoff).strip()


def _dedup_overlap(prev_tail: str, curr_tail: str) -> str:
    """If last word of prev_tail == first word of curr_tail, drop the first word of curr_tail."""
    if not prev_tail or not curr_tail:
        return curr_tail.strip()
    prev_last = prev_tail.strip().split()[-1]
    curr_words = curr_tail.strip().split()
    if curr_words and curr_words[0] == prev_last:
        curr_words = curr_words[1:]
    return " ".join(curr_words).strip()


def _strip_overlap_prefix(prev_tail: str, curr_tail: str) -> str:
    """
    Token-level overlap removal: if suffix of prev tokens == prefix of curr tokens,
    drop the overlapping prefix from curr.
    """
    prev_tokens = [t for t in prev_tail.strip().split() if t]
    curr_tokens = [t for t in curr_tail.strip().split() if t]
    if not prev_tokens or not curr_tokens:
        return curr_tail.strip()
    max_k = 0
    max_len = min(len(prev_tokens), len(curr_tokens))
    for k in range(1, max_len + 1):
        if prev_tokens[-k:] == curr_tokens[:k]:
            max_k = k
    if max_k > 0:
        curr_tokens = curr_tokens[max_k:]
    return " ".join(curr_tokens).strip()


class SherpaASRWorker:
    """
    Sherpa-ONNX ASR worker with simple 1-chunk cache to reduce boundary truncation.
    Resets cache when speaker/turn changes.
    """

    def __init__(
        self,
        model_dir: str = DEFAULT_SHERPA_MODEL_DIR,
        log_dir: str = ".",
        on_result=None,
    ):
        self.model_dir = model_dir
        self.log_dir = log_dir
        self._queue: "queue.Queue[Optional[dict]]" = queue.Queue()
        self._stop = threading.Event()
        self._ready = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._recognizer = None
        self._log_path = os.path.join(self.log_dir, "asr_transcripts.jsonl")
        self._logger = logging.getLogger("robot_gesture_client.asr")
        self._on_result = on_result
        self._prev_samples: Optional[np.ndarray] = None
        self._prev_tail_text: str = ""
        self._last_key: Optional[Tuple[str, int]] = None

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self._ready.clear()
        self._thread = threading.Thread(target=self._run, name="SherpaASR", daemon=True)
        self._thread.start()

    def stop(self, join_timeout: float = 2.0) -> None:
        self._stop.set()
        try:
            self._queue.put_nowait(None)
        except queue.Full:
            pass
        if self._thread:
            self._thread.join(timeout=join_timeout)
            self._thread = None
        self._ready.clear()
        self._prev_samples = None
        self._prev_tail_text = ""
        self._last_key = None

    def wait_until_ready(self, timeout: float = 30.0) -> bool:
        return self._ready.wait(timeout)

    def submit(
        self,
        path: Optional[str],
        sample_rate: int,
        speaker: str,
        turn: int,
        index: int,
        offset_ms: int,
        audio_bytes: Optional[bytes] = None,
    ) -> None:
        if self._stop.is_set():
            return
        try:
            self._queue.put_nowait(
                {
                    "path": path,
                    "sample_rate": sample_rate,
                    "speaker": speaker,
                    "turn": turn,
                    "index": index,
                    "offset_ms": offset_ms,
                    "audio_bytes": audio_bytes,
                    "submitted_ts": time.time(),
                }
            )
        except queue.Full:
            self._logger.warning("ASR queue full; dropped %s", path)

    # Internal ------------------------------------------------------------
    def _run(self) -> None:
        if not self._ensure_model():
            self._ready.set()
            return
        self._logger.info("Sherpa ASR worker started using model at %s", self.model_dir)
        self._ready.set()
        while True:
            try:
                job = self._queue.get(timeout=0.5 if self._stop.is_set() else None)
            except queue.Empty:
                if self._stop.is_set():
                    break
                continue
            if job is None:
                break
            try:
                self._handle_job(job)
            except Exception as exc:  # pragma: no cover - defensive
                self._logger.error("ASR job failed for %s: %s", job.get("path"), exc)

    def _handle_job(self, job: dict) -> None:
        speaker = job.get("speaker") or ""
        turn = int(job.get("turn") or 0)
        key = (speaker, turn)

        # Reset cache on speaker/turn change
        if key != self._last_key:
            self._prev_samples = None
            self._prev_tail_text = ""

        t_start = time.time()
        submit_ts = float(job.get("submitted_ts") or t_start)
        audio_bytes = job.get("audio_bytes")
        actual_rate = job.get("sample_rate")
        if audio_bytes is None:
            audio_bytes, actual_rate = self._read_wav(job.get("path"), actual_rate)
        samples = None
        if audio_bytes:
            samples, actual_rate = self._wav_to_samples(audio_bytes, actual_rate)
        if samples is None or actual_rate is None:
            return

        merged, prev_dur = self._merge_with_cache(samples)

        stream = self._recognizer.create_stream()
        stream.accept_waveform(TARGET_SAMPLE_RATE, merged)
        self._recognizer.decode_stream(stream)
        result = stream.result
        t_asr_done = time.time()

        words = _extract_words(result, offset_sec=prev_dur)
        tail_raw = _tail_from_words(words, cutoff=0.0).lower()
        tail = _dedup_overlap(self._prev_tail_text, tail_raw).lower()
        tail = _strip_overlap_prefix(self._prev_tail_text, tail)
        text_full = (result.text or "").strip().lower()

        conf = compute_confidence(result)
        use_text = True
        skip_reason = None
        if not text_full:
            use_text = False
            skip_reason = "empty"
        elif conf is not None and conf < CONF_THRESHOLD:
            use_text = False
            skip_reason = f"low_conf({conf:.2f})"

        self._write_log(job, tail if use_text else "", text_full, confidence=conf, skip_reason=skip_reason)
        if use_text and self._on_result:
            try:
                self._on_result(
                    speaker=speaker,
                    turn=turn,
                    index=job.get("index"),
                    offset_ms=job.get("offset_ms"),
                    sample_rate=TARGET_SAMPLE_RATE,
                    text=tail,
                )
            except Exception as exc:  # pragma: no cover - defensive
                self._logger.error("ASR on_result failed: %s", exc)

        # Update cache for next chunk of same speaker/turn
        self._prev_samples = samples
        if use_text:
            self._prev_tail_text = tail
        self._last_key = key

        self._logger.debug(
            "[ASR timing] queue_wait=%.1fms asr=%.1fms total=%.1fms",
            (t_start - submit_ts) * 1000.0,
            (t_asr_done - t_start) * 1000.0,
            (t_asr_done - submit_ts) * 1000.0,
        )

    def _ensure_model(self) -> bool:
        if sherpa_onnx is None:
            self._logger.error("sherpa_onnx not installed; ASR worker disabled.")
            return False
        if self._recognizer is not None:
            return True
        model_dir = os.environ.get("SHERPA_MODEL_DIR", self.model_dir)
        if not os.path.isdir(model_dir):
            self._logger.error("Sherpa model dir not found at %s", model_dir)
            return False
        try:
            self._recognizer = sherpa_onnx.OfflineRecognizer.from_transducer(
                tokens=f"{model_dir}/tokens.txt",
                encoder=f"{model_dir}/encoder-epoch-34-avg-19.onnx",
                decoder=f"{model_dir}/decoder-epoch-34-avg-19.onnx",
                joiner=f"{model_dir}/joiner-epoch-34-avg-19.onnx",
                sample_rate=TARGET_SAMPLE_RATE,
                feature_dim=80,
                decoding_method="greedy_search",
                num_threads=2,
                debug=False,
            )
            return True
        except Exception as exc:
            self._logger.error("Failed to load sherpa model: %s", exc)
            return False

    def _wav_to_samples(self, audio_bytes: bytes, sample_rate: Optional[int]) -> Tuple[Optional[np.ndarray], Optional[int]]:
        samples, sr = _wav_bytes_to_float32(audio_bytes)
        if samples is None or sr is None:
            return None, None
        if sr != TARGET_SAMPLE_RATE:
            samples = _linear_resample(samples, sr, TARGET_SAMPLE_RATE)
            sr = TARGET_SAMPLE_RATE
        return samples, sr

    def _merge_with_cache(self, samples: np.ndarray) -> Tuple[np.ndarray, float]:
        if self._prev_samples is not None and self._prev_samples.size > 0:
            prev_dur = len(self._prev_samples) / float(TARGET_SAMPLE_RATE)
            merged = np.concatenate([self._prev_samples, samples])
            return merged, prev_dur
        return samples, 0.0

    def _read_wav(self, path: str, expected_rate: Optional[int]) -> tuple[Optional[bytes], Optional[int]]:
        if not path or not os.path.isfile(path):
            self._logger.warning("ASR wav file missing: %s", path)
            return None, None
        try:
            with open(path, "rb") as f:
                data = f.read()
            with wave.open(io.BytesIO(data), "rb") as wf:
                framerate = wf.getframerate()
            return data, framerate
        except Exception as exc:
            self._logger.error("Failed to read wav for ASR %s: %s", path, exc)
        return None, None

    def _write_log(
        self,
        job: dict,
        tail_text: str,
        full_text: str,
        confidence: Optional[float] = None,
        skip_reason: Optional[str] = None,
    ) -> None:
        try:
            os.makedirs(self.log_dir, exist_ok=True)
            payload = {
                "timestamp_ms": int(time.time() * 1000),
                "speaker": job.get("speaker"),
                "turn": job.get("turn"),
                "index": job.get("index"),
                "offset_ms": job.get("offset_ms"),
                "sample_rate": TARGET_SAMPLE_RATE,
                "text_tail": tail_text.strip(),
                "text_full": full_text,
                "confidence": confidence,
                "skip_reason": skip_reason,
                "path": job.get("path"),
            }
            with open(self._log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(payload, ensure_ascii=False) + "\n")
            if tail_text:
                self._logger.info(
                    "ASR %s turn=%s idx=%s: %s",
                    job.get("speaker"),
                    job.get("turn"),
                    job.get("index"),
                    tail_text,
                )
        except Exception as exc:
            self._logger.error("Failed to write ASR log for %s: %s", job.get("path"), exc)
