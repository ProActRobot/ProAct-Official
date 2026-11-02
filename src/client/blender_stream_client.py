import base64
import json
import logging
import os
import socket
import struct
import threading
import time
import wave
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence
from types import SimpleNamespace
import importlib

import pandas as pd

BLENDER_STREAM_HOST = os.environ.get("BLENDER_STREAM_HOST", "localhost")
BLENDER_STREAM_PORT = int(os.environ.get("BLENDER_STREAM_PORT", "9999"))
BLENDER_FRAME_INTERVAL = float(os.environ.get("BLENDER_FRAME_INTERVAL", str(1.0 / 30.0)))
BLENDER_AUDIO_CHUNK_SECONDS = float(os.environ.get("BLENDER_AUDIO_CHUNK_SECONDS", "1.0"))
BLENDER_CONJUGATE_CONFIG = os.environ.get(
    "BLENDER_CONJUGATE_CONFIG",
    os.path.join(os.path.dirname(__file__), "blender_conjugate_config.json"),
)
BLENDER_CONJUGATE_REF_BVH = os.environ.get(
    "BLENDER_CONJUGATE_REF_BVH",
)

logger = logging.getLogger(__name__)


def mat3_mul(a: List[List[float]], b: List[List[float]]) -> List[List[float]]:
    return [
        [sum(a[i][k] * b[k][j] for k in range(3)) for j in range(3)]
        for i in range(3)
    ]


def mat3_transpose(m: List[List[float]]) -> List[List[float]]:
    return [[m[j][i] for j in range(3)] for i in range(3)]


def mat3_identity() -> List[List[float]]:
    return [
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
    ]


def vec_normalize(v: Sequence[float]) -> List[float]:
    l = math.sqrt(sum(c * c for c in v))
    if l < 1e-6:
        return [0.0, 1.0, 0.0]
    return [v[0] / l, v[1] / l, v[2] / l]


def vec_cross(a: Sequence[float], b: Sequence[float]) -> List[float]:
    return [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]


def vec_dot(a: Sequence[float], b: Sequence[float]) -> float:
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def direction_to_matrix(target_vec: Sequence[float]) -> List[List[float]]:
    v_y = vec_normalize(target_vec)
    v_z_up = [0.0, 0.0, 1.0]
    if abs(vec_dot(v_y, v_z_up)) > 0.99:
        v_z_up = [1.0, 0.0, 0.0]
    v_x = vec_normalize(vec_cross(v_y, v_z_up))
    v_z = vec_normalize(vec_cross(v_x, v_y))
    return [
        [v_x[0], v_y[0], v_z[0]],
        [v_x[1], v_y[1], v_z[1]],
        [v_x[2], v_y[2], v_z[2]],
    ]


def euler_to_mat3(vals: Sequence[float], order: str = "ZXY") -> List[List[float]]:
    """Interpret angles in the supplied axis order, such as ``ZXY``."""
    angle_map = {ax: math.radians(v) for ax, v in zip(order, vals)}
    rx = angle_map.get("X", 0.0)
    ry = angle_map.get("Y", 0.0)
    rz = angle_map.get("Z", 0.0)

    cx, sx = math.cos(rx), math.sin(rx)
    cy, sy = math.cos(ry), math.sin(ry)
    cz, sz = math.cos(rz), math.sin(rz)

    Rx = [[1, 0, 0], [0, cx, -sx], [0, sx, cx]]
    Ry = [[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]]
    Rz = [[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]]
    d = {"X": Rx, "Y": Ry, "Z": Rz}
    mats = [d[a] for a in order]
    return mat3_mul(mat3_mul(mats[0], mats[1]), mats[2])


def mat3_to_euler_xyz_deg(R: List[List[float]]) -> List[float]:
    sy = math.sqrt(R[0][0] * R[0][0] + R[1][0] * R[1][0])
    singular = sy < 1e-6
    if not singular:
        x = math.atan2(R[2][1], R[2][2])
        y = math.atan2(-R[2][0], sy)
        z = math.atan2(R[1][0], R[0][0])
    else:
        x = math.atan2(-R[1][2], R[1][1])
        y = math.atan2(-R[2][0], sy)
        z = 0.0
    return [math.degrees(x), math.degrees(y), math.degrees(z)]


def parse_conjugation_from_bvh(bvh_path: str) -> Optional[Dict[str, Any]]:
    if not os.path.exists(bvh_path):
        logger.warning("[Blender] BVH reference not found: %s", bvh_path)
        return None
    try:
        with open(bvh_path, "r", encoding="utf-8", errors="ignore") as f:
            lines = f.readlines()
    except Exception as exc:
        logger.warning("[Blender] Failed to read BVH: %s", exc)
        return None

    class Node:
        def __init__(self, name, parent=None):
            self.name = name
            self.parent = parent
            self.offset = [0.0, 0.0, 0.0]
            self.channels: List[str] = []
            self.children: List["Node"] = []

    nodes: Dict[str, Node] = {}
    stack: List[Node] = []
    root: Optional[Node] = None
    for raw in lines:
        line = raw.strip()
        if not line:
            continue
        tokens = line.split()
        head = tokens[0]
        if head in ("ROOT", "JOINT"):
            name = tokens[1]
            parent = stack[-1] if stack else None
            node = Node(name, parent)
            nodes[name] = node
            if parent:
                parent.children.append(node)
            else:
                root = node
            stack.append(node)
        elif head == "End":
            parent = stack[-1] if stack else None
            name = (parent.name + "_End") if parent else "EndSite"
            node = Node(name, parent)
            if parent:
                parent.children.append(node)
            nodes[name] = node
            stack.append(node)
        elif head == "OFFSET" and stack:
            stack[-1].offset = [float(tokens[1]), float(tokens[2]), float(tokens[3])]
        elif head == "CHANNELS" and stack:
            stack[-1].channels = tokens[2:]
        elif head == "}":
            if stack:
                stack.pop()
        elif head == "MOTION":
            break

    if root is None:
        logger.warning("[Blender] BVH parse failed: no ROOT")
        return None

    bones: Dict[str, Dict[str, Any]] = {}
    for name, node in nodes.items():
        if not node.channels:
            continue
        order = "".join([c[0].upper() for c in node.channels if "rotation" in c.lower()])
        if not order:
            continue
        if node.children:
            child = node.children[0]
            off = child.offset
            length = math.sqrt(off[0] ** 2 + off[1] ** 2 + off[2] ** 2)
            rest = mat3_identity() if length < 1e-6 else direction_to_matrix(off)
        else:
            rest = mat3_identity()
        bones[name] = {
            "rot_order": order,
            "rest_matrix": rest,
        }

    return {
        "root_name": root.name,
        "root_remap_Q": [[0, -1, 0], [1, 0, 0], [0, 0, 1]],
        "bones": bones,
    }


def load_conjugation_config(config_path: Optional[str], ref_bvh_path: Optional[str]) -> Optional[Dict[str, Any]]:
    """Load a local bone alignment config or generate one from a reference BVH."""
    if config_path and os.path.exists(config_path):
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as exc:
            logger.warning("[Blender] Failed to load conjugation config %s: %s", config_path, exc)
    config: Optional[Dict[str, Any]] = None
    if ref_bvh_path:
        config = parse_conjugation_from_bvh(ref_bvh_path)
        if config and config_path:
            try:
                with open(config_path, "w", encoding="utf-8") as f:
                    json.dump(config, f, ensure_ascii=False, indent=2)
                logger.info("[Blender] Saved conjugation config to %s", config_path)
            except Exception as exc:
                logger.warning("[Blender] Failed to save conjugation config: %s", exc)
    return config


class BlenderStreamClient:
    """Stream queued motion frames and synchronized audio to Blender over TCP."""

    def __init__(
        self,
        host: str = BLENDER_STREAM_HOST,
        port: int = BLENDER_STREAM_PORT,
        enabled: bool = False,
        frame_interval: float = BLENDER_FRAME_INTERVAL,
        audio_chunk_seconds: float = BLENDER_AUDIO_CHUNK_SECONDS,
        audio_dir: Optional[str] = None,
        conjugate_config_path: Optional[str] = BLENDER_CONJUGATE_CONFIG,
        ref_bvh_path: Optional[str] = BLENDER_CONJUGATE_REF_BVH,
        enable_visualizer: bool = False,
        visualizer_title: str = "Blender Stream",
    ) -> None:
        self.host = host
        self.port = port
        self.enabled = enabled
        self.frame_interval = frame_interval
        self.audio_chunk_seconds = audio_chunk_seconds
        self.audio_dir = audio_dir
        self.conjugate_config_path = conjugate_config_path
        self.ref_bvh_path = ref_bvh_path
        self._sock: Optional[socket.socket] = None
        self._lock = threading.Lock()

        self.conjugate = load_conjugation_config(conjugate_config_path, ref_bvh_path)


        self._motion_queue: List[Dict[str, Any]] = []
        self._chunk_ranges: List[tuple] = []
        self._current_frame = 0
        self._current_chunk_audio_id: Optional[int] = None
        self._sent_audio_ids: set = set()
        self._queue_lock = threading.Lock()
        self._stop_event = threading.Event()
        self._playback_thread: Optional[threading.Thread] = None


        self._pending_interrupt = False
        self._interrupt_after_audio_id: Optional[int] = None

        self.mapping: Optional[Dict[str, Dict[str, Any]]] = None
        self._fast_mapping: Optional[List[Dict[str, Any]]] = None
        self._mapping_cols_sig: Optional[tuple] = None



        self._viz = None
        if enable_visualizer:
            try:
                streaming_common = importlib.import_module("streaming_common_blender")
                clock = streaming_common.Clock()
                clock.set_to(0.0)
                audio_stub = SimpleNamespace(available_audio={})
                fps_for_viz = (1.0 / frame_interval) if frame_interval > 0 else streaming_common.FPS
                motion_stub = SimpleNamespace(
                    chunk_ranges=[],
                    unplayed_chunk_start_ids=[],
                    audio_queue=audio_stub,
                    fps=fps_for_viz,
                    current_frame=0,
                )
                viz = streaming_common.GanttVisualizer(clock, motion_stub, title=visualizer_title)
                t = threading.Thread(target=viz.run, name="BlenderViz", daemon=True)
                t.start()
                self._viz = SimpleNamespace(clock=clock, motion=motion_stub, viz=viz, thread=t)
                logger.info("[Blender] Visualizer started")
            except Exception as exc:
                logger.warning("[Blender] Failed to start visualizer: %s", exc)

    def update_audio_dir(self, audio_dir: Optional[str]) -> None:
        self.audio_dir = audio_dir

    def start_playback(self) -> None:
        """Start the background playback thread for streaming frames to Blender."""
        if not self.enabled:
            return
        if self._playback_thread is not None and self._playback_thread.is_alive():
            return  # Already running
        self._stop_event.clear()
        self._playback_thread = threading.Thread(
            target=self._playback_loop,
            name="BlenderPlaybackThread",
            daemon=True,
        )
        self._playback_thread.start()
        logger.info("[Blender] Playback thread started")

    def stop_playback(self) -> None:
        """Stop the background playback thread."""
        self._stop_event.set()
        if self._playback_thread is not None:
            self._playback_thread.join(timeout=2.0)
            self._playback_thread = None
        logger.info("[Blender] Playback thread stopped")

    def _playback_loop(self) -> None:
        """Consume the queue at a fixed rate and send audio when a chunk starts."""
        next_frame_time = time.time()

        while not self._stop_event.is_set():
            with self._queue_lock:
                if not self._motion_queue or self._current_frame >= len(self._motion_queue):
                    # No frames to play, wait a bit
                    time.sleep(0.01)
                    continue


                chunk = self._get_chunk_for_frame(self._current_frame)
                if chunk is None:

                    if self._current_chunk_audio_id is not None:
                        logger.info(
                            "[Blender] Finished motion chunk %s (frames %s-%s)",
                            self._current_chunk_audio_id,
                            "?",
                            "?",
                        )
                        self._current_chunk_audio_id = None
                    time.sleep(0.01)
                    continue
                current_audio_id, cur_start, cur_end = chunk

                if current_audio_id != self._current_chunk_audio_id:
                    if self._current_chunk_audio_id is not None:
                        logger.info(
                            "[Blender] Finished motion chunk %s (frames %s-%s)",
                            self._current_chunk_audio_id,
                            (cur_start - 30) if cur_start is not None else "?",
                            (cur_end - 30) if cur_end is not None else "?",
                        )
                    logger.info(
                        "[Blender] Started motion chunk %s (frames %s-%s)",
                        current_audio_id,
                        cur_start,
                        cur_end,
                    )
                    if self._viz is not None:
                        try:
                            if current_audio_id in self._viz.motion.unplayed_chunk_start_ids:
                                self._viz.motion.unplayed_chunk_start_ids.remove(current_audio_id)
                        except Exception:
                            pass

                    logger.debug(
                        "[BlenderDebug] _current_frame=%s, total_frames=%s, chunk_ranges_tail=%s",
                        self._current_frame,
                        len(self._motion_queue),
                        self._chunk_ranges[-5:],
                    )


                    self._current_chunk_audio_id = current_audio_id
                if current_audio_id not in self._sent_audio_ids:


                    if self._pending_interrupt:
                        logger.debug(
                            "[Blender] Skipping audio id=%s during pending interrupt (waiting for new motion)",
                            current_audio_id
                        )

                        self._sent_audio_ids.add(current_audio_id)
                    else:
                        self._try_send_audio(current_audio_id)
                        self._sent_audio_ids.add(current_audio_id)

                # Get current frame
                pose = self._motion_queue[self._current_frame]
                self._current_frame += 1
                if self._viz is not None:
                    try:
                        self._viz.motion.current_frame = self._current_frame
                    except Exception:
                        pass

            # Send frame to Blender
            ok = self._send_packet({"type": "frames", "frames": [pose]})
            if not ok:
                logger.warning("[Blender] Failed to send frame, will retry")
                time.sleep(0.1)
                continue

            # Control frame rate
            next_frame_time += self.frame_interval
            sleep_time = next_frame_time - time.time()
            if sleep_time > 0:
                time.sleep(sleep_time)
            elif sleep_time < -1.0:
                # If lagging significantly, reset timer
                next_frame_time = time.time()
                logger.debug("[Blender] Playback lagging, resetting timer")

    def _get_chunk_for_frame(self, frame_idx: int) -> Optional[tuple]:
        """
        Return the chunk covering a frame index: (audio_id, start_frame, end_frame).
        Audio IDs can repeat; return the range that actually covers the frame.
        """
        for audio_id, start_frame, end_frame in self._chunk_ranges:
            if start_frame <= frame_idx <= end_frame:
                return (audio_id, start_frame, end_frame)
        return None

    def _try_send_audio(self, audio_id: int) -> bool:
        """Try to send an audio file immediately."""
        audio_root = self.audio_dir
        if not audio_root:
            logger.debug("[Blender] No audio dir set, skipping audio for id=%s", audio_id)
            return False

        target = os.path.join(audio_root, f"output_{audio_id}.wav")

        # Quick non-blocking check - if file exists, send it immediately
        if os.path.exists(target):
            try:
                self.send_audio_file(target)
                logger.info("[Blender] Sent audio for id=%s", audio_id)
                return True
            except Exception as exc:
                logger.warning("[Blender] Failed to send audio for id=%s: %s", audio_id, exc)
                return False
        else:
            logger.debug("[Blender] Audio file not found for id=%s, continuing with motion", audio_id)
            return False

    def _close_locked(self) -> None:
        if self._sock:
            try:
                self._sock.close()
            except Exception:
                pass
        self._sock = None

    def connect(self) -> bool:
        """Connect to Blender lazily and close stale sockets after failures."""
        if not self.enabled:
            return False
        with self._lock:
            if self._sock:
                return True
            try:
                sock = socket.create_connection((self.host, self.port), timeout=2.0)
                sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                sock.settimeout(None)
                self._sock = sock
                logger.info("[Blender] Connected to %s:%s", self.host, self.port)
                return True
            except Exception as exc:
                self._sock = None
                logger.warning("[Blender] Connect failed: %s", exc)
                return False

    def close(self) -> None:
        with self._lock:
            self._close_locked()

    def _send_packet(self, payload: dict) -> bool:
        """Send a length-prefixed JSON packet."""
        if not self.enabled:
            return False
        if not self.connect():
            return False
        data = json.dumps(payload).encode("utf-8")
        header = struct.pack(">I", len(data))
        try:
            with self._lock:
                if not self._sock:
                    return False
                self._sock.sendall(header + data)
            return True
        except Exception as exc:
            logger.warning("[Blender] Send failed, closing socket: %s", exc)
            with self._lock:
                self._close_locked()
            return False

    def start_audio_stream(
        self,
        sample_rate: int,
        channels: int,
        sample_width: int,
        chunk_seconds: Optional[float] = None,
        total_frames: Optional[int] = None,
    ) -> None:
        """Send audio metadata before streaming individual chunks."""
        if not self.enabled:
            return
        seconds_per_chunk = chunk_seconds if chunk_seconds is not None else self.audio_chunk_seconds
        frames_per_chunk = int(sample_rate * seconds_per_chunk) if seconds_per_chunk else 0
        total_chunks = 0
        if total_frames and frames_per_chunk:
            total_chunks = math.ceil(total_frames / frames_per_chunk)
        meta = {
            "type": "audio_meta",
            "sample_rate": sample_rate,
            "channels": channels,
            "sample_width": sample_width,
            "frames": total_frames or 0,
            "duration": (total_frames / float(sample_rate)) if (total_frames and sample_rate) else 0,
            "chunk_seconds": seconds_per_chunk,
            "frames_per_chunk": frames_per_chunk,
            "total_chunks": total_chunks,
        }
        self._send_packet(meta)

    def send_audio_chunk_bytes(self, chunk: bytes, index: int, is_last: bool = False) -> None:
        if not self.enabled or not chunk:
            return
        payload = {
            "type": "audio_chunk",
            "index": index,
            "is_last": bool(is_last),
            "data_b64": base64.b64encode(chunk).decode("ascii"),
        }
        self._send_packet(payload)

    def send_audio_file(self, audio_path: str, chunk_seconds: Optional[float] = None) -> None:
        """Read a WAV file and send it in configured-sized chunks."""
        if not self.enabled:
            return
        if not os.path.exists(audio_path):
            logger.warning("[Blender] Audio file not found: %s", audio_path)
            return
        try:
            with wave.open(audio_path, "rb") as wf:
                sr = wf.getframerate()
                ch = wf.getnchannels()
                sw = wf.getsampwidth()
                total_frames = wf.getnframes()
                seconds_per_chunk = chunk_seconds if chunk_seconds is not None else self.audio_chunk_seconds
                frames_per_chunk = int(sr * seconds_per_chunk) if seconds_per_chunk else total_frames
                total_chunks = math.ceil(total_frames / frames_per_chunk) if frames_per_chunk else 1
                self.start_audio_stream(sr, ch, sw, seconds_per_chunk, total_frames)
                if self._viz is not None:
                    try:
                        duration = total_frames / float(sr) if sr else 0.0
                        self._viz.motion.audio_queue.available_audio[int(Path(audio_path).stem.split("_")[-1])] = {
                            "duration": duration,
                            "played": False,
                        }
                    except Exception:
                        pass
                for idx in range(total_chunks):
                    data = wf.readframes(frames_per_chunk)
                    if not data:
                        break
                    self.send_audio_chunk_bytes(data, idx, is_last=(idx == total_chunks - 1))
            logger.info("[Blender] Sent audio %s", audio_path)
        except Exception as exc:
            logger.error("[Blender] Failed to send audio %s: %s", audio_path, exc)

    def _build_mapping(self, columns: Sequence[str]) -> Dict[str, Dict[str, Any]]:
        """Build a mapping from CSV/DataFrame columns to bone and axis names.

        Handles column formats like:
        - 'Hips_Xrotation', 'Hips_Yrotation', 'Hips_Zrotation'
        - 'Hips_Xposition', 'Hips_Yposition', 'Hips_Zposition'
        """
        mapping: Dict[str, Dict[str, Any]] = {}
        for col in columns:
            # Match patterns like 'BoneName_Xrotation' or 'BoneName_Xposition'
            if "rotation" in col.lower():
                # Split to get bone name and axis
                # e.g., 'Hips_Xrotation' -> bone='Hips', axis='X'
                parts = col.rsplit("_", 1)
                if len(parts) != 2:
                    continue
                bone = parts[0]
                axis_part = parts[1]  # e.g., 'Xrotation'
                axis = axis_part[0].upper()  # 'X', 'Y', or 'Z'
                entry = mapping.setdefault(bone, {"rot_cols": {}, "rot_order": [], "pos_cols": {}})
                entry["rot_cols"][axis] = col
                if axis not in entry["rot_order"]:
                    entry["rot_order"].append(axis)
            elif "position" in col.lower():
                parts = col.rsplit("_", 1)
                if len(parts) != 2:
                    continue
                bone = parts[0]
                axis_part = parts[1]  # e.g., 'Xposition'
                axis = axis_part[0].upper()
                entry = mapping.setdefault(bone, {"rot_cols": {}, "rot_order": [], "pos_cols": {}})
                entry["pos_cols"][axis] = col
        return mapping

    def _build_fast_mapping(self, df: pd.DataFrame, mapping: Dict[str, Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Precompute column indices and matrix transforms for fast frame conversion.
        """
        columns = list(df.columns)
        col_to_idx = {col: i for i, col in enumerate(columns)}
        bones_cfg = (self.conjugate or {}).get("bones") or {}
        root_name = (self.conjugate or {}).get("root_name", "Hips")
        q_remap = (self.conjugate or {}).get("root_remap_Q")
        q_remap_t = mat3_transpose(q_remap) if q_remap else None

        fast_entries: List[Dict[str, Any]] = []
        for bone, meta in mapping.items():
            rot_order = meta.get("rot_order") or []
            rot_cols = meta.get("rot_cols") or {}
            rot_idx: List[int] = []
            missing = False
            for axis in rot_order:
                col = rot_cols.get(axis)
                if col not in col_to_idx:
                    missing = True
                    break
                rot_idx.append(col_to_idx[col])
            if missing or not rot_idx:
                continue

            pos_idx = None
            if bone == root_name:
                pos_cols = meta.get("pos_cols") or {}
                pos_idx = [
                    col_to_idx.get(pos_cols.get("X", ""), None),
                    col_to_idx.get(pos_cols.get("Y", ""), None),
                    col_to_idx.get(pos_cols.get("Z", ""), None),
                ]

            cfg = bones_cfg.get(bone)
            rest_matrix = cfg.get("rest_matrix") if cfg else None
            rest_inv = mat3_transpose(rest_matrix) if rest_matrix else None

            fast_entries.append(
                {
                    "bone": bone,
                    "rot_idx": rot_idx,
                    "order": "".join(rot_order),
                    "pos_idx": pos_idx,
                    "is_root": bone == root_name,
                    "rest": rest_matrix,
                    "rest_inv": rest_inv,
                    "use_conjugate": bool(rest_matrix),
                    "q_remap": q_remap if bone == root_name else None,
                    "q_remap_t": q_remap_t if bone == root_name else None,
                }
            )
        return fast_entries

    def _row_to_pose(self, row: pd.Series, mapping: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """Convert one DataFrame row to a pose dictionary for Blender.

        Uses the same transformation as test_blender_client_new.py:
        R_target = M_rest_inv @ R_bvh @ M_rest
        With root Q remap for correct orientation.
        """
        pose: Dict[str, Any] = {}
        root_name = (self.conjugate or {}).get("root_name", "Hips")

        for bone, meta in mapping.items():
            rot_cols = meta.get("rot_cols") or {}
            rot_order_list = meta.get("rot_order") or []

            if not rot_cols or not rot_order_list:
                continue

            # Build rotation order string and collect angles in that order
            order = "".join(rot_order_list)
            angles: List[float] = []
            missing = False

            for axis in rot_order_list:
                col = rot_cols.get(axis)
                if col is None or col not in row.index:
                    missing = True
                    break
                angles.append(float(row[col]))

            if missing or not angles:
                continue

            # Get bone config from conjugate if available
            entry_config = None
            if self.conjugate:
                entry_config = (self.conjugate.get("bones") or {}).get(bone)

            # Use config order or fallback to detected order
            order_in = order

            # print(self.mapping)
            # print(bone, angles, order_in,order)
            # assert 0

            entry: Dict[str, Any]

            # Apply conjugation transformation (same as test_blender_client_new.py)
            if entry_config and entry_config.get("rest_matrix"):
                try:
                    # 1) BVH euler -> rotation matrix
                    R_bvh = euler_to_mat3(angles, order_in)

                    # 2) Apply base transform: R_target = M_rest_inv @ R_bvh @ M_rest
                    M_rest = entry_config["rest_matrix"]
                    M_rest_inv = mat3_transpose(M_rest)
                    temp = mat3_mul(R_bvh, M_rest)
                    R_target = mat3_mul(M_rest_inv, temp)

                    # 3) Root Q remap (same as test_blender_client_new.py)
                    if bone == root_name:
                        Q = (self.conjugate or {}).get("root_remap_Q")
                        if Q:
                            R_target = mat3_mul(mat3_mul(Q, R_target), mat3_transpose(Q))

                    # 4) Convert to XYZ Euler
                    angles_out = mat3_to_euler_xyz_deg(R_target)
                    entry = {"angles": list(angles_out), "order": "XYZ"}
                except Exception as exc:
                    logger.debug("[Blender] Conjugation failed for %s: %s", bone, exc)
                    entry = {"angles": angles, "order": order_in}
            else:
                entry = {"angles": angles, "order": order_in}

            # Add position for root bone
            if bone == root_name:
                pos_cols = meta.get("pos_cols") or {}
                # print(pos_cols)
                # print(row.get(pos_cols.get("X", ""), 0.0))
                if pos_cols:
                    entry["location"] = [
                        float(row.get(pos_cols.get("X", ""), 0.0)),
                        float(row.get(pos_cols.get("Y", ""), 0.0)),
                        float(row.get(pos_cols.get("Z", ""), 0.0)),
                    ]
                    # entry["location"] = [
                    #     float(row.get(pos_cols.get("X", ""), 0.0)) + self.last_time_position[0],
                    #     float(row.get(pos_cols.get("Y", ""), 0.0)),
                    #     float(row.get(pos_cols.get("Z", ""), 0.0)) + self.last_time_position[1],
                    # ]
                    # if is_last_time:
                    #     self.last_time_position[0] = entry["location"][0]
                    #     self.last_time_position[1] = entry["location"][2]

            pose[bone] = entry

        return pose

    def _rows_to_poses_fast(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """
        Convert a DataFrame to pose dictionaries using precomputed indices.
        """
        if not self._fast_mapping:
            return [
                self._row_to_pose(row, self.mapping)  # type: ignore[arg-type]
                for _, row in df.iterrows()
            ]

        data = df.to_numpy()
        frames: List[Dict[str, Any]] = []

        mat_mul = mat3_mul
        euler_to_mat = euler_to_mat3
        mat_to_euler = mat3_to_euler_xyz_deg

        for row_vals in data:
            pose: Dict[str, Any] = {}
            for item in self._fast_mapping:
                angles = [float(row_vals[idx]) for idx in item["rot_idx"]]
                entry_order = item["order"]
                entry_angles = angles

                if item["use_conjugate"]:
                    R_bvh = euler_to_mat(angles, entry_order)
                    temp = mat_mul(R_bvh, item["rest"])
                    R_target = mat_mul(item["rest_inv"], temp)
                    if item["is_root"] and item["q_remap"]:
                        R_target = mat_mul(mat_mul(item["q_remap"], R_target), item["q_remap_t"])
                    entry_angles = mat_to_euler(R_target)
                    entry_order = "XYZ"

                entry: Dict[str, Any] = {"angles": list(entry_angles), "order": entry_order}

                if item["is_root"] and item["pos_idx"]:
                    px, py, pz = item["pos_idx"]
                    entry["location"] = [
                        float(row_vals[px]) if px is not None else 0.0,
                        float(row_vals[py]) if py is not None else 0.0,
                        float(row_vals[pz]) if pz is not None else 0.0,
                    ]

                pose[item["bone"]] = entry

            frames.append(pose)

        return frames

    def _queue_frames(self, frames: List[Dict[str, Any]], audio_id: Optional[int]) -> None:
        """Add frames to the playback queue with associated audio_id."""
        if not frames:
            return

        with self._queue_lock:

            if self._pending_interrupt and audio_id is not None:
                if self._interrupt_after_audio_id is None or audio_id > self._interrupt_after_audio_id:

                    logger.info(
                        "[Blender] === INTERRUPT SWITCH === New motion arrived (audio_id=%s > interrupt_after=%s)",
                        audio_id, self._interrupt_after_audio_id
                    )
                    logger.info("[Blender] Clearing %d old frames, starting fresh", len(self._motion_queue))


                    self._motion_queue.clear()
                    self._chunk_ranges.clear()
                    self._current_frame = 0
                    self._current_chunk_audio_id = None
                    self._sent_audio_ids.clear()
                    if self._viz is not None:
                        try:
                            self._viz.motion.chunk_ranges.clear()
                            self._viz.motion.unplayed_chunk_start_ids.clear()
                            self._viz.motion.current_frame = 0
                        except Exception:
                            pass


                    self._pending_interrupt = False
                    self._interrupt_after_audio_id = None

            start_frame = len(self._motion_queue)
            self._motion_queue.extend(frames)
            end_frame = len(self._motion_queue) - 1



            self._chunk_ranges.append((audio_id, start_frame, end_frame))
            if self._viz is not None:
                duration_sec = len(frames) * self.frame_interval if self.frame_interval else 0.0
                end_id = (audio_id or 0) + duration_sec
                self._viz.motion.chunk_ranges.append((audio_id, start_frame, end_frame, end_id))
                if audio_id is not None:
                    try:
                        self._viz.motion.unplayed_chunk_start_ids.append(audio_id)
                    except Exception:
                        pass

            logger.info("[Blender] Queued %d frames (audio_id=%s), total queue: %d frames",
                       len(frames), audio_id, len(self._motion_queue))


    def stream_motion(self, motion_df: Any, audio_id: Optional[int], audio_dir: Optional[str] = None) -> None:
        """Stream motion data to Blender using a queue and synchronized audio.

        Similar to g1_csv_sim_mujoco.py's load_csv_to_queue:
        - Converts DataFrame to pose frames
        - Queues frames for playback
        - Audio is sent when playback reaches the chunk (non-blocking)

        Args:
            motion_df: DataFrame or dict-like with motion data
            audio_id: ID of the audio file to send (output_{audio_id}.wav)
            audio_dir: Directory containing audio files (updates self.audio_dir if provided)
        """
        if not self.enabled or motion_df is None:
            return

        # Update audio dir if provided
        if audio_dir is not None:
            self.audio_dir = audio_dir

        logger.info("[Blender] Revieving %d frames (audio_id=%s)", len(motion_df), audio_id)

        try:
            df = motion_df if isinstance(motion_df, pd.DataFrame) else pd.DataFrame(motion_df)
        except Exception as exc:
            logger.error("[Blender] Failed to prepare motion dataframe: %s", exc)
            return

        cols_sig = tuple(df.columns)
        if (self.mapping is None) or (self._mapping_cols_sig != cols_sig):
            self.mapping = self._build_mapping(df.columns)
            self._fast_mapping = None
            self._mapping_cols_sig = cols_sig
        if not self.mapping:
            logger.warning("[Blender] No valid bone mapping found in columns: %s", list(df.columns)[:10])
            return
        if self._fast_mapping is None:
            self._fast_mapping = self._build_fast_mapping(df, self.mapping)

        frames = self._rows_to_poses_fast(df)

        if not frames:
            logger.warning("[Blender] No valid frames generated from motion data")
            return


        frames_per_audio = int(round(1.0 / self.frame_interval)) if self.frame_interval > 0 else len(frames)
        if audio_id is not None and frames_per_audio > 0 and len(frames) > frames_per_audio:
            for idx in range(0, len(frames), frames_per_audio):
                sub_frames = frames[idx : idx + frames_per_audio]
                sub_audio_id = audio_id + (idx // frames_per_audio)
                logger.info(
                    "[Blender] Adding %d frames to queue (audio_id=%s, slice=%d-%d)",
                    len(sub_frames),
                    sub_audio_id,
                    idx,
                    idx + len(sub_frames) - 1,
                )
                self._queue_frames(sub_frames, sub_audio_id)
        else:
            logger.info("[Blender] Adding %d frames to queue (audio_id=%s)", len(frames), audio_id)
            self._queue_frames(frames, audio_id)

        # Ensure playback thread is running
        self.start_playback()

    def clear_queue(self) -> None:
        """Clear queued motion and audio state."""
        with self._queue_lock:
            self._motion_queue.clear()
            self._chunk_ranges.clear()
            self._current_frame = 0
            self._current_chunk_audio_id = None
            self._sent_audio_ids.clear()
            self._pending_interrupt = False
            self._interrupt_after_audio_id = None
            if self._viz is not None:
                try:
                    self._viz.motion.chunk_ranges.clear()
                    self._viz.motion.unplayed_chunk_start_ids.clear()
                    self._viz.motion.current_frame = 0
                except Exception:
                    pass
        logger.info("[Blender] Motion queue cleared")

    def set_pending_interrupt(self, current_audio_id: int) -> None:
        """Keep the current chunk playing until a newer motion chunk arrives."""
        with self._queue_lock:
            self._pending_interrupt = True
            self._interrupt_after_audio_id = current_audio_id
            remaining_frames = len(self._motion_queue) - self._current_frame
        logger.info(
            "[Blender] Pending interrupt set: waiting for audio_id > %s, continuing %d frames",
            current_audio_id, remaining_frames
        )

    def is_pending_interrupt(self) -> bool:
        """Return whether playback is waiting for a post-interruption chunk."""
        with self._queue_lock:
            return self._pending_interrupt
