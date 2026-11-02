"""
interruption_core.py - interruption state and queue coordination.

This module contains the interruption state and queue coordination classes.
"""

import logging
import os
import queue
import threading
import time
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional

import requests



_local_session = requests.Session()
_local_session.trust_env = False



class InterruptionType(Enum):
    NONE = "none"
    USER_INTERRUPT = "user_interrupt"
    AGENT_SELF_INTERRUPT = "agent_self"
    MANUAL_INTERRUPT = "manual"


@dataclass
class InterruptionState:
    """Interruption state data."""
    type: InterruptionType
    timestamp: float
    source_turn: int
    target_turn: int
    audio_position_ms: int


class InterruptionManager:
    """
    Coordinate interruption state and recovery.

    Responsibilities:
    1. Detect interruption conditions.
    2. Coordinate component responses.
    3. Manage state transitions.
    4. Send interruption signals to the server.
    5. Create a local marker for MuJoCo and playback.
    """

    def __init__(
        self,
        server_url: str,
        uid: str,
        logger: logging.Logger,
        motion_dir: Optional[str] = None,
    ):
        self.server_url = server_url
        self.uid = uid
        self.logger = logger
        self.motion_dir = motion_dir

        self._lock = threading.RLock()
        self._interruption_active = False
        self._current_interruption: Optional[InterruptionState] = None
        self._interruption_sequence = 0
        self._current_audio_id: Optional[int] = None


        self._on_interrupt_callbacks: List[callable] = []
        self._on_resume_callbacks: List[callable] = []


        self._waiting_for_new_audio = False
        self._waiting_for_new_motion = False


        self.stats = {
            "user_interrupts": 0,
            "agent_self_interrupts": 0,
            "manual_interrupts": 0,
            "total_interrupted_audio_ms": 0,
        }

    def set_motion_dir(self, motion_dir: str) -> None:
        """Set the motion directory used for interruption markers."""
        self.motion_dir = motion_dir

    def register_on_interrupt(self, callback: callable) -> None:
        """Register an interruption callback."""
        with self._lock:
            self._on_interrupt_callbacks.append(callback)

    def register_on_resume(self, callback: callable) -> None:
        """Register a recovery callback."""
        with self._lock:
            self._on_resume_callbacks.append(callback)

    def is_interrupted(self) -> bool:
        """Return whether interruption is active."""
        with self._lock:
            return self._interruption_active

    def get_current_interruption(self) -> Optional[InterruptionState]:
        """Return the current interruption state."""
        with self._lock:
            return self._current_interruption

    def trigger_interruption(
        self,
        interrupt_type: InterruptionType,
        source_turn: int,
        target_turn: int,
        audio_position_ms: int = 0,
        current_audio_id: Optional[int] = None,
    ) -> bool:
        """
        Trigger an interruption.

        Args:
            interrupt_type: interruption type
            source_turn: Trigger an interruption.
            target_turn: turn being interrupted
            audio_position_ms: audio position when interrupted
            current_audio_id: current audio ID used to filter post-interruption motion

        Returns:
            Trigger an interruption.
        """
        with self._lock:
            if self._interruption_active:
                self.logger.info(
                    "Interruption already active, ignoring new interruption request"
                )
                return False

            self._interruption_sequence += 1
            self._interruption_active = True
            self._current_interruption = InterruptionState(
                type=interrupt_type,
                timestamp=time.time(),
                source_turn=source_turn,
                target_turn=target_turn,
                audio_position_ms=audio_position_ms,
            )
            self._current_audio_id = current_audio_id


            if interrupt_type == InterruptionType.USER_INTERRUPT:
                self.stats["user_interrupts"] += 1
            elif interrupt_type == InterruptionType.AGENT_SELF_INTERRUPT:
                self.stats["agent_self_interrupts"] += 1
            elif interrupt_type == InterruptionType.MANUAL_INTERRUPT:
                self.stats["manual_interrupts"] += 1
            self.stats["total_interrupted_audio_ms"] += audio_position_ms

            self._waiting_for_new_audio = True
            self._waiting_for_new_motion = True

        self.logger.info(
            "Interruption triggered: type=%s, source_turn=%s, target_turn=%s, audio_pos=%sms, audio_id=%s",
            interrupt_type.value,
            source_turn,
            target_turn,
            audio_position_ms,
            current_audio_id,
        )


        self._send_interrupt_to_server(interrupt_type, source_turn, target_turn, audio_position_ms, current_audio_id)


        self._create_local_interrupt_file()


        for callback in self._on_interrupt_callbacks:
            try:
                callback(self._current_interruption)
            except Exception as e:
                self.logger.error("Interrupt callback error: %s", e)

        return True

    def clear_interruption(self) -> None:
        """Clear interruption state and resume normal playback."""
        with self._lock:
            if not self._interruption_active:
                return

            old_state = self._current_interruption
            self._interruption_active = False
            self._current_interruption = None
            self._waiting_for_new_audio = False
            self._waiting_for_new_motion = False

            self._current_audio_id = None

        self.logger.info("Interruption cleared")


        for callback in self._on_resume_callbacks:
            try:
                callback(old_state)
            except Exception as e:
                self.logger.error("Resume callback error: %s", e)

    def notify_new_audio_received(self) -> None:
        """Notify the manager that post-interruption audio arrived."""
        with self._lock:
            if self._waiting_for_new_audio:
                self._waiting_for_new_audio = False
                self.logger.info("New audio received after interruption")


            if not self._waiting_for_new_audio and not self._waiting_for_new_motion:
                self.clear_interruption()

    def notify_new_motion_received(self) -> None:
        """Notify the manager that post-interruption motion arrived."""
        with self._lock:
            if self._waiting_for_new_motion:
                self._waiting_for_new_motion = False
                self.logger.info("New motion received after interruption")


            if not self._waiting_for_new_audio and not self._waiting_for_new_motion:
                self.clear_interruption()

    def is_waiting_for_new_audio(self) -> bool:
        """Return whether new audio is pending."""
        with self._lock:
            return self._waiting_for_new_audio

    def is_waiting_for_new_motion(self) -> bool:
        """Return whether new motion is pending."""
        with self._lock:
            return self._waiting_for_new_motion

    def get_interrupt_after_audio_id(self) -> Optional[int]:
        """Return the interruption audio ID; only newer motion is accepted."""
        with self._lock:
            return self._current_audio_id

    def _send_interrupt_to_server(
        self,
        interrupt_type: InterruptionType,
        source_turn: int,
        target_turn: int,
        audio_position_ms: int,
        current_audio_id: Optional[int] = None,
    ) -> None:
        """Send the interruption signal in a background thread."""
        interruption_data = {
            "type": interrupt_type.value,
            "source_turn": source_turn,
            "target_turn": target_turn,
            "audio_position_ms": audio_position_ms,
            "timestamp_ms": int(time.time() * 1000),
            "sequence": self._interruption_sequence,
        }

        if current_audio_id is not None:
            interruption_data["current_audio_id"] = current_audio_id

        payload = {
            "messages": [
                {
                    "role": "system",
                    "content": [
                        {
                            "type": "interruption",
                            "interruption": interruption_data,
                        }
                    ],
                }
            ]
        }

        def _send_async():
            try:
                res = _local_session.post(
                    f"{self.server_url}/api/v1/stream",
                    headers={"uid": self.uid},
                    json=payload,
                    timeout=5.0,
                )
                self.logger.debug(
                    "Sent interruption signal to server: status=%s",
                    res.status_code,
                )
            except Exception as e:
                self.logger.error("Failed to send interruption signal: %s", e)


        thread = threading.Thread(target=_send_async, daemon=True)
        thread.start()

    def _create_local_interrupt_file(self) -> None:
        """
        Create a local marker that signals MuJoCo and playback.

        MuJoCo g1_csv_sim_mujoco.py  motion_dir/interrupt.temp
         set_pending_interrupt()
        """
        if not self.motion_dir:
            self.logger.warning("Motion dir not set, cannot create interrupt file")
            return

        interrupt_file = os.path.join(self.motion_dir, "interrupt.temp")
        try:
            with open(interrupt_file, 'w') as f:
                f.write(f"interrupt_sequence={self._interruption_sequence}\n")
                f.write(f"timestamp={time.time()}\n")
                if self._current_audio_id is not None:
                    f.write(f"interrupt_after_audio_id={self._current_audio_id}\n")
            self.logger.info("Created interrupt file: %s (audio_id=%s)", interrupt_file, self._current_audio_id)
        except Exception as e:
            self.logger.error("Failed to create interrupt file: %s", e)


class PlaybackController:
    """
    .

    Responsibilities:
    1.
    2. ()
    3.
    4.
    """

    def __init__(
        self,
        interruption_manager: InterruptionManager,
        logger: logging.Logger,
    ):
        self.interruption_manager = interruption_manager
        self.logger = logger

        self._lock = threading.Lock()


        self.audio_queue: queue.Queue[Optional[bytes]] = queue.Queue()
        self._audio_playing = False
        self._audio_position_ms = 0
        self._audio_start_time = 0.0
        self._total_audio_bytes = 0


        self.motion_queue: queue.Queue[Optional[Dict]] = queue.Queue()
        self._motion_playing = False
        self._current_motion_id: Optional[int] = None


        self._current_turn = 1


        self.audio_stream = None


        interruption_manager.register_on_interrupt(self._on_interrupt)

    def _on_interrupt(self, state: InterruptionState) -> None:
        """Handle an active interruption."""
        self.logger.info("PlaybackController: handling interruption")
        self.clear_queues()

    def clear_queues(self) -> None:
        """Clear all playback queues."""
        with self._lock:

            while not self.audio_queue.empty():
                try:
                    self.audio_queue.get_nowait()
                except queue.Empty:
                    break


            while not self.motion_queue.empty():
                try:
                    self.motion_queue.get_nowait()
                except queue.Empty:
                    break


            self._audio_playing = False
            self._motion_playing = False
            self._audio_position_ms = 0
            self._total_audio_bytes = 0

            self.logger.info("Playback queues cleared")

    def enqueue_audio(self, audio_bytes: bytes) -> None:
        """Queue audio data."""
        if self.interruption_manager.is_interrupted():
            if self.interruption_manager.is_waiting_for_new_audio():
                self.interruption_manager.notify_new_audio_received()

        self.audio_queue.put(audio_bytes)

    def enqueue_motion(self, motion_data: Dict) -> None:
        """Queue motion data."""
        if self.interruption_manager.is_interrupted():
            if self.interruption_manager.is_waiting_for_new_motion():
                self.interruption_manager.notify_new_motion_received()

        self.motion_queue.put(motion_data)

    def get_audio_position_ms(self) -> int:
        """Return the current audio position in milliseconds."""
        with self._lock:
            if not self._audio_playing:
                return self._audio_position_ms

            elapsed = time.time() - self._audio_start_time
            return int(elapsed * 1000)

    def mark_audio_started(self) -> None:
        """Mark audio playback as started."""
        with self._lock:
            self._audio_playing = True
            self._audio_start_time = time.time()

    def mark_audio_stopped(self) -> None:
        """Mark audio playback as stopped."""
        with self._lock:
            if self._audio_playing:
                self._audio_position_ms = self.get_audio_position_ms()
            self._audio_playing = False

    def add_audio_bytes(self, byte_count: int) -> None:
        """Track received audio bytes."""
        with self._lock:
            self._total_audio_bytes += byte_count

    def set_current_turn(self, turn: int) -> None:
        """Set the current turn."""
        with self._lock:
            self._current_turn = turn

    def get_current_turn(self) -> int:
        """Return the current turn."""
        with self._lock:
            return self._current_turn


class MotionSyncController:
    """
    Synchronize remote motion generation.

    Responsibilities:
    1. Manage the motion WebSocket connection.
    2. Synchronize motion during interruption.
    3. Discard stale pre-interruption motion.
    """

    def __init__(
        self,
        interruption_manager: InterruptionManager,
        playback_controller: PlaybackController,
        logger: logging.Logger,
    ):
        self.interruption_manager = interruption_manager
        self.playback_controller = playback_controller
        self.logger = logger

        self._lock = threading.Lock()


        self._interruption_sequence = 0
        self._last_valid_motion_id: Optional[int] = None


        interruption_manager.register_on_interrupt(self._on_interrupt)

    def _on_interrupt(self, state: InterruptionState) -> None:
        """Handle an active interruption."""
        with self._lock:
            self._interruption_sequence += 1
            self.logger.info(
                "MotionSyncController: interruption sequence updated to %s",
                self._interruption_sequence,
            )

    def should_accept_motion(self, motion_id: int, motion_sequence: Optional[int] = None) -> bool:
        """
        Check whether motion data should be accepted.

        Discard stale motion received after interruption.
        Accept only motion newer than the interruption audio ID.

        If the response has not been interrupted, retain the interrupt-after audio ID
        so playback can stop at the correct boundary.
        """
        with self._lock:

            interrupt_after_id = self.interruption_manager.get_interrupt_after_audio_id()


            if interrupt_after_id is not None:
                if motion_id <= interrupt_after_id:
                    self.logger.info(
                        "Rejecting old motion: id=%s <= interrupt_after=%s (is_interrupted=%s)",
                        motion_id,
                        interrupt_after_id,
                        self.interruption_manager.is_interrupted(),
                    )
                    return False


            self._last_valid_motion_id = motion_id
            return True
