import time
import os
import csv
import threading
import numpy as np
import soundfile as sf
try:
    import pygame
except ImportError:  # Optional; playback does not require the Gantt visualizer.
    pygame = None

kPi = 3.141592654
kPi_2 = 1.57079632
FPS = 30.0

zero_point_ns = None
local_session_dir = None
client_session_start_time = None
is_time_log = False
_DEFAULT_AUDIO_DIR = os.path.abspath(
    os.getenv("PROACT_AUDIO_DIR", os.path.join(os.path.dirname(__file__), "runtime", "assistant_audio"))
)

def time_logging(event: str, audio_id: str):
    """
    Logs an event with a timestamp relative to the synchronized zero point.
    """
    try:
        if zero_point_ns is None:
            print(f"Warning: zero_point_ns is not set. Log timestamp will be incorrect. in {event}, {audio_id}")
            return
        if local_session_dir is None:
            print(f"Warning: local_session_dir is not set. Cannot write log. in {event}, {audio_id}")
            return
        current_time_ns = time.time_ns()
        relative_timestamp_ns = current_time_ns - zero_point_ns
        log_file_path = os.path.join(local_session_dir, "log_mujoco.csv")
        with open(log_file_path, 'a', newline='') as f:
            writer = csv.writer(f)
            # if write_header:
            #     writer.writerow(["timestamp_ms", "event", "audio_id"])
            writer.writerow([relative_timestamp_ns // 1_000_000, event, audio_id])

    except Exception as e:
        print(f"Error in mujoco time_logging: {e}")

class Clock:
    """A simple clock to track time relative to a zero point."""
    def __init__(self):
        self._zero_point_ns = None

    def set_to(self, time_sec: float):
        """
        Realigns the clock's zero point so that the current time is `time_sec`.
        """
        self._zero_point_ns = time.time_ns() - int(time_sec * 1_000_000_000)
        current_time = self.get()
        if current_time is not None:
            print(f"Clock realigned. Current time is now {current_time:.2f}s.")

    def get(self) -> float | None:
        """Gets the current time in seconds relative to the zero point. Returns None if not set."""
        if self._zero_point_ns is None:
            return None
        return (time.time_ns() - self._zero_point_ns) / 1_000_000_000.0

class AudioQueue:
    """
    Scans for, manages, and plays available audio files in a dedicated thread
    using a clock-based mechanism for robust timing.
    """
    def __init__(self, audio_interface, audio_dir=None, on_play_start=None, on_play_end=None):
        self.audio_interface = audio_interface
        self.audio_dir = audio_dir or _DEFAULT_AUDIO_DIR
        self.available_audio = {} # {audio_id: {'path': str, 'played': bool}}
        self.audio_cache = {}
        self.attempted_ids = set()
        self.chunk_endpoint_id = None
        self.stop_event = threading.Event()
        self.clock = Clock()
        self.last_audio_start_play_time = None
        self.on_play_start = on_play_start
        self.on_play_end = on_play_end
        self.play_end_timer = None


        self.pending_interrupt = False
        self.interrupt_after_audio_id = None

        self.playback_thread = threading.Thread(target=self._playback_loop, daemon=True)
        self.playback_thread.start()

        self.scan_thread = threading.Thread(target=self._scan_loop, daemon=True)
        self.scan_thread.start()
        print(f"AudioQueue initialized to scan '{audio_dir}'.")

    def _playback_loop(self):
        """A thread that continuously attempts to play audio based on the clock's current ID."""
        print("Audio playback thread started.")
        while not self.stop_event.is_set():
            current_time = self.clock.get()
            if current_time is None:
                time.sleep(0.1) # Clock not set, wait
                continue

            current_id_to_check = int(current_time)

            if current_id_to_check not in self.attempted_ids:
                if self.chunk_endpoint_id is not None and current_id_to_check >= self.chunk_endpoint_id:
                    # Reached the end of the current chunk, do not attempt further playback
                    time.sleep(0.01)
                    time_logging("AUDIO_PLAYBACK_SKIPPED_PAST_ENDPOINT", f"{current_id_to_check}>={self.chunk_endpoint_id}")
                    continue
                self.attempted_ids.add(current_id_to_check)
                time_logging("AUDIO_PLAY_ATTEMPT", str(current_id_to_check))



                if self.pending_interrupt:
                    # print(f"[AudioQueue] Skipping audio id={current_id_to_check} during pending interrupt (waiting for new motion)")
                    time_logging("AUDIO_PLAYBACK_SKIPPED_PENDING_INTERRUPT", str(current_id_to_check))

                    self.mark_as_played(current_id_to_check)
                    continue

                if self.is_available(current_id_to_check) and not self.is_played(current_id_to_check):
                    audio_path = self.available_audio[current_id_to_check]['path']
                    duration = self.available_audio[current_id_to_check].get('duration', 1.0)
                    try:
                        time_logging("AUDIO_PLAYBACK_START", str(current_id_to_check))
                        print(f"Playing audio: {os.path.basename(audio_path)}")

                        if self.play_end_timer:
                            self.play_end_timer.cancel()
                            self.play_end_timer = None

                        if self.on_play_start:
                            self.on_play_start()

                        self.last_audio_start_play_time = time.time_ns()
                        self.audio_interface.play(audio_path)

                        if self.on_play_end:
                            def _timer_end():
                                self.play_end_timer = None
                                self.on_play_end()
                            self.play_end_timer = threading.Timer(duration, _timer_end)
                            self.play_end_timer.start()

                        self.mark_as_played(current_id_to_check)
                    except Exception as e:
                        print(f"Failed to play audio for id {current_id_to_check}: {e}")
                        if self.on_play_end: # Ensure we reset state even on failure
                            self.on_play_end()
                else:
                    # Only call on_play_end if no audio is scheduled to play soon
                    # and no previous segment's timer is still running.
                    if self.play_end_timer is None and self.on_play_end:
                        self.on_play_end()  # No audio to play, ensure state reset

            time.sleep(0.01) # Prevent busy-waiting

        print("Audio playback thread stopped.")

    def _scan_loop(self):
        """Continuously scans the audio directory for new .wav files."""
        print("Audio file scanner thread started.")
        while not self.stop_event.is_set():
            try:
                if os.path.exists(self.audio_dir):
                    for filename in os.listdir(self.audio_dir):
                        if filename.startswith("output_") and filename.endswith(".wav"):
                            try:
                                audio_id = int(filename.split("_")[1].split(".")[0])
                                if audio_id not in self.available_audio:
                                    file_path = os.path.join(self.audio_dir, filename)
                                    try:

                                        with sf.SoundFile(file_path) as f:
                                            duration = len(f) / f.samplerate
                                        self.available_audio[audio_id] = {
                                            'path': file_path,
                                            'played': False,
                                            'duration': duration
                                        }
                                    except Exception as e:
                                        print(f"Error reading soundfile {file_path}: {e}")
                            except (IndexError, ValueError):
                                continue
            except Exception as e:
                print(f"Error in AudioQueue scan loop: {e}")
            time.sleep(0.2)
        print("Audio file scanner thread stopped.")

    def is_available(self, audio_id: int) -> bool:
        """Check if a specific audio ID is available."""
        return audio_id in self.available_audio

    def is_played(self, audio_id: int) -> bool:
        """Check if a specific audio ID has been played."""
        return self.available_audio.get(audio_id, {}).get('played', False)

    def mark_as_played(self, audio_id: int):
        """Mark a specific audio ID as played."""
        if audio_id in self.available_audio:
            self.available_audio[audio_id]['played'] = True

    def get_earliest_unplayed_audio_id(self) -> int | None:
        """Gets the smallest audio_id that is available, not played, and not in the past relative to the clock."""
        current_time = self.clock.get()
        if current_time is None:
            return None


        current_clock_id = int(current_time)
        unplayed_ids = [aid for aid, data in self.available_audio.items()
                        if not data['played'] and aid >= current_clock_id]

        if not unplayed_ids:
            return None

        result_id = min(unplayed_ids)
        print(f"DEBUG: [SkipCheck] Clock: {current_time:.2f}, Returning ID: {result_id}")

        return result_id

    def realign_clock_to_id(self, time_id: int):
        """Realigns the internal clock so the current time corresponds to the given ID."""
        if time_id is not None:
            self.clock.set_to(time_id)

    def stop(self):
        """Stops the audio scanning thread."""
        self.stop_event.set()
        self.playback_thread.join()
        self.scan_thread.join()
        print("AudioQueue stopped.")

class MotionQueue:
    def __init__(self, audio_interface, audio_dir=None, on_play_start=None, on_play_end=None):
        self.motion_data = []  # List of motion frames
        self.current_frame = 0
        self.chunk_ranges = []  # List of (audio_id, start_frame, end_frame, end_id) tuples
        self.unplayed_chunk_start_ids = [] # List of start_ids for chunks not yet played
        self.current_chunk_start_id = None
        self.audio_queue = AudioQueue(audio_interface, audio_dir, on_play_start, on_play_end)
        self.stop_event = threading.Event()

        # Smoothing state
        self.last_qpos = None
        self.was_smoothed = False
        self.smoothing_accel_boost = 0.0


        self.pending_interrupt = False
        self.interrupt_after_audio_id = None
        self.interrupt_sequence = 0

    def stop(self):
        """Stops the audio monitoring thread."""
        self.stop_event.set()
        self.audio_queue.stop()
        print("MotionQueue stopped.")

    def append(self, motion_frames, start_id=None):
        """
        Append motion frames to the queue.
        motion_frames: list of frames, each frame is a list of joint values
        start_id: identifier for the audio/motion sequence
        """

        if self.pending_interrupt and start_id is not None:
            if self.interrupt_after_audio_id is None or start_id > self.interrupt_after_audio_id:

                print(f"=== INTERRUPT SWITCH === New motion arrived (audio_id={start_id} > interrupt_after={self.interrupt_after_audio_id})")
                print(f"Clearing {len(self.motion_data)} old frames, starting fresh with new motion")
                time_logging(f"INTERRUPT_SWITCH_TO_NEW_{start_id}", str(self.interrupt_after_audio_id or 0))


                self.motion_data.clear()
                self.current_frame = 0
                self.chunk_ranges.clear()
                self.unplayed_chunk_start_ids.clear()
                self.current_chunk_start_id = None
                self.was_smoothed = False
                self.smoothing_accel_boost = 0.0


                self.pending_interrupt = False
                self.interrupt_after_audio_id = None

                self.audio_queue.pending_interrupt = False
                self.audio_queue.interrupt_after_audio_id = None

                self.audio_queue.clock.set_to(start_id)
                self.audio_queue.attempted_ids.clear()
                self.audio_queue.chunk_endpoint_id = None
                print(f"=== CLOCK JUMPED TO {start_id} === Skipping wait for old audio segments")

        start_frame = len(self.motion_data)
        self.motion_data.extend(motion_frames)
        end_frame = len(self.motion_data) - 1

        if start_id is not None:
            end_id = start_id + (end_frame - start_frame) / FPS
            self.chunk_ranges.append((start_id, start_frame, end_frame, end_id))
            self.unplayed_chunk_start_ids.append(start_id)
            print(f"Appended chunk start_id {start_id}: {len(motion_frames)} frames ({start_frame}-{end_frame}). end_id: {end_id}. Current {self.current_frame}/{len(self.motion_data)}")
            time_logging(f"CHUNK_APPENDED_{start_id}_{start_frame}_{end_frame}", str(start_id))
        else:
            print(f"Appended {len(motion_frames)} frames to queue. Current at {self.current_frame}/{len(self.motion_data)}")

    def get_motion(self, current_qpos=None, dt=None):
        """
        Get the current motion frame and advance to next frame.
        If current_qpos and dt are provided, apply boundary smoothing.
        Returns None if no motion data available.
        """
        if not self.motion_data or self.current_frame >= len(self.motion_data):
            # print("No motion data available or end of motion reached.")
            return None

        # Skipping logic: if buffer is large and new audio is available, skip to it
        buffer_size = len(self.motion_data) - self.current_frame
        buffer_threshold = 5
        audio_length_max_ns = 1_000_000_000  # 1 second
        is_playing = False
        if self.audio_queue.last_audio_start_play_time is not None and time.time_ns() < self.audio_queue.last_audio_start_play_time + audio_length_max_ns:
            is_playing = True
            # print('DEBUG: Audio playing, no skip')
        if buffer_size > buffer_threshold and not is_playing and not self.pending_interrupt:
            next_audio_id = self.audio_queue.get_earliest_unplayed_audio_id()
            if next_audio_id is not None:
                # Find the motion chunk for this audio_id
                target_frame = None
                for start_id, start_frame, end_frame, end_id in self.chunk_ranges:
                    if start_id <= next_audio_id <= end_id - 2/FPS:
                        # Check if skipping would actually move us forward
                        skip_offset = 1
                        if start_frame > self.current_frame + skip_offset:
                            # target_frame = start_frame - skip_offset + (next_audio_id - start_id)*FPS
                            target_frame = start_frame - skip_offset
                            time_logging(f'SKIP_AHEAD_FROM_{self.current_frame}_TO_{target_frame}', 0)
                            break

                if target_frame is not None:
                    print(f"--- SKIP --- Large buffer ({buffer_size} frames) and new audio ({next_audio_id}) found. Skipping from frame {self.current_frame} to {target_frame}.")
                    self.current_frame = target_frame

        # Check if we're entering a new audio segment
        current_chunk_start_id, current_chunk_end_id = self._get_chunk_info_for_frame(self.current_frame)
        if current_chunk_start_id != self.current_chunk_start_id:
            self.current_chunk_start_id = current_chunk_start_id
            if current_chunk_start_id is not None:
                if current_chunk_start_id in self.unplayed_chunk_start_ids:
                    self.unplayed_chunk_start_ids.remove(current_chunk_start_id)
                print(f"                           Motion Entering new id {current_chunk_start_id} at frame {self.current_frame}, {self.current_frame}/{len(self.motion_data)}")
                time_logging("MOTION_CHUNK_START", str(current_chunk_start_id))
                # Signal AudioQueue to handle this new chunk
                self.audio_queue.realign_clock_to_id(current_chunk_start_id)
                self.audio_queue.chunk_endpoint_id = current_chunk_end_id

        target_qpos = np.array(self.motion_data[self.current_frame])

        # --- Internal Smoothing Logic ---
        if current_qpos is not None and dt is not None and self.last_qpos is not None:
            relative_idx, frames_remaining = self.get_relative_frame_info()

            # Boundary check
            is_boundary = False
            if relative_idx is not None:
                if relative_idx < 5 or frames_remaining < 3:
                    is_boundary = True

            # Apply smoothing if at boundary OR if previous frame was smoothed
            if is_boundary or self.was_smoothed:
                current_vel = (current_qpos - self.last_qpos) / dt
                ideal_target_vel = (target_qpos - current_qpos) / dt
                vel_delta = ideal_target_vel - current_vel

                # Base threshold: 1.5 rad/s^2
                # Boost increases by 0.1 for each consecutive smoothed frame
                current_max_accel = 1.5 + self.smoothing_accel_boost

                if np.any(np.abs(vel_delta) > current_max_accel):
                    clipped_accel = np.clip(vel_delta, -current_max_accel, current_max_accel)
                    smoothed_target_vel = current_vel + clipped_accel
                    target_qpos = current_qpos + smoothed_target_vel * dt
                    # if self.was_smoothed and not is_boundary:
                        # print(f'smoothing due to was smoothed (threshold: {current_max_accel:.2f})')
                    self.was_smoothed = True
                    self.smoothing_accel_boost += 0.1
                else:
                    self.was_smoothed = False
                    self.smoothing_accel_boost = 0.0

            self.last_qpos = current_qpos

        self.current_frame += 1
        return target_qpos.tolist() if isinstance(target_qpos, np.ndarray) else target_qpos

    def _get_chunk_info_for_frame(self, frame_idx):
        """Get the start_id and end_id for a given frame index"""
        for start_id, start_frame, end_frame, end_id in self.chunk_ranges:
            if start_frame <= frame_idx <= end_frame:
                return start_id, end_id
        return None, None

    def get_relative_frame_info(self):
        """
        Get relative frame index and remaining frames in the current chunk.
        """
        for start_id, start_frame, end_frame, end_id in self.chunk_ranges:
            if start_frame <= self.current_frame <= end_frame:
                relative_idx = self.current_frame - start_frame
                frames_remaining = end_frame - self.current_frame
                return relative_idx, frames_remaining
        return None, None

    def clear_all(self):
        """Immediately delete all motions from queue"""
        print(f"Clearing all motion data - {len(self.motion_data)} frames deleted")
        self.motion_data.clear()
        self.current_frame = 0
        self.chunk_ranges.clear()
        self.unplayed_chunk_start_ids.clear()
        self.current_chunk_start_id = None
        self.pending_interrupt = False
        self.interrupt_after_audio_id = None

        self.audio_queue.pending_interrupt = False
        self.audio_queue.interrupt_after_audio_id = None
        self.audio_queue.chunk_endpoint_id = None
        self.audio_queue.attempted_ids.clear()
        self.was_smoothed = False
        self.smoothing_accel_boost = 0.0
        print("Motion queue cleared completely")

    def set_pending_interrupt(self, current_audio_id: int, sequence: int = 0):
        """
        Keep the current chunk playing until a newer motion chunk arrives.

        Args:
            current_audio_id: Current audio ID.
            sequence:
        """
        if sequence <= self.interrupt_sequence and sequence != 0:

            return

        self.pending_interrupt = True
        self.interrupt_after_audio_id = current_audio_id
        self.interrupt_sequence = sequence

        self.audio_queue.pending_interrupt = True
        self.audio_queue.interrupt_after_audio_id = current_audio_id
        print(f"=== PENDING INTERRUPT SET === Waiting for motion with audio_id > {current_audio_id} (seq={sequence})")
        print(f"Continuing to play {len(self.motion_data) - self.current_frame} remaining frames until new motion arrives")
        time_logging(f"PENDING_INTERRUPT_SET_{current_audio_id}", str(sequence))

    def check_interrupt_file(self, motion_dir):
        """
        Process interruption markers without deleting them; use sequence numbers for deduplication.
        """
        interrupt_file = os.path.join(motion_dir, "interrupt.temp")
        if os.path.exists(interrupt_file):
            interrupt_after_id = None
            interrupt_seq = 0
            try:
                with open(interrupt_file, 'r') as f:
                    for line in f:
                        line = line.strip()
                        if line.startswith("interrupt_sequence="):
                            try:
                                interrupt_seq = int(line.split("=")[1])
                            except (IndexError, ValueError):
                                pass
                        elif line.startswith("interrupt_after_audio_id="):
                            try:
                                interrupt_after_id = int(line.split("=")[1])
                            except (IndexError, ValueError):
                                pass


                if interrupt_seq > self.interrupt_sequence:
                    print(f"New interrupt detected via file (seq {interrupt_seq} > {self.interrupt_sequence})")
                    if interrupt_after_id is not None:
                        self.set_pending_interrupt(interrupt_after_id, interrupt_seq)
                    else:
                        current_id = self.current_chunk_start_id
                        if current_id is not None:
                            self.set_pending_interrupt(current_id, interrupt_seq)
                        else:
                            self.clear_all()
                            self.interrupt_sequence = interrupt_seq
            except Exception as e:
                print(f"Error checking interrupt file: {e}")

    def is_pending_interrupt(self) -> bool:
        """Return whether playback is waiting for a post-interruption chunk."""
        return self.pending_interrupt


class GanttVisualizer:
    """
    Use Pygame to visualize audio and motion channels on a Gantt chart.
    """
    def __init__(self, clock, motion_queue, screen_width=1500, screen_height=200, title=""):
        self.clock = clock
        self.motion_queue = motion_queue
        self.screen_width = screen_width
        self.screen_height = screen_height
        self.title = title
        self.colors = {
            "background": (20, 20, 40),
            "text": (255, 230, 230),
            "now_line": (255, 0, 0),
            "grid_line": (50, 50, 70),
            "motion_cache": (0, 150, 255),
            "motion_played": (80, 80, 120),
            "audio_cache": (0, 255, 150),
            "audio_played": (0, 120, 80),
            "session_start_line": (255, 255, 0),
        }

        self.pixels_per_second = 20
        self.channel_height = 40
        self.channel_padding = 20

        self.font = None
        self.screen = None
        self.stop_event = threading.Event()

    def _draw_text(self, text, position, size=18, color=None):
        if color is None:
            color = self.colors["text"]
        text_surface = self.font.render(text, True, color)
        self.screen.blit(text_surface, position)

    def _draw_timeline(self, now_time):
        center_x = self.screen_width // 2


        seconds_in_view = self.screen_width / self.pixels_per_second
        start_time = now_time - (seconds_in_view / 2)
        end_time = now_time + (seconds_in_view / 2)

        for i in range(int(start_time) - 1, int(end_time) + 2):
            time_val = float(i)
            x_pos = center_x + (time_val - now_time) * self.pixels_per_second
            if 0 < x_pos < self.screen_width:
                pygame.draw.line(self.screen, self.colors["grid_line"], (x_pos, 0), (x_pos, self.screen_height))
                self._draw_text(f"{time_val:.1f}s", (x_pos + 5, 5), size=16)

    def _draw_channel(self, y_offset, label, segments, now_time):
        self._draw_text(label, (10, y_offset + 10))
        center_x = self.screen_width // 2

        for start_id, start_frame, end_frame, end_id in segments:

            start_x = center_x + (start_id - now_time) * self.pixels_per_second
            end_x = center_x + (end_id - now_time) * self.pixels_per_second
            width = end_x - start_x

            if start_x + width > 0 and start_x < self.screen_width:

                is_unplayed = start_id in self.motion_queue.unplayed_chunk_start_ids
                color = self.colors["motion_cache"] if is_unplayed else self.colors["motion_played"]

                pygame.draw.rect(self.screen, color, (start_x, y_offset, width, self.channel_height))

                pygame.draw.rect(self.screen, self.colors["text"], (start_x, y_offset, width, self.channel_height), 1)
                self._draw_text(f"{int(start_id)}", (start_x + 5, y_offset + 10), size=16)

    def _draw_audio_channel(self, y_offset, label, audio_segments, now_time):
        """Draw the audio channel."""
        self._draw_text(label, (10, y_offset + 10))
        center_x = self.screen_width // 2

        for audio_id, data in audio_segments.items():
            start_time = float(audio_id)

            duration = data.get('duration', 1.0)
            end_time = start_time + duration

            start_x = center_x + (start_time - now_time) * self.pixels_per_second
            end_x = center_x + (end_time - now_time) * self.pixels_per_second
            width = end_x - start_x

            if start_x + width > 0 and start_x < self.screen_width:

                color = self.colors["audio_played"] if data.get('played', False) else self.colors["audio_cache"]

                pygame.draw.rect(self.screen, color, (start_x, y_offset, width, self.channel_height))
                pygame.draw.rect(self.screen, self.colors["text"], (start_x, y_offset, width, self.channel_height), 1)
                self._draw_text(f"{audio_id}", (start_x + 5, y_offset + 10), size=16)

    def run(self):
        """Run the visualization loop in its own thread."""
        try:
            if pygame is None:
                raise RuntimeError("pygame is required for the Gantt visualizer")
            pygame.init()
            self.screen = pygame.display.set_mode((self.screen_width, self.screen_height))
            pygame.display.set_caption(f"Streaming Gantt Chart: {self.title}")
            self.font = pygame.font.Font(None, 24)

            while not self.stop_event.is_set():
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        self.stop()
                        return

                now = self.clock.get()
                if now is None:
                    now = 0


                self.screen.fill(self.colors["background"])


                self._draw_timeline(now)


                motion_segments = self.motion_queue.chunk_ranges.copy()
                self._draw_channel(50, "Motion", motion_segments, now)


                audio_segments = self.motion_queue.audio_queue.available_audio.copy()
                self._draw_audio_channel(120, "Audio", audio_segments, now)


                center_x = self.screen_width // 2
                pygame.draw.line(self.screen, self.colors["now_line"], (center_x, 0), (center_x, self.screen_height), 2)


                if client_session_start_time is not None:
                    session_elapsed_time = time.time() - client_session_start_time
                    line_x = center_x + (session_elapsed_time - now) * self.pixels_per_second
                    if 0 < line_x < self.screen_width:
                        pygame.draw.line(self.screen, self.colors["session_start_line"], (line_x, 0), (line_x, self.screen_height), 1)


                pygame.display.flip()

                time.sleep(1 / 60)

            pygame.quit()
            print("Gantt visualizer stopped.")
        except Exception as e:
            print(f"GanttVisualizer failed: {e}")

    def stop(self):
        self.stop_event.set()
