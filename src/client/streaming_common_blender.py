import time

try:
    import pygame
except ImportError:  # Optional; motion streaming works without the visualizer.
    pygame = None
import streaming_common as base

# Re-export commonly used helpers/classes from base
Clock = base.Clock
AudioQueue = base.AudioQueue
MotionQueue = base.MotionQueue
time_logging = base.time_logging
FPS = base.FPS


class GanttVisualizer(base.GanttVisualizer):
    """
    A thin extension of the base GanttVisualizer that also draws
    the current playback position (yellow line) based on motion frame index.
    """

    def __init__(self, clock, motion_queue, screen_width=1500, screen_height=200, title=""):
        super().__init__(clock, motion_queue, screen_width=screen_width, screen_height=screen_height, title=title)
        self.colors.setdefault("playback_line", (255, 215, 0))

    def _get_playback_time(self, current_frame, chunk_ranges):
        """
        Estimate the playback timestamp (seconds) using motion frame index
        and chunk ranges (start_id, start_frame, end_frame[, end_id]).
        """
        if current_frame is None or current_frame < 0:
            return None
        if not chunk_ranges:
            return None

        fps = getattr(self.motion_queue, "fps", FPS)
        if not fps or fps <= 0:
            fps = FPS

        for chunk in chunk_ranges:
            if len(chunk) == 4:
                start_id, start_frame, end_frame, end_id = chunk
            elif len(chunk) == 3:
                start_id, start_frame, end_frame = chunk
                end_id = None
            else:
                continue

            if start_id is None or start_frame is None or end_frame is None:
                continue

            if start_frame <= current_frame <= end_frame:
                frame_offset = max(0, current_frame - start_frame)
                return start_id + (frame_offset / fps)

        return None

    def _draw_channel(self, y_offset, label, segments, now_time):
        self._draw_text(label, (10, y_offset + 10))
        center_x = self.screen_width // 2
        fps = getattr(self.motion_queue, "fps", FPS) or FPS

        for segment in segments:
            if len(segment) == 4:
                start_id, start_frame, end_frame, end_id = segment
            elif len(segment) == 3:
                start_id, start_frame, end_frame = segment
                end_id = start_id + (end_frame - start_frame) / fps if start_id is not None else None
            else:
                continue

            if start_id is None or end_id is None:
                continue

            start_x = center_x + (start_id - now_time) * self.pixels_per_second
            end_x = center_x + (end_id - now_time) * self.pixels_per_second
            width = end_x - start_x

            if start_x + width > 0 and start_x < self.screen_width:
                is_unplayed = start_id in getattr(self.motion_queue, "unplayed_chunk_start_ids", [])
                color = self.colors["motion_cache"] if is_unplayed else self.colors["motion_played"]

                pygame.draw.rect(self.screen, color, (start_x, y_offset, width, self.channel_height))
                pygame.draw.rect(self.screen, self.colors["text"], (start_x, y_offset, width, self.channel_height), 1)
                self._draw_text(f"{int(start_id)}", (start_x + 5, y_offset + 10), size=16)

    def run(self):
        """Run visualization loop with playback position indicator."""
        try:
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

                center_x = self.screen_width // 2
                motion_segments = list(getattr(self.motion_queue, "chunk_ranges", []))
                current_frame = getattr(self.motion_queue, "current_frame", None)
                playback_time = self._get_playback_time(current_frame, motion_segments)

                # Background
                self.screen.fill(self.colors["background"])

                # Timeline/grid
                self._draw_timeline(now)

                # Motion/Audio channels
                self._draw_channel(50, "Motion", motion_segments, now)

                audio_segments = {}
                audio_queue = getattr(self.motion_queue, "audio_queue", None)
                if audio_queue is not None:
                    available_audio = getattr(audio_queue, "available_audio", {})
                    audio_segments = available_audio.copy() if hasattr(available_audio, "copy") else dict(available_audio)
                self._draw_audio_channel(120, "Audio", audio_segments, now)

                # Playback position (yellow)
                if playback_time is not None:
                    playback_x = center_x + (playback_time - now) * self.pixels_per_second
                    if 0 < playback_x < self.screen_width:
                        pygame.draw.line(self.screen, self.colors["playback_line"], (playback_x, 0), (playback_x, self.screen_height), 2)
                        self._draw_text(f"{playback_time:.2f}s", (playback_x + 5, self.screen_height - 25), size=14, color=self.colors["playback_line"])

                # "Now" line (red)
                pygame.draw.line(self.screen, self.colors["now_line"], (center_x, 0), (center_x, self.screen_height), 2)

                # Session start line (from base module)
                if base.client_session_start_time is not None:
                    session_elapsed_time = time.time() - base.client_session_start_time
                    line_x = center_x + (session_elapsed_time - now) * self.pixels_per_second
                    if 0 < line_x < self.screen_width:
                        pygame.draw.line(self.screen, self.colors["session_start_line"], (line_x, 0), (line_x, self.screen_height), 1)

                pygame.display.flip()
                time.sleep(1 / 60)

            pygame.quit()
            print("Gantt visualizer stopped.")
        except Exception as e:
            print(f"GanttVisualizer failed: {e}")
