import time
import sys
import pandas as pd
import soundfile as sf
import threading
import mujoco
import mujoco.viewer
import numpy as np
import traceback
import os
import subprocess
import copy
import streaming_common
from streaming_common import *

class AudioInterface:
    def __init__(self):
        pass
    def play(self, audio_path):
        # --- Use AudioClient for playback ---
        command = ['mpv', '--no-terminal', '--force-window=no', '--audio-display=no', audio_path]
        clean_env = copy.deepcopy(os.environ)
        if 'LD_LIBRARY_PATH' in clean_env:
            del clean_env['LD_LIBRARY_PATH']
        process = subprocess.Popen(
            command,
            env=clean_env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        # --- End AudioClient playback ---

class MujocoController:
    def __init__(self, motion_queue, robot_type='g1'):
        self.motion_queue = motion_queue
        self.frame_step_ = 0

        # MuJoCo setup
        model_xml_path = os.path.join(os.path.dirname(__file__), '..', 'assets', 'g1', 'scene_empty_coll.xml')
        self.model = mujoco.MjModel.from_xml_path(model_xml_path)
        self.data = mujoco.MjData(self.model)
        self.viewer = mujoco.viewer.launch_passive(model=self.model, data=self.data, show_left_ui=True, show_right_ui=True)

        # Set camera position
        self.viewer.cam.azimuth = 180         # Azimuth, in degrees
        self.viewer.cam.elevation = -5      # Elevation, in degrees
        self.viewer.cam.distance = 2.0       # Distance from the lookat point
        self.viewer.cam.lookat[:] = [0, 0, 0.9] # Lookat point
        self.model.opt.gravity[2] = 0
        mujoco.mj_fwdPosition(self.model, self.data)

        # Initialize smoothing state in motion_queue
        joint_slice = slice(19, 19 + 31)
        self.motion_queue.last_qpos = np.copy(self.data.qpos[joint_slice])
        self.dt = 1.0 / FPS  # simulation timestep

    def Start(self):
        print("Starting Mujoco visualization...")

        def visualization_loop():
            def update_overlay(viewer, text):
                """Update the overlay in the MuJoCo viewer."""
                viewer.user_scn.ngeom = 0
                mujoco.mjv_initGeom(
                    viewer.user_scn.geoms[0],
                    type=mujoco.mjtGeom.mjGEOM_LABEL,
                    size=np.array([0.05, 0.05, 0.05]),
                    pos=np.array([0.0, 0.0, 1.5]),
                    mat=np.eye(3).flatten(),
                    rgba=np.array([1.0, 1.0, 1.0, 1.0])  # White text
                )
                viewer.user_scn.geoms[0].label = text[:99]
                viewer.user_scn.ngeom = 1
            start_time = time.time()
            next_frame_time = time.time()
            while self.viewer.is_running():
                total_frames = len(self.motion_queue.motion_data)
                current_frame_in_queue = self.motion_queue.current_frame

                # Get current state for slice
                joint_slice = slice(19, 19 + 31)
                current_sim_qpos = self.data.qpos[joint_slice]

                # Use unified smoothing logic in MotionQueue
                motion_targets = self.motion_queue.get_motion(current_qpos=current_sim_qpos, dt=self.dt)

                if motion_targets is not None:
                    # Set root position and orientation (e.g., stand on the ground)
                    self.data.qpos[0] = 0      # x
                    self.data.qpos[1] = 0      # y
                    self.data.qpos[2] = 0.79    # z (height)
                    self.data.qpos[3] = 1      # qw
                    self.data.qpos[4] = 0      # qx
                    self.data.qpos[5] = 0      # qy
                    self.data.qpos[6] = 0      # qz

                    self.data.qpos[joint_slice] = np.array(motion_targets)
                    mujoco.mj_fwdPosition(self.model, self.data)
                    self.frame_step_ += 1

                clock_time = self.motion_queue.audio_queue.clock.get()
                clock_time_str = f"{clock_time:.2f}s" if clock_time is not None else "Clock not set"
                overlay_text = f"{clock_time_str} | {current_frame_in_queue}\t/{total_frames} frames | id {self.motion_queue.current_chunk_start_id} .. {self.motion_queue.unplayed_chunk_start_ids}"
                update_overlay(self.viewer, overlay_text)

                self.viewer.sync()

                # Real-time playback at ~30 FPS (0.0333s per frame)
                # next_frame_time += 1.0/FPS
                next_frame_time += 0.033
                # next_frame_time += 0.016667
                sleep_time = next_frame_time - time.time()
                if sleep_time > 0:
                    time.sleep(sleep_time)
                elif sleep_time < -1:  # If lagging significantly, reset timer
                    next_frame_time = time.time()
                    print("Visualization loop is lagging behind real-time.")

        visualization_thread = threading.Thread(target=visualization_loop, daemon=True)
        visualization_thread.start()

    def load_csv_to_queue(self, csv_file):
        """
        Load arm and waist joint targets from a CSV file into the motion queue.
        """
        time.sleep(0.04) # once csv file is detected, a small chance that the file is partial so wait.
        df = pd.read_csv(csv_file)
        # print(f'loading csv file: {csv_file}')

        # Extract audio_id from filename (e.g., "motion_123.csv" -> 123)
        import os
        filename = os.path.basename(csv_file)
        audio_id = None
        if filename.startswith("motion_") and filename.endswith(".csv"):
            try:
                audio_id = int(filename.split("_")[1].split(".")[0])
            except (IndexError, ValueError):
                audio_id = None

        joint_lists = [
            'waist_yaw', 'waist_roll', 'waist_pitch',
            'left_shoulder_pitch', 'left_shoulder_roll', 'left_shoulder_yaw', 'left_elbow',
            'left_wrist_roll', 'left_wrist_pitch', 'left_wrist_yaw',
            'left_hand_thumb_0','left_hand_thumb_1','left_hand_thumb_2','left_hand_middle_0','left_hand_middle_1','left_hand_index_0','left_hand_index_1',
            'right_shoulder_pitch', 'right_shoulder_roll', 'right_shoulder_yaw', 'right_elbow',
            'right_wrist_roll', 'right_wrist_pitch', 'right_wrist_yaw',
            'right_hand_thumb_0','right_hand_thumb_1','right_hand_thumb_2','right_hand_middle_0','right_hand_middle_1','right_hand_index_0','right_hand_index_1',
        ]
        joint_lists = [i+'_joint' for i in joint_lists]
        motion_frames = df[joint_lists].astype(float).values.tolist()
        self.motion_queue.append(motion_frames, audio_id)
        # print(f'loaded csv shape: {len(motion_frames)} frames')
        # print(f'dof: {len(motion_frames[0])}')

    def enter_damping_state(self):
        print("Damping state is not applicable for visualization.")

def main_mujoco(audio_dir=None,
                  motion_dir=None,
                  zero_point_ns_arg=0,
                  session_dir=None,
                  session_start_time=None,
                  is_mujoco_playing=None,
                  last_play_end_time=None):
    runtime_dir = os.path.join(os.path.dirname(__file__), "runtime")
    audio_dir = audio_dir or os.getenv("PROACT_AUDIO_DIR", os.path.join(runtime_dir, "assistant_audio"))
    motion_dir = motion_dir or os.getenv("PROACT_MOTION_DIR", os.path.join(runtime_dir, "motion_data"))
    os.makedirs(audio_dir, exist_ok=True)
    os.makedirs(motion_dir, exist_ok=True)
    streaming_common.zero_point_ns = zero_point_ns_arg
    streaming_common.local_session_dir = session_dir
    streaming_common.client_session_start_time = session_start_time
    audio_interface = AudioInterface()

    # Define callbacks for playback status
    def on_play_start():
        if is_mujoco_playing is not None:
            is_mujoco_playing.value = True
            # print("DEBUG: Mujoco started playing audio (shared flag set)")
        if last_play_end_time is not None:
            last_play_end_time.value = time.time()

    def on_play_end():
        if is_mujoco_playing is not None:
            is_mujoco_playing.value = False
            # print("DEBUG: Mujoco finished playing audio (shared flag cleared)")


    # Create motion queue with audio client
    motion_queue = MotionQueue(audio_interface, audio_dir, on_play_start=on_play_start, on_play_end=on_play_end)

    # Initialize Motion Control with motion queue
    mjc = MujocoController(motion_queue)

    # Start the playback timeline visualizer.
    gantt_visualizer = GanttVisualizer(motion_queue.audio_queue.clock, motion_queue, title='Simulation Mujoco')
    gantt_thread = threading.Thread(target=gantt_visualizer.run, daemon=True)
    gantt_thread.start()

    # Start the control thread once - it will run forever
    mjc.Start()
    print("Motion control thread started, waiting for motion queue data...")

    processed_motion_files = set()

    # main loop - only scan for CSV files
    try:
        while True:
            # Check for interrupt file before processing anything
            mjc.motion_queue.check_interrupt_file(motion_dir)

            # Scan for new motion files
            current_motion_files = set(f for f in os.listdir(motion_dir) if f.startswith("motion_") and f.endswith(".csv"))
            new_motion_files = current_motion_files - processed_motion_files

            # Sort new motion files by motion number if there are multiple files
            if len(new_motion_files) > 1:
                new_motion_files = sorted(new_motion_files, key=lambda x: int(x.split("_")[1].split(".")[0]))

            # Process new motion files
            for motion_file in new_motion_files:
                motion_number = motion_file.split("_")[1].split(".")[0]
                processing_wait_file = f"processing_{motion_number}.wait"

                # Check if the processing wait file exists
                if not os.path.exists(os.path.join(motion_dir, processing_wait_file)):
                    print(f"New motion file detected: {motion_file}")
                    mjc.load_csv_to_queue(os.path.join(motion_dir, motion_file))
                    processed_motion_files.add(motion_file)
                else:
                    print(f"Skipping {motion_file} because {processing_wait_file} exists.")

            time.sleep(0.1)  # Add a delay to reduce CPU usage

    except KeyboardInterrupt:
        print("Ctrl+C detected")

    gantt_visualizer.stop()
    mjc.enter_damping_state()
    sys.exit(0)


if __name__ == '__main__':
    main_mujoco(
        audio_dir=os.getenv("PROACT_AUDIO_DIR"),
        motion_dir=os.getenv("PROACT_MOTION_DIR"),
        zero_point_ns_arg=0,
        session_dir=None,
        session_start_time=None,
    )
