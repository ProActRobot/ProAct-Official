import time
import sys
import pandas as pd
import soundfile as sf
import threading
import numpy as np
import scipy.signal
import traceback
import os
import streaming_common
import subprocess
import copy

_CLIENT_DIR = os.path.dirname(__file__)
_BUNDLED_SDK_DIR = os.path.abspath(os.path.join(_CLIENT_DIR, os.pardir, "third_party"))
if os.path.isdir(_BUNDLED_SDK_DIR) and _BUNDLED_SDK_DIR not in sys.path:
    sys.path.insert(0, _BUNDLED_SDK_DIR)

from streaming_common import *
from unitree_sdk2py.core.channel import ChannelPublisher, ChannelFactoryInitialize
from unitree_sdk2py.core.channel import ChannelSubscriber
from unitree_sdk2py.idl.default import unitree_hg_msg_dds__LowCmd_
from unitree_sdk2py.idl.default import unitree_hg_msg_dds__LowState_
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowCmd_
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowState_
from unitree_sdk2py.utils.crc import CRC
from unitree_sdk2py.utils.thread import RecurrentThread
from unitree_sdk2py.g1.audio.g1_audio_client import AudioClient
from unitree_sdk2py.g1.loco.g1_loco_client import LocoClient
try:
    from unitree_sdk2py.go2.video.video_client import VideoClient
except ImportError:
    VideoClient = None

def read_wav_pcm_data(filename):
    import wave
    with wave.open(filename, 'rb') as f:
        params = f.getparams()
        nchannels, sampwidth, framerate, nframes = params[:4]
        content = f.readframes(nframes)

        # Convert samples to a NumPy int16 array.
        samples = np.frombuffer(content, dtype=np.int16)

        # Resample audio when the source rate is not 16 kHz.
        target_samplerate = 16000
        if framerate != target_samplerate:
            num_samples = int(len(samples) * target_samplerate / framerate)
            samples = scipy.signal.resample(samples, num_samples).astype(np.int16)

        return list(samples.tobytes())

class AudioInterface:
    def __init__(self):
        self.audio_client = AudioClient()
        self.audio_client.SetTimeout(10.0)
        self.audio_client.Init()
        self.volume = 100
        self.audio_client.SetVolume(100)
        ret = self.audio_client.GetVolume()
        print("debug GetVolume: ", ret)

    def set_volume(self, volume):
        self.volume = volume
        if self.volume <= 100:
            self.audio_client.SetVolume(self.volume)
        else:
            self.audio_client.SetVolume(100)

    def play(self, audio_path):
        if self.audio_client:
            pcm_data = read_wav_pcm_data(audio_path)
            if self.volume > 100:
                volume_factor = self.volume / 100.0
                if isinstance(pcm_data, (list, tuple)):
                    raw_bytes = bytes(pcm_data)
                else:
                    raw_bytes = pcm_data
                pcm_array = np.frombuffer(raw_bytes, dtype=np.int16).copy()
                pcm_array = np.clip(pcm_array.astype(np.float32) * volume_factor, -32768, 32767).astype(np.int16)
                output_bytes = pcm_array.tobytes()
                if isinstance(pcm_data, (list, tuple)):
                    pcm_data = list(output_bytes)
                else:
                    pcm_data = output_bytes
            self.audio_client.PlayStream('00', '0', pcm_data)
        else:
            print("AudioClient not initialized, skipping playback.")

# class AudioInterface:
#     def __init__(self):
#         pass
#     def play(self, audio_path):
#         # --- Use AudioClient for playback ---
#         command = ['mpv', '--no-terminal', '--force-window=no', '--audio-display=no', audio_path]
#         clean_env = copy.deepcopy(os.environ)
#         if 'LD_LIBRARY_PATH' in clean_env:
#             del clean_env['LD_LIBRARY_PATH']
#         process = subprocess.Popen(
#             command,
#             env=clean_env,
#             stdout=subprocess.PIPE,
#             stderr=subprocess.PIPE,
#         )
#         # --- End AudioClient playback ---


class G1JointIndex:
    # Left leg
    LeftHipPitch = 0
    LeftHipRoll = 1
    LeftHipYaw = 2
    LeftKnee = 3
    LeftAnklePitch = 4
    LeftAnkleB = 4
    LeftAnkleRoll = 5
    LeftAnkleA = 5

    # Right leg
    RightHipPitch = 6
    RightHipRoll = 7
    RightHipYaw = 8
    RightKnee = 9
    RightAnklePitch = 10
    RightAnkleB = 10
    RightAnkleRoll = 11
    RightAnkleA = 11

    WaistYaw = 12
    WaistRoll = 13        # NOTE: INVALID for g1 23dof/29dof with waist locked
    WaistA = 13           # NOTE: INVALID for g1 23dof/29dof with waist locked
    WaistPitch = 14       # NOTE: INVALID for g1 23dof/29dof with waist locked
    WaistB = 14           # NOTE: INVALID for g1 23dof/29dof with waist locked

    # Left arm
    LeftShoulderPitch = 15
    LeftShoulderRoll = 16
    LeftShoulderYaw = 17
    LeftElbow = 18
    LeftWristRoll = 19
    LeftWristPitch = 20   # NOTE: INVALID for g1 23dof
    LeftWristYaw = 21     # NOTE: INVALID for g1 23dof

    # Right arm
    RightShoulderPitch = 22
    RightShoulderRoll = 23
    RightShoulderYaw = 24
    RightElbow = 25
    RightWristRoll = 26
    RightWristPitch = 27  # NOTE: INVALID for g1 23dof
    RightWristYaw = 28    # NOTE: INVALID for g1 23dof

    kNotUsedJoint = 29 # NOTE: Weight

class RobotController:
    def __init__(self, motion_queue):
        self.motion_queue = motion_queue
        self.frame_step_ = 0  # Frame counter
        self.control_dt_ = 0.033
        self.kp = 70.
        self.kd = 2
        self.low_cmd = unitree_hg_msg_dds__LowCmd_()
        self.low_state = None
        self.first_update_low_state = False
        self.crc = CRC()

        self.arm_joints = [
          G1JointIndex.WaistYaw,
          G1JointIndex.WaistRoll,
          G1JointIndex.WaistPitch,
          G1JointIndex.LeftShoulderPitch,  G1JointIndex.LeftShoulderRoll,
          G1JointIndex.LeftShoulderYaw,    G1JointIndex.LeftElbow,
          G1JointIndex.LeftWristRoll,      G1JointIndex.LeftWristPitch,
          G1JointIndex.LeftWristYaw,
          G1JointIndex.RightShoulderPitch, G1JointIndex.RightShoulderRoll,
          G1JointIndex.RightShoulderYaw,   G1JointIndex.RightElbow,
          G1JointIndex.RightWristRoll,     G1JointIndex.RightWristPitch,
          G1JointIndex.RightWristYaw,
        ]

        # Smoothing state
        self.dt = 1.0 / FPS

    def Init(self):
        # create publisher #
        self.arm_sdk_publisher = ChannelPublisher("rt/arm_sdk", LowCmd_)
        self.arm_sdk_publisher.Init()

        # create subscriber #
        self.lowstate_subscriber = ChannelSubscriber("rt/lowstate", LowState_)
        self.lowstate_subscriber.Init(self.LowStateHandler, 10)

    def Start(self):
        self.lowCmdWriteThreadPtr = RecurrentThread(
            interval=self.control_dt_, target=self.LowCmdWrite, name="control"
        )
        while self.first_update_low_state == False:
            time.sleep(1)

        if self.first_update_low_state == True:
            self.lowCmdWriteThreadPtr.Start()

    def LowStateHandler(self, msg: LowState_):
        self.low_state = msg
        if self.first_update_low_state == False:
            self.first_update_low_state = True
            # Initialize last_qpos in motion_queue from current state
            self.motion_queue.last_qpos = np.array([self.low_state.motor_state[j].q for j in self.arm_joints])

    def LowCmdWrite(self):
        self.frame_step_ += 1  # Increment frame counter

        if self.low_state is not None:
            # Get current qpos for arm joints
            current_qpos = np.array([self.low_state.motor_state[j].q for j in self.arm_joints])

            # Use MotionQueue's internal smoothing logic
            motion_targets = self.motion_queue.get_motion(current_qpos=current_qpos, dt=self.dt)

            if motion_targets is not None:
                new_target_qpos = np.array(motion_targets[:len(self.arm_joints)])

                # Apply to LowCmd
                for i, joint in enumerate(self.arm_joints):
                    self.low_cmd.motor_cmd[joint].tau = 0.
                    if joint == G1JointIndex.RightWristYaw:
                        self.low_cmd.motor_cmd[joint].q = new_target_qpos[i]
                    else:
                        self.low_cmd.motor_cmd[joint].q = new_target_qpos[i]
                    self.low_cmd.motor_cmd[joint].dq = 0.
                    self.low_cmd.motor_cmd[joint].kp = self.kp
                    self.low_cmd.motor_cmd[joint].kd = self.kd


                # Enable arm_sdk
                self.low_cmd.motor_cmd[G1JointIndex.kNotUsedJoint].q = 1

                if self.frame_step_ % 30 == 0:
                    print(f"[Playing motion, frame {self.frame_step_}, id={self.motion_queue.current_chunk_start_id}]")
            else:
                # No motion in queue
                if self.frame_step_ % 60 == 0:
                    print(f"[***Warning: No motion in queue, frame {self.frame_step_}]")

                # Keep arm_sdk enabled
                self.low_cmd.motor_cmd[G1JointIndex.kNotUsedJoint].q = 1
        else:
            print("Error: low_state is None")

        self.low_cmd.crc = self.crc.Crc(self.low_cmd)
        self.arm_sdk_publisher.Write(self.low_cmd)

    def load_csv_to_queue(self, csv_file):
        """
        Load arm and waist joint targets from a CSV file into the motion queue.
        """
        time.sleep(0.04) # once csv file is detected, a small chance that the file is partial so wait.
        df = pd.read_csv(csv_file)
        print(f'loading csv file: {csv_file}')

        filename = os.path.basename(csv_file)
        audio_id = None
        if filename.startswith("motion_") and filename.endswith(".csv"):
            try:
                audio_id = int(filename.split("_")[1].split(".")[0])
            except (IndexError, ValueError):
                audio_id = None

        # Use the joint list that matches arm_joints in RobotController
        joint_lists = [
            'waist_yaw', 'waist_roll', 'waist_pitch',
            'left_shoulder_pitch', 'left_shoulder_roll', 'left_shoulder_yaw', 'left_elbow',
            'left_wrist_roll', 'left_wrist_pitch', 'left_wrist_yaw',
            'right_shoulder_pitch', 'right_shoulder_roll', 'right_shoulder_yaw', 'right_elbow',
            'right_wrist_roll', 'right_wrist_pitch', 'right_wrist_yaw'
        ]
        joint_lists = [i+'_joint' for i in joint_lists]

        # Check if columns exist
        available_cols = [c for c in joint_lists if c in df.columns]
        if len(available_cols) != len(joint_lists):
            print(f"Warning: Missing columns in CSV. Expected {len(joint_lists)}, found {len(available_cols)}")

        motion_frames = df[joint_lists].astype(float).values.tolist()
        self.motion_queue.append(motion_frames, audio_id)
        print(f'loaded csv shape: {len(motion_frames)} frames')

    def enter_damping_state(self):
        # Damping: set all joints to q=0, dq=0, kp=0, kd=8, tau=0
        print("Entering damping state...")
        for i in range(len(self.low_cmd.motor_cmd)):
            self.low_cmd.motor_cmd[i].dq = 0
            self.low_cmd.motor_cmd[i].kp = 0
            self.low_cmd.motor_cmd[i].kd = 4
            self.low_cmd.motor_cmd[i].tau = 0
        self.low_cmd.crc = self.crc.Crc(self.low_cmd)
        self.arm_sdk_publisher.Write(self.low_cmd)
        print("Exit")

class LocoController:
    def __init__(self):
        try:
            self.loco_client = LocoClient()
            self.loco_client.SetTimeout(10.0)
            self.loco_client.Init()
            print('locomotion init success')
        except Exception as e:
            print("locomotion init failed: ", e)

    def _move(self, vx: float, vy: float, vyaw: float, duration: float):
        duration = 5.0 if duration > 5.0 or duration < 0.0 else duration
        self.loco_client.SetVelocity(vx, vy, vyaw, duration=duration)
        time.sleep(duration) # Wait for the movement to complete before processing the next command in the queue

    def turn_left(self, approx_angle: float = 90):
        if approx_angle < 0:
            self.turn_right(approx_angle=-approx_angle)
            return
        approx_angle = approx_angle + 30 if approx_angle <= 30 else approx_angle
        approx_angle = 180 if approx_angle > 180 else approx_angle
        # self._move(0.0, 0.0, 0.3, approx_angle/15)
        self._move(0.0, 0.0, 1, approx_angle/45)

    def turn_right(self, approx_angle: float = 90):
        if approx_angle < 0:
            self.turn_left(approx_angle=-approx_angle)
            return
        approx_angle = approx_angle + 30 if approx_angle <= 30 else approx_angle
        approx_angle = 180 if approx_angle > 180 else approx_angle
        # self._move(0.0, 0.0, -0.3, approx_angle/15)
        self._move(0.0, 0.0, -1, approx_angle/45)

    def move_forward(self, approx_distance: float = 1):
        approx_distance = 1.5 if approx_distance > 1.5 else approx_distance
        approx_distance = 0.0 if approx_distance < 0.0 else approx_distance
        # self._move(0.3, 0.0, 0.0, approx_distance/0.3)
        self._move(0.5, 0.0, 0.0, approx_distance/0.5)
        # self._move(0.15, 0.0, 0.0, approx_distance/0.15)

    def move_backward(self, approx_distance: float = 1):
        approx_distance = 1.5 if approx_distance > 1.5 else approx_distance
        approx_distance = 0.0 if approx_distance < 0.0 else approx_distance
        self._move(-0.3, 0.0, 0.0, approx_distance/0.3)

    def move_left(self, approx_distance: float = 1):
        if approx_distance < 0:
            self.move_right(-approx_distance)
            return
        approx_distance = 1.5 if approx_distance > 1.5 else approx_distance
        self._move(0.0, 0.3, 0.0, approx_distance/0.3)

    def move_right(self, approx_distance: float = 1):
        if approx_distance < 0:
            self.move_left(-approx_distance)
            return
        approx_distance = 1.5 if approx_distance > 1.5 else approx_distance
        self._move(0.0, -0.3, 0.0, approx_distance/0.3)

    def execute_locomotion(self, instruction: str, parameter: float = None):
        val = None
        if parameter is not None:
            try:
                val = float(parameter)
            except ValueError:
                print(f"Invalid parameter for locomotion: {parameter}")
                return

        if instruction == "turn_left":
            if val is not None: self.turn_left(val)
            else: self.turn_left()
        elif instruction == "turn_right":
            if val is not None: self.turn_right(val)
            else: self.turn_right()
        elif instruction == "move_forward":
            if val is not None: self.move_forward(val)
            else: self.move_forward()
        elif instruction == "move_backward":
            if val is not None: self.move_backward(val)
            else: self.move_backward()
        elif instruction == "move_left":
            if val is not None: self.move_left(val)
            else: self.move_left()
        elif instruction == "move_right":
            if val is not None: self.move_right(val)
            else: self.move_right()
        else:
            print(f"Unknown locomotion instruction: {instruction}")

def locomotion_worker(queue, loco_controller):
    print("Locomotion worker started")
    while True:
        try:
            cmd = queue.get()
            if cmd is None:
                break
            instruction, parameter = cmd
            print(f"Received locomotion command: {instruction}, {parameter}")
            loco_controller.execute_locomotion(instruction, parameter)
        except Exception as e:
            print(f"Error in locomotion worker: {e}")

def robot_camera_worker(queue, interval=1.0):
    if VideoClient is None:
        print("Unitree video client is unavailable; robot camera disabled.")
        return
    print("Robot camera worker started")
    try:
        client = VideoClient()
        client.SetTimeout(3.0)
        client.Init()
        code, encoded = client.GetImageSample()
        if code != 0:
             print(f"Robot camera init failed with code {code}")
    except Exception as e:
        print(f"Robot camera worker failed to init: {e}")
        return

    while True:
        start_time = time.time()
        try:
            code, encoded = client.GetImageSample()
            if code == 0:
                # Send image bytes to queue
                queue.put(bytes(encoded))
            else:
                print(f"GetImageSample failed: {code}")
        except Exception as e:
            print(f"Error in robot camera worker: {e}")

        elapsed = time.time() - start_time
        sleep_time = max(0, interval - elapsed)
        time.sleep(sleep_time)

def main_real(audio_dir=None,
            motion_dir=None,
            zero_point_ns_arg=0,
            session_dir=None,
            session_start_time=None,
            locomotion_queue=None,
            image_queue=None,
            is_robot_playing=None,
            last_robot_play_end_time=None):
    runtime_dir = os.path.join(os.path.dirname(__file__), "runtime")
    audio_dir = audio_dir or os.getenv("PROACT_AUDIO_DIR", os.path.join(runtime_dir, "assistant_audio"))
    motion_dir = motion_dir or os.getenv("PROACT_MOTION_DIR", os.path.join(runtime_dir, "motion_data"))
    os.makedirs(audio_dir, exist_ok=True)
    os.makedirs(motion_dir, exist_ok=True)
    ChannelFactoryInitialize(0, os.getenv("PROACT_UNITREE_INTERFACE", "enp7s0"))
    streaming_common.zero_point_ns = zero_point_ns_arg
    streaming_common.local_session_dir = session_dir
    streaming_common.client_session_start_time = session_start_time
    # Initialize AudioClient
    audio_interface = AudioInterface()

    # Define callbacks for playback status
    def on_play_start():
        if is_robot_playing is not None:
            is_robot_playing.value = True
        if last_robot_play_end_time is not None:
            last_robot_play_end_time.value = time.time()
            # print("DEBUG: Robot started playing audio (shared flag set)")

    def on_play_end():
        if is_robot_playing is not None:
            is_robot_playing.value = False
            # print("DEBUG: Robot finished playing audio (shared flag cleared)")


    # Create motion queue with audio client
    motion_queue = MotionQueue(audio_interface, audio_dir, on_play_start=on_play_start, on_play_end=on_play_end)

    # Initialize Robot Controller with motion queue
    robot_controller = RobotController(motion_queue)
    robot_controller.Init()

    # Initialize LocoController and start worker thread if queue is provided
    if locomotion_queue is not None:
        try:
            loco_controller = LocoController()
            loco_thread = threading.Thread(target=locomotion_worker, args=(locomotion_queue, loco_controller), daemon=True)
            loco_thread.start()
            print("Locomotion worker thread started")
        except Exception as e:
            print(f"Failed to start locomotion worker: {e}")

    # Initialize Robot Camera Worker if image_queue is provided
    if image_queue is not None:
        try:
            camera_thread = threading.Thread(target=robot_camera_worker, args=(image_queue, 1.0), daemon=True)
            camera_thread.start()
            print("Robot camera worker thread started")
        except Exception as e:
            print(f"Failed to start robot camera worker: {e}")

    # Start the playback timeline visualizer.
    gantt_visualizer = GanttVisualizer(motion_queue.audio_queue.clock, motion_queue, title="Real")
    gantt_thread = threading.Thread(target=gantt_visualizer.run, daemon=True)
    gantt_thread.start()
    # Start the control thread once - it will run forever
    robot_controller.Start()
    print("Robot control thread started, waiting for motion queue data...")

    processed_motion_files = set()

    # main loop - scan for CSV files
    try:
        while True:
            # Check for interrupt file before processing anything
            motion_queue.check_interrupt_file(motion_dir)

            # Scan for new motion files
            if os.path.exists(motion_dir):
                current_motion_files = set(f for f in os.listdir(motion_dir) if f.startswith("motion_") and f.endswith(".csv"))
                new_motion_files = current_motion_files - processed_motion_files
                if len(new_motion_files) > 1:
                    new_motion_files = sorted(new_motion_files, key=lambda x: int(x.split("_")[1].split(".")[0]))

                # Process new motion files
                for motion_file in new_motion_files:
                    motion_number = motion_file.split("_")[1].split(".")[0]
                    processing_wait_file = f"processing_{motion_number}.wait"

                    # Check if the processing wait file exists
                    if not os.path.exists(os.path.join(motion_dir, processing_wait_file)):
                        print(f"New motion file detected: {motion_file}")
                        robot_controller.load_csv_to_queue(os.path.join(motion_dir, motion_file))
                        processed_motion_files.add(motion_file)
                    else:
                        print(f"Skipping {motion_file} because {processing_wait_file} exists.")

            time.sleep(0.1)  # Add a delay to reduce CPU usage

    except KeyboardInterrupt:
        print("Ctrl+C detected")

    gantt_visualizer.stop()
    robot_controller.enter_damping_state()
    sys.exit(0)


if __name__ == '__main__':
    main_real(
        audio_dir=os.getenv("PROACT_AUDIO_DIR"),
        motion_dir=os.getenv("PROACT_MOTION_DIR"),
        zero_point_ns_arg=0,
        session_dir=None,
        session_start_time=None,
    )
