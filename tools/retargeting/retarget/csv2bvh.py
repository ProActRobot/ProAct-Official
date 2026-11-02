import numpy as np
import pandas as pd


def csv_to_bvh(hierarchy_bvh_file, csv_file, out_bvh_file, fps=30):
    with open(out_bvh_file, "w", encoding="utf-8") as fout:
        with open(hierarchy_bvh_file, "r") as hierarchy_file:
            hierarchy_lines = hierarchy_file.readlines()
        for line in hierarchy_lines:
            fout.write(line)
        fout.write("MOTION\n")
        df = pd.read_csv(csv_file)
        joint_names = [
            "waist_yaw",
            "waist_roll",
            "waist_pitch",
            "left_shoulder_pitch",
            "left_shoulder_roll",
            "left_shoulder_yaw",
            "left_elbow",
            "left_wrist_roll",
            "left_wrist_pitch",
            "left_wrist_yaw",
            "left_hand_thumb_0",
            "left_hand_thumb_1",
            "left_hand_thumb_2",
            "left_hand_middle_0",
            "left_hand_middle_1",
            "left_hand_index_0",
            "left_hand_index_1",
            "right_shoulder_pitch",
            "right_shoulder_roll",
            "right_shoulder_yaw",
            "right_elbow",
            "right_wrist_roll",
            "right_wrist_pitch",
            "right_wrist_yaw",
            "right_hand_thumb_0",
            "right_hand_thumb_1",
            "right_hand_thumb_2",
            "right_hand_middle_0",
            "right_hand_middle_1",
            "right_hand_index_0",
            "right_hand_index_1",
        ]
        joint_seq_of = {
            "left_hip_pitch": 2,
            "left_hip_roll": 4,
            "left_hip_yaw": 5,
            "left_knee": 7,
            "left_ankle_pitch": 8,
            "left_ankle_roll": 9,
            "right_hip_pitch": 10,
            "right_hip_roll": 12,
            "right_hip_yaw": 13,
            "right_knee": 15,
            "right_ankle_pitch": 16,
            "right_ankle_roll": 17,
            "waist_yaw": 18,
            "waist_roll": 19,
            "waist_pitch": 20,
            "left_shoulder_pitch": 22,
            "left_shoulder_roll": 24,
            "left_shoulder_yaw": 25,
            "left_elbow": 26,
            "left_wrist_roll": 27,
            "left_wrist_pitch": 28,
            "left_wrist_yaw": 29,
            "left_hand_thumb_0": 30,
            "left_hand_thumb_1": 31,
            "left_hand_thumb_2": 32,
            "left_hand_middle_0": 33,
            "left_hand_middle_1": 34,
            "left_hand_index_0": 35,
            "left_hand_index_1": 36,
            "right_shoulder_pitch": 38,
            "right_shoulder_roll": 40,
            "right_shoulder_yaw": 41,
            "right_elbow": 42,
            "right_wrist_roll": 43,
            "right_wrist_pitch": 44,
            "right_wrist_yaw": 45,
            "right_hand_thumb_0": 46,
            "right_hand_thumb_1": 47,
            "right_hand_thumb_2": 48,
            "right_hand_middle_0": 49,
            "right_hand_middle_1": 50,
            "right_hand_index_0": 51,
            "right_hand_index_1": 52,
        }

        def which_axis_offset(joint_name):
            id = -1
            if "pitch" in joint_name or "knee" in joint_name or "elbow" in joint_name:
                id = 1
            elif "roll" in joint_name:
                id = 2
            elif "yaw" in joint_name:
                id = 0
            if "thumb" in joint_name or "middle" in joint_name or "index" in joint_name:
                id = 0
            if "thumb_0" in joint_name:
                id = 1
            if id == -1:
                raise NotImplementedError
            return id

        index_of = {
            joint_name: joint_seq_of[joint_name] * 3 + which_axis_offset(joint_name)
            for joint_name in joint_names
        }
        new_df = pd.DataFrame(0.0, index=range(df.shape[0]), columns=range(159))
        new_df[5] = -90.0
        new_df[3 * 3 + 1] = -10.02103094
        new_df[3 * 6 + 1] = 10.02103094
        new_df[3 * 11 + 1] = -10.02103094
        new_df[3 * 14 + 1] = 10.02103094
        new_df[3 * 21] = -0.0114198554
        new_df[3 * 21 + 1] = -1.84476243e-08
        new_df[3 * 21 + 2] = 16.0032308
        new_df[3 * 23 + 2] = -15.99987616
        new_df[3 * 30] = 0.0114198554
        new_df[3 * 30 + 1] = -1.84476243e-08
        new_df[3 * 30 + 2] = -16.0032308
        new_df[3 * 32 + 2] = 15.99987616
        for index, row in df.iterrows():
            for joint_name in joint_names:
                if joint_name + "_joint" in row:
                    df.at[index, joint_name + "_rotation"] = np.rad2deg(row[joint_name + "_joint"])
                    new_df.at[index, index_of[joint_name]] = np.rad2deg(row[joint_name + "_joint"])
                else:
                    raise NotImplementedError
        fout.write("Frames: {}\n".format(df.shape[0]))
        fout.write("Frame Time: {}\n".format(1.0 / fps))
        for index, row in new_df.iterrows():
            frame_data = " ".join((f"{value:.6f}" for value in row))
            fout.write(frame_data + "\n")
