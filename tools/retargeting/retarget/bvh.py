from scipy.spatial.transform import Rotation as R
import numpy as np
import re
from transforms3d.euler import euler2mat, mat2euler


class BvhJoint:
    def __init__(self, name, parent):
        self.name = name
        self.parent = parent
        self.offset = np.zeros(3)
        self.channels = []
        self.children = []

    def add_child(self, child):
        self.children.append(child)

    def __repr__(self):
        return self.name

    def position_animated(self):
        return any([x.endswith("position") for x in self.channels])

    def rotation_animated(self):
        return any([x.endswith("rotation") for x in self.channels])


class Bvh:
    def __init__(self):
        self.joints = {}
        self.root = None
        self.keyframes = None
        self.frames = 0
        self.fps = 0
        self.joint_channels = []
        self.joint_position = None
        self.joint_rotation = None
        self.idx_of = {}

    def _parse_hierarchy(self, text):
        lines = re.split("\\s*\\n+\\s*", text)
        joint_stack = []
        self.joint_channels = []
        joint_order = []
        for line in lines:
            words = re.split("\\s+", line)
            instruction = words[0]
            if instruction == "JOINT" or instruction == "ROOT":
                parent = joint_stack[-1] if instruction == "JOINT" else None
                joint = BvhJoint(words[1], parent)
                self.joints[joint.name] = joint
                if parent:
                    parent.add_child(joint)
                joint_stack.append(joint)
                if instruction == "ROOT":
                    self.root = joint
                joint_order.append(joint.name)
            elif instruction == "CHANNELS":
                for i in range(2, len(words)):
                    joint_stack[-1].channels.append(words[i])
                self.joint_channels.append(len(words) - 2)
            elif instruction == "OFFSET":
                for i in range(1, len(words)):
                    joint_stack[-1].offset[i - 1] = float(words[i])
            elif instruction == "End":
                joint = BvhJoint(joint_stack[-1].name + "_end", joint_stack[-1])
                joint_stack[-1].add_child(joint)
                joint_stack.append(joint)
                self.joints[joint.name] = joint
                joint_order.append(joint.name)
                self.joint_channels.append(0)
            elif instruction == "}":
                joint_stack.pop()
        self.joint_order = joint_order
        self.idx_of = {name: idx for idx, name in enumerate(self.joint_order)}

    def parse_motion(self, text):
        lines = re.split("\\s*\\n+\\s*", text)
        frame = 0
        for line in lines:
            if line == "":
                continue
            words = re.split("\\s+", line)
            if line.startswith("Frame Time:"):
                self.frame_time = float(words[2])
                self.fps = 1 / self.frame_time
                continue
            if line.startswith("Frames:"):
                self.frames = int(words[1])
                continue
            if self.keyframes is None:
                self.keyframes = np.empty((self.frames, len(words)), dtype=np.float32)
            for angle_index in range(len(words)):
                self.keyframes[frame, angle_index] = float(words[angle_index])
            frame += 1
        M = len(self.joint_order)
        N = self.frames
        self.joint_position = np.zeros((N, M, 3), dtype=np.float32)
        self.joint_rotation = np.zeros((N, M, 4), dtype=np.float32)
        self.joint_rotation[:, :, 3] = 1.0
        for f in range(N):
            idx = 0
            for j, joint_name in enumerate(self.joint_order):
                joint = self.joints[joint_name]
                channels = joint.channels
                pos = np.zeros(3)
                rot = np.zeros(3)
                for c in channels:
                    if c == "Xposition":
                        pos[0] = self.keyframes[f, idx]
                        idx += 1
                    elif c == "Yposition":
                        pos[1] = self.keyframes[f, idx]
                        idx += 1
                    elif c == "Zposition":
                        pos[2] = self.keyframes[f, idx]
                        idx += 1
                    elif c == "Xrotation":
                        rot[0] = self.keyframes[f, idx]
                        idx += 1
                    elif c == "Yrotation":
                        rot[1] = self.keyframes[f, idx]
                        idx += 1
                    elif c == "Zrotation":
                        rot[2] = self.keyframes[f, idx]
                        idx += 1
                self.joint_position[f, j, :] = (
                    pos if any((x.endswith("position") for x in channels)) else joint.offset
                )
                self.joint_rotation[f, j, :] = (
                    R.from_euler("XYZ", rot, degrees=True).as_quat()
                    if any((x.endswith("rotation") for x in channels))
                    else np.array([0, 0, 0, 1])
                )

    def parse_string(self, text):
        hierarchy, motion = text.split("MOTION")
        self._parse_hierarchy(hierarchy)
        self.parse_motion(motion)

    def _extract_rotation(self, frame_pose, index_offset, joint):
        local_rotation = np.zeros(3)
        for channel in joint.channels:
            if channel.endswith("position"):
                continue
            if channel == "Xrotation":
                local_rotation[0] = frame_pose[index_offset]
            elif channel == "Yrotation":
                local_rotation[1] = frame_pose[index_offset]
            elif channel == "Zrotation":
                local_rotation[2] = frame_pose[index_offset]
            else:
                raise Exception(f"Unknown channel {channel}")
            index_offset += 1
        local_rotation = np.deg2rad(local_rotation)
        M_rotation = np.eye(3)
        for channel in joint.channels:
            if channel.endswith("position"):
                continue
            if channel == "Xrotation":
                euler_rot = np.array([local_rotation[0], 0.0, 0.0])
            elif channel == "Yrotation":
                euler_rot = np.array([0.0, local_rotation[1], 0.0])
            elif channel == "Zrotation":
                euler_rot = np.array([0.0, 0.0, local_rotation[2]])
            else:
                raise Exception(f"Unknown channel {channel}")
            M_channel = euler2mat(*euler_rot)
            M_rotation = M_rotation.dot(M_channel)
        return (M_rotation, index_offset)

    def _extract_position(self, joint, frame_pose, index_offset):
        offset_position = np.zeros(3)
        for channel in joint.channels:
            if channel.endswith("rotation"):
                continue
            if channel == "Xposition":
                offset_position[0] = frame_pose[index_offset]
            elif channel == "Yposition":
                offset_position[1] = frame_pose[index_offset]
            elif channel == "Zposition":
                offset_position[2] = frame_pose[index_offset]
            else:
                raise Exception(f"Unknown channel {channel}")
            index_offset += 1
        return (offset_position, index_offset)

    def _recursive_apply_frame(self, joint, frame_pose, index_offset, p, r, M_parent, p_parent):
        if joint.position_animated():
            offset_position, index_offset = self._extract_position(joint, frame_pose, index_offset)
        else:
            offset_position = np.zeros(3)
        if len(joint.channels) == 0:
            joint_index = list(self.joints.values()).index(joint)
            p[joint_index] = p_parent + M_parent.dot(joint.offset)
            r[joint_index] = mat2euler(M_parent)
            return index_offset
        if joint.rotation_animated():
            M_rotation, index_offset = self._extract_rotation(frame_pose, index_offset, joint)
        else:
            M_rotation = np.eye(3)
        M = M_parent.dot(M_rotation)
        position = p_parent + M_parent.dot(joint.offset) + offset_position
        rotation = np.rad2deg(mat2euler(M))
        joint_index = list(self.joints.values()).index(joint)
        p[joint_index] = position
        r[joint_index] = rotation
        for c in joint.children:
            index_offset = self._recursive_apply_frame(
                c, frame_pose, index_offset, p, r, M, position
            )
        return index_offset

    def frame_pose(self, frame):
        p = np.empty((len(self.joints), 3))
        r = np.empty((len(self.joints), 3))
        frame_pose = self.keyframes[frame]
        M_parent = np.zeros((3, 3))
        M_parent[0, 0] = 1
        M_parent[1, 1] = 1
        M_parent[2, 2] = 1
        self._recursive_apply_frame(self.root, frame_pose, 0, p, r, M_parent, np.zeros(3))
        return (p, r)

    def joint_names(self):
        return self.joints.keys()

    def parse_file(self, path):
        with open(path, "r") as f:
            self.parse_string(f.read())

    def __repr__(self):
        return f"BVH {len(self.joints.keys())} joints, {self.frames} frames"
