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
        return any([x.endswith('position') for x in self.channels])

    def rotation_animated(self):
        return any([x.endswith('rotation') for x in self.channels])


class Bvh:
    def __init__(self):
        self.joints = {}
        self.root = None
        self.keyframes = None
        self.frames = 0
        self.fps = 0
        self.joint_channels = []  # number of channels per joint
        self.joint_position = None  # (N, M, 3) local positions
        self.joint_rotation = None  # (N, M, 4) local rotations (quaternion)
        self.idx_of = {}  # maps joint name to index

    def _parse_hierarchy(self, text):
        lines = re.split('\\s*\\n+\\s*', text)

        joint_stack = []
        self.joint_channels = []  # reset for new hierarchy
        joint_order = []

        for line in lines:
            words = re.split('\\s+', line)
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
            elif instruction == '}':
                joint_stack.pop()
        self.joint_order = joint_order
        # Precompute idx_of dict
        self.idx_of = {name: idx for idx, name in enumerate(self.joint_order)}

    def _add_pose_recursive(self, joint, offset, poses):
        pose = joint.offset + offset
        poses.append(pose)

        for c in joint.children:
            self._add_pose_recursive(c, pose, poses)

    def plot_hierarchy(self):
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import axes3d, Axes3D
        poses = []
        self._add_pose_recursive(self.root, np.zeros(3), poses)
        pos = np.array(poses)

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(pos[:, 0], pos[:, 2], pos[:, 1])
        ax.set_xlim(-30, 30)
        ax.set_ylim(-30, 30)
        ax.set_zlim(-30, 30)
        plt.show()

    def parse_motion(self, text):
        lines = re.split('\\s*\\n+\\s*', text)

        frame = 0
        for line in lines:
            if line == '':
                continue
            words = re.split('\\s+', line)

            if line.startswith("Frame Time:"):
                self.fps = round(1 / float(words[2]))
                continue
            if line.startswith("Frames:"):
                self.frames = int(words[1])
                continue

            if self.keyframes is None:
                self.keyframes = np.empty((self.frames, len(words)), dtype=np.float32)

            for angle_index in range(len(words)):
                self.keyframes[frame, angle_index] = float(words[angle_index])

            frame += 1

        # Fill local joint_position and joint_rotation (quaternion)
        M = len(self.joint_order)
        N = self.frames
        self.joint_position = np.zeros((N, M, 3), dtype=np.float32)
        self.joint_rotation = np.zeros((N, M, 4), dtype=np.float32)
        self.joint_rotation[:, :, 3] = 1.0  # w=1 for identity
        for f in range(N):
            idx = 0
            for j, joint_name in enumerate(self.joint_order):
                joint = self.joints[joint_name]
                channels = joint.channels
                pos = np.zeros(3)
                rot = np.zeros(3)
                for c in channels:
                    if c == 'Xposition':
                        pos[0] = self.keyframes[f, idx]
                        idx += 1
                    elif c == 'Yposition':
                        pos[1] = self.keyframes[f, idx]
                        idx += 1
                    elif c == 'Zposition':
                        pos[2] = self.keyframes[f, idx]
                        idx += 1
                    elif c == 'Xrotation':
                        rot[0] = self.keyframes[f, idx]
                        idx += 1
                    elif c == 'Yrotation':
                        rot[1] = self.keyframes[f, idx]
                        idx += 1
                    elif c == 'Zrotation':
                        rot[2] = self.keyframes[f, idx]
                        idx += 1
                self.joint_position[f, j, :] = pos if any(x.endswith('position') for x in channels) else joint.offset
                from scipy.spatial.transform import Rotation as R
                self.joint_rotation[f, j, :] = R.from_euler('XYZ', rot, degrees=True).as_quat() if any(x.endswith('rotation') for x in channels) else np.array([0, 0, 0, 1])
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
                euler_rot = np.array([local_rotation[0], 0., 0.])
            elif channel == "Yrotation":
                euler_rot = np.array([0., local_rotation[1], 0.])
            elif channel == "Zrotation":
                euler_rot = np.array([0., 0., local_rotation[2]])
            else:
                raise Exception(f"Unknown channel {channel}")

            M_channel = euler2mat(*euler_rot)
            M_rotation = M_rotation.dot(M_channel)

        return M_rotation, index_offset

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

        return offset_position, index_offset

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
            index_offset = self._recursive_apply_frame(c, frame_pose, index_offset, p, r, M, position)

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

        return p, r

    def all_frame_poses(self):
        p = np.empty((self.frames, len(self.joints), 3))
        r = np.empty((self.frames, len(self.joints), 3))

        for frame in range(len(self.keyframes)):
            p[frame], r[frame] = self.frame_pose(frame)

        return p, r

    def _plot_pose(self, p, r, fig=None, ax=None):
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import axes3d, Axes3D
        if fig is None:
            fig = plt.figure()
        if ax is None:
            ax = fig.add_subplot(111, projection='3d')

        ax.cla()
        ax.scatter(p[:, 0], -p[:, 2], p[:, 1])
        ax.set_xlim(-100, 100)
        ax.set_ylim(-100, 100)
        ax.set_zlim(-100, 100)

        plt.draw()
        plt.pause(0.001)

    def plot_frame(self, frame, fig=None, ax=None):
        p, r = self.frame_pose(frame)
        self._plot_pose(p, r, fig, ax)

    def joint_names(self):
        return self.joints.keys()

    def parse_file(self, path):
        with open(path, 'r') as f:
            self.parse_string(f.read())

    def plot_all_frames(self):
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import axes3d, Axes3D
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        for i in range(self.frames):
            self.plot_frame(i, fig, ax)

    def __repr__(self):
        return f"BVH {len(self.joints.keys())} joints, {self.frames} frames"

import numpy as np
import copy
from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp

def load_meta_data(bvh_path):
    with open(bvh_path, 'r') as f:
        channels = []
        joints = []
        joint_parents = []
        joint_offsets = []
        end_sites = []

        parent_stack = [None]
        end_site_flag = False
        for line in f:
            if 'ROOT' in line or 'JOINT' in line:
                joints.append(line.split()[-1])
                joint_parents.append(parent_stack[-1])
                channels.append('')
                joint_offsets.append([0, 0, 0])

            elif 'End Site' in line:
                # end_sites.append(len(joints))
                # joints.append(parent_stack[-1] + '_end')
                # joint_parents.append(parent_stack[-1])
                # channels.append('')
                # joint_offsets.append([0, 0, 0])
                end_site_flag = True
                pass

            elif '{' in line:
                parent_stack.append(joints[-1])

            elif '}' in line:
                parent_stack.pop()
                end_site_flag = False

            elif 'OFFSET' in line:
                if end_site_flag:
                    pass
                else:
                    joint_offsets[-1] = np.array([float(x) for x in line.split()[-3:]]).reshape(1,3)

            elif 'CHANNELS' in line:
                trans_order = []
                rot_order = []
                for token in line.split():
                    if 'position' in token:
                        trans_order.append(token[0])

                    if 'rotation' in token:
                        rot_order.append(token[0])

                channels[-1] = ''.join(trans_order)+ ''.join(rot_order)

            elif 'Frame Time:' in line:
                break

    joint_parents = [-1]+ [joints.index(i) for i in joint_parents[1:]]
    channels = [len(i) for i in channels]
    return joints, joint_parents, channels, joint_offsets

def load_motion_data(bvh_path):
    with open(bvh_path, 'r') as f:
        lines = f.readlines()
        for i in range(len(lines)):
            if lines[i].startswith('Frame Time'):
                break
        motion_data = []
        for line in lines[i+1:]:
            data = [float(x) for x in line.split()]
            if len(data) == 0:
                break
            motion_data.append(np.array(data).reshape(1,-1))
        motion_data = np.concatenate(motion_data, axis=0)
    return motion_data



'''
N denotes frames and M denotes joints.
position and rotation denote local transforms.
translation and orientation denote global transforms.
'''

class BVHMotion():
    def __init__(self, bvh_file_name = None) -> None:


        self.joint_name = []
        self.joint_channel = []
        self.joint_parent = []
        self.num_frames = 0


        self.joint_position = None
        self.joint_rotation = None

        if bvh_file_name is not None:
            self.load_motion(bvh_file_name)
        pass


    def load_motion(self, bvh_file_path):
        '''
            Load a BVH file and initialize metadata and local transforms.
        '''
        self.joint_name, self.joint_parent, self.joint_channel, joint_offset = \
            load_meta_data(bvh_file_path)

        motion_data = load_motion_data(bvh_file_path)


        self.joint_position = np.zeros((motion_data.shape[0], len(self.joint_name), 3))
        self.joint_rotation = np.zeros((motion_data.shape[0], len(self.joint_name), 4))
        self.joint_rotation[:,:,3] = 1.0

        cur_channel = 0
        for i in range(len(self.joint_name)):
            if self.joint_channel[i] == 0:
                self.joint_position[:,i,:] = joint_offset[i].reshape(1,3)
                continue
            elif self.joint_channel[i] == 3:
                self.joint_position[:,i,:] = joint_offset[i].reshape(1,3)
                rotation = motion_data[:, cur_channel:cur_channel+3]
            elif self.joint_channel[i] == 6:
                self.joint_position[:, i, :] = motion_data[:, cur_channel:cur_channel+3]
                rotation = motion_data[:, cur_channel+3:cur_channel+6]
            self.joint_rotation[:, i, :] = R.from_euler('XYZ', rotation,degrees=True).as_quat()
            cur_channel += self.joint_channel[i]
        self.num_frames = self.joint_position.shape[0]
        return

    def batch_forward_kinematics(self, joint_position = None, joint_rotation = None, frame_id_list = None, root_pos = None, root_quat = None):
        '''
        Run batched forward kinematics from the stored metadata.
        joint_position: (N,M,3)ndarray,
        joint_rotation: (N,M,4)ndarray,
        frame_id_list: list, frame IDs to retain, or None.
        root_pos: (3,)ndarray, root joint
        root_quat: (4,)ndarray, root joint
        '''
        if joint_position is None:
            joint_position = self.joint_position.copy()
        if joint_rotation is None:
            joint_rotation = self.joint_rotation.copy()

        if frame_id_list is not None:
            joint_position = joint_position[frame_id_list]
            joint_rotation = joint_rotation[frame_id_list]
        if root_pos is not None:
            root_pos = root_pos.reshape(1,3).repeat(joint_position.shape[0], axis=0)
            joint_position[:,0,:] = root_pos
        if root_quat is not None:
            root_quat = root_quat.reshape(1,4).repeat(joint_position.shape[0], axis=0)
            joint_rotation[:,0,:] = root_quat

        joint_translation = np.zeros_like(joint_position)
        joint_orientation = np.zeros_like(joint_rotation)
        joint_orientation[:,:,3] = 1.0




        for i in range(len(self.joint_name)):
            pi = self.joint_parent[i]
            parent_orientation = R.from_quat(joint_orientation[:,pi,:])
            joint_translation[:, i, :] = joint_translation[:, pi, :] + \
                parent_orientation.apply(joint_position[:, i, :])
            joint_orientation[:, i, :] = (parent_orientation * R.from_quat(joint_rotation[:, i, :])).as_quat()
        return joint_translation, joint_orientation


    def adjust_joint_name(self, target_joint_name):
        '''
        Reorder joints to match target_joint_name.
        '''
        idx = [self.joint_name.index(joint_name) for joint_name in target_joint_name]
        idx_inv = [target_joint_name.index(joint_name) for joint_name in self.joint_name]
        self.joint_name = [self.joint_name[i] for i in idx]
        self.joint_parent = [idx_inv[self.joint_parent[i]] for i in idx]
        self.joint_parent[0] = -1
        self.joint_channel = [self.joint_channel[i] for i in idx]
        self.joint_position = self.joint_position[:,idx,:]
        self.joint_rotation = self.joint_rotation[:,idx,:]
        pass

    def raw_copy(self):
        '''
        Return a copy.
        '''
        return copy.deepcopy(self)

    @property
    def motion_length(self):
        return self.joint_position.shape[0]


    def sub_sequence(self, start, end):
        '''
        Return a subsequence.
        start:
        end:
        '''
        res = self.raw_copy()
        res.joint_position = res.joint_position[start:end,:,:]
        res.joint_rotation = res.joint_rotation[start:end,:,:]
        return res

    def append(self, other):
        '''
        Append another motion sequence.
        '''
        other = other.raw_copy()
        other.adjust_joint_name(self.joint_name)
        self.joint_position = np.concatenate((self.joint_position, other.joint_position), axis=0)
        self.joint_rotation = np.concatenate((self.joint_rotation, other.joint_rotation), axis=0)
        pass

    def translation_and_rotation(self, frame_num, target_translation_xz, target_facing_direction_xz):
        '''
        Compute adjusted joint positions and rotations.
        Align the root translation and horizontal facing direction at frame_num.
        frame_num: int
        target_translation_xz: (2,)ndarray
        target_faceing_direction_xz: A `(2,)` array describing the target facing
            direction in the horizontal plane. The second component is the z axis.
        '''

        res = self.raw_copy()

        offset = target_translation_xz - res.joint_position[frame_num, 0, [0,2]]
        res.joint_position[:, 0, [0,2]] += offset

        Ry, Rxz = self.decompose_rotation_with_yaxis(res.joint_rotation[frame_num, 0, :])
        axis = np.array([0,1,0]).reshape(3,)
        angle = np.arctan2(target_facing_direction_xz[0], target_facing_direction_xz[1])
        desired_rot = R.from_rotvec(axis * angle)
        rot = R.from_quat(res.joint_rotation[:, 0, :])
        rot_diff = desired_rot*(R.from_quat(Ry).inv())
        res.joint_rotation[:, 0, :] = (rot_diff*rot).as_quat()

        pos_offset = res.joint_position[:, 0, :] - res.joint_position[frame_num, 0, :]
        rotated_pos_offset = rot_diff.apply(pos_offset)
        res.joint_position[:,0,:] = rotated_pos_offset + res.joint_position[frame_num, 0]
        return res


if __name__ == '__main__':
    # create Bvh parser
    anim = Bvh()
    # parser file
    anim.parse_file("motion.bvh")

    # draw the skeleton in T-pose
    # anim.plot_hierarchy()

    # extract single frame pose: axis0=joint, axis1=positionXYZ/rotationXYZ
    p, r = anim.frame_pose(0)

    # extract all poses: axis0=frame, axis1=joint, axis2=positionXYZ/rotationXYZ
    all_p, all_r = anim.all_frame_poses()

    # print all joints, their positions and orientations

    for _p, _r, _j in zip(p, r, anim.joint_names()):
        print(f"{_j}: p={_p}, r={_r}")
    print("max of x in p: ", np.max(all_p[:, :, 0]))
    print("max of y in p: ", -np.min(all_p[:, :, 2]))
    print("max of z in p: ", np.max(all_p[:, :, 1]))
    print("min of x in p: ", np.min(all_p[:, :, 0]))
    print("min of y in p: ", -np.max(all_p[:, :, 2]))
    print("min of z in p: ", np.min(all_p[:, :, 1]))

    print("max of x in p in frame 0: ", np.max(all_p[0, :, 0]))
    print("max of y in p in frame 0: ", -np.min(all_p[0, :, 2]))
    print("max of z in p in frame 0: ", np.max(all_p[0, :, 1]))
    print("min of x in p in frame 0: ", np.min(all_p[0, :, 0]))
    print("min of y in p in frame 0: ", -np.max(all_p[0, :, 2]))
    print("min of z in p in frame 0: ", np.min(all_p[0, :, 1]))
    # draw the skeleton for the given frame
    # anim.plot_frame(22)

    # show full animation
    anim.plot_all_frames()
