import os
import math
import torch
import torch.nn as nn
import numpy as np
from einops import repeat
from sklearn.preprocessing import StandardScaler
import time as time_test

from typing import List, Tuple, Optional
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.figure import Figure
import matplotlib.image as mpimg
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Line3D, Text3D

from .writers import BVHWriter

def print_tensor_features(tensor, name="Tensor"):
    print(f"=== Features of {name} ===")
    print(f"Shape (shape): {tensor.shape}")
    print(f"Size (size): {tensor.size()}")
    print(f"Number of elements (numel): {tensor.numel()}")
    print(f"Number of dimensions (dim): {tensor.dim()}")
    print(f"Data type (dtype): {tensor.dtype}")
    print(f"Device (device): {tensor.device}")
    print(f"Is CUDA (is_cuda): {tensor.is_cuda}")
    print(f"Requires gradient (requires_grad): {tensor.requires_grad}")
    print(f"Is leaf node (is_leaf): {tensor.is_leaf}")
    print(f"Gradient (grad): {tensor.grad}")
    print(f"Is contiguous (is_contiguous): {tensor.is_contiguous()}")
    print(f"Stride (stride): {tensor.stride()}")
    print(f"Storage offset (storage_offset): {tensor.storage_offset()}")
    print(f"Is sparse (is_sparse): {tensor.is_sparse}")
    print(f"Layout (layout): {tensor.layout}")
    if tensor.names is not None:
        print(f"Names (names): {tensor.names}")
    else:
        print("Names (names): None")
    print("============================\n")


def extract(a, t, x_shape):
    b, *_ = t.shape
    out = a.gather(-1, t)
    out = out.reshape(b, *((1,) * (len(x_shape) - 1)))
    return out


def noise_like(shape, device, repeat=False):
    repeat_noise = lambda: torch.randn((1, *shape[1:]), device=device).repeat(shape[0], *((1,) * (len(shape) - 1)))
    noise = lambda: torch.randn(shape, device=device)
    return repeat_noise() if repeat else noise()


def make_ddim_timesteps(ddim_discr_method, num_ddim_timesteps, num_ddpm_timesteps, verbose=True):
    if ddim_discr_method == 'uniform':
        c = num_ddpm_timesteps // num_ddim_timesteps
        ddim_timesteps = np.asarray(list(range(0, num_ddpm_timesteps, c)))
    elif ddim_discr_method == 'quad':
        ddim_timesteps = ((np.linspace(0, np.sqrt(num_ddpm_timesteps * .8), num_ddim_timesteps)) ** 2).astype(int)
    else:
        raise NotImplementedError(f'There is no ddim discretization method called "{ddim_discr_method}"')

    # assert ddim_timesteps.shape[0] == num_ddim_timesteps
    # add one to get the final alpha values right (the ones from first scale to data during sampling)
    steps_out = ddim_timesteps + 1
    if verbose:
        print(f'Selected timesteps for ddim sampler: {steps_out}')
    return steps_out


def make_ddim_sampling_parameters(alphacums, ddim_timesteps, eta, verbose=True):
    # select alphas for computing the variance schedule
    alphas = alphacums[ddim_timesteps]
    alphas_prev = np.asarray([alphacums[0]] + alphacums[ddim_timesteps[:-1]].tolist())

    # according the the formula provided in https://arxiv.org/abs/2010.02502


    sigmas = eta * np.sqrt((1 - alphas_prev) / (1 - alphas) * (1 - alphas / alphas_prev))
    if verbose:
        print(f'Selected alphas for ddim sampler: a_t: {alphas}; a_(t-1): {alphas_prev}')
        print(f'For the chosen value of eta, which is {eta}, '
              f'this results in the following sigma_t schedule for ddim sampler {sigmas}')
    return sigmas, alphas, alphas_prev


def fit_and_standardize(data):
    shape = data.shape
    print(f"fit_and_standarize data.shape: {data.shape}")
    flat = data.copy().reshape((shape[0] * shape[1], shape[2]))
    scaler = StandardScaler().fit(flat)
    scaled = scaler.transform(flat).reshape(shape)
    return scaled, scaler


def standardize(data, scaler):
    shape = data.shape
    print(f"standarize data.shape: {data.shape}")
    flat = data.copy().reshape((shape[0] * shape[1], shape[2]))  # StandardScaler expected <= 2
    scaled = scaler.transform(flat).reshape(shape)
    return scaled


def inv_standardize(data, scaler):
    shape = data.shape
    flat = data.copy().reshape((shape[0] * shape[1], shape[2]))
    scaled = scaler.inverse_transform(flat).reshape(shape)
    return scaled


def standardize_with_gradient(data, scaler):
    # Ensure input is a torch tensor
    if not isinstance(data, torch.Tensor):
        raise ValueError("Input data must be a torch.Tensor.")

    # Extract scaler parameters
    mean = torch.tensor(scaler.mean_, dtype=data.dtype, device=data.device)
    scale = torch.tensor(scaler.scale_, dtype=data.dtype, device=data.device)

    # Perform inverse standardization
    shape = data.shape
    # flat = data.clone().reshape((shape[0] * shape[1], shape[2]))
    scaled = (data - mean) / scale
    # return scaled.reshape(shape)
    return scaled


def inv_standardize_with_gradient(data, mean, scale):
    """
    Perform inverse standardization while preserving gradient information using PyTorch.

    Parameters:
        data (torch.Tensor): The input tensor with gradients.
        scaler: A scaler object that provides 'mean_' and 'scale_' (e.g., from sklearn).

    Returns:
        torch.Tensor: The inverse standardized tensor with preserved gradients.
    """
    # Ensure input is a torch tensor
    if not isinstance(data, torch.Tensor):
        raise ValueError("Input data must be a torch.Tensor.")

    # # Extract scaler parameters
    # mean = torch.tensor(scaler.mean_, dtype=data.dtype, device=data.device)
    # scale = torch.tensor(scaler.scale_, dtype=data.dtype, device=data.device)

    # Perform inverse standardization
    shape = data.shape
    if not data.is_contiguous():
        data = data.contiguous()
    flat = data.reshape((shape[0] * shape[1], shape[2]))
    scaled = flat * scale + mean
    # return scaled.reshape(shape)
    # print_tensor_features(scaled)

    return scaled


def inv_standardize_with_gradient_2(data, scaler):
    """
    Perform inverse standardization while preserving gradient information using PyTorch.

    Parameters:
        data (torch.Tensor): The input tensor with gradients.
        scaler: A scaler object that provides 'mean_' and 'scale_' (e.g., from sklearn).

    Returns:
        torch.Tensor: The inverse standardized tensor with preserved gradients.
    """
    # Ensure input is a torch tensor
    if not isinstance(data, torch.Tensor):
        raise ValueError("Input data must be a torch.Tensor.")

    # Extract scaler parameters
    mean = torch.tensor(scaler.mean_, dtype=data.dtype, device=data.device)
    scale = torch.tensor(scaler.scale_, dtype=data.dtype, device=data.device)

    # Perform inverse standardization
    shape = data.shape
    if not data.is_contiguous():
        data = data.contiguous()
    flat = data.reshape((shape[0] * shape[1], shape[2]))
    scaled = flat * scale + mean
    # return scaled.reshape(shape)
    # print_tensor_features(scaled)

    return scaled


def expmap_to_quaternion(expmap: torch.Tensor) -> torch.Tensor:
    """
    Convert 3D rotation in exponential map (axis-angle) format to quaternion representation.
    This operation is differentiable and compatible with PyTorch autograd.

    Args:
        expmap (torch.Tensor): A tensor of shape (..., 3), where the last dimension represents
                               the exponential map [x, y, z] (axis-angle).

    Returns:
        torch.Tensor: A tensor of shape (..., 4), where the last dimension represents the quaternion [w, x, y, z].
    """
    # Compute the rotation angle (theta)
    theta = torch.norm(expmap, dim=-1, keepdim=True)  # Shape: (..., 1)

    # Avoid division by zero for small angles (use Taylor expansion near 0)
    small_angle = theta < 1e-6
    half_theta = theta / 2.0

    # Compute sin(theta/2) and cos(theta/2)
    sin_half_theta = torch.where(
        small_angle, 0.5 - (theta ** 2) / 48.0, torch.sin(half_theta) / (theta + 1e-8)
    ) * theta
    cos_half_theta = torch.where(small_angle, 1.0, torch.cos(half_theta))

    # Normalize the axis of rotation (expmap / theta)
    axis = torch.where(small_angle, expmap, expmap / (theta + 1e-8))

    # Compute the quaternion [w, x, y, z]
    w = cos_half_theta.squeeze(-1)  # Shape: (...)
    xyz = axis * sin_half_theta  # Shape: (..., 3)

    # Concatenate w and xyz to form the quaternion
    quaternion = torch.cat([w.unsqueeze(-1), xyz], dim=-1)  # Shape: (..., 4)

    return quaternion


def quaternion_to_expmap(quaternion: torch.Tensor) -> torch.Tensor:
    """
    Convert quaternion to exponential map (axis-angle) representation.
    This operation is differentiable and compatible with PyTorch autograd.

    Args:
        quaternion (torch.Tensor): A tensor of shape (..., 4), where the last dimension represents
                                   the quaternion [w, x, y, z].

    Returns:
        torch.Tensor: A tensor of shape (..., 3), where the last dimension represents the exponential map [x, y, z].
    """
    if quaternion.shape[-1] != 4:
        raise ValueError("Input must have shape (..., 4), representing quaternions as [w, x, y, z].")

    epsilon = 1e-6

    # Normalize the quaternion to ensure it is a unit quaternion
    quaternion = quaternion / torch.norm(quaternion, dim=-1, keepdim=True)

    # Extract the scalar (w) and vector (x, y, z) parts
    w, xyz = quaternion[..., 0], quaternion[..., 1:]  # w: (...), xyz: (..., 3)

    # Compute the angle (theta)
    theta = 2.0 * torch.acos(w.clamp(-1.0+epsilon, 1.0-epsilon))  # Clamp w to avoid numerical issues

    # Compute the axis of rotation
    # print(w, xyz)
    # nan_positions = torch.isnan(w).nonzero(as_tuple=True)
    # print(f"Positions of NaN values: {nan_positions}")
    # abs_gt_one_positions = (torch.abs(w) > 1).nonzero(as_tuple=True)
    # print(f"Positions of values with absolute value greater than 1: {abs_gt_one_positions}")

    # print(w)
    # print(w.clamp(-1.0+epsilon, 1.0-epsilon)**2)
    sin_half_theta = torch.sqrt(1.0 - w.clamp(-1.0+epsilon, 1.0-epsilon)**2).unsqueeze(-1)  # Shape: (..., 1)
    # print(sin_half_theta)
    axis = torch.where(sin_half_theta > 1e-6, xyz / (sin_half_theta + epsilon), xyz)  # Handle small angles

    # Compute the exponential map: theta * axis
    expmap = axis * theta.unsqueeze(-1)

    # Handle small angles (theta ~ 0)
    expmap = torch.where(theta.unsqueeze(-1) > 1e-6, expmap, xyz.new_zeros(xyz.shape))

    return expmap


def angle_axis_to_quat(angles: torch.Tensor, axis: torch.Tensor) -> torch.Tensor:
    """
    Create quaternions (wxyz format) from rotation angles and axes using PyTorch tensors.

    Args:
        angles (torch.Tensor): Rotation angles in radians, shape (...,).
        axis (torch.Tensor): Rotation axes, shape (..., 3).

    Returns:
        torch.Tensor: Quaternions in wxyz format, shape (..., 4).
    """
    # Normalize the axis to unit vectors
    axis = axis / (torch.sqrt(torch.sum(axis**2, dim=-1, keepdim=True)) + 1e-10)

    # Compute sine and cosine of half angles
    sines = torch.sin(angles / 2.0).unsqueeze(-1)  # Shape (..., 1)
    cosines = torch.cos(angles / 2.0).unsqueeze(-1)  # Shape (..., 1)

    # Concatenate to form quaternion: [w, x, y, z]
    quaternions = torch.cat([cosines, axis * sines], dim=-1)  # Shape (..., 4)

    return quaternions


# x-forward
def quat_to_angle_axis_y(quaternions: torch.Tensor) -> torch.Tensor:

    x_forward = torch.tensor([[1, 0, 0]]).to(quaternions.device).to(quaternions.dtype)

    forward_x = rotate_vector_by_quaternion(x_forward, quaternions)
    forward_x[:, 1] = 0

    forward_x = forward_x / torch.sqrt((forward_x ** 2).sum(axis=-1, keepdim=True))

    def between(v0s, v1s):
        a = torch.cross(v0s, v1s, dim=-1)
        w = torch.sqrt((v0s**2).sum(dim=-1) * (v1s**2).sum(dim=-1)) + (v0s * v1s).sum(dim=-1)
        quat = torch.cat([w.unsqueeze(-1), a], dim=-1)
        quat = quat / (torch.sqrt(torch.sum(quat**2, dim=-1, keepdim=True)) + 1e-10)
        return quat

    target_x = x_forward.expand_as(forward_x)
    rotation_x = between(target_x, forward_x)

    return rotation_x

def quat_to_angle_axis_z(quaternions: torch.Tensor) -> torch.Tensor:
    y_forward = torch.tensor([[0, 1, 0]]).to(quaternions.device).to(quaternions.dtype)

    forward_y = rotate_vector_by_quaternion(y_forward, quaternions)
    forward_y[:, 2] = 0

    forward_y = forward_y / torch.sqrt((forward_y ** 2).sum(axis=-1, keepdim=True) + 1e-10)

    def between(v0s, v1s):
        a = torch.cross(v0s, v1s, dim=-1)
        w = torch.sqrt((v0s**2).sum(dim=-1) * (v1s**2).sum(dim=-1)) + (v0s * v1s).sum(dim=-1)
        quat = torch.cat([w.unsqueeze(-1), a], dim=-1)
        quat = quat / (torch.sqrt(torch.sum(quat**2, dim=-1, keepdim=True)) + 1e-10)
        return quat

    target_y = y_forward.expand_as(forward_y)
    rotation_z = between(target_y, forward_y)

    return rotation_z


def quat_to_angle_axis_all(quaternions: torch.Tensor) -> torch.Tensor:
    """
    Convert quaternions to rotation angles around a specific plane.

    Args:
        quaternions (torch.Tensor): A tensor of shape (..., 4), representing quaternions [w, x, y, z].
        forward (str): Axis to project forward direction, default is 'z'.
        plane (str): Two axes defining the plane of rotation, default is 'xz'.

    Returns:
        torch.Tensor: Rotation angles in radians, shape (...,).
    """
    if quaternions.shape[-1] != 4:
        raise ValueError("Input must have shape (..., 4), representing quaternions as [w, x, y, z].")

    # Normalize the quaternion to ensure it is a unit quaternion
    quaternions = quaternions / (torch.norm(quaternions, dim=-1, keepdim=True) + 1e-10)

    # Create a direction vector aligned with the `forward` axis
    forward_vector_z = torch.zeros_like(quaternions[..., 1:])
    forward_vector_z[..., 2] = 1.0  # Set the forward axis to 1.0
    forward_vector_x = torch.zeros_like(quaternions[..., 1:])
    forward_vector_x[..., 0] = 1.0  # Set the forward axis to 1.0

    # Rotate the forward vector using the quaternion
    rotated_vector_z = rotate_vector_by_quaternion(forward_vector_z, quaternions)
    rotated_vector_x = rotate_vector_by_quaternion(forward_vector_x, quaternions)

    # Extract the components of the rotated vector in the specified plane
    x_component_z = rotated_vector_z[..., 0]
    y_component_z = rotated_vector_z[..., 1]
    z_component_z = rotated_vector_z[..., 2]
    x_angles = torch.atan2(y_component_z, z_component_z)
    y_angles = torch.atan2(x_component_z, z_component_z)

    x_component_x = rotated_vector_x[..., 0]
    y_component_x = rotated_vector_x[..., 1]
    z_component_x = rotated_vector_x[..., 2]
    z_angles = torch.atan2(y_component_x, x_component_x)
    y_angles_2 = torch.atan2(z_component_x, x_component_x)

    print(y_angles, y_angles_2)

    angles = torch.stack([x_angles, y_angles, z_angles], dim=-1)
    return angles


def quaternion_multiply(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
    """
    Multiply two quaternions in wxyz format.

    Args:
        q1 (torch.Tensor): First quaternion tensor in wxyz format, shape (..., 4).
        q2 (torch.Tensor): Second quaternion tensor in wxyz format, shape (..., 4).

    Returns:
        torch.Tensor: Resulting quaternion tensor in wxyz format, shape (..., 4).
    """
    # Extract components
    w1, x1, y1, z1 = q1[..., 0], q1[..., 1], q1[..., 2], q1[..., 3]
    w2, x2, y2, z2 = q2[..., 0], q2[..., 1], q2[..., 2], q2[..., 3]

    # Compute the product
    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2

    # Stack the results into a single tensor in wxyz format
    return torch.stack([w, x, y, z], dim=-1)


def rotate_vector_by_quaternion(v: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    """
    Rotate a 3D vector by a quaternion in wxyz format.

    Args:
        v (torch.Tensor): Input vector(s), shape (..., 3).
        q (torch.Tensor): Quaternion(s) in wxyz format, shape (..., 4).

    Returns:
        torch.Tensor: Rotated vector(s), shape (..., 3).
    """
    # Extract quaternion components
    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]

    # Compute intermediate terms
    t2 = w * x
    t3 = w * y
    t4 = w * z
    t5 = -x * x
    t6 = x * y
    t7 = x * z
    t8 = -y * y
    t9 = y * z
    t10 = -z * z

    # Extract input vector components
    vx, vy, vz = v[..., 0], v[..., 1], v[..., 2]

    # Apply rotation to the vector
    rx = 2 * ((t8 + t10) * vx + (t6 - t4) * vy + (t3 + t7) * vz) + vx
    ry = 2 * ((t4 + t6) * vx + (t5 + t10) * vy + (t9 - t2) * vz) + vy
    rz = 2 * ((t7 - t3) * vx + (t2 + t9) * vy + (t5 + t8) * vz) + vz

    # Combine results into a single tensor
    return torch.stack([rx, ry, rz], dim=-1)


def process_skeleton_info(skeleton_info, joints_name):
    """
    Select specific joints and their related information from skeleton_info.

    Args:
        skeleton_info (dict): Full skeleton information containing parent, offsets, order, and children for each joint.
        joints_name (list): List of joint names to extract.

    Returns:
        dict: Selected skeleton information containing:
            - joint_chain: List of parent indices for the selected joints.
            - offsets: List of offsets for the selected joints.
            - orders: List of rotation orders for the selected joints.
            - joint_names: List of selected joint names.
    """
    name_to_index = {name: idx for idx, name in enumerate(joints_name)}

    # Initialize storage for the selected joints
    joint_chain = []
    offsets = []
    orders = []
    selected_joint_names = []

    for joint_name in joints_name:
        if joint_name not in skeleton_info:
            raise ValueError(f"Joint '{joint_name}' not found in skeleton_info.")

        # Append joint name
        selected_joint_names.append(joint_name)

        # Get joint info
        joint_data = skeleton_info[joint_name]

        # Parent index (only include parent if it's in the selected joints)
        parent_name = joint_data['parent']
        parent_index = name_to_index[parent_name] if parent_name in joints_name else -1
        joint_chain.append(parent_index)

        # Offset
        offsets.append(torch.tensor(joint_data['offsets'], dtype=torch.float32))

        # Rotation order
        orders.append(joint_data['order'])

    return {
        'joint_chain': joint_chain,
        'offsets': torch.stack(offsets),  # Shape: [num_selected_joints, 3]
        'orders': orders,
        'joint_names': selected_joint_names
    }


def visualize_joints(num_joints, joint_chain, pos_3d, render_joint_id: bool = False,
                        save_path: Optional[str] = './test_files/pymotionlib4gesture/test1.gif'):
    """
    Visualize the input 3d joint position by matplotlib.
    """

    joint_pairs = [(parent, child) for child, parent in enumerate(joint_chain) if parent != -1]
    if isinstance(pos_3d, torch.Tensor):
        pos_3d = pos_3d.cpu().numpy()
    elif isinstance(pos_3d, np.ndarray):
        pass
    else:
        raise ValueError("Input pos_3d must be a torch.Tensor or np.ndarray.")

    pos_3d_mapped = pos_3d.copy()
    pos_3d_mapped[..., 0] = pos_3d[..., 0]  # Keep x as is
    pos_3d_mapped[..., 1] = -pos_3d[..., 2]  # z becomes vertical y
    pos_3d_mapped[..., 2] = pos_3d[..., 1]  # y becomes depth z

    fig: Figure = plt.figure(figsize=(8, 8), dpi=150)
    sub3: Axes3D = fig.add_subplot(projection='3d')
    sub3.set_xlabel("x")
    sub3.set_ylabel("-z")
    sub3.set_zlabel("y")
    sub3.set_xlim(-100, 100)
    sub3.set_ylim(-100, 100)
    sub3.set_zlim(0, 200)

    lines: List[Line3D] = [
        sub3.plot(*(pos_3d_mapped[0, joint_pair, axis] for axis in range(3)))[0]
        for joint_pair in joint_pairs]

    if render_joint_id:
        text_3d: Text3D = [sub3.text(*(pos_3d_mapped[0, j_index, axis] for axis in range(3)), str(j_index), color='r')
            for j_index in range(num_joints)]
    else:
        text_3d = None

    # here for debug, we can also render joint index.
    def update_human_3d(index: int):
        if text_3d is not None:
            for j_index in range(num_joints):
                text_3d[j_index].set_position_3d(pos_3d_mapped[index, j_index])
        for pair_idx, joint_pair in enumerate(joint_pairs):
            lines[pair_idx].set_data(pos_3d_mapped[index, joint_pair, axis] for axis in range(2))
            lines[pair_idx].set_3d_properties(pos_3d_mapped[index, joint_pair, 2])

    anim = FuncAnimation(fig, update_human_3d, pos_3d_mapped.shape[0], interval=1, repeat=False)

    add_axis_indicator_with_image(fig, image_path='./ddpm/utils/coordinate.png')

    writer = PillowWriter(fps=30)  # Adjust FPS as needed
    anim.save(save_path, writer=writer)
    print(f"Animation saved to {save_path}")
    plt.show()


def visualize_joints_for_two_people(num_joints, joint_chain, pos_3d, ret_forward_direction, head_forward_direction, render_joint_id: bool = False,
                        save_path: Optional[str] = './test_files/pymotionlib4gesture/test1.gif'):
    """
    Visualize the input 3d joint position for two people using matplotlib.
    """

    joint_pairs = [(parent, child) for child, parent in enumerate(joint_chain) if parent != -1]
    head_joint_index = 6

    pos_3d_1, pos_3d_2 = pos_3d
    ret_forward_direction_1, ret_forward_direction_2 = ret_forward_direction
    ret_forward_direction_1 = ret_forward_direction_1 * 5
    ret_forward_direction_2 = ret_forward_direction_2 * 5

    head_forward_direction_1, head_forward_direction_2 = head_forward_direction
    head_forward_direction_1 = head_forward_direction_1 * 20
    head_forward_direction_2 = head_forward_direction_2 * 20

    # Mapping the 3D positions for proper visualization
    pos_3d_mapped_1 = pos_3d_1.copy()
    pos_3d_mapped_1[..., 0] = pos_3d_1[..., 0]  # Keep x as is
    pos_3d_mapped_1[..., 1] = pos_3d_1[..., 2]  # z becomes vertical y
    pos_3d_mapped_1[..., 2] = pos_3d_1[..., 1]  # y becomes depth z

    pos_3d_mapped_2 = pos_3d_2.copy()
    pos_3d_mapped_2[..., 0] = pos_3d_2[..., 0]  # Keep x as is
    pos_3d_mapped_2[..., 1] = pos_3d_2[..., 2]  # z becomes vertical y
    pos_3d_mapped_2[..., 2] = pos_3d_2[..., 1]  # y becomes depth z

    fig: plt.Figure = plt.figure(figsize=(8, 8), dpi=150)
    sub3: Axes3D = fig.add_subplot(projection='3d')
    sub3.set_xlabel("x")
    sub3.set_ylabel("z")
    sub3.set_zlabel("y")
    sub3.set_xlim(-100, 100)
    sub3.set_ylim(-100, 100)
    sub3.set_zlim(0, 200)
    sub3.invert_yaxis()

    color_1 = 'b'
    color_2 = 'g'
    # Draw lines for the first person
    lines_1: List[Line3D] = [
        sub3.plot(*(pos_3d_mapped_1[0, joint_pair, axis] for axis in range(3)), color=color_1)[0]
        for joint_pair in joint_pairs]

    # Draw lines for the second person
    lines_2: List[Line3D] = [
        sub3.plot(*(pos_3d_mapped_2[0, joint_pair, axis] for axis in range(3)), color=color_2)[0]
        for joint_pair in joint_pairs]

    scatter_1 = sub3.scatter(pos_3d_mapped_1[0, 0, 0], pos_3d_mapped_1[0, 0, 1], 0, color='red', s=50)
    scatter_2 = sub3.scatter(pos_3d_mapped_2[0, 0, 0], pos_3d_mapped_2[0, 0, 1], 0, color='y', s=50)
    scatter_1_head = sub3.scatter(pos_3d_mapped_1[0, head_joint_index, 0], pos_3d_mapped_1[0, head_joint_index, 1], pos_3d_mapped_1[0, head_joint_index, 2], color='red', s=50)
    scatter_2_head = sub3.scatter(pos_3d_mapped_2[0, head_joint_index, 0], pos_3d_mapped_2[0, head_joint_index, 1], pos_3d_mapped_2[0, head_joint_index, 2], color='y', s=50)
    root_line_1 = sub3.plot([pos_3d_mapped_1[0, 0, 0], pos_3d_mapped_1[0, 0, 0]], [pos_3d_mapped_1[0, 0, 1], pos_3d_mapped_1[0, 0, 1]], [pos_3d_mapped_1[0, 0, 2], 0], color='orange', linestyle='-', linewidth=2, zorder=1)
    root_line_2 = sub3.plot([pos_3d_mapped_2[0, 0, 0], pos_3d_mapped_2[0, 0, 0]], [pos_3d_mapped_2[0, 0, 1], pos_3d_mapped_2[0, 0, 1]], [pos_3d_mapped_2[0, 0, 2], 0], color='orange', linestyle='-', linewidth=2, zorder=1)
    root_direction_1 = sub3.quiver(pos_3d_mapped_1[0, 0, 0], pos_3d_mapped_1[0, 0, 1], 0, ret_forward_direction_1[0, 0], ret_forward_direction_1[0, 1], 0, color='red', length=0.1, arrow_length_ratio=0.5)
    root_direction_2 = sub3.quiver(pos_3d_mapped_2[0, 0, 0], pos_3d_mapped_2[0, 0, 1], 0, ret_forward_direction_2[0, 0], ret_forward_direction_2[0, 1], 0, color='y', length=0.1, arrow_length_ratio=0.5)
    head_direction_1 = sub3.quiver(pos_3d_mapped_1[0, head_joint_index, 0], pos_3d_mapped_1[0, head_joint_index, 1], pos_3d_mapped_1[0, head_joint_index, 2], head_forward_direction_1[0, 0], head_forward_direction_1[0, 1], pos_3d_mapped_1[0, head_joint_index, 2], color='red', length=0.1, arrow_length_ratio=0.5)
    head_direction_2 = sub3.quiver(pos_3d_mapped_2[0, head_joint_index, 0], pos_3d_mapped_2[0, head_joint_index, 1], pos_3d_mapped_2[0, head_joint_index, 2], head_forward_direction_2[0, 0], head_forward_direction_2[0, 1], pos_3d_mapped_2[0, head_joint_index, 2], color='y', length=0.1, arrow_length_ratio=0.5)

    # Draw joint indices if required
    if render_joint_id:
        text_3d_1 = [sub3.text(*(pos_3d_mapped_1[0, j_index, axis] for axis in range(3)), str(j_index), color='b')
            for j_index in range(num_joints)]

        text_3d_2 = [sub3.text(*(pos_3d_mapped_2[0, j_index, axis] for axis in range(3)), str(j_index), color='b')
            for j_index in range(num_joints)]
    else:
        text_3d_1 = None
        text_3d_2 = None

    # Update function for both people
    def update_human_3d(index: int):
        if text_3d_1 is not None:
            for j_index in range(num_joints):
                text_3d_1[j_index].set_position_3d(pos_3d_mapped_1[index, j_index])
                text_3d_2[j_index].set_position_3d(pos_3d_mapped_2[index, j_index])

        for pair_idx, joint_pair in enumerate(joint_pairs):
            # Update the lines for the first person
            lines_1[pair_idx].set_data(pos_3d_mapped_1[index, joint_pair, axis] for axis in range(2))
            lines_1[pair_idx].set_3d_properties(pos_3d_mapped_1[index, joint_pair, 2])

            # Update the lines for the second person
            lines_2[pair_idx].set_data(pos_3d_mapped_2[index, joint_pair, axis] for axis in range(2))
            lines_2[pair_idx].set_3d_properties(pos_3d_mapped_2[index, joint_pair, 2])

        scatter_1._offsets3d = (np.array([pos_3d_mapped_1[index, 0, 0]]), np.array([pos_3d_mapped_1[index, 0, 1]]), np.array([0]))  # Use the new coordinates
        scatter_2._offsets3d = (np.array([pos_3d_mapped_2[index, 0, 0]]), np.array([pos_3d_mapped_2[index, 0, 1]]), np.array([0]))  # Use the new coordinates
        scatter_1_head._offsets3d = (np.array([pos_3d_mapped_1[index, head_joint_index, 0]]), np.array([pos_3d_mapped_1[index, head_joint_index, 1]]), np.array([pos_3d_mapped_1[index, head_joint_index, 2]]))  # Use the new coordinates
        scatter_2_head._offsets3d = (np.array([pos_3d_mapped_2[index, head_joint_index, 0]]), np.array([pos_3d_mapped_2[index, head_joint_index, 1]]), np.array([pos_3d_mapped_2[index, head_joint_index, 2]]))  # Use the new coordinates
        root_line_1[0].set_data([pos_3d_mapped_1[index, 0, 0], pos_3d_mapped_1[index, 0, 0]], [pos_3d_mapped_1[index, 0, 1], pos_3d_mapped_1[index, 0, 1]])
        root_line_1[0].set_3d_properties([pos_3d_mapped_1[index, 0, 2], 0])
        root_line_2[0].set_data([pos_3d_mapped_2[index, 0, 0], pos_3d_mapped_2[index, 0, 0]], [pos_3d_mapped_2[index, 0, 1], pos_3d_mapped_2[index, 0, 1]])
        root_line_2[0].set_3d_properties([pos_3d_mapped_2[index, 0, 2], 0])
        root_direction_1._segments3d = [[[pos_3d_mapped_1[index, 0, 0], pos_3d_mapped_1[index, 0, 1], 0], [pos_3d_mapped_1[index, 0, 0] + ret_forward_direction_1[index, 0], pos_3d_mapped_1[index, 0, 1] + ret_forward_direction_1[index, 1], 0]]]
        root_direction_2._segments3d = [[[pos_3d_mapped_2[index, 0, 0], pos_3d_mapped_2[index, 0, 1], 0], [pos_3d_mapped_2[index, 0, 0] + ret_forward_direction_2[index, 0], pos_3d_mapped_2[index, 0, 1] + ret_forward_direction_2[index, 1], 0]]]
        head_direction_1._segments3d = [[[pos_3d_mapped_1[index, head_joint_index, 0], pos_3d_mapped_1[index, head_joint_index, 1], pos_3d_mapped_1[index, head_joint_index, 2]], [pos_3d_mapped_1[index, head_joint_index, 0] + head_forward_direction_1[index, 0], pos_3d_mapped_1[index, head_joint_index, 1] + head_forward_direction_1[index, 1], pos_3d_mapped_1[index, head_joint_index, 2]]]]
        head_direction_2._segments3d = [[[pos_3d_mapped_2[index, head_joint_index, 0], pos_3d_mapped_2[index, head_joint_index, 1], pos_3d_mapped_2[index, head_joint_index, 2]], [pos_3d_mapped_2[index, head_joint_index, 0] + head_forward_direction_2[index, 0], pos_3d_mapped_2[index, head_joint_index, 1] + head_forward_direction_2[index, 1], pos_3d_mapped_2[index, head_joint_index, 2]]]]

    # Create animation
    anim = FuncAnimation(fig, update_human_3d, pos_3d_mapped_1.shape[0], interval=1, repeat=False)

    # Optional: Add additional axis indicator with image
    add_axis_indicator_with_image(fig, image_path='./ddpm/utils/coordinate.png')

    writer = PillowWriter(fps=30)  # Adjust FPS as needed
    anim.save(save_path, writer=writer)
    print(f"Animation saved to {save_path}")


def add_axis_indicator_with_image(fig: plt.Figure, image_path: str):
    """
    Adds an external axis indicator to the figure with a custom image.
    :param fig: The matplotlib figure to add the indicator to.
    :param image_path: Path to the image file to display in the indicator.
    """
    # Replace the 3D subplot with a simple 2D subplot
    ax_indicator = fig.add_axes([0.1, 0.7, 0.2, 0.2])  # Define a larger area in the top-left corner
    ax_indicator.axis('off')  # Turn off the axis

    # Load and display the image
    img = mpimg.imread(image_path)
    ax_indicator.imshow(img)


def fk_model(motion, motion_all, other_joint_index, skeleton_info, for_save_bvh=False):
    # start_time = time_test.time()
    # motion: [frame, feature_dim], torch.tensor
    frame, feature_dim = motion.shape

    # 1. joint chain
    joint_chain = skeleton_info['joint_chain']



    # print(motion.shape, motion.dtype, motion.device)

    # 2. add constant joint values
    # constant_joint_values = data_pipe['cnst'].const_values_
    # start_time = time_test.time()
    # constant_joint_values = torch.tensor(list(data_pipe['cnst'].const_values_.values())).to(motion.device).to(motion.dtype)
    # constant_joint_index = [3, 4, 12, 13, 25, 34, 43, 52, 53, 76, 85, 94, 103, 104, 144, 146]
    # other_joint_index = [i for i in range(150) if i not in constant_joint_index]
    # assert len(other_joint_index) == feature_dim



    # start_time = time_test.time()
    # motion_all = torch.zeros((frame, 150)).to(motion.device).to(motion.dtype)
    # motion_all[:, constant_joint_index] = constant_joint_values

    motion_all = motion_all[0:motion.shape[0], :]
    motion_all[:, other_joint_index] = motion
    # print_tensor_features(motion_all, "motion_all")
    # print_tensor_features(motion, "motion")




    # 3. convert exp to quat rotation
    # start_time = time_test.time()
    all_rot_exp = motion_all[:, 0:144].reshape(frame, -1, 3)
    root_position = motion_all[:, 144:147]
    root_dx = motion_all[:, 147]
    root_dz = motion_all[:, 148]
    root_dy_rot = motion_all[:, 149]



    # start_time = time_test.time()
    all_rot_quat = expmap_to_quaternion(all_rot_exp)



    # 4. calculate root global info
    # rec_ry = torch.zeros(frame, 4).to(motion.device).to(motion.dtype)  # Initial root rotation
    # rec_ry[:, 0] = 1.0
    # rec_xp = torch.zeros(frame).to(motion.device).to(motion.dtype)  # Initial root X position
    # rec_zp = torch.zeros(frame).to(motion.device).to(motion.dtype)  # Initial root Z position

    # q_y = angle_axis_to_quat(root_dy_rot, torch.tensor([0.0, 1.0, 0.0]).to(motion.device).to(motion.dtype))


    # # Accumulate root global transformation
    # for i in range(frame - 1):
    #     # Apply Y-axis rotation from root_dy_rot
    #     rec_ry[i + 1, :] = quaternion_multiply(q_y[i+1], rec_ry[i, :])

    # rotated_velocities = rotate_vector_by_quaternion(root_vel[1:], rec_ry[1:])
    # rec_xp[1:] = torch.cumsum(rotated_velocities[:, 0], dim=0)
    # rec_zp[1:] = torch.cumsum(rotated_velocities[:, 2], dim=0)

    # # rec_root_rot = quaternion_multiply(rec_ry, root_rot)  # Shape: [frame, 4]
    # rec_root_pos = rotate_vector_by_quaternion(root_position, rec_ry)  # Shape: [frame, 3]
    # rec_root_pos[:, 0] += rec_xp  # Add root X position to joints
    # rec_root_pos[:, 2] += rec_zp  # Add root Z position to joints


    # start_time = time_test.time()
    root_rot = all_rot_quat[:, -1]
    root_vel = torch.stack([root_dx, torch.zeros_like(root_dx), root_dz], dim=-1).to(motion.device).to(motion.dtype)

    rec_root_position = torch.zeros_like(root_position)
    rec_root_position[:, 1] = root_position[:, 1]
    # print(f"Result version: {root_rot[0]._version}")
    root_rot_y_quat = quat_to_angle_axis_y(root_rot.clone())
    root_vel = rotate_vector_by_quaternion(root_vel, root_rot_y_quat)
    root_vel = torch.cumsum(root_vel, dim=0)

    rec_root_position[:, 0] += root_vel[:, 0]  # Add root X position to joints
    rec_root_position[:, 2] += root_vel[:, 2]  # Add root Z position to joints



    # 5. fk for all
    # all_rot_quat[:, -1] = rec_root_rot
    # start_time = time_test.time()
    all_rot_quat[:, -1] = root_rot
    num_joints = all_rot_quat.shape[1]
    modified_all_rot_quat = torch.flip(all_rot_quat, dims=[1])



    if for_save_bvh:
        return rec_root_position, modified_all_rot_quat

    # modified_all_rot_quat = torch.cat([rec_root_rot.unsqueeze(1), all_rot_quat[:, :-1]], dim=1).to(motion.device).to(motion.dtype)
    # num_joints = modified_all_rot_quat.shape[1]

    # start_time = time_test.time()
    global_positions = torch.zeros(frame, num_joints, 3).to(motion.device).to(motion.dtype)
    global_rotations = torch.zeros(frame, num_joints, 4).to(motion.device).to(motion.dtype)
    # global_rotations[:, 0] = rec_root_rot  # Root rotation
    global_rotations[:, 0] = root_rot
    global_positions[:, 0] = rec_root_position  # Root position

    # Iterate over joint chain to compute global positions and rotations
    local_offsets = torch.stack([
        skeleton_info['offsets'][joint_idx].to(motion.device).to(motion.dtype)
        for joint_idx in range(len(joint_chain))
    ])  # Shape: [num_joints, 3]
    local_offsets = local_offsets.unsqueeze(0).repeat(frame, 1, 1)  # Shape: [frame, num_joints, 3]

    for joint_idx, parent_idx in enumerate(joint_chain):
        if parent_idx == -1:  # Root joint
            continue

        parent_global_rot = global_rotations[:, parent_idx].clone()  # Shape: [frame, 4]
        parent_global_pos = global_positions[:, parent_idx].clone()  # Shape: [frame, 3]

        local_rot = modified_all_rot_quat[:, joint_idx]  # Shape: [frame, 4]
        global_rotations[:, joint_idx] = quaternion_multiply(parent_global_rot.clone(), local_rot)

        local_offset = local_offsets[:, joint_idx]
        global_offset = rotate_vector_by_quaternion(local_offset, parent_global_rot.clone())  # Shape: [frame, 3]
        global_positions[:, joint_idx] = parent_global_pos.clone() + global_offset

    global_rotations = quaternion_to_expmap(global_rotations.reshape(-1, 4)).reshape(frame, num_joints, 3)



    # visualize_joints(num_joints, joint_chain, global_positions)

    return torch.cat([global_positions, global_rotations], dim=-1)


def fk_model_root(motion, motion_all, other_joint_index, dataset_type='interact'):
    # motion: [frame, feature_dim], torch.tensor
    frame, feature_dim = motion.shape

    motion_all = motion_all[0:motion.shape[0], :]
    motion_all[:, other_joint_index] = motion

    # 3. convert exp to quat rotation
    if dataset_type in ['interact', 'zeggs']:
        all_rot_exp = motion_all[:, 0:144].reshape(frame, -1, 3)
        root_position = motion_all[:, 144:147]
        root_dx = motion_all[:, 147]
        root_dz = motion_all[:, 148]
    elif dataset_type == 'photoreal':
        all_rot_exp = motion_all[:, 0:177].reshape(frame, -1, 3)
        root_position = motion_all[:, 177:180]
        root_dx = motion_all[:, 180]
        root_dz = motion_all[:, 181]

    root_rot = expmap_to_quaternion(all_rot_exp[:, -1])

    # 4. calculate root global info

    root_vel = torch.stack([root_dx, torch.zeros_like(root_dx), root_dz], dim=-1).to(motion.device).to(motion.dtype)

    rec_root_position = torch.zeros_like(root_position)
    rec_root_position[:, 1] = root_position[:, 1]
    # print(f"Result version: {root_rot[0]._version}")
    root_rot_y_quat = quat_to_angle_axis_y(root_rot.clone())
    root_vel = rotate_vector_by_quaternion(root_vel, root_rot_y_quat)
    root_vel = torch.cumsum(root_vel, dim=0)

    rec_root_position[:, 0] += root_vel[:, 0]  # Add root X position to joints
    rec_root_position[:, 2] += root_vel[:, 2]  # Add root Z position to joints

    return rec_root_position


def fk_model_head(motion, motion_all, other_joint_index, skeleton_info):
    # motion: [frame, feature_dim], torch.tensor
    frame, feature_dim = motion.shape

    # 1. joint chain
    joint_chain = skeleton_info['joint_chain']

    motion_all = motion_all[0:motion.shape[0], :]
    motion_all[:, other_joint_index] = motion

    # 3. convert exp to quat rotation
    all_rot_exp = motion_all[:, 0:144].reshape(frame, -1, 3)
    root_position = motion_all[:, 144:147]
    root_dx = motion_all[:, 147]
    root_dz = motion_all[:, 148]

    all_rot_quat = expmap_to_quaternion(all_rot_exp)

    # 4. calculate root global info

    root_rot = all_rot_quat[:, -1]
    root_vel = torch.stack([root_dx, torch.zeros_like(root_dx), root_dz], dim=-1).to(motion.device).to(motion.dtype)

    rec_root_position = torch.zeros_like(root_position)
    rec_root_position[:, 1] = root_position[:, 1]
    # print(f"Result version: {root_rot[0]._version}")
    root_rot_y_quat = quat_to_angle_axis_y(root_rot.clone())
    root_vel = rotate_vector_by_quaternion(root_vel, root_rot_y_quat)
    root_vel = torch.cumsum(root_vel, dim=0)

    rec_root_position[:, 0] += root_vel[:, 0]  # Add root X position to joints
    rec_root_position[:, 2] += root_vel[:, 2]  # Add root Z position to joints

    # 5. fk for all
    all_rot_quat[:, -1] = root_rot
    num_joints = all_rot_quat.shape[1]
    modified_all_rot_quat = torch.flip(all_rot_quat, dims=[1])

    global_positions = torch.zeros(frame, 7, 3).to(motion.device).to(motion.dtype)
    global_rotations = torch.zeros(frame, 7, 4).to(motion.device).to(motion.dtype)
    # global_rotations[:, 0] = rec_root_rot  # Root rotation
    global_rotations[:, 0] = root_rot
    global_positions[:, 0] = rec_root_position  # Root position

    # Iterate over joint chain to compute global positions and rotations
    local_offsets = torch.stack([
        skeleton_info['offsets'][joint_idx].to(motion.device).to(motion.dtype)
        for joint_idx in range(len(joint_chain))
    ])  # Shape: [num_joints, 3]
    local_offsets = local_offsets.unsqueeze(0).repeat(frame, 1, 1)  # Shape: [frame, num_joints, 3]

    for joint_idx, parent_idx in enumerate(joint_chain):
        if parent_idx == -1:  # Root joint
            continue

        if joint_idx >= 7:
            break

        parent_global_rot = global_rotations[:, parent_idx].clone()  # Shape: [frame, 4]
        parent_global_pos = global_positions[:, parent_idx].clone()  # Shape: [frame, 3]

        local_rot = modified_all_rot_quat[:, joint_idx]  # Shape: [frame, 4]
        global_rotations[:, joint_idx] = quaternion_multiply(parent_global_rot.clone(), local_rot)

        local_offset = local_offsets[:, joint_idx]
        global_offset = rotate_vector_by_quaternion(local_offset, parent_global_rot.clone())  # Shape: [frame, 3]
        global_positions[:, joint_idx] = parent_global_pos.clone() + global_offset

    global_rotations = quaternion_to_expmap(global_rotations.reshape(-1, 4)).reshape(frame, 7, 3)

    head_position, head_rotation = global_positions[:, -1], global_rotations[:, -1]

    return torch.cat([head_position, head_rotation], dim=-1)


def fk_model_foot(motion, motion_all, other_joint_index, skeleton_info, dataset_type='interact'):
    # motion: [frame, feature_dim], torch.tensor
    frame, feature_dim = motion.shape

    # 1. joint chain
    joint_chain = skeleton_info['joint_chain']

    motion_all = motion_all[0:motion.shape[0], :]
    motion_all[:, other_joint_index] = motion

    # 3. convert exp to quat rotation
    if dataset_type in ['interact', 'zeggs']:
        all_rot_exp = motion_all[:, 0:144].reshape(frame, -1, 3)
        root_position = motion_all[:, 144:147]
        root_dx = motion_all[:, 147]
        root_dz = motion_all[:, 148]
    elif dataset_type == 'photoreal':
        all_rot_exp = motion_all[:, 0:177].reshape(frame, -1, 3)
        root_position = motion_all[:, 177:180]
        root_dx = motion_all[:, 180]
        root_dz = motion_all[:, 181]

    all_rot_quat = expmap_to_quaternion(all_rot_exp)

    # 4. calculate root global info

    root_rot = all_rot_quat[:, -1]
    root_vel = torch.stack([root_dx, torch.zeros_like(root_dx), root_dz], dim=-1).to(motion.device).to(motion.dtype)

    rec_root_position = torch.zeros_like(root_position)
    rec_root_position[:, 1] = root_position[:, 1]
    # print(f"Result version: {root_rot[0]._version}")
    root_rot_y_quat = quat_to_angle_axis_y(root_rot.clone())
    root_vel = rotate_vector_by_quaternion(root_vel, root_rot_y_quat)
    root_vel = torch.cumsum(root_vel, dim=0)

    rec_root_position[:, 0] += root_vel[:, 0]  # Add root X position to joints
    rec_root_position[:, 2] += root_vel[:, 2]  # Add root Z position to joints

    # 5. fk for all
    all_rot_quat[:, -1] = root_rot
    num_joints = all_rot_quat.shape[1]
    modified_all_rot_quat = torch.flip(all_rot_quat, dims=[1])

    if dataset_type in ['interact', 'zeggs']:
        lower_body_start_index = 42
        relative_right_foot_index = 2
        relative_left_foot_index = 5
        lower_joints_num = 6
    elif dataset_type == 'photoreal':
        lower_body_start_index = 51
        relative_right_foot_index = 6
        relative_left_foot_index = 2
        lower_joints_num = 8

    global_positions = torch.zeros(frame, lower_joints_num, 3).to(motion.device).to(motion.dtype)
    global_rotations = torch.zeros(frame, lower_joints_num, 4).to(motion.device).to(motion.dtype)
    # global_rotations[:, 0] = rec_root_rot  # Root rotation
    global_rotations[:, 0] = root_rot
    global_positions[:, 0] = rec_root_position  # Root position

    # Iterate over joint chain to compute global positions and rotations
    local_offsets = torch.stack([
        skeleton_info['offsets'][joint_idx].to(motion.device).to(motion.dtype)
        for joint_idx in range(len(joint_chain))
    ])  # Shape: [num_joints, 3]
    local_offsets = local_offsets.unsqueeze(0).repeat(frame, 1, 1)  # Shape: [frame, num_joints, 3]

    for joint_idx, parent_idx in enumerate(joint_chain):
        if parent_idx == -1:  # Root joint
            continue

        if joint_idx < lower_body_start_index:
            continue

        if parent_idx != 0:
            parent_idx = parent_idx - lower_body_start_index

        parent_global_rot = global_rotations[:, parent_idx].clone()  # Shape: [frame, 4]
        parent_global_pos = global_positions[:, parent_idx].clone()  # Shape: [frame, 3]

        local_rot = modified_all_rot_quat[:, joint_idx]  # Shape: [frame, 4]
        global_rotations[:, joint_idx-lower_body_start_index] = quaternion_multiply(parent_global_rot.clone(), local_rot)

        local_offset = local_offsets[:, joint_idx]
        global_offset = rotate_vector_by_quaternion(local_offset, parent_global_rot.clone())  # Shape: [frame, 3]
        global_positions[:, joint_idx-lower_body_start_index] = parent_global_pos.clone() + global_offset

    # global_rotations = quaternion_to_expmap(global_rotations.reshape(-1, 4)).reshape(frame, 6, 3)

    right_foot_position = global_positions[:, relative_right_foot_index, [0,2]]
    left_foot_position = global_positions[:, relative_left_foot_index, [0,2]]
    foot_position = torch.cat([right_foot_position, left_foot_position], dim=-1)
    # foot_vel = torch.diff(foot_position, dim=0)

    return foot_position


def quaternion_inverse(q):
    """
    Compute the inverse of a quaternion.

    Parameters:
    - q: torch.tensor of shape (..., 4), where the last dimension represents the quaternion [w, x, y, z]

    Returns:
    - torch.tensor of shape (..., 4), the inverse quaternion
    """
    if q.shape[-1] != 4:
        raise ValueError("Input must have shape (..., 4), representing quaternions as [w, x, y, z].")

    # Compute the conjugate: [w, -x, -y, -z]
    q_conjugate = q.clone()
    q_conjugate[..., 1:] = -q[..., 1:]  # Negate x, y, z components

    # Compute the squared norm: w^2 + x^2 + y^2 + z^2
    q_norm_squared = torch.sum(q ** 2, dim=-1, keepdim=True)

    # Compute the inverse: conjugate / norm_squared
    q_inverse = q_conjugate / q_norm_squared
    return q_inverse


def between_torch(v0s: torch.Tensor, v1s: torch.Tensor) -> torch.Tensor:
    """
    Compute the quaternion representing the rotation from v0s to v1s in PyTorch.

    Args:
        v0s (torch.Tensor): A tensor of shape (..., 3), representing the starting direction vectors.
        v1s (torch.Tensor): A tensor of shape (..., 3), representing the target direction vectors.

    Returns:
        torch.Tensor: A tensor of shape (..., 4), representing the resulting quaternions [w, x, y, z].
    """
    # Compute the cross product: rotation axis
    a = torch.cross(v0s, v1s, dim=-1)  # Shape (..., 3)

    # Compute the scalar part of the quaternion (w)
    w = torch.sqrt((v0s**2).sum(dim=-1) * (v1s**2).sum(dim=-1)) + (v0s * v1s).sum(dim=-1)  # Shape (...,)

    # Concatenate to form quaternion [w, x, y, z]
    quaternions = torch.cat([w.unsqueeze(-1), a], dim=-1)  # Shape (..., 4)

    # Normalize the quaternions
    quaternions = quaternions / (torch.norm(quaternions, dim=-1, keepdim=True) + 1e-10)

    return quaternions


def inv_fk_model(global_rot, global_pos, data_pipe, skeleton_info):
    # global_rot: [bs, frame, num_joints, 4], torch.tensor (quaternion format)
    # global_pos: [bs, frame, num_joints, 3], torch.tensor (position format)

    all_frame, num_joints, _ = global_rot.shape
    # global_rot = global_rot.reshape(bs*frame, num_joints, 4)
    # global_pos = global_pos.reshape(bs*frame, num_joints, 3)

    # 1. Initialize outputs
    motion_all = torch.zeros((all_frame, 150)).to(global_rot.device).to(global_rot.dtype)

    # 2. Extract root position and rotation
    root_position = global_pos[:, 0]  # Shape: [bs*frame, 3]
    root_rotation = global_rot[:, 0]  # Shape: [bs*frame, 4]

    # 3. Calculate root velocity and rotation delta
    # rec_xp = root_position[:, 0]  # X coordinate
    # rec_zp = root_position[:, 2]  # Z coordinate
    # root_dx = torch.diff(rec_xp, prepend=torch.tensor([0.0]).to(global_rot.device))  # Shape: [frame]
    # root_dz = torch.diff(rec_zp, prepend=torch.tensor([0.0]).to(global_rot.device))  # Shape: [frame]

    root_vel = torch.diff(root_position, dim=0).to(global_rot.device)
    first_frame_root_rot_y_angle = quat_to_angle_axis_y(root_rotation[0])
    first_frame_root_rot_y_quat = angle_axis_to_quat(first_frame_root_rot_y_angle, torch.tensor([0.0, 1.0, 0.0]).to(global_rot.device))
    root_vel = rotate_vector_by_quaternion(root_vel, quaternion_inverse(first_frame_root_rot_y_quat))
    root_vel = torch.vstack([torch.tensor([0.0, 0.0, 0.0]).to(global_rot.device), root_vel])
    root_dx, root_dz = root_vel[:, 0], root_vel[:, 2]

    # Calculate Y-axis rotation (assuming rotation around Y axis only)
    forward = rotate_vector_by_quaternion(torch.tensor([[0, 0, 1]]).to(global_rot.device), root_rotation)
    forward[:, 1] = 0.0
    target = torch.tensor([[0, 0, 1]]).repeat(len(forward), 1).to(global_rot.device).to(global_rot.dtype)
    rot_for_vel = between_torch(target, forward)
    q_y_inverse = quaternion_inverse(rot_for_vel)
    delta_rotations = quaternion_multiply(q_y_inverse[:-1], rot_for_vel[1:])  # Shape: [frame-1, 4]
    delta_rotations = torch.vstack([delta_rotations[0], delta_rotations])
    root_dy_rot = quat_to_angle_axis_y(delta_rotations)  # Extract angle for Y rotation
    # print(root_dy_rot)

    # Store in motion_all
    motion_all[:, 144:147] = root_position  # Root position
    motion_all[:, 147] = root_dx
    motion_all[:, 148] = root_dz
    motion_all[:, 149] = root_dy_rot

    # 4. Compute local joint rotations
    joint_chain = skeleton_info['joint_chain']
    local_rotations = torch.zeros_like(global_rot)

    for joint_idx, parent_idx in enumerate(joint_chain):
        if parent_idx == -1:  # Root joint
            local_rotations[:, joint_idx] = root_rotation
        else:
            parent_global_rot = global_rot[:, parent_idx]
            joint_global_rot = global_rot[:, joint_idx]
            local_rotations[:, joint_idx] = quaternion_multiply(quaternion_inverse(parent_global_rot), joint_global_rot)

    # Convert local rotations to exponential map
    local_rot_exp = quaternion_to_expmap(local_rotations.reshape(-1, 4)).reshape(all_frame, num_joints, 3)
    local_rot_exp = torch.flip(local_rot_exp, dims=[1])
    motion_all[:, :144] = local_rot_exp.reshape(all_frame, -1)

    # 5. Extract variable joint values (other indices)
    constant_joint_index = [3, 4, 12, 13, 25, 34, 43, 52, 53, 76, 85, 94, 103, 104, 144, 146]
    other_joint_index = [i for i in range(150) if i not in constant_joint_index]
    motion = motion_all[:, other_joint_index]  # Shape: [frame, feature_dim]

    return motion


def flip_vector_by_dot(x: np.ndarray, inplace: bool = False) -> np.ndarray:
    """
    make sure x[i] * x[i+1] >= 0
    """
    if x.ndim == 1:
        return x

    sign: np.ndarray = np.sum(x[:-1] * x[1:], axis=-1)
    sign[sign < 0] = -1
    sign[sign >= 0] = 1
    sign = np.cumprod(sign, axis=0, )

    # print(np.sum(sign[0, :] == -1))
    # print(np.where(sign[0, :] == -1)[0])

    x_res = x.copy() if not inplace else x
    x_res[1:][sign < 0] *= -1

    return x_res


def export_quaternion_motion_bvh(root_pos, joints_rot_quat, skeleton_info_raw, data_pipe, bvh_save_path):
    import pandas as pd
    from scipy.spatial.transform import Rotation as R

    root_pos = root_pos.cpu().numpy()
    joints_rot_quat = joints_rot_quat.cpu().numpy()
    all_frame, joint_num, _ = joints_rot_quat.shape

    first_frame_quat_for_flip = np.zeros((joint_num, 4))
    first_frame_quat_for_flip[:, 0] = 1
    joints_rot_quat = flip_vector_by_dot(np.concatenate([first_frame_quat_for_flip[np.newaxis, :], joints_rot_quat], axis=0))[1:]
    joints_rot_quat = joints_rot_quat.reshape(all_frame*joint_num, 4)

    # quat 2 euler(degree, ZYX) wxyz
    rotation = R.from_quat(joints_rot_quat, scalar_first=True)
    joints_rot_euler = rotation.as_euler('zyx', degrees=True)
    joints_rot_euler = joints_rot_euler.reshape(all_frame, joint_num, 3)
    # print(joints_rot_euler[0, 0:5])
    joints_rot_euler = joints_rot_euler.reshape(all_frame, joint_num*3)
    # concat root pos & joints rot
    all_motion_info = np.concatenate([root_pos, joints_rot_euler], axis = -1)

    motion = skeleton_info_raw[0]
    all_features_dim = len(motion.values.columns)
    motion.values = pd.DataFrame(np.zeros((all_frame, all_features_dim)), columns=motion.values.columns)

    select_feature_dims = data_pipe['jtsel'].selected_channels
    motion.values[select_feature_dims] = all_motion_info
    not_selected = data_pipe['jtsel'].not_selected
    not_selected_values = data_pipe['jtsel'].not_selected_values
    for d in not_selected:
        motion.values[d] = not_selected_values[d]

    writer = BVHWriter()
    with open(bvh_save_path, 'w') as f:
        writer.write(motion, f, framerate=-1)


def extract_y_rot_from_exp(rot_exp):
    # rot_exp: [len,]
    rot_quat = expmap_to_quaternion(rot_exp)
    rot_y_quat = quat_to_angle_axis_y(rot_quat.clone())
    rot_y_exp = quaternion_to_expmap(rot_y_quat)

    return rot_y_exp

def extract_z_rot_from_exp(rot_exp):
    # rot_exp: [len,]
    rot_quat = expmap_to_quaternion(rot_exp)
    rot_z_quat = quat_to_angle_axis_z(rot_quat.clone())
    rot_z_exp = quaternion_to_expmap(rot_z_quat)

    return rot_z_exp

def smart_linspace(old_rot, new_rot, num_points=10):
    delta = new_rot - old_rot


    if abs(delta) <= 180:
        return np.linspace(old_rot, new_rot, num_points)


    if old_rot > 0:
        transition_point = 180
        next_transition = -540
        new_rot = new_rot - 360
    else:
        transition_point = -180
        next_transition = 540
        new_rot = new_rot + 360


    n1 = num_points // 2
    n2 = num_points - n1


    first_half = np.linspace(old_rot, transition_point, n1, endpoint=False)


    second_half = np.linspace(next_transition, new_rot, n2)


    return np.hstack((first_half, second_half))

def simulate_z_rotation_nod(duration_sec=2.0, fps=30, angle_deg=20.0, mean_angle=26.54, nods_per_second=1.0):
    num_frames = int(duration_sec * fps)
    times = np.linspace(0, duration_sec, num_frames)


    z_angles_deg = (angle_deg * np.sin(2 * np.pi * nods_per_second * times)) * (-1)
    z_angles_deg = z_angles_deg + mean_angle

    return z_angles_deg.astype(np.float32)
