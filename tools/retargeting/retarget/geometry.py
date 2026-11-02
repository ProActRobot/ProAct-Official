import numpy as np
import mink


def get_relative_position(bvh_model, joint_ref, joint_to, frame_idx):
    positions, _ = bvh_model.frame_pose(frame_idx)
    joint_ref_idx = list(bvh_model.joints.keys()).index(joint_ref)
    joint_to_idx = list(bvh_model.joints.keys()).index(joint_to)
    relative_pos = positions[joint_to_idx] - positions[joint_ref_idx]
    relative_pos = np.array([-relative_pos[0], relative_pos[1], -relative_pos[2]])
    return relative_pos


def update_target(
    task,
    bvh_model,
    joint_ref,
    joint_to,
    frame_idx,
    norm_factor,
    base_target=None,
    offset=None,
):
    target = get_relative_position(bvh_model, joint_ref, joint_to, frame_idx)
    target = target / 100 * norm_factor
    target = [target[2], target[0], target[1]]
    if base_target is not None:
        target = np.array(target) + base_target
    if offset is not None:
        target = np.array(target) + np.array(offset)
    task.set_target(mink.SE3([1.0, 0.0, 0.0, 0.0, target[0], target[1], target[2]]))
    return target
