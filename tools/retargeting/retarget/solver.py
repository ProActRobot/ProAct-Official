from .tasks import create_tasks
import csv
import time
import mujoco
import mujoco.viewer
import mink
import numpy as np
from scipy.spatial.transform import Rotation as R
from loop_rate_limiters import RateLimiter
from .bvh import Bvh
from .geometry import update_target
from .viewer import update_overlay


def retarget_file(
    bvh_file,
    csv_file,
    *,
    model_path,
    max_frequency=1000.0,
    bvh_frame_time=None,
    visual=False,
):
    bvh_model = Bvh()
    bvh_model.parse_file(bvh_file)
    if bvh_model.frames < 1:
        raise ValueError("BVH contains no frames")
    bvh_frame_time = bvh_frame_time or bvh_model.frame_time
    model = mujoco.MjModel.from_xml_path(str(model_path))
    if model.nq != 50:
        raise ValueError("Model must use the supported 50-qpos G1 layout")
    configuration = mink.Configuration(model)
    tasks, targets = create_tasks(model)
    com_task = targets["com_task"]
    left_foot_task = targets["left_foot_task"]
    right_foot_task = targets["right_foot_task"]
    right_elbow_link_task = targets["right_elbow_link_task"]
    left_elbow_link_task = targets["left_elbow_link_task"]
    right_hand_task = targets["right_hand_task"]
    left_hand_task = targets["left_hand_task"]
    left_shoulder_task = targets["left_shoulder_task"]
    right_shoulder_task = targets["right_shoulder_task"]
    right_thumb_task = targets["right_thumb_task"]
    left_thumb_task = targets["left_thumb_task"]
    right_index_task = targets["right_index_task"]
    left_index_task = targets["left_index_task"]
    right_middle_task = targets["right_middle_task"]
    left_middle_task = targets["left_middle_task"]
    limits = [mink.ConfigurationLimit(model)]
    com_mid = model.body("com_target").mocapid[0]
    model = configuration.model
    data = configuration.data
    solver = "osqp"
    if visual:
        viewer = mujoco.viewer.launch_passive(
            model=model, data=data, show_left_ui=True, show_right_ui=True
        )
        mujoco.mjv_defaultFreeCamera(model, viewer.cam)
    mink.move_mocap_to_frame(model, data, "head_target", "head", "site")
    data.mocap_pos[com_mid] = data.subtree_com[1]
    rate = RateLimiter(frequency=max_frequency, warn=False)
    frame_idx = 0
    if csv_file:
        with open(csv_file, mode="w", newline="") as file:
            writer = csv.writer(file)
            joint_names = [model.joint(j).name for j in range(model.njnt)]
            adjusted_header = (
                ["frame_idx"]
                + ["floating_base_joint_dof_" + str(i + 1) for i in range(7)]
                + joint_names[1:]
            )
            writer.writerow(adjusted_header)

    def visualize_positions(viewer, positions, colors):
        for i, (pos, color) in enumerate(zip(positions, colors)):
            geom = viewer.user_scn.geoms[viewer.user_scn.ngeom]
            mujoco.mjv_initGeom(
                geom,
                type=mujoco.mjtGeom.mjGEOM_BOX,
                size=np.array([0.01, 0.01, 0.01]),
                pos=pos,
                mat=np.eye(3).flatten(),
                rgba=color,
            )
            viewer.user_scn.ngeom += 1

    try:
        # Converge on frame zero, then write every input frame exactly once.
        for step in range(100 + bvh_model.frames):
            initial_convergence = step < 100
            frame_idx = max(0, step - 100)
            if visual and not viewer.is_running():
                break
            if visual:
                update_overlay(viewer, frame_idx, frame_idx * bvh_frame_time)
            left_foot_zero_pose = data.xpos[model.body("left_ankle_roll_link").id]
            right_foot_zero_pose = data.xpos[model.body("right_ankle_roll_link").id]
            left_foot_task.set_target(mink.SE3([1.0, 0.0, 0.0, 0.0, *left_foot_zero_pose]))
            right_foot_task.set_target(mink.SE3([1.0, 0.0, 0.0, 0.0, *right_foot_zero_pose]))
            robot_lower_arm_norm = 36 / 36
            right_hand_target = update_target(
                right_hand_task,
                bvh_model,
                "b_r_forearm",
                "b_r_wrist",
                frame_idx,
                robot_lower_arm_norm,
                data.xpos[model.body("right_elbow_link").id],
                [0.0, -0.03, 0.0],
            )
            left_hand_target = update_target(
                left_hand_task,
                bvh_model,
                "b_l_forearm",
                "b_l_wrist",
                frame_idx,
                robot_lower_arm_norm,
                data.xpos[model.body("left_elbow_link").id],
                [0.0, 0.03, 0.0],
            )
            robot_upper_arm_norm = 25 / 36
            right_elbow_link_target = update_target(
                right_elbow_link_task,
                bvh_model,
                "b_r_arm",
                "b_r_forearm",
                frame_idx,
                robot_upper_arm_norm,
                data.xpos[model.body("right_shoulder_pitch_link").id],
                [0.0, -0.08, 0.0],
            )
            left_elbow_link_target = update_target(
                left_elbow_link_task,
                bvh_model,
                "b_l_arm",
                "b_l_forearm",
                frame_idx,
                robot_upper_arm_norm,
                data.xpos[model.body("left_shoulder_pitch_link").id],
                [0.0, 0.08, 0.0],
            )
            robot_shoulder_norm = 20 / 36
            left_shoulder_target = update_target(
                left_shoulder_task,
                bvh_model,
                "b_root",
                "b_l_arm",
                frame_idx,
                robot_shoulder_norm,
                data.xpos[model.body("pelvis").id],
            )
            right_shoulder_target = update_target(
                right_shoulder_task,
                bvh_model,
                "b_root",
                "b_r_arm",
                frame_idx,
                robot_shoulder_norm,
                data.xpos[model.body("pelvis").id],
            )
            robot_thumb_norm = 20 / 36
            right_thumb_target = update_target(
                right_thumb_task,
                bvh_model,
                "b_r_wrist",
                "b_r_thumb1",
                frame_idx,
                robot_thumb_norm,
                right_hand_target,
            )
            left_thumb_target = update_target(
                left_thumb_task,
                bvh_model,
                "b_l_wrist",
                "b_l_thumb1",
                frame_idx,
                robot_thumb_norm,
                left_hand_target,
            )
            robot_index_norm = 20 / 36
            right_index_target = update_target(
                right_index_task,
                bvh_model,
                "b_r_wrist",
                "b_r_index2",
                frame_idx,
                robot_index_norm,
                right_hand_target,
            )
            left_index_target = update_target(
                left_index_task,
                bvh_model,
                "b_l_wrist",
                "b_l_index2",
                frame_idx,
                robot_index_norm,
                left_hand_target,
            )
            robot_middle_norm = 20 / 36
            right_middle_target = update_target(
                right_middle_task,
                bvh_model,
                "b_r_wrist",
                "b_r_middle2",
                frame_idx,
                robot_middle_norm,
                right_hand_target,
            )
            left_middle_target = update_target(
                left_middle_task,
                bvh_model,
                "b_l_wrist",
                "b_l_middle2",
                frame_idx,
                robot_middle_norm,
                left_hand_target,
            )
            com_task.set_target(data.mocap_pos[com_mid])
            vel = mink.solve_ik(configuration, tasks, rate.dt, solver, 0.01, limits=limits)
            configuration.integrate_inplace(vel, rate.dt)
            data.qpos[0:19] = 0.0
            data.qpos[2] = 0.79
            data.qpos[3] = 1.0
            data.qpos[29] = 0
            data.qpos[30] = (
                np.linalg.norm(
                    R.from_quat(
                        bvh_model.joint_rotation[frame_idx, bvh_model.idx_of["b_l_thumb1"]]
                    ).as_rotvec()
                )
                - 0.1
            )
            data.qpos[31] = (
                np.linalg.norm(
                    R.from_quat(
                        bvh_model.joint_rotation[frame_idx, bvh_model.idx_of["b_l_thumb2"]]
                    ).as_rotvec()
                )
                - 0.1
            )
            data.qpos[32] = -np.linalg.norm(
                R.from_quat(
                    bvh_model.joint_rotation[frame_idx, bvh_model.idx_of["b_l_middle1"]]
                ).as_rotvec()
            )
            data.qpos[33] = -np.linalg.norm(
                R.from_quat(
                    bvh_model.joint_rotation[frame_idx, bvh_model.idx_of["b_l_middle2"]]
                ).as_rotvec()
            )
            data.qpos[34] = -np.linalg.norm(
                R.from_quat(
                    bvh_model.joint_rotation[frame_idx, bvh_model.idx_of["b_l_index1"]]
                ).as_rotvec()
            )
            data.qpos[35] = -np.linalg.norm(
                R.from_quat(
                    bvh_model.joint_rotation[frame_idx, bvh_model.idx_of["b_l_index2"]]
                ).as_rotvec()
            )
            data.qpos[43] = 0
            data.qpos[44] = (
                -np.linalg.norm(
                    R.from_quat(
                        bvh_model.joint_rotation[frame_idx, bvh_model.idx_of["b_r_thumb1"]]
                    ).as_rotvec()
                )
                + 0.1
            )
            data.qpos[45] = (
                -np.linalg.norm(
                    R.from_quat(
                        bvh_model.joint_rotation[frame_idx, bvh_model.idx_of["b_r_thumb2"]]
                    ).as_rotvec()
                )
                + 0.1
            )
            data.qpos[46] = np.linalg.norm(
                R.from_quat(
                    bvh_model.joint_rotation[frame_idx, bvh_model.idx_of["b_r_index1"]]
                ).as_rotvec()
            )
            data.qpos[47] = np.linalg.norm(
                R.from_quat(
                    bvh_model.joint_rotation[frame_idx, bvh_model.idx_of["b_r_index2"]]
                ).as_rotvec()
            )
            data.qpos[48] = np.linalg.norm(
                R.from_quat(
                    bvh_model.joint_rotation[frame_idx, bvh_model.idx_of["b_r_middle1"]]
                ).as_rotvec()
            )
            data.qpos[49] = np.linalg.norm(
                R.from_quat(
                    bvh_model.joint_rotation[frame_idx, bvh_model.idx_of["b_r_middle2"]]
                ).as_rotvec()
            )
            mujoco.mj_forward(model, data)
            left_shoulder_world_pos = data.xpos[model.body("left_shoulder_pitch_link").id]
            right_shoulder_world_pos = data.xpos[model.body("right_shoulder_pitch_link").id]
            if visual:
                visualize_positions(viewer, [left_shoulder_world_pos], [[0, 1, 1, 1]])
                visualize_positions(viewer, [right_shoulder_world_pos], [[0, 1, 1, 1]])
                visualize_positions(viewer, [right_hand_target], [[0, 1, 0, 1]])
                visualize_positions(viewer, [left_hand_target], [[0, 1, 0, 1]])
                visualize_positions(viewer, [right_elbow_link_target], [[0, 1, 0, 1]])
                visualize_positions(viewer, [left_elbow_link_target], [[0, 1, 0, 1]])
                visualize_positions(viewer, [left_shoulder_target], [[1, 0, 0, 1]])
                visualize_positions(viewer, [right_shoulder_target], [[1, 0, 0, 1]])
                visualize_positions(viewer, [right_thumb_target], [[1, 1, 0, 1]])
                visualize_positions(viewer, [left_thumb_target], [[1, 1, 0, 1]])
                visualize_positions(viewer, [right_index_target], [[0, 1, 1, 1]])
                visualize_positions(viewer, [left_index_target], [[0, 1, 1, 1]])
                visualize_positions(viewer, [right_middle_target], [[1, 0, 1, 1]])
                visualize_positions(viewer, [left_middle_target], [[1, 0, 1, 1]])
            joint_angles = np.round(data.qpos.copy(), 6)
            if not initial_convergence and csv_file and (frame_idx >= 0):
                with open(csv_file, mode="a", newline="") as file:
                    writer = csv.writer(file)
                    writer.writerow([frame_idx] + joint_angles.tolist())
            if visual:
                viewer.sync()
                if not initial_convergence:
                    time.sleep(bvh_frame_time)

    finally:
        if visual:
            viewer.close()
    return csv_file
