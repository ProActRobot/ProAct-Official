import os
import mujoco
from loop_rate_limiters import RateLimiter
import mink
import numpy as np
from bvh import Bvh
from scipy.spatial.transform import Rotation as R
import time
import csv
import multiprocessing as mp
# from smooth_motion import smooth_single_csv
import warnings
warnings.filterwarnings("ignore", message=".*Converted matrix *")

def get_relative_position(bvh_model, joint_ref, joint_to, frame_idx):
    """
    Compute the relative position from one joint to another at a specific frame.
    """
    positions, _ = bvh_model.frame_pose(frame_idx)

    joint_ref_idx = list(bvh_model.joints.keys()).index(joint_ref)
    joint_to_idx = list(bvh_model.joints.keys()).index(joint_to)

    relative_pos = positions[joint_to_idx] - positions[joint_ref_idx]
    relative_pos = np.array([-relative_pos[0], relative_pos[1], -relative_pos[2]])
    return relative_pos

def get_joint_rotation(bvh_model, joint_name, frame_idx):
    """
    Get the rotation of a joint at a specific frame.
    """
    if joint_name in bvh_model.idx_of:
        return bvh_model.joint_rotation[int(frame_idx), bvh_model.idx_of[joint_name]]
    else:
        return np.array([0, 0, 0, 1])  # Identity quaternion

def update_target(task, bvh_model, joint_ref, joint_to, frame_idx, norm_factor, base_target=None, offset=None):
    """
    Update the target for a given task dynamically.
    :param task: The task to update.
    :param bvh_model: The BVH model.
    :param joint_ref: Reference joint name.
    :param joint_to: Target joint name.
    :param frame_idx: Current frame index (can be float for interpolation).
    :param norm_factor: Normalization factor for scaling.
    :param base_target: Base target position to add (optional).
    :param offset: Offset to apply to the target (optional).
    """
    target = get_relative_position(bvh_model, joint_ref, joint_to, frame_idx)
    target = target / 100 * norm_factor
    # target = [target[2], target[0], target[1]] # to make +y to +x
    target = [-target[0], target[2], target[1]]
    if base_target is not None:
        target = np.array(target) + base_target
    if offset is not None:
        target = np.array(target) + np.array(offset)
    task.set_target(mink.lie.SE3([0.1, 0.0, 0.0, 0.0, target[0], target[1], target[2]]))
    return target

def do_retarget(bvh_model, model, max_frequency=200.0, bvh_frame_time=0.033333, start_frame=0, end_frame=None):
    configuration = mink.Configuration(model)
    tasks = [
        com_task := mink.ComTask(cost=5.0),
    ]
    left_foot_task = mink.FrameTask(
        frame_name='left_foot',
        frame_type="site",
        position_cost=100.0,
        orientation_cost=5.0,
        lm_damping=1.0,
    )
    tasks.append(left_foot_task)
    right_foot_task = mink.FrameTask(
        frame_name='right_foot',
        frame_type="site",
        position_cost=100.0,
        orientation_cost=5.0,
        lm_damping=1.0,
    )
    tasks.append(right_foot_task)
    pelvis_task = mink.FrameTask(
        frame_name="pelvis",
        frame_type="body",
        position_cost=50.0,
        orientation_cost=50.0,
        lm_damping=1e-2,
    )
    tasks.append(pelvis_task)

    pelvis_zero_pose = model.body("pelvis").pos
    pelvis_task.set_target(mink.lie.SE3([1.0, 0.0, 0.0, 0.0, *pelvis_zero_pose]))

    right_elbow_link_task = mink.FrameTask(
        frame_name="right_elbow_link",
        frame_type="body",
        position_cost=15.0,
        orientation_cost=0.0,
        lm_damping=1e-2,
    )
    tasks.append(right_elbow_link_task)

    left_elbow_link_task = mink.FrameTask(
        frame_name="left_elbow_link",
        frame_type="body",
        position_cost=15.0,
        orientation_cost=0.0,
        lm_damping=1e-2,
    )
    tasks.append(left_elbow_link_task)

    right_hand_task = mink.FrameTask(
        frame_name="right_hand_palm",
        frame_type="geom",
        position_cost=5.0,
        orientation_cost=0.0,
        lm_damping=1e-2,
    )
    tasks.append(right_hand_task)

    left_hand_task = mink.FrameTask(
        frame_name="left_hand_palm",
        frame_type="geom",
        position_cost=5.0,
        orientation_cost=0.0,
        lm_damping=1e-2,
    )
    tasks.append(left_hand_task)

    left_shoulder_task = mink.FrameTask(
        frame_name="left_shoulder_pitch_link",
        frame_type="body",
        position_cost=20.0,
        orientation_cost=0.0,
        lm_damping=1e-2,
    )
    tasks.append(left_shoulder_task)

    right_shoulder_task = mink.FrameTask(
        frame_name="right_shoulder_pitch_link",
        frame_type="body",
        position_cost=20.0,
        orientation_cost=0.0,
        lm_damping=1e-2,
    )
    tasks.append(right_shoulder_task)

    thumb_tasks = []

    right_thumb_task = mink.FrameTask(
        frame_name="right_hand_thumb_2_link",
        frame_type="body",
        position_cost=5.0,
        orientation_cost=0.0,
        lm_damping=1e-2,
    )
    thumb_tasks.append(right_thumb_task)

    left_thumb_task = mink.FrameTask(
        frame_name="left_hand_thumb_2_link",
        frame_type="body",
        position_cost=5.0,
        orientation_cost=0.0,
        lm_damping=1e-2,
    )
    thumb_tasks.append(left_thumb_task)

    tasks.extend(thumb_tasks)

    finger_tasks = []

    right_index_task = mink.FrameTask(
        frame_name="right_hand_index_1_link",
        frame_type="body",
        position_cost=5.0,
        orientation_cost=0.0,
        lm_damping=1e-2,
    )
    finger_tasks.append(right_index_task)

    left_index_task = mink.FrameTask(
        frame_name="left_hand_index_1_link",
        frame_type="body",
        position_cost=5.0,
        orientation_cost=0.0,
        lm_damping=1e-2,
    )
    finger_tasks.append(left_index_task)

    right_middle_task = mink.FrameTask(
        frame_name="right_hand_middle_1_link",
        frame_type="body",
        position_cost=5.0,
        orientation_cost=0.0,
        lm_damping=1e-2,
    )
    finger_tasks.append(right_middle_task)

    left_middle_task = mink.FrameTask(
        frame_name="left_hand_middle_1_link",
        frame_type="body",
        position_cost=5.0,
        orientation_cost=0.0,
        lm_damping=1e-2,
    )
    finger_tasks.append(left_middle_task)

    tasks.extend(finger_tasks)

    collision_pairs = [
        # (["left_hand_collision", "left_thumb_collision", "left_index_collision", "left_middle_collision"], ["right_hand_collision", "right_thumb_collision", "right_index_collision", "right_middle_collision"]),
        # (["left_hand_collision", "left_thumb_collision", "left_index_collision", "left_middle_collision"], ["head_collision", "left_thigh_collision", "right_thigh_collision"]),
        # (["right_hand_collision", "right_thumb_collision", "right_index_collision", "right_middle_collision"], ["head_collision", "left_thigh_collision", "right_thigh_collision"]),
        # (["left_hand_collision", "left_elbow_collision"], ["right_hand_collision", "right_elbow_collision"]),
        # (["left_hand_collision"], ["left_thigh_collision", "right_thigh_collision"]),
        # (["right_hand_collision"], ["left_thigh_collision", "right_thigh_collision"]),
    ]
    # collision_avoidance_limit = mink.CollisionAvoidanceLimit(
    #     model=model,
    #     geom_pairs=collision_pairs,
    #     minimum_distance_from_collisions=0.02,
    #     collision_detection_distance=0.2,
    # )

    limits = [
        mink.ConfigurationLimit(model),
        # collision_avoidance_limit,
    ]

    com_mid = model.body("com_target").mocapid[0]

    model = configuration.model
    data = configuration.data
    solver = "osqp"

    # configuration.update_from_keyframe("teleop")
    mink.move_mocap_to_frame(model, data, "head_target", "head", "site")
    data.mocap_pos[com_mid] = data.subtree_com[1]
    rate = RateLimiter(frequency=max_frequency, warn=False)
    '''
    class bvh_model has xzy order, pos[0], -pos[2], pos[1] is the correct x, y, z in BVH.
    in ZeroEGGS, human is facing -y direction, with +z up, +x left hand side.
    in photoreal pxb, human is facing +y direction, with +z up, -x left hand side.
    in xml default posture, robot is facing +x direction, with +z up, +y left hand side.
    to align with mujoco robot, we use the data as pos[2], pos[0], pos[1] as x, y, z.
    '''
    # from joint: RightShoulder, to joint: RightElbow is the upper right arm. the length is ~ 36

    # Calculate the distance between right_shoulder_pitch_link and right_elbow_link in the rest pose using world frame positions
    shoulder_world_pos = data.xpos[model.body("right_shoulder_pitch_link").id]
    # print(shoulder_world_pos)
    elbow_world_pos = data.xpos[model.body("right_elbow_link").id]
    # print(elbow_world_pos)
    distance = np.linalg.norm(elbow_world_pos - shoulder_world_pos)
    # print(f"World frame distance between right_shoulder_pitch_link and right_elbow_link: {distance}")
    # Calculate the distance between left_shoulder_pitch_link and left_elbow_link in the rest pose using world frame positions
    shoulder_world_pos = data.xpos[model.body("left_shoulder_pitch_link").id]
    # print(shoulder_world_pos)
    elbow_world_pos = data.xpos[model.body("left_elbow_link").id]
    # print(elbow_world_pos)
    distance = np.linalg.norm(elbow_world_pos - shoulder_world_pos)
    # print(f"World frame distance between left_shoulder_pitch_link and left_elbow_link: {distance}")
    # print(data.xpos[model.body("pelvis").id])

    # Calculate the direction the robot's head is facing in the world frame in the rest pose
    head_world_pos = data.xpos[model.body("pelvis").id]
    head_world_xmat = data.xmat[model.body("pelvis").id].reshape(3, 3)  # Rotation matrix
    head_facing_direction = head_world_xmat[:, 0]  # The x-axis of the head in the world frame
    # print(f"Head world position: {head_world_pos}")
    # print(f"Head facing direction in world frame: {head_facing_direction}")

    # print(f"In world frame:")
    # print(f"right_ankle_roll_link: {data.xpos[model.body("right_ankle_roll_link").id]}")
    # print(f"left_ankle_roll_link: {data.xpos[model.body("left_ankle_roll_link").id]}")

    frame_idx = 0

    # calc_start_time = time.time()
    # start_time = time.time()
    initial_convergence = True
    frame_idx = 0
    motions = []
    total_frame_idx = 0  # Track actual frame index for non-visual mode
    init_converge_frames = 30 # constant
    while True:
        if initial_convergence:
            if total_frame_idx == init_converge_frames:
                initial_convergence = False
                frame_idx = start_frame
                total_frame_idx = 0
            else:
                total_frame_idx += 1
        else:
            total_frame_idx += 1
            frame_idx = total_frame_idx + start_frame
            if frame_idx >= min(end_frame, bvh_model.frames):
                break
        joint_angles = data.qpos
        joint_angles = np.round(joint_angles, 6)
        left_foot_zero_pose = data.xpos[model.body("left_ankle_roll_link").id]
        right_foot_zero_pose = data.xpos[model.body("right_ankle_roll_link").id]
        left_foot_task.set_target(mink.lie.SE3([1.0, 0.0, 0.0, 0.0, *left_foot_zero_pose]))
        right_foot_task.set_target(mink.lie.SE3([1.0, 0.0, 0.0, 0.0, *right_foot_zero_pose]))

        robot_lower_arm_norm = 36 / 36
        right_hand_target = update_target(right_hand_task, bvh_model, "b_r_forearm", "b_r_wrist", frame_idx, robot_lower_arm_norm, data.xpos[model.body("right_elbow_link").id], [0.0, -0.03, 0.0])
        left_hand_target = update_target(left_hand_task, bvh_model, "b_l_forearm", "b_l_wrist", frame_idx, robot_lower_arm_norm, data.xpos[model.body("left_elbow_link").id], [0.0, 0.03, 0.0])

        robot_upper_arm_norm = 25 / 36
        right_elbow_link_target = update_target(right_elbow_link_task, bvh_model, "b_r_arm", "b_r_forearm", frame_idx, robot_upper_arm_norm, data.xpos[model.body("right_shoulder_pitch_link").id], [0.0, -0.08, 0.0])
        left_elbow_link_target = update_target(left_elbow_link_task, bvh_model, "b_l_arm", "b_l_forearm", frame_idx, robot_upper_arm_norm, data.xpos[model.body("left_shoulder_pitch_link").id], [0.0, 0.08, 0.0])

        robot_shoulder_norm = 20 / 36
        left_shoulder_target = update_target(left_shoulder_task, bvh_model, "b_root", "b_l_arm", frame_idx, robot_shoulder_norm, data.xpos[model.body("pelvis").id])
        right_shoulder_target = update_target(right_shoulder_task, bvh_model, "b_root", "b_r_arm", frame_idx, robot_shoulder_norm, data.xpos[model.body("pelvis").id])

        robot_thumb_norm = 20 / 36
        right_thumb_target = update_target(right_thumb_task, bvh_model, "b_r_wrist", "b_r_thumb1", frame_idx, robot_thumb_norm, right_hand_target)
        left_thumb_target = update_target(left_thumb_task, bvh_model, "b_l_wrist", "b_l_thumb1", frame_idx, robot_thumb_norm, left_hand_target)

        robot_index_norm = 20 / 36
        right_index_target = update_target(right_index_task, bvh_model, "b_r_wrist", "b_r_index2", frame_idx, robot_index_norm, right_hand_target)
        left_index_target = update_target(left_index_task, bvh_model, "b_l_wrist", "b_l_index2", frame_idx, robot_index_norm, left_hand_target)

        robot_middle_norm = 20 / 36
        right_middle_target = update_target(right_middle_task, bvh_model, "b_r_wrist", "b_r_middle2", frame_idx, robot_middle_norm, right_hand_target)
        left_middle_target = update_target(left_middle_task, bvh_model, "b_l_wrist", "b_l_middle2", frame_idx, robot_middle_norm, left_hand_target)

        com_task.set_target(data.mocap_pos[com_mid])

        vel = mink.solve_ik(configuration, tasks, rate.dt, solver, 1e-2, limits=limits)
        configuration.integrate_inplace(vel, rate.dt)

        data.qpos[0:19] = 0.0
        data.qpos[2] = 0.79
        data.qpos[3] = 1.0
        # data.qpos[25] = 1.5 - np.linalg.norm(R.from_quat(bvh_model.joint_rotation[frame_idx, bvh_model.idx_of['b_l_forearm']]).as_rotvec())
        # data.qpos[39] = 1.5 - np.linalg.norm(R.from_quat(bvh_model.joint_rotation[frame_idx, bvh_model.idx_of['b_r_forearm']]).as_rotvec())
        # data.qpos[28] = data.qpos[28] - 0.05
        data.qpos[29] = 0
        data.qpos[30] = np.clip(np.linalg.norm(R.from_quat(get_joint_rotation(bvh_model, 'b_l_thumb1', frame_idx)).as_rotvec()) - 0.1, -0.724312, 1.0472)
        data.qpos[31] = np.clip(np.linalg.norm(R.from_quat(get_joint_rotation(bvh_model, 'b_l_thumb2', frame_idx)).as_rotvec()) - 0.1, 0, 1.74533)
        data.qpos[32] = np.clip(-np.linalg.norm(R.from_quat(get_joint_rotation(bvh_model, 'b_l_middle1', frame_idx)).as_rotvec()), -1.5708, 0)
        data.qpos[33] = np.clip(-np.linalg.norm(R.from_quat(get_joint_rotation(bvh_model, 'b_l_middle2', frame_idx)).as_rotvec()), -1.74533, 0)
        data.qpos[34] = np.clip(-np.linalg.norm(R.from_quat(get_joint_rotation(bvh_model, 'b_l_index1', frame_idx)).as_rotvec()), -1.5708, 0)
        data.qpos[35] = np.clip(-np.linalg.norm(R.from_quat(get_joint_rotation(bvh_model, 'b_l_index2', frame_idx)).as_rotvec()), -1.74533, 0)
        # data.qpos[42] = data.qpos[42] + 0.05
        data.qpos[43] = 0
        data.qpos[44] = np.clip(-np.linalg.norm(R.from_quat(get_joint_rotation(bvh_model, 'b_r_thumb1', frame_idx)).as_rotvec()) + 0.1, -1.0472, 0.724312)
        data.qpos[45] = np.clip(-np.linalg.norm(R.from_quat(get_joint_rotation(bvh_model, 'b_r_thumb2', frame_idx)).as_rotvec()) + 0.1, -1.74533, 0)
        data.qpos[46] = np.clip(np.linalg.norm(R.from_quat(get_joint_rotation(bvh_model, 'b_r_index1', frame_idx)).as_rotvec()), 0, 1.5708)
        data.qpos[47] = np.clip(np.linalg.norm(R.from_quat(get_joint_rotation(bvh_model, 'b_r_index2', frame_idx)).as_rotvec()), 0, 1.74533)
        data.qpos[48] = np.clip(np.linalg.norm(R.from_quat(get_joint_rotation(bvh_model, 'b_r_middle1', frame_idx)).as_rotvec()), 0, 1.5708)
        data.qpos[49] = np.clip(np.linalg.norm(R.from_quat(get_joint_rotation(bvh_model, 'b_r_middle2', frame_idx)).as_rotvec()), 0, 1.74533)
        mujoco.mj_forward(model, data)

        # Include 7 DOF for floating_base_joint
        joint_angles = np.round(joint_angles, 6)
        if not initial_convergence:
            motions.append([frame_idx] + joint_angles.tolist())
        # rate.sleep()

    # calc_end_time = time.time()
    # print(f'average time per frame: {(calc_end_time - calc_start_time) / bvh_model.frames:.6f} seconds')
    return motions

def process_segment(bvh_model, model, start_frame, end_frame):
    return do_retarget(bvh_model, model, max_frequency=1000.0, bvh_frame_time=0.033333, start_frame=start_frame, end_frame=end_frame)

def retarget_outer(bvh_file, csv_file):
    print(f'start retargeting {bvh_file}')
    calc_start_time = time.time()

    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
    xml_path = os.getenv(
        "PROACT_RETARGET_MODEL",
        os.path.join(repo_root, "assets", "g1", "scene_empty_coll.xml"),
    )
    print(f"Using XML path: {xml_path}")
    model = mujoco.MjModel.from_xml_path(xml_path)
    output_dir = os.path.dirname(csv_file)
    name1 = os.path.splitext(os.path.basename(csv_file))[0] if csv_file else "unknown"
    wait_file = os.path.join(output_dir, f"{name1}.wait")
    with open(wait_file, "w") as f:
        f.write("Processing started.")

    bvh_model = Bvh()
    bvh_model.parse_file(bvh_file)

    if csv_file:
        with open(csv_file, mode="w", newline="") as file:
            writer = csv.writer(file)
            # Write the header (joint names)
            joint_names = [model.joint(j).name for j in range(model.njnt)]
            # Adjust header for floating_base_joint with 7 DOF
            adjusted_header = ["frame_idx"] + ["floating_base_joint_dof_" + str(i + 1) for i in range(7)] + joint_names[1:]
            writer.writerow(adjusted_header)

    total_frames = bvh_model.frames
    num_processes = 4
    frames_per_process = total_frames // num_processes

    # Calculate explicit start and end frames for each process
    process_ranges = []
    for i in range(num_processes):
        start_frame = i * frames_per_process
        if i == num_processes - 1:
            # Last process handles any remaining frames
            end_frame = total_frames
        else:
            end_frame = (i + 1) * frames_per_process
        process_ranges.append((start_frame, end_frame))
    print(process_ranges)
    # Use multiprocessing to process each segment
    with mp.Pool(processes=num_processes) as pool:
        results = pool.starmap(process_segment, [(bvh_model, model, start_frame, end_frame) for start_frame, end_frame in process_ranges])

    # Combine results
    motions = [motion for segment in results for motion in segment]

    # End timer
    calc_end_time = time.time()
    print(f'average time per frame: {(calc_end_time - calc_start_time) / bvh_model.frames:.6f} seconds')

    with open(csv_file, mode="a", newline="") as file:
        writer = csv.writer(file)
        writer.writerows(motions)

    # remove wait_file
    if os.path.exists(wait_file):
        os.remove(wait_file)
