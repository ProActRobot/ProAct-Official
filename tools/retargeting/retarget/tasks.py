import mink


def create_tasks(model):
    tasks = [(com_task := mink.ComTask(cost=5.0))]
    left_foot_task = mink.FrameTask(
        frame_name="left_foot",
        frame_type="site",
        position_cost=100.0,
        orientation_cost=5.0,
        lm_damping=1.0,
    )
    tasks.append(left_foot_task)
    right_foot_task = mink.FrameTask(
        frame_name="right_foot",
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
        lm_damping=0.01,
    )
    tasks.append(pelvis_task)
    pelvis_zero_pose = model.body("pelvis").pos
    pelvis_task.set_target(mink.SE3([1.0, 0.0, 0.0, 0.0, *pelvis_zero_pose]))
    right_elbow_link_task = mink.FrameTask(
        frame_name="right_elbow_link",
        frame_type="body",
        position_cost=15.0,
        orientation_cost=0.0,
        lm_damping=0.01,
    )
    tasks.append(right_elbow_link_task)
    left_elbow_link_task = mink.FrameTask(
        frame_name="left_elbow_link",
        frame_type="body",
        position_cost=15.0,
        orientation_cost=0.0,
        lm_damping=0.01,
    )
    tasks.append(left_elbow_link_task)
    right_hand_task = mink.FrameTask(
        frame_name="right_hand_palm",
        frame_type="geom",
        position_cost=5.0,
        orientation_cost=0.0,
        lm_damping=0.01,
    )
    tasks.append(right_hand_task)
    left_hand_task = mink.FrameTask(
        frame_name="left_hand_palm",
        frame_type="geom",
        position_cost=5.0,
        orientation_cost=0.0,
        lm_damping=0.01,
    )
    tasks.append(left_hand_task)
    left_shoulder_task = mink.FrameTask(
        frame_name="left_shoulder_pitch_link",
        frame_type="body",
        position_cost=20.0,
        orientation_cost=0.0,
        lm_damping=0.01,
    )
    tasks.append(left_shoulder_task)
    right_shoulder_task = mink.FrameTask(
        frame_name="right_shoulder_pitch_link",
        frame_type="body",
        position_cost=20.0,
        orientation_cost=0.0,
        lm_damping=0.01,
    )
    tasks.append(right_shoulder_task)
    thumb_tasks = []
    right_thumb_task = mink.FrameTask(
        frame_name="right_hand_thumb_2_link",
        frame_type="body",
        position_cost=5.0,
        orientation_cost=0.0,
        lm_damping=0.01,
    )
    thumb_tasks.append(right_thumb_task)
    left_thumb_task = mink.FrameTask(
        frame_name="left_hand_thumb_2_link",
        frame_type="body",
        position_cost=5.0,
        orientation_cost=0.0,
        lm_damping=0.01,
    )
    thumb_tasks.append(left_thumb_task)
    tasks.extend(thumb_tasks)
    finger_tasks = []
    right_index_task = mink.FrameTask(
        frame_name="right_hand_index_1_link",
        frame_type="body",
        position_cost=5.0,
        orientation_cost=0.0,
        lm_damping=0.01,
    )
    finger_tasks.append(right_index_task)
    left_index_task = mink.FrameTask(
        frame_name="left_hand_index_1_link",
        frame_type="body",
        position_cost=5.0,
        orientation_cost=0.0,
        lm_damping=0.01,
    )
    finger_tasks.append(left_index_task)
    right_middle_task = mink.FrameTask(
        frame_name="right_hand_middle_1_link",
        frame_type="body",
        position_cost=5.0,
        orientation_cost=0.0,
        lm_damping=0.01,
    )
    finger_tasks.append(right_middle_task)
    left_middle_task = mink.FrameTask(
        frame_name="left_hand_middle_1_link",
        frame_type="body",
        position_cost=5.0,
        orientation_cost=0.0,
        lm_damping=0.01,
    )
    finger_tasks.append(left_middle_task)
    tasks.extend(finger_tasks)
    return tasks, {
        "com_task": com_task,
        "left_foot_task": left_foot_task,
        "right_foot_task": right_foot_task,
        "pelvis_task": pelvis_task,
        "right_elbow_link_task": right_elbow_link_task,
        "left_elbow_link_task": left_elbow_link_task,
        "right_hand_task": right_hand_task,
        "left_hand_task": left_hand_task,
        "left_shoulder_task": left_shoulder_task,
        "right_shoulder_task": right_shoulder_task,
        "right_thumb_task": right_thumb_task,
        "left_thumb_task": left_thumb_task,
        "right_index_task": right_index_task,
        "left_index_task": left_index_task,
        "right_middle_task": right_middle_task,
        "left_middle_task": left_middle_task,
    }
