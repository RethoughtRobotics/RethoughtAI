"""adapter.labels.load_plans: each plan the policy sent, at the frame of the observation it was made from."""

import numpy as np
import pytest

rosbag2_py = pytest.importorskip('rosbag2_py')
from builtin_interfaces.msg import Duration, Time  # noqa: E402
from geometry_msgs.msg import Transform, TwistStamped  # noqa: E402
from moveit_msgs.msg import RobotTrajectory  # noqa: E402
from rclpy.serialization import serialize_message  # noqa: E402
from trajectory_msgs.msg import JointTrajectoryPoint, MultiDOFJointTrajectoryPoint  # noqa: E402

from adapter.labels import load_plans  # noqa: E402

S = 1_000_000_000
STEP = S // 25


def plan(stamp, x, steps=3):
    """A chunk observed at `stamp` (ns) going from x in +x, the trigger at 0.5."""
    msg = RobotTrajectory()
    msg.joint_trajectory.header.stamp = Time(sec=stamp // S, nanosec=stamp % S)
    for k in range(steps):
        tf = Transform()
        tf.translation.x, tf.rotation.w = x + 0.01 * k, 1.0
        at = Duration(nanosec=k * STEP)
        msg.multi_dof_joint_trajectory.points.append(MultiDOFJointTrajectoryPoint(transforms=[tf], time_from_start=at))
        msg.joint_trajectory.points.append(JointTrajectoryPoint(positions=[0.5], time_from_start=at))
    return msg


def write(path, plans, nudge=False):
    """plans: (observation time, arrival time, x)."""
    writer = rosbag2_py.SequentialWriter()
    writer.open(rosbag2_py.StorageOptions(uri=str(path), storage_id='mcap'), rosbag2_py.ConverterOptions('', ''))
    writer.create_topic(rosbag2_py.TopicMetadata(0, '/policy/left/chunk', 'moveit_msgs/msg/RobotTrajectory', 'cdr'))
    writer.create_topic(rosbag2_py.TopicMetadata(1, '/policy/left/nudge', 'geometry_msgs/msg/TwistStamped', 'cdr'))
    for observed, arrived, x in plans:
        writer.write('/policy/left/chunk', serialize_message(plan(observed, x)), arrived)
        if nudge:
            writer.write('/policy/left/nudge', serialize_message(TwistStamped()), arrived)
    writer.close()


def test_a_plan_lands_on_its_observation_frame_not_its_arrival(tmp_path):
    t = [S + STEP * k for k in range(20)]
    # observed 5 ms after frame 4 and 15 ms before frame 8, each arriving 0.1 s later
    write(tmp_path / 'bag', [(t[4] + 5_000_000, t[4] + S // 10, 0.1), (t[8] - 15_000_000, t[8] + S // 10, 0.2)])
    plans, planned = load_plans(tmp_path / 'bag', np.array(t))

    assert np.flatnonzero(planned).tolist() == [4, 8]
    np.testing.assert_allclose(plans[4, :, 0], [0.1, 0.11, 0.12])
    np.testing.assert_allclose(plans[8, 0], [0.2, 0, 0, 0, 0, 0, 1, 0.5])  # x y z, quaternion, trigger
    assert not plans[5].any()


def test_a_plan_observed_outside_the_frames_is_left_out(tmp_path):
    t = [S + STEP * k for k in range(5)]
    write(tmp_path / 'bag', [(t[0] - STEP, t[0], 0.1), (t[4] + STEP, t[4] + S // 10, 0.2)])
    assert not load_plans(tmp_path / 'bag', np.array(t))[1].any()


def test_plans_with_a_nudge_adapter_are_refused(tmp_path):
    t = [S + STEP * k for k in range(5)]
    write(tmp_path / 'bag', [(t[1], t[2], 0.1)], nudge=True)
    with pytest.raises(ValueError, match='nudge'):
        load_plans(tmp_path / 'bag', np.array(t))


def test_a_nudge_bags_offset_and_flag_reach_each_frame(tmp_path):
    from geometry_msgs.msg import PoseStamped
    from std_msgs.msg import Bool

    from adapter.labels import load_episode

    writer = rosbag2_py.SequentialWriter()
    path = tmp_path / 'bag'
    writer.open(rosbag2_py.StorageOptions(uri=str(path), storage_id='mcap'), rosbag2_py.ConverterOptions('', ''))
    topics = [
        ('/teleop/left/tcp', 'geometry_msgs/msg/PoseStamped'),
        ('/teleop/left/goal', 'geometry_msgs/msg/PoseStamped'),
        ('/teleop/left/nudging', 'std_msgs/msg/Bool'),
        ('/teleop/left/correction_offset', 'geometry_msgs/msg/TwistStamped'),
    ]
    for i, (name, kind) in enumerate(topics):
        writer.create_topic(rosbag2_py.TopicMetadata(i, name, kind, 'cdr'))
    pose = PoseStamped()
    pose.pose.orientation.w = 1.0
    for k in range(100):  # 1 s at 100 Hz: a nudge of -2 cm in y from 0.5 s
        t = S + k * S // 100
        on = k >= 50
        offset = TwistStamped()
        offset.twist.linear.y = -0.02 if on else 0.0
        writer.write('/teleop/left/tcp', serialize_message(pose), t)
        writer.write('/teleop/left/goal', serialize_message(pose), t)
        writer.write('/teleop/left/nudging', serialize_message(Bool(data=on)), t)
        writer.write('/teleop/left/correction_offset', serialize_message(offset), t)
    writer.close()
    ep = load_episode(path)

    assert ep.offset.shape == (len(ep), 6)
    early, late = ep.t < S + S // 2, ep.t >= S + S // 2
    assert not ep.offset[early].any() and not ep.intervention[early].any()
    np.testing.assert_allclose(ep.offset[late, 1], -0.02)
    assert ep.intervention[late].all()
