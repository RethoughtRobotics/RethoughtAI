import numpy as np
import rosbag2_py
from baxter_core_msgs.msg import EndEffectorState
from geometry_msgs.msg import PoseStamped
from rclpy.serialization import serialize_message

from adapter.labels import Episode, chunk_actions, load_measured
from adapter.lerobot_export import StillRuns, features, idle_start

SHAPES = {'left': (400, 640, 3)}


def test_one_arm_keeps_its_names_and_two_arms_are_prefixed():
    one = features({'/cam': 'left'}, 'left', SHAPES)
    assert one['observation.state']['shape'] == (15,) and one['action']['names'][0] == 'goal.x'
    two = features({'/cam': 'left'}, ['left', 'right'], SHAPES)
    assert two['observation.state']['shape'] == (30,) and two['action']['shape'] == (16,)
    assert two['action']['names'][8] == 'right.goal.x' and two['observation.state']['names'][8] == 'left.left_s0'
    tools = features({'/cam': 'left'}, ['left', 'right'], SHAPES, gripper=False)  # hands holding tools
    assert tools['observation.state']['shape'] == (28,) and tools['action']['shape'] == (14,)
    assert 'gripper' not in ' '.join(tools['action']['names'] + tools['observation.state']['names'])


def _bag(path):
    writer = rosbag2_py.SequentialWriter()
    writer.open(rosbag2_py.StorageOptions(uri=str(path), storage_id='mcap'), rosbag2_py.ConverterOptions('', ''))
    for i, (topic, kind) in enumerate([
        ('/teleop/left/tcp', 'geometry_msgs/msg/PoseStamped'),
        ('/teleop/right/tcp', 'geometry_msgs/msg/PoseStamped'),
        ('/robot/end_effector/left_gripper/state', 'baxter_core_msgs/msg/EndEffectorState'),
    ]):
        writer.create_topic(rosbag2_py.TopicMetadata(id=i, name=topic, type=kind, serialization_format='cdr'))
    for k in range(100):  # 1 s at 100 Hz: the left TCP moves 1 m along x, the right one 0.5 m along y
        t = 1_000_000_000 + k * 10_000_000
        for arm, axis, speed in (('left', 'x', 1.0), ('right', 'y', 0.5)):
            msg = PoseStamped()
            setattr(msg.pose.position, axis, speed * k / 100)
            msg.pose.orientation.w = 1.0
            writer.write(f'/teleop/{arm}/tcp', serialize_message(msg), t)
        grip = EndEffectorState()
        grip.position = 100.0 - k  # closing
        writer.write('/robot/end_effector/left_gripper/state', serialize_message(grip), t)
    del writer


def test_measured_labels_are_where_the_hand_took_each_arm(tmp_path):
    _bag(tmp_path / 'ep')
    eps = load_measured(tmp_path / 'ep', ['left', 'right'], fps=25)
    left, right = eps['left'], eps['right']
    assert len(left) == len(right) == 25 and (left.t == right.t).all()
    chunk = chunk_actions(left, 0, 5)  # one step ahead, 40 ms steps at 1 m/s: 40, 80 .. mm along x
    assert np.allclose(chunk[:, 0], [0.04, 0.08, 0.12, 0.16, 0.20]) and np.allclose(chunk[:, 1:6], 0)
    assert np.allclose(chunk_actions(right, 0, 2)[:, 1], [0.02, 0.04])
    assert np.allclose(left.target_p[-1], left.tcp_p[-1])  # the last step holds
    assert np.isclose(left.trigger[-1], 1 - (100 - 96) / 100) and (right.trigger == 0).all()  # no right gripper topic


def test_a_kinesthetic_bag_replays_as_an_absolute_policy(tmp_path):
    """No teleop goal: labels are the measured TCP; absolute chunks pull a displaced arm back onto it."""
    from inference.policies import ReplayPolicy

    _bag(tmp_path / 'ep')
    policy = ReplayPolicy(tmp_path / 'ep', 'left', 25.0, 5, absolute=True)
    assert policy.start is None  # absolute goals start where the arm is
    assert np.allclose(ReplayPolicy(tmp_path / 'ep', 'left', 25.0, 5).start[0], [0, 0, 0])
    live = np.array([0.0, 0.01, 0.0])  # the arm 1 cm off the demonstration
    actions = policy({'step': 0, 'tcp_position': live, 'tcp_orientation': np.array([0, 0, 0, 1.0])})
    assert np.allclose(actions[:, 0], [0.0, 0.04, 0.08, 0.12, 0.16])  # the measured TCP, same step on
    assert np.allclose(actions[:, 1], -0.01)  # relative to the live TCP: back onto the demonstration
    relative = ReplayPolicy(tmp_path / 'ep', 'left', 25.0, 5)
    assert np.allclose(relative({'step': 0})[:, 1], 0.0)  # relative to the recorded TCP


def test_measured_labels_are_interpolated_to_each_steps_time(tmp_path):
    """A TCP recorded at 76 Hz is read at each 25 Hz step's exact time, not the latest sample before it."""
    writer = rosbag2_py.SequentialWriter()
    writer.open(
        rosbag2_py.StorageOptions(uri=str(tmp_path / 'ep'), storage_id='mcap'), rosbag2_py.ConverterOptions('', '')
    )
    writer.create_topic(
        rosbag2_py.TopicMetadata(
            id=0, name='/teleop/left/tcp', type='geometry_msgs/msg/PoseStamped', serialization_format='cdr'
        )
    )
    for k in range(76):  # 1 s at 76 Hz, 1 m/s along x
        msg = PoseStamped()
        msg.pose.position.x = k / 76
        msg.pose.orientation.w = 1.0
        writer.write('/teleop/left/tcp', serialize_message(msg), 1_000_000_000 + round(k * 1e9 / 76))
    del writer
    left = load_measured(tmp_path / 'ep', ['left'], fps=25)['left']
    assert np.allclose(np.diff(left.tcp_p[:, 0]), 0.04, atol=1e-6)  # evenly 40 mm per 40 ms step


def _still_then_moving(still, n=20, step=0.01):
    p = np.zeros((n, 3))
    p[still:, 0] = step * np.arange(1, n - still + 1)
    q = np.tile([0.0, 0.0, 0.0, 1.0], (n, 1))
    return Episode(p, q, p.copy(), q.copy(), np.zeros(n), np.arange(n) * 40_000_000, 25.0)


def test_idle_start_keeps_the_last_still_step_of_the_first_arm_to_move():
    # left still for steps 0 to 7, right for 0 to 4: the episode starts at step 4
    eps = {'left': _still_then_moving(8), 'right': _still_then_moving(5)}
    assert idle_start(eps, 0.001, np.radians(1.0)) == 4
    assert idle_start({'left': _still_then_moving(0)}, 0.001, np.radians(1.0)) == 0
    assert idle_start({'left': _still_then_moving(20)}, 0.001, np.radians(1.0)) == 0  # never moves: kept whole


def test_a_still_stretch_keeps_its_first_frame_and_anything_moving():
    still = StillRuns(0.001, np.radians(1.0))
    q = [[0.0, 0.0, 0.0, 1.0]]
    keep = [still.keep([[0.0, 0, 0]], q, [0.0])]  # starts a stretch
    keep += [still.keep([[0.0005 * i, 0, 0]], q, [0.0]) for i in (1, 2)]  # within 1 mm of its start
    keep += [still.keep([[0.0015, 0, 0]], q, [0.0])]  # out: a new stretch starts here
    keep += [still.keep([[0.0015, 0, 0]], q, [0.005])]  # gripper within 0.01: still
    keep += [still.keep([[0.0015, 0, 0]], q, [0.02])]  # the gripper moving is not still
    assert keep == [True, False, False, True, False, True]
