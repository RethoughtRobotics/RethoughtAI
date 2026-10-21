import numpy as np
import rosbag2_py
from baxter_core_msgs.msg import EndEffectorState
from geometry_msgs.msg import PoseStamped
from rclpy.serialization import serialize_message

from adapter.labels import Episode, chunk_actions, load_measured
from adapter.lerobot_export import features, idle_start

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
