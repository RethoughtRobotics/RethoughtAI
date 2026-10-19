"""Offline safety tests: real ROS messages, no rclpy.init or robot interfaces."""

from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pytest
from baxter_core_msgs.msg import DigitalIOState
from builtin_interfaces.msg import Duration, Time
from geometry_msgs.msg import PoseStamped, Transform
from moveit_msgs.msg import RobotTrajectory
from scipy.spatial.transform import Rotation
from sensor_msgs.msg import Joy
from std_msgs.msg import Bool, Empty, Float32
from std_srvs.srv import SetBool
from trajectory_msgs.msg import JointTrajectoryPoint, MultiDOFJointTrajectoryPoint

from inference.bimanual_runtime import BimanualRuntime
from inference.runtime import SETTLE, TCP_STALE, Runtime

CFG = {
    'arm': 'both',
    'fps': 25,
    'chunk_size': 50,
    'publish_rate': 100,
    'entry_ramp': 0,
    'chunk': '/policy/both/chunk',
    'engage': '/policy/both/engage',
    'engaged': '/policy/both/engaged',
    'deadman': {'topic': '/joy', 'frame_id': 'controller', 'button': 0},
    'arms': {
        name: {key: f'/{name}/{key}' for key in ('tcp', 'target', 'trigger', 'cuff', 'home')}
        for name in ('left', 'right')
    },
}


def stamp(seconds, cls=Time):
    ns = round(seconds * 1e9)
    return cls(sec=ns // 1_000_000_000, nanosec=ns % 1_000_000_000)


class FakeNode:
    def __init__(self):
        self.time = 10.0
        self.subscriptions, self.publishers, self.services, self.timers = {}, {}, {}, []
        self.publisher_types, self.logs = {}, []

    def get_clock(self):
        def now():
            seconds = self.time
            return SimpleNamespace(nanoseconds=round(seconds * 1e9), to_msg=lambda: stamp(seconds))

        return SimpleNamespace(now=now)

    def get_logger(self):
        return SimpleNamespace(info=self.logs.append)

    def create_subscription(self, cls, topic, callback, qos):
        assert topic not in self.subscriptions
        self.subscriptions[topic] = callback
        return callback

    def create_publisher(self, cls, topic, qos):
        assert topic not in self.publishers
        self.publisher_types[topic] = cls
        self.publishers[topic] = []
        return SimpleNamespace(publish=lambda msg: self.publishers[topic].append(deepcopy(msg)))

    def create_service(self, cls, topic, callback):
        assert topic not in self.services
        self.services[topic] = callback
        return callback

    def create_timer(self, period, callback):
        self.timers.append((period, callback))
        return callback

    def send(self, topic, msg):
        self.subscriptions[topic](msg)


def tcp(x):
    msg = PoseStamped()
    msg.pose.position.x = float(x)
    msg.pose.orientation.w = 1.0
    return msg


def refresh(node, names=('left', 'right'), positions=(0.5, 0.8)):
    for name in names:
        node.send(CFG['arms'][name]['tcp'], tcp(positions[('left', 'right').index(name)]))


def request(node, value):
    return node.services[CFG['engage']](SetBool.Request(data=value), SetBool.Response())


def joy(value, frame='controller'):
    msg = Joy(buttons=[int(value)])
    msg.header.frame_id = frame
    return msg


def make_runtime(cfg=None, merge=None):
    node = FakeNode()
    rt = BimanualRuntime(node, deepcopy(cfg or CFG), merge=merge)
    refresh(node)
    rt._tick()
    return rt, node


def engage(rt, node, service=True):
    if service:
        assert request(node, True).success
    else:
        node.send('/joy', joy(True))
    node.time += SETTLE + 0.01
    refresh(node)
    rt._tick()
    assert rt.engaged and all(a.engaged for a in rt.arms.values())
    assert rt.accept_after == pytest.approx(node.time)


def chunk(t0, offsets=(0.0, 0.2, 0.4, 0.8), shift=0.0):
    msg = RobotTrajectory()
    msg.joint_trajectory.header.stamp = stamp(t0)
    msg.multi_dof_joint_trajectory.header.stamp = stamp(t0)
    msg.multi_dof_joint_trajectory.joint_names = ['left_tcp', 'right_tcp']
    for offset in offsets:
        transforms = []
        for x in (0.5 + shift + offset * 0.1, 0.8 + 2 * shift + offset * 0.2):
            tf = Transform()
            tf.translation.x, tf.rotation.w = float(x), 1.0
            transforms.append(tf)
        msg.multi_dof_joint_trajectory.points.append(
            MultiDOFJointTrajectoryPoint(transforms=transforms, time_from_start=stamp(offset, Duration))
        )
    return msg


def deliver(node, msg):
    node.send(CFG['chunk'], msg)


def sent(node, name):
    msg = node.publishers[CFG['arms'][name]['target']][-1]
    p = msg.pose.position
    return np.array([p.x, p.y, p.z])


def statuses(node):
    return [msg.data for msg in node.publishers[CFG['engaged']]]


def assert_cleared(rt):
    assert not rt.engaged and np.isinf(rt.accept_after)
    for arm in rt.arms.values():
        assert not arm.engaged and not arm.chunks
        assert arm.segment is None and arm.entry is None and arm.ref is None


def test_only_shared_ros_controls_and_no_gripper_publishers():
    rt, node = make_runtime()
    assert set(node.services) == {CFG['engage']}
    assert len(node.timers) == 1 and node.timers[0][0] == pytest.approx(0.01)
    assert set(node.subscriptions) == {
        '/joy',
        CFG['chunk'],
        *(arm[key] for arm in CFG['arms'].values() for key in ('tcp', 'home', 'cuff')),
    }
    assert set(node.publisher_types.values()) == {PoseStamped, Bool}
    assert Float32 not in node.publisher_types.values()
    assert statuses(node) == [False]
    engage(rt, node)
    assert statuses(node) == [False, True]


@pytest.mark.parametrize('missing', ['left', 'right'])
def test_engagement_requires_both_fresh_tcps_and_shared_settling(missing):
    rt, node = make_runtime()
    request(node, True)
    node.time += 1.0
    refresh(node, names=tuple(name for name in rt.arms if name != missing))
    rt._tick()
    assert not rt.engaged
    refresh(node)
    rt._tick()
    assert not rt.engaged
    node.time += SETTLE - 0.01
    refresh(node)
    rt._tick()
    assert not rt.engaged
    node.time += 0.02
    refresh(node)
    rt._tick()
    assert rt.engaged
    assert {arm.accept_after for arm in rt.arms.values()} == {rt.accept_after}


def test_atomic_chunk_interpolation_and_reference_mapping():
    rt, node = make_runtime()
    engage(rt, node)
    deliver(node, chunk(node.time))
    left, right = rt.arms.values()
    assert len(left.chunks) == len(right.chunks) == 1
    assert left.chunks[0].arrival == right.chunks[0].arrival
    assert np.array_equal(left.chunks[0].t, right.chunks[0].t)
    assert left.chunks[0].triggers is right.chunks[0].triggers is None
    rt._tick()
    node.time += 0.04  # Runtime finishes its initial hold segment first.
    refresh(node)
    rt._tick()
    node.time += 0.02  # Half a policy step, but the same tick for both arms.
    refresh(node)
    rt._tick()
    assert sent(node, 'left')[0] == pytest.approx(0.506, abs=2e-6)
    assert sent(node, 'right')[0] == pytest.approx(0.812, abs=2e-6)
    assert left.segment[0] == right.segment[0] and left.step == right.step
    target_msgs = [node.publishers[CFG['arms'][n]['target']][-1] for n in rt.arms]
    assert target_msgs[0].header.stamp == target_msgs[1].header.stamp
    assert left.ref[0][0] == pytest.approx(0.5)


@pytest.mark.parametrize('stale', ['left', 'right'])
def test_stale_releases_both_same_tick_then_recovers_from_fresh_chunks(stale):
    rt, node = make_runtime()
    engage(rt, node)
    old = chunk(node.time)
    deliver(node, old)
    rt._tick()
    node.time += 0.1
    refresh(node)
    rt._tick()
    frozen = {name: sent(node, name).copy() for name in rt.arms}
    node.time += TCP_STALE + 0.01
    refresh(node, names=tuple(name for name in rt.arms if name != stale))
    rt._tick()
    assert_cleared(rt)
    assert statuses(node)[-1] is False
    for name in rt.arms:
        assert np.array_equal(sent(node, name), frozen[name])
    deliver(node, old)
    assert all(not arm.chunks for arm in rt.arms.values())
    refresh(node, positions=(0.3, 1.0))
    rt._tick()
    assert not rt.engaged
    node.time += SETTLE + 0.01
    refresh(node, positions=(0.3, 1.0))
    rt._tick()
    assert rt.engaged
    deliver(node, old)  # Pre-recovery inference cannot execute.
    assert all(not arm.chunks for arm in rt.arms.values())
    for name, actual in zip(rt.arms, (0.3, 1.0)):
        arm = rt.arms[name]
        assert arm.ref[0][0] == pytest.approx(actual)
        assert sent(node, name) == pytest.approx(frozen[name], abs=2e-6)
    deliver(node, chunk(node.time))
    assert all(len(arm.chunks) == 1 for arm in rt.arms.values())
    rt._tick()
    node.time += 0.05
    refresh(node, positions=(0.3, 1.0))
    rt._tick()
    for arm in rt.arms.values():
        goal = arm.ref[0] + arm.published[0] - arm.ref[2]
        assert goal == pytest.approx(arm.last_goal[0], abs=2e-6)


def test_service_release_clears_immediately_and_targets_do_not_follow_tcp():
    rt, node = make_runtime()
    engage(rt, node)
    deliver(node, chunk(node.time))
    rt._tick()
    frozen = {name: sent(node, name).copy() for name in rt.arms}
    assert request(node, False).message == 'released'
    assert_cleared(rt)
    for _ in range(4):
        node.time += 0.1
        refresh(node, positions=(0.2, 1.5))
        rt._tick()
        for name in rt.arms:
            assert np.array_equal(sent(node, name), frozen[name])
    assert statuses(node) == [False, True, False]


def test_single_deadman_controls_both_and_ignores_other_controller():
    rt, node = make_runtime()
    node.send('/joy', joy(True, 'wrong'))
    rt._tick()
    assert not rt.held and not rt.engaged
    engage(rt, node, service=False)
    node.send('/joy', joy(False))
    assert_cleared(rt)
    assert statuses(node) == [False, True, False]


def test_deadman_alive_timeout_releases_both():
    cfg = deepcopy(CFG)
    cfg['deadman']['alive'] = '/controller_pose'
    rt, node = make_runtime(cfg)
    node.send('/joy', joy(True))
    node.time += SETTLE + 0.01
    refresh(node)
    node.send('/controller_pose', tcp(0))
    rt._tick()
    assert rt.engaged
    node.time += TCP_STALE + 0.01
    refresh(node)
    rt._tick()
    assert_cleared(rt)


@pytest.mark.parametrize('home_arm', ['left', 'right'])
def test_home_either_arm_resets_shared_policy_without_gripper_write(home_arm):
    rt, node = make_runtime()
    engage(rt, node)
    old = chunk(node.time)
    deliver(node, old)
    rt._tick()
    for arm in rt.arms.values():
        arm.trigger = 1.0  # Even inherited trigger state must not leak to ROS.
    node.send(CFG['arms'][home_arm]['home'], Empty())
    assert_cleared(rt)
    assert rt.requested and statuses(node) == [False, True, False]
    rt._tick()
    assert not rt.engaged
    node.time += SETTLE + 0.01
    refresh(node)
    rt._tick()
    assert rt.engaged
    deliver(node, old)
    assert all(not arm.chunks for arm in rt.arms.values())
    deliver(node, chunk(node.time))
    rt._tick()
    assert statuses(node) == [False, True, False, True]
    assert Float32 not in node.publisher_types.values()


@pytest.mark.parametrize('cuff_arm', ['left', 'right'])
def test_cuff_either_arm_discards_inflight_and_waits_for_both_released(cuff_arm):
    rt, node = make_runtime()
    engage(rt, node)
    old = chunk(node.time)
    deliver(node, old)
    rt._tick()
    other = 'right' if cuff_arm == 'left' else 'left'
    node.send(CFG['arms'][cuff_arm]['cuff'], DigitalIOState(state=1))
    assert_cleared(rt)
    node.send(CFG['arms'][other]['cuff'], DigitalIOState(state=1))
    node.time += 0.5
    refresh(node, positions=(0.3, 1.1))
    deliver(node, chunk(node.time))
    rt._tick()
    assert_cleared(rt)
    node.send(CFG['arms'][cuff_arm]['cuff'], DigitalIOState(state=0))
    node.time += 0.5
    refresh(node)
    rt._tick()
    assert_cleared(rt)
    node.send(CFG['arms'][other]['cuff'], DigitalIOState(state=0))
    rt._tick()
    assert not rt.engaged
    node.time += SETTLE + 0.01
    refresh(node, positions=(0.3, 1.1))
    rt._tick()
    assert rt.engaged
    deliver(node, old)
    assert all(not arm.chunks and not arm.correcting for arm in rt.arms.values())
    deliver(node, chunk(node.time))
    assert all(len(arm.chunks) == 1 for arm in rt.arms.values())


@pytest.mark.parametrize(
    'fault',
    [
        'names',
        'reverse_names',
        'missing_transform',
        'extra_transform',
        'few_points',
        'nan_left',
        'nan_right',
        'inf_position',
        'zero_quaternion',
        'unnormalized',
        'repeated_time',
        'reverse_time',
        'negative_time',
        'expired',
        'pre_engage',
        'future_stamp',
        'mismatched_stamp',
        'joint_points',
        'invalid_tail',
        'invalid_nanosecond',
        'negative_stamp',
        'expired_horizon',
    ],
)
def test_malformed_chunk_is_rejected_atomically(fault):
    rt, node = make_runtime(merge={'type': 'latest', 'horizon': 2})
    engage(rt, node)
    msg = chunk(node.time)
    pts = msg.multi_dof_joint_trajectory.points
    if fault == 'names':
        msg.multi_dof_joint_trajectory.joint_names = ['left_tcp']
    elif fault == 'reverse_names':
        msg.multi_dof_joint_trajectory.joint_names.reverse()
    elif fault == 'missing_transform':
        pts[1].transforms.pop()
    elif fault == 'extra_transform':
        pts[1].transforms.append(Transform())
    elif fault == 'few_points':
        msg.multi_dof_joint_trajectory.points = pts[:1]
    elif fault in ('nan_left', 'nan_right'):
        pts[1].transforms[int(fault == 'nan_right')].rotation.x = float('nan')
    elif fault == 'inf_position':
        pts[1].transforms[1].translation.x = float('inf')
    elif fault in ('zero_quaternion', 'unnormalized'):
        pts[1].transforms[1].rotation.w = 0.0 if fault == 'zero_quaternion' else 2.0
    elif fault == 'repeated_time':
        pts[1].time_from_start = pts[0].time_from_start
    elif fault == 'reverse_time':
        pts[2].time_from_start = stamp(0.1, Duration)
    elif fault == 'negative_time':
        pts[0].time_from_start = Duration(sec=-1)
    elif fault == 'expired':
        node.time += 1.0
        refresh(node)
    elif fault == 'pre_engage':
        msg = chunk(node.time - 0.01)
    elif fault == 'future_stamp':
        msg = chunk(node.time + 0.01)
    elif fault == 'mismatched_stamp':
        msg.multi_dof_joint_trajectory.header.stamp = stamp(node.time - 1)
    elif fault == 'joint_points':
        msg.joint_trajectory.points = [JointTrajectoryPoint(positions=[1.0])] * len(pts)
    elif fault == 'invalid_tail':
        pts[-1].transforms[1].rotation.w = 0.0
    elif fault == 'invalid_nanosecond':
        pts[1].time_from_start.nanosec = 1_000_000_000
    elif fault == 'negative_stamp':
        msg.joint_trajectory.header.stamp.sec = -1
    elif fault == 'expired_horizon':
        node.time += 0.3  # Full message still covers now, clipped horizon does not.
        refresh(node)
    deliver(node, msg)
    assert all(not arm.chunks for arm in rt.arms.values())


def test_rejected_chunk_preserves_previous_complete_pair():
    rt, node = make_runtime()
    engage(rt, node)
    deliver(node, chunk(node.time))
    before = [arm.chunks[0] for arm in rt.arms.values()]
    bad = chunk(node.time)
    bad.multi_dof_joint_trajectory.points[-1].transforms = []
    deliver(node, bad)
    assert [arm.chunks[0] for arm in rt.arms.values()] == before
    assert all(len(arm.chunks) == 1 for arm in rt.arms.values())


@pytest.mark.parametrize('missing', ['left', 'right'])
def test_no_coverage_on_either_arm_holds_both_even_mid_segment(missing):
    rt, node = make_runtime()
    engage(rt, node)
    deliver(node, chunk(node.time))
    rt._tick()
    node.time += 0.01
    refresh(node)
    rt._tick()
    frozen = {name: sent(node, name).copy() for name in rt.arms}
    rt.arms[missing].chunks.clear()
    node.time += 0.01
    refresh(node)
    rt._tick()
    for name in rt.arms:
        assert sent(node, name) == pytest.approx(frozen[name], abs=2e-6)
    assert rt.engaged
    assert rt.arms['left'].segment[0] == rt.arms['right'].segment[0]


def test_future_coverage_waits_then_expiry_holds_both():
    rt, node = make_runtime()
    engage(rt, node)
    t0 = node.time
    deliver(node, chunk(t0, offsets=(0.1, 0.2, 0.3)))
    rt._tick()
    assert sent(node, 'left')[0] == pytest.approx(0.5, abs=2e-6)
    assert sent(node, 'right')[0] == pytest.approx(0.8, abs=2e-6)
    node.time = t0 + 0.15
    refresh(node)
    rt._tick()
    assert sent(node, 'left')[0] > 0.5 and sent(node, 'right')[0] > 0.8
    frozen = {name: sent(node, name).copy() for name in rt.arms}
    node.time = t0 + 0.31
    refresh(node)
    rt._tick()
    for name in rt.arms:
        assert sent(node, name) == pytest.approx(frozen[name], abs=2e-6)


@pytest.mark.parametrize(
    'merge',
    [
        {'type': 'latest'},
        {'type': 'ensemble', 'coeff': 0.1},
        {'type': 'ramp', 'ramp': 3},
    ],
)
def test_reuses_runtime_merge_for_both_arms(merge):
    rt, node = make_runtime(merge=merge)
    engage(rt, node)
    deliver(node, chunk(node.time))
    deliver(node, chunk(node.time, shift=0.03))
    for arm in rt.arms.values():
        actual, expected = arm._combine(node.time), Runtime._combine(arm, node.time)
        assert actual[0] == pytest.approx(expected[0])
        assert actual[1].as_quat() == pytest.approx(expected[1].as_quat())
        assert actual[2] is None


def test_entry_ramp_and_quaternion_interpolation_are_shared():
    cfg = {**deepcopy(CFG), 'entry_ramp': 1.0}
    rt, node = make_runtime(cfg)
    engage(rt, node)
    t0 = node.time
    msg = chunk(t0, offsets=(0.0, 1.0, 2.0), shift=0.1)
    q = Rotation.from_euler('z', 0.4).as_quat()
    for point in msg.multi_dof_joint_trajectory.points:
        for tf in point.transforms:
            tf.rotation.x, tf.rotation.y, tf.rotation.z, tf.rotation.w = map(float, -q)
    deliver(node, msg)
    rt._tick()
    for _ in range(12):
        node.time += 0.04
        refresh(node)
        rt._tick()
    left, right = rt.arms.values()
    assert left.entry[0] == right.entry[0]
    assert 0.5 < sent(node, 'left')[0] < 0.65
    assert 0.8 < sent(node, 'right')[0] < 1.1
    for arm in rt.arms.values():
        angle = Rotation.from_quat(arm.published[1]).magnitude()
        assert 0 < angle < 0.4


@pytest.mark.parametrize('invalid_arm', ['left', 'right'])
def test_invalid_tcp_cannot_refresh_gate_or_run_one_arm(invalid_arm):
    rt, node = make_runtime()
    engage(rt, node)
    deliver(node, chunk(node.time))
    bad = tcp(0.5)
    bad.pose.orientation.w = 0.0
    node.send(CFG['arms'][invalid_arm]['tcp'], bad)
    rt._tick()
    assert_cleared(rt)


def test_chunks_before_engagement_or_after_service_release_are_ignored():
    rt, node = make_runtime()
    deliver(node, chunk(node.time))
    assert all(not arm.chunks for arm in rt.arms.values())
    engage(rt, node)
    request(node, False)
    deliver(node, chunk(node.time))
    assert_cleared(rt)


def test_one_clock_sample_for_both_arm_ticks_and_output_stamps():
    rt, node = make_runtime()
    engage(rt, node)
    deliver(node, chunk(node.time))
    original = node.get_clock
    reads = []

    def advancing_clock():
        def now():
            reads.append(node.time)
            result = original().now()
            node.time += 0.001
            return result

        return SimpleNamespace(now=now)

    node.get_clock = advancing_clock
    rt._tick()
    assert len(reads) == 1
    for name in rt.arms:
        msg = node.publishers[CFG['arms'][name]['target']][-1]
        assert msg.header.stamp == stamp(reads[0])
    assert rt._clock_sample is None


def test_horizon_clips_the_complete_pair_and_direct_publish_suppresses_trigger():
    rt, node = make_runtime(merge={'type': 'latest', 'horizon': 2})
    engage(rt, node)
    deliver(node, chunk(node.time))
    for arm in rt.arms.values():
        assert len(arm.chunks[0].t) == 2
        assert arm.chunks[0].end == pytest.approx(node.time + 0.2)
        arm._publish(arm.tcp[0], arm.tcp[1], 1.0, engaged=True)
    assert set(node.publisher_types.values()) == {Bool, PoseStamped}
