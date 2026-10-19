"""Offline policy-node tests with real ROS messages; no ROS init or commands."""

import threading
from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pytest
from baxter_core_msgs.msg import DigitalIOState
from geometry_msgs.msg import PoseStamped
from moveit_msgs.msg import RobotTrajectory
from rclpy.qos import DurabilityPolicy
from sensor_msgs.msg import Image, JointState
from std_msgs.msg import Bool, Empty

from adapter.lerobot_export import JOINTS
from inference.bimanual_policy_node import ARMS, STALE, BimanualPolicyNode

CFG = {
    'fps': 25,
    'joint_states': '/joints',
    'engaged': '/engaged',
    'chunk': '/chunk',
    'arms': {arm: {key: f'/{arm}/{key}' for key in ('tcp', 'cuff', 'home', 'gripper')} for arm in ARMS},
}
IMAGES = [{'key': f'observation.images.{name}', 'topic': f'/camera/{name}'} for name in ('left', 'right', 'third_eye')]


class FakeNode:
    def __init__(self):
        self.time = 10.0
        self.subscriptions, self.qos, self.types, self.publishers = {}, {}, {}, {}

    def get_clock(self):
        return SimpleNamespace(now=lambda: SimpleNamespace(nanoseconds=round(self.time * 1e9)))

    def get_logger(self):
        return SimpleNamespace(warning=lambda text: None)

    def create_subscription(self, cls, topic, callback, qos):
        assert topic not in self.subscriptions
        self.subscriptions[topic], self.qos[topic], self.types[topic] = callback, qos, cls
        return callback

    def create_publisher(self, cls, topic, qos):
        assert cls is RobotTrajectory and topic == CFG['chunk']
        self.publishers[topic] = []
        return SimpleNamespace(publish=lambda msg: self.publishers[topic].append(deepcopy(msg)))

    def send(self, topic, msg):
        self.subscriptions[topic](msg)


class FakePolicy:
    bimanual = True
    rtc = None
    n_obs_steps = 1
    features = {'observation.state': {}, **{c['key']: {} for c in IMAGES}}

    def __init__(self):
        self.calls, self.resets = [], 0
        self.hook = None

    def state(self, tcp_p, tcp_q, gripper, joints):
        assert gripper is None
        assert list(tcp_p) == list(tcp_q) == list(joints) == list(ARMS)
        return np.concatenate([np.r_[tcp_p[arm], tcp_q[arm], joints[arm]] for arm in ARMS])

    def reset(self):
        self.resets += 1

    def __call__(self, obs):
        self.calls.append(deepcopy(obs))
        if self.hook:
            self.hook()
        return actions()


def actions(n=4):
    result = np.zeros((n, 14))
    result[:, :3] = [4, 5, 6]  # Deliberately far from TCP: must stay absolute.
    result[:, 7:10] = [7, 8, 9]
    result[:, 6], result[:, 13] = 2.0, -3.0
    return result


def tcp(p=(1, 2, 3), q=(0, 0, 0, 1)):
    msg = PoseStamped()
    msg.pose.position.x, msg.pose.position.y, msg.pose.position.z = map(float, p)
    (msg.pose.orientation.x, msg.pose.orientation.y, msg.pose.orientation.z, msg.pose.orientation.w) = map(float, q)
    return msg


def joint_message():
    names = [f'{arm}_{j}' for arm in ARMS for j in JOINTS]
    values = list(map(float, range(14)))
    # Neither input message order nor cfg insertion order may change state order.
    return JointState(name=names[::-1], position=values[::-1])


def image():
    return Image(height=2, width=3, encoding='rgb8', step=9, data=list(range(18)))


def refresh(node):
    for arm, p in zip(ARMS, ((1, 2, 3), (11, 12, 13))):
        node.send(CFG['arms'][arm]['tcp'], tcp(p))
    node.send(CFG['joint_states'], joint_message())
    for camera in IMAGES:
        node.send(camera['topic'], image())


@pytest.fixture
def make(monkeypatch):
    starts = []
    original_start = threading.Thread.start
    monkeypatch.setattr(threading.Thread, 'start', lambda thread: starts.append(thread))

    def build(cfg=None, policy=None, every=1, ready=True):
        node, policy = FakeNode(), policy or FakePolicy()
        server = BimanualPolicyNode(node, deepcopy(cfg or CFG), IMAGES, policy, every)
        assert starts[-1] is server.worker
        if ready:
            refresh(node)
            node.send(CFG['engaged'], Bool(data=True))
        return server, node, policy

    build.thread_start = original_start
    return build


def test_construction_subscriptions_and_only_one_pose_chunk_publisher(make):
    server, node, _ = make(ready=False)
    assert not server.engaged
    assert set(node.subscriptions) == {
        *(c['topic'] for c in IMAGES),
        CFG['joint_states'],
        CFG['engaged'],
        *(CFG['arms'][arm][key] for arm in ARMS for key in ('tcp', 'cuff', 'home')),
    }
    assert set(node.publishers) == {CFG['chunk']}
    assert node.qos[CFG['engaged']].durability == DurabilityPolicy.TRANSIENT_LOCAL
    assert not any('gripper' in topic for topic in (*node.subscriptions, *node.publishers))
    assert node.types[CFG['joint_states']] is JointState


def test_state_order_and_camera_values_are_copied(make):
    cfg = deepcopy(CFG)
    cfg['arms'] = dict(reversed(list(cfg['arms'].items())))
    server, node, _ = make(cfg=cfg)
    t, obs, generation = server._observation()
    assert t == pytest.approx(10) and generation == server.generation
    expected = np.r_[[1, 2, 3, 0, 0, 0, 1], np.arange(7), [11, 12, 13, 0, 0, 0, 1], np.arange(7, 14)]
    np.testing.assert_array_equal(obs['observation.state'], expected)
    assert set(obs) == {'observation.state', *(c['key'] for c in IMAGES)}
    obs[IMAGES[0]['key']][:] = 99
    assert not np.all(server.latest[IMAGES[0]['key']][0] == 99)


@pytest.mark.parametrize('missing', ['left_tcp', 'right_tcp', 'joints', *(c['key'] for c in IMAGES)])
def test_complete_fresh_inputs_required(make, missing):
    server, node, policy = make()
    value, _ = server.latest.pop(missing)
    assert server._observation() is None
    assert not server._step() and not policy.calls
    server.latest[missing] = (value, node.time - STALE - 0.001)
    assert server._observation() is None
    server.latest[missing] = (value, node.time + 0.01)
    assert server._observation() is None
    refresh(node)
    assert server._observation() is not None


def test_skew_uses_receive_times_and_is_configurable(make):
    cfg = deepcopy(CFG)
    cfg['max_tcp_skew'] = 0.05
    server, node, _ = make(cfg=cfg)
    node.time += 0.06
    msg = tcp()
    msg.header.stamp.sec = 999999  # Header clocks are deliberately ignored.
    node.send(CFG['arms']['left']['tcp'], msg)
    assert server._observation() is None
    node.send(CFG['arms']['right']['tcp'], msg)
    assert server._observation() is not None


def test_quaternion_continuity_on_every_tcp_input_without_qy_rule(make):
    server, node, _ = make(ready=False)
    q0 = np.array([0, -0.8, 0, -0.6])
    node.send(CFG['arms']['left']['tcp'], tcp(q=q0))
    np.testing.assert_array_equal(server.latest['left_tcp'][0][1], q0)
    node.send(CFG['arms']['right']['tcp'], tcp(q=-q0))
    np.testing.assert_array_equal(server.latest['right_tcp'][0][1], -q0)
    # Intermediate TCPs must participate even though no observation/prediction occurs.
    for angle in (4.4, 5.2, 6.0, 6.6):
        q = np.array([0, np.sin(angle), 0, np.cos(angle)])
        previous = server.latest['left_tcp'][0][1].copy()
        node.send(CFG['arms']['left']['tcp'], tcp(q=-q))
        current = server.latest['left_tcp'][0][1]
        assert np.dot(previous, current) >= 0
    assert current[1] > 0  # Legitimately crossed qy=0, rather than canonicalized.
    np.testing.assert_array_equal(server.latest['right_tcp'][0][1], -q0)


@pytest.mark.parametrize('arm', ARMS)
@pytest.mark.parametrize('bad', ['position', 'nonfinite', 'zero', 'nonunit'])
def test_invalid_tcp_removes_previous_valid_input(make, arm, bad):
    server, node, _ = make()
    msg = tcp()
    if bad == 'position':
        msg.pose.position.z = float('inf')
    elif bad == 'nonfinite':
        msg.pose.orientation.x = float('nan')
    else:
        msg.pose.orientation.w = 0.0 if bad == 'zero' else 1.1
    node.send(CFG['arms'][arm]['tcp'], msg)
    assert server._observation() is None


@pytest.mark.parametrize('bad', ['missing', 'short', 'duplicate', 'nonfinite'])
def test_invalid_shared_joints_remove_previous_valid_input(make, bad):
    server, node, _ = make()
    msg = joint_message()
    if bad == 'missing':
        msg.name.pop()
    elif bad == 'short':
        msg.position.pop()
    elif bad == 'duplicate':
        msg.name.append(msg.name[0])
        msg.position.append(0.0)
    else:
        msg.position[0] = float('nan')
    node.send(CFG['joint_states'], msg)
    assert server._observation() is None


def test_absolute_two_transform_chunks_normalized_and_matching_headers(make):
    server, node, policy = make()
    assert server._step()
    assert len(policy.calls) == 1 and policy.resets == 1
    msg = node.publishers[CFG['chunk']][0]
    joint, multi = msg.joint_trajectory, msg.multi_dof_joint_trajectory
    assert joint.header == multi.header
    assert joint.header.frame_id == 'base' and joint.header.stamp.sec == 10
    assert not joint.points and not joint.joint_names
    assert list(multi.joint_names) == ['left_tcp', 'right_tcp']
    for k, point in enumerate(multi.points):
        assert len(point.transforms) == 2
        assert point.time_from_start.sec + point.time_from_start.nanosec * 1e-9 == pytest.approx(k / 25)
        for tf, p, w in zip(point.transforms, ((4, 5, 6), (7, 8, 9)), (1, -1)):
            assert [tf.translation.x, tf.translation.y, tf.translation.z] == list(p)
            assert [tf.rotation.x, tf.rotation.y, tf.rotation.z, tf.rotation.w] == [0, 0, 0, w]


@pytest.mark.parametrize(
    'bad', ['empty', 'single', 'flat', 'width', 'rank', 'ragged', 'string', 'nan', 'inf', 'left_zero', 'right_zero']
)
def test_entire_malformed_chunk_is_rejected(make, bad):
    server, node, _ = make()
    a = actions()
    if bad == 'empty':
        a = np.zeros((0, 14))
    elif bad == 'single':
        a = a[:1]
    elif bad == 'flat':
        a = a[0]
    elif bad == 'width':
        a = a[:, :13]
    elif bad == 'rank':
        a = a[None]
    elif bad == 'ragged':
        a = [[1], [1, 2]]
    elif bad == 'string':
        a = 'not actions'
    elif bad in ('nan', 'inf'):
        a[-1, -1] = float(bad)
    else:
        start = 3 if bad == 'left_zero' else 10
        a[-1, start : start + 4] = 0
    assert not server._publish(node.time, a, server.generation)
    assert not node.publishers[CFG['chunk']]


def test_extreme_nonzero_model_quaternions_normalize_without_mutating_actions(make):
    server, node, _ = make()
    a = actions()
    a[:, 3:7], a[:, 10:14] = 1e300, 1e-300
    original = a.copy()
    assert server._publish(node.time, a, server.generation)
    np.testing.assert_array_equal(a, original)
    for point in node.publishers[CFG['chunk']][0].multi_dof_joint_trajectory.points:
        for tf in point.transforms:
            assert [tf.rotation.x, tf.rotation.y, tf.rotation.z, tf.rotation.w] == [0.5] * 4


def test_no_inference_disengaged_and_reset_after_both_transitions(make):
    server, node, policy = make(ready=False)
    refresh(node)
    assert not server._step() and not policy.calls
    node.send(CFG['engaged'], Bool(data=True))
    assert server._step()
    generation, resets = server.generation, policy.resets
    node.send(CFG['engaged'], Bool(data=True))
    assert server.generation == generation
    node.send(CFG['engaged'], Bool(data=False))
    assert not server._step() and len(policy.calls) == 1
    assert policy.resets == resets + 1
    node.send(CFG['engaged'], Bool(data=True))
    assert server._step() and policy.resets == resets + 2


@pytest.mark.parametrize(
    'event', ['release', 'release_reengage', 'home_left', 'home_right', 'cuff_left', 'cuff_right', 'close']
)
def test_inflight_generation_discard_and_next_prediction_reset(make, event):
    server, node, policy = make()

    def transition():
        if event.startswith('home'):
            node.send(CFG['arms'][event.split('_')[1]]['home'], Empty())
        elif event.startswith('cuff'):
            arm = event.split('_')[1]
            node.send(CFG['arms'][arm]['cuff'], DigitalIOState(state=1))
            node.send(CFG['arms'][arm]['cuff'], DigitalIOState(state=0))
        elif event == 'close':
            server.close()
        else:
            node.send(CFG['engaged'], Bool(data=False))
        if event == 'release_reengage':
            node.send(CFG['engaged'], Bool(data=True))

    policy.hook = transition  # Executes in the middle of the model call.
    assert not server._step()
    assert not node.publishers[CFG['chunk']]
    policy.hook = None
    assert not server._publish(node.time, actions(), server.generation - 1)
    if event != 'close':
        node.send(CFG['engaged'], Bool(data=True))
        assert server._step() and policy.resets == 2


def test_cuff_blocks_even_if_shared_true_and_release_needs_reengagement(make):
    server, node, policy = make()
    node.send(CFG['arms']['left']['cuff'], DigitalIOState(state=1))
    node.send(CFG['engaged'], Bool(data=True))
    assert not server._step() and not policy.calls
    node.send(CFG['arms']['left']['cuff'], DigitalIOState(state=0))
    assert not server._step()
    node.send(CFG['engaged'], Bool(data=True))
    assert server._step()


def test_publication_rechecks_inputs_and_observation_age(make):
    server, node, policy = make()
    policy.hook = lambda: setattr(node, 'time', node.time + STALE + 0.01)
    assert not server._step() and not node.publishers[CFG['chunk']]
    refresh(node)
    assert not server._publish(node.time - STALE - 0.01, actions(), server.generation)
    assert not server._publish(node.time + 0.01, actions(), server.generation)


def test_release_between_message_build_and_publish_cannot_publish(make, monkeypatch):
    server, node, _ = make()
    generation = server.generation
    import inference.bimanual_policy_node as module

    real_transform = module.Transform

    def transform():
        node.send(CFG['engaged'], Bool(data=False))
        return real_transform()

    monkeypatch.setattr(module, 'Transform', transform)
    assert not server._publish(node.time, actions(), generation)
    assert not node.publishers[CFG['chunk']]


def test_release_callback_does_not_wait_for_blocked_model(make):
    server, node, policy = make()
    entered, finish, released = threading.Event(), threading.Event(), threading.Event()
    results = []

    def blocked_model():
        entered.set()
        assert finish.wait(2.0)

    def release():
        node.send(CFG['engaged'], Bool(data=False))
        released.set()

    policy.hook = blocked_model
    prediction = threading.Thread(target=lambda: results.append(server._step()))
    callback = threading.Thread(target=release)
    # Only these offline test threads run, never the node's infinite worker.
    make.thread_start(prediction)
    try:
        assert entered.wait(2.0)
        make.thread_start(callback)
        assert released.wait(2.0)  # Must complete before the model unblocks.
    finally:
        finish.set()
        prediction.join(2.0)
        if callback.ident is not None:
            callback.join(2.0)
    assert not prediction.is_alive() and not callback.is_alive()
    assert results == [False] and not node.publishers[CFG['chunk']]


def test_actual_publication_holds_same_lock_as_release_callback(make):
    server, node, _ = make()
    publisher = server.chunk_pub.publish
    checks = []

    def publish(msg):
        acquired = server.lock.acquire(blocking=False)
        if acquired:
            server.lock.release()
        checks.append(not acquired)
        publisher(msg)

    server.chunk_pub.publish = publish
    assert server._step() and checks == [True]


@pytest.mark.parametrize('state', [np.zeros(27), np.zeros((1, 28)), np.full(28, np.nan)])
def test_invalid_wrapper_state_never_reaches_model(make, state):
    server, node, policy = make()
    policy.state = lambda *args: state
    assert server._observation() is None
    assert not server._step() and not policy.calls and not node.publishers[CFG['chunk']]


@pytest.mark.parametrize(
    'key,value',
    [('fps', 0), ('fps', float('nan')), ('max_tcp_skew', -1), ('max_tcp_skew', float('inf')), ('engaged', None)],
)
def test_invalid_configuration(make, key, value):
    cfg = deepcopy(CFG)
    cfg[key] = value
    with pytest.raises(ValueError):
        make(cfg=cfg)


@pytest.mark.parametrize('every', [0, -1, 1.5])
def test_invalid_cadence(make, every):
    with pytest.raises(ValueError):
        make(every=every)


def test_reject_single_arm_rtc_and_missing_feature_camera(make):
    for attr, value in (('bimanual', False), ('rtc', object()), ('features', {'observation.images.missing': {}})):
        policy = FakePolicy()
        setattr(policy, attr, value)
        with pytest.raises(ValueError):
            make(policy=policy)


def test_worker_cadence_without_running_a_thread(make, monkeypatch):
    server, _, _ = make(every=3)
    waits = []
    monkeypatch.setattr(server._wake, 'wait', lambda delay: waits.append(delay))
    monkeypatch.setattr('inference.bimanual_policy_node.time.monotonic', lambda: 100.0)
    calls = []

    def step():
        calls.append(True)
        if len(calls) == 2:
            server._stop.set()
        return True

    monkeypatch.setattr(server, '_step', step)
    server._run()
    assert waits == pytest.approx([0, 3 / 25])


def test_device_lost_recognises_dead_gpu_errors_only():
    from inference.bimanual_policy_node import device_lost

    assert device_lost(RuntimeError('CUDA error: unspecified launch failure'))
    assert device_lost(RuntimeError('CUDA error: an illegal memory access was encountered'))
    assert not device_lost(ValueError('expected 2 observations, got 1'))
    assert not device_lost(RuntimeError('mat1 and mat2 shapes cannot be multiplied'))
