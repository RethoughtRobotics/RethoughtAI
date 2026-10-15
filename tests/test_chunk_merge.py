import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from adapter.labels import apply_actions, model_state, relative_actions
from inference.runtime import Chunk, Runtime, check_merge

FPS = 25.0
CFG = {"fps": FPS, "chunk_size": 100, "publish_rate": 100, "tcp": "tcp", "target": "target", "trigger": "trigger",
       "engage": "engage", "chunk": "chunk", "deadman": {"topic": "joy", "frame_id": "x", "button": 0}}


class FakeNode:
    def __getattr__(self, name):  # create_subscription, create_publisher, create_service, create_timer
        return lambda *a, **k: None

    def get_logger(self):
        return type("Logger", (), {"info": lambda self, text: None})()


def runtime(merge):
    return Runtime(FakeNode(), CFG, merge=merge)


def line(t0, arrival, x0, slope, k=100):
    """A chunk moving along x at `slope` m/s from x0, observed at t0."""
    t = t0 + np.arange(k) / FPS
    positions = np.column_stack([x0 + slope * (t - t0), np.zeros(k), np.zeros(k)])
    return Chunk(t, positions, Rotation.identity(k), np.zeros(k), arrival)


def test_chunk_interpolates_between_steps():
    c = Chunk(np.array([0.0, 0.04]), np.array([[0, 0, 0], [1, 0, 0]], float),
              Rotation.from_rotvec([[0, 0, 0], [0, 0, 0.1]]), np.array([0.0, 1.0]), 0.0)
    p, R, trigger = c.at(0.01)
    assert np.allclose(p, [0.25, 0, 0]) and trigger == pytest.approx(0.25)
    assert R.as_rotvec() == pytest.approx([0, 0, 0.025], abs=1e-4)  # nlerp = slerp at a step's rotation


def test_interpolation_takes_the_short_way_across_quaternion_sign_flips():
    q = Rotation.from_rotvec([[0, 0, 3.1], [0, 0, -3.1]]).as_quat()  # 0.08 rad apart, across +-pi
    q[1] *= -1 if np.dot(q[0], q[1]) > 0 else 1                     # stored in opposite hemispheres
    c = Chunk(np.array([0.0, 0.04]), np.zeros((2, 3)), Rotation.from_quat(q), None, 0.0)
    assert (c.at(0.02)[1].inv() * Rotation.from_rotvec([0, 0, np.pi])).magnitude() < 1e-3


def test_ramp_reaches_the_new_chunk_without_a_jump_even_when_the_overlap_is_short():
    rt = runtime({"type": "ramp", "ramp": 40})
    old = line(0.0, 0.0, 0.0, 0.1)            # ends at t = 3.96 s
    new = line(3.5, 3.6, 0.05, 0.1)           # offset 5 cm; overlap with old: ~9 steps < ramp
    rt.chunks = [old, new]
    ts = np.arange(3.6, 3.96, 1 / FPS)
    x = np.array([rt._combine(t)[0][0] for t in ts])
    assert np.diff(x).max() < 0.1 / FPS + 0.05 / 5  # the 5 cm offset spread over the overlap's steps
    assert rt._combine(old.end)[0][0] == pytest.approx(new.at(old.end)[0][0])  # weight 1 by the old chunk's end


def test_ensemble_of_identical_chunks_is_the_chunk():
    rt = runtime({"type": "ensemble", "coeff": 0.01})
    rt.chunks = [line(0.0, 0.0, 0.0, 0.1), line(0.04, 0.04, 0.004, 0.1)]
    assert rt._combine(1.0)[0][0] == pytest.approx(0.1)


def test_chunks_are_used_only_from_their_arrival():
    rt = runtime({"type": "latest"})
    rt.chunks = [line(0.0, 0.0, 0.0, 0.1), line(0.5, 0.7, 1.0, 0.1)]
    assert rt._combine(0.6)[0][0] == pytest.approx(0.06)
    assert rt._combine(0.8)[0][0] == pytest.approx(1.03)


def test_merge_guards():
    with pytest.raises(ValueError):
        check_merge({"type": "ramp", "ramp": 100}, 100)
    with pytest.raises(ValueError):
        check_merge({"type": "ensemble", "coeff": 0.01}, 50, every=50)
    assert check_merge({"type": "ramp", "ramp": 40}, 50, every=25) is not None
    assert check_merge({"type": "latest"}, 50, every=10) is None


def test_relative_actions_invert_apply_actions():
    rng = np.random.default_rng(0)
    actions = np.column_stack([rng.normal(0, 0.1, (5, 3)), rng.normal(0, 0.3, (5, 3)), rng.uniform(0, 1, 5)])
    tcp_p, tcp_q = np.array([0.6, 0.1, 0.0]), Rotation.random(random_state=1).as_quat()
    positions, rotations = apply_actions(actions, tcp_p, tcp_q)
    assert relative_actions(positions, rotations, actions[:, 6], tcp_p, tcp_q) == pytest.approx(actions)


def test_model_state_uses_rotation_columns_and_ignores_quaternion_sign():
    q = Rotation.from_euler("xyz", [0.3, -0.2, 1.0]).as_quat()
    s = model_state(np.zeros(3), q, 0.5, np.arange(7.0))
    R = Rotation.from_quat(q).as_matrix()
    assert s.shape == (17,) and np.allclose(s[3:6], R[:, 0]) and np.allclose(s[6:9], R[:, 1])
    assert np.allclose(s, model_state(np.zeros(3), -q, 0.5, np.arange(7.0)))


def test_chunks_after_a_hold_crossfade_in_instead_of_jumping():
    rt = runtime({"type": "ensemble", "coeff": 0.01})
    rt.hold, rt.tcp = (np.zeros(3), Rotation.identity()), (np.zeros(3), np.array([0, 0, 0, 1.0]), 0.0)
    rt.chunks = [line(0.0, 0.0, 0.066, 0.0)]  # a still chunk 66 mm ahead (the servo lead)
    x = []
    for k in range(40):
        rt._policy_step(k / FPS)
        x.append(rt.segment[1][0][0])
    assert x[0] == 0.0  # starts at the hold
    assert np.diff(x).max() <= 0.066 / (FPS * 1.0) + 1e-9  # spread over the 1 s entry ramp
    assert x[-1] == pytest.approx(0.066)


def test_in_process_policy_chunks_run_through_the_same_merge():
    """--replay: a policy called each step, its chunk anchored on the TCP at that step."""
    class Still:  # every chunk: 2 cm ahead in x, gripper closed
        def __call__(self, observation):
            return np.tile([0.02, 0, 0, 0, 0, 0, 1.0], (100, 1))

    rt = Runtime(FakeNode(), {**CFG, "temporal_ensemble_coeff": 0.01, "n_action_steps": 1, "entry_ramp": 0}, Still())
    rt.tcp = (np.array([0.5, 0.0, 0.1]), np.array([0, 0, 0, 1.0]), 0.0)
    rt.hold = (rt.tcp[0], Rotation.identity())
    for k in range(5):
        rt._policy_step(k / FPS)
    p, R, trigger = rt.segment[1]
    assert np.allclose(p, [0.52, 0, 0.1]) and trigger == pytest.approx(1.0) and len(rt.chunks) == 5


def test_home_clears_but_keeps_the_run_on():
    rt = runtime({"type": "ensemble", "coeff": 0.01})
    rt.requested, rt.chunks, rt.segment = True, [line(0.0, 0.0, 0.0, 0.1)], (0.0, None, None)
    rt.trigger_pub = type("Pub", (), {"publish": lambda self, msg: setattr(self, "sent", msg.data)})()
    rt.trigger, rt._now = 1.0, lambda: 0.0
    rt._on_home(None)
    assert rt.requested and rt.chunks == [] and rt.segment is None
    assert rt.trigger == 0.0 and rt.trigger_pub.sent == 0.0  # gripper open


def released_runtime():
    rt = runtime({"type": "latest"})
    clock, sent = [0.0], []
    rt._now, rt._announce = (lambda: clock[0]), (lambda: None)
    rt.target_pub = type("Pub", (), {"publish": lambda self, m: sent.append(np.array([m.pose.position.x, m.pose.position.y, m.pose.position.z]))})()
    rt.trigger_pub = type("Pub", (), {"publish": lambda self, m: None})()
    rt.node = type("Node", (), {"get_clock": lambda self: type("C", (), {"now": lambda self: type("T", (), {"to_msg": lambda self: __import__("builtin_interfaces.msg").msg.Time()})()})(),
                                "get_logger": lambda self: type("L", (), {"info": lambda self, t: None})()})()
    return rt, clock, sent


def test_released_target_never_follows_a_sagging_arm():
    """A target that follows the TCP is a stroke to the teleop: it pushes the arm on (drift)."""
    rt, clock, sent = released_runtime()
    q = np.array([0, 0, 0, 1.0])
    for k in range(100):  # the arm sags 10 cm while released
        clock[0] = k * 0.01
        rt.tcp = (np.array([0.6, 0.2, 0.1 - 0.001 * k]), q, clock[0])
        rt._tick()
    assert np.allclose(sent[0], sent[-1]) and np.allclose(sent[-1], [0.6, 0.2, 0.1])


def test_engaged_goals_map_through_the_teleops_stroke_rule():
    """Back from homing with the target still at the pre-home pose: no jump, and goals land."""
    rt, clock, sent = released_runtime()
    q = np.array([0, 0, 0, 1.0])
    rt.tcp = (np.array([0.8, 0.0, 0.0]), q, 0.0)
    rt._tick()  # frozen at the pre-home pose
    home = np.array([0.6, 0.2, 0.2])
    rt.requested = True
    clock[0] = 2.0
    rt._tick()  # homing: the teleop publishes no TCP
    clock[0] = 5.0
    rt.tcp = (home, q, 5.0)
    rt._tick()
    assert not rt.engaged  # the teleop is just back: wait for it to re-reference
    clock[0] = 5.35
    rt.tcp = (home, q, 5.35)
    rt._tick()
    assert rt.engaged and np.allclose(sent[-1], [0.8, 0.0, 0.0], atol=1e-5)  # first target: unchanged
    rt._publish(home + [0.05, 0, 0], q, None, engaged=True)  # goal 5 cm ahead of the TCP
    teleop_goal = rt.ref[0] + (sent[-1] - rt.ref[2])  # what the teleop servos to
    assert np.allclose(teleop_goal, home + [0.05, 0, 0], atol=1e-5)


def test_horizon_keeps_only_the_first_steps():
    class Still:
        def __call__(self, observation):
            return np.tile([0.02, 0, 0, 0, 0, 0, 1.0], (100, 1))

    rt = Runtime(FakeNode(), {**CFG, "temporal_ensemble_coeff": 0.01, "n_action_steps": 1, "entry_ramp": 0},
                 Still(), merge={"type": "ensemble", "coeff": 0.01, "horizon": 20})
    rt.tcp = (np.array([0.5, 0.0, 0.1]), np.array([0, 0, 0, 1.0]), 0.0)
    rt.hold = (rt.tcp[0], Rotation.identity())
    rt._policy_step(0.0)
    assert len(rt.chunks[0].t) == 20
    with pytest.raises(ValueError):
        check_merge({"type": "ensemble", "coeff": 0.01, "horizon": 10}, 100, every=10)



class Latency:
    def __init__(self, seconds):
        self.seconds = seconds

    def __len__(self):
        return 1

    def percentile(self, q):
        return self.seconds


def rtc_node(latency, last):
    """A PolicyNode with only what _rtc_args reads: RTC execution_horizon 25 at 25 fps."""
    from types import SimpleNamespace

    from inference.policy_node import PolicyNode

    node = PolicyNode.__new__(PolicyNode)
    node.fps, node.dt, node.warned, node.last, node.latency = FPS, 1 / FPS, False, last, Latency(latency)
    node.policy = SimpleNamespace(policy=SimpleNamespace(config=SimpleNamespace(rtc_config=SimpleNamespace(execution_horizon=25))))
    node.warnings = []
    node.node = SimpleNamespace(get_logger=lambda: SimpleNamespace(warning=node.warnings.append))
    return node


TCP = (np.zeros(3), np.array([0.0, 0.0, 0.0, 1.0]))


def test_rtc_continues_the_last_chunk():
    delay, prev = rtc_node(0.5, line(0.0, 0.0, 0.0, 0.1, k=50))._rtc_args(0.2, TCP)
    assert delay == 13 and prev is not None and len(prev) == 45


def test_rtc_starts_fresh_when_the_delay_reaches_the_horizon():
    """Bag real_wrist_bugs: 1.6 s inferences (delay 36 of 25) chained chunks 591 mm ahead of the arm."""
    node = rtc_node(1.4, line(0.0, 0.0, 0.0, 0.1, k=50))
    delay, prev = node._rtc_args(0.2, TCP)
    assert delay == 35 and prev is None
    node._rtc_args(0.3, TCP)
    assert len(node.warnings) == 1  # once, until chunks continue again


def test_rtc_starts_fresh_when_the_arm_is_not_following():
    node = rtc_node(0.5, line(0.0, 0.0, 0.3, 0.1, k=50))  # 300 mm ahead of the TCP
    assert node._rtc_args(0.2, TCP)[1] is None
    node.last = line(0.0, 0.0, 0.0, 0.1, k=50)
    assert node._rtc_args(0.2, TCP)[1] is not None and not node.warned


def test_reanchored_chunk_keeps_its_plan_relative_to_the_tcp():
    c = line(0.0, 0.0, 0.5, 0.1)  # planned from a TCP at x = 0.5
    R_turn = Rotation.from_euler("z", 30, degrees=True)
    moved = c.reanchored(np.array([0.5, 0, 0]), Rotation.identity(), np.array([0.6, 0.2, 0.1]), R_turn, arrival=9.0)
    assert np.allclose(moved.positions - [0.6, 0.2, 0.1], c.positions - [0.5, 0, 0]) and moved.arrival == 9.0
    assert np.allclose((moved.rotations * R_turn.inv()).as_rotvec(), c.rotations.as_rotvec())


def test_cuff_correction_follows_the_arm_then_resumes_at_once_from_the_corrected_pose():
    rt, clock, sent = released_runtime()
    q = np.array([0, 0, 0, 1.0])
    a, b = np.array([0.5, 0.0, 0.1]), np.array([0.65, 0.1, 0.05])  # where the policy had the arm; where the hand left it
    rt.requested, rt.engaged, rt.frozen = True, True, (a, q)
    rt.ref = (a, Rotation.identity(), a, Rotation.identity())  # goal -> target identity
    rt.hold, rt.accept_after = (a, Rotation.identity()), 0.0
    for k in range(10):
        clock[0] = 0.01 * k
        rt.tcp = (a, q, clock[0]); rt.history.append((clock[0], a, Rotation.identity()))
    rt.chunks = [line(0.05, 0.05, 0.5, 0.1), line(0.09, 0.09, 0.5, 0.1)]  # planned from a, moving along x
    rt.squeezed = True
    for k in range(50):  # the hand moves the arm from a to b
        clock[0] = 0.1 + 0.01 * k
        p = a + (b - a) * k / 49
        rt.tcp = (p, q, clock[0]); rt.history.append((clock[0], p, Rotation.identity()))
        rt._tick()
        assert np.allclose(sent[-1], p, atol=1e-5)  # the target (so the teleop's goal) stays on the arm
    assert len(rt.chunks) == 1 and rt.engaged
    rt.squeezed = False
    clock[0] += 0.01
    rt.tcp = (b, q, clock[0])
    rt._tick()
    assert np.allclose(sent[-1], b, atol=1e-3)  # released where the hand left it: no jump, no spring-back
    assert rt.chunks[0].arrival == pytest.approx(clock[0]) and rt.entry[2] == pytest.approx(0.3)
    planned = line(0.09, 0.09, 0.5, 0.1).at(clock[0])[0] - a  # the plan's offset from where it was planned
    assert np.allclose(rt.chunks[0].at(clock[0])[0] - b, planned, atol=1e-6)  # the same offset, now from b
    for k in range(40):  # resumes at once toward the plan, never back toward a
        clock[0] += 0.01
        rt.tcp = (b, q, clock[0])
        rt._tick()
        assert np.dot(sent[-1] - b, b - a) > -1e-4


def test_lead_sends_the_pose_planned_ahead_and_the_trigger_on_time():
    k = 30
    t = np.arange(k) / FPS
    c = Chunk(t, np.column_stack([0.1 * t, np.zeros(k), np.zeros(k)]), Rotation.identity(k),
              (t >= 0.4).astype(float), 0.0)  # the gripper closes at 0.4 s
    for kind in ({"type": "latest"}, {"type": "ensemble", "coeff": 0.01}, {"type": "ramp", "ramp": 5}):
        rt = runtime({**kind, "lead": 0.4})
        rt.chunks = [c]
        p, _, trigger = rt._combine(0.2)
        assert p[0] == pytest.approx(0.06) and trigger == 0.0  # pose at 0.6 s, gripper still open
        assert rt._combine(0.4)[2] == 1.0
        assert rt._combine(1.0)[0][0] == pytest.approx(0.1 * c.end)  # past the chunk's end: its last pose


def test_lead_that_outruns_each_chunk_is_reported():
    assert check_merge({"type": "latest", "lead": 0.4}, 32, 8) is None
    assert "lead" in check_merge({"type": "latest", "lead": 0.4}, 32, 24)
    with pytest.raises(ValueError):
        check_merge({"type": "latest", "lead": -0.1}, 32, 8)


def test_gripper_latch_closes_fully_and_ignores_flicker():
    rt = runtime({"type": "latest", "gripper_latch": [0.3, 0.6]})
    out = [rt._latched(v) for v in (0.0, 0.5, 0.65, 0.45, 0.55, 0.35, 0.2, 0.5)]
    assert out == [0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0]
    assert runtime({"type": "latest"})._latched(0.45) == 0.45  # no latch: the trigger as planned
    with pytest.raises(ValueError):
        check_merge({"type": "latest", "gripper_latch": [0.6, 0.3]}, 32, 8)


def test_gripper_lead_sends_the_planned_close_early_and_leaves_the_pose():
    k = 30
    t = np.arange(k) / FPS
    c = Chunk(t, np.column_stack([0.1 * t, np.zeros(k), np.zeros(k)]), Rotation.identity(k),
              (t >= 0.6).astype(float), 0.0)  # the plan closes the gripper at 0.6 s
    rt = runtime({"type": "ramp", "ramp": 5, "gripper_lead": 0.4})
    rt.chunks = [c]
    p, _, trigger = rt._combine(0.2)
    assert p[0] == pytest.approx(0.02) and trigger == 1.0  # pose on time, close sent 0.4 s early
    assert rt._combine(0.1)[2] == 0.0
