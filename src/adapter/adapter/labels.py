"""Action labels from a recorded teleop episode: one definition for training data and replay.

An episode is resampled at `fps` over its active span, from the first to the last teleop goal
(the B-button home move after the last goal is left out). Each step takes the latest message at
or before it, by recording time (the robot's header clocks are not aligned with this PC's):
  tcp      measured TCP (/teleop/{arm}/tcp): position, quaternion [x y z w], base frame
  target   what the teleop servo chased: its goal (/teleop/{arm}/goal) while clutched, the TCP
           while the clutch was released and the arm held still
  trigger  gripper as the VR trigger value (0 open .. 1 closed), from the last gripper command
  intervention  1 while a person drives the arm in a policy rollout (the teleop's VR takeover,
           /teleop/{arm}/takeover, or a cuff squeeze) or nudges it (/teleop/{arm}/nudging), else 0
  offset   the nudge added to the policy's goal (/teleop/{arm}/correction_offset: translation m and
           rotation vector rad, base frame; run_teleop.sh --takeover nudge), zeros before the first;
           None without the topic
  plan     (load_plans) the plan the policy made from the step's observation, as it ran: the
           policy server's chunk (/policy/{arm}/chunk), stamped with its observation's time

Kinesthetic episodes (load_measured) have no goal: the target is the measured TCP.

The action chunk observed at step t is target[t + k] relative to tcp[t], k = 0..chunk_size-1:
[dx dy dz] in base axes, [rx ry rz] the base-frame rotation vector with R_target = exp(r) R_tcp,
and the trigger. Steps past the end repeat the last one (the arm holds).
"""

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import rosbag2_py
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message
from scipy.spatial.transform import Rotation, Slerp

GOAL_HOLD = 0.05  # s: a goal older than this means the clutch was released (the teleop publishes ~60 Hz)


@dataclass
class Episode:
    tcp_p: np.ndarray  # (N, 3)
    tcp_q: np.ndarray  # (N, 4)
    target_p: np.ndarray  # (N, 3)
    target_q: np.ndarray  # (N, 4)
    trigger: np.ndarray  # (N,)
    t: np.ndarray  # (N,) recording time, ns
    fps: float
    intervention: np.ndarray | None = None  # (N,) 1 while a person drives (a policy rollout's corrections)
    offset: np.ndarray | None = None  # (N, 6) the nudge added to the policy's goal (--takeover nudge)
    plan: np.ndarray | None = None  # (N, steps, 8) the policy's plan from the step's observation (load_plans)
    planned: np.ndarray | None = None  # (N,) 1 where a plan was made from the step's observation

    def __len__(self):
        return len(self.t)


def _pose(msg):
    p, o = msg.pose.position, msg.pose.orientation
    return [p.x, p.y, p.z], [o.x, o.y, o.z, o.w]


def _gripper_trigger(msg):
    """Trigger value from a gripper command, inverting the teleop's opening = 100 * (1 - trigger)."""
    return 1.0 - float(json.loads(msg.args)['position']) / 100.0


class NoTeleopGoal(ValueError):
    """The recording has no teleop goal: the arm was guided by hand (load_measured reads it)."""


def load_episode(path, arm='left', fps=25.0) -> Episode:
    topics = {
        'tcp': f'/teleop/{arm}/tcp',
        'goal': f'/teleop/{arm}/goal',
        'gripper': f'/robot/end_effector/{arm}_gripper/command',
        'takeover': f'/teleop/{arm}/takeover',
        'nudging': f'/teleop/{arm}/nudging',
        'offset': f'/teleop/{arm}/correction_offset',
        'cuff': f'/robot/digital_io/{arm}_lower_cuff/state',
    }
    reader = rosbag2_py.SequentialReader()
    reader.open(rosbag2_py.StorageOptions(uri=str(Path(path)), storage_id='mcap'), rosbag2_py.ConverterOptions('', ''))
    types = {t.name: t.type for t in reader.get_all_topics_and_types()}
    if topics['tcp'] not in types:
        raise ValueError(f'{path}: no {topics["tcp"]} (recorded before the teleop published it)')
    if topics['goal'] not in types:
        raise NoTeleopGoal(f'{path}: no {topics["goal"]}: a kinesthetic recording (load_measured)')
    reader.set_filter(rosbag2_py.StorageFilter(topics=[t for t in topics.values() if t in types]))
    data = {n: [] for n in topics}  # name -> [(time, value)]
    while reader.has_next():
        topic, raw, t = reader.read_next()
        name = next(n for n, tp in topics.items() if tp == topic)
        msg = deserialize_message(raw, get_message(types[topic]))
        if name == 'gripper':
            data[name].append((t, _gripper_trigger(msg)))
        elif name in ('takeover', 'nudging', 'cuff'):
            data[name].append((t, bool(msg.state if name == 'cuff' else msg.data)))
        elif name == 'offset':
            v, w = msg.twist.linear, msg.twist.angular
            data[name].append((t, [v.x, v.y, v.z, w.x, w.y, w.z]))
        else:
            data[name].append((t, _pose(msg)))

    if not data['goal']:
        raise NoTeleopGoal(f'{path}: no {topics["goal"]} message: a kinesthetic recording (load_measured)')
    goal_t = np.array([t for t, _ in data['goal']])
    tcp_t = np.array([t for t, _ in data['tcp']])
    t = np.arange(goal_t[0], goal_t[-1] + 1, int(1e9 / fps))

    def latest(times, at):  # index of the latest message at or before each time (-1: none yet)
        return np.searchsorted(times, at, side='right') - 1

    ti = np.maximum(latest(tcp_t, t), 0)
    tcp_p = np.array([data['tcp'][i][1][0] for i in ti])
    tcp_q = np.array([data['tcp'][i][1][1] for i in ti])
    gi = latest(goal_t, t)
    clutched = (gi >= 0) & (t - goal_t[np.maximum(gi, 0)] <= GOAL_HOLD * 1e9)
    target_p, target_q = tcp_p.copy(), tcp_q.copy()
    for k in np.where(clutched)[0]:
        target_p[k], target_q[k] = data['goal'][gi[k]][1]

    trigger = np.zeros(len(t))  # gripper open until the first command
    if data['gripper']:
        gr_t = np.array([x for x, _ in data['gripper']])
        gr = latest(gr_t, t)
        trigger = np.where(gr >= 0, np.array([v for _, v in data['gripper']])[np.maximum(gr, 0)], 0.0)
    intervention = np.zeros(len(t))
    for name in ('takeover', 'nudging', 'cuff'):
        if data[name]:
            times = np.array([x for x, _ in data[name]])
            i = latest(times, t)
            on = np.array([v for _, v in data[name]])[np.maximum(i, 0)]
            intervention = np.maximum(intervention, np.where(i >= 0, on, False).astype(float))
    offset = None
    if topics['offset'] in types:
        offset = np.zeros((len(t), 6))
        if data['offset']:
            i = latest(np.array([x for x, _ in data['offset']]), t)
            values = np.array([v for _, v in data['offset']])
            offset = np.where((i >= 0)[:, None], values[np.maximum(i, 0)], 0.0)
    return Episode(tcp_p, tcp_q, target_p, target_q, trigger, t, fps, intervention, offset)


def load_plans(path, t, arm='left'):
    """(plan (N, steps, 8), planned (N,)) at the steps `t` (ns): each plan the policy server sent
    (/policy/{arm}/chunk: absolute TCP goals [x y z qx qy qz qw] and trigger per step, stamped with the
    time of the observation it was made from) at the step nearest that time, within half a step;
    planned is 1 there. Plans arrive ~0.1 s after their observation, every few steps: the other
    steps have none. A rollout with a nudge adapter sent the base's plan plus the nudge, not the
    base's: refused."""
    chunk, nudge = f'/policy/{arm}/chunk', f'/policy/{arm}/nudge'
    reader = rosbag2_py.SequentialReader()
    reader.open(rosbag2_py.StorageOptions(uri=str(Path(path)), storage_id='mcap'), rosbag2_py.ConverterOptions('', ''))
    types = {x.name: x.type for x in reader.get_all_topics_and_types()}
    if chunk not in types:
        raise ValueError(f'{path}: no {chunk} (recorded without the policy server)')
    reader.set_filter(rosbag2_py.StorageFilter(topics=[x for x in (chunk, nudge) if x in types]))
    plans = []  # (observation time ns, (steps, 8))
    while reader.has_next():
        topic, raw, _ = reader.read_next()
        if topic == nudge:
            raise ValueError(f"{path}: the policy ran with a nudge adapter: its plans are not the base's")
        msg = deserialize_message(raw, get_message(types[chunk]))
        stamp = msg.joint_trajectory.header.stamp
        tf = [p.transforms[0] for p in msg.multi_dof_joint_trajectory.points]
        trigger = [p.positions[0] for p in msg.joint_trajectory.points] or [0.0] * len(tf)
        plans.append((
            stamp.sec * 1_000_000_000 + stamp.nanosec,
            [
                [x.translation.x, x.translation.y, x.translation.z, x.rotation.x, x.rotation.y, x.rotation.z]
                + [x.rotation.w, g]
                for x, g in zip(tf, trigger)
            ],
        ))
    if not plans:
        raise ValueError(f'{path}: no plan on {chunk}')
    steps = min(len(p) for _, p in plans)
    plan, planned = np.zeros((len(t), steps, 8), np.float32), np.zeros(len(t))
    period = t[1] - t[0] if len(t) > 1 else 1
    for stamp, p in plans:  # in arrival order: a later plan for the same step wins
        k = int(np.searchsorted(t, stamp))
        k = k - 1 if k == len(t) or (k > 0 and stamp - t[k - 1] < t[k] - stamp) else k
        if 0 <= k < len(t) and abs(int(t[k]) - stamp) <= period / 2:
            plan[k], planned[k] = np.asarray(p[:steps], np.float32), 1.0
    return plan, planned


def load_measured(path, arms=('left',), fps=25.0, lead=1) -> dict[str, Episode]:
    """Kinesthetic episodes (the arm guided by hand, no teleop goal): per arm, the target at a step is
    the measured TCP `lead` steps later (the last step holds), so even the first action of a chunk is
    where the hand took the arm next, as a teleop goal leads the arm. The trigger is the measured
    gripper opening (0 open .. 1 closed). Steps span the time every arm's TCP is recorded, on one
    clock for all arms. The TCP is interpolated to each step's time."""
    topics = {
        (arm, kind): tp
        for arm in arms
        for kind, tp in (('tcp', f'/teleop/{arm}/tcp'), ('gripper', f'/robot/end_effector/{arm}_gripper/state'))
    }
    reader = rosbag2_py.SequentialReader()
    reader.open(rosbag2_py.StorageOptions(uri=str(Path(path)), storage_id='mcap'), rosbag2_py.ConverterOptions('', ''))
    types = {t.name: t.type for t in reader.get_all_topics_and_types()}
    missing = [tp for (arm, kind), tp in topics.items() if kind == 'tcp' and tp not in types]
    if missing:
        raise ValueError(f'{path}: no {", ".join(missing)} (the teleop publishes the measured TCP)')
    reader.set_filter(rosbag2_py.StorageFilter(topics=[tp for tp in topics.values() if tp in types]))
    data = {key: [] for key in topics}
    by_topic = {tp: key for key, tp in topics.items()}
    while reader.has_next():
        topic, raw, t = reader.read_next()
        key = by_topic[topic]
        msg = deserialize_message(raw, get_message(types[topic]))
        data[key].append((t, _pose(msg) if key[1] == 'tcp' else 1.0 - float(msg.position) / 100.0))

    start = max(data[(arm, 'tcp')][0][0] for arm in arms)
    end = min(data[(arm, 'tcp')][-1][0] for arm in arms)
    t = np.arange(start, end + 1, int(1e9 / fps))

    def latest(rows):
        times = np.array([x for x, _ in rows])
        return np.maximum(np.searchsorted(times, t, side='right') - 1, 0)

    out = {}
    for arm in arms:
        # Interpolated to each step's time, not the latest sample before it: the TCP arrives at ~76 Hz,
        # so a held sample is up to 13 ms old, 2 mm rms (13 mm max) of noise at a hand-guided pace.
        tcp = data[(arm, 'tcp')]
        times = np.array([x for x, _ in tcp], dtype=float)
        at = np.clip(t.astype(float), times[0], times[-1])
        positions = np.array([v[0] for _, v in tcp])
        p = np.stack([np.interp(at, times, positions[:, k]) for k in range(3)], axis=1)
        q = Slerp(times, Rotation.from_quat([v[1] for _, v in tcp]))(at).as_quat()
        grip = data[(arm, 'gripper')]
        trigger = np.array([grip[k][1] for k in latest(grip)]) if grip else np.zeros(len(t))
        ahead = np.minimum(np.arange(len(t)) + lead, len(t) - 1)
        out[arm] = Episode(p, q, p[ahead], q[ahead], trigger, t, fps)
    return out


def chunk_actions(episode: Episode, step: int, chunk_size: int) -> np.ndarray:
    """(chunk_size, 7) actions observed at `step`: target[step + k] relative to tcp[step], trigger."""
    k = np.minimum(np.arange(step, step + chunk_size), len(episode) - 1)
    R_tcp = Rotation.from_quat(episode.tcp_q[step])
    dp = episode.target_p[k] - episode.tcp_p[step]
    dr = (Rotation.from_quat(episode.target_q[k]) * R_tcp.inv()).as_rotvec()
    return np.column_stack([dp, dr, episode.trigger[k]])


def apply_actions(actions, tcp_p, tcp_q):
    """Absolute targets (positions (K, 3), Rotations) from actions relative to the TCP (p, q)."""
    return tcp_p + actions[:, :3], Rotation.from_rotvec(actions[:, 3:6]) * Rotation.from_quat(tcp_q)


def relative_actions(positions, rotations, triggers, tcp_p, tcp_q):
    """Inverse of apply_actions."""
    dr = (rotations * Rotation.from_quat(tcp_q).inv()).as_rotvec()
    return np.column_stack([positions - tcp_p, dr, triggers])


def model_state(tcp_p, tcp_q, gripper, joints):
    """The policy's 17-D state: TCP position, R_tcp's first two columns (sign-free, unlike the
    quaternion), gripper, 7 joints."""
    R = Rotation.from_quat(tcp_q).as_matrix()
    return np.concatenate([tcp_p, R[:, 0], R[:, 1], [gripper], joints]).astype(np.float32)
