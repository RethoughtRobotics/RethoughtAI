"""Action labels from a recorded teleop episode: one definition for training data and replay.

An episode is resampled at `fps` over its active span, from the first to the last teleop goal
(the B-button home move after the last goal is left out). Each step takes the latest message at
or before it, by recording time (the robot's header clocks are not aligned with this PC's):
  tcp      measured TCP (/teleop/{arm}/tcp): position, quaternion [x y z w], base frame
  target   what the teleop servo chased: its goal (/teleop/{arm}/goal) while clutched, the TCP
           while the clutch was released and the arm held still
  trigger  gripper as the VR trigger value (0 open .. 1 closed), from the last gripper command

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
from scipy.spatial.transform import Rotation

GOAL_HOLD = 0.05  # s: a goal older than this means the clutch was released (the teleop publishes ~60 Hz)


@dataclass
class Episode:
    tcp_p: np.ndarray      # (N, 3)
    tcp_q: np.ndarray      # (N, 4)
    target_p: np.ndarray   # (N, 3)
    target_q: np.ndarray   # (N, 4)
    trigger: np.ndarray    # (N,)
    t: np.ndarray          # (N,) recording time, ns
    fps: float

    def __len__(self):
        return len(self.t)


def _pose(msg):
    p, o = msg.pose.position, msg.pose.orientation
    return [p.x, p.y, p.z], [o.x, o.y, o.z, o.w]


def _gripper_trigger(msg):
    """Trigger value from a gripper command, inverting the teleop's opening = 100 * (1 - trigger)."""
    return 1.0 - float(json.loads(msg.args)["position"]) / 100.0


def load_episode(path, arm="left", fps=25.0) -> Episode:
    topics = {"tcp": f"/teleop/{arm}/tcp", "goal": f"/teleop/{arm}/goal",
              "gripper": f"/robot/end_effector/{arm}_gripper/command"}
    reader = rosbag2_py.SequentialReader()
    reader.open(rosbag2_py.StorageOptions(uri=str(Path(path)), storage_id="mcap"), rosbag2_py.ConverterOptions("", ""))
    types = {t.name: t.type for t in reader.get_all_topics_and_types()}
    missing = [n for n in ("tcp", "goal") if topics[n] not in types]
    if missing:
        raise ValueError(f"{path}: no {', '.join(topics[n] for n in missing)} (recorded before the teleop published it)")
    reader.set_filter(rosbag2_py.StorageFilter(topics=[t for t in topics.values() if t in types]))
    data = {n: [] for n in topics}  # name -> [(time, value)]
    while reader.has_next():
        topic, raw, t = reader.read_next()
        name = next(n for n, tp in topics.items() if tp == topic)
        msg = deserialize_message(raw, get_message(types[topic]))
        data[name].append((t, _gripper_trigger(msg) if name == "gripper" else _pose(msg)))

    goal_t = np.array([t for t, _ in data["goal"]])
    tcp_t = np.array([t for t, _ in data["tcp"]])
    t = np.arange(goal_t[0], goal_t[-1] + 1, int(1e9 / fps))

    def latest(times, at):  # index of the latest message at or before each time (-1: none yet)
        return np.searchsorted(times, at, side="right") - 1

    ti = np.maximum(latest(tcp_t, t), 0)
    tcp_p = np.array([data["tcp"][i][1][0] for i in ti])
    tcp_q = np.array([data["tcp"][i][1][1] for i in ti])
    gi = latest(goal_t, t)
    clutched = (gi >= 0) & (t - goal_t[np.maximum(gi, 0)] <= GOAL_HOLD * 1e9)
    target_p, target_q = tcp_p.copy(), tcp_q.copy()
    for k in np.where(clutched)[0]:
        target_p[k], target_q[k] = data["goal"][gi[k]][1]

    trigger = np.zeros(len(t))  # gripper open until the first command
    if data["gripper"]:
        gr_t = np.array([x for x, _ in data["gripper"]])
        gr = latest(gr_t, t)
        trigger = np.where(gr >= 0, np.array([v for _, v in data["gripper"]])[np.maximum(gr, 0)], 0.0)
    return Episode(tcp_p, tcp_q, target_p, target_q, trigger, t, fps)


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
