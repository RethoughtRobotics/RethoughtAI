"""Teleop session -> LeRobot dataset, on the action definition of labels.py.

Frames are labels.load_episode's steps (`fps`, first to last teleop goal); at each, the cameras,
joints and gripper take their latest message at or before it, by recording time. Per frame:
  observation.images.<key>  each configured camera, RGB
  observation.state         TCP [x y z qx qy qz qw] (base frame), gripper (0 open .. 1 closed),
                            the arm's 7 joint angles
  action                    the goal the teleop servo chased [x y z qx qy qz qw] (the TCP while
                            the clutch was released), and the gripper trigger (0 open .. 1 closed)
Both poses are absolute: a policy's chunk observed at frame t is action[t + k] relative to the TCP
in state[t] (labels.chunk_actions), which needs both frames, so it is formed at training time.
Quaternions are kept sign-continuous within an episode.
"""

import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import rosbag2_py
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message

from adapter.labels import load_episode

JOINTS = ("s0", "s1", "e0", "e1", "w0", "w1", "w2")
MAX_IMAGE_AGE = 0.15  # s: a frame whose image is older than this is reported (a recording stall)


def image_rgb(msg):
    if hasattr(msg, "format"):  # sensor_msgs/CompressedImage
        import cv2
        image = cv2.imdecode(np.frombuffer(bytes(msg.data), np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError("Cannot decode compressed camera image")
        return np.ascontiguousarray(image[..., ::-1])
    channels = {"rgb8": 3, "bgr8": 3, "rgba8": 4, "bgra8": 4}[msg.encoding]
    a = np.frombuffer(bytes(msg.data), np.uint8).reshape(msg.height, msg.width, channels)[..., :3]
    return np.ascontiguousarray(a[..., ::-1] if msg.encoding.startswith("bgr") else a)


def _continuous(q):
    """Flip quaternion signs so consecutive rows stay in the same hemisphere."""
    q = q.copy()
    for i in range(1, len(q)):
        if np.dot(q[i], q[i - 1]) < 0:
            q[i] = -q[i]
    return q


def _drop_leading(ep, n):
    """The episode without its first n steps."""
    return replace(ep, **{f: getattr(ep, f)[n:] for f in ("tcp_p", "tcp_q", "target_p", "target_q", "trigger", "t")})


def features(cameras, arm, image_shapes):
    pose = ["x", "y", "z", "qx", "qy", "qz", "qw"]
    f = {f"observation.images.{key}": {"dtype": "video", "shape": image_shapes[key],
                                         "names": ["height", "width", "channels"]} for key in cameras.values()}
    f["observation.state"] = {"dtype": "float32", "shape": (15,),
                              "names": [f"tcp.{n}" for n in pose] + ["gripper"] + [f"{arm}_{j}" for j in JOINTS]}
    f["action"] = {"dtype": "float32", "shape": (8,), "names": [f"goal.{n}" for n in pose] + ["gripper"]}
    return f


def read_frames(path, arm, cameras, t):
    """Latest camera images, arm joints and gripper position at each time in t (ns); None for the
    leading times before every stream has started (recording began while the clutch was held)."""
    joint_topic, gripper_topic = "/robot/joint_states", f"/robot/end_effector/{arm}_gripper/state"
    reader = rosbag2_py.SequentialReader()
    reader.open(rosbag2_py.StorageOptions(uri=str(path), storage_id="mcap"), rosbag2_py.ConverterOptions("", ""))
    types = {x.name: x.type for x in reader.get_all_topics_and_types()}
    # Configured camera names remain raw for live inference/viz. Recorded JPEG copies use
    # <topic>/compressed; old raw bags and explicitly compressed configs remain supported.
    cameras = {tp if tp in types else tp + "/compressed": key for tp, key in cameras.items()}
    missing = set(cameras) - types.keys()
    if missing:
        raise ValueError(f"{path}: missing camera topics: {sorted(missing)}")
    reader.set_filter(rosbag2_py.StorageFilter(topics=[*cameras, joint_topic, gripper_topic]))
    names = [f"{arm}_{j}" for j in JOINTS]
    latest = {}  # topic -> (time, value)
    out, k = [], 0

    def snapshot():
        missing = [tp for tp in (*cameras, joint_topic, gripper_topic) if tp not in latest]
        if missing:
            if any(f is not None for f in out):
                raise ValueError(f"{path}: {missing} missing at frame {k}")
            out.append(None)
            return
        age = max((t[k] - latest[tp][0]) / 1e9 for tp in cameras)
        out.append(({cameras[tp]: latest[tp][1] for tp in cameras}, latest[joint_topic][1],
                    latest[gripper_topic][1], age))

    while reader.has_next() and k < len(t):
        topic, raw, stamp = reader.read_next()
        while k < len(t) and stamp > t[k]:  # every frame before this message sees the state so far
            snapshot()
            k += 1
        msg = deserialize_message(raw, get_message(types[topic]))
        if topic in cameras:
            latest[topic] = (stamp, image_rgb(msg))
        elif topic == joint_topic:
            if names[0] in msg.name:
                latest[topic] = (stamp, np.array([msg.position[msg.name.index(n)] for n in names]))
        else:
            latest[topic] = (stamp, 1.0 - float(msg.position) / 100.0)  # 0 open .. 1 closed, as the trigger
    while k < len(t):
        snapshot()
        k += 1
    return out


def export(session, repo_id, root, task, cameras, arm="left", fps=25, robot_type="baxter", log=print):
    """Write every episode_* of `session` (in order) as one LeRobot dataset at `root`."""
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    episodes = sorted(Path(session).glob("episode_*"), key=lambda d: int(d.name.split("_")[1]))
    first = load_episode(episodes[0], arm, fps)
    sample = read_frames(episodes[0], arm, cameras, first.t[-1:])[0][0]  # last frame: all streams are up
    shapes = {key: img.shape for key, img in sample.items()}
    dataset = LeRobotDataset.create(repo_id, fps, features(cameras, arm, shapes), root=root, robot_type=robot_type)
    summary = []
    for index, path in enumerate(episodes):
        ep = load_episode(path, arm, fps)
        frames = read_frames(path, arm, cameras, ep.t)
        skip = sum(f is None for f in frames)  # leading frames before every stream started
        frames, ep = frames[skip:], _drop_leading(ep, skip)
        tcp_q = _continuous(ep.tcp_q)
        goal_q = _continuous(ep.target_q)
        goal_q[np.sum(goal_q * tcp_q, axis=1) < 0] *= -1  # goal in the TCP's hemisphere
        stale = int(sum(age > MAX_IMAGE_AGE for *_, age in frames))
        for k, (images, joints, gripper, _) in enumerate(frames):
            frame = {f"observation.images.{key}": img for key, img in images.items()}
            frame["observation.state"] = np.concatenate([ep.tcp_p[k], tcp_q[k], [gripper], joints]).astype(np.float32)
            frame["action"] = np.concatenate([ep.target_p[k], goal_q[k], [ep.trigger[k]]]).astype(np.float32)
            frame["task"] = task
            dataset.add_frame(frame)
        dataset.save_episode()
        summary.append({"episode_index": index, "source": path.name, "frames": len(frames),
                        "skipped_leading_frames": skip, "stale_image_frames": stale})
        log(f"{path.name} -> episode {index}: {len(frames)} frames ({len(frames) / fps:.1f} s)"
            + (f", {stale} with images older than {MAX_IMAGE_AGE} s" if stale else ""))
    dataset.finalize()
    (Path(root) / "meta" / "rethoughtai_sources.json").write_text(json.dumps(
        {"session": str(Path(session).resolve().name), "arm": arm, "fps": fps, "episodes": summary}, indent=1))
    return dataset
