"""Teleop session -> LeRobot dataset, on the action definition of labels.py.

Frames are labels.load_episode's steps (`fps`, first to last teleop goal); at each, the cameras,
joints and gripper take their latest message at or before it, by recording time. Per frame:
  observation.images.<key>  each configured camera, RGB
  observation.state         TCP [x y z qx qy qz qw] (base frame), gripper (0 open .. 1 closed),
                            the arm's 7 joint angles
  action                    the goal the teleop servo chased [x y z qx qy qz qw] (the TCP while
                            the clutch was released), and the gripper trigger (0 open .. 1 closed);
                            kinesthetic (labels="measured"): the measured TCP and gripper opening
  policy.plan, policy.planned  (plans) the policy's plan from this frame's observation as it ran,
                            flattened (steps x [x y z qx qy qz qw trigger], absolute), and 1 where
                            one was made (every few frames; labels.load_plans), else zeros and 0
Several arms (kinesthetic, bimanual): both per arm, concatenated, names prefixed with the arm.
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

from adapter.labels import load_episode, load_measured, load_plans

JOINTS = ('s0', 's1', 'e0', 'e1', 'w0', 'w1', 'w2')
MAX_IMAGE_AGE = 0.15  # s: a frame whose image is older than this is reported (a recording stall)


def image_rgb(msg):
    if hasattr(msg, 'format'):  # sensor_msgs/CompressedImage
        import cv2

        image = cv2.imdecode(np.frombuffer(bytes(msg.data), np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError('Cannot decode compressed camera image')
        return np.ascontiguousarray(image[..., ::-1])
    channels = {'rgb8': 3, 'bgr8': 3, 'rgba8': 4, 'bgra8': 4}[msg.encoding]
    a = np.frombuffer(bytes(msg.data), np.uint8).reshape(msg.height, msg.width, channels)[..., :3]
    return np.ascontiguousarray(a[..., ::-1] if msg.encoding.startswith('bgr') else a)


def _continuous(q):
    """Flip quaternion signs so consecutive rows stay in the same hemisphere."""
    q = q.copy()
    for i in range(1, len(q)):
        if np.dot(q[i], q[i - 1]) < 0:
            q[i] = -q[i]
    return q


def _drop_leading(ep, n):
    """The episode without its first n steps."""
    fields = ('tcp_p', 'tcp_q', 'target_p', 'target_q', 'trigger', 't', 'intervention', 'plan', 'planned')
    return replace(ep, **{f: getattr(ep, f)[n:] for f in fields if getattr(ep, f) is not None})


def idle_start(eps, distance, angle):
    """Steps before the last one at which every arm's TCP is still within `distance` (m) and `angle`
    (rad) of its first pose: the wait before a demonstration moves. Nothing observable says how long
    it lasts, so a policy at the start pose learns to wait and, replanning from the same still view,
    waits again (a 1.3 s idle stalled every sim cup base at home)."""
    from scipy.spatial.transform import Rotation

    onsets = []
    for e in eps.values():
        moved = np.linalg.norm(e.tcp_p - e.tcp_p[0], axis=1) > distance
        moved |= (Rotation.from_quat(e.tcp_q) * Rotation.from_quat(e.tcp_q[0]).inv()).magnitude() > angle
        if moved.any():
            onsets.append(int(np.argmax(moved)))
    return max(min(onsets) - 1, 0) if onsets else 0  # an episode that never moves is kept whole


class StillRuns:
    """Keeps only the first frame of each still stretch: every position (TCPs, goals) within `distance`
    (m), every orientation within `angle` (rad) and every gripper value (trigger, measured opening)
    within GRIP of where the stretch began. A scripted demonstration holds still for fixed times (the
    sim expert 1 s after closing the gripper, 0.8 s after opening it, 1.3 s at the start) that nothing
    observable measures; a policy replanning there sees the same view and plans the same wait."""

    GRIP = 0.01  # gripper 0 open .. 1 closed: about 0.4 mm of the stroke

    def __init__(self, distance, angle):
        self.distance, self.angle, self.anchor = float(distance), float(angle), None

    def keep(self, positions, quaternions, grips):
        """Whether the frame with these (lists of) values starts a stretch or moves out of one."""
        from scipy.spatial.transform import Rotation

        frame = (np.asarray(positions), np.asarray(quaternions), np.asarray(grips))
        if self.anchor is not None:
            p, q, g = self.anchor
            turned = (Rotation.from_quat(frame[1]) * Rotation.from_quat(q).inv()).magnitude()
            if (
                np.linalg.norm(frame[0] - p, axis=-1).max() <= self.distance
                and turned.max() <= self.angle
                and np.abs(frame[2] - g).max() <= self.GRIP
            ):
                return False
        self.anchor = frame
        return True


def features(cameras, arms, image_shapes, gripper=True, intervention=False, plan_steps=0):
    """One arm: state tcp.*, gripper, joints; action goal.*, gripper. Several: each name prefixed
    with its arm (left.tcp.x ...), arms in order. Without `gripper`, no gripper in either. With
    `intervention`, a per-frame `intervention` (1 while a person drives the arm, labels.Episode).
    With `plan_steps`, the policy's plan of that many steps as it ran (policy.plan, policy.planned)."""
    arms = [arms] if isinstance(arms, str) else list(arms)
    pose = ['x', 'y', 'z', 'qx', 'qy', 'qz', 'qw']
    prefix = (lambda arm: f'{arm}.') if len(arms) > 1 else (lambda arm: '')
    f = {
        f'observation.images.{key}': {
            'dtype': 'video',
            'shape': image_shapes[key],
            'names': ['height', 'width', 'channels'],
        }
        for key in cameras.values()
    }
    grip = ['gripper'] if gripper else []
    state = [f'{prefix(a)}{n}' for a in arms for n in [f'tcp.{x}' for x in pose] + grip + [f'{a}_{j}' for j in JOINTS]]
    action = [f'{prefix(a)}{n}' for a in arms for n in [f'goal.{x}' for x in pose] + grip]
    f['observation.state'] = {'dtype': 'float32', 'shape': (len(state),), 'names': state}
    f['action'] = {'dtype': 'float32', 'shape': (len(action),), 'names': action}
    if intervention:
        f['intervention'] = {'dtype': 'float32', 'shape': (1,), 'names': ['intervention']}
    if plan_steps:
        names = [f'{k}.{x}' for k in range(plan_steps) for x in ['x', 'y', 'z', 'qx', 'qy', 'qz', 'qw', 'trigger']]
        f['policy.plan'] = {'dtype': 'float32', 'shape': (len(names),), 'names': names}
        f['policy.planned'] = {'dtype': 'float32', 'shape': (1,), 'names': ['planned']}
    return f


def read_frames(path, arms, cameras, t, gripper=True):
    """Latest camera images, each arm's joints and gripper position at each time in t (ns): per frame
    (images, joints, gripper, image age), joints and gripper concatenated over `arms`; None for the
    leading times before every stream has started (recording began while the clutch was held)."""
    return list(iter_frames(path, arms, cameras, t, gripper))


def iter_frames(path, arms, cameras, t, gripper=True):
    """read_frames, one frame at a time: an episode's decoded images need not all fit in memory."""
    arms = [arms] if isinstance(arms, str) else list(arms)
    joint_topic = '/robot/joint_states'
    grippers = {f'/robot/end_effector/{arm}_gripper/state': arm for arm in arms} if gripper else {}
    reader = rosbag2_py.SequentialReader()
    reader.open(rosbag2_py.StorageOptions(uri=str(path), storage_id='mcap'), rosbag2_py.ConverterOptions('', ''))
    types = {x.name: x.type for x in reader.get_all_topics_and_types()}
    # Configured camera names remain raw for live inference/viz. Recorded JPEG copies use
    # <topic>/compressed; old raw bags and explicitly compressed configs remain supported.
    cameras = {tp if tp in types else tp + '/compressed': key for tp, key in cameras.items()}
    missing = set(cameras) - types.keys()
    if missing:
        raise ValueError(f'{path}: missing camera topics: {sorted(missing)}')
    reader.set_filter(rosbag2_py.StorageFilter(topics=[*cameras, joint_topic, *grippers]))
    names = [f'{arm}_{j}' for arm in arms for j in JOINTS]
    latest = {}  # topic -> (time, value)
    k, started = 0, False

    def snapshot():
        nonlocal started
        missing = [tp for tp in (*cameras, joint_topic, *grippers) if tp not in latest]
        if missing:
            if started:
                raise ValueError(f'{path}: {missing} missing at frame {k}')
            return None
        started = True
        age = max((t[k] - latest[tp][0]) / 1e9 for tp in cameras)
        grip = [latest[tp][1] for tp in grippers]
        return (
            {cameras[tp]: latest[tp][1] for tp in cameras},
            latest[joint_topic][1],
            grip[0] if len(grip) == 1 else np.array(grip),
            age,
        )  # gripper [] without grippers

    while reader.has_next() and k < len(t):
        topic, raw, stamp = reader.read_next()
        while k < len(t) and stamp > t[k]:  # every frame before this message sees the state so far
            yield snapshot()
            k += 1
        msg = deserialize_message(raw, get_message(types[topic]))
        if topic in cameras:
            latest[topic] = (stamp, image_rgb(msg))
        elif topic == joint_topic:
            if all(n in msg.name for n in names):  # arm messages, not the gripper-only ones
                latest[topic] = (stamp, np.array([msg.position[msg.name.index(n)] for n in names]))
        else:
            latest[topic] = (stamp, 1.0 - float(msg.position) / 100.0)  # 0 open .. 1 closed, as the trigger
    while k < len(t):
        yield snapshot()
        k += 1


def export(
    session,
    repo_id,
    root,
    task,
    cameras,
    arm='left',
    fps=25,
    robot_type='baxter',
    log=print,
    arms=None,
    labels='goal',
    lead=1,
    gripper=True,
    workers=1,
    trim_idle=None,
    trim_still=None,
    intervention=False,
    plans=False,
):
    """Write every episode_* of `session` (in order) as one LeRobot dataset at `root`.

    `labels`: "goal", what the teleop servo chased (labels.load_episode, one arm); "measured", the
    measured TCP of each of `arms` `lead` steps later (labels.load_measured), for kinesthetic
    demonstrations. `gripper`: False leaves the gripper out of state and action (tools held fixed).
    `trim_idle` (distance m, angle rad): each episode starts at its last step before any arm's TCP
    moves that far from its first pose (idle_start); None keeps the wait. `trim_still` (distance m,
    angle rad): every still stretch, the start's too, is cut to its first frame (StillRuns).
    Videos are encoded as frames arrive (streaming encoding), not from images written to disk after
    each episode. `workers` > 1 exports runs of consecutive episodes in parallel processes, each to
    its own dataset, then joins them in order (LeRobot's aggregate_datasets: videos and data files
    concatenated, not re-encoded). `intervention` adds each frame's intervention flag (goal labels:
    the policy rollouts of adapt corrections --whole, labels.Episode.intervention); `plans` the
    policy's plans as they ran (labels.load_plans), for ttt correction-rows."""
    import shutil
    from concurrent.futures import ProcessPoolExecutor
    from multiprocessing import get_context

    arms = list(arms or [arm])
    if labels == 'goal' and len(arms) > 1:
        raise ValueError('goal labels are per teleop arm: export one arm, or use measured labels')
    episodes = sorted(Path(session).glob('episode_*'), key=lambda d: int(d.name.split('_')[1]))
    workers = max(1, min(workers, len(episodes)))
    args = dict(
        repo_id=repo_id,
        task=task,
        cameras=cameras,
        fps=fps,
        robot_type=robot_type,
        arms=arms,
        labels=labels,
        lead=lead,
        gripper=gripper,
        trim_idle=trim_idle,
        trim_still=trim_still,
        intervention=intervention,
        plans=plans,
    )
    if workers == 1:
        summary = _export_episodes(episodes, root=root, log=log, **args)
    else:
        from lerobot.datasets.aggregate import aggregate_datasets

        shards = np.array_split(np.arange(len(episodes)), workers)
        tmp = Path(str(root) + '.shards')
        if tmp.exists():
            shutil.rmtree(tmp)
        roots = [tmp / f'{i:02d}' for i in range(workers)]
        log(f'{len(episodes)} episodes in {workers} parallel runs')
        with ProcessPoolExecutor(workers, mp_context=get_context('spawn')) as pool:
            jobs = [
                pool.submit(
                    _export_episodes,
                    [episodes[j] for j in shard],
                    root=r,
                    log=print,
                    **{**args, 'repo_id': f'{repo_id}_{i:02d}'},
                )
                for i, (shard, r) in enumerate(zip(shards, roots))
            ]
            parts = [job.result() for job in jobs]
        summary = []
        for part in parts:
            for item in part:
                summary.append({**item, 'episode_index': len(summary)})
                log(f'{item["source"]} -> episode {len(summary) - 1}: {item["frames"]} frames')
        aggregate_datasets([f'{repo_id}_{i:02d}' for i in range(workers)], repo_id, roots=roots, aggr_root=Path(root))
        shutil.rmtree(tmp)
    (Path(root) / 'meta' / 'rethoughtai_sources.json').write_text(
        json.dumps(
            {
                'session': str(Path(session).resolve().name),
                'arms': arms,
                'labels': labels,
                'fps': fps,
                **({'lead_steps': lead} if labels == 'measured' else {}),
                'gripper': gripper,
                **({'trim_idle': list(trim_idle)} if trim_idle else {}),
                **({'trim_still': list(trim_still)} if trim_still else {}),
                'episodes': summary,
            },
            indent=1,
        )
    )


def _export_episodes(
    episodes,
    repo_id,
    root,
    task,
    cameras,
    fps,
    robot_type,
    arms,
    labels,
    lead,
    gripper,
    log,
    trim_idle=None,
    trim_still=None,
    intervention=False,
    plans=False,
):
    """`episodes` as one LeRobot dataset at `root`; per episode, a summary of what was written."""
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    arm = arms[0]

    def load(path):  # arm -> Episode
        if labels == 'measured':
            return load_measured(path, arms, fps, lead)
        ep = load_episode(path, arm, fps)
        if plans:
            plan, planned = load_plans(path, ep.t, arm)
            ep = replace(ep, plan=plan, planned=planned)
        return {arm: ep}

    first = load(episodes[0])[arms[0]]
    sample = read_frames(episodes[0], arms, cameras, first.t[-1:], gripper)[0][0]  # last frame: all streams are up
    shapes = {key: img.shape for key, img in sample.items()}
    dataset = LeRobotDataset.create(
        repo_id,
        fps,
        features(cameras, arms, shapes, gripper, intervention, first.plan.shape[1] if plans else 0),
        root=root,
        robot_type=robot_type,
        streaming_encoding=True,
        encoder_threads=2,
    )
    summary = []
    for path in episodes:
        eps = load(path)
        idle = idle_start(eps, *trim_idle) if trim_idle else 0
        eps = {a: _drop_leading(e, idle) for a, e in eps.items()}
        skip, stale, poses, n, k = 0, 0, None, 0, -1
        still = StillRuns(*trim_still) if trim_still else None
        for frame in iter_frames(path, arms, cameras, eps[arms[0]].t, gripper):
            if frame is None:  # leading frames before every stream started
                skip += 1
                continue
            if poses is None:
                eps = {a: _drop_leading(e, skip) for a, e in eps.items()}
                poses = {}
                for a, ep in eps.items():
                    tcp_q = _continuous(ep.tcp_q)
                    goal_q = _continuous(ep.target_q)
                    goal_q[np.sum(goal_q * tcp_q, axis=1) < 0] *= -1  # goal in the TCP's hemisphere
                    poses[a] = (tcp_q, goal_q)
            images, joints, opening, age = frame
            k += 1  # the step of eps this frame is
            opening = np.atleast_1d(opening)
            if still is not None and not still.keep(
                [x for a in arms for x in (eps[a].tcp_p[k], eps[a].target_p[k])],
                [x for a in arms for x in (poses[a][0][k], poses[a][1][k])],
                [*opening, *(eps[a].trigger[k] for a in arms)],
            ):
                continue
            stale += age > MAX_IMAGE_AGE
            grip_state = (lambda i: [opening[i]]) if gripper else (lambda i: [])
            grip_action = (lambda a: [eps[a].trigger[k]]) if gripper else (lambda a: [])
            row = {f'observation.images.{key}': img for key, img in images.items()}
            row['observation.state'] = np.concatenate([
                np.concatenate([eps[a].tcp_p[k], poses[a][0][k], grip_state(i), joints[7 * i : 7 * i + 7]])
                for i, a in enumerate(arms)
            ]).astype(np.float32)
            row['action'] = np.concatenate([
                np.concatenate([eps[a].target_p[k], poses[a][1][k], grip_action(a)]) for a in arms
            ]).astype(np.float32)
            if intervention:
                row['intervention'] = np.array([eps[arm].intervention[k]], dtype=np.float32)
            if plans:
                row['policy.plan'] = eps[arm].plan[k, : first.plan.shape[1]].reshape(-1)
                row['policy.planned'] = np.array([eps[arm].planned[k]], dtype=np.float32)
            row['task'] = task
            dataset.add_frame(row)
            n += 1
        dataset.save_episode()
        summary.append({
            'episode_index': len(summary),
            'source': path.name,
            'frames': n,
            'skipped_leading_frames': skip,
            'idle_start_frames': idle,
            **({'still_frames_cut': k + 1 - n} if still is not None else {}),
            'stale_image_frames': int(stale),
        })
        log(
            f'{path.name}: {n} frames ({n / fps:.1f} s)'
            + (f', {stale} with images older than {MAX_IMAGE_AGE} s' if stale else '')
        )
    dataset.finalize()
    return summary
