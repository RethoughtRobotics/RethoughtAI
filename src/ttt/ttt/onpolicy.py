"""On-policy corrections (the sim's corrector, training/sim_experiments.md): the base's own attempts,
from the closed-loop evaluator's bags, with the scripted planner's chunk at each frame as the target.

The planner's chunks come with the bag: BaxterSDK's `python -m baxter_mujoco.datagen.expert BAG...`
writes <bag>/expert.npz, the attempt's states at the dataset's rate from the engage to the release and,
at each, the expert's chunk replanned from the arm's and the cup's current poses, sampled as the
datasets record their labels (measured TCP and gripper, no lead). The bag needs the policy's cameras
(the evaluator's --bag-images).

`build` (ttt onpolicy) runs the frozen base over those frames as cache.build runs it over a dataset's,
into a cache of the same rows: `chunk_demo` and `z_demo` the planner's chunk, relabelled as the
checkpoint's training labels were (the pose `lead` steps later, the gripper not led), `chunk_pi` and
`z_pi` the base's own plan from that frame; `episode` the attempt's seed, `frame` the row of
expert.npz. Frames without a plan (a fallen cup) are left out. The corrector's gate
(corrector.corrected: the base's executed steps on average more than `gate` cm from the planner's)
picks the corrected frames, and the update loop takes them as its corrections
(update.Settings.corrections, ttt update --corrections), the anchor from the dataset's cache.

`export` (ttt export) writes the corrected frames as a LeRobot dataset (plan S4): one episode per
corrected frame, `horizon` frames of its observation with the planner's chunk as their actions, so that
the action chunk of each episode's frame 0 is the correction. The later frames repeat the observation
with the rest of the chunk (padded past its end): a retraining that should see only the corrections
samples frame 0.
"""

import json
import re
import time
from pathlib import Path

import numpy as np
import torch

from . import cache as cache_mod
from . import corrector

EXPERT = 'expert.npz'
# Camera topic: key, as configs/baxter_sim.yaml records them; the policy's are used
CAMERAS = {
    '/cameras/left_hand_camera/image': 'left',
    '/cameras/right_hand_camera/image': 'right',
    '/cameras/third_eye/color/image_raw': 'third_eye',
}
SOURCES = 'meta/rethoughtai_sources.json'


def lead_steps(model, fps):
    """Steps the checkpoint's labels lead the TCP: its training dataset's `_lead<N>ms`, else 0."""
    from inference.policies import trained_lead

    name = trained_lead(model)
    return round(int(re.search(r'_lead(\d+)ms', name).group(1)) * fps / 1000) if name else 0


def relabel(chunk, horizon, lead):
    """(..., horizon, 8) labels from the planner's (..., >= horizon + lead, 8) measured chunk: the pose
    `lead` steps later, the gripper on time (tools/relabel_lead.py's rule)."""
    if chunk.shape[-2] < horizon + lead:
        raise ValueError(
            f'planner chunks of {chunk.shape[-2]} steps; {horizon} led by {lead} need {horizon + lead} '
            '(datagen.expert --horizon)'
        )
    return np.concatenate([chunk[..., lead : lead + horizon, :7], chunk[..., :horizon, 7:]], -1)


def seed(bag):
    """The attempt's seed from its bag directory (the evaluator's seed_NNNN), else None."""
    m = re.search(r'(\d+)$', Path(bag).name)
    return int(m.group(1)) if m else None


def cameras_of(lp):
    """{topic: key} of the policy's cameras."""
    out = {tp: key for tp, key in CAMERAS.items() if f'observation.images.{key}' in lp.features}
    missing = {k for k in lp.features if k.startswith('observation.images.')} - {
        f'observation.images.{key}' for key in out.values()
    }
    if missing:
        raise ValueError(f'no camera topic known for {", ".join(sorted(missing))}')
    return out


def load_expert(bag, name=EXPERT):
    """expert.npz of `bag` as a dict, with its meta."""
    path = Path(bag) / name
    if not path.exists():
        raise FileNotFoundError(f'{path}: run python -m baxter_mujoco.datagen.expert on the bag first')
    z = np.load(path)
    out = {k: z[k] for k in z.files if k != 'meta'}
    out['meta'] = json.loads(str(z['meta']))
    return out


def observations(bag, expert, rows, cameras, arm='left'):
    """Per frame of `rows` (indices into expert.npz): (images {key: (H, W, 3) uint8}, joints, TCP
    (position, quaternion), gripper), each the latest message at or before the frame, as the dataset
    export and the policy node take them."""
    from adapter.lerobot_export import iter_frames

    rows = np.asarray(rows)
    frames = iter_frames(bag, arm, cameras, expert['t'][rows])
    for i, frame in zip(rows, frames, strict=True):
        if frame is None:
            raise ValueError(f'{bag}: a camera or the joints have no message by frame {i}')
        images, joints, _, _ = frame
        tcp = expert['tcp'][i]
        yield images, joints, (tcp[:3], tcp[3:7]), float(expert['gripper'][i])


def item(lp, images, joints, tcp, gripper, action, episode, frame):
    """A frame as a LeRobotDataset item: the cameras letterboxed (or stretched) to the policy's shapes
    as the runtime feeds them, (C, H, W) in [0, 1]."""
    from inference.policies import letterbox, stretch

    fit = stretch if getattr(lp, 'resize', None) else letterbox
    out = {
        k: torch.from_numpy(fit(images[k.removeprefix('observation.images.')], *shape)).permute(2, 0, 1) / 255.0
        for k, shape in lp.image_shapes.items()
    }
    out['observation.state'] = torch.from_numpy(lp.state(*tcp, gripper, joints))
    out['action'] = torch.as_tensor(action, dtype=torch.float32)
    out['action_is_pad'] = torch.zeros(len(action), dtype=torch.bool)
    out['episode_index'] = torch.tensor(episode)
    out['frame_index'] = torch.tensor(frame)
    return out


def build(model, bags, out, device='cpu', batch_size=32, lead=None, arm='left', expert=EXPERT):
    """Run the base `model` over the planned frames of every bag and write a cache to `out` (see the
    module docstring); `lead` (steps) defaults to the checkpoint's. Returns its path."""
    start = time.time()
    lp = cache_mod.load_policy(model, device)
    if not getattr(lp, 'absolute', False) or getattr(lp, 'bimanual', False):
        raise ValueError(
            f'{model}: on-policy corrections are for one-arm absolute checkpoints (15-D state, 8-D action)'
        )
    cfg = lp.policy.config
    cameras = cameras_of(lp)
    loaded = time.time()
    parts, batch, sources, seen, fps = [], [], {}, 0, None

    def flush():
        if batch:
            collated = torch.utils.data.default_collate(batch)
            parts.append({k: v.cpu() for k, v in cache_mod.rows(lp, collated).items()})
            batch.clear()

    for n, bag in enumerate(bags):
        e = load_expert(bag, expert)
        fps = fps or e['meta']['fps']
        if e['meta']['fps'] != fps:
            raise ValueError(f'{bag}: expert frames at {e["meta"]["fps"]} fps, the others at {fps}')
        if lead is None:
            lead = lead_steps(model, fps)
        episode = seed(bag) if seed(bag) is not None else n
        if episode in sources:
            raise ValueError(f'{bag}: seed {episode} also in {sources[episode]}')
        sources[episode] = str(Path(bag).resolve())
        rows = np.flatnonzero(e['valid'])
        seen += len(e['valid'])
        labels = relabel(e['chunk'][rows], cfg.horizon, lead)
        for k, (i, obs) in enumerate(zip(rows, observations(bag, e, rows, cameras, arm), strict=True)):
            batch.append(item(lp, *obs, labels[k], episode, int(i)))
            if len(batch) == batch_size:
                flush()
    flush()
    if not parts:
        raise ValueError('no frame with a plan in the bags')
    tensors = cache_mod.typed({k: torch.cat([p[k] for p in parts]) for k in parts[0]})
    done = time.time()
    n = len(tensors['episode'])
    meta = {
        'model': str(Path(model).resolve()) if Path(model).exists() else str(model),
        'source': 'on-policy',
        'bags': sources,
        'arm': arm,
        'cameras': cameras,
        'fps': fps,
        'lead_steps': lead,
        'episodes': sorted(sources),
        'frames': n,
        'frames_without_plan': seen - n,
        'horizon': cfg.horizon,
        'n_action_steps': cfg.n_action_steps,
        'device': cache_mod.device_name(device),
        'load_s': round(loaded - start, 2),
        'encode_s': round(done - loaded, 2),
        'per_frame_s': round((done - loaded) / max(n, 1), 5),
    }
    return cache_mod.save(tensors, meta, out)


def export(cache_path, root, repo_id, task, gate=3.0, pool=None, robot_type='baxter', log=print):
    """The corrected frames of an on-policy cache (`pool` seeds, all if None; `gate` cm) as a LeRobot
    dataset at `root` (see the module docstring), with meta/rethoughtai_sources.json naming each
    episode's attempt and frame. Returns the number of episodes."""
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    from adapter.lerobot_export import features
    from inference.policies import raw_state

    tensors, meta = cache_mod.load(cache_path)
    if meta.get('source') != 'on-policy':
        raise ValueError(f'{cache_path}: not an on-policy cache (ttt onpolicy)')
    pool = sorted(meta['episodes']) if pool is None else list(pool)
    steps = meta['n_action_steps']
    mask = corrector.corrected(tensors, pool, gate, steps)
    error = corrector.path_error(tensors['chunk_pi'], tensors['chunk_demo'], tensors['pad'], steps)
    cameras = meta['cameras']
    dataset, summary = None, []
    for episode in pool:
        rows = torch.nonzero(mask & (tensors['episode'] == episode)).flatten()
        if not len(rows):
            continue
        bag = meta['bags'][str(episode)]
        e = load_expert(bag)
        frames = tensors['frame'][rows].numpy()
        for r, obs in zip(rows.tolist(), observations(bag, e, frames, cameras, meta['arm']), strict=True):
            images, joints, tcp, gripper = obs
            if dataset is None:
                shapes = {key: img.shape for key, img in images.items()}
                dataset = LeRobotDataset.create(
                    repo_id,
                    int(round(meta['fps'])),
                    features(cameras, meta['arm'], shapes),
                    root=root,
                    robot_type=robot_type,
                    streaming_encoding=True,
                    encoder_threads=2,
                )
            row = {f'observation.images.{key}': img for key, img in images.items()}
            row['observation.state'] = raw_state(*tcp, gripper, joints)
            row['task'] = task
            for action in tensors['chunk_demo'][r].numpy():
                dataset.add_frame({**row, 'action': action.astype(np.float32)})
            dataset.save_episode()
            frame = int(tensors['frame'][r])
            summary.append({
                'episode_index': len(summary),
                'seed': int(episode),
                'frame': frame,
                't': int(e['t'][frame]),
                'path_error_cm': round(float(error[r]) * 100, 2),
            })
        log(f'seed {episode}: {len(rows)} corrected frames')
    if dataset is None:
        raise ValueError(f'no corrected frame in seeds {pool} at gate {gate} cm')
    dataset.finalize()
    (Path(root) / SOURCES).write_text(
        json.dumps(
            {
                'source': 'on-policy corrections',
                'cache': str(Path(cache_path).resolve()),
                'model': meta['model'],
                'gate_cm': gate,
                'lead_steps': meta['lead_steps'],
                'horizon': meta['horizon'],
                'episodes': summary,
            },
            indent=1,
        )
    )
    return len(summary)
