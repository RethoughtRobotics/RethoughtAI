"""The latent cache: a frozen VITA base run once over every frame of a LeRobot dataset.

One row per frame: `feat`, the observer's output (each camera's ResNet features and the normalised
state), which the observation encoder maps to `z0`; `z_pi`, the base's flowed latent, and
`chunk_pi`, its decoded chunk; `chunk_demo`, the demonstration's chunk from that frame (past the
episode's end padded with its last action, `pad`, as training pads it) and `z_demo`, its action
encoder latent; `state`; `episode`, `frame`. Chunks are unnormalised, the dataset's absolute goals
(position in m, quaternion, gripper). Latents are stored as float16, chunks as float32.

Frames go through the policy as the runtime feeds them: the checkpoint's own normaliser, the
cameras at their recorded size, resized by the policy (bilinear, no antialias, as training resized
them) and centre cropped.
"""

import json
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

FILE = 'cache.safetensors'
LATENTS = ('feat', 'z0', 'z_pi', 'z_demo')


def load_policy(model, device='cpu'):
    """The checkpoint through the runtime's loading (inference.policies.LeRobotPolicy), in eval mode."""
    from inference.policies import LeRobotPolicy

    return LeRobotPolicy(str(model), device)


def normalise(lp, batch):
    """The batch's observation features and action, normalised by the checkpoint's preprocessor."""
    keys = [*lp.features, 'action']
    return lp.pre({k: batch[k] for k in keys if k in batch})


def flow(vita, z0):
    """The base's plan from z0: the flowed latent and its decoded (normalised) chunk."""
    z = vita.flow_matcher.sample(vita.flow_net, start=z0)
    return z, vita.action_decoder(z)


@torch.no_grad()
def rows(lp, batch):
    """One batch of frames (a LeRobotDataset batch with the action chunk) to cache rows."""
    policy, vita = lp.policy, lp.policy.vita
    nb = policy._with_time_dim(normalise(lp, batch))
    feat = vita.observer(nb)
    z0 = vita.obs_encoder(feat)
    z_pi, chunk = flow(vita, z0)
    return {
        'feat': feat,
        'z0': z0,
        'z_pi': z_pi,
        'z_demo': vita.action_encoder(nb['action']),
        'chunk_pi': lp.post(chunk),
        'chunk_demo': batch['action'],
        'pad': batch['action_is_pad'],
        'state': batch['observation.state'],
        'episode': batch['episode_index'],
        'frame': batch['frame_index'],
    }


def dataset(repo_id, revision, episodes, horizon, fps=None):
    """The LeRobotDataset of `episodes` with each frame's action chunk of `horizon` steps."""
    from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata

    fps = fps or LeRobotDatasetMetadata(repo_id, revision=revision).fps
    return LeRobotDataset(
        repo_id,
        revision=revision,
        episodes=list(episodes) if episodes is not None else None,
        delta_timestamps={'action': [k / fps for k in range(horizon)]},
        video_backend='torchcodec',
    )


def build(model, repo_id, revision, out, episodes=None, device='cpu', batch_size=32, workers=4, frames=None):
    """Run the base `model` over the dataset's `episodes` (all if None) and write the cache to the
    directory `out`; `frames`, a dataset to read instead of the Hub's (tests). Returns its path."""
    start = time.time()
    lp = load_policy(model, device)
    cfg = lp.policy.config
    ds = frames if frames is not None else dataset(repo_id, revision, episodes, cfg.horizon)
    loaded = time.time()
    parts = []
    for batch in DataLoader(ds, batch_size=batch_size, num_workers=workers, shuffle=False):
        parts.append({k: v.cpu() for k, v in rows(lp, batch).items()})
    tensors = typed({k: torch.cat([p[k] for p in parts]) for k in parts[0]})
    done = time.time()
    n = len(tensors['episode'])
    meta = {
        'model': str(Path(model).resolve()) if Path(model).exists() else str(model),
        'repo_id': repo_id,
        'revision': revision,
        'fps': getattr(ds, 'fps', None),
        'episodes': sorted({int(e) for e in tensors['episode']}),
        'frames': n,
        'horizon': cfg.horizon,
        'n_action_steps': cfg.n_action_steps,
        'device': device_name(device),
        'load_s': round(loaded - start, 2),
        'encode_s': round(done - loaded, 2),
        'per_frame_s': round((done - loaded) / max(n, 1), 5),
    }
    return save(tensors, meta, out)


def typed(tensors):
    """The rows' tensors in the cache's dtypes."""
    for k in LATENTS:
        tensors[k] = tensors[k].half()
    for k in ('chunk_pi', 'chunk_demo', 'state'):
        tensors[k] = tensors[k].float()
    for k in ('episode', 'frame'):
        tensors[k] = tensors[k].int()
    return tensors


def save(tensors, meta, out):
    """Write a cache (and its cache.json) to the directory `out`; returns its path."""
    from safetensors.torch import save_file

    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    save_file({k: v.contiguous() for k, v in tensors.items()}, str(out / FILE), metadata={'meta': json.dumps(meta)})
    (out / 'cache.json').write_text(json.dumps(meta, indent=2) + '\n')
    return out / FILE


def load(path, device='cpu'):
    """(tensors, meta) of the cache at `path` (its directory or file)."""
    from safetensors import safe_open

    path = Path(path)
    if path.is_dir():
        path = path / FILE
    with safe_open(str(path), framework='pt', device=str(device)) as f:
        meta = json.loads(f.metadata()['meta'])
        tensors = {k: f.get_tensor(k) for k in f.keys()}
    return tensors, meta


def device_name(device):
    device = torch.device(device)
    if device.type == 'cuda':
        return torch.cuda.get_device_name(device)
    return f'cpu ({torch.get_num_threads()} threads)'
