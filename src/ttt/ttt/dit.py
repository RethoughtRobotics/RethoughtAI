"""Test-time training of a frozen Multi-Task DiT: the base's cache rows, its LoRA and its loss.

The DiT has no latent to flow: its denoiser predicts the noise of an action chunk conditioned on the
observation encoder's vector (CLIP features of each camera, the state and the task's text features
over its n_obs_steps frames). The encoder stays frozen, so the cache holds that vector, `cond`, per
frame, and the update trains LoRA on the denoiser (lora.DIT_TARGETS) with the DiT's own training loss
(its noise or velocity error at random timesteps) on the cached `cond` and the target chunk.

Rows: `cond`; `target`, the demonstration's chunk as training aligns it (from frame t - n_obs_steps + 1,
`horizon` steps) and `target_pad`; `chunk_demo` and `pad`, its steps from frame t on, and `chunk_pi`, the
base's plan over the same steps (the corrector and the open-loop error compare these as on VITA);
`state`, `episode`, `frame`. A corrected frame's target is the demonstration's chunk, as M0's z_demo.

Plans are sampled from seeded noise (PLAN_SEED per batch of rows), so the base and an adapter plan from
the same noise and their open-loop errors pair.
"""

import torch

from inference import lora

TYPE = 'multi_task_dit'
PLAN_SEED = 0


def is_dit(policy):
    return getattr(policy.config, 'type', None) == TYPE


def check(policy):
    """The DiT variants the cached conditioning covers: no bind (its context is image patches), no energy
    (its prediction is a gradient), RoPE attention (LoRA wraps its projections)."""
    cfg = policy.config
    if cfg.bind or cfg.energy or not cfg.use_rope:
        raise ValueError('ttt on the Multi-Task DiT needs bind and energy off and use_rope on')


def start(policy):
    """Index of frame t in a chunk as training aligns it."""
    return policy.config.n_obs_steps - 1


def dataset(repo_id, revision, episodes, cfg, fps=None):
    """The LeRobotDataset of `episodes` with each frame's observations and action chunk as the DiT trains
    on them (its observation and action delta indices)."""
    from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata

    fps = fps or LeRobotDatasetMetadata(repo_id, revision=revision).fps
    obs = [k / fps for k in cfg.observation_delta_indices]
    delta = {k: obs for k in cfg.input_features}
    delta['action'] = [k / fps for k in cfg.action_delta_indices]
    return LeRobotDataset(
        repo_id,
        revision=revision,
        episodes=list(episodes) if episodes is not None else None,
        delta_timestamps=delta,
        video_backend='torchcodec',
    )


def conditioning(lp, batch):
    """The observation encoder's vector of a batch (normalised by the checkpoint's preprocessor, the
    task tokenised)."""
    policy = lp.policy
    keys = [*lp.features, 'action', 'task']
    nb = lp.pre({k: batch[k] for k in keys if k in batch})
    return policy.observation_encoder.encode(policy._prepare_batch(nb))


def plan(policy, cond, seed=PLAN_SEED):
    """The normalised chunk the DiT samples from `cond` with noise seeded by `seed`, from frame t on."""
    cuda = [cond.device] if cond.device.type == 'cuda' else []
    with torch.random.fork_rng(devices=cuda):
        torch.manual_seed(seed)
        chunk = policy.objective.conditional_sample(policy.noise_predictor, len(cond), cond)
    return chunk[:, start(policy) :]


@torch.no_grad()
def rows(lp, batch):
    """One batch of frames to cache rows."""
    cond = conditioning(lp, batch)
    s = start(lp.policy)
    return {
        'cond': cond,
        'chunk_pi': lp.post(plan(lp.policy, cond)),
        'target': batch['action'],
        'target_pad': batch['action_is_pad'],
        'chunk_demo': batch['action'][:, s:],
        'pad': batch['action_is_pad'][:, s:],
        'state': batch['observation.state'][:, -1],
        'episode': batch['episode_index'],
        'frame': batch['frame_index'],
    }


def prepare(policy, rank=16, alpha=1.0):
    """LoRA on the denoiser's attention and MLP layers, everything else frozen; returns (parameters,
    (adapter root, its name))."""
    check(policy)
    policy.requires_grad_(False)
    root = policy.get_submodule(lora.DIT_ROOT)
    lora.wrap(root, lora.targets(root, lora.DIT_TARGETS), rank, alpha)
    return lora.parameters(root), (root, lora.DIT_ROOT)


def loss(policy, cond, action, pad=None):
    """The DiT's training loss (batch mean) on the normalised target chunks `action` given `cond`."""
    batch = {'action': action, **({'action_is_pad': pad} if pad is not None else {})}
    return policy.objective.compute_loss(policy.noise_predictor, batch, cond)


class Cached:
    """Rows of a DiT cache on the device: conditioning and normalised target chunks."""

    def __init__(self, lp, tensors, rows, device):
        from .update import normalised_actions

        self.device = device
        self.cond = tensors['cond'][rows].float().to(device)
        self.action = normalised_actions(lp, tensors['target'][rows].float().to(device))
        self.pad = tensors['target_pad'][rows].to(device)

    def __len__(self):
        return len(self.cond)

    def batches(self, index):
        for i in index:
            yield i.to(self.device)

    def loss(self, policy, i):
        return loss(policy, self.cond[i], self.action[i], self.pad[i])


@torch.no_grad()
def decode(lp, tensors, rows, batch=1024):
    """The unnormalised chunks (from frame t on) the policy plans from the cached `cond` of `rows`."""
    out = []
    for i in range(0, len(rows), batch):
        cond = tensors['cond'][rows[i : i + batch]].float().to(lp.device)
        out.append(lp.post(plan(lp.policy, cond)).float().cpu())
    return torch.cat(out) if out else torch.zeros(0)
