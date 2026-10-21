"""Seeded exploration in VITA's latent, for test-time training from reward (plan S7 of
training/sim_experiments.md, the real plan's design B): an attempt plans from z0 + sigma eps, eps drawn
once from `seed` and kept for every plan of the attempt, so that attempts differ by their offset.

With `noise` s the flow's Euler integration is stochastic, each step x' = x + f(x, t) dt + s n with
n ~ N(0, I): a denoising MDP over the latent, as DPPO treats a diffusion policy's steps, whose per-step
log-probability (`step_logp`, a Gaussian of standard deviation s around the deterministic step) the
policy gradient takes. The noise is drawn from a generator seeded by `seed` as well.

With `record` (a file) every plan is appended to it as it is made, so an attempt cut off keeps the plans
it made: per plan, np.save arrays in the order of FIELDS (read back by `records`), with a json header
next to it (<record>.json: sigma, seed, noise, flow steps, latent size). The policy node tells the
explorer whether the runtime is engaged (`engaged`), so the attempt's own plans can be told from those
made before the engage and after the release.
"""

import json
import math
import time
import types
from pathlib import Path

import numpy as np
import torch

# feat: the observer's output (the observation encoder's input); z0: the observation latent, without the
# offset; trajectory: the latent at each flow step, (steps + 1, D), the first z0 + sigma eps and the last
# the flowed latent the chunk is decoded from; logp: each step's log-probability (empty without noise);
# engaged: 1 engaged, 0 not, -1 unknown; time: wall time of the plan
FIELDS = ('feat', 'z0', 'trajectory', 'logp', 'engaged', 'time')


def offset(dim, seed):
    """eps of attempt `seed`: (dim,) standard normal, the same on every device."""
    return torch.randn(dim, generator=torch.Generator().manual_seed(seed))


def step_logp(x, mean, std):
    """log N(x; mean, std^2 I) per row, summed over the last dimension."""
    d = x.shape[-1]
    return -0.5 * (((x - mean) / std) ** 2).sum(-1) - d * (math.log(std) + 0.5 * math.log(2 * math.pi))


def step_mean(vita, x, k, steps):
    """The deterministic Euler step k of `steps` from x; k an int, or a tensor of one per row."""
    if torch.is_tensor(k):
        t = (k / steps).to(x.device, x.dtype)
    else:
        t = torch.full((x.shape[0],), k / steps, device=x.device, dtype=x.dtype)
    return x + vita.flow_net(x, t) * (1.0 / steps)  # as flow_matcher.sample


def flow(vita, start, noise=0.0, generator=None):
    """(trajectory (B, steps + 1, D), per-step log-probabilities (B, steps), empty without noise) of
    the flow from `start`; without noise the trajectory ends where flow_matcher.sample ends."""
    steps = vita.flow_matcher.num_sampling_steps
    x, xs, logps = start, [start], []
    for k in range(steps):
        mean = step_mean(vita, x, k, steps)
        if noise > 0:
            n = torch.randn(x.shape, generator=generator, device=x.device, dtype=x.dtype)
            x = mean + noise * n
            logps.append(step_logp(x, mean, noise))
        else:
            x = mean
        xs.append(x)
    logp = torch.stack(logps, 1) if logps else start.new_zeros(start.shape[0], 0)
    return torch.stack(xs, 1), logp


class Explorer:
    """Installs itself as the VITA model's generate_actions (see the module docstring)."""

    def __init__(self, vita, sigma, seed, noise=0.0, record=None):
        if sigma < 0 or noise < 0:
            raise ValueError(f'sigma {sigma} and noise {noise}: at least 0')
        w = next(vita.parameters())
        self.vita, self.sigma, self.seed, self.noise = vita, float(sigma), int(seed), float(noise)
        self.eps = offset(vita.config.latent_dim, self.seed).to(w.device, w.dtype)
        self.generator = torch.Generator(w.device).manual_seed(self.seed)
        self.engaged = None
        self.file = None
        if record:
            record = Path(record)
            record.parent.mkdir(parents=True, exist_ok=True)
            header = {
                'sigma': self.sigma,
                'seed': self.seed,
                'noise': self.noise,
                'steps': vita.flow_matcher.num_sampling_steps,
                'latent_dim': vita.config.latent_dim,
                'fields': list(FIELDS),
            }
            Path(f'{record}.json').write_text(json.dumps(header, indent=2) + '\n')
            self.file = open(record, 'ab')  # noqa: SIM115  open for the runtime's life, flushed per plan
        vita.generate_actions = types.MethodType(lambda _, batch: self.generate(batch), vita)

    def generate(self, batch):
        vita = self.vita
        feat = vita.observer(batch)
        z0 = vita.obs_encoder(feat)
        trajectory, logp = flow(vita, z0 + self.sigma * self.eps, self.noise, self.generator)
        chunk = vita.action_decoder(trajectory[:, -1])
        residual = getattr(vita, 'residual', None)
        if residual is not None:
            chunk = chunk + residual(chunk, z0)
        if self.file is not None:
            self.write(feat, z0, trajectory, logp)
        return chunk

    def write(self, feat, z0, trajectory, logp):
        n = len(feat)
        engaged = -1 if self.engaged is None else int(bool(self.engaged))
        values = {
            'feat': feat,
            'z0': z0,
            'trajectory': trajectory,
            'logp': logp,
            'engaged': np.full(n, engaged, np.int8),
            'time': np.full(n, time.time()),
        }
        for k in FIELDS:
            v = values[k]
            np.save(self.file, v.float().cpu().numpy() if torch.is_tensor(v) else v)
        self.file.flush()


def records(path):
    """{field: array} of every plan in a record file, concatenated; a last plan cut off while it was
    written is left out."""
    rows = []
    with open(path, 'rb') as f:
        while True:
            row = []
            try:
                for _ in FIELDS:
                    row.append(np.load(f))
            except (EOFError, ValueError, OSError):
                break
            rows.append(row)
    if not rows:
        raise ValueError(f'{path}: no plan recorded')
    return {k: np.concatenate([r[i] for r in rows]) for i, k in enumerate(FIELDS)}


def header(path):
    """The record's <record>.json."""
    return json.loads(Path(f'{path}.json').read_text())
