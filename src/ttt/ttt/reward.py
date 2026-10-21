"""Test-time training from reward instead of corrections (plan S7 of training/sim_experiments.md, the
real plan's design B): the policy's own attempts, scored, and no corrector.

An attempt is one closed-loop evaluator attempt run with seeded exploration (inference.explore: every
plan from z0 + sigma eps_i, eps_i fixed for the attempt) and its plans recorded: the evaluator's row
and the record file. The evaluator runs one attempt per runtime, so `attempts_of` takes them from an
evaluator run whose stacks each scored one seed (stack_*/results.jsonl and stack_*/RECORD). Only the
plans made while the runtime was engaged count (all of them if the record has no engaged flag).

Reward r_i = success (0 or 1) + 0.5 x progress, progress the phase reached as a fraction of the four
(PHASES: reach 1/4, grasp 1/2, transfer 3/4, release 1; none 0).

Two updates of a LoRA adapter on the flow net (M0's: rank 16, lr 1e-4, the anchor replaying the
dataset cache's base pairs with weight lambda 0.5), continuing the adapter the attempts ran with
(`init`; none for the base's attempts):
  rwr  reward-weighted regression (S7a): w = softmax(beta r), beta solving KL(w || U) = ln 2
       (TTT-Discover's temperature, `temperature`; equal rewards give uniform weights), and the flow
       error of each attempt's own pairs (z0, z1_i), z1_i the latent the policy flowed from
       z0 + sigma eps_i, weighted so that the loss is sum_i w_i E_{z0 in i} l(z0, z1_i)
       (`row_weights`, `weighted_loss`): batches drawn uniformly over the frames, each frame's error
       times w_i / n_i * N
  pg   a DPPO-style policy gradient (S7b): the flow's Euler steps, run stochastic by the runtime
       (its flow noise s), are the actions of a denoising MDP over the latent, each a Gaussian of
       standard deviation s around the deterministic step (inference.explore.step_logp). The PPO
       clipped surrogate (`pg_loss`) over the recorded steps, every step of attempt i with the
       advantage r_i - mean r (no critic, no normalisation); the old log-probabilities are the
       adapter's the attempts ran with, recomputed before the first update step (the runtime's own,
       recorded, are kept as a check: `logp_gap`). Equal rewards leave nothing to follow: the
       adapter is saved as it was.
Every output gets an update.json as ttt update's: settings, attempts (seed, reward, weight, frames),
the temperature, losses, wall time and device.
"""

import dataclasses
import json
import math
import time
from pathlib import Path

import torch

from inference import explore, lora

from . import BASE, corrector, update
from . import cache as cache_mod

PHASES = ('reach', 'grasp', 'transfer', 'release')
KL = math.log(2)
RECORD = 'plans.npy'  # the evaluator stack's record file


@dataclasses.dataclass
class Settings:
    method: str = 'rwr'
    rank: int = 16
    alpha: float = 1.0
    lam: float = 0.5
    steps: int = 200
    batch: int = 256
    lr: float = 1e-4
    seed: int = 0
    anchor: tuple = tuple(BASE)
    kl: float = KL
    progress: float = 0.5
    clip: float = 0.1
    log_every: int = 10


def progress(phase):
    """The phase reached as a fraction of PHASES, 0 for none."""
    return 0.0 if phase is None else (PHASES.index(phase) + 1) / len(PHASES)


def reward(row, weight=0.5):
    """r = success + weight x progress of an evaluator row."""
    return float(bool(row['success'])) + weight * progress(row.get('phase'))


def kl_uniform(w):
    """KL(w || U) of weights over K attempts: sum_i w_i ln(K w_i)."""
    w = torch.as_tensor(w, dtype=torch.float64)
    return float(torch.special.xlogy(w, w * len(w)).sum())


def temperature(r, kl=KL, iterations=200):
    """beta >= 0 with KL(softmax(beta r) || U) = kl: 0 if the rewards are equal, inf if no beta gets
    there (more than K e^-kl attempts tie at the best: the weights' limit, uniform over those, is as far
    from uniform as they go)."""
    r = torch.as_tensor(r, dtype=torch.float64)
    r = r - r.max()
    k = len(r)
    if bool((r == 0).all()):
        return 0.0
    if math.log(k / int((r == 0).sum())) <= kl:
        return math.inf

    def gap(beta):
        return kl_uniform(torch.softmax(beta * r, 0)) - kl

    lo, hi = 0.0, 1.0
    while gap(hi) < 0:
        lo, hi = hi, 2 * hi
    for _ in range(iterations):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if gap(mid) < 0 else (lo, mid)
    return (lo + hi) / 2


def weights(r, kl=KL):
    """(w, beta): w = softmax(beta r), beta by `temperature`; with beta inf, uniform over the best."""
    r = torch.as_tensor(r, dtype=torch.float64)
    beta = temperature(r, kl)
    if math.isinf(beta):
        best = (r == r.max()).double()
        return best / best.sum(), beta
    return torch.softmax(beta * (r - r.max()), 0), beta


def row_weights(attempt, w):
    """Per frame of attempt index `attempt` (N,), w_i / n_i * N: the mean over uniformly drawn frames
    of this times their error estimates sum_i w_i E_{frames of i} error."""
    n = torch.bincount(attempt, minlength=len(w)).double()
    return (w.double()[attempt] / n[attempt] * len(attempt)).float()


def weighted_loss(error, row_weight):
    """The reward-weighted flow loss of a batch: mean(row weight x per-frame error)."""
    return (row_weight * error).mean()


def pg_loss(logp, old_logp, advantage, clip):
    """(loss, fraction clipped): PPO's clipped surrogate, -mean(min(rho A, clip(rho, 1 - c, 1 + c) A)),
    rho = exp(logp - old_logp) per step."""
    ratio = torch.exp(logp - old_logp)
    clipped = ratio.clamp(1 - clip, 1 + clip)
    loss = -torch.minimum(ratio * advantage, clipped * advantage).mean()
    return loss, ((ratio - 1).abs() > clip).float().mean()


@dataclasses.dataclass
class Attempt:
    seed: int
    row: dict
    record: Path

    def plans(self):
        """The record's plans while engaged (all if it has no engaged flag), as tensors."""
        rec = explore.records(self.record)
        keep = rec['engaged'] == 1
        if not keep.any():
            keep = rec['engaged'] == -1
        if not keep.any():
            raise ValueError(f'{self.record}: no plan while the runtime was engaged')
        return {k: torch.from_numpy(v[keep]) for k, v in rec.items()}

    def header(self):
        return explore.header(self.record)


def attempts_of(runs, record=RECORD):
    """The attempts of evaluator runs (directories with stack_*/results.jsonl), one seed per stack,
    in seed order."""
    found = {}
    for run in [runs] if isinstance(runs, (str, Path)) else runs:
        stacks = sorted(Path(run).glob('stack_*/results.jsonl'))
        if not stacks:
            raise ValueError(f'{run}: no stack_*/results.jsonl')
        for path in stacks:
            rows = [json.loads(x) for x in path.read_text().splitlines() if x.strip()]
            if len(rows) != 1:
                raise ValueError(f'{path}: {len(rows)} attempts; an explored attempt needs a runtime of its own')
            rec = path.parent / record
            if not rec.exists():
                raise ValueError(f'{rec}: no record of the attempt (serve --record)')
            seed = rows[0]['seed']
            if seed in found:
                raise ValueError(f'{path}: seed {seed} also in {found[seed].record.parent}')
            found[seed] = Attempt(seed, rows[0], rec)
    return [found[s] for s in sorted(found)]


def resume(root, path):
    """Load the LoRA adapter at `path` into root's freshly wrapped adapters (the same layers and rank)."""
    from safetensors.torch import load_file

    meta, layers = lora.meta(path), lora.adapters(root)
    if meta.get('kind') or set(meta['targets']) != set(layers) or meta['rank'] != next(iter(layers.values())).rank:
        raise ValueError(f'{path}: not a LoRA adapter on the same layers at the same rank')
    tensors = load_file(str(Path(path) / lora.WEIGHTS))
    with torch.no_grad():
        for name, m in layers.items():
            for k in 'AB':
                getattr(m, k).copy_(tensors[f'{name}.{k}'])


def run(model, cache_path, runs, out, settings=None, device='cpu', init=None, record=RECORD):
    """Update on the attempts of the evaluator `runs` (see the module docstring), the anchor from the
    cache at `cache_path`, from the adapter `init` (a fresh one if None); save the adapter to `out`.
    Returns the update.json record."""
    s = settings or Settings()
    if s.method not in ('rwr', 'pg'):
        raise ValueError(f'method {s.method!r}: rwr or pg')
    started = time.time()
    attempts = attempts_of(runs, record)
    headers = [a.header() for a in attempts]
    if len({h['sigma'] for h in headers}) > 1:
        raise ValueError(f'attempts explored with different sigmas {sorted({h["sigma"] for h in headers})}')
    if s.method == 'pg' and any(h['noise'] <= 0 for h in headers):
        raise ValueError('the policy gradient needs a stochastic flow: attempts recorded with --flow-noise')
    if s.method == 'pg' and len({h['noise'] for h in headers}) > 1:
        raise ValueError('attempts with different flow noise')
    plans = [a.plans() for a in attempts]
    rewards = torch.tensor([reward(a.row, s.progress) for a in attempts], dtype=torch.float64)
    w, beta = weights(rewards, s.kl)

    tensors, meta = cache_mod.load(cache_path)
    model = model or meta['model']
    lp = cache_mod.load_policy(model, device)
    policy, vita = lp.policy, lp.policy.vita
    torch.manual_seed(s.seed)  # LoRA's A initialisation, when no adapter is continued
    params, (root, name) = update.prepare(policy, 'lora', s.rank, s.alpha)
    if init:
        resume(root, init)
    policy.eval()
    use_anchor = s.lam > 0
    d_rows = torch.nonzero(corrector.in_episodes(tensors, s.anchor)).flatten()
    if use_anchor and len(d_rows) == 0:
        raise ValueError(f'no anchor frame in episodes {list(s.anchor)}')
    anchor_set = update.Cached(lp, tensors, d_rows, device) if use_anchor else None

    attempt = torch.cat([torch.full((len(p['feat']),), i) for i, p in enumerate(plans)])
    feat = torch.cat([p['feat'] for p in plans]).to(device)
    trajectory = torch.cat([p['trajectory'] for p in plans]).to(device)
    n_steps = trajectory.shape[1] - 1
    extra = {}
    g = torch.Generator().manual_seed(s.seed)
    steps = s.steps
    if s.method == 'rwr':
        data = {'feat': feat, 'z1': trajectory[:, -1], 'weight': row_weights(attempt, w).to(device)}
        n = len(feat)
    else:
        noise = headers[0]['noise']
        k = torch.arange(n_steps).repeat(len(feat))
        frame = torch.arange(len(feat)).repeat_interleave(n_steps)
        advantage = (rewards - rewards.mean()).float()[attempt[frame]]
        data = {
            'x': trajectory[frame, k],
            'next': trajectory[frame, k + 1],
            'k': k.to(device),
            'advantage': advantage.to(device),
        }
        with torch.no_grad():
            data['old'] = torch.cat([
                explore.step_logp(data['next'][i], explore.step_mean(vita, data['x'][i], data['k'][i], n_steps), noise)
                for i in torch.arange(len(k)).split(4096)
            ])
        recorded = torch.cat([p['logp'] for p in plans]).flatten().to(device)
        extra['logp_gap'] = float((recorded - data['old']).abs().max()) if recorded.numel() == len(k) else None
        n = len(k)
        if bool((rewards == rewards[0]).all()):
            steps, extra['skipped'] = 0, 'equal rewards: no advantage'

    shape = (steps, s.batch)
    idx, tc = torch.randint(n, shape, generator=g), torch.rand(shape, generator=g)
    if use_anchor:
        id_, td = torch.randint(len(anchor_set), shape, generator=g), torch.rand(shape, generator=g)
    opt = torch.optim.AdamW(params, lr=s.lr, betas=(0.95, 0.999), weight_decay=0.0)
    log = []
    update.sync(device)
    t0 = time.time()
    batches_d = anchor_set.batches(id_) if use_anchor else None
    for step in range(steps):
        i = idx[step].to(device)
        if s.method == 'rwr':
            z0 = vita.obs_encoder(data['feat'][i])
            error = update.pair_error(vita, z0, data['z1'][i], tc[step].to(device), 'flow')
            loss_c = weighted_loss(error, data['weight'][i])
            row = {'step': step + 1, 'weighted': loss_c.item()}
        else:
            mean = explore.step_mean(vita, data['x'][i], data['k'][i], n_steps)
            logp = explore.step_logp(data['next'][i], mean, noise)
            loss_c, clipped = pg_loss(logp, data['old'][i], data['advantage'][i], s.clip)
            row = {'step': step + 1, 'pg': loss_c.item(), 'clipped': clipped.item()}
        total = (1 - s.lam) * loss_c
        if use_anchor:
            z0d, z1d, _ = anchor_set.pair(vita, next(batches_d))
            loss_d = update.pair_error(vita, z0d, z1d, td[step].to(device), 'flow').mean()
            total = total + s.lam * loss_d
            row['anchor'] = loss_d.item()
        opt.zero_grad(set_to_none=True)
        total.backward()
        opt.step()
        row['total'] = total.item()
        if step == 0 or (step + 1) % s.log_every == 0 or step + 1 == steps:
            log.append(row)
    update.sync(device)
    update_s = time.time() - t0

    out = Path(out)
    lora.save(root, out, name=name, trainable='lora')
    frames = torch.bincount(attempt, minlength=len(attempts)).tolist()
    record_ = {
        'settings': {**dataclasses.asdict(s), 'anchor': list(s.anchor)},
        'method': s.method,
        'model': str(model),
        'cache': str(Path(cache_path).resolve()),
        'init': str(Path(init).resolve()) if init else None,
        'output': 'adapter',
        'attempts': [
            {
                'seed': a.seed,
                'success': bool(a.row['success']),
                'phase': a.row.get('phase'),
                'reward': float(r),
                'weight': float(wi),
                'frames': f,
                'sigma': h['sigma'],
                'explore_seed': h['seed'],
                'record': str(a.record.resolve()),
            }
            for a, r, wi, f, h in zip(attempts, rewards, w, frames, headers, strict=True)
        ],
        'beta': beta if math.isfinite(beta) else 'inf',
        'kl': kl_uniform(w),
        'frames': sum(frames),
        'samples': n,
        'flow_steps': n_steps,
        **extra,
        'anchor_frames': len(d_rows) if use_anchor else 0,
        'trainable_parameters': sum(p.numel() for p in params),
        'update_s': round(update_s, 3),
        'total_s': round(time.time() - started, 3),
        'device': cache_mod.device_name(device),
        'losses': log,
    }
    (out / 'update.json').write_text(json.dumps(record_, indent=2) + '\n')
    return record_
