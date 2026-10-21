"""The update loop: L1 (flow matching on corrected pairs, the anchor replaying base pairs with weight
lambda), optionally plus L_pref (the base's own flowed latent as the rejected sample, beta its scale).

For a pair (z0, z1) and t ~ U(0, 1), z_t = (1 - t) z0 + t z1 and the per-pair error is
|f(z_t, t) - (z1 - z0)|^2 (the `flow` objective). Pairs are the cache's own (z0, z_demo), with no
optimal-transport re-pairing: the correction says which latent this z0 should flow to. Trainable sets
that include the observation encoder or the action decoder use the base's own training objective
(`vita`: that error plus its consistency and reconstruction terms, with the checkpoint's weights),
since the flow error alone does not reach the decoder.

Trainable sets (TRAINABLE), named after the plans that use them:
  lora       LoRA on the flow net's fc1 and fc2 (M0); adapter names relative to the flow net
  flow       the whole flow net (S3b)
  flow_last  the flow net's output layer only (S3c)
  all        everything: ResNet, observation encoder, action encoder, flow net, decoder (S3a); it reads
             frames, since z0 and z_demo change with it
  lora_obs   LoRA on the observation encoder, the linear map from the observer's features (S8a)
  lora_dec   LoRA on the action decoder's linear layers (S8b)
  lora_all   LoRA on all three (S8c)
LoRA sets save an adapter (inference.lora, loadable by the runtime's --adapter); the others a full
checkpoint directory (the runtime's --model). Every output gets an update.json: settings, corrected
frames, wall time of the update and the device it ran on.

Corrections come from the cache's own `pool` episodes (the demonstration as the corrector), or with
`corrections` from an on-policy cache (ttt.onpolicy: the base's attempts, the planner as the corrector),
its `pool` attempts (all if None) gated the same way; the anchor is always the cache's.
"""

import copy
import dataclasses
import json
import shutil
import time
from pathlib import Path

import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, Subset

from inference import lora

from . import BASE, M0_POOL, corrector
from . import cache as cache_mod

LORA_SITES = {
    'obs': ('obs_encoder',),
    'flow': tuple(f'flow_net.{p}' for p in lora.VITA_TARGETS),
    'dec': ('action_decoder.*',),
}
TRAINABLE = {
    'lora': ('flow',),
    'lora_obs': ('obs',),
    'lora_dec': ('dec',),
    'lora_all': ('obs', 'flow', 'dec'),
    'flow': None,
    'flow_last': None,
    'all': None,
}
FLOW_ONLY = ('lora', 'flow', 'flow_last')


@dataclasses.dataclass
class Settings:
    trainable: str = 'lora'
    rank: int = 16
    alpha: float = 1.0
    lam: float = 0.5
    steps: int = 200
    batch: int = 256
    lr: float = 1e-4
    seed: int = 0
    beta: float | None = None
    objective: str = 'auto'
    gate: float | None = 3.0
    pool: tuple | None = M0_POOL
    anchor: tuple = tuple(BASE)
    corrections: str | None = None
    log_every: int = 10

    def resolved_objective(self):
        if self.objective != 'auto':
            return self.objective
        return 'flow' if self.trainable in FLOW_ONLY else 'vita'


def prepare(policy, trainable, rank=16, alpha=1.0):
    """Make `trainable` of the policy trainable and everything else frozen; returns
    (parameters, adapter root module and its name, or None for a full checkpoint)."""
    if trainable not in TRAINABLE:
        raise ValueError(f'trainable {trainable!r}: one of {", ".join(TRAINABLE)}')
    vita = policy.vita
    policy.requires_grad_(False)
    sites = TRAINABLE[trainable]
    if sites == ('flow',):
        root, name = vita.flow_net, lora.FLOW_NET
        lora.wrap(root, lora.targets(root), rank, alpha)
        return lora.parameters(root), (root, name)
    if sites:
        root, name = vita, 'vita'
        patterns = [p for s in sites for p in LORA_SITES[s]]
        lora.wrap(root, lora.targets(root, patterns), rank, alpha)
        return lora.parameters(root), (root, name)
    module = {'flow': vita.flow_net, 'flow_last': vita.flow_net.out_proj, 'all': vita}[trainable]
    params = [p for p in module.parameters() if p.dtype.is_floating_point]
    for p in params:
        p.requires_grad_(True)
    return params, None


def pair_error(vita, z0, z1, t, objective, action=None):
    """Per-pair loss (B,): the flow error at t, plus with `vita` the base's consistency and
    reconstruction terms against the normalised chunk `action`."""
    tt = t[:, None]
    loss = ((vita.flow_net((1 - tt) * z0 + tt * z1, t) - (z1 - z0)) ** 2).mean(-1)
    if objective == 'flow':
        return loss
    cfg = vita.config

    def recon(chunk):
        diff = chunk - action
        return (diff.abs() if cfg.recon_loss_type == 'l1' else diff**2).mean((1, 2))

    if cfg.enc_recon_weight > 0:
        loss = loss + cfg.enc_recon_weight * recon(vita.action_decoder(z1))
    if cfg.decode_flow_latents:
        z = vita.flow_matcher.sample(vita.flow_net, start=z0)
        if cfg.consistency_weight > 0:
            loss = loss + cfg.consistency_weight * ((z - z1) ** 2).mean(-1)
        if cfg.flow_recon_weight > 0:
            loss = loss + cfg.flow_recon_weight * recon(vita.action_decoder(z))
    return loss


class Cached:
    """Pairs from the cache: z0 through the (possibly adapted) observation encoder from `feat`."""

    def __init__(self, lp, tensors, rows, device):
        self.device = device
        self.feat = tensors['feat'][rows].float().to(device)
        self.z = {False: tensors['z_demo'][rows].float().to(device), True: tensors['z_pi'][rows].float().to(device)}
        self.action = {
            False: normalised_actions(lp, tensors['chunk_demo'][rows].to(device)),
            True: normalised_actions(lp, tensors['chunk_pi'][rows].to(device)),
        }

    def __len__(self):
        return len(self.feat)

    def batches(self, index):
        for i in index:
            yield i.to(self.device)

    def pair(self, vita, i, negative=False):
        """(z0, z1, normalised chunk of z1) of rows i: z1 the demonstration's, or with `negative`
        the base's own."""
        return vita.obs_encoder(self.feat[i]), self.z[negative][i], self.action[negative][i]


class Frames:
    """Pairs from frames (trainable set `all`): z0 and z_demo through the model's own encoders."""

    def __init__(self, lp, frames, positions, keys, device, workers=4):
        self.lp, self.frames, self.device, self.workers = lp, frames, device, workers
        where = {p: i for i, p in enumerate(positions)}
        self.index = [where[k] for k in keys]

    def __len__(self):
        return len(self.index)

    def batches(self, index):
        flat = [self.index[j] for j in index.flatten().tolist()]
        loader = DataLoader(Subset(self.frames, flat), batch_size=index.shape[1], num_workers=self.workers)
        for batch in loader:
            yield cache_mod.normalise(self.lp, batch)

    def pair(self, vita, batch, negative=False):
        if negative:
            raise ValueError('the preference loss needs cached latents; not available with trainable all')
        nb = self.lp.policy._with_time_dim(batch)
        return vita.encode_observations(nb), vita.action_encoder(nb['action']), nb['action']


def normalised_actions(lp, chunk):
    from lerobot.processor import TransitionKey, create_transition

    return lp.normalizer(create_transition(action=chunk))[TransitionKey.ACTION]


def frame_positions(frames):
    """(episode, frame) of each item of a frames dataset."""
    if hasattr(frames, 'hf_dataset'):
        cols = frames.hf_dataset.select_columns(['episode_index', 'frame_index'])
        return [(int(e), int(f)) for e, f in zip(cols['episode_index'], cols['frame_index'], strict=True)]
    return [(int(frames[i]['episode_index']), int(frames[i]['frame_index'])) for i in range(len(frames))]


def run(model, cache_path, out, settings=None, device='cpu', frames=None, workers=4):
    """Update the base `model` (default: the cache's) on the cache at `cache_path` with `settings` and
    save the result to `out`; `frames`, a dataset for trainable set `all` (default: the cache's
    dataset). Returns the update.json record."""
    s = settings or Settings()
    objective = s.resolved_objective()
    if s.beta is not None and s.trainable == 'all':
        raise ValueError('the preference loss needs cached latents; not available with trainable all')
    started = time.time()
    tensors, meta = cache_mod.load(cache_path)
    model = model or meta['model']
    lp = cache_mod.load_policy(model, device)
    policy, vita = lp.policy, lp.policy.vita
    ref = copy.deepcopy(vita).requires_grad_(False) if s.beta is not None else None
    torch.manual_seed(s.seed)  # LoRA's A initialisation
    params, adapter = prepare(policy, s.trainable, s.rank, s.alpha)
    policy.eval()  # dropout off, the inference crop

    steps = meta['n_action_steps']
    c_tensors, pool = corrections(s, tensors, meta)
    c_rows = torch.nonzero(corrector.corrected(c_tensors, pool, s.gate, steps)).flatten()
    d_rows = torch.nonzero(corrector.in_episodes(tensors, s.anchor)).flatten()
    if len(c_rows) == 0:
        raise ValueError(f'no corrected frame in episodes {list(pool)} at gate {s.gate} cm')
    if s.lam > 0 and len(d_rows) == 0:
        raise ValueError(f'no anchor frame in episodes {list(s.anchor)}')
    if s.trainable == 'all':
        if frames is None:
            frames = cache_mod.dataset(
                meta['repo_id'], meta['revision'], sorted(set(s.pool) | set(s.anchor)), meta['horizon']
            )
        positions = frame_positions(frames)
        keys = list(zip(tensors['episode'].tolist(), tensors['frame'].tolist(), strict=True))
        sets = [Frames(lp, frames, positions, [keys[i] for i in r], device, workers) for r in (c_rows, d_rows)]
    else:
        sets = [Cached(lp, c_tensors, c_rows, device), Cached(lp, tensors, d_rows, device)]
    corrected_set, anchor_set = sets

    g = torch.Generator().manual_seed(s.seed)
    shape = (s.steps, s.batch)
    ic, tc = torch.randint(len(corrected_set), shape, generator=g), torch.rand(shape, generator=g)
    use_anchor = s.lam > 0
    if use_anchor:
        id_, td = torch.randint(len(anchor_set), shape, generator=g), torch.rand(shape, generator=g)
    opt = torch.optim.AdamW(params, lr=s.lr, betas=(0.95, 0.999), weight_decay=0.0)
    log = []
    sync(device)
    t0 = time.time()
    batches_c = corrected_set.batches(ic)
    batches_d = anchor_set.batches(id_) if use_anchor else None
    for step in range(s.steps):
        t = tc[step].to(device)
        bc = next(batches_c)
        z0, z1, a = corrected_set.pair(vita, bc)
        loss_c = pair_error(vita, z0, z1, t, objective, a).mean()
        total, row = (1 - s.lam) * loss_c, {'step': step + 1, 'corrected': loss_c.item()}
        if use_anchor:
            bd = next(batches_d)
            z0d, z1d, ad = anchor_set.pair(vita, bd)
            loss_d = pair_error(vita, z0d, z1d, td[step].to(device), objective, ad).mean()
            total = total + s.lam * loss_d
            row['anchor'] = loss_d.item()
        if ref is not None:
            pref = preference(vita, ref, corrected_set, bc, t, objective, s.beta)
            total = total + pref
            row['pref'] = pref.item()
        opt.zero_grad(set_to_none=True)
        total.backward()
        opt.step()
        row['total'] = total.item()
        if step == 0 or (step + 1) % s.log_every == 0 or step + 1 == s.steps:
            log.append(row)
    sync(device)
    update_s = time.time() - t0

    out = Path(out)
    if adapter is not None:
        lora.save(adapter[0], out, name=adapter[1])
    else:
        save_checkpoint(policy, model, out)
    record = {
        'settings': {**dataclasses.asdict(s), 'objective': objective, 'pool': list(pool)},
        'model': str(model),
        'cache': str(Path(cache_path).resolve()),
        'output': 'adapter' if adapter is not None else 'checkpoint',
        'corrections': str(Path(s.corrections).resolve()) if s.corrections else None,
        'corrected_frames': len(c_rows),
        'corrected_per_episode': corrector.summary(c_tensors, pool, s.gate, steps)['per_episode'],
        'anchor_frames': len(d_rows) if use_anchor else 0,
        'trainable_parameters': sum(p.numel() for p in params),
        'update_s': round(update_s, 3),
        'total_s': round(time.time() - started, 3),
        'device': cache_mod.device_name(device),
        'losses': log,
    }
    (out / 'update.json').write_text(json.dumps(record, indent=2) + '\n')
    return record


def corrections(s, tensors, meta):
    """(tensors, pool) the corrected frames are gated from: the cache's own, or the on-policy cache
    `s.corrections`, which must come from the same base."""
    if s.corrections is None:
        if s.pool is None:
            raise ValueError('pool: the episodes to correct (all only with an on-policy corrections cache)')
        return tensors, tuple(s.pool)
    if s.trainable == 'all':
        raise ValueError('on-policy corrections have no dataset frames; trainable all needs them')
    c_tensors, c_meta = cache_mod.load(s.corrections)
    if c_meta.get('source') != 'on-policy':
        raise ValueError(f'{s.corrections}: not an on-policy cache (ttt onpolicy)')
    for k in ('model', 'horizon', 'n_action_steps'):
        if c_meta[k] != meta[k]:
            raise ValueError(f"{s.corrections}: {k} {c_meta[k]}, the cache's {meta[k]}")
    return c_tensors, tuple(c_meta['episodes'] if s.pool is None else s.pool)


def preference(vita, ref, pairs, batch, t, objective, beta):
    """L_pref: -log sigmoid(-beta [(l(z_h) - l_ref(z_h)) - (l(z_pi) - l_ref(z_pi))]), the same t."""
    z0, zh, ah = pairs.pair(vita, batch)
    _, zpi, api = pairs.pair(vita, batch, negative=True)
    with torch.no_grad():
        r0, _, _ = pairs.pair(ref, batch)
        ref_w = pair_error(ref, r0, zh, t, objective, ah)
        ref_l = pair_error(ref, r0, zpi, t, objective, api)
    won = pair_error(vita, z0, zh, t, objective, ah) - ref_w
    lost = pair_error(vita, z0, zpi, t, objective, api) - ref_l
    return -F.logsigmoid(-beta * (won - lost)).mean()


def save_checkpoint(policy, model, out):
    """A full checkpoint directory: the base's files (config, processors) with the updated weights."""
    from safetensors.torch import save_model

    out.mkdir(parents=True, exist_ok=True)
    for f in Path(model).iterdir():
        if f.is_file() and f.name != 'model.safetensors':
            shutil.copy2(f, out / f.name)
    save_model(policy, str(out / 'model.safetensors'))


def sync(device):
    if torch.device(device).type == 'cuda':
        torch.cuda.synchronize(device)
