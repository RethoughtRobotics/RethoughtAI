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
  full       the observation encoder, the flow net and the action decoder, every weight (S3a): the
             ResNet and the action encoder stay frozen, so it runs on cached latents as lora_all does
  flow       the whole flow net (S3b)
  flow_last  the flow net's output layer only (S3c)
  all        everything: ResNet, observation encoder, action encoder, flow net, decoder; it reads
             frames, since z0 and z_demo change with it
  lora_obs   LoRA on the observation encoder, the linear map from the observer's features (S8a)
  lora_dec   LoRA on the action decoder's linear layers (S8b)
  lora_all   LoRA on all three (S8c)
  residual   a residual head on the decoded chunk (S5): an MLP from the base's chunk and z0 to a
             correction added to the chunk, everything else frozen. It trains on the chunks, not the
             flow (objective `chunk`): the base's chunk plus the correction against the corrector's
             chunk, and on anchor frames the correction against zero, with the checkpoint's
             reconstruction loss (l1 or mse) over the normalised chunk
  latent_offset
             a latent offset (S6): an MLP from z0 to an offset added to z0, every weight frozen, trained
             through the frozen flow net with the flow error (or the vita objective)
Both heads have `hidden` units, by default as many as make their parameter count M0's adapter's at
`rank` (S5: the same parameter count as M0's).
LoRA sets save an adapter (inference.lora, loadable by the runtime's --adapter); full, flow and
flow_last a weights adapter, their tuned weights only (the runtime's --adapter too, alpha blending the
base's weights and these); residual and latent_offset a head adapter (the runtime's --adapter, alpha
scaling the head's output); all a full checkpoint directory (the runtime's --model) that is also a
weights adapter. Every adapter.json records its trainable set. Every output gets an update.json:
settings, corrected frames, wall time of the update and the device it ran on.

Corrections come from the cache's own `pool` episodes (the demonstration as the corrector), or with
`corrections` from an on-policy cache (ttt.onpolicy: the base's attempts, the planner as the corrector),
its `pool` attempts (all if None) gated the same way; the anchor is always the cache's.

The gated frames can be capped and the corrector degraded (plans S15 and S16), in this order, every
draw seeded by the update seed, on cached latents only (not trainable `all`, which reads frames):
  max_frames     a random `max_frames` of the gated frames (S15: M0's corrected frame count)
  drop           that fraction of them left out, the operator missing some (S16d: 0.3)
  late           s; each frame's target the corrector's chunk from the frame round(late * fps) later
                 of the same episode or attempt (S16a: 0.5, 12 frames at 25 fps; expert.npz has a row
                 per frame at the dataset's rate, the cache's `frame`), the gate still on the frame's
                 own chunk; frames with no plan that late (the attempt ended or the cup fell) are lost
  chunk_fraction the first round(chunk_fraction * horizon) steps of each target chunk kept and the rest
                 held at the last kept waypoint, marked padded, as training pads a chunk past an
                 episode's end (S16b: 0.5, the 16 executed steps of 32); z_h encodes the whole chunk,
                 so the flow loss cannot mask steps: the truncated chunk is re-encoded instead
  noise_cm       Gaussian noise of this standard deviation (cm) on each waypoint's x, y and z, every step
                 of the chunk on its own (S16c: 2); orientation and gripper unchanged
Truncated or noisy chunks go through the frozen action encoder again for their z_h, and the `vita`
objective reconstructs them; a late target is the later frame's cached chunk and latent. Applied to
S15's capped frames, each S16 variant degrades the frames S15 trains on.
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
    'full': None,
    'flow': None,
    'flow_last': None,
    'all': None,
    'residual': None,
    'latent_offset': None,
}
FLOW_ONLY = ('lora', 'flow', 'flow_last', 'latent_offset')


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
    max_frames: int | None = None
    drop: float = 0.0
    late: float = 0.0
    chunk_fraction: float = 1.0
    noise_cm: float = 0.0
    hidden: int | None = None
    log_every: int = 10

    def degraded(self):
        """Whether the corrector's chunks change (late, truncated or noisy), not only which frames."""
        return self.late > 0 or self.chunk_fraction < 1 or self.noise_cm > 0

    def resolved_objective(self):
        if self.trainable == 'residual':
            return 'chunk'
        if self.objective != 'auto':
            return self.objective
        return 'flow' if self.trainable in FLOW_ONLY else 'vita'


def lora_parameters(policy, rank):
    """The parameter count of M0's adapter (LoRA of `rank` on the flow net's targets) on `policy`."""
    root = lora.flow_net(policy)
    layers = [root.get_submodule(n) for n in lora.targets(root)]
    return sum(rank * (m.in_features + m.out_features) for m in layers)


def head_hidden(policy, kind, rank):
    """The hidden width that makes a `kind` head's parameter count closest to M0's adapter's at `rank`."""
    inputs, outputs = lora.head_size(policy, kind)
    return max(1, round((lora_parameters(policy, rank) - outputs) / (inputs + 1 + outputs)))


def prepare(policy, trainable, rank=16, alpha=1.0, hidden=None):
    """Make `trainable` of the policy trainable and everything else frozen; returns
    (parameters, adapter root module and its name, or None for a full checkpoint or a head)."""
    if trainable not in TRAINABLE:
        raise ValueError(f'trainable {trainable!r}: one of {", ".join(TRAINABLE)}')
    vita = policy.vita
    policy.requires_grad_(False)
    if trainable in lora.HEADS:
        head = lora.add_head(policy, trainable, hidden or head_hidden(policy, trainable, rank), alpha)
        return list(head.parameters()), None
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
    modules = {
        'full': (vita.obs_encoder, vita.flow_net, vita.action_decoder),
        'flow': (vita.flow_net,),
        'flow_last': (vita.flow_net.out_proj,),
        'all': (vita,),
    }[trainable]
    params = [p for m in modules for p in m.parameters() if p.dtype.is_floating_point]
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


def residual_error(vita, pairs, i, anchor=False):
    """Per-pair loss (B,) of the residual head on rows i: the base's chunk plus its correction against
    the corrector's chunk, or with `anchor` against the base's own (the correction zero)."""
    base = pairs.action[True][i]
    diff = vita.residual(base, vita.obs_encoder(pairs.feat[i])) + (0 if anchor else base - pairs.action[False][i])
    return (diff.abs() if vita.config.recon_loss_type == 'l1' else diff**2).mean((1, 2))


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
    if s.beta is not None and s.trainable == 'residual':
        raise ValueError('the preference loss is on the flow; the residual head trains on chunks')
    if s.trainable == 'residual' and s.objective != 'auto':
        raise ValueError(f'objective {s.objective}: the residual head trains on chunks (auto)')
    check(s)
    started = time.time()
    tensors, meta = cache_mod.load(cache_path)
    model = model or meta['model']
    lp = cache_mod.load_policy(model, device)
    policy, vita = lp.policy, lp.policy.vita
    ref = copy.deepcopy(vita).requires_grad_(False) if s.beta is not None else None
    torch.manual_seed(s.seed)  # LoRA's A initialisation
    params, adapter = prepare(policy, s.trainable, s.rank, s.alpha, s.hidden)
    policy.eval()  # dropout off, the inference crop

    steps = meta['n_action_steps']
    c_tensors, pool, c_meta = corrections(s, tensors, meta)
    c_rows = torch.nonzero(corrector.corrected(c_tensors, pool, s.gate, steps)).flatten()
    gated = len(c_rows)
    c_rows = selected(s, c_rows)
    c_tensors, c_rows, late_lost = degrade(lp, c_tensors, c_rows, s, fps_of(c_meta) if s.late > 0 else None)
    d_rows = torch.nonzero(corrector.in_episodes(tensors, s.anchor)).flatten()
    if len(c_rows) == 0:
        raise ValueError(f'no corrected frame in episodes {list(pool)} at gate {s.gate} cm ({gated} gated)')
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
        if objective == 'chunk':
            loss_c = residual_error(vita, corrected_set, bc).mean()
        else:
            z0, z1, a = corrected_set.pair(vita, bc)
            loss_c = pair_error(vita, z0, z1, t, objective, a).mean()
        total, row = (1 - s.lam) * loss_c, {'step': step + 1, 'corrected': loss_c.item()}
        if use_anchor:
            bd = next(batches_d)
            if objective == 'chunk':
                loss_d = residual_error(vita, anchor_set, bd, anchor=True).mean()
            else:
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
        lora.save(adapter[0], out, name=adapter[1], trainable=s.trainable)
        output = 'adapter'
    elif s.trainable in lora.HEADS:
        lora.save_head(policy, out, s.trainable)
        output = 'head'
    elif s.trainable == 'all':
        save_checkpoint(policy, model, out)
        lora.save_weights(policy, None, out, s.trainable, weights='model.safetensors')
        output = 'checkpoint'
    else:
        lora.save_weights(policy, [n for n, p in policy.named_parameters() if p.requires_grad], out, s.trainable)
        output = 'weights'

    record = {
        'settings': {
            **dataclasses.asdict(s),
            'objective': objective,
            'pool': list(pool),
            **({'hidden': lora.heads(vita)[0].fc1.out_features} if s.trainable in lora.HEADS else {}),
        },
        'model': str(model),
        'cache': str(Path(cache_path).resolve()),
        'output': output,
        'corrections': str(Path(s.corrections).resolve()) if s.corrections else None,
        'gated_frames': gated,
        'corrected_frames': len(c_rows),
        'late_lost_frames': late_lost,
        'corrected_per_episode': {int(e): int((c_tensors['episode'][c_rows] == e).sum()) for e in pool},
        'anchor_frames': len(d_rows) if use_anchor else 0,
        'trainable_parameters': sum(p.numel() for p in params),
        'update_s': round(update_s, 3),
        'total_s': round(time.time() - started, 3),
        'device': cache_mod.device_name(device),
        'losses': log,
    }
    (out / 'update.json').write_text(json.dumps(record, indent=2) + '\n')
    return record


def check(s):
    """The cap and the degradations in range, and on cached latents."""
    if s.max_frames is not None and s.max_frames < 1:
        raise ValueError(f'max_frames {s.max_frames}: at least 1')
    if not 0 <= s.drop < 1 or not 0 < s.chunk_fraction <= 1 or s.late < 0 or s.noise_cm < 0:
        raise ValueError(
            f'drop {s.drop} in [0, 1), chunk_fraction {s.chunk_fraction} in (0, 1], late {s.late} and '
            f'noise_cm {s.noise_cm} at least 0'
        )
    if s.degraded() and s.trainable == 'all':
        raise ValueError('degraded chunks are re-encoded from cached rows; not available with trainable all')


def stream(seed, k):
    """A generator for draw `k` of the update seed, apart from the update's own batches."""
    return torch.Generator().manual_seed(1000 * seed + k)


def selected(s, rows):
    """The gated `rows` capped at `s.max_frames`, then with the fraction `s.drop` left out, at random
    (seeded) and in their order."""
    if s.max_frames is not None and len(rows) > s.max_frames:
        rows = rows[torch.randperm(len(rows), generator=stream(s.seed, 1))[: s.max_frames]].sort().values
    if s.drop > 0:
        keep = len(rows) - round(s.drop * len(rows))
        rows = rows[torch.randperm(len(rows), generator=stream(s.seed, 2))[:keep]].sort().values
    return rows


def later(tensors, rows, shift):
    """Per row of `rows`, the row of its episode `shift` frames later, -1 where there is none."""
    where = {k: i for i, k in enumerate(zip(tensors['episode'].tolist(), tensors['frame'].tolist(), strict=True))}
    keys = zip(tensors['episode'][rows].tolist(), tensors['frame'][rows].tolist(), strict=True)
    return torch.tensor([where.get((e, f + shift), -1) for e, f in keys], dtype=torch.long)


def fps_of(meta):
    """A cache's frame rate: its own (on-policy caches and caches built since), else its dataset's."""
    if meta.get('fps'):
        return meta['fps']
    from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata

    return LeRobotDatasetMetadata(meta['repo_id'], revision=meta['revision']).fps


def degrade(lp, tensors, rows, s, fps):
    """The corrector's chunks at `rows` as plan S16 degrades them (see the module docstring):
    (tensors with chunk_demo, z_demo and pad replaced at the kept rows, the kept rows, the rows lost to
    lateness). The tensors are returned as they are when nothing is degraded."""
    if not s.degraded() or len(rows) == 0:
        return tensors, rows, 0
    out = {**tensors, **{k: tensors[k].clone() for k in ('chunk_demo', 'z_demo', 'pad')}}
    lost = 0
    if s.late > 0:
        src = later(tensors, rows, round(s.late * fps))
        lost = int((src < 0).sum())
        rows, src = rows[src >= 0], src[src >= 0]
        for k in ('chunk_demo', 'z_demo', 'pad'):
            out[k][rows] = tensors[k][src]
    if s.chunk_fraction < 1 or s.noise_cm > 0:
        chunk, pad = out['chunk_demo'][rows].clone(), out['pad'][rows].clone()
        horizon = chunk.shape[1]
        if s.chunk_fraction < 1:
            keep = max(1, round(s.chunk_fraction * horizon))
            chunk[:, keep:] = chunk[:, keep - 1 : keep]
            pad[:, keep:] = True
        if s.noise_cm > 0:
            noise = torch.randn(chunk[..., :3].shape, generator=stream(s.seed, 3))
            chunk[..., :3] += noise * s.noise_cm / 100
        vita = lp.policy.vita
        device = next(vita.parameters()).device
        with torch.no_grad():
            z = vita.action_encoder(normalised_actions(lp, chunk.to(device)))
        out['chunk_demo'][rows], out['pad'][rows] = chunk, pad
        out['z_demo'][rows] = z.cpu().to(out['z_demo'].dtype)
    return out, rows, lost


def corrections(s, tensors, meta):
    """(tensors, pool, meta) the corrected frames are gated from: the cache's own, or the on-policy
    cache `s.corrections`, which must come from the same base."""
    if s.corrections is None:
        if s.pool is None:
            raise ValueError('pool: the episodes to correct (all only with an on-policy corrections cache)')
        return tensors, tuple(s.pool), meta
    if s.trainable == 'all':
        raise ValueError('on-policy corrections have no dataset frames; trainable all needs them')
    c_tensors, c_meta = cache_mod.load(s.corrections)
    if c_meta.get('source') != 'on-policy':
        raise ValueError(f'{s.corrections}: not an on-policy cache (ttt onpolicy)')
    for k in ('model', 'horizon', 'n_action_steps'):
        if c_meta[k] != meta[k]:
            raise ValueError(f"{s.corrections}: {k} {c_meta[k]}, the cache's {meta[k]}")
    return c_tensors, tuple(c_meta['episodes'] if s.pool is None else s.pool), c_meta


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
