"""Open-loop evaluation over the cache: decode every frame of the test and base episodes with an
adapter (LoRA, weights or a head) or an updated checkpoint at any alpha, and score path error (mm) against the
demonstration overall and per cup phase, and forgetting (on the base episodes, adapted minus base),
paired over episodes with bootstrap intervals.

Phases come from the demonstration's gripper channel: reach until it first closes, transfer while
closed, release after it last opens.
"""

import json
from pathlib import Path

import torch

from . import BASE, TEST, corrector, stats
from . import cache as cache_mod

PHASES = ('reach', 'transfer', 'release')
CLOSED = 0.5  # gripper command above this is closed (the sim's: 0 open, about 0.87 closed)


@torch.no_grad()
def decode(lp, tensors, rows, batch=1024):
    """The unnormalised chunks the policy plans from the cached observer features of `rows`."""
    vita, device = lp.policy.vita, lp.device
    out = []
    for i in range(0, len(rows), batch):
        z0 = vita.obs_encoder(tensors['feat'][rows[i : i + batch]].float().to(device))
        out.append(lp.post(cache_mod.flow(vita, z0)[1]).float().cpu())
    return torch.cat(out) if out else torch.zeros(0)


def phases(tensors):
    """Per row: 0 reach, 1 transfer, 2 release, from the demonstration's gripper command."""
    closed = tensors['chunk_demo'][:, 0, 7] > CLOSED
    label = torch.zeros(len(closed), dtype=torch.long)
    for e in tensors['episode'].unique():
        rows = torch.nonzero(tensors['episode'] == e).flatten()
        rows = rows[tensors['frame'][rows].argsort()]
        on = torch.nonzero(closed[rows]).flatten()
        if len(on) == 0:
            continue
        label[rows[on[0] :]] = 1
        label[rows[on[-1] + 1 :]] = 2
    return label


def per_episode(error, tensors, rows, phase):
    """{episode: {'all': mean mm, phase: mean mm or None}} over `rows` (error indexed like rows)."""
    episodes = tensors['episode'][rows]
    out = {}
    for e in episodes.unique().tolist():
        mine = episodes == e
        row = {'all': error[mine].mean().item(), 'frames': int(mine.sum())}
        for k, name in enumerate(PHASES):
            sel = mine & (phase[rows] == k)
            row[name] = error[sel].mean().item() if sel.any() else None
        out[int(e)] = row
    return out


def score(chunks, tensors, rows, steps, phase):
    error = 1000 * corrector.path_error(chunks, tensors['chunk_demo'][rows], tensors['pad'][rows], steps)
    return per_episode(error, tensors, rows, phase)


def compare(adapted, base=None, n_boot=10000):
    """Median of the per-episode values and, given `base`, of the paired differences (adapted -
    base), overall and per phase, with bootstrap intervals over episodes."""
    out = {}
    for k in ('all', *PHASES):
        a = {e: r[k] for e, r in adapted.items()}
        known = [v for v in a.values() if stats.known(v)]
        out[k] = {'median_mm': stats.bootstrap(known, n=n_boot) if known else None}
        if base is not None:
            b = {e: r[k] for e, r in base.items()}
            out[k]['minus_base_mm'] = stats.paired(a, b, n=n_boot) if known else None
    return out


class Blend:
    """A policy whose weights are the base's interpolated toward an update's by alpha (WiSE-FT), as the
    runtime blends a weights adapter (lora.blend); alpha 0 is the base exactly, 1 the update. `updated`
    may hold only the tensors it changed."""

    def __init__(self, lp, updated):
        from inference import lora

        self.lp, self.blend = lp, lora.blend
        self.base = {k: v.clone() for k, v in lp.policy.state_dict().items()}
        self.updated = {k: updated[k].to(v.device) for k, v in self.base.items() if k in updated}

    def __call__(self, alpha):
        state = {k: self.blend(v, self.updated[k], alpha) if k in self.updated else v for k, v in self.base.items()}
        self.lp.policy.load_state_dict(state)


def run(
    cache_path,
    out,
    adapter=None,
    checkpoint=None,
    plans=None,
    model=None,
    alphas=(1.0,),
    device='cpu',
    test=TEST,
    forget=BASE,
    n_boot=10000,
):
    """Score the base and the adapter (a lora.save, save_weights or save_head directory) or checkpoint (an
    updated full checkpoint) at each alpha, or the plans of another cache's model over the same frames (alpha 1:
    an update that changes the observer, whose cached features no longer hold); write the results
    json to `out` and return it."""
    from inference import lora

    tensors, meta = cache_mod.load(cache_path)
    model = model or meta['model']
    steps = meta['n_action_steps']
    lp = cache_mod.load_policy(model, device)
    weights = adapter is not None and lora.meta(adapter).get('kind') == 'weights'
    if adapter is not None and not weights:
        lora.load_policy(lp.policy, adapter, 1.0)

        def set_alpha(alpha):
            lora.set_alpha(lp.policy, alpha)

    elif weights or checkpoint is not None:
        from safetensors.torch import load_file

        updated = lora.tuned(adapter) if weights else load_file(str(Path(checkpoint) / 'model.safetensors'))
        changed = [
            k
            for k, v in lp.policy.state_dict().items()
            if k.startswith('vita.observer.') and k in updated and not torch.equal(updated[k].to(v.device), v)
        ]
        if changed:
            raise ValueError(
                f'{adapter if weights else checkpoint} changes the observer ({changed[0]}, ...): the cached features '
                'do not hold for it; build a cache with it as the model and score its plans'
            )
        set_alpha = Blend(lp, updated)
    else:

        def set_alpha(alpha):
            pass

        alphas = (1.0,) if plans is not None else ()
    if plans is not None:
        other, _ = cache_mod.load(plans)
        for k in ('episode', 'frame'):
            if not torch.equal(other[k], tensors[k]):
                raise ValueError(f'{plans} holds other frames than {cache_path}')
    phase = phases(tensors)
    sets = {'test': list(test), 'forget': list(forget)}
    rows = {k: torch.nonzero(corrector.in_episodes(tensors, v)).flatten() for k, v in sets.items()}

    def scores(alpha):
        if plans is not None and alpha:
            return {k: score(other['chunk_pi'][r], tensors, r, steps, phase) for k, r in rows.items()}
        set_alpha(alpha)
        return {k: score(decode(lp, tensors, r), tensors, r, steps, phase) for k, r in rows.items()}

    base = scores(0.0)
    results = {
        'cache': str(Path(cache_path).resolve()),
        'model': str(model),
        'adapter': str(Path(adapter).resolve()) if adapter else None,
        'checkpoint': str(Path(checkpoint).resolve()) if checkpoint else None,
        'plans': str(Path(plans).resolve()) if plans else None,
        'test_episodes': sets['test'],
        'forget_episodes': sets['forget'],
        'executed_steps': steps,
        'base': {
            'test': base['test'],
            'forget': base['forget'],
            'summary': compare(base['test'], n_boot=n_boot),
            'forget_summary': compare(base['forget'], n_boot=n_boot),
        },
        'alphas': {},
    }
    for alpha in alphas:
        s = scores(float(alpha))
        results['alphas'][str(float(alpha))] = {
            'test': s['test'],
            'forget': s['forget'],
            'summary': compare(s['test'], base['test'], n_boot),
            'forgetting': compare(s['forget'], base['forget'], n_boot),
        }
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2) + '\n')
    return results
