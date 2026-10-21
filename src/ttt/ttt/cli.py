import json

import click

from . import BASE, M0_POOL, TEST


def episodes(text):
    """'21,25,29-33' to [21, 25, 29, 30, 31, 32, 33]."""
    out = []
    for part in str(text).split(','):
        lo, _, hi = part.strip().partition('-')
        out += list(range(int(lo), int(hi or lo) + 1))
    return out


def gate(text):
    return None if text == 'all' else float(text)


def pool_of(text, meta):
    """--pool: episodes, or all of the cache's."""
    return list(meta['episodes']) if text == 'all' else episodes(text)


def default_device():
    import torch

    return 'cuda' if torch.cuda.is_available() else 'cpu'


def span(r):
    return f'{r.start}-{r.stop - 1}'


@click.group()
def main():
    """Test-time training of a frozen VITA over cached latents."""


@main.command()
@click.option('--model', required=True, help='Base checkpoint directory (pretrained_model).')
@click.option('--repo', default='andnetdeboer/BlueCupRedContainer_sim', show_default=True)
@click.option('--revision', required=True, help='Dataset commit.')
@click.option('--out', required=True, help='Cache directory.')
@click.option('--episodes', 'eps', default=None, help='e.g. 0-59 (default: all).')
@click.option('--device', default=None)
@click.option('--batch-size', default=32, show_default=True)
@click.option('--workers', default=4, show_default=True)
def cache(model, repo, revision, out, eps, device, batch_size, workers):
    """Run the frozen base over every frame once and store its latents and chunks."""
    from . import cache as cache_mod

    path = cache_mod.build(
        model, repo, revision, out, episodes(eps) if eps else None, device or default_device(), batch_size, workers
    )
    click.echo(path.with_name('cache.json').read_text())


@main.command()
@click.option('--model', required=True, help='Base checkpoint directory (pretrained_model).')
@click.option('--out', required=True, help='Cache directory.')
@click.option('--device', default=None)
@click.option('--batch-size', default=32, show_default=True)
@click.option('--lead', type=int, default=None, help="Steps the labels lead the TCP (default: the checkpoint's).")
@click.option('--arm', default='left', show_default=True)
@click.option('--rows', multiple=True, help="The evaluator's attempt rows (its --out): their bags, after BAGS.")
@click.option('--failed', is_flag=True, help='With --rows, only the failed attempts.')
@click.option('--attempts', type=int, default=None, help='With --rows, the first N attempts by seed.')
@click.argument('bags', nargs=-1)
def onpolicy(model, out, device, batch_size, lead, arm, rows, failed, attempts, bags):
    """Cache the base over evaluator bags (with their expert.npz), the planner's chunks as the targets."""
    from . import onpolicy as onpolicy_mod

    bags = list(bags) + (onpolicy_mod.bags_of(rows, failed, attempts) if rows else [])
    if not bags:
        raise click.UsageError('no bag: give BAGS or --rows')
    path = onpolicy_mod.build(model, bags, out, device or default_device(), batch_size, lead, arm)
    click.echo(path.with_name('cache.json').read_text())


@main.command()
@click.option('--cache', 'cache_path', required=True, help='On-policy cache (ttt onpolicy).')
@click.option('--root', required=True, help='Dataset directory to write.')
@click.option('--repo-id', required=True)
@click.option('--task', required=True, help='Language instruction stored with every frame.')
@click.option('--pool', default='all', show_default=True, help='Seeds, e.g. 1000-1019, or all.')
@click.option('--gate', 'gate_', default='3', show_default=True, help='cm, or all.')
def export(cache_path, root, repo_id, task, pool, gate_):
    """The corrected frames of an on-policy cache as a LeRobot dataset (plan S4)."""
    from . import cache as cache_mod
    from . import onpolicy as onpolicy_mod

    _, meta = cache_mod.load(cache_path)
    n = onpolicy_mod.export(cache_path, root, repo_id, task, gate(gate_), pool_of(pool, meta))
    click.echo(f'{n} episodes in {root}')


@main.command()
@click.option('--cache', 'cache_path', required=True)
@click.option('--pool', default=None, help='Episodes, or all (default: 21,25,29,33; all for an on-policy cache).')
@click.option('--gate', 'gate_', default='3', show_default=True, help='cm, or all.')
def corrections(cache_path, pool, gate_):
    """Count the frames the corrector takes over (the cost column)."""
    from . import cache as cache_mod
    from . import corrector

    tensors, meta = cache_mod.load(cache_path)
    if pool is None:
        pool = 'all' if meta.get('source') == 'on-policy' else ','.join(map(str, M0_POOL))
    summary = corrector.summary(tensors, pool_of(pool, meta), gate(gate_), meta['n_action_steps'])
    click.echo(json.dumps(summary, indent=2))


@main.command()
@click.option('--cache', 'cache_path', required=True)
@click.option('--out', required=True, help='Adapter (LoRA or weights) or checkpoint directory to write.')
@click.option('--model', default=None, help="Base checkpoint (default: the cache's).")
@click.option(
    '--trainable',
    default='lora',
    show_default=True,
    type=click.Choice([
        'lora',
        'lora_obs',
        'lora_dec',
        'lora_all',
        'full',
        'flow',
        'flow_last',
        'all',
        'residual',
        'latent_offset',
    ]),
)
@click.option(
    '--pool', default=None, help='Episodes to correct, or all (default: 21,25,29,33; all with --corrections).'
)
@click.option('--gate', 'gate_', default='3', show_default=True, help='cm, or all.')
@click.option('--corrections', 'corrections_', default=None, help='On-policy cache (ttt onpolicy) of the corrections.')
@click.option('--anchor', default=span(BASE), show_default=True)
@click.option('--lam', default=0.5, show_default=True, help='Anchor weight lambda.')
@click.option('--steps', default=200, show_default=True)
@click.option('--batch', default=256, show_default=True)
@click.option('--lr', default=1e-4, show_default=True)
@click.option('--seed', default=0, show_default=True)
@click.option('--max-frames', type=int, default=None, help='At most this many corrected frames, by --seed (S15).')
@click.option('--drop', default=0.0, show_default=True, help='Fraction of the corrected frames left out (S16d).')
@click.option('--late', default=0.0, show_default=True, help="s; the corrector's chunk this much later (S16a).")
@click.option('--chunk-fraction', default=1.0, show_default=True, help='Leading part of each chunk kept (S16b).')
@click.option('--noise-cm', default=0.0, show_default=True, help='Gaussian noise per waypoint, cm (S16c: 2).')
@click.option('--rank', default=16, show_default=True)
@click.option(
    '--hidden', type=int, default=None, help="Head width (residual, latent_offset; default: M0's parameters at --rank)."
)
@click.option('--alpha', default=1.0, show_default=True)
@click.option('--beta', default=None, type=float, help='Adds the preference loss with this beta.')
@click.option('--objective', default='auto', show_default=True, type=click.Choice(['auto', 'flow', 'vita']))
@click.option('--device', default=None)
@click.option('--workers', default=4, show_default=True, help='Frame loaders (trainable all).')
def update(
    cache_path,
    out,
    model,
    trainable,
    pool,
    gate_,
    corrections_,
    anchor,
    lam,
    steps,
    batch,
    lr,
    seed,
    max_frames,
    drop,
    late,
    chunk_fraction,
    noise_cm,
    rank,
    hidden,
    alpha,
    beta,
    objective,
    device,
    workers,
):
    """Update the base on corrected frames, with the anchor; save an adapter or a checkpoint."""
    from . import update as update_mod

    if pool is None:
        pool = 'all' if corrections_ else ','.join(map(str, M0_POOL))
    s = update_mod.Settings(
        trainable=trainable,
        rank=rank,
        alpha=alpha,
        lam=lam,
        steps=steps,
        batch=batch,
        lr=lr,
        seed=seed,
        beta=beta,
        objective=objective,
        gate=gate(gate_),
        pool=None if pool == 'all' else tuple(episodes(pool)),
        anchor=tuple(episodes(anchor)),
        corrections=corrections_,
        max_frames=max_frames,
        drop=drop,
        late=late,
        chunk_fraction=chunk_fraction,
        noise_cm=noise_cm,
        hidden=hidden,
    )
    record = update_mod.run(model, cache_path, out, s, device or default_device(), workers=workers)
    click.echo(
        json.dumps({
            k: record[k]
            for k in (
                'output',
                'gated_frames',
                'corrected_frames',
                'anchor_frames',
                'trainable_parameters',
                'update_s',
                'device',
            )
        })
    )
    click.echo(json.dumps(record['losses'][-1]))


@main.command()
@click.option('--cache', 'cache_path', required=True, help="The dataset's cache, for the anchor.")
@click.option('--out', required=True, help='Adapter directory to write.')
@click.option(
    '--method',
    default='rwr',
    show_default=True,
    type=click.Choice(['rwr', 'pg']),
    help='Reward-weighted regression (S7a) or the clipped policy gradient (S7b).',
)
@click.option('--init', default=None, help='The LoRA adapter the attempts ran with, continued (default: a new one).')
@click.option('--model', default=None, help="Base checkpoint (default: the cache's).")
@click.option('--anchor', default=span(BASE), show_default=True)
@click.option('--lam', default=0.5, show_default=True, help='Anchor weight lambda.')
@click.option('--steps', default=200, show_default=True)
@click.option('--batch', default=256, show_default=True)
@click.option('--lr', default=1e-4, show_default=True)
@click.option('--rank', default=16, show_default=True)
@click.option('--seed', default=0, show_default=True)
@click.option('--clip', default=0.1, show_default=True, help="pg: the ratio's clip range.")
@click.option('--record', default='plans.npy', show_default=True, help="Each stack's record file (serve --record).")
@click.option('--device', default=None)
@click.argument('runs', nargs=-1, required=True)
def reward(cache_path, out, method, init, model, anchor, lam, steps, batch, lr, rank, seed, clip, record, device, runs):
    """Update a LoRA adapter on the reward of explored attempts (plan S7): RUNS, evaluator runs with a
    seed per stack and each stack's plans recorded."""
    from . import reward as reward_mod

    s = reward_mod.Settings(
        method=method,
        rank=rank,
        lam=lam,
        steps=steps,
        batch=batch,
        lr=lr,
        seed=seed,
        anchor=tuple(episodes(anchor)),
        clip=clip,
    )
    r = reward_mod.run(model, cache_path, runs, out, s, device or default_device(), init, record)
    keys = ('method', 'beta', 'kl', 'frames', 'samples', 'skipped', 'update_s', 'device')
    click.echo(json.dumps({k: r[k] for k in keys if k in r}))
    click.echo(json.dumps([{k: a[k] for k in ('seed', 'reward', 'weight', 'frames')} for a in r['attempts']]))
    if r['losses']:
        click.echo(json.dumps(r['losses'][-1]))


@main.command()
@click.option('--cache', 'cache_path', required=True)
@click.option('--out', required=True, help='Results json.')
@click.option('--adapter', default=None, help='Adapter directory: LoRA, weights or a head (ttt update).')
@click.option('--checkpoint', default=None, help='Updated full checkpoint directory.')
@click.option('--plans', default=None, help='Cache built with the updated checkpoint (an updated observer).')
@click.option('--model', default=None, help="Base checkpoint (default: the cache's).")
@click.option('--alphas', default='1', show_default=True, help='e.g. 0,0.25,0.5,0.75,1')
@click.option('--test', default=span(TEST), show_default=True)
@click.option('--forget', default=span(BASE), show_default=True)
@click.option('--device', default=None)
def evaluate(cache_path, out, adapter, checkpoint, plans, model, alphas, test, forget, device):
    """Open-loop path error, per phase and forgetting, for the base and the update at each alpha."""
    from . import evaluate as evaluate_mod

    if sum(x is not None for x in (adapter, checkpoint, plans)) > 1:
        raise click.UsageError('give one of --adapter, --checkpoint, --plans')
    results = evaluate_mod.run(
        cache_path,
        out,
        adapter,
        checkpoint,
        plans,
        model,
        [float(a) for a in alphas.split(',')],
        device or default_device(),
        episodes(test),
        episodes(forget),
    )
    line = {'base': results['base']['summary']['all']['median_mm']['value']}
    for a, r in results['alphas'].items():
        line[a] = {
            'test': r['summary']['all']['median_mm']['value'],
            'minus_base': r['summary']['all']['minus_base_mm']['value'],
            'forgetting': r['forgetting']['all']['minus_base_mm']['value'],
        }
    click.echo(json.dumps(line))


if __name__ == '__main__':
    main()
