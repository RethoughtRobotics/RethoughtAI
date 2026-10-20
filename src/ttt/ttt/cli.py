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
@click.option('--cache', 'cache_path', required=True)
@click.option('--pool', default=','.join(map(str, M0_POOL)), show_default=True)
@click.option('--gate', 'gate_', default='3', show_default=True, help='cm, or all.')
def corrections(cache_path, pool, gate_):
    """Count the frames the offline corrector takes over (the cost column)."""
    from . import cache as cache_mod
    from . import corrector

    tensors, meta = cache_mod.load(cache_path)
    click.echo(json.dumps(corrector.summary(tensors, episodes(pool), gate(gate_), meta['n_action_steps']), indent=2))


@main.command()
@click.option('--cache', 'cache_path', required=True)
@click.option('--out', required=True, help='Adapter or checkpoint directory to write.')
@click.option('--model', default=None, help="Base checkpoint (default: the cache's).")
@click.option(
    '--trainable',
    default='lora',
    show_default=True,
    type=click.Choice(['lora', 'flow', 'flow_last', 'all', 'lora_obs', 'lora_dec', 'lora_all']),
)
@click.option('--pool', default=','.join(map(str, M0_POOL)), show_default=True)
@click.option('--gate', 'gate_', default='3', show_default=True, help='cm, or all.')
@click.option('--anchor', default=span(BASE), show_default=True)
@click.option('--lam', default=0.5, show_default=True, help='Anchor weight lambda.')
@click.option('--steps', default=200, show_default=True)
@click.option('--batch', default=256, show_default=True)
@click.option('--lr', default=1e-4, show_default=True)
@click.option('--seed', default=0, show_default=True)
@click.option('--rank', default=16, show_default=True)
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
    anchor,
    lam,
    steps,
    batch,
    lr,
    seed,
    rank,
    alpha,
    beta,
    objective,
    device,
    workers,
):
    """Update the base on corrected frames, with the anchor; save an adapter or a checkpoint."""
    from . import update as update_mod

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
        pool=tuple(episodes(pool)),
        anchor=tuple(episodes(anchor)),
    )
    record = update_mod.run(model, cache_path, out, s, device or default_device(), workers=workers)
    click.echo(
        json.dumps({
            k: record[k]
            for k in ('output', 'corrected_frames', 'anchor_frames', 'trainable_parameters', 'update_s', 'device')
        })
    )
    click.echo(json.dumps(record['losses'][-1]))


@main.command()
@click.option('--cache', 'cache_path', required=True)
@click.option('--out', required=True, help='Results json.')
@click.option('--adapter', default=None, help='LoRA adapter directory.')
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
