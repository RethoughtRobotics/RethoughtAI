"""Test-time training from reward (ttt.reward, plan S7) and the runtime's seeded exploration
(inference.explore), on test_ttt's small randomly initialised VITA and its cache."""

import json
import math

import numpy as np
import pytest
import torch
from click.testing import CliRunner
from test_ttt import DIM, Frames, built, model  # noqa: F401  (fixtures)

from inference import explore, lora
from ttt import cache, reward


def test_reward_is_success_plus_half_the_progress():
    rows = [
        {'success': True, 'phase': 'release'},
        {'success': False, 'phase': 'transfer'},
        {'success': False, 'phase': 'grasp'},
        {'success': False, 'phase': 'reach'},
        {'success': False, 'phase': None},
    ]
    assert [reward.reward(r) for r in rows] == [1.5, 0.375, 0.25, 0.125, 0.0]


def test_the_temperature_puts_the_weights_ln2_from_uniform():
    r = torch.tensor([0.0, 0.125, 0.25, 0.25, 0.375, 1.5, 0.0, 0.125])
    w, beta = reward.weights(r)
    assert 0 < beta < math.inf
    assert reward.kl_uniform(w) == pytest.approx(math.log(2), abs=1e-9)
    assert float(w.sum()) == pytest.approx(1)
    order = torch.argsort(r, stable=True)
    assert (w[order].diff() >= -1e-12).all()  # more reward, more weight
    assert reward.weights(r * 10)[0] == pytest.approx(w)  # the scale of the rewards does not matter
    # small spreads still end ln 2 from uniform
    assert reward.kl_uniform(reward.weights(torch.tensor([0.0, 0.0, 0.0, 0.125]))[0]) == pytest.approx(math.log(2))


def test_equal_rewards_weigh_uniformly_and_ties_cap_the_temperature():
    w, beta = reward.weights(torch.full((8,), 0.25))
    assert beta == 0 and torch.allclose(w, torch.full((8,), 1 / 8, dtype=w.dtype))
    # 5 of 8 at the best: at most ln(8 / 5) < ln 2 from uniform, the limit uniform over those 5
    r = torch.tensor([1.5, 1.5, 1.5, 1.5, 1.5, 0.0, 0.25, 0.0])
    w, beta = reward.weights(r)
    assert math.isinf(beta)
    assert w.tolist() == [0.2] * 5 + [0.0] * 3
    # exactly half at the best: ln 2 is the limit
    assert math.isinf(reward.temperature(torch.tensor([1.0, 1.0, 0.0, 0.0])))


def test_the_weighted_loss_weighs_each_attempt_by_its_weight():
    g = torch.Generator().manual_seed(0)
    attempt = torch.tensor([0] * 3 + [1] * 10 + [2] * 1 + [3] * 6)
    error = torch.rand(len(attempt), generator=g)
    w = torch.tensor([0.1, 0.2, 0.6, 0.1], dtype=torch.float64)
    rw = reward.row_weights(attempt, w)
    expected = sum(float(w[i]) * float(error[attempt == i].mean()) for i in range(4))
    assert float(reward.weighted_loss(error, rw)) == pytest.approx(expected, rel=1e-6)
    # uniformly drawn batches estimate it
    idx = torch.randint(len(attempt), (20000,), generator=g)
    assert float(reward.weighted_loss(error[idx], rw[idx])) == pytest.approx(expected, rel=0.03)
    # uniform weights: each attempt counts once, however many frames it has
    rw = reward.row_weights(attempt, torch.full((4,), 0.25, dtype=torch.float64))
    assert float(reward.weighted_loss(error, rw)) == pytest.approx(
        sum(float(error[attempt == i].mean()) for i in range(4)) / 4, rel=1e-6
    )


def test_the_pg_loss_is_ppo_clipped():
    old = torch.zeros(4)
    advantage = torch.tensor([1.0, -1.0, 0.5, 0.0])
    logp = torch.zeros(4, requires_grad=True)
    loss, clipped = reward.pg_loss(logp, old, advantage, 0.1)
    assert float(loss.detach()) == pytest.approx(-float(advantage.mean())) and float(clipped) == 0
    loss.backward()
    assert torch.allclose(logp.grad, -advantage / 4)  # up the log-probability of the better attempts

    # beyond the clip in the advantage's direction: no gradient; against it: still pulled back
    logp = torch.tensor([0.5, -0.5, -0.5, 0.5], requires_grad=True)
    loss, clipped = reward.pg_loss(logp, old, torch.tensor([1.0, -1.0, 1.0, -1.0]), 0.1)
    loss.backward()
    assert float(clipped) == 1
    assert logp.grad[0] == 0 and logp.grad[1] == 0
    assert logp.grad[2] < 0 and logp.grad[3] > 0
    surrogate = -(1.1 - 0.9 + math.exp(-0.5) - math.exp(0.5)) / 4
    assert float(loss.detach()) == pytest.approx(surrogate)


def test_step_logp_is_the_gaussian_density():
    x, mean = torch.randn(3, 5), torch.randn(3, 5)
    expected = torch.distributions.Normal(mean, 0.3).log_prob(x).sum(-1)
    assert torch.allclose(explore.step_logp(x, mean, 0.3), expected, atol=1e-5)


def test_the_offset_is_fixed_per_seed(model):  # noqa: F811
    from inference.policies import LeRobotPolicy

    assert torch.equal(explore.offset(DIM, 7), explore.offset(DIM, 7))
    assert not torch.equal(explore.offset(DIM, 7), explore.offset(DIM, 8))
    a = LeRobotPolicy(str(model), 'cpu', explore={'sigma': 0.2, 'seed': 7})
    b = LeRobotPolicy(str(model), 'cpu', explore={'sigma': 0.2, 'seed': 7})
    assert torch.equal(a.explorer.eps, explore.offset(DIM, 7)) and torch.equal(a.explorer.eps, b.explorer.eps)
    eps = a.explorer.eps.clone()
    item = Frames()[0]
    plan_of(a, item), plan_of(a, item)
    assert torch.equal(a.explorer.eps, eps)  # kept for every plan of the attempt


def plan_of(lp, item):
    batch = torch.utils.data.default_collate([item])
    with torch.no_grad():
        return lp.post(lp.policy.predict_action_chunk(cache.normalise(lp, batch)))[0]


def test_exploration_offsets_z0_and_records_every_plan(model, tmp_path):  # noqa: F811
    from inference.policies import LeRobotPolicy

    items = [Frames()[i] for i in (0, 5, 30)]
    base = LeRobotPolicy(str(model), 'cpu')
    zero = LeRobotPolicy(str(model), 'cpu', explore={'sigma': 0.0, 'seed': 1})
    assert torch.allclose(plan_of(zero, items[0]), plan_of(base, items[0]), atol=1e-6)

    rec = tmp_path / 'stack' / 'plans.npy'
    a = LeRobotPolicy(str(model), 'cpu', explore={'sigma': 0.5, 'seed': 3, 'record': rec})
    b = LeRobotPolicy(str(model), 'cpu', explore={'sigma': 0.5, 'seed': 3})
    c = LeRobotPolicy(str(model), 'cpu', explore={'sigma': 0.5, 'seed': 4})
    a.explorer.engaged = True
    plans = [plan_of(a, it) for it in items[:2]]
    a.explorer.engaged = False
    plans.append(plan_of(a, items[2]))
    assert torch.allclose(plans[0], plan_of(b, items[0]))
    assert not torch.allclose(plans[0], plan_of(c, items[0]))
    assert not torch.allclose(plans[0], plan_of(base, items[0]), atol=1e-4)

    r = explore.records(rec)
    assert r['trajectory'].shape == (3, 7, DIM) and r['logp'].shape == (3, 0)
    assert r['engaged'].tolist() == [1, 1, 0]
    vita = base.policy.vita
    z0 = torch.from_numpy(r['z0'])
    with torch.no_grad():
        assert torch.allclose(vita.obs_encoder(torch.from_numpy(r['feat'])), z0, atol=1e-5)
        start = z0 + 0.5 * explore.offset(DIM, 3)
        assert torch.allclose(torch.from_numpy(r['trajectory'][:, 0]), start, atol=1e-6)
        end = vita.flow_matcher.sample(vita.flow_net, start=start)
    assert torch.allclose(torch.from_numpy(r['trajectory'][:, -1]), end, atol=1e-5)
    assert explore.header(rec) == {
        'sigma': 0.5,
        'seed': 3,
        'noise': 0.0,
        'steps': 6,
        'latent_dim': DIM,
        'fields': list(explore.FIELDS),
    }
    # a plan cut off while it was written is left out
    with open(rec, 'ab') as f:
        np.save(f, np.zeros(3))
    assert len(explore.records(rec)['z0']) == 3


def test_the_stochastic_flow_records_each_steps_logp(model, tmp_path):  # noqa: F811
    from inference.policies import LeRobotPolicy

    rec = tmp_path / 'plans.npy'
    lp = LeRobotPolicy(str(model), 'cpu', explore={'sigma': 0.1, 'seed': 2, 'noise': 0.05, 'record': rec})
    item = Frames()[3]
    first, second = plan_of(lp, item), plan_of(lp, item)
    assert not torch.allclose(first, second)  # fresh noise every plan, the offset kept
    r = explore.records(rec)
    assert r['logp'].shape == (2, 6) and r['engaged'].tolist() == [-1, -1]
    vita = lp.policy.vita
    x = torch.from_numpy(r['trajectory'])
    with torch.no_grad():
        for k in range(6):
            mean = explore.step_mean(vita, x[:, k], torch.full((2,), k), 6)
            logp = explore.step_logp(x[:, k + 1], mean, 0.05)
            assert torch.allclose(logp, torch.from_numpy(r['logp'][:, k]), atol=1e-3)


ROWS = [  # an evaluator row per attempt: the reward from success and phase
    {'success': True, 'phase': 'release'},
    {'success': False, 'phase': 'grasp'},
    {'success': False, 'phase': None},
    {'success': False, 'phase': 'transfer'},
]


def evaluator_run(model, built, path, sigma=0.3, noise=0.0, rows=ROWS, init=None):  # noqa: F811
    """An evaluator run as the sim evaluator leaves it with exploration: a stack per attempt, its row and
    its record, the plans the explorer makes over cached frames of episodes 21 and 25 (engaged) and 0 (not)."""
    tensors, _ = cache.load(built)
    lp = cache.load_policy(model)
    if init:
        lora.load_policy(lp.policy, init)
    vita = lp.policy.vita
    for i, row in enumerate(rows):
        stack = path / f'stack_{i}'
        stack.mkdir(parents=True)
        seed = 500 + i
        (stack / 'results.jsonl').write_text(json.dumps({'seed': seed, **row}) + '\n')
        ex = explore.Explorer(vita, sigma, seed, noise, record=stack / reward.RECORD)
        for e, engaged in ((21 + 4 * (i % 2), True), (0, False)):
            feat = tensors['feat'][tensors['episode'] == e].float()
            ex.engaged = engaged
            with torch.no_grad():
                z0 = vita.obs_encoder(feat)
                trajectory, logp = explore.flow(vita, z0 + ex.sigma * ex.eps, noise, ex.generator)
            ex.write(feat, z0, trajectory, logp)
        ex.file.close()
    return path


def settings(**kw):
    return reward.Settings(**{'steps': 30, 'batch': 16, 'lr': 1e-3, 'anchor': (0, 1), **kw})


def test_attempts_of_the_evaluator_stacks(model, built, tmp_path):  # noqa: F811
    run = evaluator_run(model, built, tmp_path / 'r1')
    attempts = reward.attempts_of(run)
    assert [a.seed for a in attempts] == [500, 501, 502, 503]
    plans = attempts[0].plans()
    assert len(plans['feat']) == 10 and (plans['engaged'] == 1).all()  # episode 21's frames, not 0's
    (run / 'stack_0' / 'results.jsonl').write_text('{"seed": 500}\n{"seed": 504}\n')
    with pytest.raises(ValueError, match='a runtime of its own'):
        reward.attempts_of(run)


def test_reward_weighted_regression_follows_the_better_attempts(model, built, tmp_path):  # noqa: F811
    run = evaluator_run(model, built, tmp_path / 'r1')
    record = reward.run(model, built, [run], tmp_path / 'a', settings(steps=80, lr=3e-3))
    w = [a['weight'] for a in record['attempts']]
    assert record['kl'] == pytest.approx(math.log(2)) and 0 < record['beta'] < math.inf
    assert w[0] == max(w) and w[2] == min(w)
    assert [a['frames'] for a in record['attempts']] == [10, 10, 10, 10] and record['frames'] == 40
    assert {'weighted', 'anchor', 'total'} <= set(record['losses'][-1])
    assert record['init'] is None and record['trainable_parameters'] == 4 * 16 * (DIM + 4 * DIM) * 2
    assert json.loads((tmp_path / 'a' / lora.META).read_text())['root'] == lora.FLOW_NET

    # the adapted policy plans from z0 closer to the best attempt's flowed latents than the base did
    best = reward.attempts_of(run)[0].plans()
    feat, z1 = best['feat'], best['trajectory'][:, -1]
    errors = {}
    for name, adapter in (('base', None), ('adapted', tmp_path / 'a')):
        lp = cache.load_policy(model)
        if adapter:
            lora.load_policy(lp.policy, adapter)
        vita = lp.policy.vita
        with torch.no_grad():
            end = vita.flow_matcher.sample(vita.flow_net, start=vita.obs_encoder(feat))
        errors[name] = float(((end - z1) ** 2).mean())
    assert errors['adapted'] < errors['base']


def test_the_next_round_continues_the_adapter(model, built, tmp_path):  # noqa: F811
    run = evaluator_run(model, built, tmp_path / 'r1')
    reward.run(model, built, [run], tmp_path / 'a1', settings())
    run2 = evaluator_run(model, built, tmp_path / 'r2', init=tmp_path / 'a1')
    zero = reward.run(model, built, [run2], tmp_path / 'a2', settings(steps=0), init=tmp_path / 'a1')
    from safetensors.torch import load_file

    assert zero['init'] == str((tmp_path / 'a1').resolve())
    one, two = load_file(str(tmp_path / 'a1' / lora.WEIGHTS)), load_file(str(tmp_path / 'a2' / lora.WEIGHTS))
    assert all(torch.equal(one[k], two[k]) for k in one)


def test_the_policy_gradient_starts_on_policy(model, built, tmp_path):  # noqa: F811
    run = evaluator_run(model, built, tmp_path / 'r1', noise=0.05)
    record = reward.run(model, built, [run], tmp_path / 'a', settings(method='pg', steps=20, lr=1e-3))
    assert record['logp_gap'] < 1e-2  # the recomputed old log-probabilities are the recorded ones
    assert record['losses'][0]['clipped'] == 0  # ratio 1 at the first step
    assert record['samples'] == 40 * 6 and record['flow_steps'] == 6
    assert (tmp_path / 'a' / lora.WEIGHTS).exists()

    with pytest.raises(ValueError, match='stochastic flow'):
        reward.run(model, built, [evaluator_run(model, built, tmp_path / 'r0')], tmp_path / 'b', settings(method='pg'))

    flat = [{'success': False, 'phase': 'grasp'}] * 3
    same = evaluator_run(model, built, tmp_path / 'r2', noise=0.05, rows=flat)
    record = reward.run(model, built, [same], tmp_path / 'c', settings(method='pg'))
    assert record['skipped'] and record['losses'] == []


def test_a_ppo_step_trains_only_the_adapter(model, built):  # noqa: F811
    from ttt import update

    lp = cache.load_policy(model)
    vita = lp.policy.vita
    params, _ = update.prepare(lp.policy, 'lora', 4)
    feat = cache.load(built)[0]['feat'][:8].float()
    with torch.no_grad():
        trajectory, old = explore.flow(vita, vita.obs_encoder(feat), 0.05, torch.Generator().manual_seed(0))
    k = torch.arange(6).repeat(8)
    frame = torch.arange(8).repeat_interleave(6)
    mean = explore.step_mean(vita, trajectory[frame, k], k, 6)
    logp = explore.step_logp(trajectory[frame, k + 1], mean, 0.05)
    assert torch.allclose(logp.detach(), old.flatten(), atol=1e-3)
    loss, _ = reward.pg_loss(logp, old.flatten(), torch.linspace(-1, 1, 8)[frame], 0.1)
    assert torch.isfinite(loss)
    loss.backward()
    trainable = {id(p) for p in params}
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in params)
    assert all(p.grad is None for p in lp.policy.parameters() if id(p) not in trainable)


def test_reward_cli(model, built, tmp_path):  # noqa: F811
    from ttt.cli import main

    run = evaluator_run(model, built, tmp_path / 'r1')
    args = ['reward', '--cache', str(built), '--out', str(tmp_path / 'a'), '--anchor', '0-1', '--steps', '5']
    result = CliRunner().invoke(main, [*args, '--batch', '8', '--device', 'cpu', str(run)])
    assert result.exit_code == 0, result.output
    summary = next(json.loads(x) for x in result.output.splitlines() if x.startswith('{'))
    assert summary['method'] == 'rwr' and summary['frames'] == 40
    assert json.loads((tmp_path / 'a' / 'update.json').read_text())['settings']['steps'] == 5
