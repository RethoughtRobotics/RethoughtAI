"""Every trainable set of ttt update, round trip into the runtime: the update's own module, saved as a
LoRA adapter (lora, lora_obs, lora_dec, lora_all) or a weights adapter (full, flow, flow_last; all, a
checkpoint that is one too), loaded by the runtime's --adapter at any alpha, plans the same chunk from a
raw observation (alpha 0 the base exactly, 1 the trained module), and the open-loop evaluation scores
the runtime's plans."""

import json

import numpy as np
import pytest
import torch
from test_ttt import CAMERAS, Frames, built, model, settings  # noqa: F401  (fixtures)

from inference import lora
from inference.cli import adapted
from inference.policies import LeRobotPolicy
from ttt import cache, evaluate, update

ADAPTERS = ('lora', 'lora_obs', 'lora_dec', 'lora_all')
WEIGHTS = ('full', 'flow', 'flow_last')
SITES = {
    'lora': {'flow_net'},
    'lora_obs': {'obs_encoder'},
    'lora_dec': {'action_decoder'},
    'lora_all': {'obs_encoder', 'flow_net', 'action_decoder'},
    'full': {'obs_encoder', 'flow_net', 'action_decoder'},
    'flow': {'flow_net'},
    'flow_last': {'flow_net.out_proj'},
}
TCP = (np.array([0.6, 0.2, 0.1]), np.array([0.0, 1.0, 0.0, 0.0]))


def observation(seed=0):
    """A raw observation as the policy node hands it: uint8 (H, W, 3) images and the 15-D state."""
    rng = np.random.default_rng(seed)
    obs = {k: rng.integers(0, 255, (h, w, 3), dtype=np.uint8) for k, (_, h, w) in CAMERAS.items()}
    return {**obs, 'observation.state': rng.normal(size=15).astype(np.float32)}


def trained(model, built, out, trainable, monkeypatch):  # noqa: F811
    """update.run with `trainable`, and the policy it trained, as it was when saved to `out`."""
    seen = []
    prepare = update.prepare

    def keep(policy, *args, **kw):
        seen.append(policy)
        return prepare(policy, *args, **kw)

    monkeypatch.setattr(update, 'prepare', keep)
    s = settings(trainable=trainable, steps=10)
    if trainable == 'all':
        update.run(model, built, out, s, frames=Frames(), workers=0)
    else:
        update.run(model, built, out, s)
    return seen[0]


def as_runtime(model, policy):  # noqa: F811
    """A runtime policy (the checkpoint's processors) running `policy`'s module."""
    lp = LeRobotPolicy(str(model), 'cpu')
    lp.policy = policy.eval()
    return lp


def changed(model, policy):  # noqa: F811
    """The state dict names `policy` changed from the base `model`'s."""
    base = cache.load_policy(model).policy.state_dict()
    return {k for k, v in policy.state_dict().items() if not torch.equal(v, base[k])}


@pytest.mark.parametrize('trainable', ADAPTERS + WEIGHTS + ('all',))
def test_the_runtime_plans_as_the_trained_module(model, built, tmp_path, trainable, monkeypatch):  # noqa: F811
    out = tmp_path / 'out'
    policy = trained(model, built, out, trainable, monkeypatch)
    meta = json.loads((out / lora.META).read_text())
    assert meta['trainable'] == trainable
    runtime = LeRobotPolicy(str(model), 'cpu', adapter=out, alpha=1.0)
    if trainable in ADAPTERS:
        assert 'kind' not in meta and not (out / 'model.safetensors').exists()
        logged = {part.split(' on ')[1] for part in adapted(runtime, out).split(': ')[1].split(', ')}
        assert logged == SITES[trainable]
    elif trainable in WEIGHTS:
        assert meta['kind'] == 'weights' and not (out / 'model.safetensors').exists()
        assert adapted(runtime, out) == f'{trainable}: tuned weights'
        held = set(lora.tuned(out))
        assert changed(model, policy) <= held
        assert held and all(any(k.startswith(f'vita.{s}.') for s in SITES[trainable]) for k in held)
    else:  # a checkpoint (the runtime's --model) and a weights adapter on the base alike
        assert meta == {'kind': 'weights', 'trainable': 'all', 'weights': 'model.safetensors'}
        assert any(k.startswith('vita.observer.') for k in changed(model, policy))
    obs = observation()
    plan = runtime(obs, tcp=TCP)
    assert np.array_equal(plan, as_runtime(model, policy)(obs, tcp=TCP))
    assert not np.allclose(plan, LeRobotPolicy(str(model), 'cpu')(obs, tcp=TCP))
    if trainable == 'all':
        assert np.array_equal(plan, LeRobotPolicy(str(out), 'cpu')(obs, tcp=TCP))


@pytest.mark.parametrize('trainable', ADAPTERS + WEIGHTS)
def test_alpha_blends_the_adapter_in_the_runtime(model, built, tmp_path, trainable, monkeypatch):  # noqa: F811
    out = tmp_path / 'out'
    policy = trained(model, built, out, trainable, monkeypatch)
    obs = observation(1)
    base_lp = LeRobotPolicy(str(model), 'cpu')
    base, base_state = base_lp(obs, tcp=TCP), base_lp.policy.state_dict()
    tuned = policy.state_dict()
    plans = {}
    for alpha in (0.0, 0.5, 1.0):
        runtime = LeRobotPolicy(str(model), 'cpu', adapter=out, alpha=alpha)
        plans[alpha] = runtime(obs, tcp=TCP)
        if trainable in ADAPTERS:
            deltas = {name: m.delta() for name, m in lora.adapters(policy).items()}
            layers = lora.adapters(runtime.policy)
            assert layers.keys() == deltas.keys()
            for name, m in layers.items():
                assert m.alpha == alpha
            lora.set_alpha(policy, alpha)
            assert np.array_equal(plans[alpha], as_runtime(model, policy)(obs, tcp=TCP))
            lora.set_alpha(policy, 1.0)
        else:
            state = runtime.policy.state_dict()
            for k in lora.tuned(out):
                assert torch.equal(state[k], torch.lerp(base_state[k], tuned[k], alpha))
    assert np.array_equal(plans[0.0], base)
    assert np.array_equal(plans[1.0], as_runtime(model, policy)(obs, tcp=TCP))
    assert not np.allclose(plans[0.5], base) and not np.allclose(plans[0.5], plans[1.0])


def flat(scores):
    """Per-episode scores as one list, None (a phase the episode lacks) as nan."""
    return [np.nan if v is None else v for e in sorted(scores) for _, v in sorted(scores[e].items())]


@pytest.mark.parametrize('trainable', ADAPTERS + WEIGHTS)
def test_open_loop_scores_the_runtimes_plans(model, built, tmp_path, trainable, monkeypatch):  # noqa: F811
    out = tmp_path / 'out'
    trained(model, built, out, trainable, monkeypatch)
    results = evaluate.run(built, tmp_path / 'r.json', adapter=out, alphas=(0.0, 0.5, 1.0), n_boot=50)
    tensors, meta = cache.load(built)
    rows = torch.nonzero(torch.isin(tensors['episode'], torch.tensor(results['test_episodes']))).flatten()
    phase = evaluate.phases(tensors)
    for alpha in (0.0, 0.5, 1.0):
        runtime = LeRobotPolicy(str(model), 'cpu', adapter=out, alpha=alpha)
        scores = evaluate.score(evaluate.decode(runtime, tensors, rows), tensors, rows, meta['n_action_steps'], phase)
        assert flat(scores) == pytest.approx(flat(results['alphas'][str(alpha)]['test']), abs=1e-4, nan_ok=True)
    assert results['alphas']['0.0']['test'] == results['base']['test']
    assert results['alphas']['0.5']['test'] != results['alphas']['1.0']['test']


def test_a_weights_adapter_on_another_policy_is_refused(model, tmp_path):  # noqa: F811
    lp = cache.load_policy(model)
    lora.save_weights(lp.policy, ['vita.flow_net.out_proj.weight'], tmp_path, 'flow_last')
    from safetensors.torch import save_file

    save_file({'vita.nowhere.weight': torch.zeros(2)}, str(tmp_path / lora.WEIGHTS))
    with pytest.raises(ValueError, match='not in the policy'):
        lora.load_policy(cache.load_policy(model).policy, tmp_path)


def test_the_cli_offers_every_trainable_set():
    from ttt.cli import update as update_cli

    option = next(p for p in update_cli.params if p.name == 'trainable')
    assert set(option.type.choices) == set(update.TRAINABLE)
