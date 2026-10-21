"""Test-time training of a Multi-Task DiT (ttt.dit) on a small randomly initialised denoiser: LoRA only on
its attention and MLP layers, the loss and its gradients, adapters through save and load and alpha, and
the update and open-loop evaluation over a DiT cache."""

import json
import types

import pytest
import torch
from torch import nn

from inference import lora
from inference.policies import check_adapter
from ttt import cache, dit, evaluate, update

H, COND, LAYERS = 16, 24, 2
EPISODES = {0: 12, 1: 12, 21: 10, 25: 10, 38: 8, 39: 8}


def config():
    from lerobot.configs.types import FeatureType, PolicyFeature
    from lerobot.policies.multi_task_dit.configuration_multi_task_dit import MultiTaskDiTConfig

    return MultiTaskDiTConfig(
        input_features={'observation.state': PolicyFeature(FeatureType.STATE, (15,))},
        output_features={'action': PolicyFeature(FeatureType.ACTION, (8,))},
        horizon=H,
        n_action_steps=8,
        hidden_dim=32,
        num_layers=LAYERS,
        num_heads=4,
        timestep_embed_dim=16,
        num_train_timesteps=20,
        push_to_hub=False,
        device='cpu',
    )


class TinyDiT(nn.Module):
    """The DiT's denoiser and objective; its observation encoder (CLIP) stays out, as ttt caches its output."""

    def __init__(self, seed=0):
        from lerobot.policies.multi_task_dit.modeling_multi_task_dit import DiffusionObjective, DiffusionTransformer

        super().__init__()
        torch.manual_seed(seed)
        self.config = config()
        self.noise_predictor = DiffusionTransformer(self.config, COND)
        self.objective = DiffusionObjective(self.config, 8, H)
        with torch.no_grad():  # AdaLN-Zero starts every block shut; a trained one is open
            for block in self.noise_predictor.transformer_blocks:
                block.adaLN_modulation[-1].weight.normal_(std=0.05)
        self.eval()


def runtime(policy):
    """What ttt reads of the runtime's LeRobotPolicy, with identity normalisation."""
    return types.SimpleNamespace(policy=policy, device=torch.device('cpu'), normalizer=lambda t: t, post=lambda x: x)


def toy(n=32, seed=1):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(n, COND, generator=g), torch.rand(n, H, 8, generator=g) * 2 - 1


def test_lora_sits_only_on_the_denoisers_attention_and_mlp():
    policy = TinyDiT()
    params, (root, name) = dit.prepare(policy, rank=4)
    assert name == 'noise_predictor' and root is policy.noise_predictor
    assert sorted(lora.adapters(root)) == sorted(
        f'transformer_blocks.{i}.{p}'
        for i in range(LAYERS)
        for p in ('attn.qkv_proj', 'attn.out_proj', 'mlp.0', 'mlp.2')
    )
    trainable = {n for n, p in policy.named_parameters() if p.requires_grad}
    assert trainable == {f'noise_predictor.{n}.{k}' for n in lora.adapters(root) for k in 'AB'}
    layers = lora.adapters(root).values()
    assert sum(p.numel() for p in params) == sum(4 * (m.base.in_features + m.base.out_features) for m in layers)


def test_the_loss_is_finite_its_gradients_reach_only_lora_and_it_falls():
    policy = TinyDiT()
    params, _ = dit.prepare(policy, rank=4)
    cond, action = toy()
    opt = torch.optim.AdamW(params, lr=1e-2)
    losses = []
    for _ in range(60):
        torch.manual_seed(3)  # the same noise and timesteps every step
        loss = dit.loss(policy, cond, action)
        assert torch.isfinite(loss)
        opt.zero_grad()
        loss.backward()
        assert all(p.grad is None for n, p in policy.named_parameters() if not n.endswith(('.A', '.B')))
        assert any(p.grad.abs().sum() > 0 for p in params)
        opt.step()
        losses.append(loss.item())
    assert losses[-1] < 0.8 * losses[0]


def trained(seed=0):
    policy = TinyDiT(seed)
    dit.prepare(policy, rank=4)
    g = torch.Generator().manual_seed(5)
    with torch.no_grad():
        for m in lora.adapters(policy.noise_predictor).values():
            m.B.copy_(0.1 * torch.randn(m.B.shape, generator=g))
    return policy


def test_an_adapter_survives_save_and_load_and_alpha_0_is_the_base(tmp_path):
    policy = trained()
    lora.save(policy.noise_predictor, tmp_path / 'a', name=lora.DIT_ROOT, trainable='lora')
    cond, _ = toy(4)
    adapted = dit.plan(policy, cond)

    loaded = TinyDiT()
    base = dit.plan(loaded, cond)
    assert not torch.allclose(base, adapted)
    lora.load_policy(loaded, tmp_path / 'a', alpha=1.0)
    assert torch.allclose(dit.plan(loaded, cond), adapted, atol=1e-6)
    lora.set_alpha(loaded, 0.0)
    assert torch.equal(dit.plan(loaded, cond), base)
    check_adapter('multi_task_dit', tmp_path / 'a')


def test_the_runtime_takes_only_lora_on_the_dit(tmp_path):
    lora.save_weights(TinyDiT(), ['noise_predictor.input_proj.weight'], tmp_path / 'w', 'flow')
    with pytest.raises(ValueError, match='weights adapter is for VITA'):
        check_adapter('multi_task_dit', tmp_path / 'w')
    check_adapter('vita', tmp_path / 'w')


def test_plans_are_seeded():
    policy, (cond, _) = TinyDiT(), toy(4)
    assert torch.equal(dit.plan(policy, cond), dit.plan(policy, cond))


@pytest.fixture
def built(tmp_path):
    """A DiT cache: a straight demonstrated path per episode, the base 5 cm off on the pool episodes."""
    g = torch.Generator().manual_seed(7)
    rows = []
    for e, n in EPISODES.items():
        for f in range(n):
            steps = torch.arange(f - 1, f - 1 + H).clamp(0, n - 1).float()
            target = torch.zeros(H, 8)
            target[:, 0] = 0.01 * steps
            target[:, 6] = 1.0
            target[:, 7] = (steps > n / 2).float() * 0.87
            pad = (torch.arange(f - 1, f - 1 + H) < 0) | (torch.arange(f - 1, f - 1 + H) >= n)
            rows.append((e, f, target, pad))
    target = torch.stack([r[2] for r in rows])
    pad = torch.stack([r[3] for r in rows])
    episode = torch.tensor([r[0] for r in rows])
    off = torch.isin(episode, torch.tensor([21, 25])).float()[:, None, None] * torch.tensor([0.05] + [0] * 7)
    tensors = cache.typed({
        'cond': torch.randn(len(rows), COND, generator=g),
        'target': target,
        'target_pad': pad,
        'chunk_demo': target[:, 1:],
        'pad': pad[:, 1:],
        'chunk_pi': target[:, 1:] + off,
        'state': torch.zeros(len(rows), 15),
        'episode': episode,
        'frame': torch.tensor([r[1] for r in rows]),
    })
    meta = {'model': 'tiny', 'type': 'multi_task_dit', 'episodes': sorted(EPISODES), 'horizon': H, 'n_action_steps': 8}
    return cache.save(tensors, meta, tmp_path / 'cache').parent


@pytest.fixture
def tiny(monkeypatch):
    monkeypatch.setattr(cache, 'load_policy', lambda model, device='cpu': runtime(TinyDiT()))


def test_update_and_open_loop_on_a_dit_cache(built, tiny, tmp_path):
    s = update.Settings(steps=20, batch=16, lr=1e-3, pool=(21, 25), anchor=(0, 1))
    record = update.run(None, built, tmp_path / 'a', s)
    assert record['settings']['objective'] == 'dit' and record['output'] == 'adapter'
    assert record['corrected_frames'] == 20 and record['anchor_frames'] == 24
    assert all(torch.isfinite(torch.tensor(r['total'])) for r in record['losses'])
    meta = json.loads((tmp_path / 'a' / 'adapter.json').read_text())
    assert meta['root'] == 'noise_predictor' and len(meta['targets']) == 4 * LAYERS

    results = evaluate.run(
        built, tmp_path / 'r.json', adapter=tmp_path / 'a', alphas=(0.0, 1.0), test=(38, 39), forget=(0, 1), n_boot=20
    )
    zero = results['alphas']['0.0']
    assert zero['test'] == results['base']['test'] and zero['forget'] == results['base']['forget']
    assert results['alphas']['1.0']['test'] != results['base']['test']


def test_a_dit_base_refuses_vita_settings(built, tiny, tmp_path):
    for s in (update.Settings(trainable='flow'), update.Settings(beta=1.0), update.Settings(noise_cm=2.0)):
        with pytest.raises(ValueError):
            update.run(None, built, tmp_path / 'x', s)
