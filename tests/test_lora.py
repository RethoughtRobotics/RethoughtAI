"""LoRA adapters (inference.lora) on VITA's flow net: alpha 0 and a zero B are the base exactly, and an
adapter survives a save and load."""

import json
import math

import pytest
import torch

from inference import lora
from inference.vita.modeling_vita import SimpleFlowNet


def flow_net(seed=0, layers=4, dim=512):
    torch.manual_seed(seed)
    return SimpleFlowNet(dim, dim, dim, layers).eval()


def inputs(dim=512, batch=8):
    g = torch.Generator().manual_seed(1)
    return torch.randn(batch, dim, generator=g), torch.rand(batch, generator=g)


def randomize(net):
    g = torch.Generator().manual_seed(2)
    with torch.no_grad():
        for m in lora.adapters(net).values():
            m.B.copy_(0.05 * torch.randn(m.B.shape, generator=g))


def test_vita_targets_are_the_flow_nets_eight_mlp_layers():
    names = lora.targets(flow_net())
    assert names == [f'layers.{i}.mlp.fc{k}' for i in range(4) for k in (1, 2)]
    net = flow_net()
    lora.wrap(net, names, rank=16)
    assert sum(p.numel() for p in lora.parameters(net)) == 4 * 16 * (512 + 2048) * 2  # 0.33 M


def test_zero_initialised_b_is_the_base():
    x, t = inputs()
    base, net = flow_net(), flow_net()
    lora.wrap(net, lora.targets(net), rank=16)
    assert torch.equal(net(x, t), base(x, t))


def test_alpha_zero_is_the_base_exactly():
    x, t = inputs()
    base, net = flow_net(), flow_net()
    lora.wrap(net, lora.targets(net), rank=16)
    randomize(net)
    assert not torch.allclose(net(x, t), base(x, t))
    lora.set_alpha(net, 0.0)
    assert torch.equal(net(x, t), base(x, t))


def test_layer_computes_w_plus_alpha_ba():
    torch.manual_seed(0)
    linear = torch.nn.Linear(6, 5)
    layer = lora.LoRALinear(linear, rank=2, alpha=0.7)
    with torch.no_grad():
        layer.B.normal_()
    x = torch.randn(3, 6)
    expected = x @ (linear.weight + 0.7 * layer.B @ layer.A).T + linear.bias
    assert torch.allclose(layer(x), expected, atol=1e-6)
    assert not linear.weight.requires_grad and layer.A.requires_grad and layer.B.requires_grad


def test_save_and_load_round_trip(tmp_path):
    x, t = inputs()
    net = flow_net()
    lora.wrap(net, lora.targets(net), rank=8, alpha=0.5)
    randomize(net)
    lora.save(net, tmp_path / 'adapter')
    meta = json.loads((tmp_path / 'adapter' / lora.META).read_text())
    assert meta == {'rank': 8, 'targets': lora.targets(flow_net())}
    from safetensors.torch import load_file

    saved = load_file(str(tmp_path / 'adapter' / lora.WEIGHTS))
    assert set(saved) == {f'{n}.{k}' for n in meta['targets'] for k in 'AB'}  # A and B only, not W

    loaded = flow_net()
    lora.load(loaded, tmp_path / 'adapter', alpha=0.5)
    assert torch.equal(loaded(x, t), net(x, t))
    lora.set_alpha(loaded, 0.0)
    assert torch.equal(loaded(x, t), flow_net()(x, t))


def test_load_refuses_another_shape(tmp_path):
    net = flow_net(dim=64)
    lora.wrap(net, lora.targets(net), rank=4)
    lora.save(net, tmp_path)
    with pytest.raises(ValueError):
        lora.load(flow_net(), tmp_path)


def test_wrap_refuses_what_is_not_linear():
    with pytest.raises(TypeError):
        lora.wrap(flow_net(), ['layers.0.norm'], rank=4)


def test_a_is_initialised_as_a_linear_weight():
    layer = lora.LoRALinear(torch.nn.Linear(512, 512), rank=16)
    bound = 1 / math.sqrt(512)
    assert layer.A.abs().max() <= bound and torch.count_nonzero(layer.B) == 0


def test_runtime_policy_with_an_adapter(tmp_path):
    """LeRobotPolicy(adapter=...): alpha 0 plans exactly the base's chunk, alpha 1 another."""
    import numpy as np
    from lerobot.configs.types import FeatureType, PolicyFeature

    from inference.policies import LeRobotPolicy
    from inference.vita import VitaConfig, VitaPolicy

    dim = 64
    cfg = VitaConfig(
        input_features={
            'observation.images.left': PolicyFeature(FeatureType.VISUAL, (3, 32, 40)),
            'observation.state': PolicyFeature(FeatureType.STATE, (15,)),
        },
        output_features={'action': PolicyFeature(FeatureType.ACTION, (8,))},
        horizon=32,
        n_action_steps=16,
        pretrained_backbone_weights=None,
        resize_shape=(32, 40),
        crop_shape=None,
        latent_dim=dim,
        flow_hidden_dim=dim,
        ae_enc_hidden_dim=dim,
        ae_dec_hidden_dim=dim,
        device='cpu',
        push_to_hub=False,
    )
    torch.manual_seed(0)
    model = tmp_path / 'vita'
    VitaPolicy(cfg).save_pretrained(model)
    stats = {k: {'mean': [0.0] * n, 'std': [1.0] * n} for k, n in (('observation.state', 15), ('action', 8))}
    (model / 'normalization_stats.json').write_text(json.dumps(stats))

    net = flow_net(dim=dim)
    lora.wrap(net, lora.targets(net), rank=4)
    randomize(net)
    lora.save(net, tmp_path / 'adapter')

    rng = np.random.default_rng(0)
    obs = {
        'observation.images.left': rng.integers(0, 255, (32, 40, 3), dtype=np.uint8),
        'observation.state': rng.normal(size=15).astype(np.float32),
    }
    tcp = (np.array([0.6, 0.2, 0.1]), np.array([0.0, 1.0, 0.0, 0.0]))
    base = LeRobotPolicy(str(model), 'cpu')(obs, tcp=tcp)
    off = LeRobotPolicy(str(model), 'cpu', adapter=tmp_path / 'adapter', alpha=0.0)
    on = LeRobotPolicy(str(model), 'cpu', adapter=tmp_path / 'adapter', alpha=1.0)
    assert len(lora.adapters(lora.flow_net(off.policy))) == 8
    assert np.array_equal(off(obs, tcp=tcp), base)
    assert not np.allclose(on(obs, tcp=tcp), base)
