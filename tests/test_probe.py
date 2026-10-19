import torch
from torch import nn

from inference import probe


class Toy(nn.Module):
    """Alternating layers run in a loop (as a flow policy's steps), and one module never runs."""

    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList([nn.Linear(4, 4) if i % 2 == 0 else nn.Linear(4, 4, bias=False) for i in range(6)])
        self.unused = nn.Linear(4, 2)
        self.attn = nn.MultiheadAttention(4, 2)

    def forward(self, x):
        for _ in range(3):
            for layer in self.layers:
                x = layer(x)
        return self.attn(x, x, x)[0]


def test_trace_counts_calls_and_shapes():
    model = Toy()
    root = probe.trace(model, lambda: model(torch.zeros(5, 1, 4)))
    nodes = {n.name: n for n in probe.walk(root)}
    assert nodes['layers.0'].calls == 3 and nodes['layers.0'].outputs == [[5, 1, 4]]
    assert nodes['unused'].calls == 0
    assert nodes['attn'].attention and not nodes['layers.0'].attention
    assert root.params == sum(p.numel() for p in model.parameters())


def test_alternating_stack_folds_to_its_unit():
    model = Toy()
    root = probe.trace(model, lambda: model(torch.zeros(5, 1, 4)))
    layers = next(n for n in root.children if n.name == 'layers')
    assert probe.period(layers) == 2
    assert probe.period(root) == 0  # named children are not a stack
    out = probe.text(root)
    assert '2–5  repeat 0–1' in out and '    2  Linear' not in out


def test_json_round_trip():
    model = Toy()
    root = probe.trace(model, lambda: model(torch.zeros(5, 1, 4)))
    assert probe.from_json(probe.to_json(root)) == root


def test_blank_observation_is_hwc_images_and_state():
    obs = probe.blank_observation({
        'observation.images.left': {'type': 'VISUAL', 'shape': [3, 4, 6]},
        'observation.state': {'type': 'STATE', 'shape': [17]},
    })
    assert obs['observation.images.left'].shape == (4, 6, 3) and obs['observation.state'].shape == (17,)
