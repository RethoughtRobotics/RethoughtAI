"""LoRA adapters on a frozen policy's linear layers: each wrapped layer computes (W + alpha B A) x.

A (rank x in) and B (out x rank) are all that trains; W and everything else stay frozen. B starts at
zero, so a new adapter is the base policy, and alpha blends: 0 is the base exactly, 1 the adapted one.
On VITA they sit on fc1 and fc2 of each flow net block (VITA_TARGETS: 8 layers with the repository's
4 blocks), named relative to the flow net (flow_net()), the only part a test-time update over cached
latents runs.

An adapter on disk is a directory: adapter.safetensors (each target's A and B) and adapter.json
(rank, target names and, if saved with one, `root`: the module of the policy the names are relative
to, e.g. 'vita' for adapters on the observation encoder or action decoder too). An adapter without
a root is on the flow net.

A weights adapter (adapter.json's `kind` 'weights', saved by save_weights) holds tuned weights instead
of A and B: a fine-tuned part of the policy (ttt update's flow, flow_last, full) or, with `weights`
naming another file, a whole policy's state dict. Loaded, each held tensor is the base's interpolated
toward the tuned one, alpha 0 the base and 1 the tuned weights exactly. Either kind may record the
`trainable` set it was trained as.
"""

import fnmatch
import json
import math
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

VITA_TARGETS = ('layers.*.mlp.fc1', 'layers.*.mlp.fc2')
WEIGHTS, META = 'adapter.safetensors', 'adapter.json'


class LoRALinear(nn.Module):
    """A frozen nn.Linear plus alpha B A."""

    def __init__(self, base: nn.Linear, rank: int, alpha: float = 1.0):
        super().__init__()
        self.base, self.rank, self.alpha = base, rank, alpha
        base.requires_grad_(False)
        w = base.weight
        self.A = nn.Parameter(torch.empty(rank, base.in_features, device=w.device, dtype=w.dtype))
        self.B = nn.Parameter(torch.zeros(base.out_features, rank, device=w.device, dtype=w.dtype))
        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))  # as nn.Linear's weight, and PEFT's A

    def forward(self, x):
        y = self.base(x)
        if self.alpha == 0:
            return y
        return y + self.alpha * F.linear(F.linear(x, self.A), self.B)

    def delta(self):
        """alpha B A, the change to W."""
        return self.alpha * self.B @ self.A


FLOW_NET = 'vita.flow_net'


def flow_net(policy):
    """The module VITA's adapter names are relative to: VitaPolicy's flow net."""
    return policy.get_submodule(FLOW_NET)


def targets(root, patterns=VITA_TARGETS):
    """Names (relative to root) of the nn.Linear layers matching any of the fnmatch patterns."""
    return [
        name
        for name, m in root.named_modules()
        if isinstance(m, nn.Linear) and any(fnmatch.fnmatchcase(name, p) for p in patterns)
    ]


def wrap(root, names, rank, alpha=1.0):
    """Replace each named nn.Linear of root by a LoRALinear around it; returns {name: LoRALinear}."""
    if not names:
        raise ValueError('no layers to adapt')
    wrapped = {}
    for name in names:
        parent, _, child = name.rpartition('.')
        owner = root.get_submodule(parent) if parent else root
        base = getattr(owner, child)
        if not isinstance(base, nn.Linear):
            raise TypeError(f'{name} is a {type(base).__name__}, not an nn.Linear')
        wrapped[name] = LoRALinear(base, rank, alpha)
        setattr(owner, child, wrapped[name])
    return wrapped


def adapters(root):
    """{name: LoRALinear} of root."""
    return {name: m for name, m in root.named_modules() if isinstance(m, LoRALinear)}


def set_alpha(root, alpha):
    for m in adapters(root).values():
        m.alpha = float(alpha)


def parameters(root):
    """The trainable parameters: every adapter's A and B."""
    return [p for m in adapters(root).values() for p in (m.A, m.B)]


def save(root, path, name=None, trainable=None):
    """Write root's adapters (A, B, rank, target names) to the directory `path`; `name`, root's
    module name in the policy (as policy.get_submodule takes it), is stored as the adapter's root,
    and `trainable` as the set it was trained as."""
    from safetensors.torch import save_file

    layers = adapters(root)
    if not layers:
        raise ValueError('no adapters to save')
    ranks = {m.rank for m in layers.values()}
    if len(ranks) != 1:
        raise ValueError(f'adapters of different ranks {sorted(ranks)}')
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    tensors = {f'{name}.{k}': getattr(m, k).detach().cpu().contiguous() for name, m in layers.items() for k in 'AB'}
    save_file(tensors, str(path / WEIGHTS))
    meta = {'rank': ranks.pop(), 'targets': list(layers)}
    if name is not None:
        meta['root'] = name
    if trainable is not None:
        meta['trainable'] = trainable
    (path / META).write_text(json.dumps(meta, indent=2) + '\n')
    return path


def load(root, path, alpha=1.0):
    """Wrap root's layers named in the adapter at `path` and load its A and B; returns {name: LoRALinear}."""
    from safetensors.torch import load_file

    path = Path(path)
    meta = json.loads((path / META).read_text())
    tensors = load_file(str(path / WEIGHTS))
    expected = {f'{name}.{k}' for name in meta['targets'] for k in 'AB'}
    if set(tensors) != expected:
        raise ValueError(f'{path / WEIGHTS} holds {sorted(set(tensors) ^ expected)} not in or missing from {META}')
    wrapped = wrap(root, meta['targets'], meta['rank'], alpha)
    with torch.no_grad():
        for name, m in wrapped.items():
            for k in 'AB':
                param, value = getattr(m, k), tensors[f'{name}.{k}']
                if param.shape != value.shape:
                    raise ValueError(f'{name}.{k}: adapter {tuple(value.shape)}, layer {tuple(param.shape)}')
                param.copy_(value)
    return wrapped


def meta(path):
    """The adapter's adapter.json."""
    return json.loads((Path(path) / META).read_text())


def root_of(policy, path):
    """The module of `policy` that the adapter at `path` is named relative to."""
    return policy.get_submodule(meta(path).get('root', FLOW_NET))


def save_weights(policy, names, path, trainable, weights=None):
    """Write the tensors `names` of `policy` (state dict names) as a weights adapter at `path`; with
    `weights`, a file at `path` already holding them (a whole checkpoint's model.safetensors), only
    the adapter.json naming it."""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    if weights is None:
        from safetensors.torch import save_file

        state = policy.state_dict()
        save_file({k: state[k].detach().cpu().contiguous() for k in names}, str(path / WEIGHTS))
    entry = {'kind': 'weights', 'trainable': trainable, **({'weights': weights} if weights else {})}
    (path / META).write_text(json.dumps(entry, indent=2) + '\n')
    return path


def tuned(path):
    """A weights adapter's tensors, by state dict name."""
    from safetensors.torch import load_file

    return load_file(str(Path(path) / meta(path).get('weights', WEIGHTS)))


def blend(base, value, alpha):
    """base toward the tuned value by alpha, 0 and 1 exactly the ends (tensors not floating: the value)."""
    if not base.dtype.is_floating_point:
        return value.to(base.device)
    return torch.lerp(base, value.to(base.device, base.dtype), float(alpha))


def load_weights(policy, path, alpha=1.0):
    """Set each tensor of the weights adapter at `path` in `policy`, blended with the base's by
    `alpha`; returns the names set."""
    tensors, state = tuned(path), policy.state_dict()
    unknown = sorted(set(tensors) - set(state))
    if unknown:
        raise ValueError(f'{path}: {unknown[0]} (and {len(unknown) - 1} more) not in the policy')
    with torch.no_grad():
        for k, v in tensors.items():
            if v.shape != state[k].shape:
                raise ValueError(f'{k}: adapter {tuple(v.shape)}, policy {tuple(state[k].shape)}')
            state[k].copy_(blend(state[k], v, alpha))
    return list(tensors)


def load_policy(policy, path, alpha=1.0):
    """Load the adapter at `path` into `policy` at `alpha`: a weights adapter by load_weights, a LoRA
    one by load() on the module it names as its root (the flow net if none)."""
    if meta(path).get('kind') == 'weights':
        return load_weights(policy, path, alpha)
    return load(root_of(policy, path), path, alpha)
