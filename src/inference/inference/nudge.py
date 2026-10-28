"""A nudge adapter: a small corrector on a frozen policy, fitted from human corrections (ttt corrector;
the methods are in ttt.correctors, README.md there).

The frozen policy plans as always; the adapter adds one offset to every step of the plan, or, fitted
per step (adapter.json `steps`), each of the plan's first `steps` its own (the last also to any later):

    plan' = plan + alpha * f(x),   x = (z - mean) / std

where z is the policy's own observation encoder output for the frame (the Multi-Task DiT's
conditioning vector), followed by the frame's observation.state when the adapter was fitted with it
(adapter.json `state`, its length), and f is the fitted model (adapter.json `model`):
  linear  f(x) = x W + b
  mlp     layers of weight (out, in) and bias, GELU between them, none after the last
  kernel  a memory of corrections (ttt.correctors.kernel): the stored nudges weighted by tricube on
          the cosine similarity of x to their keys, exactly 0 when no key is within s_min
The offset covers position and rotation (the first 6 of the 7 relative actions) and, in an adapter
fitted with it (adapter.json `outputs` 7), the gripper trigger, kept within 0 (open) .. 1 (closed). It
is not capped: what bounds the arm is the teleop's servo speed limits, the operator's takeover and the
e-stop.

On disk, next to the checkpoint it was fitted on (models/<policy>/adapters/<name>/):
  adapter.json         kind 'nudge'; `model`; `base` (the checkpoint and the sha256 of its
                       model.safetensors, which a load must match); `corrections` (the dataset it
                       learned from); `fit` (the method and what it chose)
  adapter.safetensors  mean and std of the inputs; linear: W (inputs x outputs), b; mlp:
                       layer<i>.weight, layer<i>.bias; kernel: mask (the inputs it compares),
                       keys (stored x inputs, unit length), values (stored x outputs)
"""

import hashlib
import json
from pathlib import Path

import numpy as np

KIND = 'nudge'
WEIGHTS, META = 'adapter.safetensors', 'adapter.json'
POSE = 6  # dx dy dz, rotation vector; a 7th output is the gripper trigger


def fingerprint(model_dir):
    """sha256 of the checkpoint's model.safetensors: what a nudge adapter was fitted on."""
    digest = hashlib.sha256()
    with open(Path(model_dir) / 'model.safetensors', 'rb') as f:
        for block in iter(lambda: f.read(1 << 24), b''):
            digest.update(block)
    return digest.hexdigest()


def save(path, model, arrays, model_dir, state=0, steps=0, **meta):
    """Write a nudge adapter at `path` for the checkpoint at `model_dir`: `model` its description
    ({'type': 'linear'} or {'type': 'mlp', 'layers': n, 'activation': 'gelu'}), `arrays` its weights
    (module docstring), `state` how many of the last inputs are the observation.state, `steps` the
    plan steps of a per-step model (its outputs step by step), 0 for one offset per plan; `meta`
    (corrections, fit, ...) goes into adapter.json."""
    from safetensors.numpy import save_file

    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    # Contiguous float32 copies: safetensors writes an array's memory as it lies, so a column slice
    # would be stored with its entries in the wrong places.
    arrays = {k: np.ascontiguousarray(np.asarray(v, np.float32)) for k, v in arrays.items()}
    save_file(arrays, str(path / WEIGHTS))
    entry = {
        'kind': KIND,
        'model': model,
        'base': {'checkpoint': str(Path(model_dir).resolve()), 'sha256': fingerprint(model_dir)},
        'inputs': int(len(arrays['mean'])),
        'state': int(state),  # the last `state` inputs are the observation.state; the rest the encoder's z
        # per step: 6 position and rotation; 7 and the gripper trigger
        'outputs': int(len(_model(model, arrays).bias)) // max(int(steps), 1),
        'steps': int(steps),
        **meta,
    }
    (path / META).write_text(json.dumps(entry, indent=2) + '\n')
    return path


class _Linear:
    def __init__(self, a):
        self.W, self.bias = a['W'].astype(np.float64), a['b'].astype(np.float64)

    def __call__(self, x):
        return x @ self.W + self.bias


class _MLP:
    def __init__(self, a, layers):
        self.layers = [
            (a[f'layer{i}.weight'].astype(np.float64), a[f'layer{i}.bias'].astype(np.float64)) for i in range(layers)
        ]
        self.bias = self.layers[-1][1]

    def __call__(self, x):
        from scipy.special import erf

        for i, (W, b) in enumerate(self.layers):
            x = x @ W.T + b
            if i < len(self.layers) - 1:
                x = 0.5 * x * (1.0 + erf(x / np.sqrt(2.0)))  # GELU, exact as PyTorch's default
        return x


class _Kernel:
    """ttt.correctors.kernel's recall, served."""

    def __init__(self, a, s_min):
        self.mask = a['mask'] > 0.5
        self.keys, self.values = a['keys'].astype(np.float64), a['values'].astype(np.float64)
        self.s_min = float(s_min)
        self.bias = np.zeros(self.values.shape[1])

    def __call__(self, x):
        x = np.where(self.mask, x, 0.0)
        x = x / max(float(np.linalg.norm(x)), 1e-12)
        u = (1.0 - self.keys @ x) / (1.0 - self.s_min)
        phi = np.where(u < 1.0, (1.0 - np.clip(u, 0.0, 1.0) ** 3) ** 3, 0.0)
        return (phi @ self.values) / max(1.0, float(phi.sum()))


def _model(model, arrays):
    kind = (model or {}).get('type', 'linear')  # adapters written before `model` was recorded are linear
    if kind == 'linear':
        return _Linear(arrays)
    if kind == 'mlp':
        if model.get('activation', 'gelu') != 'gelu':
            raise ValueError(f'nudge mlp activation {model["activation"]!r}: only gelu is served')
        return _MLP(arrays, int(model['layers']))
    if kind == 'kernel':
        if model.get('kernel', 'tricube') != 'tricube':
            raise ValueError(f'nudge kernel {model["kernel"]!r}: only tricube is served')
        return _Kernel(arrays, model['s_min'])
    raise ValueError(f'nudge model type {kind!r}: linear, mlp or kernel')


class Nudge:
    """A loaded nudge adapter on a policy: records the encoder's output of each plan and offsets it."""

    def __init__(self, path, policy, model_dir, alpha):
        from safetensors.numpy import load_file

        meta = json.loads((Path(path) / META).read_text())
        if meta.get('kind') != KIND:
            raise ValueError(f'{path}: not a nudge adapter')
        if fingerprint(model_dir) != meta['base']['sha256']:
            raise ValueError(f'{path} was fitted on {meta["base"]["checkpoint"]}, not on {model_dir}')
        arrays = load_file(str(Path(path) / WEIGHTS))
        self.mean, self.std = arrays['mean'].astype(np.float64), arrays['std'].astype(np.float64)
        self.model = _model(meta.get('model'), arrays)
        self.kind = (meta.get('model') or {}).get('type', 'linear')
        self.alpha = float(alpha)
        self.state = int(meta.get('state', 0))
        self.outputs = int(meta.get('outputs', POSE))
        self.steps = int(meta.get('steps', 0))
        self.z = self.offset = None  # of the last plan; per step: its first step's (`offsets` all of them)
        self.offsets = None
        encoder = policy.observation_encoder
        original = encoder.encode

        def encode(batch):
            out = original(batch)
            self.z = out[0].float().cpu().numpy()
            return out

        encoder.encode = encode

    def __call__(self, actions, state=None):
        """The plan (K, 7) relative actions plus alpha times the offset for the last encoded frame
        (and its observation.state `state`, when the adapter takes it), the trigger kept within 0 .. 1;
        the offset (6 or 7,) applied is kept in `offset` (per step: at the first step, all in `offsets`)."""
        inputs = self.z
        if self.state:
            if state is None:
                raise ValueError('this nudge adapter needs the observation.state')
            inputs = np.concatenate([self.z, np.asarray(state, np.float64).reshape(-1)])
        offset = self.alpha * self.model((inputs - self.mean) / self.std)
        out = np.array(actions, dtype=np.float64, copy=True)
        if self.steps:
            offsets = offset.reshape(self.steps, self.outputs)
            k = np.minimum(np.arange(len(out)), self.steps - 1)  # past the fitted steps: the last one's
            out[:, : self.outputs] += offsets[k]
            self.offsets, offset = offsets, offsets[0]
        else:
            out[:, : self.outputs] += offset
        self.offset = offset
        if self.outputs > POSE:
            out[:, POSE] = np.clip(out[:, POSE], 0.0, 1.0)
        return out
