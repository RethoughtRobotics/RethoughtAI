"""A nudge adapter: a linear corrector on a frozen policy, fitted from human corrections (ttt linear).

The frozen policy plans as always; the adapter adds one offset to every step of the plan:

    plan' = plan + alpha * (W z + b)

where z is the policy's own observation encoder output for the frame (the Multi-Task DiT's
conditioning vector, standardised as at fitting time). The offset covers position and rotation (the
first 6 of the 7 relative actions); the gripper is left to the policy. The offset is not capped: what
bounds the arm is the teleop's servo speed limits, the operator's takeover and the e-stop.

On disk, next to the checkpoint it was fitted on (models/<policy>/adapters/<name>/):
  adapter.json         kind 'nudge'; `base` (the checkpoint and the sha256 of its model.safetensors,
                       which a load must match), `corrections` (the dataset it learned from), `fit`
  adapter.safetensors  W (inputs x 6), b (6), mean and std (inputs) of z
"""

import hashlib
import json
from pathlib import Path

import numpy as np

KIND = 'nudge'
WEIGHTS, META = 'adapter.safetensors', 'adapter.json'
OUTPUTS = 6  # dx dy dz, rotation vector


def fingerprint(model_dir):
    """sha256 of the checkpoint's model.safetensors: what a nudge adapter was fitted on."""
    digest = hashlib.sha256()
    with open(Path(model_dir) / 'model.safetensors', 'rb') as f:
        for block in iter(lambda: f.read(1 << 24), b''):
            digest.update(block)
    return digest.hexdigest()


def save(path, W, b, mean, std, model_dir, **meta):
    """Write a nudge adapter at `path` for the checkpoint at `model_dir`; `meta` (corrections, fit, ...)
    goes into adapter.json."""
    from safetensors.numpy import save_file

    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    W, b = np.asarray(W, np.float32)[:, :OUTPUTS], np.asarray(b, np.float32)[:OUTPUTS]
    save_file(
        {'W': W, 'b': b, 'mean': np.asarray(mean, np.float32), 'std': np.asarray(std, np.float32)},
        str(path / WEIGHTS),
    )
    entry = {
        'kind': KIND,
        'base': {'checkpoint': str(Path(model_dir).resolve()), 'sha256': fingerprint(model_dir)},
        'inputs': int(W.shape[0]),
        'outputs': OUTPUTS,
        **meta,
    }
    (path / META).write_text(json.dumps(entry, indent=2) + '\n')
    return path


class Nudge:
    """A loaded nudge adapter on a policy: records the encoder's output of each plan and offsets it."""

    def __init__(self, path, policy, model_dir, alpha):
        from safetensors.numpy import load_file

        meta = json.loads((Path(path) / META).read_text())
        if meta.get('kind') != KIND:
            raise ValueError(f'{path}: not a nudge adapter')
        if fingerprint(model_dir) != meta['base']['sha256']:
            raise ValueError(f'{path} was fitted on {meta["base"]["checkpoint"]}, not on {model_dir}')
        w = load_file(str(Path(path) / WEIGHTS))
        self.W, self.b, self.mean, self.std = (w[k].astype(np.float64) for k in ('W', 'b', 'mean', 'std'))
        self.alpha = float(alpha)
        self.z = self.offset = None  # of the last plan
        encoder = policy.observation_encoder
        original = encoder.encode

        def encode(batch):
            out = original(batch)
            self.z = out[0].float().cpu().numpy()
            return out

        encoder.encode = encode

    def __call__(self, actions):
        """The plan (K, 7) relative actions plus alpha times the offset for the last encoded frame; the
        offset (6,) applied is kept in `offset`."""
        offset = self.alpha * (((self.z - self.mean) / self.std) @ self.W + self.b)
        self.offset = offset
        out = np.array(actions, dtype=np.float64, copy=True)
        out[:, :OUTPUTS] += offset
        return out
