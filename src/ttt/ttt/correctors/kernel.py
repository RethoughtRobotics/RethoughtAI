"""Kernel memory: the corrected rows stored as (key, nudge) pairs, the nudge at a frame the stored
nudges weighted by a compact-support kernel on the keys' cosine similarity s:

    f(x) = sum_i phi(u_i) v_i / max(1, sum_i phi(u_i)),   u_i = (1 - s_i) / (1 - s_min)
    phi(u) = (1 - u^3)^3 for u < 1, else 0              (tricube, phi(0) = 1)

Keys are the standardised inputs, the constant features masked out, scaled to unit length.

What follows from the form, with no data:
  - f(x) = 0 exactly when every stored key has s <= s_min: the plan is the base's.
  - |f(x)| <= max_i |v_i|: the weights sum to at most 1.
  - a new pair changes f only where its own s > s_min.
  - f is continuous, C^2 at the support's edge, as tricube and its first two derivatives vanish
    there; dividing by sum phi alone (Nadaraya and Watson) would jump from v_i to 0 at that edge.

s_min is chosen on held-out training episodes by the weighted error ridge uses for lambda, or fixed.
"""

import numpy as np

from .data import WEIGHT, Scaler

S_MINS = np.round(np.arange(0.0, 1.0, 0.05), 2)


def tricube(u):
    """(1 - u^3)^3 on 0 <= u < 1, 0 beyond."""
    return np.where(u < 1.0, (1.0 - np.clip(u, 0.0, 1.0) ** 3) ** 3, 0.0)


def unit(x, mask):
    """Rows of `x` with the masked features 0, scaled to unit length (a zero row stays 0)."""
    x = np.where(mask, x, 0.0)
    return x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), 1e-12)


def recall(keys, values, s_min, x):
    """The memory's nudge (N, k) for unit query rows `x` (N, d)."""
    phi = tricube((1.0 - x @ keys.T) / (1.0 - s_min))
    return (phi @ values) / np.maximum(1.0, phi.sum(axis=1, keepdims=True))


class Kernel:
    name = 'kernel'

    def __init__(self, s_min=None, s_mins=S_MINS):
        self.fixed, self.s_mins = s_min, np.asarray(s_mins, float)
        self.s_min = s_min
        self.scaler = self.keys = self.values = None

    def _store(self, z, y, w):
        self.scaler = Scaler(z)
        stored = w >= WEIGHT  # the rows whose plan holds a correction
        self.keys, self.values = unit(self.scaler(z[stored]), self.scaler.varies), y[stored]

    def _recall(self, z, s_min):
        return recall(self.keys, self.values, s_min, unit(self.scaler(z), self.scaler.varies))

    def fit(self, z, y, w, groups):
        """Store the corrected rows; s_min fixed, or the one with the lowest weighted error on held-out
        `groups` (episodes)."""
        if self.fixed is None:
            errors = np.zeros(len(self.s_mins))
            for held in np.unique(groups):
                tr, te = groups != held, groups == held
                self._store(z[tr], y[tr], w[tr])
                for j, s_min in enumerate(self.s_mins):
                    e = np.linalg.norm(self._recall(z[te], s_min) - y[te], axis=1)
                    errors[j] += (w[te] * e).sum() / w[te].sum()
            self.s_min = float(self.s_mins[int(np.argmin(errors))])
        self._store(z, y, w)
        return self

    def predict(self, z):
        return self._recall(z, self.s_min)

    def summary(self):
        return {'s_min': self.s_min, 'stored': int(len(self.keys))}

    def export(self, columns):
        """(model description, arrays) for the nudge adapter (inference.nudge): the target `columns`."""
        arrays = {
            'mean': self.scaler.mean,
            'std': self.scaler.std,
            'mask': self.scaler.varies.astype(np.float32),
            'keys': self.keys,
            'values': self.values[:, columns],
        }
        return {'type': 'kernel', 'kernel': 'tricube', 's_min': self.s_min}, arrays
