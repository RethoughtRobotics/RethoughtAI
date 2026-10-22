"""Ridge: the nudge linear in the standardised inputs, W x + b, weighted least squares with an L2
penalty lambda chosen by holding out training episodes in turn."""

import numpy as np

from .data import Scaler, weighted_mean

LAMBDAS = np.logspace(0, 6, 13)


def _solver(z, y, w):
    """lam -> (W, b, scaler): weighted ridge, |sqrt(w) (x W + b - y)|^2 + lam |W|^2 over standardised
    features, from one eigendecomposition (of the features' or the rows' Gram matrix, whichever is
    smaller), so every lam costs a matrix product. The constant features get weight exactly 0."""
    scaler = Scaler(z)
    x = scaler(z)[:, scaler.varies]
    b = weighted_mean(y, w)
    root = np.sqrt(w)[:, None]
    xs, ys = x * root, (y - b) * root
    if len(x) >= x.shape[1]:
        s, V = np.linalg.eigh(xs.T @ xs)
        h = V.T @ (xs.T @ ys)
        solve = lambda lam: V @ (h / (s + lam)[:, None])  # noqa: E731
    else:
        s, U = np.linalg.eigh(xs @ xs.T)
        h = U.T @ ys
        solve = lambda lam: xs.T @ (U @ (h / (s + lam)[:, None]))  # noqa: E731

    def fit(lam):
        W = np.zeros((len(scaler.mean), y.shape[1]))
        W[scaler.varies] = solve(lam)
        return W, b, scaler

    return fit


def fit_ridge(z, y, lam, w=None):
    """(W (d, k), b (k,), scaler): weighted ridge at `lam`, weights 1 by default."""
    return _solver(z, y, np.ones(len(y)) if w is None else np.asarray(w, float))(lam)


def predict(model, z):
    W, b, scaler = model
    return scaler(z) @ W + b


class Ridge:
    name = 'ridge'

    def __init__(self, lambdas=LAMBDAS):
        self.lambdas = np.asarray(lambdas, float)
        self.lam = self.model = None

    def fit(self, z, y, w, groups):
        """Fit on all rows at the lambda with the lowest weighted error on held-out `groups` (episodes)."""
        errors = np.zeros(len(self.lambdas))
        for held in np.unique(groups):
            tr, te = groups != held, groups == held
            solve = _solver(z[tr], y[tr], w[tr])
            for j, lam in enumerate(self.lambdas):
                e = np.linalg.norm(predict(solve(lam), z[te]) - y[te], axis=1)
                errors[j] += (w[te] * e).sum() / w[te].sum()
        self.lam = float(self.lambdas[int(np.argmin(errors))])
        self.model = _solver(z, y, w)(self.lam)
        return self

    def predict(self, z):
        return predict(self.model, z)

    def summary(self):
        return {'lambda': self.lam}

    def export(self, outputs):
        """(model description, arrays) for the nudge adapter (inference.nudge), the first `outputs`."""
        W, b, scaler = self.model
        return {'type': 'linear'}, {'W': W[:, :outputs], 'b': b[:outputs], 'mean': scaler.mean, 'std': scaler.std}
