"""Paired comparisons over units (test episodes or layouts): the statistic of the per-unit
differences with a percentile bootstrap interval over units."""

import numpy as np

STATS = {'median': np.median, 'mean': np.mean}


def bootstrap(values, stat='median', n=10000, level=0.95, seed=0):
    """`stat` of `values` (one per unit) and its `level` percentile bootstrap interval over units."""
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 1 or len(values) == 0:
        raise ValueError(f'need a 1-D array of per-unit values, got shape {values.shape}')
    f = STATS[stat]
    rng = np.random.default_rng(seed)
    draws = f(values[rng.integers(0, len(values), (n, len(values)))], axis=1)
    lo, hi = np.quantile(draws, [(1 - level) / 2, (1 + level) / 2])
    return {'stat': stat, 'value': float(f(values)), 'lo': float(lo), 'hi': float(hi), 'units': len(values)}


def paired(a, b, stat='median', n=10000, level=0.95, seed=0):
    """`stat` of a - b over the units both have ({unit: value} each) with its bootstrap interval;
    `excludes_zero` says whether the interval does. Units missing a value (None or NaN) are left out."""
    units = sorted(u for u in set(a) & set(b) if known(a[u]) and known(b[u]))
    if not units:
        raise ValueError('no unit in common')
    out = bootstrap([a[u] - b[u] for u in units], stat, n, level, seed)
    out['excludes_zero'] = bool(out['lo'] > 0 or out['hi'] < 0)
    return out


def known(value):
    return value is not None and bool(np.isfinite(value))
