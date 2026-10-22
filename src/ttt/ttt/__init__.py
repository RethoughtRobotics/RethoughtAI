"""Test-time training of a frozen VITA policy over cached latents (training/training_experiments.md,
shared code items 4 to 8): the latent cache, the offline corrector, the update loop and the open-loop
evaluation, with bootstrap intervals in ttt.stats."""

import os

# numpy's BLAS and PyTorch each start a thread per core; together they oversubscribe the CPU and a
# 1789 x 1789 eigendecomposition took 13.5 s instead of 2.1 s (ttt.corrections). Set before numpy
# loads; an explicit setting in the environment wins.
for _name in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(_name, str(min(8, os.cpu_count() or 1)))

# The fixed split of the cup datasets: base policy and anchor, correction pool, open-loop test
BASE, POOL, TEST = range(0, 21), range(21, 38), range(38, 56)
M0_POOL = (21, 25, 29, 33)
