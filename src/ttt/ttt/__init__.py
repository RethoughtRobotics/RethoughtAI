"""Test-time training of a frozen VITA policy over cached latents (training/training_experiments.md,
shared code items 4 to 8): the latent cache, the offline corrector, the update loop and the open-loop
evaluation, with bootstrap intervals in ttt.stats."""

# The fixed split of the cup datasets: base policy and anchor, correction pool, open-loop test
BASE, POOL, TEST = range(0, 21), range(21, 38), range(38, 56)
M0_POOL = (21, 25, 29, 33)
