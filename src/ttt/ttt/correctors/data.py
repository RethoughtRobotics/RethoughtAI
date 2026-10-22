"""What every corrector learns from: the rows of ttt correction-rows turned into inputs, targets and
weights, and the plan error the evaluation scores.

Trained as CR-DAgger trains its residual (Xu et al. 2025, github.com/yifan-hou/cr-dagger): on whole
rollouts, where the policy drove labelled with no correction, so a corrector learns to leave the base
alone there, and the corrections weighted WEIGHT times (its weighted_sampling: 4).

Which steps were corrected is decided per step of each row's plan (rows' step_intervention), as
FlowCorrect labels a window positive when any of its steps is corrected (Welte et al. 2026): a row
whose plan runs into a correction gets the part of the nudge its corrected steps carry, and its
weight, so the target rises into each correction over one plan instead of jumping from a weighted 0.
Rows built before step_intervention existed are labelled per frame, with the LEAD_UP s before each
correction weighted as CR-DAgger's correction_horizon.
"""

import numpy as np

WEIGHT = 4.0
LEAD_UP = 1.0  # s
CONSTANT = 1e-6  # a feature's spread over the rows below this: it does not vary


def nudge(chunk_pi, chunk_demo, mask=None):
    """The whole-plan nudge per row: the person's plan minus the base's, averaged over the plan's
    steps, (N, 7); with `mask` (N, steps), the steps outside it count as 0 (the policy's own steps).
    Rotation vectors are subtracted directly, a first-order approximation for the small angles of a
    correction."""
    difference = chunk_demo - chunk_pi
    return (difference if mask is None else difference * mask[:, :, None]).mean(axis=1)


def problem(tensors, fps=25):
    """(inputs, targets, weights, corrected, untouched) of the rows: inputs the base's features and
    state; targets the nudge over the plan's corrected steps, 0 where none is; weights WEIGHT on rows
    with a corrected step (rows without step_intervention: on correction frames and the LEAD_UP
    before each), 1 elsewhere; corrected the rows whose frame the person drove, untouched those no
    step of whose plan they did (what the evaluation scores)."""
    z = tensors['cond'].float().numpy().astype(np.float64)
    if 'state' in tensors:
        z = np.hstack([z, tensors['state'].numpy().astype(np.float64)])
    pi, demo = (tensors[k].numpy().astype(np.float64) for k in ('chunk_pi', 'chunk_demo'))
    n = len(pi)
    corrected = tensors['intervention'].numpy() > 0.5 if 'intervention' in tensors else np.ones(n, bool)
    if 'step_intervention' in tensors:
        steps = tensors['step_intervention'].numpy() > 0.5
        y = nudge(pi, demo, steps)
        touched = steps.any(axis=1)
        return z, y, np.where(touched, WEIGHT, 1.0), corrected, ~touched
    y = np.where(corrected[:, None], nudge(pi, demo), 0.0)
    episode, frame = tensors['episode'].numpy(), tensors['frame'].numpy()
    lead_up = np.zeros(n, bool)
    for i in np.flatnonzero(corrected & ~np.r_[False, corrected[:-1]]):  # where a correction starts
        lead_up |= (episode == episode[i]) & (frame >= frame[i] - LEAD_UP * fps) & (frame < frame[i])
    return z, y, np.where(corrected | lead_up, WEIGHT, 1.0), corrected, ~corrected


class Scaler:
    """Standardises inputs as fitted on training rows. A feature that does not vary over them (e.g. the
    task text's embedding) carries no information: scale 1 and masked out, so a live value that differs
    from the rows' by rounding (they are cached as float16) cannot be amplified."""

    def __init__(self, z):
        self.mean, spread = z.mean(0), z.std(0)
        self.varies = spread > CONSTANT
        self.std = np.where(self.varies, spread, 1.0)

    def __call__(self, z):
        """(N, d) standardised, the constant features 0."""
        return np.where(self.varies, (z - self.mean) / self.std, 0.0)


def weighted_mean(y, w):
    return (w[:, None] * y).sum(0) / w.sum()


def plan_error(chunk_pi, chunk_demo, offset):
    """Per row: the mean distance (cm) between the person's plan and the base's plan shifted by
    `offset` (N, 7) over every step, the mean gripper difference, and whether the shifted plan's
    gripper decision now (step 0, closed above 0.5, the trigger kept within 0 .. 1) is the person's."""
    shifted = chunk_pi + offset[:, None, :]
    position = np.linalg.norm((shifted - chunk_demo)[:, :, :3], axis=2).mean(1) * 100
    trigger = np.clip(shifted[:, :, 6], 0.0, 1.0)
    gripper = np.abs(trigger - chunk_demo[:, :, 6]).mean(1)
    decision = (trigger[:, 0] > 0.5) == (chunk_demo[:, 0, 6] > 0.5)
    return position, gripper, decision
