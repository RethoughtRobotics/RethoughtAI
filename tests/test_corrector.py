"""ttt.corrections: the corrector's labels and its linear fit."""

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

pytest.importorskip('torch')
from ttt import corrections  # noqa: E402
from ttt.correctors import data, ridge  # noqa: E402


def raw_state(p, q, gripper=0.0):
    return np.r_[p, q, gripper, np.zeros(7)]


def test_goals_at_the_tcp_are_zero_relative_actions():
    q = Rotation.from_euler('xyz', [0.3, -0.2, 1.0]).as_quat()
    state = raw_state([0.6, 0.2, 0.1], q)
    goals = np.tile(np.r_[0.6, 0.2, 0.1, q, 1.0], (4, 1))
    np.testing.assert_allclose(corrections.relative(goals, state), np.tile([0, 0, 0, 0, 0, 0, 1.0], (4, 1)), atol=1e-6)


def test_relative_actions_are_base_frame_offsets_from_the_tcp():
    q = Rotation.from_euler('z', 90, degrees=True).as_quat()
    state = raw_state([0.6, 0.2, 0.1], q)
    turned = (Rotation.from_euler('x', 10, degrees=True) * Rotation.from_quat(q)).as_quat()
    goals = np.array([np.r_[0.65, 0.2, 0.1, turned, 0.0]])
    out = corrections.relative(goals, state)[0]
    np.testing.assert_allclose(out[:3], [0.05, 0, 0], atol=1e-6)  # base axes, not the TCP's
    np.testing.assert_allclose(out[3:6], [np.radians(10), 0, 0], atol=1e-6)


def test_the_nudge_is_the_mean_difference_over_the_plan():
    pi = np.zeros((2, 3, 7))
    demo = np.zeros((2, 3, 7))
    demo[0, :, 0] = [0.01, 0.02, 0.03]
    np.testing.assert_allclose(data.nudge(pi, demo)[0], [0.02, 0, 0, 0, 0, 0, 0])


def test_ridge_recovers_a_linear_map():
    rng = np.random.default_rng(0)
    z = rng.normal(size=(300, 20))
    W = rng.normal(size=(20, 3))
    y = z @ W + [0.1, -0.3, 0.2]
    model = ridge.fit_ridge(z, y, lam=1e-6)
    z_new = rng.normal(size=(5, 20))
    np.testing.assert_allclose(ridge.predict(model, z_new), z_new @ W + [0.1, -0.3, 0.2], atol=0.05)


def test_with_more_features_than_rows_ridge_fits_the_rows_it_saw():
    rng = np.random.default_rng(0)
    z, y = rng.normal(size=(60, 200)), rng.normal(size=(60, 2))
    np.testing.assert_allclose(ridge.predict(ridge.fit_ridge(z, y, lam=1e-6), z), y, atol=1e-4)


def test_a_large_penalty_predicts_the_training_mean():
    rng = np.random.default_rng(1)
    z, y = rng.normal(size=(40, 30)), rng.normal(size=(40, 7))
    model = ridge.fit_ridge(z, y, lam=1e9)
    np.testing.assert_allclose(ridge.predict(model, rng.normal(size=(3, 30))), np.tile(y.mean(0), (3, 1)), atol=1e-4)


def test_a_feature_constant_in_training_gets_no_weight():
    rng = np.random.default_rng(2)
    z = np.c_[rng.normal(size=(50, 5)), np.full((50, 3), 0.25)]  # the last 3 never vary
    y = z[:, :1] * [1.0, -1.0]
    model = ridge.fit_ridge(z, y, lam=1.0)
    np.testing.assert_array_equal(model[0][5:], 0.0)
    live = z[:4].copy()
    live[:, 5:] += 5e-4  # live float32 against rows cached as float16
    np.testing.assert_allclose(ridge.predict(model, live), ridge.predict(model, z[:4]))


def test_policy_frames_are_labelled_with_no_correction_and_corrections_weighted():
    import torch

    n, steps = 60, 4
    pi = torch.zeros(n, steps, 7)
    demo = torch.zeros(n, steps, 7)
    demo[:, :, 0] = 0.05  # the person would be 5 cm further everywhere
    intervention = torch.zeros(n)
    intervention[40:50] = 1.0  # a correction from frame 40
    tensors = {
        'cond': torch.zeros(n, 3),
        'state': torch.zeros(n, 2),
        'chunk_pi': pi,
        'chunk_demo': demo,
        'intervention': intervention,
        'episode': torch.zeros(n, dtype=torch.int32),
        'frame': torch.arange(n, dtype=torch.int32),
    }
    z, y, w, corrected, untouched = data.problem(tensors, fps=10)
    assert z.shape == (n, 5)  # features and state
    np.testing.assert_allclose(y[40:50, 0], 0.05)
    np.testing.assert_allclose(y[:40], 0.0)  # the policy drove: no correction
    np.testing.assert_allclose(y[50:], 0.0)
    assert (w[40:50] == data.WEIGHT).all() and (w[30:40] == data.WEIGHT).all()  # and the 1 s before
    assert (w[:30] == 1.0).all() and (w[50:] == 1.0).all()


def test_a_plan_running_into_a_correction_carries_the_nudge_of_its_corrected_steps():
    import torch

    n, steps = 60, 4
    demo = torch.zeros(n, steps, 7)
    demo[:, :, 0] = 0.05  # the person would be 5 cm further everywhere
    intervention = torch.zeros(n)
    intervention[40:50] = 1.0  # a correction from frame 40 to 49
    k = torch.clamp(torch.arange(n)[:, None] + torch.arange(steps), max=n - 1)
    tensors = {
        'cond': torch.zeros(n, 3),
        'chunk_pi': torch.zeros(n, steps, 7),
        'chunk_demo': demo,
        'intervention': intervention,
        'step_intervention': intervention[k],
        'episode': torch.zeros(n, dtype=torch.int32),
        'frame': torch.arange(n, dtype=torch.int32),
    }
    z, y, w, corrected, untouched = data.problem(tensors)
    # frames 37, 38, 39 plan 1, 2, 3 of their 4 steps into the correction; 36 none of them
    np.testing.assert_allclose(y[36:41, 0], [0.0, 0.0125, 0.025, 0.0375, 0.05])
    np.testing.assert_allclose(y[46:51, 0], [0.05, 0.0375, 0.025, 0.0125, 0.0])  # and out of it
    assert (w[37:50] == data.WEIGHT).all() and (w[:37] == 1.0).all() and (w[50:] == 1.0).all()
    assert corrected[40:50].all() and not corrected[37:40].any()  # scored as corrections: the frame's own
    assert untouched[:37].all() and not untouched[37:50].any() and untouched[50:].all()


def test_weighted_ridge_follows_the_heavier_rows():
    rng = np.random.default_rng(3)
    z = rng.normal(size=(200, 4))
    y = np.where(np.arange(200)[:, None] < 100, 1.0, -1.0) * np.ones((200, 1))
    w = np.where(np.arange(200) < 100, 9.0, 1.0)
    model = ridge.fit_ridge(z, y, lam=1e6, w=w)  # all penalty: the weighted mean
    np.testing.assert_allclose(model[1], [0.8])
