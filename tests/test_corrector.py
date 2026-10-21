"""ttt.corrections: the corrector's labels and its linear fit."""

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

pytest.importorskip('torch')
from ttt import corrections  # noqa: E402


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
    np.testing.assert_allclose(corrections.nudge(pi, demo)[0], [0.02, 0, 0, 0, 0, 0, 0])


def test_ridge_recovers_a_linear_map():
    rng = np.random.default_rng(0)
    z = rng.normal(size=(300, 20))
    W = rng.normal(size=(20, 3))
    y = z @ W + [0.1, -0.3, 0.2]
    model = corrections.fit_ridge(z, y, lam=1e-6)
    z_new = rng.normal(size=(5, 20))
    np.testing.assert_allclose(corrections.predict(model, z_new), z_new @ W + [0.1, -0.3, 0.2], atol=0.05)


def test_with_more_features_than_rows_ridge_fits_the_rows_it_saw():
    rng = np.random.default_rng(0)
    z, y = rng.normal(size=(60, 200)), rng.normal(size=(60, 2))
    np.testing.assert_allclose(corrections.predict(corrections.fit_ridge(z, y, lam=1e-6), z), y, atol=1e-4)


def test_a_large_penalty_predicts_the_training_mean():
    rng = np.random.default_rng(1)
    z, y = rng.normal(size=(40, 30)), rng.normal(size=(40, 7))
    model = corrections.fit_ridge(z, y, lam=1e9)
    np.testing.assert_allclose(
        corrections.predict(model, rng.normal(size=(3, 30))), np.tile(y.mean(0), (3, 1)), atol=1e-4
    )
