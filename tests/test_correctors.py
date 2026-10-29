"""ttt.correctors: every method, fitted, exported and loaded as the robot loads it, predicts what it
predicted when fitted; the MLP learns what a linear map cannot; the kernel memory keeps the guarantees its form gives."""

import types

import numpy as np
import pytest

pytest.importorskip('torch')
from inference import nudge  # noqa: E402
from ttt.correctors import METHODS, kernel, mlp, ridge  # noqa: E402


def rows(n=600, d=6, seed=0):
    rng = np.random.default_rng(seed)
    z = np.c_[rng.normal(size=(n, d)), np.full((n, 2), 0.3)]  # the last two features never vary
    y = np.zeros((n, 7))
    y[:, 0] = 0.05 * z[:, 0] - 0.02 * z[:, 1]
    y[:, 1] = 0.04 * np.sin(2 * z[:, 2])  # not linear
    return z, y, np.ones(n), np.arange(n) % 4


@pytest.fixture
def checkpoint(tmp_path):
    (tmp_path / 'ckpt').mkdir()
    (tmp_path / 'ckpt' / 'model.safetensors').write_bytes(b'weights')
    return tmp_path / 'ckpt'


def served(tmp_path, checkpoint, corrector, outputs, z, steps=0):
    """The offsets the robot's nudge adapter gives for the rows z (per step: (steps, outputs) each),
    from the adapter `corrector` wrote."""
    columns = [s * 7 + j for s in range(max(steps, 1)) for j in range(outputs)]
    description, arrays = corrector.export(columns)
    path = nudge.save(tmp_path / corrector.name, description, arrays, checkpoint, steps=steps)
    out = []
    for row in z:
        encoder = types.SimpleNamespace(encode=lambda batch: batch)
        loaded = nudge.Nudge(path, types.SimpleNamespace(observation_encoder=encoder), checkpoint, 1.0)
        loaded.z = row  # what the policy's encoder gave for the frame
        plan = loaded(np.zeros((max(steps, 2) + 1, 7)))
        out.append(plan[:steps] if steps else loaded.offset)
    return np.array(out)


@pytest.mark.parametrize('method', sorted(METHODS))
@pytest.mark.parametrize('outputs', [6, 7])
def test_the_served_adapter_predicts_what_the_fitted_corrector_does(tmp_path, checkpoint, method, outputs):
    z, y, w, groups = rows()
    options = {'steps': 300} if method == 'mlp' else {}
    corrector = METHODS[method](**options).fit(z, y, w, groups)
    live = z[:8].copy()
    live[:, -2:] += 5e-4  # constant features differ live by float16-sized rounding
    np.testing.assert_allclose(
        served(tmp_path, checkpoint, corrector, outputs, live), corrector.predict(z[:8])[:, :outputs], atol=1e-5
    )


def test_the_mlp_learns_what_a_linear_map_cannot():
    z, y, w, groups = rows(n=1500)
    zt, yt, _, _ = rows(n=300, seed=1)
    linear = ridge.Ridge().fit(z, y, w, groups)
    net = mlp.MLP(steps=2000).fit(z, y, w, groups)
    error = lambda c: np.abs(c.predict(zt)[:, 1] - yt[:, 1]).mean()  # noqa: E731
    assert error(net) < 0.5 * error(linear)


def test_mlp_forward_matches_pytorch():
    import torch

    net = torch.nn.Sequential(torch.nn.Linear(3, 5), torch.nn.GELU(), torch.nn.Linear(5, 2)).double()
    x = np.random.default_rng(2).normal(size=(4, 3))
    layers = [(m.weight.detach().numpy(), m.bias.detach().numpy()) for m in net if isinstance(m, torch.nn.Linear)]
    np.testing.assert_allclose(mlp.forward(layers, x), net(torch.as_tensor(x)).detach().numpy(), atol=1e-10)


@pytest.mark.parametrize('method', sorted(METHODS))
def test_a_per_step_adapter_serves_each_steps_fitted_offset(tmp_path, checkpoint, method):
    z, y, w, groups = rows()
    steps = 3
    y = np.concatenate([y * (k + 1) for k in range(steps)], axis=1)  # a different offset at each step
    options = {'steps': 300} if method == 'mlp' else {}
    corrector = METHODS[method](**options).fit(z, y, w, groups)
    got = served(tmp_path, checkpoint, corrector, 7, z[:5], steps=steps)
    want = corrector.predict(z[:5]).reshape(5, steps, 7)
    want[:, :, 6] = np.clip(want[:, :, 6], 0.0, 1.0)  # the served plan keeps the trigger within 0 .. 1
    np.testing.assert_allclose(got, want, atol=1e-5)


def test_a_per_step_target_keeps_an_up_then_down_that_the_plan_average_loses():
    import torch

    from ttt.correctors import data

    n, steps = 8, 4
    offset = torch.zeros(n, steps, 6)
    offset[:, :2, 2], offset[:, 2:, 2] = 0.03, -0.03  # up for half the plan, then down
    tensors = {
        'cond': torch.zeros(n, 3),
        'chunk_pi': torch.zeros(n, steps, 7),
        'chunk_demo': torch.zeros(n, steps, 7),
        'intervention': torch.ones(n),
        'step_intervention': torch.ones(n, steps),
        'offset': offset,
        'episode': torch.zeros(n, dtype=torch.int32),
        'frame': torch.arange(n, dtype=torch.int32),
    }
    plan = data.problem(tensors, target='plan')[1]
    step = data.problem(tensors, target='step')[1].reshape(n, steps, 7)
    np.testing.assert_allclose(plan[:, 2], 0.0, atol=1e-12)  # averaged over the plan: nothing
    np.testing.assert_allclose(step[0, :, 2], [0.03, 0.03, -0.03, -0.03])


def test_the_kernel_memory_is_zero_beyond_its_radius_bounded_and_continuous_at_its_edge():
    keys = np.eye(4)[:3]  # the fourth direction holds no stored key
    values = np.array([[0.02, 0.0], [0.0, -0.05], [0.01, 0.01]])
    s_min = 0.8

    def at(s):  # a unit query whose cosine similarity to the first key is s, 0 to the others
        return kernel.recall(keys, values, s_min, np.array([[s, 0.0, 0.0, np.sqrt(1 - s * s)]]))[0]

    assert np.all(at(0.8) == 0.0) and np.all(at(0.5) == 0.0)  # exactly 0: the base's plan
    np.testing.assert_allclose(at(1.0), values[0])  # on a stored key: its own nudge
    rng = np.random.default_rng(3)
    x = kernel.unit(rng.normal(size=(500, 4)), np.ones(4, bool))
    assert np.all(np.abs(kernel.recall(keys, values, s_min, x)) <= np.abs(values).max(axis=0) + 1e-12)
    assert np.abs(at(0.8 + 1e-4)).max() < 1e-9  # no jump at the edge, as dividing by sum phi alone would


def test_weighted_frame_keeps_only_the_nudged_frames_not_the_lead_up_or_the_decaying_handback():
    import torch

    from ttt.correctors import data

    n, steps = 10, 4
    nudging = torch.zeros(n)
    nudging[3:6] = 1.0  # the person nudged frames 3 to 5
    offset = torch.zeros(n, steps, 6)
    offset[1:6, :, 0] = 0.02  # the plans from frame 1 run into the nudge
    offset[6:, :, 0] = 0.02 * 0.5 ** torch.arange(1, n - 5).reshape(-1, 1)  # released: decaying, never 0
    tensors = {
        'cond': torch.zeros(n, 3),
        'chunk_pi': torch.zeros(n, steps, 7),
        'chunk_demo': torch.zeros(n, steps, 7),
        'intervention': nudging,
        'step_intervention': torch.zeros(n, steps),
        'offset': offset,
        'episode': torch.zeros(n, dtype=torch.int32),
        'frame': torch.arange(n, dtype=torch.int32),
    }
    w_plan = data.problem(tensors, weighted='plan')[2]
    w_frame = data.problem(tensors, weighted='frame')[2]
    assert np.flatnonzero(w_plan == data.WEIGHT).tolist() == list(range(1, n))
    assert np.flatnonzero(w_frame == data.WEIGHT).tolist() == [3, 4, 5]
