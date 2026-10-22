"""ttt.correctors: every method, fitted, exported and loaded as the robot loads it, predicts what it
predicted when fitted; the MLP learns what a linear map cannot."""

import types

import numpy as np
import pytest

pytest.importorskip('torch')
from inference import nudge  # noqa: E402
from ttt.correctors import METHODS, mlp, ridge  # noqa: E402


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


def served(tmp_path, checkpoint, corrector, outputs, z):
    """The offsets the robot's nudge adapter gives for the rows z, from the adapter `corrector` wrote."""
    description, arrays = corrector.export(outputs)
    path = nudge.save(tmp_path / corrector.name, description, arrays, checkpoint)
    out = []
    for row in z:
        encoder = types.SimpleNamespace(encode=lambda batch: batch)
        loaded = nudge.Nudge(path, types.SimpleNamespace(observation_encoder=encoder), checkpoint, 1.0)
        loaded.z = row  # what the policy's encoder gave for the frame
        loaded(np.zeros((2, 7)))
        out.append(loaded.offset)
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
