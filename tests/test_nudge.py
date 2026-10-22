"""inference.nudge: a linear corrector's offset on every step of the frozen policy's plan."""

import types

import numpy as np
import pytest

pytest.importorskip('safetensors')
from inference import nudge  # noqa: E402


class Encoder:
    """Stands in for the policy's observation encoder: encode() returns the batch's vector."""

    def encode(self, batch):
        return batch


class Vector:
    def __init__(self, z):
        self.z = np.asarray([z], np.float64)

    def __getitem__(self, i):
        return types.SimpleNamespace(
            float=lambda: types.SimpleNamespace(cpu=lambda: types.SimpleNamespace(numpy=lambda: self.z[i]))
        )


@pytest.fixture
def checkpoint(tmp_path):
    (tmp_path / 'ckpt').mkdir()
    (tmp_path / 'ckpt' / 'model.safetensors').write_bytes(b'weights')
    return tmp_path / 'ckpt'


def linear(tmp_path, checkpoint, W, b, outputs=6, state=0):
    """A linear nudge adapter of W (inputs, 7) and b (7,), inputs unscaled, its first `outputs`."""
    arrays = {'W': W[:, :outputs], 'b': b[:outputs], 'mean': np.zeros(len(W)), 'std': np.ones(len(W))}
    return nudge.save(tmp_path / 'adapter', {'type': 'linear'}, arrays, checkpoint, state=state, fit={'method': 'test'})


def adapter(tmp_path, checkpoint, W, b):
    return linear(tmp_path, checkpoint, W, b)


def loaded(path, checkpoint, alpha=1.0, z=(1.0, 0.0)):
    policy = types.SimpleNamespace(observation_encoder=Encoder())
    n = nudge.Nudge(path, policy, checkpoint, alpha)
    policy.observation_encoder.encode(Vector(z))  # the policy encodes the frame while it plans
    return n


def test_every_step_is_offset_and_the_gripper_left_alone(tmp_path, checkpoint):
    W = np.zeros((2, 7))
    W[0, :3] = [0.02, -0.03, 0.0]
    path = adapter(tmp_path, checkpoint, W, np.zeros(7))
    plan = np.zeros((4, 7))
    plan[:, 6] = 1.0
    out = loaded(path, checkpoint)(plan)
    np.testing.assert_allclose(out[:, :3], np.tile([0.02, -0.03, 0.0], (4, 1)))
    np.testing.assert_allclose(out[:, 6], 1.0)


def test_alpha_zero_is_the_policy_exactly(tmp_path, checkpoint):
    path = adapter(tmp_path, checkpoint, np.ones((2, 7)), np.ones(7))
    plan = np.random.default_rng(0).normal(size=(5, 7))
    np.testing.assert_array_equal(loaded(path, checkpoint, alpha=0.0)(plan), plan)


def test_the_offset_is_not_capped(tmp_path, checkpoint):
    W = np.zeros((2, 7))
    W[0, 0] = 1.0  # 1 m for z = (1, 0)
    path = adapter(tmp_path, checkpoint, W, np.zeros(7))
    n = loaded(path, checkpoint, alpha=0.5)
    n(np.zeros((3, 7)))
    np.testing.assert_allclose(n.offset, [0.5, 0, 0, 0, 0, 0])


def test_another_checkpoint_is_refused(tmp_path, checkpoint):
    path = adapter(tmp_path, checkpoint, np.zeros((2, 7)), np.zeros(7))
    other = tmp_path / 'other'
    other.mkdir()
    (other / 'model.safetensors').write_bytes(b'other weights')
    with pytest.raises(ValueError, match='fitted on'):
        loaded(path, other)


def test_an_adapter_fitted_with_the_state_reads_it(tmp_path, checkpoint):
    W = np.zeros((3, 7))  # z (2) then the state (1)
    W[2, 1] = 0.1
    path = linear(tmp_path, checkpoint, W, np.zeros(7), state=1)
    n = loaded(path, checkpoint)
    out = n(np.zeros((2, 7)), state=[0.5])
    np.testing.assert_allclose(out[:, 1], 0.05)
    with pytest.raises(ValueError, match='observation.state'):
        n(np.zeros((2, 7)))


def test_saved_weights_load_back_exactly(tmp_path, checkpoint):
    from safetensors.numpy import load_file

    W = np.random.default_rng(4).normal(size=(5, 7))
    path = linear(tmp_path, checkpoint, W, np.arange(7.0))  # W[:, :6]: a view, not contiguous in memory
    w = load_file(str(path / nudge.WEIGHTS))
    np.testing.assert_array_equal(w['W'], W[:, :6].astype(np.float32))
    np.testing.assert_array_equal(w['b'], np.arange(6.0, dtype=np.float32))


def test_a_gripper_adapter_offsets_the_trigger_within_its_range(tmp_path, checkpoint):
    W = np.zeros((2, 7))
    W[0, 6] = 0.4  # close by 0.4 for z = (1, 0)
    path = linear(tmp_path, checkpoint, W, np.zeros(7), outputs=7)
    plan = np.zeros((3, 7))
    plan[:, 6] = [0.0, 0.5, 0.9]
    out = loaded(path, checkpoint)(plan)
    np.testing.assert_allclose(out[:, 6], [0.4, 0.9, 1.0])  # clipped at closed


def test_an_mlp_adapter_runs_its_layers_with_gelu_between(tmp_path, checkpoint):
    from scipy.special import erf

    rng = np.random.default_rng(5)
    W0, b0, W1, b1 = rng.normal(size=(4, 2)), rng.normal(size=4), rng.normal(size=(6, 4)), rng.normal(size=6)
    arrays = {'mean': np.zeros(2), 'std': np.ones(2), 'layer0.weight': W0, 'layer0.bias': b0}
    arrays |= {'layer1.weight': W1, 'layer1.bias': b1}
    model = {'type': 'mlp', 'layers': 2, 'activation': 'gelu'}
    path = nudge.save(tmp_path / 'adapter', model, arrays, checkpoint)
    n = loaded(path, checkpoint)  # z = (1, 0)
    n(np.zeros((2, 7)))
    h = W0 @ [1.0, 0.0] + b0
    expected = W1 @ (0.5 * h * (1 + erf(h / np.sqrt(2)))) + b1
    np.testing.assert_allclose(n.offset, expected, rtol=1e-5)
