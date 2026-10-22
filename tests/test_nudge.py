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


def adapter(tmp_path, checkpoint, W, b):
    return nudge.save(tmp_path / 'adapter', W, b, np.zeros(len(W)), np.ones(len(W)), checkpoint, fit={'method': 'test'})


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
