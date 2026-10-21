import numpy as np
import pytest
import torch

from inference import inpaint


class Objective:
    """LeRobot's DiffusionObjective as inpaint.sample reads it: a DDPM, 10 steps of 100."""

    def __init__(self, horizon=8, action_dim=3):
        from diffusers.schedulers.scheduling_ddpm import DDPMScheduler

        self.noise_scheduler = DDPMScheduler(num_train_timesteps=100, clip_sample=True, prediction_type='epsilon')
        self.num_inference_steps, self.horizon, self.action_dim = 10, horizon, action_dim


class Model(torch.nn.Module):
    """Predicts noise as if every clean action were `level`."""

    def __init__(self, level=0.5):
        super().__init__()
        self.level, self.weight = level, torch.nn.Parameter(torch.zeros(1))

    def forward(self, x, t, conditioning_vec):
        alpha = Objective().noise_scheduler.alphas_cumprod[t].reshape(-1, 1, 1)
        return (x - alpha.sqrt() * self.level) / (1 - alpha).sqrt()


def test_soft_mask_frozen_then_decays_to_zero_at_the_overlap():
    w = inpaint.soft_mask(12, 3, 8)
    assert np.all(w[:3] == 1)
    assert np.all(np.diff(w[2:9]) <= 0) and 0 < w[3] < 1
    assert np.all(w[8:] == 0)


def test_full_weight_keeps_the_known_steps_and_zero_weight_changes_nothing():
    known = torch.full((4, 3), -0.25)
    torch.manual_seed(0)
    x = inpaint.sample(Objective(), Model(), None, known, np.r_[1.0, 1.0, 0.0, 0.0])
    torch.manual_seed(0)
    free = inpaint.sample(Objective(), Model(), None, known, np.zeros(4))
    assert torch.allclose(x[0, :2], known[:2])
    assert torch.allclose(free[0], torch.full((8, 3), 0.5), atol=1e-4)
    assert x[0, 4:].sub(0.5).abs().max() < 1e-4  # beyond the known steps: the policy's own


@pytest.mark.parametrize('frozen', [0, 5])
def test_soft_mask_lengths(frozen):
    assert inpaint.soft_mask(5, frozen, 5).shape == (5,)


class Latency:
    def __init__(self, s):
        self.s = s

    def __len__(self):
        return 1

    def percentile(self, _):
        return self.s


def test_a_single_arm_chunk_continues_the_last_one_from_its_observation_time():
    """PolicyNode._known: the last chunk's raw absolute actions from t on, qy >= 0, frozen over the delay."""
    from types import SimpleNamespace

    from scipy.spatial.transform import Rotation

    from inference.policy_node import PolicyNode
    from inference.runtime import Chunk

    dt, n = 0.04, 31
    positions = np.c_[0.01 * np.arange(n), np.zeros(n), np.zeros(n)]
    rotations = Rotation.from_quat(np.tile([0.0, -1.0, 0.0, 0.0], (n, 1)))  # qy < 0: the sign flips
    last = Chunk(10.0 + dt * np.arange(n), positions, rotations, np.linspace(0, 1, n), 10.0)
    node = SimpleNamespace(last=last, dt=dt, fps=25.0, latency=Latency(0.1))
    known, weights = PolicyNode._known(node, 10.0 + 24 * dt)
    assert known.shape == (7, 8)  # steps 24 .. 30 of the last chunk
    assert np.allclose(known[:, 0], 0.01 * np.arange(24, 31))
    assert np.all(known[:, 4] > 0)
    assert np.allclose(known[:, 7], np.linspace(0, 1, n)[24:])
    assert np.all(weights[:4] == 1) and weights[-1] < 1  # 0.1 s is 3 steps, plus the step it is published in
    assert PolicyNode._known(node, 10.0 + 31 * dt) is None  # the last chunk has run out
