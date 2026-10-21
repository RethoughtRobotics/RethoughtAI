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
