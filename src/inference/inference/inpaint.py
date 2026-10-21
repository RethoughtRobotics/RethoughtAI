"""Chunks that continue the chunk being played: inpainting during a diffusion policy's denoising.

  Black, Galliker, Levine, "Real-Time Execution of Action Chunking Flow Policies", 2025 (RTC): the
  steps of a new chunk that the robot plays while it is computed are frozen to the current chunk, and
  the rest of the overlap is pulled toward it with weights decaying to zero (their soft mask, eq. 5).
  Lugmayr et al., "RePaint: Inpainting using Denoising Diffusion Probabilistic Models", CVPR 2022: a
  known part of a sample is imposed at every denoising step by replacing it with the known values
  noised to that step's level.

RTC steers a flow policy with a gradient; the MultiTask DiT is a DDPM, for which replacement needs no
gradient and no retraining. A new chunk generated from its own observation disagreed with the chunk it
replaced by 18 to 53 mm and 5 to 16 deg over their overlap (SweepTrash on the robot); here the part
already decided is kept and the policy fills in what follows.
"""

import numpy as np


def soft_mask(length, frozen, overlap):
    """(length,) weights: 1 over the first `frozen` steps, decaying to 0 at `overlap` (RTC eq. 5), 0 after."""
    k = np.arange(length, dtype=float)
    c = np.clip((overlap - k) / max(overlap - frozen + 1, 1), 0.0, 1.0)
    w = c * np.expm1(c) / np.expm1(1.0)
    w[: min(frozen, length)] = 1.0
    w[int(max(overlap, 0)) :] = 0.0
    return w


def sample(objective, model, conditioning, known, weights):
    """A DDPM sample (batch 1) of `objective` (LeRobot's DiffusionObjective) whose first len(weights)
    steps are pulled toward `known` (P, action_dim, normalized) by `weights` (P,) at every denoising step."""
    import torch

    scheduler = objective.noise_scheduler
    parameter = next(model.parameters())
    x = torch.randn((1, objective.horizon, objective.action_dim), dtype=parameter.dtype, device=parameter.device)
    known = torch.as_tensor(known, dtype=x.dtype, device=x.device)[None]
    w = torch.as_tensor(weights, dtype=x.dtype, device=x.device)[None, :, None]
    p = known.shape[1]
    scheduler.set_timesteps(objective.num_inference_steps)
    steps = list(scheduler.timesteps)
    for i, t in enumerate(steps):
        output = model(x, torch.full((1,), t, dtype=torch.long, device=x.device), conditioning_vec=conditioning)
        x = scheduler.step(output, t, x).prev_sample
        if i + 1 < len(steps):  # known part at the noise level x now has
            level = torch.full((1,), steps[i + 1], dtype=torch.long, device=x.device)
            target = scheduler.add_noise(known, torch.randn_like(known), level)
        else:
            target = known
        x[:, :p] = w * target + (1 - w) * x[:, :p]
    return x
