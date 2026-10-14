#!/usr/bin/env python

# Copyright 2025 The VITA authors (Gao et al., ICLR 2026) and The HuggingFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from dataclasses import dataclass, field

from lerobot.configs import NormalizationMode, PreTrainedConfig
from lerobot.optim import AdamConfig, DiffuserSchedulerConfig


@PreTrainedConfig.register_subclass("vita")
@dataclass
class VitaConfig(PreTrainedConfig):
    """Configuration for VITA: Vision-to-Action Flow Matching Policy (Gao et al., ICLR 2026).

    Port of https://github.com/ucd-dare/VITA (`flare/configs/policy/vita.yaml` and
    `default_policy.yaml`). Instead of flowing from Gaussian noise, VITA flows from the visual
    latent (ResNet features + state, projected to `latent_dim`) to the latent of an action
    autoencoder, then decodes the action chunk.

    Defaults match upstream except `horizon` / `n_action_steps`, which upstream sets per task
    (16 / 8 for 8-20 fps simulators); pick them for your control rate.
    """

    # Inputs / outputs
    n_obs_steps: int = 1  # upstream `obs_horizon`
    horizon: int = 16  # upstream `pred_horizon`: action steps predicted per chunk
    n_action_steps: int = 8  # upstream `action_horizon`: steps executed per policy call

    normalization_mapping: dict[str, NormalizationMode] = field(
        default_factory=lambda: {
            # ImageNet normalization is applied inside the ResNet observer, as upstream does via
            # `override_stats`, so images only need to arrive in [0, 1].
            "VISUAL": NormalizationMode.IDENTITY,
            "STATE": NormalizationMode.MEAN_STD,
            "ACTION": NormalizationMode.MEAN_STD,
        }
    )

    # Observer: ResNet-18 (ImageNet weights, frozen BatchNorm), global-average-pooled per camera.
    vision_backbone: str = "resnet18"
    pretrained_backbone_weights: str | None = "ResNet18_Weights.IMAGENET1K_V1"
    resize_shape: tuple[int, int] = (240, 320)  # every camera is resized to this before stacking
    crop_shape: tuple[int, int] | None = (224, 308)  # random crop in training, center crop in eval

    # VITA objective
    latent_dim: int = 512
    decode_flow_latents: bool = True  # decode flowed latents during training (FLC + FLD losses)
    consistency_weight: float = 1.0  # FLC: MSE(flowed latent, encoded action latent)
    enc_contrastive_weight: float = 0.0
    flow_contrastive_weight: float = 0.0

    # Flow matching (upstream `flow_matcher`: exact OT-CFM)
    flow_matcher: str = "exact"  # "exact" (minibatch OT pairing) or "conditional" (index pairing)
    flow_sigma: float = 0.0
    num_sampling_steps: int = 6  # Euler steps from visual latent to action latent

    # Flow network (upstream `flow_net`: simple_flow_net)
    flow_hidden_dim: int = 512
    flow_num_layers: int = 4
    flow_mlp_ratio: float = 4.0
    flow_dropout: float = 0.0
    flow_time_embed_dim: int = 256

    # Action autoencoder (upstream `action_ae`: cnn encoder, simple MLP decoder, not variational)
    enc_recon_weight: float = 0.5  # AE: decode(encode(actions)) vs actions
    flow_recon_weight: float = 0.5  # FLD: decode(flowed latent) vs actions
    recon_loss_type: str = "l1"  # upstream: l1 outperforms l2
    ae_enc_hidden_dim: int = 512
    ae_dec_hidden_dim: int = 512
    ae_num_layers: int = 4
    ae_dropout: float = 0.0

    # Training presets (upstream default_policy.yaml)
    optimizer_lr: float = 1e-4
    optimizer_betas: tuple[float, float] = (0.95, 0.999)
    optimizer_eps: float = 1e-8
    optimizer_weight_decay: float = 1e-6
    scheduler_name: str = "cosine"
    scheduler_warmup_steps: int = 500
    # Upstream also trains with EMA (power 0.75) and grad-norm clipping at 1.0; those live in the
    # training loop, not the policy.

    def __post_init__(self):
        super().__post_init__()
        if not self.vision_backbone.startswith("resnet"):
            raise ValueError(f"`vision_backbone` must be a ResNet variant. Got {self.vision_backbone}.")
        if self.flow_matcher not in ("exact", "conditional"):
            raise ValueError(f"`flow_matcher` must be 'exact' or 'conditional'. Got {self.flow_matcher}.")
        if self.recon_loss_type not in ("l1", "l2"):
            raise ValueError(f"`recon_loss_type` must be 'l1' or 'l2'. Got {self.recon_loss_type}.")
        if self.n_action_steps > self.horizon - self.n_obs_steps + 1:
            raise ValueError(
                f"`n_action_steps` ({self.n_action_steps}) must be <= horizon - n_obs_steps + 1 "
                f"({self.horizon - self.n_obs_steps + 1})."
            )
        # The CNN action encoder halves the sequence `ae_num_layers` times.
        if self.horizon % (2**self.ae_num_layers) != 0:
            raise ValueError(
                f"`horizon` ({self.horizon}) must be divisible by 2**ae_num_layers ({2**self.ae_num_layers})."
            )
        if self.crop_shape is not None and (
            self.crop_shape[0] > self.resize_shape[0] or self.crop_shape[1] > self.resize_shape[1]
        ):
            raise ValueError(f"`crop_shape` {self.crop_shape} must fit in `resize_shape` {self.resize_shape}.")

    def get_optimizer_preset(self) -> AdamConfig:
        return AdamConfig(
            lr=self.optimizer_lr,
            betas=self.optimizer_betas,
            eps=self.optimizer_eps,
            weight_decay=self.optimizer_weight_decay,
        )

    def get_scheduler_preset(self) -> DiffuserSchedulerConfig:
        return DiffuserSchedulerConfig(name=self.scheduler_name, num_warmup_steps=self.scheduler_warmup_steps)

    def validate_features(self) -> None:
        if len(self.image_features) == 0:
            raise ValueError("VITA flows from visual latents and needs at least one image input.")
        if self.robot_state_feature is None:
            raise ValueError("VITA requires `observation.state` as an input.")

    @property
    def observation_delta_indices(self) -> list:
        return list(range(1 - self.n_obs_steps, 1))

    @property
    def action_delta_indices(self) -> list:
        return list(range(1 - self.n_obs_steps, 1 - self.n_obs_steps + self.horizon))

    @property
    def reward_delta_indices(self) -> None:
        return None
