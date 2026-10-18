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
"""VITA: Vision-to-Action Flow Matching Policy (Gao et al., ICLR 2026).

Port of https://github.com/ucd-dare/VITA (`flare/policies/vita/vita_policy.py` and the modules it
uses). VITA flows directly from a visual latent to an action latent (no Gaussian noise, no
cross-attention conditioning) with a small MLP flow network, and decodes the flowed latent into
an action chunk with an action autoencoder's decoder.

Differences from upstream, all behaviour-preserving for training:
  - torchcfm's exact OT-CFM is re-implemented here (assignment via scipy), so neither torchcfm
    nor POT is needed. With uniform batch weights the exact OT plan is a permutation, which is
    what scipy's `linear_sum_assignment` returns; pairs are then sampled from it with
    replacement as torchcfm does.
  - timm's `Mlp` is inlined.
  - Each camera is resized separately before stacking, so cameras may have different sizes.
  - Eval uses a center crop (upstream crops randomly in eval too).
"""

import math
from collections import deque

import einops
import numpy as np
import torch
import torch.nn.functional as F  # noqa: N812
import torchvision
from scipy.optimize import linear_sum_assignment
from torch import Tensor, nn
from torchvision.models._utils import IntermediateLayerGetter
from torchvision.ops.misc import FrozenBatchNorm2d

from lerobot.utils.constants import ACTION, OBS_STATE

from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.policies.utils import populate_queues
from .configuration_vita import VitaConfig

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class VitaPolicy(PreTrainedPolicy):
    config_class = VitaConfig
    name = "vita"

    def __init__(self, config: VitaConfig, **kwargs):
        super().__init__(config)
        config.validate_features()
        self.config = config
        self.vita = VitaModel(config)
        self._queues: dict[str, deque[Tensor]] = {}
        self.reset()

    def get_optim_params(self) -> dict:
        return self.vita.parameters()

    def reset(self):
        """Clear observation and action queues. Call on `env.reset()`."""
        self._queues = {key: deque(maxlen=self.config.n_obs_steps) for key in self._obs_keys}
        self._queues[ACTION] = deque(maxlen=self.config.n_action_steps)

    @property
    def _obs_keys(self) -> list[str]:
        return [OBS_STATE, *self.config.image_features]

    def _with_time_dim(self, batch: dict[str, Tensor]) -> dict[str, Tensor]:
        """Give state (B, D) and images (B, C, H, W) a time dimension when n_obs_steps == 1."""
        batch = dict(batch)
        if batch[OBS_STATE].ndim == 2:
            batch[OBS_STATE] = batch[OBS_STATE].unsqueeze(1)
        for key in self.config.image_features:
            if batch[key].ndim == 4:
                batch[key] = batch[key].unsqueeze(1)
        return batch

    @torch.no_grad()
    def predict_action_chunk(self, batch: dict[str, Tensor], **kwargs) -> Tensor:
        """Actions to execute, (B, n_action_steps, action_dim), starting at the current step.

        Uses the observation queues when populated by `select_action`, else the batch directly.
        """
        if any(len(self._queues[k]) > 0 for k in self._obs_keys):
            batch = {k: torch.stack(list(self._queues[k]), dim=1) for k in self._obs_keys}
        batch = self._with_time_dim(batch)
        actions = self.vita.generate_actions(batch)  # (B, horizon, action_dim)
        start = self.config.n_obs_steps - 1
        return actions[:, start : start + self.config.n_action_steps]

    @torch.no_grad()
    def select_action(self, batch: dict[str, Tensor], **kwargs) -> Tensor:
        batch = {k: v for k, v in batch.items() if k != ACTION}
        self._queues = populate_queues(self._queues, batch)
        if len(self._queues[ACTION]) == 0:
            actions = self.predict_action_chunk(batch)
            self._queues[ACTION].extend(actions.transpose(0, 1))
        return self._queues[ACTION].popleft()

    def forward(self, batch: dict[str, Tensor]) -> tuple[Tensor, dict]:
        batch = self._with_time_dim(batch)
        return self.vita.compute_loss(batch)


class VitaModel(nn.Module):
    def __init__(self, config: VitaConfig):
        super().__init__()
        self.config = config
        self.image_keys = list(config.image_features)
        state_dim = config.robot_state_feature.shape[0]
        self.action_dim = config.action_feature.shape[0]

        self.observer = ResNetObserver(config, self.image_keys)
        obs_dim = config.n_obs_steps * (len(self.image_keys) * self.observer.feature_dim + state_dim)
        self.obs_encoder = nn.Linear(obs_dim, config.latent_dim)

        self.action_encoder = CNNActionEncoder(
            pred_horizon=config.horizon,
            action_dim=self.action_dim,
            latent_dim=config.latent_dim,
            hidden_dim=config.ae_enc_hidden_dim,
            num_layers=config.ae_num_layers,
        )
        self.action_decoder = SimpleActionDecoder(
            dec_hidden_dim=config.ae_dec_hidden_dim,
            latent_dim=config.latent_dim,
            pred_horizon=config.horizon,
            action_dim=self.action_dim,
            num_layers=config.ae_num_layers,
            dropout=config.ae_dropout,
        )
        self.flow_net = SimpleFlowNet(
            input_dim=config.latent_dim,
            hidden_dim=config.flow_hidden_dim,
            output_dim=config.latent_dim,
            num_layers=config.flow_num_layers,
            mlp_ratio=config.flow_mlp_ratio,
            dropout=config.flow_dropout,
            time_embed_dim=config.flow_time_embed_dim,
        )
        self.flow_matcher = FlowMatcher(
            exact_ot=config.flow_matcher == "exact",
            sigma=config.flow_sigma,
            num_sampling_steps=config.num_sampling_steps,
        )
        self.recon_loss_fn = F.l1_loss if config.recon_loss_type == "l1" else F.mse_loss

    def encode_observations(self, batch: dict[str, Tensor]) -> Tensor:
        return self.obs_encoder(self.observer(batch))

    def generate_actions(self, batch: dict[str, Tensor]) -> Tensor:
        obs_latents = self.encode_observations(batch)
        action_latents = self.flow_matcher.sample(self.flow_net, start=obs_latents)
        return self.action_decoder(action_latents)

    def compute_loss(self, batch: dict[str, Tensor]) -> tuple[Tensor, dict]:
        cfg = self.config
        obs_latents = self.encode_observations(batch)
        gt_actions = batch[ACTION]
        batch_size = gt_actions.shape[0]
        action_latents = self.action_encoder(gt_actions)

        # Flow matching: visual latents (source) -> action latents (target).
        flow_loss = self.flow_matcher.compute_loss(self.flow_net, target=action_latents, start=obs_latents)
        loss = flow_loss
        metrics = {"flow_loss": flow_loss.item()}

        if cfg.enc_contrastive_weight > 0:
            c_loss = contrastive_loss(obs_latents.view(batch_size, -1), action_latents.view(batch_size, -1))
            loss = loss + cfg.enc_contrastive_weight * c_loss
            metrics["enc_contrastive_loss"] = c_loss.item()

        if cfg.decode_flow_latents:
            action_latents_pred = self.flow_matcher.sample(self.flow_net, start=obs_latents)
            if cfg.consistency_weight > 0:  # FLC
                consistency = F.mse_loss(action_latents_pred, action_latents)
                loss = loss + cfg.consistency_weight * consistency
                metrics["consistency_loss"] = consistency.item()
            if cfg.flow_contrastive_weight > 0:
                c_loss = contrastive_loss(
                    obs_latents.view(batch_size, -1), action_latents_pred.view(batch_size, -1)
                )
                loss = loss + cfg.flow_contrastive_weight * c_loss
                metrics["flow_contrastive_loss"] = c_loss.item()
            if cfg.flow_recon_weight > 0:  # FLD
                flow_recon = self.recon_loss_fn(self.action_decoder(action_latents_pred), gt_actions)
                loss = loss + cfg.flow_recon_weight * flow_recon
                metrics["flow_action_recon_loss"] = flow_recon.item()

        if cfg.enc_recon_weight > 0:  # AE
            enc_recon = self.recon_loss_fn(self.action_decoder(action_latents), gt_actions)
            loss = loss + cfg.enc_recon_weight * enc_recon
            metrics["enc_action_recon_loss"] = enc_recon.item()

        return loss, metrics


class ResNetObserver(nn.Module):
    """ResNet features per camera (global average pool) concatenated with the state, flattened over time."""

    def __init__(self, config: VitaConfig, image_keys: list[str]):
        super().__init__()
        self.config = config
        self.image_keys = image_keys
        weights = config.pretrained_backbone_weights
        backbone = getattr(torchvision.models, config.vision_backbone)(
            weights=weights, norm_layer=FrozenBatchNorm2d
        )
        self.feature_dim = backbone.fc.in_features
        self.backbone = IntermediateLayerGetter(backbone, return_layers={"layer4": "feature_map"})
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.register_buffer("img_mean", torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1), persistent=False)
        self.register_buffer("img_std", torch.tensor(IMAGENET_STD).view(1, 3, 1, 1), persistent=False)

    def _images(self, batch: dict[str, Tensor]) -> Tensor:
        """(B, S, C, H, W) per camera -> (B*S*N, C, h, w), resized, cropped and ImageNet-normalized."""
        cams = []
        for key in self.image_keys:
            x = einops.rearrange(batch[key], "b s c h w -> (b s) c h w")
            if x.shape[-2:] != tuple(self.config.resize_shape):
                x = F.interpolate(x, size=tuple(self.config.resize_shape), mode="bilinear", align_corners=False)
            cams.append(x)
        images = einops.rearrange(torch.stack(cams, dim=1), "bs n c h w -> (bs n) c h w")

        if self.config.crop_shape is not None:
            ch, cw = self.config.crop_shape
            h, w = images.shape[-2:]
            if self.training:  # one random offset per batch, as upstream
                top = torch.randint(0, h - ch + 1, (1,)).item()
                left = torch.randint(0, w - cw + 1, (1,)).item()
            else:
                top, left = (h - ch) // 2, (w - cw) // 2
            images = images[..., top : top + ch, left : left + cw]
        return (images - self.img_mean) / self.img_std

    def forward(self, batch: dict[str, Tensor]) -> Tensor:
        state = batch[OBS_STATE]
        b, s = state.shape[:2]
        n = len(self.image_keys)
        img_feats = self.pool(self.backbone(self._images(batch))["feature_map"])
        img_feats = einops.rearrange(img_feats, "(b s n) c h w -> b (s n c h w)", b=b, s=s, n=n)
        return torch.cat([state.flatten(start_dim=1), img_feats], dim=1)


class FlowMatcher:
    """Conditional flow matching from a given source (the visual latent), optionally with exact
    minibatch optimal-transport pairing (torchcfm's ExactOptimalTransportConditionalFlowMatcher)."""

    def __init__(self, exact_ot: bool = True, sigma: float = 0.0, num_sampling_steps: int = 6):
        self.exact_ot = exact_ot
        self.sigma = sigma
        self.num_sampling_steps = num_sampling_steps

    @torch.no_grad()
    def _ot_pairs(self, x0: Tensor, x1: Tensor) -> tuple[Tensor, Tensor]:
        cost = torch.cdist(x0.flatten(1).float(), x1.flatten(1).float()).pow(2).cpu().numpy()
        rows, cols = linear_sum_assignment(cost)
        plan = np.zeros_like(cost)
        plan[rows, cols] = 1.0 / len(rows)
        choices = np.random.choice(plan.size, p=plan.ravel(), size=x0.shape[0], replace=True)
        i, j = np.divmod(choices, plan.shape[1])
        return torch.as_tensor(i, device=x0.device), torch.as_tensor(j, device=x1.device)

    def compute_loss(self, model: nn.Module, target: Tensor, start: Tensor) -> Tensor:
        x0, x1 = start, target
        if self.exact_ot:
            i, j = self._ot_pairs(x0, x1)
            x0, x1 = x0[i], x1[j]
        t = torch.rand(x0.shape[0], device=x0.device, dtype=x0.dtype)
        t_ = t.view(-1, *([1] * (x0.ndim - 1)))
        xt = t_ * x1 + (1 - t_) * x0
        if self.sigma > 0:
            xt = xt + self.sigma * torch.randn_like(xt)
        ut = x1 - x0
        vt = model(xt, t)
        return torch.mean((vt - ut) ** 2)

    def sample(self, model: nn.Module, start: Tensor, num_steps: int | None = None) -> Tensor:
        """Euler integration of the learned velocity field from `start` (t=0) to t=1."""
        num_steps = num_steps or self.num_sampling_steps
        x = start
        dt = 1.0 / num_steps
        for k in range(num_steps):
            t = torch.full((x.shape[0],), k / num_steps, device=x.device, dtype=x.dtype)
            x = x + model(x, t) * dt
        return x


def contrastive_loss(features_a: Tensor, features_b: Tensor, temperature: float = 0.07) -> Tensor:
    """Symmetric InfoNCE between matching rows of two feature sets."""
    a = F.normalize(features_a, dim=-1)
    b = F.normalize(features_b, dim=-1)
    logits = a @ b.t() / temperature
    labels = torch.arange(a.shape[0], device=a.device)
    return 0.5 * (F.cross_entropy(logits, labels) + F.cross_entropy(logits.t(), labels))


class Mlp(nn.Module):
    """timm.layers.Mlp: fc1 -> act -> dropout -> fc2 -> dropout."""

    def __init__(self, in_features, hidden_features, out_features=None, act_layer=nn.GELU, drop=0.0):
        super().__init__()
        out_features = out_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop)
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop2 = nn.Dropout(drop)

    def forward(self, x: Tensor) -> Tensor:
        return self.drop2(self.fc2(self.drop1(self.act(self.fc1(x)))))


class SinusoidalPosEmbed(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, x: Tensor) -> Tensor:
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=x.device) * -emb)
        emb = x[:, None] * emb[None, :]
        return torch.cat((emb.sin(), emb.cos()), dim=-1)


class FlowNetLayer(nn.Module):
    """Residual MLP block with adaLN-style timestep modulation (gate, scale, shift), zero-initialized."""

    def __init__(self, dim: int, mlp_ratio: float = 4.0, dropout: float = 0.0):
        super().__init__()
        self.dim = dim
        self.norm = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.mlp = Mlp(dim, int(dim * mlp_ratio), act_layer=lambda: nn.GELU(approximate="tanh"), drop=dropout)
        self.time_modulator = nn.Sequential(nn.SiLU(), nn.Linear(dim, 3 * dim))
        nn.init.constant_(self.time_modulator[-1].weight, 0)
        nn.init.constant_(self.time_modulator[-1].bias, 0)

    def forward(self, x: Tensor, t: Tensor) -> Tensor:
        gamma, scale, shift = self.time_modulator(t).view(x.shape[0], 3, self.dim).unbind(1)
        x_norm = self.norm(x) * (scale + 1) + shift
        return x + self.mlp(x_norm) * gamma


class SimpleFlowNet(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers, mlp_ratio=4.0, dropout=0.0,
                 time_embed_dim=256):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.time_embed = nn.Sequential(
            SinusoidalPosEmbed(time_embed_dim),
            nn.Linear(time_embed_dim, time_embed_dim * 4),
            nn.Mish(),
            nn.Linear(time_embed_dim * 4, hidden_dim),
        )
        self.layers = nn.ModuleList(
            [FlowNetLayer(hidden_dim, mlp_ratio=mlp_ratio, dropout=dropout) for _ in range(num_layers)]
        )
        self.norm = nn.LayerNorm(hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, output_dim)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        nn.init.normal_(self.time_embed[1].weight, std=0.02)
        nn.init.normal_(self.time_embed[3].weight, std=0.02)
        # Note: as upstream, this xavier pass also overrides each block's zero-initialized modulator.

    def forward(self, x: Tensor, t: Tensor) -> Tensor:
        x = self.input_proj(x)
        t = self.time_embed(t)
        for block in self.layers:
            x = block(x, t)
        return self.out_proj(self.norm(x))


class CNNActionEncoder(nn.Module):
    """Strided 1D convolutions over the action chunk, then a linear projection to the latent."""

    def __init__(self, pred_horizon: int, action_dim: int, latent_dim: int, hidden_dim: int = 128,
                 num_layers: int = 3):
        super().__init__()
        layers, in_dim = [], action_dim
        for _ in range(num_layers):
            layers += [nn.Conv1d(in_dim, hidden_dim, kernel_size=5, stride=2, padding=2), nn.ReLU()]
            in_dim = hidden_dim
        self.encoder = nn.Sequential(*layers)
        self.latent_proj = nn.Linear(hidden_dim * (pred_horizon // (2**num_layers)), latent_dim)
        for m in self.modules():
            if isinstance(m, (nn.Linear, nn.Conv1d)):
                nn.init.orthogonal_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.fill_(0.0)

    def forward(self, actions: Tensor) -> Tensor:
        x = self.encoder(actions.transpose(1, 2))  # (B, hidden, horizon / 2**num_layers)
        return self.latent_proj(x.flatten(start_dim=1))


class SimpleActionDecoder(nn.Module):
    """MLP from the latent to the full (horizon, action_dim) chunk."""

    def __init__(self, dec_hidden_dim: int, latent_dim: int, pred_horizon: int, action_dim: int,
                 num_layers: int = 4, dropout: float = 0.0):
        super().__init__()
        self.pred_horizon = pred_horizon
        self.action_dim = action_dim
        self.input_proj = nn.Linear(latent_dim, dec_hidden_dim)
        self.layers = nn.ModuleList(
            [Mlp(dec_hidden_dim, dec_hidden_dim, dec_hidden_dim, drop=dropout) for _ in range(num_layers)]
        )
        self.output_proj = nn.Linear(dec_hidden_dim, pred_horizon * action_dim)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, z: Tensor) -> Tensor:
        x = self.input_proj(z)
        for layer in self.layers:
            x = layer(x)
        return self.output_proj(x).view(-1, self.pred_horizon, self.action_dim)
