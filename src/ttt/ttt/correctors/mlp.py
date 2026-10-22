"""MLP: the nudge as CR-DAgger's residual head (Xu et al. 2025, github.com/yifan-hou/cr-dagger,
PyriteML/diffusion_policy/model/residual/mlp.py and config/train_online_residual_mlp_workspace.yaml):
three hidden layers of 256, GELU, dropout 0.1, mean squared error on normalised targets, AdamW at
3e-4 with a cosine schedule and an exponential moving average of the weights. Its settings are fixed,
not chosen on held-out episodes. Rows are weighted in the loss.

Exported for the nudge adapter with the input scaling separate (mean, std) and the target scaling
folded into the last layer, so the runtime only runs the layers.
"""

import numpy as np

from .data import Scaler, weighted_mean


class MLP:
    name = 'mlp'

    def __init__(
        self,
        hidden=(256, 256, 256),
        dropout=0.1,
        steps=3000,
        batch=256,
        lr=3e-4,
        weight_decay=1e-6,
        warmup=300,
        ema=0.999,
        seed=0,
        device='cpu',
    ):
        self.hidden, self.dropout, self.steps, self.batch = tuple(hidden), dropout, steps, batch
        self.lr, self.weight_decay, self.warmup, self.ema, self.seed = lr, weight_decay, warmup, ema, seed
        self.device = device
        self.layers = self.scaler = self.loss = None

    def _net(self, d, k):
        import torch
        from torch import nn

        torch.manual_seed(self.seed)
        mods, width = [], d
        for h in self.hidden:
            mods += [nn.Linear(width, h), nn.GELU(), nn.Dropout(self.dropout)]
            width = h
        mods.append(nn.Linear(width, k))
        return nn.Sequential(*mods)

    def fit(self, z, y, w, groups=None):
        """Train on all rows (`groups` unused: nothing is chosen on held-out episodes)."""
        import copy
        import math

        import torch

        self.scaler = Scaler(z)
        x = torch.as_tensor(self.scaler(z), dtype=torch.float32)
        self.y_mean = weighted_mean(y, w)
        self.y_std = np.sqrt(weighted_mean((y - self.y_mean) ** 2, w)) + 1e-6
        t = torch.as_tensor((y - self.y_mean) / self.y_std, dtype=torch.float32)
        wt = torch.as_tensor(w, dtype=torch.float32)
        net = self._net(x.shape[1], y.shape[1]).to(self.device)
        avg = copy.deepcopy(net).eval()
        opt = torch.optim.AdamW(net.parameters(), lr=self.lr, betas=(0.95, 0.999), weight_decay=self.weight_decay)
        schedule = torch.optim.lr_scheduler.LambdaLR(
            opt,
            lambda s: (s + 1) / self.warmup
            if s < self.warmup
            else 0.5 * (1 + math.cos(math.pi * (s - self.warmup) / max(1, self.steps - self.warmup))),
        )
        g = torch.Generator().manual_seed(self.seed)
        x, t, wt = x.to(self.device), t.to(self.device), wt.to(self.device)
        net.train()
        for _ in range(self.steps):
            i = torch.randint(len(x), (self.batch,), generator=g).to(self.device)
            per_row = ((net(x[i]) - t[i]) ** 2).mean(1)
            loss = (wt[i] * per_row).sum() / wt[i].sum()
            opt.zero_grad()
            loss.backward()
            opt.step()
            schedule.step()
            with torch.no_grad():
                for pa, pn in zip(avg.parameters(), net.parameters()):
                    pa.mul_(self.ema).add_(pn, alpha=1 - self.ema)
        self.loss = float(loss.detach())
        self.layers = self._fold([m for m in avg if isinstance(m, torch.nn.Linear)])
        return self

    def _fold(self, linears):
        """[(weight (out, in), bias)] as numpy, the constant inputs' weights 0 and the target scaling
        in the last layer."""
        layers = [(m.weight.detach().cpu().double().numpy(), m.bias.detach().cpu().double().numpy()) for m in linears]
        W0, b0 = layers[0]
        layers[0] = (np.where(self.scaler.varies[None, :], W0, 0.0), b0)
        Wl, bl = layers[-1]
        layers[-1] = (Wl * self.y_std[:, None], bl * self.y_std + self.y_mean)
        return layers

    def predict(self, z):
        return forward(self.layers, self.scaler(z))

    def summary(self):
        return {'final_training_loss': self.loss, 'hidden': list(self.hidden), 'steps': self.steps}

    def export(self, outputs):
        """(model description, arrays) for the nudge adapter (inference.nudge), the first `outputs`."""
        arrays = {'mean': self.scaler.mean, 'std': self.scaler.std}
        for i, (W, b) in enumerate(self.layers):
            last = i == len(self.layers) - 1
            arrays[f'layer{i}.weight'] = W[:outputs] if last else W
            arrays[f'layer{i}.bias'] = b[:outputs] if last else b
        return {'type': 'mlp', 'layers': len(self.layers), 'activation': 'gelu'}, arrays


def gelu(x):
    """GELU as PyTorch's default (exact, with the error function)."""
    from scipy.special import erf

    return 0.5 * x * (1.0 + erf(x / np.sqrt(2.0)))


def forward(layers, x):
    """The MLP on standardised inputs x (N, d): GELU after every layer but the last."""
    for i, (W, b) in enumerate(layers):
        x = x @ W.T + b
        if i < len(layers) - 1:
            x = gelu(x)
    return x
