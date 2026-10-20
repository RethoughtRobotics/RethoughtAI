"""Run the SweepTrash VITA checkpoint on episode 0 at t = 20 s and save every intermediate tensor
for vita.typ: pooled image features, observation latent, the 6 Euler steps, decoded chunk, and the
encoded latent of the demonstrated chunk. Needs a GPU and the local checkpoint and dataset.

    cd ~/FinalProject/rethought_ai
    uv run python training/figures/src/assets/vita_trace.py
"""

import glob
import io
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image

sys.path.insert(0, 'src/inference')
from inference.policies import LeRobotPolicy  # noqa: E402

ROOT = 'datasets/SweepTrash_Kin-Full'
MODEL = 'models/vita_SweepTrash_Kin-Full_inpainted_minimax_lead640ms_aug_30k_r1/step-030000'
T, FPS = 20.0, 25
OUT = Path(__file__).resolve().parent / 'vita.json'


def frame(cam):
    png = subprocess.run(
        [
            'ffmpeg',
            '-loglevel',
            'error',
            '-ss',
            str(T),
            '-i',
            f'{ROOT}/videos/observation.images.{cam}/chunk-000/file-000.mp4',
            '-frames:v',
            '1',
            '-f',
            'image2',
            '-c:v',
            'png',
            '-',
        ],
        capture_output=True,
        check=True,
    ).stdout
    return np.asarray(Image.open(io.BytesIO(png)).convert('RGB'))


data = pd.read_parquet(sorted(glob.glob(f'{ROOT}/data/*/*.parquet'))[0])
ep = data[data.episode_index == 0]
i = int(T * FPS)
state = np.asarray(ep['observation.state'].values[i], np.float32)
# This checkpoint's labels lead the measured pose by 0.64 s (16 frames): its chunk at i is the
# recorded pose over [i + 16, i + 48).
demo = np.stack(ep['action'].values[i + 16 : i + 48]).astype(np.float32)

policy = LeRobotPolicy(MODEL, 'cuda')
vita = policy.policy.vita
obs = {f'observation.images.{c}': frame(c) for c in ('left', 'right', 'third_eye')}
obs['observation.state'] = state

trace = {}
vita.observer.register_forward_hook(lambda m, a, o: trace.__setitem__('observer', o[0].float().cpu().numpy()))
vita.obs_encoder.register_forward_hook(lambda m, a, o: trace.__setitem__('z0', o[0].float().cpu().numpy()))
steps = []
sample = vita.flow_matcher.sample


def traced_sample(model, start, num_steps=None):
    n = num_steps or vita.flow_matcher.num_sampling_steps
    x = start
    steps.append(x[0].float().cpu().numpy())
    for k in range(n):
        t = torch.full((x.shape[0],), k / n, device=x.device, dtype=x.dtype)
        x = x + model(x, t) / n
        steps.append(x[0].float().cpu().numpy())
    return x


vita.flow_matcher.sample = traced_sample
pred = policy(obs)

with torch.inference_mode():
    norm = policy.normalizer
    from lerobot.processor import TransitionKey, create_transition

    a = norm(create_transition(action=torch.as_tensor(demo, device='cuda')))[TransitionKey.ACTION]
    z_demo = vita.action_encoder(a[None])[0].float().cpu().numpy()

z1 = steps[-1]
cos = float(np.dot(z1, z_demo) / np.linalg.norm(z1) / np.linalg.norm(z_demo))
pos_err = float(np.linalg.norm(pred[:, :3] - demo[: len(pred), :3], axis=1).mean())
print('observer', trace['observer'].shape, 'z0', trace['z0'].shape, 'steps', len(steps))
print(f'cos(z1, z_demo) = {cos:.3f}; mean left xyz error {pos_err * 1000:.1f} mm')
json.dump(
    {
        'state': state.round(3).tolist(),
        'img_feats': trace['observer'][28:].reshape(3, 512).round(3).tolist(),
        'steps': [s.round(3).tolist() for s in steps],
        'z_demo': z_demo.round(3).tolist(),
        'pred': pred.round(3).tolist(),
        'demo': demo.round(3).tolist(),
        'cos': cos,
        'pos_err_mm': pos_err * 1000,
    },
    open(OUT, 'w'),
)
