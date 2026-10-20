"""Data for the figures: camera frames, a real action chunk, the DiT's parameter split and a noised
chunk, and the TTT example. Writes next to this file. vita.json comes from vita_trace.py (GPU);
ttt.json is built from it, so run that first.

    cd ~/FinalProject/rethought_ai
    uv run python training/figures/src/assets/make_assets.py
"""

import glob
import json
import math
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
from safetensors import safe_open

HERE = Path(__file__).resolve().parent
ROOT = 'datasets/SweepTrash_Kin-Full'
DIT = 'models/multi_task_dit_SweepTrash_Kin-Full_inpainted_aug_30k_r1/step-030000'
T, FPS = 20.0, 25  # episode 0, the moment every figure shows

# Frames at t and one step earlier (the DiT sees two), 320 px wide.
for cam in ('third_eye', 'left', 'right'):
    for suffix, t in (('', T), ('_prev', T - 1 / FPS)):
        subprocess.run(
            [
                'ffmpeg',
                '-loglevel',
                'error',
                '-y',
                '-ss',
                str(round(t, 2)),
                '-i',
                f'{ROOT}/videos/observation.images.{cam}/chunk-000/file-000.mp4',
                '-frames:v',
                '1',
                '-vf',
                'scale=320:-1',
                '-q:v',
                '3',
                str(HERE / f'{cam}{suffix}.jpg'),
            ],
            check=True,
        )

data = pd.read_parquet(sorted(glob.glob(f'{ROOT}/data/*/*.parquet'))[0])
ep = data[data.episode_index == 0]
i = int(T * FPS)

# model.typ: the 16-step chunk and the state at t.
(HERE / 'chunk.json').write_text(
    json.dumps({
        'chunk': np.stack(ep['action'].values[i : i + 16]).round(3).tolist(),
        'state': np.asarray(ep['observation.state'].values[i]).round(3).tolist(),
        't': T,
    })
    + '\n'
)

# dit.typ: parameters by module, and the 32-step chunk min-max normalised as the DiT trains on it,
# noised at step 50 of 100 under its squaredcos_cap_v2 schedule (one seeded draw).
params = {}
with safe_open(f'{DIT}/model.safetensors', 'pt') as f:
    for k in f.keys():
        if 'vision_encoder' in k:
            key = 'clip_vision'
        elif 'text_encoder.text_encoder' in k:
            key = 'clip_text'
        elif 'text_encoder.projection' in k:
            key = 'text_proj'
        elif 'adaLN' in k:
            key = 'adaln'
        elif 'transformer_blocks' in k:
            key = 'attn_mlp'
        elif 'time_mlp' in k:
            key = 'time_mlp'
        else:
            key = 'io_proj'
        params[key] = params.get(key, 0) + math.prod(f.get_slice(k).get_shape())
stats = json.load(open(f'{ROOT}/meta/stats.json'))
lo, hi = np.array(stats['action']['min']), np.array(stats['action']['max'])
a = np.stack(ep['action'].values[i : i + 32])
x0 = np.clip(2 * (a - lo) / np.maximum(hi - lo, 1e-8) - 1, -1, 1)
steps = 100


def alpha_bar(s):
    return math.cos((s + 0.008) / 1.008 * math.pi / 2) ** 2


betas = np.array([min(1 - alpha_bar((k + 1) / steps) / alpha_bar(k / steps), 0.999) for k in range(steps)])
abar = np.cumprod(1 - betas)[50]
eps = np.random.default_rng(0).standard_normal(x0.shape)
(HERE / 'dit.json').write_text(
    json.dumps({
        'params': params,
        'x0': x0.round(3).tolist(),
        'xt': (math.sqrt(abar) * x0 + math.sqrt(1 - abar) * eps).round(3).tolist(),
        'eps': eps.round(3).tolist(),
        'abar_t': float(abar),
    })
    + '\n'
)

# ttt_lora.typ: eight seeded latent offsets of VITA's z0, and illustrative rewards.
vita = json.load(open(HERE / 'vita.json'))
z0, sigma = np.array(vita['steps'][0]), 0.03
starts = z0 + sigma * np.random.default_rng(0).standard_normal((8, 512))
(HERE / 'ttt.json').write_text(
    json.dumps({
        'z0': z0.round(3).tolist(),
        'z_demo': vita['z_demo'],
        'sigma': sigma,
        'starts': starts.round(3).tolist(),
        'rewards': [0.25, 0.3, 1.4, 0.2, 0.55, 1.45, 0.3, 0.6],
    })
    + '\n'
)
