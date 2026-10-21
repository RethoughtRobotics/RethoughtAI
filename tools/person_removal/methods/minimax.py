"""MiniMax-Remover (Zi et al., NeurIPS 2025): Wan2.1-1.3B video diffusion distilled for object removal.

No text prompt and no classifier-free guidance, 12 steps: the fastest of the video models here, and
temporally consistent. Code: github.com/zibojia/MiniMax-Remover; weights: huggingface zibojia/minimax-remover.
"""

import sys
from pathlib import Path

import numpy as np

DEFAULTS = {'steps': 12, 'clip_len': 81, 'overlap': 16, 'dilation': 6, 'temporal': 2, 'feather': 6}
SEED = 42


def load(models: Path):
    import torch

    sys.path.insert(0, str(Path(models) / 'MiniMax-Remover'))
    from diffusers.models import AutoencoderKLWan
    from diffusers.schedulers import UniPCMultistepScheduler
    from pipeline_minimax_remover import Minimax_Remover_Pipeline
    from transformer_minimax_remover import Transformer3DModel

    weights = Path(models) / 'weights/minimax-remover'
    # The pipeline puts its tensors on 'cuda:0': run one process per GPU (remove.py does).
    pipe = Minimax_Remover_Pipeline(
        transformer=Transformer3DModel.from_pretrained(weights / 'transformer', torch_dtype=torch.float16),
        vae=AutoencoderKLWan.from_pretrained(weights / 'vae', torch_dtype=torch.float16),
        scheduler=UniPCMultistepScheduler.from_pretrained(weights / 'scheduler'),
    ).to('cuda')

    def inpaint(frames: np.ndarray, masks: np.ndarray, steps: int) -> np.ndarray:
        t, h, w = masks.shape
        out = pipe(
            images=torch.from_numpy(frames).float() / 127.5 - 1.0,
            masks=torch.from_numpy(masks[..., None].astype(np.float32)),
            num_frames=t,
            height=h,
            width=w,
            num_inference_steps=steps,
            generator=torch.Generator(device='cuda').manual_seed(SEED),
            iterations=1,
        ).frames[0]
        return (np.clip(out, 0, 1) * 255 + 0.5).astype(np.uint8)

    return inpaint
