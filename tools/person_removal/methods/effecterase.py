"""EffectErase (FudanCVL, CVPR 2026): Wan2.1-Fun-1.3B-InP with a LoRA trained to remove an object together
with its effects (shadows, reflections), guided by a crop of the object.

The best results here at 20 steps (its default is 50), several times slower than MiniMax-Remover. Licence:
CC BY-NC 4.0 (non-commercial). Code: github.com/FudanCVL/EffectErase; weights: huggingface
alibaba-pai/Wan2.1-Fun-1.3B-InP and FudanCVL/EffectErase (EffectErase.ckpt).
"""

import sys
from pathlib import Path

import numpy as np

DEFAULTS = {'steps': 50, 'clip_len': 81, 'overlap': 16, 'dilation': 6, 'temporal': 2, 'feather': 6}
SEED = 2025
PROMPT = 'Remove the specified object and all related effects, then restore a clean background.'
# The negative prompt the authors use (Wan's standard one, in Chinese): blur, artefacts, deformed limbs, crowds.
NEGATIVE = (
    '细节模糊不清，字幕，作品，画作，画面，静止，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，'
    '画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，杂乱的背景，三条腿，背景人很多，倒着走'
)


def load(models: Path):
    import torch
    from PIL import Image

    models = Path(models)
    sys.path.insert(0, str(models / 'EffectErase'))
    sys.path.insert(0, str(models / 'EffectErase/examples/remove_wan'))
    from diffsynth import ModelManager, WanRemovePipeline
    from infer_remove_wan import crop_square_from_pil

    wan = models / 'weights/Wan2.1-Fun-1.3B-InP'
    manager = ModelManager(device='cuda')
    manager.load_models(
        [str(wan / name) for name in ('diffusion_pytorch_model.safetensors', 'models_t5_umt5-xxl-enc-bf16.pth',
                                      'Wan2.1_VAE.pth', 'models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth')],
        torch_dtype=torch.bfloat16,
    )  # fmt: skip
    manager.load_lora_v2(str(models / 'weights/effecterase/EffectErase.ckpt'), lora_alpha=1.0)
    pipe = WanRemovePipeline.from_model_manager(manager, torch_dtype=torch.bfloat16, device='cuda')
    pipe.enable_vram_management(num_persistent_param_in_dit=6 * 10**9)

    def to_tensor(x):  # (T, H, W, C) uint8 to (C, T, H, W) in [-1, 1]
        return torch.from_numpy(x).float().permute(3, 0, 1, 2).cuda() / 127.5 - 1.0

    def inpaint(frames: np.ndarray, masks: np.ndarray, steps: int) -> np.ndarray:
        t, h, w = masks.shape
        masks_rgb = np.repeat((masks[..., None] * 255).astype(np.uint8), 3, axis=-1)
        k = int(masks.reshape(t, -1).sum(1).argmax())  # the object's crop from the frame showing most of it
        ref = crop_square_from_pil(Image.fromarray(masks_rgb[k]), Image.fromarray(frames[k]), target_size=224).cuda()
        with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16):
            video, _ = pipe(
                video_mask=to_tensor(masks_rgb), video_fg_bg=to_tensor(frames), video_bg=None, task='remove',
                fg_first_img=ref, prompt_remove=PROMPT, negative_prompt=NEGATIVE, num_inference_steps=steps,
                cfg_scale=1.0, seed=SEED, tiled=False, height=h, width=w,
            )  # fmt: skip
        return np.stack([np.asarray(f.convert('RGB')) for f in video])

    return inpaint
