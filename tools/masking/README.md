# Removing people from demonstration videos

Kinesthetic demos show the operator's hand pushing the arm, so a policy can learn "move when a hand is
visible". These tools erase the person from a LeRobot dataset's camera videos, keeping everything else
identical (states, actions, timestamps, other cameras), and push the result to the Hub.

```
dataset ──sam3_masks.py──▶ person masks ──video_remove.py (MiniMax-Remover / EffectErase)──▶ X_inpainted_<method>
        └─sam3_inpaint_dataset.py (SAM3 + OpenCV Telea, one pass) ──────────────────────────▶ X_inpainted
```

| Script | What it does |
| --- | --- |
| `sam3_masks.py` | SAM3 "person" masks per frame, saved as lossless FFV1 mask videos. |
| `video_remove.py` | Video inpainting in overlapping 81-frame clips at native resolution; output pasted back only inside a feathered mask, so the robot and objects stay pixel-exact. |
| `sam3_inpaint_dataset.py` | Original single-pass method: SAM3 masks + Telea inpainting. Fast, but leaves pale smears over large regions such as a torso. |
| `../relabel_lead.py` | Rebuilds measured-TCP actions for another lead (e.g. 16 steps = 640 ms). |

## Methods

| Method | Model | Steps | Notes |
| --- | --- | --- | --- |
| Telea | OpenCV | n/a | Per frame. Smears and flicker on large masks. |
| MiniMax-Remover | Wan2.1-1.3B, distilled (NeurIPS 2025) | 12 | Fast, no CFG; temporally consistent. |
| EffectErase | Wan2.1-Fun-1.3B + LoRA (CVPR 2026) | 50 | Also removes shadows and reflections. CC BY-NC 4.0. |

Environments live in `~/.cache/vor` (one venv per method, plus weights);
SAM3 needs a separate venv with `transformers>=5`.

## Usage

```bash
# 1. masks (GPU-bound: ~8 fps per A100 worker; spread workers over GPUs)
python sam3_masks.py --source DATASET --out MASKS --camera observation.images.third_eye --workers 4

# 2. removal
python video_remove.py --method minimax --source DATASET --masks MASKS \
    --out datasets/NAME_inpainted_minimax --camera observation.images.third_eye
```

Masks are unioned over ±2 frames and dilated by 6 px before inpainting, because SAM3 segments each frame
independently and flickers. Output videos keep the source codec settings (AV1, GOP 2) so training decodes
them as fast as the originals.

## Caveats

- SAM3 is prompted with "person". Hands entering from the frame edge are the hardest case; check a contact
  sheet before training on a new dataset.
- Only the third-eye camera is inpainted by default; wrist cameras may also show the operator's hand.
- `evaluate.py` and `mask_dataset.py` are the earlier DeepLab / Mask R-CNN baselines for sampled frames.

Batch jobs that run these tools on a cluster are kept out of this repository, in a private `hpc/`
checkout that `.gitignore` excludes.
