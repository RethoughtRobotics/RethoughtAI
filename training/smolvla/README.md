# SmolVLA

Hugging Face's small VLA: a SmolVLM2-500M vision-language backbone with a flow-matching action
expert. It is language-conditioned and fine-tuned from the pretrained SmolVLM2 weights.

```bash
training/smolvla/train.sh --mode tmux --gpu 1
training/smolvla/train.sh --set 'args.cameras=[observation.images.left]' --set hub_suffix=_wrist   # wrist camera only
```

## Config

| Setting | Value | Why |
|---|---|---|
| backend | `task` | rethought-lerobot `tasks/cup_to_container/train.py` |
| dataset | `cup_to_container` | SmolVLA resizes images to 512x512 internally; 224x224 sources lose detail |
| batch / steps | 64 / 30k | LeRobot recommends batch 64 for 20k steps; we ran 30k |
| cameras | all (default) | `args.cameras` trains on a subset, e.g. the wrist camera only |

## Cost

Measured on an RTX 6000 Ada: 2.8 steps/s, about 3 h for 30k steps. It is GPU-bound, not data-bound.
`gpu_mem_mib` in the config is an estimate.

## Runs

- `andnetdeboer/smolvla_CupToContainer` (all cameras)
- `andnetdeboer/smolvla_CupToContainer_wrist` (left wrist camera only, 2026-10-11)
