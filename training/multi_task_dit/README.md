# Multi-Task DiT

A diffusion-transformer policy conditioned on CLIP ViT-B/16 image features and the task text.
Built for multi-task data.

```bash
training/multi_task_dit/train.sh --mode queue --cron
```

## Config

| Setting | Value | Why |
|---|---|---|
| backend | `task` | rethought-lerobot `tasks/cup_to_container/train.py` |
| dataset | `blue_cup_to_container` | 224x224 matches the CLIP input |
| batch / steps | 96 / 30k | the largest batch that fits in 48 GB |
| retries | batch 64, then 48 | after out-of-memory |

## Cost

Measured on an RTX 6000 Ada: 1.4 steps/s, about 6 h for 30k steps. Batch 96 peaks around 39 GB, so
it needs most of a 48 GB card (`gpu_mem_mib: 43000`).

## Runs

- `andnetdeboer/multitaskdit_BlueCupToContainer` (2026-10-12), W&B project `rethoughtai-dit`
