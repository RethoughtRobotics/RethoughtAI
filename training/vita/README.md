# VITA

Vision-to-action flow matching: it flows from image latents directly to action latents, with no
noise source. A small model trained from scratch with a long schedule.

```bash
training/vita/train.sh --mode tmux --gpu 0
```

## Config

These are the upstream VITA settings, not the task-script defaults:

| Setting | Value | Why |
|---|---|---|
| backend | `task` | rethought-lerobot `tasks/cup_to_container/train.py` (VITA support added there) |
| dataset | `cup_to_container` | |
| batch / steps | 128 / 100k | upstream recipe |
| precision | fp32 | upstream trains in fp32 (the task default is bf16) |
| grad clip | 1.0 | upstream (the task default is 10.0) |
| EMA power | 0.75 | upstream EMA warmup |

## Cost

Measured on an RTX 6000 Ada: 4.8 steps/s, about 6 h for 100k steps. `gpu_mem_mib` in the config is
an estimate.

## Runs

- `andnetdeboer/vita_CupToContainer` (2026-10-11)
