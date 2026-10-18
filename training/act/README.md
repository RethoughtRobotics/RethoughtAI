# ACT

Action Chunking Transformer: a ResNet-18 image encoder and a CVAE transformer that predicts a chunk
of future joint targets. Trained from scratch, with no pretrained policy.

```bash
training/act/train.sh --mode tmux --gpu 0
```

## Config

| Setting | Value | Why |
|---|---|---|
| backend | `task` | rethought-lerobot `tasks/cup_to_container/train.py` |
| dataset | `cup_to_container` | ACT works on the raw images, so full resolution |
| chunk size | 20 | 0.8 s at 25 fps; the `act_cup_chunk20` run |
| batch / steps | 8 / 100k | the ACT paper's small-batch, long-schedule recipe |

## Cost

Measured on an RTX 6000 Ada: 17 steps/s, about 2 h for 100k steps, with little memory.
`gpu_mem_mib` in the config is an estimate.

## Runs

- `andnetdeboer/act_CupToContainer_chunk20` (2026-10-08)

## Notes

- Chunk size changes the action normalization stats, so the task script keys its stats files by
  chunk size (`stats_<dataset>_chunk<N>.json` in rethought-lerobot).
- ACT has no language input; one policy per task.
