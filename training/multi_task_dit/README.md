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

## Fine-tune on corrections with LoRA

Flags on the same recipe, all off by default (rethought-lerobot `main` from `3346cb4`): start from a trained
checkpoint, train only LoRA on the denoiser's attention and MLP layers, and add correction episodes
recorded on the robot beside the demonstrations, sampled more often.

```bash
training/multi_task_dit/train.sh --mode run \
    --set args.init_from=<checkpoint>/pretrained_model --set args.lora_rank=16 \
    --set args.corrections_repo_id=<corrections dataset> --set args.corrections_weight=4 \
    --set steps=2000 --set batch_size=64
```

| Flag | Effect |
|---|---|
| `init_from` | the checkpoint's own config and normalization stats are kept |
| `lora_rank` | PEFT LoRA of this rank on the denoiser (524k parameters at 16); everything else frozen |
| `lora_alpha`, `lr` | LoRA alpha (default: the rank) and learning rate (default 1e-4) |
| `corrections_repo_id`, `corrections_dir` | the correction episodes, frames resized to the training cameras' shape |
| `corrections_weight` | each correction frame is sampled this many times per pass over the demonstrations |
| `corrections_val_episodes` | correction episodes held out, logged as `val_corrections/` (default: the last one) |

A LoRA checkpoint holds `lora_adapter/` and, in `pretrained_model/`, the base with the adapter merged in,
which the runtime loads like any checkpoint. 2000 steps at batch 64 take about 6 min on an A100.

## Runs

- `andnetdeboer/multitaskdit_BlueCupToContainer` (2026-10-12), W&B project `rethoughtai-dit`
- `dit_lora_papercup_v1` (2026-10-28): LoRA rank 16 on the above, 2000 steps, the 10-22 PaperCup nudge
  corrections (7 episodes, weight 4). Demo validation loss 0.117 to 0.087, held-out correction episode
  0.050 to 0.015. On the robot it picked up the paper cup at new positions, still picked up the blue
  cup, and worked with the container moved (one trial each). W&B project `rethoughtai-dit-lora`
