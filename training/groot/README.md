# GR00T N1.7

NVIDIA's 3B humanoid/generalist VLA: an Eagle VLM with a flow-matching diffusion-transformer action head. This is a
full fine-tune in bf16 through `lerobot-train`.

```bash
training/groot/train.sh --mode queue --cron
```

## Config

| Setting | Value | Why |
|---|---|---|
| backend | `lerobot` | `--policy.type=groot` |
| base model | `nvidia/GR00T-N1.7-3B` | |
| embodiment | `new_embodiment` | Baxter is not one of GR00T's pretraining embodiments |
| dataset | `blue_cup_to_container_46ep` | episodes 4, 14, 24, 34, 44 excluded |
| chunk / executed | 16 / 16 | 0.64 s at 25 fps |
| actions | absolute (`use_relative_actions: false`) | |
| batch / steps | 64 / 20k | retries 32, then 16, after out-of-memory |
| uploads | steps 10k and 20k | as `step-*` branches |

## Cost

Measured on an RTX 6000 Ada: 1.38 s/step at batch 64, 36 GB peak, 7 h 40 min for 20k steps.

## Runs

- `andnetdeboer/groot_BlueCupToContainer` (2026-10-12). That run had validation turned off
  (`--eval_steps=0`), so it has no `eval/*` curves. The config here keeps validation on, so a rerun
  holds out 10% of the 46 episodes.
