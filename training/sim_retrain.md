# Sim retraining: idle-trimmed demonstrations, the robot's control pipeline

What to retrain for the sim corrector experiments, and how to run and score it. Everything else in
[sim_experiments.md](sim_experiments.md) (layouts, seeds, scoring, statistics) is unchanged.

## Why

Every sim cup demonstration waits 1.32 s at home (median, 1.12 to 1.40 s) before the TCP moves:
33 frames, longer than the 32-step plan. Nothing in the observation says how long the wait has
lasted, so at home every plan says wait, the arm does not move, the next observation is the same,
and the policy stalls. That is the main failure of the sim bases (the DiT: 0 of 50, 38 attempts never
reaching the cup). The 640 ms label lead only shortened the wait to 0.68 s.

The fix is in the data: each episode now starts at its last frame before the TCP first moves 1 mm
or 1 deg (`adapt lerobot --trim-idle 0.001 1`). Pauses later in an episode are kept; the gripper
state explains them.

The sim also runs the robot's control pipeline now: the runtime publishes each step's timed plan
(`/policy/left/path`) and the teleop plays it through tcp_ik and the replay stream (`--input plan`),
with no servo lag. Labels are therefore not led: a led label would be played 640 ms early.

## Datasets

All on the Hub, private, exported from the same 60 recordings as `BlueCupRedContainer_sim`, measured
TCP labels, lead 0, 25 fps.

| Dataset | Episodes | Use |
|---|---|---|
| `andnetdeboer/BlueCupRedContainer_sim_idletrim@0ece535665ed` | 0 to 59 | correction pool (21, 25, 29, 33), anchor replay, open-loop test (38 to 55) |
| `andnetdeboer/BlueCupRedContainer_sim_idletrim_train21@2e306270161d` | 0 to 20 of the above | trains the base |
| `andnetdeboer/BlueCupRedContainer_sim_idletrim_heldout5@bcb8b9b969af` | 55 to 59 of the above | validation loss |

Episode indices match the untrimmed set: episode k is the same demonstration in both, starting
`idle_start_frames` recording steps in (`meta/rethoughtai_sources.json`; 27 to 34, median 31). From
there on, state and action equal the untrimmed episode's exactly; the TCP moves 5 mm within 4 frames
(median) of the start, against 33 before. 22263 frames in all (24055 untrimmed); train21 7851.

Do not use `*_sim_lead640ms_*` or the untrimmed `*_sim_*` sets or checkpoints for new runs.

## What to train

| Run | Policy | Dataset | Seed | Hub repo |
|---|---|---|---|---|
| S0 (required) | Multi-Task DiT | `sim_cup_idletrim_train21` | 0 | `andnetdeboer/multi_task_dit_BlueCupRedContainer_sim_idletrim_train21_s0` |
| S0b, S0c (noise floor, after S0) | Multi-Task DiT | `sim_cup_idletrim_train21` | 1, 2 | `..._s1`, `..._s2` |

Same configuration as `multi_task_dit_BlueCupRedContainer_sim_lead640ms_train21_s0` apart from the
dataset: `training/multi_task_dit/config.yaml`, 30k steps, batch 64, n_obs_steps 2, horizon 32,
24 steps run, DDPM, LeRobot's default image transforms.

```bash
python training/train.py multi_task_dit --dataset sim_cup_idletrim_train21 \
    --set seed=0 --set batch_size=64 --set hub_name=multi_task_dit --set hub_suffix=_s0 --mode tmux --gpu 0
```

Check before launching: the printed command names `andnetdeboer/BlueCupRedContainer_sim_idletrim_train21`
and the eval set `..._idletrim_heldout5`, and pushes to the repo in the table.

Check after training, offline, before any closed-loop run: on the first frame of a few training
episodes, the predicted chunk should move the TCP more than 1 cm within its first 8 steps. If it
does not, the trim did not take; stop and report.

## How to evaluate

The robot's pipeline, nothing sim-specific:

- BaxterSDK `main` (with `feat/replay-tcp`): `scripts/run_sim_teleop.sh --cup --cameras --left --policy`.
  `--policy` now plays the plan through `--input plan`.
- RethoughtAI `main` (with `feat/dit-ensemble`): `scripts/run_inference --model <repo>` with the
  default `configs/baxter.yaml`. `configs/baxter_sim.yaml` (run-time leads for the untrimmed sim
  checkpoints) is removed; do not recreate it. The DiT runs the `multi_task_dit_absolute` profile:
  a plan every 4 frames, temporally ensembled, 10 DDPM steps, no lead.
- `python -m baxter_mujoco.datagen.evaluate --task cup --seeds 1000-1049 --out <rows>.jsonl`, as
  before.

Score S0 twice on seeds 1000 to 1049 (closed-loop repeat noise was 3 to 4 of 50 before). Add one
metric per attempt: the time until the TCP has moved 2 cm from home. It should be close to the
inference latency; if attempts still sit at home, report it before anything else.

## Then the corrector runs

Only once S0 solves some layouts (forgetting and gains need solved layouts to measure against):
E1 (linear), E2 (MLP) and E3 (LoRA on the denoiser, plan S20-M's code), three update seeds each,
corrections from pool episodes 21, 25, 29, 33 of `BlueCupRedContainer_sim_idletrim`, anchor on
`..._idletrim_train21`. Each corrector's target is the whole chunk the DiT predicts (its action steps by 8), not one action.
