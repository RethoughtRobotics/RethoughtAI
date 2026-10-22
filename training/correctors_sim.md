# Correctors in the sim: the red and black cup

Does a corrector learned from a handful of demonstrations with a new cup colour make the frozen blue
base succeed with that colour, without breaking blue? Ridge against the MLP (`src/ttt/ttt/correctors`,
its README), every number paired over the same test layouts.

## What is fixed

| | |
|---|---|
| base | `andnetdeboer/multi_task_dit_BlueCupRedContainer_sim_goal_trimmed_train55_s0` (55 blue demos, teleop-goal labels) |
| base, measured 2026-10-21 | blue 6 of 10, red 0 of 10, black 0 of 10 (seeds 1000 to 1009) |
| corrections | `andnetdeboer/BlueCupRedContainer_sim_goal_trimmed_redcup10` (10 episodes, 3272 frames) and `..._blackcup10` (10, 2947): the scripted expert on 10 of the base's own training layouts (BaxterSDK `conditions/cup_train10.jsonl`) |
| test layouts | seeds 1000 to 1019, never used for a correction or a choice |
| executor | the target servo, as the robot: `run_sim_teleop.sh --cup --cameras --left --policy-target` |

The correction sets are whole demonstrations without an intervention flag, so every frame is a
correction (no "leave the policy alone" frames): the comparison is ridge against MLP on the same rows.

## Run (sim on ROS domain 43, visible)

```bash
# 0. the sim and the base policy (both on domain 43)
tmux new-session -d -s sim43 -e ROS_DOMAIN_ID=43 -e DISPLAY=:1 -n sim \
  "cd ~/FinalProject/baxter_sdk && scripts/run_sim_teleop.sh --cup --cameras --left --policy-target"
BASE=andnetdeboer/multi_task_dit_BlueCupRedContainer_sim_goal_trimmed_train55_s0
CKPT=$(cd ~/FinalProject/rethought_ai && .venv/bin/python -c "from huggingface_hub import snapshot_download as s; print(s('$BASE'))")

# 1. the corrections as local datasets, and their rows (GPU, about 4 min per 3000 frames)
cd ~/FinalProject/rethought_ai
for c in red black; do
  hf download andnetdeboer/BlueCupRedContainer_sim_goal_trimmed_${c}cup10 --repo-type dataset --revision v3.0 \
    --local-dir datasets/sim_${c}cup10
  ttt correction-rows --model $CKPT --dataset datasets/sim_${c}cup10 \
    --repo-id andnetdeboer/BlueCupRedContainer_sim_goal_trimmed_${c}cup10 --out ../data/corrector_rows/sim_${c}cup10
done

# 2. held-out tables and adapters, ridge and MLP (CPU)
for c in red black; do for m in ridge mlp; do
  ttt corrector --rows ../data/corrector_rows/sim_${c}cup10 --method $m --gripper \
    --save models/sim_goal55/adapters/sim-${c}cup10-$m-r1 --model $CKPT
done; done

# 3. closed loop: base, ridge, MLP on the same red test layouts, then blue with each (does it break blue?)
cd ~/FinalProject/baxter_sdk/src/baxter_mujoco && source ../../scripts/env.sh
eval_run() {  # $1 label, $2 cup colour, $3 adapter or empty
  (cd ~/FinalProject/rethought_ai && ROS_DOMAIN_ID=43 scripts/run_inference --model $BASE ${3:+--adapter $3 --alpha 1.0})
  sleep 60
  ROS_DOMAIN_ID=43 uv run --group ros python -m baxter_mujoco.datagen.evaluate --task cup --cup-color $2 \
    --seeds 1000-1019 --out ~/FinalProject/data/sim_eval/$1.jsonl
}
A=~/FinalProject/rethought_ai/models/sim_goal55/adapters
eval_run base_red red ""
eval_run ridge_red red $A/sim-redcup10-ridge-r1
eval_run mlp_red red $A/sim-redcup10-mlp-r1
eval_run ridge_red_on_blue blue $A/sim-redcup10-ridge-r1
eval_run mlp_red_on_blue blue $A/sim-redcup10-mlp-r1
```

About 50 s per attempt: each line of step 3 is about 17 minutes. The adapter path must point at a
checkpoint whose `model.safetensors` matches the adapter's fingerprint (the snapshot in `$CKPT`).

## What decides it

- **Success on red** (of 20, paired by seed) for base, ridge, MLP; red is 0 for the base, so any
  success is the corrector's.
- **Blue with the red corrector** against blue without: what the corrector costs where the base was fine.
- The held-out tables of step 2 first: if neither method beats the constant nudge offline, step 3 waits.
