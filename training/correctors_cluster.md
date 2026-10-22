# Cluster brief: correctors in the sim (priority 1), then re-scoring the earlier sim bases (priority 2)

For the cluster agent. Run everything headless (`MUJOCO_GL=egl`) with the job setup in your private
`hpc/` checkout; nothing about the cluster goes into this repository. Report every result as a
table in `reports/sim_experiments/` of BaxterSDK, paired by test layout, with the noise floor.

## Code

- RethoughtAI branch `feat/whole-episode-corrections` (on `main`): `ttt correction-rows`,
  `ttt corrector --method ridge|mlp`, the nudge adapter (`inference serve --adapter`). Read
  `src/ttt/ttt/correctors/README.md` first: the problem, the methods, the evaluation, the adapter.
- BaxterSDK `main`: the sim, the evaluator (`baxter_mujoco.datagen.evaluate --cup-color`), the target
  servo (`run_sim_teleop.sh --policy-target`).
- The local recipe these runs scale up is `training/correctors_sim.md`; follow it for every command.

## Priority 1: do correctors learned from a new cup colour make the frozen blue base succeed with it?

| | |
|---|---|
| base S0 | `andnetdeboer/multi_task_dit_BlueCupRedContainer_sim_goal_trimmed_train55_s0` (17-D state, 7-D TCP-relative actions, teleop-goal labels): run through the target servo, `--policy-target` |
| measured locally | blue 6 of 10, red 0 of 10, black 0 of 10 (seeds 1000 to 1009) |
| corrections | `andnetdeboer/BlueCupRedContainer_sim_goal_trimmed_redcup10`, `..._blackcup10` (revision `v3.0`; 10 scripted episodes each, on the base's own training layouts) |
| test layouts | seeds **1000 to 1049**, every condition on the same 50, never used to fit or choose anything |
| methods | `ridge`, `mlp` (`ttt corrector`), each with `--gripper`; alpha 1 |

Every frame of these sets is a correction (no intervention flag): the targets are the nudge
everywhere. Use the sets exactly as published.

### Runs, in this order

| # | condition | evaluate on | purpose |
|---|---|---|---|
| C0 | S0, no adapter, twice (two independent runs of all 50) | red, black, blue | baselines and the repeat noise floor |
| C1 | ridge from redcup10 | red | does linear make red work? |
| C2 | MLP from redcup10 | red | does the MLP? better than ridge? |
| C3 | C1 and C2 adapters | blue | forgetting: what the red corrector costs where S0 was fine |
| C4 | C1 to C3 with blackcup10 | black, blue | the same for black |
| C5 | ridge and MLP from the first 1, 3 and 5 episodes of redcup10 (`ttt correction-rows` on a dataset of those episodes) | red | how many corrections are needed |
| C6 | LoRA on S0's denoiser from redcup10 (`ttt update --trainable lora`, the anchor from S0's training set), served with `--adapter` | red, blue | the "adapt the policy itself" baseline every corrector must justify itself against |

For every adapter keep `ttt corrector`'s held-out table (its stdout and the adapter's `results.json`):
if neither method beats the constant nudge offline for a colour, run C1 to C4 for it anyway, but
say so next to the closed-loop numbers.

### Report (one table per colour)

Per condition: success of 50; the paired difference against S0 on the same layouts with a 95%
bootstrap interval over layouts; on blue, the change on the layouts S0 solved (forgetting); time to
success; main failure mode; the offline held-out error (cm) and gripper decision. An effect counts
only if its interval excludes zero and it exceeds C0's repeat noise. Write the predictions first:
C1 and C2 above 0 on red (S0 is at 0), C2 at least C1, C3 within noise of S0 on blue.

### Pitfalls

- An adapter loads only on the checkpoint whose `model.safetensors` it was fitted on (its sha256 is
  in `adapter.json`): build rows and serve from the same snapshot of S0.
- The policy server and the sim share the GPU; one policy server per sim, on its own
  `ROS_DOMAIN_ID`. `scripts/run_inference` stops any other inference process on the machine.
- `ttt` caps numpy's and PyTorch's threads at 8 (`OPENBLAS_NUM_THREADS` overrides): several fits on
  one node should each get their own cores.
- Never overwrite an adapter a run used; a new fit gets a new name (`sim-<colour>cup<N>-<method>-r<k>`).

## Priority 2 (last): re-score the earlier sim bases with the current inference

The results in `reports/sim_experiments/README.md` were measured with the old VR-style target servo
pipeline (`exp/old-servo-pipeline`). Re-score each base on seeds 1000 to 1049, blue, once, with the
current `main` of both repositories and the executor its labels need:

| checkpoint | labels | executor |
|---|---|---|
| `multi_task_dit_BlueCupRedContainer_sim_lead640ms_train21_s0`, `..._bind_s0`, `..._energy_s0`, `..._bindenergy_s0` | measured TCP, 640 ms lead | `--policy` (the plan input, replay stream) |
| `multi_task_dit_BlueCupRedContainer_sim_idletrim_train21_s0`, `..._sim_train21_s0` | measured TCP, no lead | `--policy` |
| `vita_BlueCupRedContainer_sim_train{5,21,25,37,55}_s0` (and `train21_s1`, `_s2`), the `lead640ms` VITAs | measured TCP | `--policy` |
| `multi_task_dit_BlueCupRedContainer_sim_goal_trimmed_train55_s0` | teleop goals | `--policy-target` (already C0) |

Check each checkpoint's `config.json` before running: 15-D state and 8-D absolute actions take
`--policy`; 17-D state and 7-D relative actions are teleop-goal policies and take `--policy-target`.
Use the `-ema` variant where one exists, as before. Report: success of 50 next to the old number,
the main failure mode, and whether the 640 ms models over-lead on the plan input (the current
executor trails by about 60 ms, so their labels lead about 580 ms too far).
