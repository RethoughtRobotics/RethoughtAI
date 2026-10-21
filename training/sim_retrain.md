# Sim base and colour corrections: what to generate, train and score

The sim experiment for the corrector plan: a strong base trained on blue cups only, then corrections
for a cup colour no demonstration shows (red, black), each from 1 to 10 scripted episodes, scored on
whether the corrected policy picks up the new colour without losing the blue one.

## The base: teleop goals, the recipe that works on the robot

The real Multitask DiT trained on teleop data with teleop-goal labels (what the servo was commanded,
`andnetdeboer/multitaskdit_BlueCupToContainer`) runs smoothly and does not hesitate: it reproduces
the operator's commands and the same servo absorbs the tracking lag at run time as during recording.
The sim's scripted expert drove the arm the same way (it publishes `/policy/left/target`, the teleop
servoes onto it), and every recording has the commanded target (`/teleop/left/goal`), so the sim
data is exported with those labels. Kinesthetic labels (measured pose) and the plan input are a
separate path, not used here.

The expert holds still for fixed times nothing observable measures (1.3 s at the start, 1.0 s after
closing the gripper, 0.8 s after opening it); a policy replanning there sees the same view and plans
the same wait. The export cuts every still stretch (TCP, goal and gripper all unchanged) to its first
frame.

| Dataset (blue cup only) | Revision | Episodes | Use |
|---|---|---|---|
| `andnetdeboer/BlueCupRedContainer_sim_goal_trimmed` | `4a39e23c5d6e` | 60 | all |
| `andnetdeboer/BlueCupRedContainer_sim_goal_trimmed_train55` | `4941730b2261` | 0 to 54 (18534 frames) | trains the base |
| `andnetdeboer/BlueCupRedContainer_sim_goal_trimmed_heldout5` | `5a260bc71ec0` | 55 to 59 | validation loss |

Made with `adapt lerobot recorder/data/BlueCupRedContainer_sim --labels goal --trim-still 0.001 1`
(46 to 64 still frames cut per episode, 20221 frames of 24055 kept).

### Train

The real teleop DiT's recipe, unchanged (`training/multi_task_dit/config.yaml`, the task backend):

```bash
python training/train.py multi_task_dit --dataset sim_cup_goal_train55 \
    --set seed=0 --set hub_name=multi_task_dit --set hub_suffix=_s0 --mode tmux --gpu 0
```

Hub repo: `andnetdeboer/multi_task_dit_BlueCupRedContainer_sim_goal_trimmed_train55_s0`. Check that its
`config.json` has the real teleop DiT's layout: `n_obs_steps` 2, `horizon` 32, `n_action_steps` 24, a
7-D action (TCP-relative goal and trigger). If it trains 8-D absolute actions instead, stop and report.

### Score the base

The policy runs through the target servo, as the teleop DiT ran on the robot:

- BaxterSDK `main`: `scripts/run_sim_teleop.sh --cup --cameras --left --policy-target`
  (`--policy` now plays a plan through the replay stream, for kinesthetic checkpoints; not this one).
- RethoughtAI `main`: `scripts/run_inference --model <repo>` with the default `configs/baxter.yaml`:
  the relative `multi_task_dit` profile, a plan every 4 frames, temporally ensembled.
- `python -m baxter_mujoco.datagen.evaluate --task cup --seeds 1000-1049 --out <rows>.jsonl`
  (blue, the default `--cup-color`).

S0 on blue twice (repeat noise was 3 to 4 of 50), then S0 on red and on black
(`--cup-color red`, `--cup-color black`), the same seeds.

## Colour corrections: generate

Two correction sets, 10 scripted episodes each, on 10 of the base's own training layouts (episodes
0, 6, .., 54: same cup and container places, container yaw and lighting), only the cup's colour
changed. The layouts are in BaxterSDK `src/baxter_mujoco/conditions/cup_train10.jsonl`.

The sim stack the generator drives must follow the target stream:

```bash
scripts/run_sim_teleop.sh --cup --cameras --left --policy-target --session <SESSION>   # BaxterSDK
python -m baxter_mujoco.datagen.generate --task cup --episodes 10 \
    --cup-color red --conditions src/baxter_mujoco/conditions/cup_train10.jsonl         # baxter_mujoco
```

Sessions `BlueCupRedContainer_sim_redcup10` and `BlueCupRedContainer_sim_blackcup10`. A failed
episode is discarded; with `--conditions` there is one attempt per layout, so a set can come out with
fewer than 10: note which layouts.

Export each as the base was, with the same task string (the policy is text-conditioned; the new
colour is a visual change only):

```bash
adapt lerobot recorder/data/BlueCupRedContainer_sim_redcup10 --config recorder/data/BlueCupRedContainer_sim_redcup10/baxter.yaml \
    --repo-id andnetdeboer/BlueCupRedContainer_sim_goal_trimmed_redcup10 \
    --task "Pick up the blue cup and place it in the red container" --labels goal --trim-still 0.001 1
adapt push datasets/BlueCupRedContainer_sim_redcup10 --repo-id andnetdeboer/BlueCupRedContainer_sim_goal_trimmed_redcup10
```

and the same for black. Before pushing, look at one third_eye frame per set: the cup must be red or
black, not blue.

## Then

Corrections from 1 or all 10 episodes of a colour, the base frozen: E1 linear and E2 MLP correctors on
the encoder's observation embedding, E3 LoRA on the denoiser. Each target is the whole chunk the DiT
predicts, in its native action space. Score on the 50 test layouts in that colour and in blue
(forgetting). If a colour is learned, add the second colour's corrections on top and score all three.
