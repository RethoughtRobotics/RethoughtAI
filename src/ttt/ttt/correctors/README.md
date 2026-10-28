# Correctors

A corrector is a small model that learns, from a person's corrections of a frozen policy, a **nudge**
to add to the policy's plans. The policy itself never changes:

```
plan' = plan + alpha * f(x)        x = the policy's own image features z and its 17-D state, standardised
```

`f` is fitted here and served on the robot or in the sim by the nudge adapter (`inference.nudge`).
`alpha` is the run-time dial: 0 is the policy exactly.

## From rollouts to an adapter

```
rollout bags (VR takeovers or cuff squeezes)
  adapt corrections BAG... --out recorder/data/<Name> --whole     one episode per policy rollout
  adapt lerobot recorder/data/<Name> --labels goal --intervention --plans  per-frame flag, plans as run
  ttt correction-rows --model CKPT --dataset datasets/<Name> --repo-id ... --out ../data/corrector_rows/<name>
  ttt corrector --rows ../data/corrector_rows/<name> --method ridge|mlp [--gripper] [--save DIR]
  inference serve ... --model CKPT --adapter DIR --alpha A          (scripts/run_inference --adapter)
```

`ttt correction-rows` calls the frozen policy at every frame exactly as the runtime does and keeps
its features, its plan and the person's plan (`ttt/corrections.py`). Everything after that is here.

## The problem (`data.py`, the same for every method)

| | |
|---|---|
| inputs | image features `z` (4130 for the Multi-Task DiT) and the 17-D state, standardised on the training rows; features that never vary there (the task text) are masked out |
| plan | the policy's plan as it ran on the robot (`adapt lerobot --plans`; rows only at the frames a plan was made from), else sampled again offline with a fixed seed |
| target | the **whole-plan nudge**: the person's plan minus the policy's, averaged over its steps (7: position, rotation vector, gripper), **the steps the policy drove counting 0**: a plan running into a correction carries the part it overlaps (FlowCorrect's window label) |
| nudges | with BaxterSDK `run_teleop.sh --takeover nudge` the policy keeps driving and the person's hand adds an offset to its goal (FlowCorrect's relative correction): the target's pose is then **the recorded offset** (`correction.offset`), averaged over the plan, not the person's path minus a plan made while the policy watched them correct, which a takeover's is and which can point the wrong way |
| weights | 4 on rows whose plan holds a corrected step, 1 elsewhere (CR-DAgger's `weighted_sampling`); rows without the per-step mask: 4 on correction frames and the 1 s before each |

## Methods

| name | model | chosen on held-out episodes | file |
|---|---|---|---|
| `ridge` | `W x + b`, L2 penalty | lambda (13 values, 1 to 1e6) | `ridge.py` |
| `kernel` | memory of the corrected rows, tricube on cosine similarity, exactly 0 beyond `s_min` | s_min (0 to 0.95) | `kernel.py` |
| `mlp` | CR-DAgger's residual head: 3 x 256, GELU, dropout 0.1; AdamW 3e-4, cosine, EMA, weighted MSE on normalised targets | nothing: settings fixed in advance | `mlp.py` |

## Evaluation (`evaluate.py`)

Each episode is held out in turn and the method fitted on the others. On the held-out episode's
**correction frames**: the plan's distance to the person's (cm) and how often its gripper decision
(open or closed now) matches, for the policy alone, a constant nudge, the corrector and the true nudge
(the best any one nudge per frame can do). On its **policy frames**: how far the corrector moves the
plan where the policy was left to drive (it should stay small).

```
ttt corrector --rows ../data/corrector_rows/upsidedown_whole --method mlp
```

## Adapters

`--save` fits on every episode and writes `models/<policy>/adapters/<corrections>-<method>-r<round>/`:
`adapter.json` (the model type, the policy's sha256, which a load must match, the corrections, the
fit) and `adapter.safetensors` (input scaling and weights). `--gripper` also offsets the gripper
trigger; without it the policy keeps the gripper. Never overwrite an adapter a run used: a new round
gets a new name.

## Adding a method

Write a class with `name`, `fit(z, y, w, groups)`, `predict(z)`, `summary()` and `export(outputs)`
(the model description and arrays for the adapter), add it to `METHODS` in `__init__.py`, and if its
model is not `linear` or `mlp`, teach `inference/nudge.py` to run it. `tests/test_correctors.py` then
checks that the served adapter predicts what the fitted corrector does.
