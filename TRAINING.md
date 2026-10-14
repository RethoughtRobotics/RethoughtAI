# Training spec: CupToContainer, TCP-goal actions

This spec tells a training server how to turn the `andnetdeboer/CupToContainer` LeRobot dataset into a
policy that `inference` can run on Baxter's left arm. The source of truth for the action definition
is `src/adapter/adapter/labels.py`. If this document and that file disagree, the file is right.

## 1. What the trained policy must do

At each policy step (25 Hz), the policy gets the current camera images and arm state. It returns a
**chunk of 100 future TCP goals, each relative to the TCP measured at that step**:

```
actions: (100, 7) float, physical units, NOT normalized
  [:, 0:3]  dx dy dz   goal position minus TCP position, base-frame axes, metres
  [:, 3:6]  rx ry rz   base-frame rotation vector r with  R_goal = exp(r) · R_tcp,  radians
  [:, 6]    trigger    gripper, 0 open .. 1 closed (absolute, not relative)
```

The runtime (`src/inference/inference/runtime.py`) anchors each chunk on the TCP measured at
observation time with `labels.apply_actions`. It ensembles overlapping chunks
(`temporal_ensemble_coeff: 0.01`, `n_action_steps: 1`) and streams the result to the teleop as a
VR-controller target. The robot's own servo and IK then track it, exactly as they did for the human.

**Why goals and not measured poses or joints:** the labels are what the teleop servo was
*commanded* to reach (`/teleop/left/goal`), not where the arm ended up. The policy reproduces the
operator's intent, and the same servo absorbs the tracking lag at runtime that it absorbed during
recording. Because the actions are relative to the TCP, the policy does not depend on absolute
workspace position, so small changes in where the robot or table sit cost less.

## 2. Input dataset

| | |
|---|---|
| Hub repo | `andnetdeboer/CupToContainer` (private, dataset, revision tag = LeRobot `CODEBASE_VERSION`) |
| Format | LeRobot v3, 25 fps, 51 episodes, 25,934 frames (17.3 min); episodes are 13–41 s |
| Task string | `Pick up the blue cup and place it in the red container` (single task) |
| `observation.images.left` | left hand camera, 400×640 RGB video |
| `observation.images.third_eye` | fixed RealSense, 480×640 RGB video |
| `observation.state` (15) | `tcp.x y z qx qy qz qw` (base frame), `gripper` (measured, 0 open..1 closed), `left_s0 s1 e0 e1 w0 w1 w2` |
| `action` (8) | **absolute** `goal.x y z qx qy qz qw` and `gripper` (trigger). While the clutch was released, goal = TCP |

`meta/rethoughtai_sources.json` maps each episode to its source recording.

Download it with `hf download andnetdeboer/CupToContainer --repo-type dataset --local-dir DIR`.
The repo is private, so the server needs an `HF_TOKEN` with read access.

## 3. Building action targets (the TCP-goal transform)

The dataset stores both poses as absolute values on purpose. A chunk observed at frame `t` pairs
`action[t+k]` with `state[t]`, so you can only form it once the frames are loaded together:

1. Load with `delta_timestamps = {"action": [k / 25 for k in range(100)]}`. Frames past the episode
   end repeat the last frame, which is what `labels.chunk_actions` does: the arm holds.
2. For every sample, convert the (100, 8) absolute chunk using `state[t]`:

```python
import numpy as np
from scipy.spatial.transform import Rotation

def tcp_relative(action, state):
    """(K, 8) absolute goals + (15,) state at t -> (K, 7) targets; mirrors labels.chunk_actions."""
    R_tcp = Rotation.from_quat(state[3:7])                     # xyzw
    dp = action[:, :3] - state[:3]
    dr = (Rotation.from_quat(action[:, 3:7]) * R_tcp.inv()).as_rotvec()
    return np.column_stack([dp, dr, action[:, 7]]).astype(np.float32)
```

3. **Keep the loss on padded steps.** Do not mask with `action_is_pad`. The repeated last frame is a
   real label meaning "hold still at the end", and leaving it in is what teaches the policy to stop.

**Do not use LeRobot's `use_relative_actions` / `RelativeActionsProcessorStep`.** It computes
`action - state` element by element. That is meaningless for quaternions, and it would also make the
action 8-D instead of 7-D.

## 4. Observation preprocessing

- **State → 17-D:** `[tcp.xyz (3), first two columns of R_tcp (6), gripper (1), joints (7)]`.
  Use the 6-D rotation in place of the quaternion: `q` and `-q` describe the same rotation, and the
  export only keeps quaternion signs consistent within an episode, not across episodes.
- **Images:** use native resolution, RGB, with ImageNet normalization (ACT default). Light
  augmentation is allowed (colour jitter, ±5 % random crop). Do not flip or rotate the images: a
  flipped image would no longer match the base-frame actions.
- **Normalization:** compute mean/std **over the transformed 7-D targets** (all 100 steps pooled)
  and over the 17-D state, from the training split only. The `action` stats in `meta/stats.json`
  describe absolute goals and must not be used. Save the stats with the model.

## 5. Model and hyperparameters

The primary model is **ACT** (LeRobot `act`), which matches the runtime's settings in `configs/baxter.yaml`:

| param | value |
|---|---|
| `chunk_size` | 100 (4 s) |
| `n_action_steps` | 1 (the runtime ensembles with coeff 0.01) |
| vision backbone | ResNet-18, ImageNet weights, one pass per camera |
| `dim_model` / heads / enc / dec layers | 512 / 8 / 4 / 1 (LeRobot ACT defaults) |
| VAE | on, `latent_dim` 32, `kl_weight` 10 |
| optimizer | AdamW, lr 1e-5, backbone lr 1e-5, weight decay 1e-4 |
| batch / steps | batch 8, 100k steps (about 31 epochs); checkpoint every 10k |
| loss | L1 on normalized 7-D targets (+ KL) |
| seed | fixed; log it |

The alternative is **SmolVLA**, with the same action and state definitions (§3–4), `chunk_size` 50,
`n_action_steps` 50, and no ensembling. If you use it, also change `inference.chunk_size`,
`n_action_steps` and `temporal_ensemble_coeff` in `configs/baxter.yaml` to match.

## 6. Validation

- **Split by episode, not by frame.** Hold out episodes 4, 14, 24, 34 and 44 (5 of 51); train on the rest.
- Report the following on held-out episodes, in physical units after unnormalizing:
  - position error (mm) at k = 0, 25 and 99;
  - rotation error (deg) at the same k;
  - gripper MAE.
- **Transform check before training:** for a few frames, rebuild absolute goals with
  `labels.apply_actions(targets, state[:3], state[3:7])` and confirm they match `action[t+k]` within 1e-5.
  This catches errors in quaternion order (xyzw) or rotation side.
- Pick the checkpoint by held-out error, then confirm it on the robot. Offline loss alone does not
  establish the final result.

## 7. Deliverable

Push the model to a private Hub repo, e.g. `andnetdeboer/act_CupToContainer`, containing:

- weights and the policy config;
- normalization stats for the 17-D state and the 7-D targets;
- `action_definition.json`:

```json
{"dataset": "andnetdeboer/CupToContainer", "arm": "left", "fps": 25, "chunk_size": 100,
 "action": ["dx", "dy", "dz", "rx", "ry", "rz", "trigger"],
 "action_frame": "relative to TCP at observation time, base axes; R_goal = exp(r) R_tcp",
 "state": ["tcp.xyz", "R_tcp[:, :2] (column-major)", "gripper", "left_s0..w2"],
 "cameras": ["left", "third_eye"], "quaternion_order": "xyzw"}
```

- the held-out metrics from §6 and the training command or config used.

## 8. Running it

`inference serve` loads the pushed model through LeRobot (`make_pre_post_processors`; a checkpoint
without LeRobot processor files is normalized from its `normalization_stats.json`). It builds the
17-D state from the robot's topics with `labels.model_state` and publishes unnormalized `(100, 7)`
chunks, anchored on the TCP observed. `inference run` executes them; see the README.
