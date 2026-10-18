
<h1 align="center">
  <font color="#8e7cc3">R</font>ethought<font color="#8e7cc3">A</font><font color="#8e7cc3">I</font>
</h1>

<p align="center">
  <img src="https://img.shields.io/badge/ROS2-compatible-brightgreen?logo=ros&logoColor=white" />
  <img src="https://img.shields.io/badge/LeRobot-v3-brightgreen?logo=huggingface&logoColor=white" />

</p>

<p align="center"><b>RAI - Record · Adapt · Infer</b></p>

  A ROS2-native pipeline for robot learning. Collect demonstrations, convert to training-ready datasets, and deploy trained policies.


---

## About

RAI is a 3-step pipeline for learning from demonstrations with ROS2: record, adapt, and infer.

---

## The Pipeline

<img src="assets/overview.png" alt="RAI" width="100%" />

**[recorder](recorder/)** — Record demonstrations to MCAP with live Rerun visualization. Trigger via physical buttons (Baxter, Sawyer) or keyboard (any robot).

**[adapter](adapter/)** — Convert MCAP recordings to LeRobot v3 datasets, quality-filter episodes, and push to HuggingFace Hub.

**[training](training/)** — Train ACT, SmolVLA, VITA, Multi-Task DiT, GR00T or pi0.5 on those datasets, from composable YAML configs, with an unattended queue and watchdog.

**[inference](inference/)** — Deploy a trained policy from HuggingFace back to your ROS2 robot.

---

## Supported Robots

| Robot | Config | Trigger |
|---|---|---|
| Baxter | [baxter.yaml](recorder/configs/baxter.yaml) | Digital IO buttons |
| Sawyer | coming soon | Digital IO buttons |
| Generic ROS2 | [generic.yaml](recorder/configs/generic.yaml) | Keyboard |

---

## Installation

Requires [uv](https://docs.astral.sh/uv/).

```bash
git clone https://github.com/RethoughtRobotics/RethoughtAI.git
cd RethoughtAI
uv sync
```

All three packages install into a single virtual environment.

```bash
uv run recorder --help
uv run adapter --help
uv run inference --help
```

---

## Quick Start

```bash
# 1. Record demonstrations
uv run recorder record --config baxter.yaml

# 2. Convert to LeRobot dataset and push to HuggingFace
uv run adapter convert --config configs/baxter.yaml \
  --input ./recordings \
  --push-to-hub yourname/your-dataset

# 3. Train a policy (see training/README.md)
training/act/train.sh --mode tmux --gpu 0

# 4. Deploy a policy (ACT, SmolVLA, any LeRobot checkpoint): two ROS 2 nodes
uv run inference serve baxter.yaml --model yourname/your-model   # policy node (GPU)
uv run inference run baxter.yaml --model yourname/your-model     # runtime: drives the teleop
```

### Inference on ROS 2

```
cameras, /robot/joint_states,      ┌──────────────────────┐  /policy/left/chunk        ┌──────────────┐  /policy/left/target  ┌────────┐
gripper state, /teleop/left/tcp ──▶│ policy node (serve)  │──────────────────────────▶│ runtime (run) │──────────────────────▶│ teleop │──▶ arm
                                   │ LeRobot policy + RTC │  moveit_msgs/             │ merge chunks  │  /policy/left/trigger │ servo  │
                                   └──────────────────────┘  RobotTrajectory           └──────────────┘                       └────────┘
                                                             stamped at observation time
```

- **The policy node** runs any LeRobot checkpoint on the robot's topics in a worker thread and publishes
  each action chunk as a `moveit_msgs/RobotTrajectory`, plus `nav_msgs/Path` for RViz. The chunk's
  header stamp is the time of the observation it was computed from. Loading, normalization and
  tokenization are LeRobot's own (`make_pre_post_processors`). For flow policies (SmolVLA) the node
  uses LeRobot's Real-Time Chunking.
- **The runtime** merges the chunks by time and streams a target to the teleop at 100 Hz. Merge
  modes: ACT temporal ensembling, `latest` (for RTC), or a linear crossfade. Because each chunk
  carries its observation time, inference latency only shifts where execution starts in the chunk.
- **The two are separate processes,** so inference never blocks the control loop. Putting the GPU
  on another machine is a ROS networking question, not new code.
- **Per policy type settings** live in the config's `inference.policies` (`act`, `smolvla`, `vita`,
  `multi_task_dit`, `groot`). On a laptop RTX 500 (4 GB) next to the sim: ACT ~30 ms, run every step;
  SmolVLA with RTC ~450 ms, run every 10 steps with `execution_horizon` 25 (it must exceed the delay
  in steps, ~13); MultiTask DiT ~100 ms, every 4 steps.
- **Two checkpoint layouts** are told apart by their feature shapes. A 17-D state and 7-D actions
  relative to the TCP (`training/TRAINING.md`'s transform: ACT, SmolVLA, VITA, MultiTask DiT), or the dataset as
  exported, a 15-D state and 8-D absolute goals (GR00T, kinesthetic cup VITA), which the policy node makes relative to the
  TCP it observed. Images are letterboxed to a checkpoint's image size when it differs from the
  camera's, as `BlueCupToContainer`'s 224 x 224 frames were.

**Running a checkpoint** from `models/<name>_*/step-<N>/` opens tmux. When a prefix like
`--smolvla` matches several models, give the full name. The top pane runs the policy
(`launch/policy.launch.py`). The bottom pane has `./start` typed, ready to connect the policy to the
arm. `./stop` holds the arm where it is. `./home` sends it to the untuck pose with the gripper open,
as the controller's B button (with `./start` still on, the policy resumes fresh from there).

```bash
# bringbackbaxter: ./run_teleop.sh --left --policy (robot) or ./run_sim_teleop.sh --left --policy
./run_inference --act 100000                  # real robot: its cameras; sim: shadow of episode 4
./run_inference --smolvla_CupToContainer 30000 --episode 24   # sim: shadow of episode 24
./run_inference --smolvla_CupToContainer_wrist  # newest checkpoint, wrist camera only
```

With the robot's cameras up, the policy watches them. With none (the sim), it runs a **shadow
rollout**: it drives the arm while the policy watches a recorded episode's cameras, played by
`ros2 bag play`. `--episode` with live cameras is refused: the bag's frames would mix with theirs. The arm first moves to the episode's start pose, then the bag plays from the
episode's first step.

### Pinned kinesthetic checkpoints (2026-10-14)

Downloaded into `models/<repo-name>/<branch>/`, not the moving Hub `main`. All eight files per
checkpoint were checked against the Hub file sizes. These folders remain ignored by Git.

| Repository (`andnetdeboer/` prefix) | Local branch | Pinned commit | Deployment |
|---|---|---|---|
| `vita_BlueCupRedContainer_kin_pose` | `step-070000` | `603760a332302cfb55037caf0dd4abd8eb762ed9` | Left arm, 15-D state / 8-D action |
| `vita_BlueCupRedContainer_kin_lead640ms` | `step-080000` | `4cef9dd9b7594f8ac6eae27236e759fadd79012f` | Left arm, 15-D state / 8-D action |
| `vita_SweepTrash_Kin` | `step-050000` | `f1b5e0f11e1ded03fe76b7b5d3c66295c0edbff9` | Coordinated two-arm pose-only pipeline; offline validated, hardware rollout pending |
| `multi_task_dit_kin_pose_combo_30k_r5h100` | `step-030000` | `e18064d84d62a480cad64683c2e71c53d6857cc0` | Left arm DiT, `kin_pose`; `multi_task_dit_absolute` settings |
| `multi_task_dit_kin_pose_sam3_inpaint_aug_30k_r1` | `step-030000` | `f8fc9cb367ed521b593af3043e7fe71b7d354390` | Left arm DiT, `kin_pose_sam3_inpaint_v1` |
| `multi_task_dit_kin_lead640ms_sam3_inpaint_aug_30k_r1` | `step-030000` | `42417dd7dfcbaad5c53e2974e168ea4bb7d1e1e7` | Left arm DiT, `kin_lead640ms_sam3_inpaint_v1` (lead learned: no extra lead) |
| `vita_kin_pose_mask_spatial_30k_r5` | `step-030000` | `f904216f5b648c0ea2a7f2a6dffeed06f2c9f437` | Left arm VITA, `kin_pose` |
| `vita_kin_pose_mask_spatial_30k_r5-ema` | `step-030000` | `0affdfa3da41764b5931ea69d301722c26a8a46e` | Left arm VITA, `kin_pose`, EMA weights |
| `vita_kin_pose_drop_top30_30k_r5-ema` | `step-030000` | `71ad0cb4aa30cbc18c0987cc38436b0d02590566` | Left arm VITA, `kin_pose`, EMA weights |
| `multi_task_dit_SweepTrash_Kin-Full_inpainted_effecterase_lead640ms_aug_30k_r1` | `step-030000` | `697dcb9bfa6759027f70d77c1f023ada576fc441` | Two-arm SweepTrash DiT, `lead640ms` (no run-time lead); `multi_task_dit_bimanual` settings |
| `multi_task_dit_BlueCupRedContainer_kin_pose_inpainted_minimax_lead640ms_aug_30k_r1` | `step-030000` | `e5f62fc9b794fb7aa359e97b2b634b6ba7d81d01` | Left arm DiT, `kin_pose` minimax inpainting, `lead640ms` (no run-time lead); `multi_task_dit_absolute` settings |
| `multi_task_dit_SweepTrash_Kin-Full_inpainted_minimax_lead640ms_aug_30k_r1` | `step-030000` | `3f1ca67b0e6920283aa89b87be34403e7c8fa77c` | As above, `minimax` inpainting dataset |
| `multi_task_dit_SweepTrash_Kin-Full_inpainted_lead640ms_aug_30k_r1` | `step-030000` | `7c7cff626a0c6596b2b6f8504853a1da2bb7dcfa` | As above, plain `inpainted` dataset |
| `vita_SweepTrash_Kin-Full_inpainted_effecterase_lead640ms_aug_30k_r1` | `step-030000` | `439b6c56e53d14933867762e1a08d19cb13ae551` | Two-arm SweepTrash VITA, `lead640ms`; `vita_bimanual` settings |
| `vita_SweepTrash_Kin-Full_inpainted_effecterase_lead640ms_aug_30k_r1-ema` | `step-030000` | `439b6c56e53d14933867762e1a08d19cb13ae551` | As above, EMA weights (`pretrained_model_ema`) |
| `vita_SweepTrash_Kin-Full_inpainted_minimax_lead640ms_aug_30k_r1` | `step-030000` | `f96e6314e1a8bcc20fe388c82407be902af2011b` | Two-arm SweepTrash VITA, `minimax` inpainting, `lead640ms`; `vita_bimanual` settings |
| `vita_SweepTrash_Kin-Full_inpainted_minimax_lead640ms_aug_30k_r1-ema` | `step-030000` | `467cd3d9fcd8d23b4b7e4445c618efba5f61bc6d` | As above, EMA weights (own repo) |
| `smolvla_SweepTrash_Kin-Full_inpainted_lead640ms_30k_r1` | `step-030000` | `e6460cce7ef63c9d825578548a116c49ce6a705c` | Two-arm SweepTrash SmolVLA, `lead640ms`; `smolvla_bimanual` settings |
| `smolvla_SweepTrash_Kin-Full_inpainted_30k_r1` | `step-030000` | `d2e8c604105e6fd18311945556260585a8afdec3` | As above, without the 640 ms label lead (`SweepTrash_Kin-Full_inpainted`) |

**Cup models use the existing pipeline, including `./start`, `./stop`, cuff correction and homing.**
Bring up the left-arm policy teleop first, as above, with the live cameras and the same calibrated
TCP/tool offset as the demonstrations. From this directory:

```bash
# Compatibility checks only: no ROS node, weights or robot motion.
# Source launch/env.sh first if ROS/Baxter Python messages are not on your environment's path.
source launch/env.sh
.venv/bin/inference check models/vita_BlueCupRedContainer_kin_pose/step-070000
.venv/bin/inference check models/vita_BlueCupRedContainer_kin_lead640ms/step-080000

# Opens the existing tmux pipeline; ./start is typed but NOT executed.
./run_inference --vita_BlueCupRedContainer_kin_pose 70000
# Alternative run (replaces the previous pipeline):
./run_inference --vita_BlueCupRedContainer_kin_lead640ms 80000
```

- Feature shapes select `inference.policies.vita_absolute` in `configs/baxter.yaml`; old relative
  VITA retains its existing every-frame temporal ensemble. Both cup models use left wrist and
  third-eye images, not the right wrist, at their exported camera resolutions.
- State is `[TCP xyz, quaternion xyzw, gripper, seven joints]`. Action is absolute base-frame
  `[xyz, quaternion xyzw, gripper]`. `LeRobotPolicy` converts to the internal relative format using
  the **observed** TCP, then `PolicyNode` reconstructs the same absolute targets. The TCP is not
  added twice; no change to the teleop's servo/IK is required.
- The trained horizon is 32 frames (1.28 s at 25 Hz); its saved `n_action_steps` is 16. Deployment
  overrides only the returned slice to 32 and predicts every 16 frames (0.64 s), using `latest`.
  The tail covers inference latency: returning only 16 ends at 0.60 s and leaves a coverage gap.
  The next prediction replaces the tail; the runtime skips elapsed steps by observation timestamp,
  as it already did. This is not a guarantee that exactly 16 targets execute under every latency.
- Dataset metadata confirms `pose` has `lead_steps=0`; `lead640ms` has `lead_steps=16`.
  **Do not add a 0.64 s timestamp offset, slice another 16 steps, or enable relative-action RTC.**
  The future target shift is already learned. The two policies use the same schedule for comparison.
- Automatic shadow playback of the old `CupToContainer` episode is refused for these kinesthetic
  cup model names. A proper kinesthetic shadow mode still needs measured start-pose loading and
  raw/JPEG camera playback compatibility. Test live initially with engagement off, then a supervised
  short rollout with clear workspace and `./stop`/physical emergency stop available.

**SweepTrash is not an 8-D single-arm checkpoint.** Its train config references
`andnetdeboer/SweepTrash_Kin`; local dataset feature names and source metadata confirm:

- Arms ordered **left, right**; state is 14 values per arm: TCP xyz + quaternion xyzw + seven joints.
- Actions are seven absolute pose values per arm, left then right (14 total).
- No gripper channels: tools stay held; never invent an opening command from the last pose value.
- Three cameras; `lead_steps=1` (40 ms), not the cup model's 640 ms lead.
- The dataset preserves sign-continuous quaternions. The single-arm `raw_state()` rule `qy >= 0`
  cannot blindly be reused: SweepTrash's left `qy` crosses zero in the actual data.

**SweepTrash now has its own coordinated configuration, `configs/baxter_bimanual.yaml`.**
The compatibility check still rejects it with the single-arm config, before replacing an existing
run. `run_inference` selects the two-arm config automatically for the exact SweepTrash model name.

```bash
# Compatibility only: no node or motion.
source launch/env.sh
.venv/bin/inference check models/vita_SweepTrash_Kin/step-050000 --config configs/baxter_bimanual.yaml

# When ready for a supervised test, from ../bringbackbaxter:
./run_teleop.sh --both --policy --no-gripper
# Then from RethoughtAI (opens tmux; DOES NOT execute ./start):
./run_inference --vita_SweepTrash_Kin 50000
```

- One model server observes both TCPs, all 14 joints and all three cameras. State ordering matches
  the export exactly, with per-arm sign-continuous quaternions and no invented gripper state.
- Each chunk contains both absolute TCP trajectories in **one** `RobotTrajectory`, stamped at the
  same observation time. The returned slice is 32 targets, predicted every 16 frames with `latest`
  merge; no added label lead or relative-action RTC. Predicted quaternions are normalized before
  publication; malformed/nonfinite chunks are rejected atomically.
- The runtime uses one `/policy/both/engage` service and `/policy/both/engaged` signal, with target
  outputs on `/policy/left/target` and `/policy/right/target`. Both feedback streams must be fresh
  before either engages. Either stale TCP releases both; missing chunk coverage holds both.
- A home event or cuff squeeze on either side clears both plans. Cuff handling conservatively
  releases the coordinated run; both cuffs must be released and fresh feedback settled before
  re-engagement. In-flight predictions from before a control transition are discarded.
- No gripper subscription or command is used by this model. **Use `--no-gripper` on the teleop**
  so VR triggers and homing do not open the tool grippers; cuff gripper buttons remain manual.
- In the opened control pane, `CONFIG` already selects the shared service for `./start`/`./stop`.
  Outside that pane, use `CONFIG=configs/baxter_bimanual.yaml ./start` (or `./stop`, `./home`).
  Two-arm `./home` requests stop first and sends both home topics. Release the VR deadman while
  homing so it cannot request re-engagement. Automatic shadowing of old cup recordings is refused;
  bimanual replay/start-pose playback is not implemented.

These checks validate software coordination, **not collision-free execution or task success**.
Check TCP calibration, tool clearance and both servo instances in simulation before a short,
supervised hardware rollout. Keep `./stop` and the physical emergency stop available.

**Validation performed (no robot commands):** 29 focused regression tests passed
(`test_kinesthetic_inference.py`, `test_chunk_merge.py`, `test_kinesthetic_export.py`). Actual GPU
inference used the first exported state and decoded real AV1 video frames (CPU `libdav1d`, since
OpenCV's AV1 decoder failed here). Cup state reconstruction error was zero; outputs were finite
32-step internal chunks, warmed p50 ~17.1 ms / ~16.6 ms respectively (five measured calls after two
warmups). SweepTrash's model and saved processors loaded and returned finite `(16, 14)` raw actions;
this verifies loading only, **not bimanual deployment or task success**. Full physical rollouts have
not been run. The pre-existing untracked `tests/test_inference_runtime.py` imports nonexistent APIs
and was left untouched rather than treated as a test of this pipeline.

**Bimanual validation added:** paired policy/runtime, launcher and control tests use fake nodes and
mocked shell commands, without starting ROS or moving hardware. Actual GPU inference through the
integrated wrapper on SweepTrash's first exported state and real decoded camera frames returned
finite `(32, 14)` absolute actions; state reconstruction error was zero. The full focused suite and
real `inference check` pass. A live simulation/hardware rollout has not been performed.

