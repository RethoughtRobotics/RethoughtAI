
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

<img src="overview.png" alt="RAI" width="100%" />

**[recorder](recorder/)** — Record demonstrations to MCAP with live Rerun visualization. Trigger via physical buttons (Baxter, Sawyer) or keyboard (any robot).

**[adapter](adapter/)** — Convert MCAP recordings to LeRobot v3 datasets, quality-filter episodes, and push to HuggingFace Hub.

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

# 3. Train with LeRobot (standard LeRobot workflow)
# https://github.com/huggingface/lerobot

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
- **Per policy type settings** live in the config's `inference.policies` (`act`, `smolvla`).
  On a laptop RTX 500 (4 GB) next to the sim: ACT ~30 ms, run every step; SmolVLA with RTC ~450 ms,
  run every 10 steps with `execution_horizon` 25 (it must exceed the delay in steps, ~13).

**Shadow rollout** (`launch/shadow.launch.py`): the policy drives a simulated arm (BaxFlow-T)
while it watches a held-out episode's recorded cameras, played by `ros2 bag play`. The runtime moves
the arm to the episode's start pose, then seeks the bag player to the episode's first step and
resumes it.

```bash
# bringbackbaxter: ./run_sim_teleop.sh --left --policy --objects
ros2 launch launch/shadow.launch.py model:=yourname/your-model episode:=recorder/data/CupToContainer/episode_004
ros2 service call /policy/left/engage std_srvs/srv/SetBool "{data: true}"
```

---

## Part of Rethought Robotics

<p>
  <a href="https://github.com/RethoughtRobotics/baxter-zenoh">
    <img src="https://gh-card.dev/repos/RethoughtRobotics/baxter-zenoh.svg?fullname=" width="40%" />
  </a>
  <a href="https://github.com/RethoughtRobotics/BaxterSDK">
    <img src="https://gh-card.dev/repos/RethoughtRobotics/BaxterSDK.svg?fullname=" width="40%" />
  </a>
</p>
