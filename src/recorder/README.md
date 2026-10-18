# recorder

Records robot demonstrations as ROS2 bags, one bag per episode. Sessions contain multiple episodes and are stored under `recorder/data/`.

## Requirements

- ROS2 (Kilted or later), workspace sourced
- Python 3.12+
- For digital I/O trigger: Baxter workspace sourced (`baxter_env`)
- For visualization: Rerun viewer

## Installation

```bash
uv sync
```

## Usage

```bash
record record <config>
record record baxter.yaml
record record baxter.yaml --output ./recorder/data --no-viz
```

## Options

| Flag | Default | Description |
|---|---|---|
| `--output` | `./recorder/data` | Directory to write sessions into |
| `--no-viz` | off | Disable Rerun live visualization |

## Config

Configs live in `configs/`. Pass by name (`baxter.yaml`) or full path.

| Field | Description |
|---|---|
| `robot` | Robot identifier |
| `topics.images` | Camera topics and their LeRobot key names |
| `topics.state` | State topics (joint states, end effector) |
| `topics.extra` | Additional topics to record |
| `trigger.type` | `keyboard` or `digital_io` |
| `trigger.record` | Topic or key to toggle recording |
| `trigger.success` | Topic or key to save episode |
| `trigger.discard` | Topic or key to discard episode |
| `recording.fps` | Target frame rate |

## Output Structure

```
recorder/data/
  session_YYYY_MM_DD_HH_MM_SS/
    baxter.yaml
    episode_000/
      metadata.yaml
      episode_000.db3
    episode_001/
      ...
```

## Trigger Types

**keyboard** -- press keys in the terminal to control recording.

**digital_io** -- Baxter cuff buttons drive recording. The left inner cuff light mirrors recording state (on = recording).

## Session Controls

| Action | Effect |
|---|---|
| Record | Start or stop the current episode |
| Save | Save the episode and advance the counter |
| Discard | Delete the episode, keep the counter |
| Ctrl+C | End the session |
