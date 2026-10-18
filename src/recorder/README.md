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
| `recording.image_transport` | `raw` (generic default) or `compressed` (Baxter default): record JPEG copies on `<camera topic>/compressed` |
| `recording.jpeg_quality` | JPEG quality, 1–100 (default 90); only used with compressed recording |

The configured `topics.images` names remain the **raw live camera topics** used by visualization
and inference. With compressed recording, a session-owned ROS node publishes JPEG copies and
only those image copies are bagged. Original image headers are preserved. The exporter accepts
both old raw bags and new compressed bags, resolving the `/compressed` suffix automatically.
Existing bags and their saved configs are never rewritten. Set `recording.image_transport: raw`
for lossless raw recording. JPEG is for colour/mono8 cameras, not depth. If a JPEG publisher is
already running (for example a rollout recorder), stop it before starting this recorder, to avoid
two publishers generating duplicate frames on the same `/compressed` topic.

The additional Baxter cuff/safety overrides are **recorded commands only**: the recorder never
publishes robot motion, speed, or safety settings. Diagnostics and navigator events provide context
for hand-guided demos. Pure kinesthetic and bimanual demos still require a suitable action-label
exporter; compressed-image support does not add that label mode.

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

**digital_io** -- Configured Baxter navigator buttons drive recording (left show/OK/back by default).
The cuff squeeze enables hand guidance, independently of recording. The left inner cuff light
mirrors recording state (on = recording).

## Session Controls

| Action | Effect |
|---|---|
| Record | Start or stop the current episode |
| Save | Save the episode and advance the counter |
| Discard | Delete the episode, keep the counter |
| Ctrl+C | End the session |
