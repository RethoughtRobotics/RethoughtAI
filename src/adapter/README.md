# adapter

Converts recorder sessions (ROS2 bags) into LeRobot v3 datasets and optionally pushes them to HuggingFace Hub.

## Requirements

- Python 3.12+
- forge-robotics
- huggingface-hub (required only for Hub upload)
- HuggingFace account and `huggingface-cli login` (required only for Hub upload)

## Installation

```bash
uv sync
```

## Usage

```bash
adapt adapt <config> <session> [<session> ...]
adapt adapt baxter.yaml recorder/data/session_2026_06_03_18_52_04
adapt adapt baxter.yaml recorder/data/session_* --hub-repo username/dataset-name
```

## Options

| Flag | Default | Description |
|---|---|---|
| `--output` | value from config | Override output directory |
| `--hub-repo` | value from config | Override HuggingFace repo ID |

## Config

Configs live in `adapter/configs/`. Pass by name (`baxter.yaml`) or full path.

| Field | Description |
|---|---|
| `robot_type` | Robot identifier written into dataset metadata |
| `fps` | Video frame rate for encoded episodes |
| `output` | Output directory for the LeRobot v3 dataset |
| `hub_repo` | HuggingFace repo ID, or `null` to skip upload |
| `camera_mapping` | Maps ROS topic camera names to LeRobot observation keys |

## Output Structure

Each session directory is scanned for `episode_*/` subdirectories. Episodes are numbered sequentially across all sessions in the order they are passed.

```
dataset/
  meta/
    info.json
    stats.json
    episodes.jsonl
    tasks.jsonl
  data/
    chunk-000/
      episode_000000.parquet
      episode_000001.parquet
  videos/
    chunk-000/
      observation.images.left/
        episode_000000.mp4
```

## HuggingFace Upload

Set `hub_repo` in the config or pass `--hub-repo` on the command line. Requires authentication:

```bash
huggingface-cli login
```

The dataset is pushed as a public dataset repository. The URL will be printed on completion.
