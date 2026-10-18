#!/bin/bash
# Train multi_task_dit with multi_task_dit/config.yaml (see multi_task_dit/README.md). Any train.py option passes through; without
# --mode it only prints the command.
#   training/multi_task_dit/train.sh --mode tmux --gpu 0
#   training/multi_task_dit/train.sh --mode queue --cron
#   training/multi_task_dit/train.sh --dataset cup_to_container --set steps=5000 --mode run
cd "$(dirname "$0")/../.." && exec uv run python training/train.py multi_task_dit "$@"
