#!/bin/bash
# Train smolvla with smolvla/config.yaml (see smolvla/README.md). Any train.py option passes through; without
# --mode it only prints the command.
#   training/smolvla/train.sh --mode tmux --gpu 0
#   training/smolvla/train.sh --mode queue --cron
#   training/smolvla/train.sh --dataset cup_to_container --set steps=5000 --mode run
cd "$(dirname "$0")/../.." && exec uv run python training/train.py smolvla "$@"
