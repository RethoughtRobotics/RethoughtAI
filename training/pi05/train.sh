#!/bin/bash
# Train pi05 with pi05/config.yaml (see pi05/README.md). Any train.py option passes through; without
# --mode it only prints the command.
#   training/pi05/train.sh --mode tmux --gpu 0
#   training/pi05/train.sh --mode queue --cron
#   training/pi05/train.sh --dataset cup_to_container --set steps=5000 --mode run
cd "$(dirname "$0")/../.." && exec uv run python training/train.py pi05 "$@"
