#!/bin/bash
# Train groot with groot/config.yaml (see groot/README.md). Any train.py option passes through; without
# --mode it only prints the command.
#   training/groot/train.sh --mode tmux --gpu 0
#   training/groot/train.sh --mode queue --cron
#   training/groot/train.sh --dataset cup_to_container --set steps=5000 --mode run
cd "$(dirname "$0")/../.." && exec uv run python training/train.py groot "$@"
