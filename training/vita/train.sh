#!/bin/bash
# Train vita with vita/config.yaml (see vita/README.md). Any train.py option passes through; without
# --mode it only prints the command.
#   training/vita/train.sh --mode tmux --gpu 0
#   training/vita/train.sh --mode queue --cron
#   training/vita/train.sh --dataset cup_to_container --set steps=5000 --mode run
cd "$(dirname "$0")/../.." && exec uv run python training/train.py vita "$@"
