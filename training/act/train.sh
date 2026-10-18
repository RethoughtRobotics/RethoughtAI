#!/bin/bash
# Train act with act/config.yaml (see act/README.md). Any train.py option passes through; without
# --mode it only prints the command.
#   training/act/train.sh --mode tmux --gpu 0
#   training/act/train.sh --mode queue --cron
#   training/act/train.sh --dataset cup_to_container --set steps=5000 --mode run
cd "$(dirname "$0")/../.." && exec uv run python training/train.py act "$@"
