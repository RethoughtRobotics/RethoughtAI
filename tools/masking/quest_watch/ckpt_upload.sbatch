#!/bin/bash
# Upload local checkpoints to the Hub (repos made public), then verify every file (presence, size, hash). No deletion.
#SBATCH -J ckpt_upload
#SBATCH -A e52938
#SBATCH -p short
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH -t 04:00:00
#SBATCH -o /projects/e52938/kmy2091/logs/%x_%j.out
set -euo pipefail
WS=/projects/e52938/kmy2091
export HF_HOME=$WS/cache/huggingface HF_TOKEN_PATH=$HOME/.cache/huggingface/token PYTHONUNBUFFERED=1 HF_HUB_ENABLE_HF_TRANSFER=0
$WS/rethought-lerobot/.venv/bin/python $WS/cache/vor/ckpt_upload.py upload
$WS/rethought-lerobot/.venv/bin/python $WS/cache/vor/ckpt_upload.py verify
