#!/bin/bash
#SBATCH -J sam3_masks_kinpose
#SBATCH -A e52938
#SBATCH -p gengpu
#SBATCH --gres=gpu:a100:1
#SBATCH --exclude=qgpu2014
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH -t 04:00:00
#SBATCH -o /projects/e52938/kmy2091/logs/%x_%j.out
set -euo pipefail
WS=/projects/e52938/kmy2091
export PATH="$WS/cache/ffmpeg7/bin:$PATH" LD_LIBRARY_PATH="$WS/cache/ffmpeg7/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export HF_HOME="$WS/cache/huggingface" HF_TOKEN_PATH="$HOME/.cache/huggingface/token" PYTHONUNBUFFERED=1
nvidia-smi -L
"$WS/cache/sam3-venv/bin/python" "$WS/RethoughtAI-adapter/tools/masking/sam3_masks.py" \
    --source "$WS/cache/lerobot/hub/datasets--andnetdeboer--BlueCupRedContainer_kin_pose/snapshots/e2594aed18862a36bac83d27bd6217faefa709c5" --out "$WS/cache/vor/masks/BlueCupRedContainer_kin_pose" \
    --camera observation.images.third_eye --workers 1 2>&1 | grep -v -i "warn\|rope_theta\|Loading weights"
