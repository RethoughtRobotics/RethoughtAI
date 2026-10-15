#!/bin/bash
#SBATCH -J sam3_masks_sweeptrash
#SBATCH -A e52938
#SBATCH -p gengpu
#SBATCH --gres=gpu:a100:2
#SBATCH --exclude=qgpu2014
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH -t 04:00:00
#SBATCH -o /projects/e52938/kmy2091/logs/%x_%j.out
set -euo pipefail
WS=/projects/e52938/kmy2091
export PATH="$WS/cache/ffmpeg7/bin:$PATH" LD_LIBRARY_PATH="$WS/cache/ffmpeg7/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export HF_HOME="$WS/cache/huggingface" HF_TOKEN_PATH="$HOME/.cache/huggingface/token" PYTHONUNBUFFERED=1
nvidia-smi -L
"$WS/cache/sam3-venv/bin/python" "$WS/RethoughtAI-adapter/tools/masking/sam3_masks.py" \
    --source "$WS/datasets/SweepTrash_Kin-Full" --out "$WS/cache/vor/masks/SweepTrash_Kin-Full" \
    --camera observation.images.third_eye --workers 4 2>&1 | grep -v -i "warn\|rope_theta\|Loading weights"
