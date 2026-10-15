#!/bin/bash
# One shard of EffectErase (20 steps) over episode-aligned segments of SweepTrash_Kin-Full third_eye.
#SBATCH -J effecterase_shard
#SBATCH -A e52938
#SBATCH -p gengpu
#SBATCH --gres=gpu:a100:2
#SBATCH --exclude=qgpu2014
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH -t 06:00:00
#SBATCH -o /projects/e52938/kmy2091/logs/%x_%j.out
set -euo pipefail
: "${SHARD:?K/M}"
WS=/projects/e52938/kmy2091; V=$WS/cache/vor
export PATH="$WS/cache/ffmpeg7/bin:$PATH" LD_LIBRARY_PATH="$WS/cache/ffmpeg7/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" PYTHONUNBUFFERED=1
nvidia-smi -L
$V/venv-effecterase/bin/python $WS/RethoughtAI-adapter/tools/masking/video_remove.py --method effecterase --steps 20 \
  --source $WS/datasets/SweepTrash_Kin-Full --masks $V/masks/SweepTrash_Kin-Full --out $WS/datasets/SweepTrash_Kin-Full_inpainted_effecterase \
  --camera observation.images.third_eye --segment-frames 1500 --shard "$SHARD" --parts-dir $V/parts/SweepTrash_Kin-Full_effecterase \
  --workers 2 2>&1 | grep --line-buffered -v -i -E "warn|it/s\]|fg_emb|noise shape"
