#!/bin/bash
# One shard of a video_remove.py method (METHOD, NAME) over episode-aligned segments; SHARD=K/M.
#SBATCH -J vor_shard
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
$V/venv-${METHOD:?}/bin/python $WS/RethoughtAI-adapter/tools/masking/video_remove.py --method "$METHOD" ${STEPS:+--steps "$STEPS"} \
  --source $WS/datasets/${NAME:?} --masks $V/masks/$NAME --out $WS/datasets/${NAME}_inpainted_$METHOD \
  --camera observation.images.third_eye --segment-frames 1500 --shard "$SHARD" --parts-dir $V/parts/${NAME}_$METHOD \
  --workers "${WORKERS:-2}" 2>&1 | grep --line-buffered -v -i -E "warn|it/s\]|fg_emb|noise shape"
