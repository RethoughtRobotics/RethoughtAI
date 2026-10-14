#!/bin/bash
#SBATCH -J sam3_inpaint_dataset
#SBATCH -A your_allocation
#SBATCH -p gengpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH -t 24:00:00
#SBATCH -o sam3_inpaint_dataset_%j.out

set -euo pipefail

ROOT_DIR=${ROOT_DIR:-$HOME/FinalProject/RethoughtAI}
SRC_DATASET=${SRC_DATASET:-$ROOT_DIR/datasets/BlueCupRedContainer_kin_lead640ms}
OUT_DATASET=${OUT_DATASET:-$ROOT_DIR/datasets/BlueCupRedContainer_kin_lead640ms_sam3_inpaint_v1}
DEVICE=${DEVICE:-cuda}

# Optional overrides:
# export PROMPT=person
# export THRESHOLD=0.3
# export DILATION=3
# export INPAINT_RADIUS=6
# export CAMERA=observation.images.third_eye
# export MAX_VIDEOS=0
# export MAX_FRAMES=0

cd "$ROOT_DIR"
source .venv/bin/activate

bash tools/masking/slurm_sam3_inpaint_dataset.sh "$SRC_DATASET" "$OUT_DATASET" "$DEVICE"
