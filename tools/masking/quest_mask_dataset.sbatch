#!/bin/bash
#SBATCH -J mask_dataset
#SBATCH -A your_allocation
#SBATCH -p gengpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH -t 08:00:00
#SBATCH -o mask_dataset_%j.out

set -euo pipefail

# Edit these before submission.
ROOT_DIR=${ROOT_DIR:-$HOME/FinalProject/RethoughtAI}
SRC_DATASET=${SRC_DATASET:-$ROOT_DIR/datasets/BlueCupRedContainer_kin_lead640ms}
OUT_DATASET=${OUT_DATASET:-$ROOT_DIR/datasets/BlueCupRedContainer_kin_lead640ms_maskrcnn_v1}
BACKEND=${BACKEND:-maskrcnn}
DEVICE=${DEVICE:-cuda}

cd "$ROOT_DIR"
source .venv/bin/activate

bash tools/masking/slurm_mask_dataset.sh "$SRC_DATASET" "$OUT_DATASET" "$BACKEND" "$DEVICE"