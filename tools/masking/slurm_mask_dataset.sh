#!/usr/bin/env bash
# Build a masked LeRobot dataset variant on Quest using an allocated GPU node.
set -euo pipefail

SRC_DATASET="${1:?Usage: $0 <source_dataset> <output_dataset> [backend] [device]}"
OUT_DATASET="${2:?Usage: $0 <source_dataset> <output_dataset> [backend] [device]}"
BACKEND="${3:-maskrcnn}"
DEVICE="${4:-cuda}"

ROOT_DIR="${ROOT_DIR:-$PWD}"
VENV="${VENV:-$ROOT_DIR/.venv}"

if [[ ! -d "$SRC_DATASET" ]]; then
  echo "Source dataset missing: $SRC_DATASET" >&2
  exit 2
fi

source "$VENV/bin/activate"
python "$ROOT_DIR/tools/masking/mask_dataset.py" \
  --source "$SRC_DATASET" \
  --out "$OUT_DATASET" \
  --backend "$BACKEND" \
  --device "$DEVICE" \
  --threshold 0.5 \
  --fill 127 \
  --dilation 2

echo "Done: $OUT_DATASET"