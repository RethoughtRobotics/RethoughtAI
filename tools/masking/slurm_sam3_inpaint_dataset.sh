#!/usr/bin/env bash
# Build a SAM3+inpaint dataset variant on an allocated node.
set -euo pipefail

SRC_DATASET="${1:?Usage: $0 <source_dataset> <output_dataset> [device]}"
OUT_DATASET="${2:?Usage: $0 <source_dataset> <output_dataset> [device]}"
DEVICE="${3:-cuda}"

ROOT_DIR="${ROOT_DIR:-$PWD}"
VENV="${VENV:-$ROOT_DIR/.venv}"

PROMPT="${PROMPT:-person}"
THRESHOLD="${THRESHOLD:-0.3}"
DILATION="${DILATION:-3}"
INPAINT_RADIUS="${INPAINT_RADIUS:-6}"
CAMERA="${CAMERA:-observation.images.third_eye}"
MAX_VIDEOS="${MAX_VIDEOS:-0}"
MAX_FRAMES="${MAX_FRAMES:-0}"
WORKERS="${WORKERS:-4}"

if [[ ! -d "$SRC_DATASET" ]]; then
  echo "Source dataset missing: $SRC_DATASET" >&2
  exit 2
fi

source "$VENV/bin/activate"

CMD=(python "$ROOT_DIR/tools/masking/sam3_inpaint_dataset.py"
  --source "$SRC_DATASET"
  --out "$OUT_DATASET"
  --device "$DEVICE"
  --prompt "$PROMPT"
  --threshold "$THRESHOLD"
  --dilation "$DILATION"
  --inpaint-radius "$INPAINT_RADIUS"
  --workers "$WORKERS")

if [[ -n "$CAMERA" ]]; then
  CMD+=(--camera "$CAMERA")
fi
if [[ "$MAX_VIDEOS" != "0" ]]; then
  CMD+=(--max-videos "$MAX_VIDEOS")
fi
if [[ "$MAX_FRAMES" != "0" ]]; then
  CMD+=(--max-frames "$MAX_FRAMES")
fi

"${CMD[@]}"

echo "Done: $OUT_DATASET"
