#!/bin/bash
# Hold an H100 now; start training as soon as the chain has pushed the dataset and written its commit to $SHA_FILE.
#SBATCH -A e52938
#SBATCH -p gengpu
#SBATCH --gres=gpu:h100:1
#SBATCH --exclude=qgpu2014
#SBATCH --cpus-per-task=32
#SBATCH --mem=96G
#SBATCH -t 04:00:00
#SBATCH -o /projects/e52938/kmy2091/logs/%x_%j.out
set -euo pipefail
: "${SHA_FILE:?}" "${SCRIPT:?}" "${REPO:?}" "${RUN:?}"
echo "$(date +%T) holding $(nvidia-smi -L | head -1); waiting for $SHA_FILE"
for i in $(seq 1 360); do [[ -s "$SHA_FILE" ]] && break; sleep 30; done
[[ -s "$SHA_FILE" ]] || { echo "dataset never became ready" >&2; exit 3; }
SHA=$(cat "$SHA_FILE"); echo "$(date +%T) dataset ready at $SHA; starting $RUN"
cd /projects/e52938/kmy2091
exec bash "$SCRIPT" "$REPO" "$SHA" ${VARIANT_ARG:-} "$RUN"
