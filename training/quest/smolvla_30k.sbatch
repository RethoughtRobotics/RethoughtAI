#!/bin/bash
# Submit from /projects/e52938/kmy2091:
# sbatch RethoughtAI/training/quest/smolvla_30k.sbatch DATASET_ID REVISION RUN_NAME
# SmolVLA (SmolVLM2-500M backbone + flow-matching action expert), recipe of training/smolvla (smolvla_cup_run1):
# VLM weights loaded, vision encoder frozen, expert + state projection trained, chunk 50, lr 1e-4,
# 1k warmup / 30k decay, batch 64, bf16. Standard LeRobot augmentation only. Workers scale each camera to fit
# 512x512 (aspect kept) before augmenting, so SmolVLA's resize_with_pad only pads (bit-identical result).
#SBATCH --job-name=smolvla-30k
#SBATCH --account=e52938
#SBATCH --partition=gengpu
#SBATCH --gres=gpu:a100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=96G
#SBATCH --time=12:00:00

set -euo pipefail
: "${SLURM_JOB_ID:?Submit this script through Slurm; do not train on the login node}"
DS="${1:?Dataset ID required}"
REV="${2:?Pinned dataset commit SHA required}"
NAME="${3:?Unique run name required}"
[[ "$NAME" =~ ^[A-Za-z0-9_-]+$ ]] || { echo 'Unsafe run name' >&2; exit 2; }

ROOT="${VITA_WORKSPACE:-/projects/e52938/kmy2091}"
BACKEND="$ROOT/rethought-lerobot"
OUT="$ROOT/outputs/$NAME"
export HF_HOME="$ROOT/cache/huggingface"
export HF_TOKEN_PATH="${HF_TOKEN_PATH:-$HOME/.cache/huggingface/token}"
export HF_LEROBOT_HOME="$ROOT/cache/lerobot"
export UV_CACHE_DIR="$ROOT/cache/uv"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=4
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export WANDB_API_KEY="${WANDB_API_KEY:-}"
mkdir -p "$ROOT/outputs" "$ROOT/logs" "$HF_HOME" "$HF_LEROBOT_HOME"

cd "$BACKEND"
[[ -x .venv/bin/lerobot-train ]] || { echo 'Install the training environment first.' >&2; exit 2; }
nvidia-smi

# torchcodec needs an FFmpeg with a software AV1 decoder. The cluster's ffmpeg/6.1 module only has the
# hwaccel-only native av1 decoder ("Could not push packet to decoder: Function not implemented"), so use
# conda-forge FFmpeg 7.1 + libdav1d, installed once with:
#   module load mamba/24.3.0 && mamba create -p $ROOT/cache/ffmpeg7 -c conda-forge ffmpeg=7.1 dav1d
FFMPEG_PREFIX="${FFMPEG_PREFIX:-$ROOT/cache/ffmpeg7}"
[[ -e "$FFMPEG_PREFIX/lib/libdav1d.so" ]] || { echo "Missing $FFMPEG_PREFIX; install it (see above)." >&2; exit 2; }
export LD_LIBRARY_PATH="$FFMPEG_PREFIX/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

# Expected layout; override for other robots, e.g. bimanual: STATE_DIM=28 ACTION_DIM=14
# CAMERAS=observation.images.left,observation.images.right,observation.images.third_eye
export STATE_DIM="${STATE_DIM:-15}" ACTION_DIM="${ACTION_DIM:-8}"
export CAMERAS="${CAMERAS:-observation.images.left,observation.images.third_eye}"
.venv/bin/python - "$DS" "$REV" <<'PY'
import json
import os
import sys
import torch
from huggingface_hub import HfApi, hf_hub_download

assert torch.cuda.is_available(), 'Allocated GPU is not available to PyTorch'
api = HfApi()
assert api.whoami()['name'] == 'andnetdeboer', 'Authenticate as andnetdeboer first'
info = json.load(open(hf_hub_download(sys.argv[1], 'meta/info.json', repo_type='dataset', revision=sys.argv[2])))
features = info['features']
assert features['observation.state']['shape'] == [int(os.environ['STATE_DIM'])], features['observation.state']['shape']
assert features['action']['shape'] == [int(os.environ['ACTION_DIM'])], features['action']['shape']
assert info['fps'] == 25, 'Review horizon timing for non-25-Hz data'
cameras = {k for k, v in features.items() if v['dtype'] in ('image', 'video')}
assert cameras == set(os.environ['CAMERAS'].split(',')), cameras
print('Verified dimensions/cameras at revision', sys.argv[2])
PY

if [[ -e "$OUT" ]]; then
    echo "Refusing existing output $OUT; pick another run name or resume explicitly." >&2
    exit 2
fi

# Decode a real AV1 frame. Do not fall back to pyav: it is ~9x slower (data_s ~1.2 s/step, ~11 h per run).
VIDEO_BACKEND=torchcodec
.venv/bin/python - "$DS" "$REV" <<'PY' || { echo 'torchcodec cannot decode the dataset videos' >&2; exit 2; }
import json, sys
from huggingface_hub import hf_hub_download
from torchcodec.decoders import VideoDecoder
info = json.load(open(hf_hub_download(sys.argv[1], 'meta/info.json', repo_type='dataset', revision=sys.argv[2])))
key = next(k for k, v in info['features'].items() if v['dtype'] == 'video')
path = info['video_path'].format(video_key=key, chunk_index=0, file_index=0)
import time
for attempt in range(5):  # concurrent jobs can race on the shared HF cache
    try:
        video = hf_hub_download(sys.argv[1], path, repo_type='dataset', revision=sys.argv[2])
        break
    except (FileNotFoundError, OSError):
        if attempt == 4:
            raise
        time.sleep(20)
print('torchcodec decoded', tuple(VideoDecoder(video).get_frames_at(indices=[0, 1]).data.shape), 'from', key)
PY
echo "Using video backend: $VIDEO_BACKEND"

exec .venv/bin/lerobot-train \
    --dataset.repo_id="$DS" --dataset.revision="$REV" \
    --dataset.video_backend="$VIDEO_BACKEND" --dataset.eval_split=0.1 \
    --dataset.image_transforms.enable=true \
    --dataset.image_transforms.resize_fit="[512,512]" --dataset.image_transforms.resize_antialias=false \
    --policy.type=smolvla \
    --policy.vlm_model_name=HuggingFaceTB/SmolVLM2-500M-Video-Instruct \
    --policy.load_vlm_weights=true --policy.freeze_vision_encoder=true \
    --policy.train_expert_only=true --policy.train_state_proj=true \
    --policy.n_obs_steps=1 --policy.chunk_size=50 --policy.n_action_steps=50 \
    --policy.optimizer_lr=1e-4 --policy.scheduler_warmup_steps=1000 --policy.scheduler_decay_steps=30000 \
    --policy.device=cuda \
    --accelerator.mixed_precision="${MIXED_PRECISION:-bf16}" \
    --policy.push_to_hub=true --policy.private=true \
    --policy.repo_id="andnetdeboer/$NAME" \
    --seed=42 --batch_size="${BATCH_SIZE:-64}" --steps="${STEPS:-30000}" --save_freq=5000 \
    --log_freq=100 --num_workers="${NUM_WORKERS:-28}" --dataloader_multiprocessing_context=fork --eval_steps=2000 --env_eval_freq=0 \
    --output_dir="$OUT" --job_name="$NAME" \
    --wandb.enable=true --wandb.project=rethoughtai-smolvla --wandb.entity=deboerandnet-northwestern-university --wandb.mode=online
