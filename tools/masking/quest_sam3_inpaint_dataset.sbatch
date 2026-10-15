#!/bin/bash
# Build a SAM3 person-inpainted copy of a LeRobot dataset, verify it, and push it to the Hub (private).
# Submit from /projects/e52938/kmy2091:
#   sbatch RethoughtAI-adapter/tools/masking/quest_sam3_inpaint_dataset.sbatch
#SBATCH -J sam3_inpaint_dataset
#SBATCH -A e52938
#SBATCH -p gengpu
#SBATCH --gres=gpu:a100:1
#SBATCH --exclude=qgpu2014
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH -t 12:00:00
#SBATCH -o /projects/e52938/kmy2091/logs/%x_%j.out

set -euo pipefail
: "${SLURM_JOB_ID:?Submit this script through Slurm}"

WS=${WS:-/projects/e52938/kmy2091}
ROOT_DIR=${ROOT_DIR:-$WS/RethoughtAI-adapter}
SRC_DATASET=${SRC_DATASET:-$WS/datasets/BlueCupRedContainer_kin_lead640ms}
OUT_DATASET=${OUT_DATASET:-$WS/datasets/BlueCupRedContainer_kin_lead640ms_sam3_inpaint_v1}
REPO_ID=${REPO_ID:-andnetdeboer/$(basename "$OUT_DATASET")}
PUSH=${PUSH:-1}
# Separate env with transformers>=5 (Sam3Model); the robot/training envs are left untouched.
export VENV=${VENV:-$WS/cache/sam3-venv}
# conda-forge FFmpeg 7.1: libdav1d to decode the AV1 sources, libsvtav1 to re-encode like LeRobot.
FFMPEG_PREFIX=${FFMPEG_PREFIX:-$WS/cache/ffmpeg7}
export PATH="$FFMPEG_PREFIX/bin:$PATH"
export LD_LIBRARY_PATH="$FFMPEG_PREFIX/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export HF_HOME="$WS/cache/huggingface"
export HF_TOKEN_PATH="${HF_TOKEN_PATH:-$HOME/.cache/huggingface/token}"
export PYTHONUNBUFFERED=1

# Optional overrides:
# export PROMPT=person
# export THRESHOLD=0.3
# export DILATION=3
# export INPAINT_RADIUS=6
# export CAMERA=observation.images.third_eye
# export MAX_VIDEOS=0
# export MAX_FRAMES=0

nvidia-smi -L
cd "$ROOT_DIR"
bash tools/masking/slurm_sam3_inpaint_dataset.sh "$SRC_DATASET" "$OUT_DATASET" cuda

# Same frame count per video as the source, and decodable at random indices by the training decoder.
"$VENV/bin/python" -I - "$SRC_DATASET" "$OUT_DATASET" <<'PY'
import json, subprocess, sys
from pathlib import Path
src, out = Path(sys.argv[1]), Path(sys.argv[2])
def frames(p):
    r = subprocess.check_output(["ffprobe", "-v", "error", "-count_packets", "-select_streams", "v:0",
                                 "-show_entries", "stream=nb_read_packets,codec_name", "-of", "json", str(p)], text=True)
    s = json.loads(r)["streams"][0]
    return int(s["nb_read_packets"]), s["codec_name"]
for v in sorted((src / "videos").rglob("*.mp4")):
    a, b = frames(v), frames(out / v.relative_to(src))
    print(v.relative_to(src), "source", a, "output", b)
    assert a == b, f"mismatch for {v}"
PY
"$WS/rethought-lerobot/.venv/bin/python" -I - "$OUT_DATASET" <<'PY'
import os, sys
from pathlib import Path
from torchcodec.decoders import VideoDecoder
for v in sorted(Path(sys.argv[1], "videos").rglob("*.mp4")):
    d = VideoDecoder(str(v))
    n = d.metadata.num_frames
    print(v.name, v.parent.parent.name, tuple(d.get_frames_at(indices=[0, n // 2, n - 1]).data.shape))
PY
[[ "$PUSH" == 1 ]] || { echo "Built $OUT_DATASET (PUSH=0, not uploaded)"; exit 0; }

# huggingface_hub in the training venv has upload_large_folder (the SAM3 venv's newer one does not).
"$WS/rethought-lerobot/.venv/bin/python" -I - "$OUT_DATASET" "$REPO_ID" <<'PY'
import sys
from huggingface_hub import HfApi
out, repo = sys.argv[1], sys.argv[2]
api = HfApi()
api.create_repo(repo, repo_type="dataset", private=True, exist_ok=True)
api.upload_large_folder(repo_id=repo, folder_path=out, repo_type="dataset", private=True)
# LeRobotDataset loads the revision tagged with its format version (see adapter `push`).
api.create_tag(repo, tag="v3.0", repo_type="dataset", exist_ok=True)
print(f"Pushed https://huggingface.co/datasets/{repo} (private, tag v3.0) at", api.dataset_info(repo).sha)
PY
