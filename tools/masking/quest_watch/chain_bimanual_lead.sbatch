#!/bin/bash
# After SAM3 inpainting: relabel to a 640 ms lead, push to the Hub (private, v3.0), verify it loads, then
# submit the bimanual MultiTask DiT (augmentation only) on an H100 pinned to the pushed revision.
#SBATCH -J chain_bimanual_lead
#SBATCH -A e52938
#SBATCH -p short
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH -t 01:00:00
#SBATCH -o /projects/e52938/kmy2091/logs/%x_%j.out
set -euo pipefail
WS=/projects/e52938/kmy2091
NAME=${NAME:-SweepTrash_Kin-Full}
LEAD=${LEAD:-16}
# TAG names the inpainting method in the outputs, e.g. TAG=effecterase -> NAME_inpainted_effecterase_lead640ms
TAG=${TAG:+_$TAG}
SRC=${SRC:-$WS/datasets/h100/${NAME}_inpainted}
OUT=$WS/datasets/${NAME}_inpainted${TAG}_lead640ms
REPO=andnetdeboer/${NAME}_inpainted${TAG}_lead640ms
RUN=${RUN:-multi_task_dit_${NAME}_inpainted${TAG}_lead640ms_aug_30k_r1}
export HF_HOME=$WS/cache/huggingface HF_LEROBOT_HOME=$WS/cache/lerobot HF_TOKEN_PATH=$HOME/.cache/huggingface/token
export LD_LIBRARY_PATH="$WS/cache/ffmpeg7/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" PYTHONUNBUFFERED=1
PY=$WS/rethought-lerobot/.venv/bin/python

"$PY" "$WS/RethoughtAI-adapter/tools/relabel_lead.py" --source "$SRC" --out "$OUT" --lead-steps "$LEAD"

SHA=$("$PY" -I - "$OUT" "$REPO" <<'PY'
import sys
from huggingface_hub import HfApi
out, repo = sys.argv[1], sys.argv[2]
api = HfApi()
api.create_repo(repo, repo_type="dataset", private=True, exist_ok=True)
api.upload_large_folder(repo_id=repo, folder_path=out, repo_type="dataset", private=True, print_report=False)
api.create_tag(repo, tag="v3.0", repo_type="dataset", exist_ok=True)
print(api.dataset_info(repo).sha)
PY
)
SHA=$(echo "$SHA" | tail -1)
echo "Pushed $REPO at $SHA"

"$PY" -I - "$REPO" "$SHA" <<'PY'
import sys
from lerobot.datasets.lerobot_dataset import LeRobotDataset
ds = LeRobotDataset(sys.argv[1], revision=sys.argv[2], video_backend="torchcodec")
x = ds[ds.num_frames // 2]
print("loaded", ds.num_episodes, "episodes", ds.num_frames, "frames",
      {k: tuple(v.shape) for k, v in x.items() if hasattr(v, "shape") and v.ndim})
# First and last frame of every episode, every camera: torchcodec raises if a timestamp misses its frame.
import pandas as pd
eps = ds.meta.episodes
for i in range(len(eps)):
    for idx in (int(eps[i]["dataset_from_index"]), int(eps[i]["dataset_to_index"]) - 1):
        item = ds[idx]
        assert int(item["episode_index"]) == int(eps[i]["episode_index"]), (i, idx)
print("checked first/last frame of all", len(eps), "episodes on", len(ds.meta.video_keys), "cameras")
PY

cd "$WS"
STATE_DIM=28 ACTION_DIM=14 DIT_VARIANT=aug \
CAMERAS=observation.images.left,observation.images.right,observation.images.third_eye \
sbatch --export=ALL --gres=gpu:h100:1 --exclude=qgpu2014 --time=03:00:00 --output=logs/%x_%j.out \
    RethoughtAI/training/quest/multitask_dit_combo_30k.sbatch "$REPO" "$SHA" "$RUN"
