#!/bin/bash
# Person removal with a video-inpainting model (video_remove.py) on saved SAM3 masks, then a private Hub push
# (tag v3.0) and a check that every episode decodes. Submit from /projects/e52938/kmy2091, e.g.
#   METHOD=minimax SRC=datasets/X MASKS=cache/vor/masks/X OUT=datasets/X_inpainted_minimax \
#   sbatch --gres=gpu:a100:2 RethoughtAI-adapter/tools/masking/quest_video_remove.sbatch
#SBATCH -J video_remove
#SBATCH -A e52938
#SBATCH -p gengpu
#SBATCH --gres=gpu:a100:2
#SBATCH --exclude=qgpu2014
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH -t 12:00:00
#SBATCH -o /projects/e52938/kmy2091/logs/%x_%j.out
set -euo pipefail
WS=/projects/e52938/kmy2091
V=$WS/cache/vor
: "${METHOD:?minimax or effecterase}" "${SRC:?source dataset dir}" "${MASKS:?sam3_masks.py output dir}" "${OUT:?output dataset dir}"
REPO=${REPO:-andnetdeboer/$(basename "$OUT")}
CAMERA=${CAMERA:-observation.images.third_eye}
WORKERS=${WORKERS:-2}
export PATH="$WS/cache/ffmpeg7/bin:$PATH" LD_LIBRARY_PATH="$WS/cache/ffmpeg7/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export HF_HOME=$WS/cache/huggingface HF_LEROBOT_HOME=$WS/cache/lerobot HF_TOKEN_PATH=$HOME/.cache/huggingface/token PYTHONUNBUFFERED=1
cd "$WS"
nvidia-smi -L

"$V/venv-$METHOD/bin/python" RethoughtAI-adapter/tools/masking/video_remove.py --method "$METHOD" \
    --source "$SRC" --masks "$MASKS" --out "$OUT" --camera "$CAMERA" --workers "$WORKERS" ${STEPS:+--steps "$STEPS"} \
    2>&1 | grep -v -i -E "warn|it/s\]"

SHA=$("$WS/rethought-lerobot/.venv/bin/python" -I - "$OUT" "$REPO" <<'PY' | tail -1
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
echo "Pushed https://huggingface.co/datasets/$REPO at $SHA"

"$WS/rethought-lerobot/.venv/bin/python" -I - "$REPO" "$SHA" <<'PY'
import sys
from lerobot.datasets.lerobot_dataset import LeRobotDataset
ds = LeRobotDataset(sys.argv[1], revision=sys.argv[2], video_backend="torchcodec")
eps = ds.meta.episodes
for i in range(len(eps)):
    for idx in (int(eps[i]["dataset_from_index"]), int(eps[i]["dataset_to_index"]) - 1):
        assert int(ds[idx]["episode_index"]) == int(eps[i]["episode_index"]), (i, idx)
print("checked first/last frame of all", len(eps), "episodes on", len(ds.meta.video_keys), "cameras")
PY
