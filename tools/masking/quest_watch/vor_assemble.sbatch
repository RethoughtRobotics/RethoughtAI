#!/bin/bash
# Join EffectErase parts into SweepTrash_Kin-Full_inpainted_effecterase, push (private, v3.0), decode every episode.
#SBATCH -J vor_assemble
#SBATCH -A e52938
#SBATCH -p short
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH -t 02:00:00
#SBATCH -o /projects/e52938/kmy2091/logs/%x_%j.out
set -euo pipefail
WS=/projects/e52938/kmy2091; V=$WS/cache/vor
OUT=$WS/datasets/${NAME:?}_inpainted_${METHOD:?}; REPO=andnetdeboer/${NAME}_inpainted_${METHOD}
export PATH="$WS/cache/ffmpeg7/bin:$PATH" LD_LIBRARY_PATH="$WS/cache/ffmpeg7/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" PYTHONUNBUFFERED=1
export HF_HOME=$WS/cache/huggingface HF_LEROBOT_HOME=$WS/cache/lerobot HF_TOKEN_PATH=$HOME/.cache/huggingface/token
$V/venv-effecterase/bin/python $WS/RethoughtAI-adapter/tools/masking/video_remove.py --method "$METHOD" \
  --source $WS/datasets/$NAME --masks $V/masks/$NAME --out $OUT \
  --camera observation.images.third_eye --parts-dir $V/parts/${NAME}_$METHOD --assemble
SHA=$($WS/rethought-lerobot/.venv/bin/python -I - "$OUT" "$REPO" <<'PY' | tail -1
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
$WS/rethought-lerobot/.venv/bin/python -I - "$REPO" "$SHA" <<'PY'
import sys
from lerobot.datasets.lerobot_dataset import LeRobotDataset
ds = LeRobotDataset(sys.argv[1], revision=sys.argv[2], video_backend="torchcodec")
eps = ds.meta.episodes
for i in range(len(eps)):
    for idx in (int(eps[i]["dataset_from_index"]), int(eps[i]["dataset_to_index"]) - 1):
        assert int(ds[idx]["episode_index"]) == int(eps[i]["episode_index"]), (i, idx)
print("checked first/last frame of all", len(eps), "episodes on", len(ds.meta.video_keys), "cameras")
PY
