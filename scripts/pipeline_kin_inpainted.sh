#!/bin/bash
# Offline inpaint + measured export + HF push for KinInpaintedBlueCupRedContainer.
# Runs without internet (HF_HUB_OFFLINE=1 for models; push retries until online).
set -e
cd ~/FinalProject/rethought_ai
source /opt/ros/kilted/setup.bash
export HF_HUB_OFFLINE=1

SESSION=../data/sessions/KinInpaintedBlueCupRedContainer
DATASET=datasets/KinInpaintedBlueCupRedContainer_measured
REPO_ID=andnetdeboer/KinInpaintedBlueCupRedContainer_measured
TASK="Pick up the blue cup and place it in the red container"
LOG=datasets/KinInpaintedBlueCupRedContainer_measured.log

INPAINT_VENV=../ros2_inpaint/.venv/bin/python

echo "=== offline inpaint ($(date +%T))" | tee "$LOG"
$INPAINT_VENV -u scripts/offline_inpaint_session.py "$SESSION" 2>&1 | tee -a "$LOG"

echo "=== export measured ($(date +%T))" | tee -a "$LOG"
# Use a config that reads the inpainted topic for third_eye
.venv/bin/adapt lerobot "$SESSION" \
  --config configs/baxter_inpainted.yaml \
  --repo-id "$REPO_ID" \
  --labels measured --lead 1 \
  --trim-idle 0.01 1 \
  --task "$TASK" \
  --output "$DATASET" \
  --workers 6 2>&1 | tee -a "$LOG"

echo "=== push ($(date +%T))" | tee -a "$LOG"
unset HF_HUB_OFFLINE
until .venv/bin/adapt push "$DATASET" --repo-id "$REPO_ID" --public \
    --collection andnetdeboer/rethoughtai-6ac6adec399ec52e90e21556 >> "$LOG" 2>&1; do
  echo "$(date +%T) push failed (offline?); retrying in 2 min" | tee -a "$LOG"
  sleep 120
done
echo "=== done ($(date +%T))" | tee -a "$LOG"
echo "https://huggingface.co/datasets/$REPO_ID" | tee -a "$LOG"
