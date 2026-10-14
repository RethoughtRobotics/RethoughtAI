# Offline masking evaluation

Evaluate pretrained models before fine-tuning or changing training videos. Nothing here publishes
ROS messages, changes policy inputs or modifies datasets. Run from the RethoughtAI directory with
its existing Python environment. CPU is the default to avoid contention with live robot inference.

## Status

- DeepLabV3/ResNet50, official torchvision COCO-with-VOC-labels weights: ungated **person baseline**,
  not claimed to be the best available model or a cropped-hand specialist.
- Mask R-CNN ResNet50 FPN v2, official torchvision COCO weights: ungated person-instance baseline.
  Often better boundaries than DeepLab, but still not a dedicated hand segmenter.
- SAM 3 backend: prepared, but not runtime-validated. The current authenticated Hugging Face request
  for `facebook/sam3` / `sam3.pt` returned **403 GatedRepoError** on 2026-10-14. Request authorized
  access at https://huggingface.co/facebook/sam3; never work around the gate.
- SAM 3 installation/weights and fine-tuning have not been performed by this pipeline.
  Use a separate environment following https://github.com/facebookresearch/sam3 rather than changing
  the robot runtime's dependencies. Model API/version compatibility must be checked after installation.

## Extract a small reproducible sample

```bash
.venv/bin/python tools/masking/evaluate.py extract \
  --dataset datasets/BlueCupRedContainer_kin_pose \
  --episodes 0 4 8 --max-episodes 3 --samples 3 \
  --out ../reports/2026-10-14-person-shortcut/masking_samples
```

Existing directories are deliberately not overwritten; choose a fresh output name for new runs.
Extraction reads LeRobot v3 episode/video offsets and samples both policy cameras. It stores
episode/frame IDs, requested video time and decoded RGB hashes. FFmpeg/libdav1d seek is used for
AV1 videos; this is a visual-review sample, **not guaranteed exact training-decoder parity**.
This manifest is not a complete frame-aligned mask cache suitable for dataset export.

## Run an ungated baseline

```bash
.venv/bin/python tools/masking/evaluate.py run \
  --samples-dir ../reports/2026-10-14-person-shortcut/masking_samples \
  --out ../reports/2026-10-14-person-shortcut/masking_deeplab_v1 \
  --backend deeplab --device cpu --threads 4

.venv/bin/python tools/masking/evaluate.py run \
  --samples-dir ../reports/2026-10-14-person-shortcut/masking_samples \
  --out ../reports/2026-10-14-person-shortcut/masking_maskrcnn_v1 \
  --backend maskrcnn --device cpu --threads 4
```

The first run downloads official pretrained weights. Outputs include original-resolution binary
masks, per-prompt masks, original/overlay/neutral-fill previews, an HTML gallery and JSON provenance.
Input hashes are checked. DeepLab's threshold is a person-pixel probability, not instance confidence.
Recorded elapsed time includes preprocessing/transfer and the first-call warmup; it is not a
standardized latency benchmark. Mask area and confidence **are not accuracy**.

Create a compact overview, or open the run's HTML gallery:

```bash
.venv/bin/python tools/masking/evaluate.py contact-sheet \
  --run ../reports/2026-10-14-person-shortcut/masking_deeplab_v1 \
  --out ../reports/2026-10-14-person-shortcut/masking_deeplab_v1_overview.jpg
```

## SAM 3, after authorized setup

In the separate environment containing a verified SAM 3 installation:

```bash
python /absolute/path/to/RethoughtAI/tools/masking/evaluate.py run \
  --samples-dir /absolute/path/to/masking_samples \
  --out /absolute/path/to/masking_sam3_v1 \
  --backend sam3 --device cuda --checkpoint /absolute/path/to/sam3.pt \
  --prompts person "human hand" "human forearm"
```

Run away from live robot control, or on an allocated Quest GPU—not a Quest login-node GPU job.
Compare prompt masks individually before union: a false-positive hand prompt can erase the robot.
SAM 3 thresholds are instance-confidence thresholds, not comparable to DeepLab pixel thresholds.
Point/box-prompt SAM 2 video tracking is a useful subsequent comparator but is not implemented here.

## Review before training

Inspect both wrist and third-eye views across reach, grasp, transfer, release and no-person frames.
Mark misses of cropped hands/forearms, false removal of robot/gripper/cup/container, shadows and
ambiguous occlusions. Review the sample visually before scaling to full videos. A person silhouette
can still expose hand location; erasure is not proof the policy has become operator-invariant.

For a quantitative benchmark, create independent manually labelled binary human masks, named
exactly like the predicted union masks (white human, black everything else), in a separate folder:

```bash
.venv/bin/python tools/masking/evaluate.py score \
  --run /absolute/path/to/masking_deeplab_v1 \
  --truth /absolute/path/to/manual_truth \
  --out /absolute/path/to/new_metrics.json
```

Scores report labelled coverage, IoU, precision and recall overall and by camera. No human labels
are fabricated and no accuracy claim is made without them. Separately annotate protected task
objects to measure accidental removal; a high person IoU alone does not establish training utility.
Train/validation splits must be episode-separated; masked derivatives inherit their source split.

Fine-tuning should wait until reviewed failures identify the need: detector misses, segmentation
boundaries, hand-specific coverage or temporal tracking. Keep masks/versioned provenance separate
from original images, and validate temporal alignment before any future masked dataset export.

## Build a masked dataset variant

If you want a materialized dataset for training, build a separate copy where only camera videos are
masked and metadata/parquet labels stay unchanged:

```bash
.venv/bin/python tools/masking/mask_dataset.py \
  --source datasets/BlueCupRedContainer_kin_lead640ms \
  --out datasets/BlueCupRedContainer_kin_lead640ms_maskrcnn_v1 \
  --backend maskrcnn --device cuda --threshold 0.5 --fill 127 --dilation 2

# quick smoke test (first 64 frames of first camera video)
.venv/bin/python tools/masking/mask_dataset.py \
  --source datasets/BlueCupRedContainer_kin_lead640ms \
  --out /tmp/mask_smoke \
  --backend deeplab --device cpu --max-videos 1 --max-frames 64
```

This keeps frame count and ordering fixed, writes `meta/masking_report.json`, and never edits the
source dataset. Use a new output path each run.

Quest launcher (run on an allocated compute node, not a login node):

```bash
bash tools/masking/slurm_mask_dataset.sh \
  /path/to/source_dataset \
  /path/to/output_masked_dataset \
  maskrcnn cuda

# or submit as a batch job after editing allocation and paths
sbatch tools/masking/quest_mask_dataset.sbatch
```

### Training recommendation

Do **not** train only on fully blacked-out/person-erased videos. Use a mixture:

1. Raw dataset only (baseline).
2. Masked-dataset-only (stress test).
3. Mixed raw+masked episodes (recommended default).

The mixed setup is usually the best first step for inference invariance: the policy still sees normal
images but cannot depend entirely on demonstrator pixels. Keep evaluation scenes unmasked to verify
real deployment behavior.

### Invariance without relying on masks at inference

To reduce dependence on person regions while remaining robust to unmasked deployment views:

1. Train on mixed raw + masked episodes (for example 50/50 batch sampling).
2. Add random non-person Cutout in both raw and masked branches.
3. Optional: consistency loss between raw and masked views for the same frame/state.

Use masked inputs as a training-time perturbation, not a hard runtime requirement. This avoids
introducing a brittle dependency on live segmentation quality during robot inference.