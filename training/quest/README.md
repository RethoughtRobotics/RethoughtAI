# Quest: 30,000-step kinesthetic VITA comparison

## Verified October 14, 2026

- SSH host `quser32` is a login node, without an allocated GPU. A failed `nvidia-smi` there is not a reason to install drivers.
- Slurm account: `e52938`, normal QOS. `gengpu` exposes A100 and H100 resources, maximum 48 hours; `gengpu-long` exposes A100 resources, maximum 10 days.
- A **test-only**, unsubmitted request for one H100, 8 CPUs, 64 GB system RAM and 12 hours was accepted by Slurm. GPU memory is separate from `--mem`.
- RethoughtAI `main` is still June 9 (`dae06cf`); training is on `add-training` (`d009bb0`). Adapter, recorder and inference have October 14 changes.
- The actual backend is https://github.com/andnet-deboer/lerobot, **main**, now `c90443e13a4af9c1b8817280ca803a23329026ee`. The `vita` branch is older (`220c6fb7`). The local backend checkout has been updated to main.

### New backend commits

| Commit | What changed |
| --- | --- |
| `c90443e1` | `tools/train_vita_hub.sh`: generic VITA training on raw Hub layouts, resume support; dataset completeness checker; shared GPU lock for workstation pipelines |
| `809c17f0` | `policy.camera_dropout`: training-only feature dropout per camera; restores all cameras if every camera would be dropped |
| `496a89de` | Adds `--val_freq` and held-out evaluation to the older TCP-relative cup task trainer |
| `6ab7b1fb` | Generic trainer defaults to 10% held-out episodes and evaluation every 2,000 steps |

Use generic `lerobot-train`, not `tasks/cup_to_container/train.py`: the latter transforms 15-D state / 8-D absolute actions into 17-D / 7-D TCP-relative features. Generic training derives the raw shapes from dataset metadata. This resolves the earlier backend-layout concern, but does not establish the private dataset's identity or label alignment.

The original RethoughtAI `image_augmentation` setting still only reaches its `lerobot` backend, not its VITA `task` backend. The Quest script bypasses that mismatch and passes image transforms explicitly.

## Remaining prerequisites

There is currently **no Hugging Face token** and **no installed training environment** in this session. Private kinesthetic datasets/checkpoints therefore cannot be enumerated or checked. None of the public RethoughtAI branches names the exact private cup export. Do not substitute public `CupToContainer` for the zero-lead kinesthetic dataset.

1. Load `python/3.12.10` with Quest's module system. System `python3` is only 3.6.8.
2. Lightweight tooling is installed at `/projects/e52938/kmy2091/.quest-tools/bin/` (including `uv` and `hf`). Use that `uv` executable if it is not on PATH.
3. Set `UV_CACHE_DIR=/projects/e52938/kmy2091/cache/uv`, then install the backend environment in `rethought-lerobot`: `uv sync --locked --extra training --extra diffusion --extra scipy-dep --python /software/2025/python/3.12.10/bin/python`. These are targeted extras, not `--extra all` (which pulls unrelated robot/VLA packages). Dependency installation and binary compatibility have **not yet been tested**.
4. Set `HF_TOKEN_PATH=$HOME/.cache/huggingface/token` and authenticate interactively with `hf auth login` as `andnetdeboer`. Never put a token in scripts, YAML, chat, or Git. The same token path is preserved when project-local `HF_HOME` is set.
5. List your authenticated datasets with `HfApi().list_datasets(author="andnetdeboer")`; identify the **zero-lead** cup export. Check its original export settings: 15-D state, 8-D absolute pose/gripper actions, 25 Hz, left wrist and third-eye cameras. Shape checks alone cannot detect future label shifts.
6. Run backend `tools/dataset_ready.py DATASET_ID` to verify upload completeness. Obtain its exact dataset commit SHA with `HfApi().dataset_info(DATASET_ID).sha`. Use the same SHA for both runs. Pre-download with `snapshot_download(..., repo_type="dataset", revision=SHA)` into the project Hub cache before allocating a GPU.

## Runs

From `/projects/e52938/kmy2091`, submit:

```bash
sbatch RethoughtAI/training/quest/vita_pose_30k.sbatch DATASET_ID DATASET_SHA raw vita_kin_pose_raw_30k
sbatch RethoughtAI/training/quest/vita_pose_30k.sbatch DATASET_ID DATASET_SHA aug vita_kin_pose_aug_30k
```

Replace both placeholders with verified values. These are instructions only: **no job has been submitted**. Override GPU selection with `sbatch --gres=gpu:a100:1 ...` if appropriate. Do not assume all A100s have the same VRAM. For a short GPU inspection allocation, use `srun --account=e52938 --partition=gengpu --gres=gpu:h100:1 --cpus-per-task=2 --mem=8G --time=00:10:00 --pty bash`, then `nvidia-smi` inside it. This does request billable resources.

The script uses:

- 30,000 optimizer steps, batch 128, fp32, seed 42, EMA power 0.75.
- 32-frame horizon, 16 action steps (1.28 / 0.64 seconds at 25 Hz).
- Same revision and deterministic episode split for raw and augmented runs. This split is **not proven identical to the earlier 70k/80k experiments**, so compare these new paired runs to one another.
- Built-in VITA resize/random crop in **both** variants. “Raw” means no *additional* dataset augmentation, not literally no image changes.
- Augmented variant: mild brightness/contrast/saturation plus random erasing with probability 0.35 and 2–8% area. No hue jitter, flips, or camera dropout. Both configured transforms are selected; the eraser's probability controls whether masking occurs.
- PyAV decoder to avoid making initial setup depend on TorchCodec/system FFmpeg compatibility. Use the same decoder for both variants.
- Generic trainer's policy optimizer preset (including its default clipping), not a claim of exact parity with the older task trainer's optimizer.
- No automatic Hub push, no W&B authentication dependency, no tmux/cron GPU picker, and no overwrite of existing output folders.

Inspect augmentation examples before full training: random masks may hide the cup/target and cannot restore occluded information. They do **not** segment/remove the demonstrator. Person-targeted masks and SAM evaluation remain unimplemented. Camera dropout is available upstream but should be a separate follow-up condition; it is feature-level rather than image masking and preserves at least one camera.

## Monitor, resume, upload

- `squeue -u "$USER"`; output defaults to Slurm's `slurm-JOBID.out` in the submission directory.
- `sacct -j JOBID --format=JobID,State,Elapsed,MaxRSS,AllocTRES` after completion/failure.
- Check checkpoints under `outputs/RUN_NAME/checkpoints/`. Benchmark actual seconds per step; the repo's RTX 6000 Ada measurement (4.8 steps/s, roughly 1.7 hours for 30k training steps before overhead) is **not** a Quest runtime measurement. The 12-hour request is an initial budget, not a guarantee.
- Resume explicitly inside a new allocation using `lerobot-train --resume=true --config_path=.../checkpoints/last/pretrained_model/train_config.json`. Do not change variant/data during resume. The generic backend's `tools/train_vita_hub.sh` also resumes automatically, but its default is **100k**, not 30k, and it otherwise pushes to the Hub/W&B.
- After reviewing the run, use backend `tools/upload_checkpoints.py RUN_DIR andnetdeboer/NEW_MODEL_REPO --only_steps 30000` for an intentional private upload. Use distinct repos for raw/augmented runs; do not overwrite the tested 70k/80k models.

Evaluate matched no-contact autonomous trials, hand-near/hand-away prediction sensitivity, object sensitivity, interventions, overlapping-plan disagreement, and tracking error. Validation loss alone cannot establish removal of hand dependence or solve physical-guidance mismatch. Leave label lead, controller, replanning and chunk merge strategy unchanged in this first visual comparison.