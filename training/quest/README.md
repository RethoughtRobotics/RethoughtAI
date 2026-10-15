# Quest training

Slurm scripts for training LeRobot policies on Northwestern Quest with `lerobot-train` from
[andnet-deboer/lerobot](https://github.com/andnet-deboer/lerobot) (branch `quest-training-speedups`).
Submit from `/projects/e52938/kmy2091`. Every run pins a dataset commit and pushes a private checkpoint
to `andnetdeboer/<RUN_NAME>`.

| Script | Policy | Usage |
| --- | --- | --- |
| `multitask_dit_combo_30k.sbatch` | MultiTask DiT, bf16 | `DS SHA RUN_NAME`, `DIT_VARIANT=combo\|aug` |
| `vita_pose_30k.sbatch` | VITA, fp32 + EMA | `DS SHA raw\|aug\|drop\|mask RUN_NAME` |
| `smolvla_30k.sbatch` | SmolVLA (SmolVLM2-500M), bf16 | `DS SHA RUN_NAME` |

```bash
sbatch --gres=gpu:h100:1 RethoughtAI/training/quest/multitask_dit_combo_30k.sbatch \
    andnetdeboer/BlueCupRedContainer_kin_pose e2594aed18862a36bac83d27bd6217faefa709c5 my_run
```

**Augmentation variants.** `aug`: LeRobot's six default image transforms. `combo` (DiT) / `drop`, `mask`
(VITA) add third-eye camera dropout (30%) and/or random spatial masking.

**Bimanual / other robots.** Shapes are checked before training. Override the single-arm defaults:
`STATE_DIM=28 ACTION_DIM=14 CAMERAS=observation.images.left,observation.images.right,observation.images.third_eye`
(and pass `--export=ALL` to `sbatch`).

## Speed

What each script does to keep the GPU busy, and typical step times for 30k steps:

- **AV1 decoding.** The cluster `ffmpeg` modules cannot software-decode AV1, so torchcodec uses conda-forge
  FFmpeg 7.1 + dav1d from `cache/ffmpeg7` (setup command in each script). The script decodes a real frame
  first and stops instead of falling back to PyAV, which is ~9x slower.
- **Resize before augmenting.** Workers shrink frames to the policy's input size before augmentation
  (`resize_shape` for DiT/VITA, aspect-preserving `resize_fit` for SmolVLA), ~4x less CPU per sample.
- **Workers.** 28 `fork` workers on 32 CPUs: startup in seconds instead of ~65 s per spawned worker.

| Policy | GPU | Single-arm | Bimanual (3 cameras) |
| --- | --- | --- | --- |
| DiT (bf16) | H100 | 0.14 s/step, 1.2 h | 0.20 s/step, 1.7 h |
| DiT (bf16) | A100 | 0.28 s/step, 2.3 h | — |
| VITA | H100 / A100 | — / 0.21 s/step, 1.8 h | 0.12 / 0.23 s/step |

## Notes

- Exclude node `qgpu2014` (`CUDA device busy or unavailable`); the scripts' examples do.
- SmolVLA needs `num2words` in the training venv for the SmolVLM processor.
- `lead640ms` datasets label each action with the TCP pose 640 ms (16 steps) later. The zero-lead DiT moved
  slowly and hesitantly on the robot. Relabel any measured-TCP dataset with `tools/relabel_lead.py`
  (adapter branch).
