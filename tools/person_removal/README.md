# Person removal

Kinesthetic demonstrations show the operator's hands and body pushing the arm, so a policy can learn
"move when a person is in view". These tools erase the person from a LeRobot dataset's camera videos and
leave everything else as recorded: states, actions, timestamps, the other cameras and every pixel away from
the person. The result is a new dataset with the same episodes and different pixels.

| Step | Command | What it does |
| --- | --- | --- |
| 0 | `fetch` | Downloads a video method's code and weights (once per machine). |
| 1 | `masks` | Segments the person in every frame with SAM3 and saves lossless mask videos. |
| 2 | `remove` | Inpaints the masked region with Telea, MiniMax-Remover or EffectErase, into a new dataset. |
| 3 | `publish` | Pushes the dataset to the Hub and checks every episode decodes. |

Masks are made once and shared by every method, so comparing methods costs only step 2.

## Methods

| Method | `--method` | Model | GPU memory | Quality |
| --- | --- | --- | --- | --- |
| Telea | `telea` | OpenCV, no model | none (CPU) | Pale smear over large regions such as a torso; flickers. A baseline. |
| MiniMax-Remover | `minimax` | Wan2.1-1.3B, distilled (NeurIPS 2025), 12 steps | 7 GB | Clean and temporally consistent; can hallucinate an object in a large hole. |
| EffectErase | `effecterase` | Wan2.1-Fun-1.3B + LoRA (CVPR 2026) | 20 GB | Best: also removes the person's shadows. Non-commercial licence (CC BY-NC 4.0). |

On SweepTrash (third-eye camera, 56 episodes) EffectErase at 20 steps looked best, MiniMax-Remover once
painted a red blob into the hole, and Telea smeared. Run `remove --max-frames 200` with each method on one
dataset and look before committing GPU hours.

## Speed

End to end (decoding, the model, compositing, AV1 encoding), one worker per GPU, 640 x 480 frames, the
operator in view in nearly every clip. Measured on the third-eye cameras of the SweepTrash and
BlueCupRedContainer datasets (81,000 and 21,000 frames); EffectErase at 50 steps is scaled from its 20-step runs by
the two settings' measured model times.

| Step | Hardware | Seconds per frame | Hours per hour of 25 fps video |
| --- | --- | --- | --- |
| `masks` (SAM3) | A100 | 0.08 | 2 |
| `remove --method telea` | 1 CPU worker | 0.05 | 1.3 |
| `remove --method minimax` | A100 | 0.35 | 9 |
| `remove --method minimax` | H100 | 0.18 | 4.5 |
| `remove --method effecterase --steps 20` | A100 | 0.6 (0.53 to 0.71) | 15 |
| `remove --method effecterase` (50 steps) | A100 | about 1.2 | about 30 |

- A clip with nobody in it is copied without running the model, at about Telea's speed: a dataset where the
  operator is in view half the time takes about half as long.
- More GPUs divide the time: `--workers N` over N GPUs, or shards (below) across machines. Two workers on one
  GPU are no faster than one (SAM3 went from 12.9 to 2 x 4.7 frames per second).
- Every run prints seconds per frame as it goes and records them per video in its report, so a smoke run
  with `--max-frames 200` gives the estimate for your own data and GPU.

### Live use

Masks and removal add up per frame (SAM3 on an A100 for the masks):

| Combination | Seconds per frame | Frames per second | Short of 25 fps by |
| --- | --- | --- | --- |
| SAM3 + Telea (one CPU worker) | 0.13 | 8 | 3x |
| SAM3 + MiniMax-Remover, A100 | 0.43 | 2.3 | 11x |
| SAM3 + EffectErase, 20 steps, A100 | 0.68 | 1.5 | 17x |

None keeps up with a 25 fps camera on one GPU, and the video models add latency on top: they inpaint a clip
of 81 frames at once, so a frame's result waits for the 80 frames after it to be recorded (3.2 s) and then
for the clip's compute (about 28 s for MiniMax-Remover on an A100). These tools are for processing recorded
datasets, not a live stream. A live version would need shorter clips (the models accept 4k + 1 frames, but
quality below 81 is untested here), SAM3 run only every few frames with its masks tracked in between, and
several GPUs taking turns on consecutive clips.

## Setup

ffmpeg 6 or newer must be on the PATH, with an AV1 encoder (libsvtav1) and decoder (libdav1d), as LeRobot
datasets are AV1 by default: `ffmpeg -encoders | grep svtav1` and `ffmpeg -decoders | grep dav1d`. The
conda-forge build has both (`conda install -c conda-forge "ffmpeg>=7" dav1d`).

The three models need incompatible library versions (SAM3 needs transformers 5, the video methods older
diffusers), so each gets its own environment, from `requirements/`:

```bash
cd tools/person_removal
for env in sam3 minimax effecterase; do
    uv venv --python 3.12 .venv-$env && uv pip install --python .venv-$env -r requirements/$env.txt
done

# SAM3 is gated: accept its licence at huggingface.co/facebook/sam3, then
.venv-sam3/bin/hf auth login

# The video methods' code and weights, into any directory (about 3 GB for minimax, 20 GB for effecterase)
.venv-minimax/bin/python . fetch --method minimax --models ~/person_removal_models
.venv-effecterase/bin/python . fetch --method effecterase --models ~/person_removal_models
uv pip install --python .venv-effecterase -e ~/person_removal_models/EffectErase
```

Telea needs only the SAM3 environment.

## Process a dataset

All commands run from the repository root as `python tools/person_removal COMMAND`, each with its method's
environment. `--source` is a local dataset directory or a Hub repo id (with `--revision` to pin it). Only
the third-eye camera is processed unless `--camera` says otherwise (repeat it for several cameras).

**1. Masks** (once per dataset, any method):

```bash
.venv-sam3/bin/python tools/person_removal masks \
    --source andnetdeboer/SweepTrash_Kin-Full --out masks/SweepTrash_Kin-Full --workers 4
```

Use one worker per GPU (`--workers` = the number of GPUs); a second worker on the same GPU is slower. The
output mirrors the dataset (`videos/<camera>/chunk-000/file-000.mkv`, 255 where the person is) and
`masks_report.json` gives each video's mean person area. A video at 0% means SAM3 never saw anyone: check
the prompt (`--prompt`) before going on.

**2. Remove**, with one of:

```bash
# Telea
.venv-sam3/bin/python tools/person_removal remove --method telea \
    --source andnetdeboer/SweepTrash_Kin-Full --masks masks/SweepTrash_Kin-Full \
    --out datasets/SweepTrash_Kin-Full_inpainted_telea --workers 8

# MiniMax-Remover
.venv-minimax/bin/python tools/person_removal remove --method minimax --models ~/person_removal_models \
    --source andnetdeboer/SweepTrash_Kin-Full --masks masks/SweepTrash_Kin-Full \
    --out datasets/SweepTrash_Kin-Full_inpainted_minimax --workers 2

# EffectErase, 20 steps
.venv-effecterase/bin/python tools/person_removal remove --method effecterase --steps 20 \
    --models ~/person_removal_models --source andnetdeboer/SweepTrash_Kin-Full \
    --masks masks/SweepTrash_Kin-Full --out datasets/SweepTrash_Kin-Full_inpainted_effecterase --workers 2
```

`--workers` runs that many processes with a model each, over the visible GPUs (one video file each at a
time). The output directory must not exist yet. `meta/person_removal_<method>.json` in it records the
settings and each video's speed.

**3. Publish**:

```bash
.venv-sam3/bin/python tools/person_removal publish \
    --folder datasets/SweepTrash_Kin-Full_inpainted_minimax --repo-id USER/SweepTrash_Kin-Full_inpainted_minimax
```

The repo is private unless `--public`; it is tagged `v3.0`, as LeRobot expects. The check decodes the first
and last frame of every episode of the pushed revision and needs lerobot installed (the training
environment, or `--no-check`). Train on the commit SHA it prints.

### More GPUs than one machine has

A dataset's videos are few and long (one file holds many episodes), so `--workers` stops helping beyond the
number of files. Cut the videos into segments at episode starts instead, run shards of them anywhere, then
join them:

```bash
# M runs, K = 0 .. M-1, on as many machines as you have; each writes lossless parts
python tools/person_removal remove --method minimax --models ~/person_removal_models \
    --source DATASET --masks MASKS --out OUT --segment-frames 1500 --shard K/M --parts parts/NAME

# once every shard is done (no model, CPU only)
python tools/person_removal assemble --method minimax --source DATASET --masks MASKS --out OUT --parts parts/NAME
```

Segments never split an episode, so joining them needs no blending and the result is identical to an
unsharded run. `assemble` refuses to run while any span of any video is missing.

## How a video is processed

1. Frames and masks are read in clips of `--clip-len` frames (81 for the video models, which take 4k + 1
   frames), consecutive clips sharing `--overlap` frames (16).
2. Each frame's mask is the union of its own and its neighbours' (`--temporal` 2 frames either side), grown
   by `--dilation` pixels (6): SAM3 segments frames independently and its edges flicker.
3. A clip without any person is copied as it is. Otherwise the method inpaints it.
4. The output enters each frame only through the mask, feathered over `--feather` pixels (6), so the robot
   and the objects away from the person keep their recorded pixels. Overlapping clips are cross-faded.
5. The video is encoded with the source camera's settings from `meta/info.json` (AV1, GOP 2, CRF 30 by
   default), so training reads it as fast as the original.

Telea defaults to one frame per call, no temporal union, 3 px dilation and no feathering.

## Caveats

- SAM3 is prompted with "person". A hand entering from the frame edge is the hardest case; look at a few
  frames of the masks of every new dataset.
- Wrist cameras can see the operator's hand too; add them with `--camera`.
- The video models generate the hole's content: a large hole can come back with an object that was never
  there. Compare a few episodes side by side with the source before training.

## Code

| File | Contents |
| --- | --- |
| `__main__.py` | The command line. |
| `masks.py` | SAM3 masks, and the mask clean-up every method shares (temporal union, dilation, feathering). |
| `remove.py` | Clip streaming, compositing and cross-fading; workers, segments, shards and assembly. |
| `methods/` | One module per method: its `DEFAULTS` and `load(models)`, which returns `inpaint(frames, masks, steps)`. |
| `dataset.py` | A LeRobot dataset's videos, copying the rest, encoder settings, episode segments. |
| `video.py` | Reading and writing video through ffmpeg. |
| `fetch.py`, `publish.py` | Steps 0 and 3. |

A new method is one module in `methods/` with `DEFAULTS` and `load`, plus its name in `methods.METHODS` and
the `--method` choices.
