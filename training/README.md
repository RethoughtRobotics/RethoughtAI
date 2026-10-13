# Training

Train a policy on a dataset made by [`adapter`](../adapter/), then deploy it with [`inference`](../inference/).

Training runs in a checkout of **rethought-lerobot** (our LeRobot fork) and its `.venv`. This folder holds
what to run and how: shared tools at the top level, one folder per policy with its config, a launcher
and notes.

```
training/
├── README.md               this file: the overall process
├── train.py                compose a run from YAML and launch it (print / run / tmux / queue)
├── pipeline.py             unattended queue supervisor: GPU wait, retries, OOM/stall/NaN watchdog, upload
├── tmux_train.sh           run any command in a detached tmux session on one GPU, teed to a log
├── upload_checkpoints.py   push checkpoints to a HF model repo (step-* branches + main)
├── defaults.yaml           shared settings: paths, HF/W&B accounts, validation, seed, ...
├── datasets/               one YAML per dataset (repo id, episodes)
└── <policy>/               act, smolvla, multi_task_dit, vita, groot, pi05
    ├── config.yaml         backend, steps, batch size, GPU memory, policy arguments
    ├── train.sh            launcher: training/<policy>/train.sh [train.py options]
    └── README.md           what the policy is, why these settings, measured cost, gotchas
```

## 1. Setup

```bash
git clone <rethought-lerobot> ~/rethought-lerobot && cd ~/rethought-lerobot && uv sync --locked --extra all
cd ~/RethoughtAI && uv sync                       # train.py itself only needs pyyaml
huggingface-cli login && wandb login              # checkpoints go to HF, metrics to W&B
```

Then check `defaults.yaml`: `lerobot_dir`, `output_root`, `hub_owner` and the W&B entity are per user.
The fork must include held-out validation (`dataset.eval_split` / `eval_steps`) and PEFT
`rank_pattern` support, which the pi0.5 config needs.

## 2. Pick a policy

Measured on one RTX 6000 Ada (48 GB) with the Baxter cup datasets:

| Policy | Backend | Batch | Steps | Speed | Peak memory | Wall clock |
|---|---|---|---|---|---|---|
| [ACT](act/) | task | 8 | 100k | 17 steps/s | small | ~2 h |
| [SmolVLA](smolvla/) | task | 64 | 30k | 2.8 steps/s | – | ~3 h |
| [VITA](vita/) | task | 128 | 100k | 4.8 steps/s | – | ~6 h |
| [Multi-Task DiT](multi_task_dit/) | task | 96 | 30k | 1.4 steps/s | ~39 GB | ~6 h |
| [GR00T N1.7](groot/) | lerobot | 64 | 20k | 1.38 s/step | 36 GB | ~8 h |
| [pi0.5 (LoRA)](pi05/) | lerobot | 32 | 60k | 8.4 s/step | 30.5 GB | ~6 days |

Rule of thumb: start with ACT on a single-task setup, since it trains in hours and is often the best
return. Move to a VLA (SmolVLA, GR00T, pi0.5) for multi-task or language-conditioned work.

## 3. Configure

A run is built from three YAML layers. Later layers win: dicts merge and everything else replaces.

1. `defaults.yaml`: shared for every run
2. `datasets/<name>.yaml`: the dataset. The policy config picks one, or pass `--dataset <name>`.
3. `<policy>/config.yaml`: the policy

After those come `--set key=value` overrides. Dotted keys reach into dicts, and values are parsed as YAML.
Everything under `args:` is passed straight to the backend as a flag. For the lerobot backend that
means any `lerobot-train` option (`--set args.policy.chunk_size=50`). For the task backend it means
any `tasks/cup_to_container/train.py` option (`--set args.cameras=[observation.images.left]`).

There are two backends:

| Backend | Command | Used for | Notes |
|---|---|---|---|
| `lerobot` | `lerobot-train` | GR00T, pi0.5, any LeRobot policy | can select episodes; pushes to HF itself |
| `task` | `tasks/cup_to_container/train.py` | ACT, SmolVLA, Multi-Task DiT, VITA | trains on all episodes; pushes to HF itself |

Names follow from the config: run `<name>_<dataset short_name>`, output
`<output_root>/<run>/`, log `<output_root>/logs/<run>.log`, HF repo
`<hub_owner>/<hub_name or name>_<short_name><hub_suffix>`, W&B project `rethoughtai-<wandb_name or name>`.

- **New dataset:** add `datasets/<name>.yaml` with `repo_id`, `short_name`, `episodes` (`null` = all).
- **New policy:** add `<policy>/config.yaml` (copy the closest one), `train.sh` (the same one-liner) and a `README.md`.

## 4. Launch

Always look at the command first. With no `--mode`, `train.py` only prints it.

```bash
training/pi05/train.sh                                  # print the command
training/act/train.sh --mode run --gpu 0                # foreground
training/smolvla/train.sh --mode tmux --gpu 1           # detached tmux session <run-name>, attach with tmux attach -t
uv run python training/train.py groot pi05 --mode queue --cron   # GR00T, then pi0.5, unattended
```

**Queue mode is the default for anything long.** It writes `pipeline.json` to
`<output_root>/pipeline_<date>_<policies>/` and starts `pipeline.py`, which for each job in order:

1. waits for the previous job
2. runs the `precheck` until it passes (e.g. gated-model access)
3. waits for a GPU with `gpu_mem_mib` free
4. launches the job in tmux
5. watches the log: success, crash, out of memory, stall (`stall_minutes` without output), NaN loss
6. retries with the next `retry_batch_sizes` entry after OOM (or once more if nothing was saved yet)
7. on success, uploads the `upload_steps` checkpoints

`--cron` installs a watchdog that restarts the supervisor if it dies. Alerts go to
`alerts.log` and, as W&B alerts, to the `watchdog_wandb_project`, which can email you.

**To change a queued job, pause the pipeline first.** Otherwise cron restarts the old config while you edit:

```bash
S=<output_root>/pipeline_...;  touch $S/PAUSED;  kill $(cat $S/supervisor.pid);  tmux kill-session -t <run>
# edit $S/pipeline.json, move the old output dir away, delete $S/state.json, then:
rm $S/PAUSED     # cron restarts it within 10 min, or start it now:
setsid nohup ~/rethought-lerobot/.venv/bin/python training/pipeline.py $S/pipeline.json >> $S/supervisor.out 2>&1 &
```

Only one supervisor can run per pipeline directory; a second one exits immediately.

## 5. Monitor

- **W&B**: `train/*` every `log_freq` steps and `eval/*` every `val.every` steps. Validation is on for
  every run: the lerobot backend holds out the last 10% of episodes per task. Watch `eval/eval_loss`
  for overfitting, since train loss alone hides it.
- **Logs**: `tail -f <output_root>/logs/<run>.log`, or `tmux attach -t <run-name>`.
- **Pipeline**: `pipeline.log`, `alerts.log` and `state.json` in the pipeline directory.

## 6. Checkpoints

Each run saves `<output_root>/<run>/checkpoints/<step>/pretrained_model/` every `save_freq` steps and
pushes the final one to the HF repo (private by default). `upload_checkpoints.py` adds intermediate
steps as `step-XXXXXX` branches and adds the repo to `hub_collection`:

```bash
~/rethought-lerobot/.venv/bin/python training/upload_checkpoints.py <output_root>/<run> <hub_repo> --only_steps 10000,20000
```

LoRA runs (pi0.5) save `adapter_model.safetensors` plus the fully trained modules, not the full
model. Loading one also loads its base model (`lerobot/pi05_base`).

## 7. Deploy

Load the HF repo (or one of its `step-*` revisions) with [`inference`](../inference/).
