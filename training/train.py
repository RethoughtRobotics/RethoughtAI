"""
Compose a training run from YAML configs and launch it.

A run is built from three layers, later ones winning (dicts merge, everything else replaces):

    defaults.yaml                 where things live, W&B/HF accounts, validation, seed, ...
    datasets/<dataset>.yaml       which LeRobot dataset and episodes
    <policy>/config.yaml          backend, steps, batch size, GPU memory, policy arguments

plus `--set key=value` overrides (dotted keys reach into dicts, values are parsed as YAML).

Two backends turn the merged config into a command, run inside `lerobot_dir`:
    lerobot   `lerobot-train` (GR00T, pi0.5, and any other LeRobot policy)
    task      rethought-lerobot's `tasks/cup_to_container/train.py` (ACT, SmolVLA, Multi-Task DiT, VITA)

Modes:
    print     show the command and exit (default)
    run       run it in the foreground
    tmux      run it in a detached tmux session on one GPU
    queue     run one or more policies one after another under the watchdog supervisor (pipeline.py)

Examples:
    python training/train.py pi05                                  # print the pi0.5 command
    python training/train.py act --dataset cup_to_container --mode tmux --gpu 1
    python training/train.py smolvla --set steps=5000 --set args.cameras=observation.images.left
    python training/train.py groot pi05 --mode queue --cron        # GR00T, then pi0.5, unattended
"""
import argparse
import copy
import json
import os
import shlex
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent
BACKENDS = ("lerobot", "task")


# ── config ────────────────────────────────────────────────────────────────────
def merge(base: dict, over: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in over.items():
        out[k] = merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else copy.deepcopy(v)
    return out


def set_dotted(cfg: dict, assignment: str) -> None:
    key, sep, raw = assignment.partition("=")
    if not sep:
        raise SystemExit(f"--set expects key=value, got {assignment!r}")
    # `args.*` keys are CLI flags that may themselves contain dots (policy.type), so split only once there.
    parts = ["args", key[5:]] if key.startswith("args.") else key.split(".")
    node = cfg
    for p in parts[:-1]:
        node = node.setdefault(p, {})
    node[parts[-1]] = yaml.safe_load(raw)


def policy_config_path(policy: str) -> Path:
    p = Path(policy)
    if p.suffix in (".yaml", ".yml") and p.exists():
        return p.resolve()
    p = HERE / policy / "config.yaml"
    if not p.exists():
        known = sorted(d.name for d in HERE.iterdir() if (d / "config.yaml").exists())
        raise SystemExit(f"unknown policy {policy!r}; known: {', '.join(known)}")
    return p


def load_config(policy: str, dataset: str | None, sets: list[str]) -> dict:
    cfg = yaml.safe_load((HERE / "defaults.yaml").read_text())
    pol = yaml.safe_load(policy_config_path(policy).read_text())
    ds_name = dataset or pol.get("dataset") or cfg["dataset"]
    ds_path = HERE / "datasets" / f"{ds_name}.yaml"
    if not ds_path.exists():
        raise SystemExit(f"unknown dataset {ds_name!r}: no {ds_path}")
    cfg = merge(cfg, {"dataset": yaml.safe_load(ds_path.read_text())})
    cfg = merge(cfg, {k: v for k, v in pol.items() if k != "dataset"})
    for s in sets:
        set_dotted(cfg, s)
    if cfg.get("backend") not in BACKENDS:
        raise SystemExit(f"{policy}: backend must be one of {BACKENDS}, got {cfg.get('backend')!r}")

    ds = cfg["dataset"]
    cfg.setdefault("run_name", f"{cfg['name']}_{ds['short_name']}")
    root = Path(os.path.expanduser(cfg["output_root"]))
    cfg["output_dir"] = str(root / cfg["run_name"])
    cfg["log_file"] = str(root / "logs" / f"{cfg['run_name']}.log")  # outside output_dir: lerobot-train refuses a non-empty one
    cfg["hub_repo_id"] = f"{cfg['hub_owner']}/{cfg.get('hub_name', cfg['name'])}_{ds['short_name']}{cfg.get('hub_suffix', '')}"
    cfg["wandb_project"] = cfg["wandb"]["project_prefix"] + cfg.get("wandb_name", cfg["name"])
    cfg["lerobot_dir"] = str(Path(os.path.expanduser(cfg["lerobot_dir"])).resolve())
    return cfg


# ── backends ──────────────────────────────────────────────────────────────────
def flag_value(v) -> str:
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, (list, dict)):
        return json.dumps(v, separators=(",", ":"))
    return str(v)


def lerobot_cmd(cfg: dict) -> list[str]:
    ds, val, wb = cfg["dataset"], cfg["val"], cfg["wandb"]
    common = {
        "dataset.repo_id": ds["repo_id"],
        "dataset.episodes": ds.get("episodes"),
        "dataset.video_backend": cfg["video_backend"],
        "dataset.image_transforms.enable": cfg["image_augmentation"],
        "dataset.eval_split": val["split"],
        "eval_steps": val["every"] if val["split"] else 0,
        "policy.device": "cuda",
        "policy.push_to_hub": True,
        "policy.private": cfg["private"],
        "policy.repo_id": cfg["hub_repo_id"],
        "seed": cfg["seed"],
        "batch_size": cfg["batch_size"],
        "steps": cfg["steps"],
        "save_freq": cfg["save_freq"],
        "log_freq": cfg["log_freq"],
        "num_workers": cfg["num_workers"],
        "use_policy_training_preset": True,
        "env_eval_freq": 0,
        "output_dir": cfg["output_dir"],
        "job_name": cfg["run_name"],
        "wandb.enable": True,
        "wandb.project": cfg["wandb_project"],
        "wandb.entity": wb["entity"],
        "wandb.disable_artifact": True,
    }
    args = {**common, **cfg.get("args", {})}
    return ["lerobot-train"] + [f"--{k}={flag_value(v)}" for k, v in args.items() if v is not None]


def task_cmd(cfg: dict) -> list[str]:
    ds = cfg["dataset"]
    if ds.get("episodes") is not None:
        raise SystemExit(f"{cfg['name']}: the task backend trains on every episode; this dataset config "
                         "selects episodes, which only the lerobot backend supports")
    common = {
        "dataset_repo_id": ds["repo_id"],
        "video_backend": cfg["video_backend"],
        "output_dir": cfg["output_dir"],
        "steps": cfg["steps"],
        "batch_size": cfg["batch_size"],
        "save_freq": cfg["save_freq"],
        "log_freq": cfg["log_freq"],
        "val_freq": cfg["val"]["every"] if cfg["val"]["split"] else 0,
        "num_workers": cfg["num_workers"],
        "hub_repo_id": cfg["hub_repo_id"],
        "wandb_project": cfg["wandb_project"],
        "wandb_entity": cfg["wandb"]["entity"],
    }
    cmd = ["python", "tasks/cup_to_container/train.py"]
    for k, v in {**common, **cfg.get("args", {})}.items():
        if v is None or v is False:
            continue
        cmd.append(f"--{k}")
        if v is True:
            continue
        cmd += [str(x) for x in v] if isinstance(v, list) else [str(v)]
    return cmd


def build_cmd(cfg: dict) -> list[str]:
    return lerobot_cmd(cfg) if cfg["backend"] == "lerobot" else task_cmd(cfg)


# ── modes ─────────────────────────────────────────────────────────────────────
def run_foreground(cfg: dict, cmd: list[str], gpu: int | None) -> int:
    env = {**os.environ, **{k: str(v) for k, v in cfg.get("env", {}).items()}}
    env["PATH"] = f"{cfg['lerobot_dir']}/.venv/bin:{env['PATH']}"
    if gpu is not None:
        env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    Path(cfg["log_file"]).parent.mkdir(parents=True, exist_ok=True)
    with open(cfg["log_file"], "w") as log:
        proc = subprocess.Popen(cmd, cwd=cfg["lerobot_dir"], env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        for chunk in iter(lambda: proc.stdout.read1(4096), b""):
            sys.stdout.buffer.write(chunk)
            sys.stdout.flush()
            log.write(chunk.decode(errors="replace"))
            log.flush()
    return proc.wait()


def run_tmux(cfg: dict, cmd: list[str], gpu: int) -> int:
    env_prefix = ["env"] + [f"{k}={v}" for k, v in cfg.get("env", {}).items()]
    session = cfg["run_name"].replace("_", "-")
    return subprocess.run([str(HERE / "tmux_train.sh"), session, str(gpu), cfg["log_file"], "--", *env_prefix, *cmd],
                          env={**os.environ, "TRAIN_WORKDIR": cfg["lerobot_dir"]}).returncode


def pipeline_job(cfg: dict, cmd: list[str], after: str | None) -> dict:
    q = cfg["queue"]
    job = {
        "name": cfg["run_name"],
        "session": cfg["run_name"].replace("_", "-"),
        "gpu_mem_mib": cfg["gpu_mem_mib"],
        "match": cfg["output_dir"],
        "log": cfg["log_file"],
        "run_dir": cfg["output_dir"],
        "success_regex": "End of training" if cfg["backend"] == "lerobot" else "Training complete",
        "env": {k: str(v) for k, v in cfg.get("env", {}).items()},
        "launch": cmd,
        "retries": [{"--batch_size": str(b)} for b in cfg.get("retry_batch_sizes", [])],
        "stall_minutes": q["stall_minutes"],
    }
    if after:
        job["after"] = after
    if cfg.get("precheck"):
        job.update(precheck=f"PATH={cfg['lerobot_dir']}/.venv/bin:$PATH {cfg['precheck']['cmd']}", precheck_hint=cfg["precheck"].get("hint", ""),
                   precheck_poll_minutes=q["precheck_poll_minutes"], precheck_timeout_hours=q["precheck_timeout_hours"])
    if cfg.get("upload_steps"):
        steps = ",".join(str(s) for s in cfg["upload_steps"])
        collection = f" --collection {cfg['hub_collection']}" if cfg.get("hub_collection") else ""
        job["on_success"] = [f"{cfg['lerobot_dir']}/.venv/bin/python {HERE}/upload_checkpoints.py {cfg['output_dir']} "
                             f"{cfg['hub_repo_id']}{collection} --only_steps {steps}"]
    return job


def run_queue(cfgs: list[dict], cmds: list[list[str]], install_cron: bool, dry: bool) -> int:
    first = cfgs[0]
    names = "_".join(c["name"] for c in cfgs)
    state_dir = Path(os.path.expanduser(first["output_root"])) / f"pipeline_{datetime.now():%Y-%m-%d_%H%M}_{names}"
    jobs, after = [], None
    for cfg, cmd in zip(cfgs, cmds):
        jobs.append(pipeline_job(cfg, cmd, after))
        after = cfg["run_name"]
    pipeline = {"state_dir": str(state_dir), "workdir": first["lerobot_dir"], "jobs": jobs}
    if first.get("watchdog_wandb_project"):
        pipeline["wandb_alert"] = {"entity": first["wandb"]["entity"], "project": first["watchdog_wandb_project"]}
    if dry:
        print(json.dumps(pipeline, indent=2))
        return 0

    state_dir.mkdir(parents=True)
    (state_dir / "pipeline.json").write_text(json.dumps(pipeline, indent=2))
    python = f"{first['lerobot_dir']}/.venv/bin/python"
    start = f"setsid nohup {python} {HERE}/pipeline.py {state_dir}/pipeline.json >> {state_dir}/supervisor.out 2>&1 < /dev/null &"
    watchdog = state_dir / "cron_watchdog.sh"
    watchdog.write_text(f"""#!/bin/bash
# Restart the pipeline supervisor if it died before finishing (run from cron every 10 min).
# Does nothing while {state_dir}/PAUSED exists.
export PATH=/usr/local/bin:/usr/bin:/bin HOME={Path.home()}
S={state_dir}
[ -e $S/PAUSED ] && exit 0
grep -q "pipeline finished" $S/pipeline.log 2>/dev/null && exit 0
pid=$(cat $S/supervisor.pid 2>/dev/null)
[ -n "$pid" ] && kill -0 "$pid" 2>/dev/null && exit 0
echo "$(date '+%F %T') cron watchdog: supervisor not running, restarting" >> $S/pipeline.log
echo "$(date '+%F %T') [WARN] watchdog: supervisor was not running; restarted by cron" >> $S/alerts.log
cd {first['lerobot_dir']} && {start}
""")
    watchdog.chmod(0o755)
    subprocess.run(start, shell=True, cwd=first["lerobot_dir"], check=True)
    if install_cron:
        current = subprocess.run(["crontab", "-l"], capture_output=True, text=True).stdout
        subprocess.run(["crontab", "-"], input=current + f"*/10 * * * * {watchdog}\n", text=True, check=True)
    print(f"Supervisor started for {', '.join(c['run_name'] for c in cfgs)}")
    print(f"  State:   {state_dir}  (pipeline.log, alerts.log, state.json)")
    print(f"  Pause:   touch {state_dir}/PAUSED, then kill $(cat {state_dir}/supervisor.pid)")
    if not install_cron:
        print(f"  Cron:    not installed; add with: (crontab -l; echo '*/10 * * * * {watchdog}') | crontab -")
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("policies", nargs="+", help="policy folder name(s) (act, pi05, ...) or path(s) to a config.yaml")
    p.add_argument("--dataset", help="dataset config name in datasets/ (default: from the policy, else defaults.yaml)")
    p.add_argument("--set", action="append", default=[], metavar="KEY=VALUE",
                   help="override any config key, e.g. steps=5000, val.every=1000, args.policy.chunk_size=50")
    p.add_argument("--name", help="run name (default: <policy>_<dataset short name>)")
    p.add_argument("--mode", choices=["print", "run", "tmux", "queue"], default="print")
    p.add_argument("--gpu", type=int, default=None, help="GPU index for run/tmux (queue picks a free one)")
    p.add_argument("--cron", action="store_true", help="queue: install the cron watchdog")
    p.add_argument("--dry-run", action="store_true", help="queue: print pipeline.json instead of starting it")
    a = p.parse_args()

    if len(a.policies) > 1 and a.mode != "queue":
        raise SystemExit("several policies only make sense with --mode queue")
    if a.name and len(a.policies) > 1:
        raise SystemExit("--name applies to a single policy")
    sets = a.set + ([f"run_name={a.name}"] if a.name else [])
    cfgs = [load_config(pol, a.dataset, sets) for pol in a.policies]
    cmds = [build_cmd(c) for c in cfgs]

    if a.mode == "print":
        for cfg, cmd in zip(cfgs, cmds):
            env = " ".join(f"{k}={v}" for k, v in cfg.get("env", {}).items())
            print(f"# {cfg['run_name']}: run in {cfg['lerobot_dir']} (log: {cfg['log_file']})")
            print(f"{env} {shlex.join(cmd)}".strip())
        return 0
    if a.mode == "queue":
        return run_queue(cfgs, cmds, a.cron, a.dry_run)
    if Path(cfgs[0]["output_dir"]).exists():
        raise SystemExit(f"{cfgs[0]['output_dir']} already exists; pick another --name or move it away")
    if a.mode == "run":
        return run_foreground(cfgs[0], cmds[0], a.gpu)
    return run_tmux(cfgs[0], cmds[0], 0 if a.gpu is None else a.gpu)


if __name__ == "__main__":
    sys.exit(main())
