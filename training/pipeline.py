"""
Run a queue of training jobs unattended, with watchdogs.

For each job, in order:
  1. wait for the job named in `after` to finish (success or failure, unless `require_success`)
  2. optional `precheck` command, polled until it succeeds (e.g. gated-model access)
  3. wait for a GPU with `gpu_mem_mib` free, then launch `launch` via tools/tmux_train.sh
     (or `attach_pid` to a job that is already running)
  4. watch: process exit, `success_regex` in the log, stalled log (no writes for `stall_minutes`),
     NaN/inf loss; kill and fail on stall/NaN
  5. on failure: retry with the next entry of `retries` (argument overrides, e.g. a smaller batch)
     after an out-of-memory error, or once more if no checkpoint was written yet
  6. on success: run the `on_success` commands (e.g. upload checkpoints)

Alerts go to `<state_dir>/alerts.log` and, if `wandb_alert` is set, to W&B (which can email you).
State is kept in `<state_dir>/state.json`, so the supervisor can be restarted at any time
(e.g. from cron) and resumes: finished jobs are skipped and running jobs are re-attached.

Only one supervisor runs per state dir (a lock on `<state_dir>/supervisor.lock`), and none starts
while `<state_dir>/PAUSED` exists. Touch PAUSED before stopping the supervisor to change a job, so
the cron watchdog does not restart it with the old config in the meantime.

Commands run in `workdir` (the rethought-lerobot checkout whose .venv does the training).

Usage:
    python tools/pipeline.py path/to/pipeline.json
"""
import fcntl
import json
import os
import re
import shlex
import signal
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

TOOLS_DIR = Path(__file__).resolve().parent
POLL_S = int(os.environ.get("PIPELINE_POLL_S", "60"))
OOM_RE = re.compile(r"OutOfMemoryError|CUDA out of memory", re.I)
NAN_RE = re.compile(r"\bloss\s*[=:]\s*-?(nan|inf)\b", re.I)  # train.py "loss=" and lerobot-train "loss:"


class Pipeline:
    def __init__(self, cfg_path: Path):
        self.cfg = json.loads(cfg_path.read_text())
        self.state_dir = Path(self.cfg["state_dir"])
        self.state_dir.mkdir(parents=True, exist_ok=True)
        self.state_path = self.state_dir / "state.json"
        self.state = json.loads(self.state_path.read_text()) if self.state_path.exists() else {}
        self.workdir = Path(self.cfg.get("workdir", ".")).expanduser().resolve()

    # ── bookkeeping ───────────────────────────────────────────────────────────
    def log(self, msg: str) -> None:
        line = f"{datetime.now():%F %T} {msg}"
        print(line, flush=True)
        with open(self.state_dir / "pipeline.log", "a") as f:
            f.write(line + "\n")

    def alert(self, title: str, text: str, level: str = "WARN") -> None:
        self.log(f"ALERT [{level}] {title}: {text}")
        with open(self.state_dir / "alerts.log", "a") as f:
            f.write(f"{datetime.now():%F %T} [{level}] {title}: {text}\n")
        wb = self.cfg.get("wandb_alert")
        if wb:
            code = (
                "import wandb,sys; r=wandb.init(project=sys.argv[1], entity=sys.argv[2], job_type='watchdog', "
                "id='pipeline-alerts', name='pipeline-alerts', resume='allow', settings=wandb.Settings(silent=True)); "
                "wandb.alert(title=sys.argv[3], text=sys.argv[4], level=getattr(wandb.AlertLevel, sys.argv[5])); r.finish()"
            )
            try:
                subprocess.run([sys.executable, "-c", code, wb["project"], wb["entity"], title, text, level],
                               timeout=120, capture_output=True)
            except Exception as e:  # alerting must never take the supervisor down
                self.log(f"W&B alert failed: {e}")

    def save(self) -> None:
        tmp = self.state_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.state, indent=2))
        tmp.replace(self.state_path)

    def job_state(self, name: str) -> dict:
        return self.state.setdefault(name, {"status": "pending", "attempt": 0})

    def set_status(self, name: str, status: str, **extra) -> None:
        st = self.job_state(name)
        st.update(status=status, updated=f"{datetime.now():%F %T}", **extra)
        self.save()
        self.log(f"[{name}] -> {status} {extra if extra else ''}")

    # ── helpers ───────────────────────────────────────────────────────────────
    @staticmethod
    def alive(pid: int) -> bool:
        try:
            os.kill(pid, 0)
            return True
        except OSError:
            return False

    @staticmethod
    def find_pid(match: str) -> int | None:
        """Oldest `python`/`lerobot-train` process whose command line contains `match`."""
        out = subprocess.run(["ps", "-eo", "pid,etimes,comm,args", "--no-headers"],
                             capture_output=True, text=True).stdout
        cands = []
        for line in out.splitlines():
            parts = line.split(None, 3)
            if len(parts) == 4 and parts[2] in ("python", "python3", "lerobot-train") and match in parts[3]:
                cands.append((-int(parts[1]), int(parts[0])))
        return max(cands)[1] if cands else None

    @staticmethod
    def best_gpu() -> tuple[int, int]:
        out = subprocess.run(["nvidia-smi", "--query-gpu=index,memory.free", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True).stdout
        gpus = [tuple(int(x) for x in line.split(",")) for line in out.strip().splitlines()]
        return max(gpus, key=lambda g: g[1])

    @staticmethod
    def log_text(path: Path, tail_bytes: int = 200_000) -> str:
        if not path.exists():
            return ""
        with open(path, "rb") as f:
            f.seek(max(0, path.stat().st_size - tail_bytes))
            return f.read().decode(errors="replace").replace("\r", "\n")

    def has_checkpoint(self, job: dict) -> bool:
        ck = Path(job["run_dir"]) / "checkpoints"
        return ck.exists() and any(ck.iterdir())

    def kill_job(self, job: dict, pid: int | None) -> None:
        if pid and self.alive(pid):
            os.kill(pid, signal.SIGINT)
            for _ in range(60):
                if not self.alive(pid):
                    break
                time.sleep(2)
            if self.alive(pid):
                os.kill(pid, signal.SIGKILL)
        if job.get("session"):
            subprocess.run(["tmux", "kill-session", "-t", job["session"]], capture_output=True)

    def run_cmds(self, name: str, cmds: list[str]) -> None:
        for cmd in cmds:
            self.log(f"[{name}] post-step: {cmd}")
            r = subprocess.run(cmd, shell=True, cwd=self.workdir, capture_output=True, text=True)
            with open(self.state_dir / f"{name}_post.log", "a") as f:
                f.write(f"$ {cmd}\n{r.stdout}\n{r.stderr}\n")
            if r.returncode != 0:
                self.alert(f"{name}: post-step failed", f"`{cmd}` exited {r.returncode}; see {name}_post.log")

    # ── phases ────────────────────────────────────────────────────────────────
    def wait_after(self, job: dict) -> bool:
        dep = job.get("after")
        if not dep:
            return True
        while self.job_state(dep)["status"] not in ("succeeded", "failed"):
            time.sleep(POLL_S)
        if job.get("require_success") and self.job_state(dep)["status"] != "succeeded":
            self.alert(f"{job['name']}: skipped", f"dependency {dep} failed and require_success is set")
            self.set_status(job["name"], "failed", reason="dependency failed")
            return False
        return True

    def precheck(self, job: dict) -> bool:
        cmd = job.get("precheck")
        if not cmd:
            return True
        deadline = time.time() + 3600 * job.get("precheck_timeout_hours", 12)
        alerted = False
        while time.time() < deadline:
            r = subprocess.run(cmd, shell=True, cwd=self.workdir, capture_output=True, text=True)
            if r.returncode == 0:
                if alerted:
                    self.log(f"[{job['name']}] precheck now passes")
                return True
            if not alerted:
                self.set_status(job["name"], "waiting_precheck")
                self.alert(f"{job['name']}: waiting", job.get("precheck_hint", f"precheck failing: {cmd}"))
                alerted = True
            time.sleep(60 * job.get("precheck_poll_minutes", 10))
        self.alert(f"{job['name']}: gave up", "precheck still failing after timeout", "ERROR")
        self.set_status(job["name"], "failed", reason="precheck timeout")
        return False

    def wait_gpu(self, job: dict) -> int | None:
        need = job.get("gpu_mem_mib", 0)
        deadline = time.time() + 3600 * job.get("max_gpu_wait_hours", 6)
        warned = False
        while time.time() < deadline:
            gpu, free = self.best_gpu()
            if free >= need:
                return gpu
            if not warned:
                self.set_status(job["name"], "waiting_gpu")
                self.alert(f"{job['name']}: waiting for GPU", f"needs {need} MiB free; best is GPU {gpu} with {free} MiB")
                warned = True
            time.sleep(POLL_S)
        self.alert(f"{job['name']}: no GPU", f"no GPU with {need} MiB free within the wait limit", "ERROR")
        return None

    @staticmethod
    def check_job(job: dict) -> None:
        """lerobot-train refuses an existing --output_dir, and tmux_train.sh creates the log's folder."""
        out = next((a.split("=", 1)[1] for a in job.get("launch", []) if a.startswith("--output_dir=")), None)
        if out and Path(job["log"]).resolve().is_relative_to(Path(out).resolve()):
            raise SystemExit(f"[{job['name']}] log {job['log']} is inside lerobot-train --output_dir {out}; "
                             "put the log outside it, or lerobot-train will refuse to start")

    def build_cmd(self, job: dict, attempt: int) -> list[str]:
        args = list(job["launch"])
        retries = job.get("retries", [])
        if attempt > 0 and retries:
            override = retries[min(attempt, len(retries)) - 1]
            for flag, value in override.items():
                if flag in args:
                    args[args.index(flag) + 1] = value
                elif any(a.startswith(flag + "=") for a in args):
                    i = next(i for i, a in enumerate(args) if a.startswith(flag + "="))
                    args[i] = f"{flag}={value}"
                else:
                    args += [flag, value]
        return args

    def launch(self, job: dict, gpu: int, attempt: int) -> int | None:
        args = self.build_cmd(job, attempt)
        subprocess.run(["tmux", "kill-session", "-t", job["session"]], capture_output=True)
        log = Path(job["log"])
        if log.exists():
            log.rename(log.with_name(f"{log.stem}.attempt{attempt - 1}{log.suffix}"))
        env_prefix = ["env"] + [f"{k}={v}" for k, v in job.get("env", {}).items()]
        cmd = [str(TOOLS_DIR / "tmux_train.sh"), job["session"], str(gpu), str(log), "--", *env_prefix, *args]
        self.log(f"[{job['name']}] launch attempt {attempt} on GPU {gpu}: {shlex.join(args)}")
        subprocess.run(cmd, cwd=self.workdir, check=True, capture_output=True,
                       env={**os.environ, "TRAIN_WORKDIR": str(self.workdir)})
        for _ in range(60):  # wait up to 2 min for the process to appear
            time.sleep(2)
            pid = self.find_pid(job["match"])
            if pid:
                return pid
        return None

    def watch(self, job: dict, pid: int) -> str:
        """Return 'succeeded', 'oom', 'stalled', 'nan' or 'crashed'."""
        log = Path(job["log"])
        stall_s = 60 * job.get("stall_minutes", 20)
        start = time.time()
        while self.alive(pid):
            time.sleep(POLL_S)
            text = self.log_text(log, 20_000)
            if NAN_RE.search(text):
                self.kill_job(job, pid)
                return "nan"
            age = time.time() - (log.stat().st_mtime if log.exists() else start)
            if age > stall_s and time.time() - start > stall_s:
                self.kill_job(job, pid)
                return "stalled"
        time.sleep(5)
        text = self.log_text(log)
        if re.search(job.get("success_regex", "Training complete"), text):
            return "succeeded"
        return "oom" if OOM_RE.search(text) else "crashed"

    def run_job(self, job: dict) -> None:
        name = job["name"]
        st = self.job_state(name)
        if st["status"] in ("succeeded", "failed"):
            self.log(f"[{name}] already {st['status']}, skipping")
            return
        if not self.wait_after(job) or not self.precheck(job):
            return

        max_attempts = 1 + max(len(job.get("retries", [])), 1)
        while True:
            pid = job.get("attach_pid") if st["attempt"] == 0 else None
            if pid and not self.alive(pid):
                pid = None
            pid = pid or self.find_pid(job["match"])  # re-attach after a supervisor restart
            if pid:
                self.log(f"[{name}] attached to running PID {pid}")
            elif job.get("launch"):
                gpu = self.wait_gpu(job)
                if gpu is None:
                    self.set_status(name, "failed", reason="no GPU")
                    return
                pid = self.launch(job, gpu, st["attempt"])
                if pid is None:
                    outcome = "crashed"
            if pid:
                self.set_status(name, "running", pid=pid, attempt=st["attempt"])
                outcome = self.watch(job, pid)
            elif not job.get("launch"):
                outcome = "succeeded" if re.search(job.get("success_regex", "Training complete"),
                                                   self.log_text(Path(job["log"]))) else "crashed"

            if outcome == "succeeded":
                self.set_status(name, "succeeded")
                self.alert(f"{name}: finished", "training completed", "INFO")
                self.run_cmds(name, job.get("on_success", []))
                return

            tail = "\n".join(self.log_text(Path(job["log"]), 4000).strip().splitlines()[-6:])
            st["attempt"] += 1
            retry = job.get("launch") and st["attempt"] < max_attempts and (
                outcome == "oom" or not self.has_checkpoint(job))
            self.alert(f"{name}: {outcome}", f"attempt {st['attempt'] - 1} ended ({outcome})."
                       f"{' Retrying.' if retry else ' Not retrying.'}\n{tail}", "ERROR")
            if not retry:
                self.set_status(name, "failed", reason=outcome)
                return
            self.save()

    def run(self) -> None:
        if (self.state_dir / "PAUSED").exists():
            raise SystemExit(f"{self.state_dir}/PAUSED exists; remove it to run the pipeline")
        self._lock = open(self.state_dir / "supervisor.lock", "w")  # held until this process exits
        try:
            fcntl.flock(self._lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise SystemExit(f"another supervisor is already running for {self.state_dir}")
        for job in self.cfg["jobs"]:
            self.check_job(job)
        self.log(f"pipeline start (PID {os.getpid()})")
        (self.state_dir / "supervisor.pid").write_text(str(os.getpid()))
        for job in self.cfg["jobs"]:
            self.run_job(job)
        self.log("pipeline finished: " + ", ".join(f"{j['name']}={self.job_state(j['name'])['status']}"
                                                   for j in self.cfg["jobs"]))
        self.alert("pipeline finished", ", ".join(f"{j['name']}: {self.job_state(j['name'])['status']}"
                                                  for j in self.cfg["jobs"]), "INFO")


if __name__ == "__main__":
    Pipeline(Path(sys.argv[1])).run()
