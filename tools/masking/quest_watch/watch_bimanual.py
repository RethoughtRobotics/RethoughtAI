"""Watch andnetdeboer's HF datasets; when a new, fully uploaded bimanual dataset appears, start the SAM3 inpaint job.

One line per event on stdout (for a Claude Code Monitor). State lives in seen.json next to this file, so the
watcher can be restarted without re-triggering. Run with the rethought-lerobot venv python.
"""

import json
import os
import subprocess
import sys
import time
from pathlib import Path

from huggingface_hub import HfApi, hf_hub_download, snapshot_download

WS = Path("/projects/e52938/kmy2091")
STATE = Path(__file__).with_name("seen.json")
SBATCH = WS / "RethoughtAI-adapter/tools/masking/quest_sam3_inpaint_dataset.sbatch"
AUTHOR = "andnetdeboer"
SKIP_SUFFIXES = ("_inpainted", "_inpaint_v1")
POLL_S = int(os.environ.get("POLL_S", "120"))

api = HfApi()


def say(msg: str) -> None:
    print(f"{time.strftime('%H:%M:%S')} {msg}", flush=True)


def load_state() -> dict:
    if STATE.exists():
        return json.loads(STATE.read_text())
    ids = sorted(d.id for d in api.list_datasets(author=AUTHOR))
    state = {"baseline": ids, "handled": {}}
    STATE.write_text(json.dumps(state, indent=2))
    say(f"baseline recorded: {len(ids)} existing datasets ignored")
    return state


def bimanual(repo: str, sha: str) -> tuple[bool, str]:
    info = json.load(open(hf_hub_download(repo, "meta/info.json", repo_type="dataset", revision=sha)))
    cams = sorted(k for k, v in info["features"].items() if v.get("dtype") == "video")
    arms = None
    try:
        src = json.load(open(hf_hub_download(repo, "meta/rethoughtai_sources.json", repo_type="dataset", revision=sha)))
        arms = src.get("arms")
    except Exception:
        pass
    is_bi = (arms is not None and len(arms) >= 2) or (
        "observation.images.left" in cams and "observation.images.right" in cams
    )
    return is_bi, f"arms={arms} cameras={cams}"


def launch(repo: str, sha: str) -> str:
    name = repo.split("/")[1]
    src = WS / "datasets" / name
    out = WS / "datasets" / f"{name}_inpainted"
    if out.exists():
        raise RuntimeError(f"output exists: {out}")
    snapshot_download(repo, repo_type="dataset", revision=sha, local_dir=src)
    env = dict(os.environ, SRC_DATASET=str(src), OUT_DATASET=str(out), REPO_ID=f"{AUTHOR}/{name}_inpainted", WORKERS="4")
    # SAM3 is GPU-bound: 2 H100s with 2 workers each is ~4x one A100 (single A100: ~6.5-8.5 fps).
    r = subprocess.run(["sbatch", "--parsable", "--gres=gpu:h100:2", "--cpus-per-task=16", "--mem=128G", str(SBATCH)],
                       env=env, capture_output=True, text=True, check=True)
    return r.stdout.strip()


def main() -> None:
    state = load_state()
    waiting: set[str] = set()
    say(f"watching {AUTHOR} every {POLL_S}s; {len(state['handled'])} already handled")
    while True:
        try:
            for d in api.list_datasets(author=AUTHOR):
                repo = d.id
                if repo in state["baseline"] or repo in state["handled"] or repo.endswith(SKIP_SUFFIXES):
                    continue
                refs = api.list_repo_refs(repo, repo_type="dataset")
                if "v3.0" not in [t.name for t in refs.tags]:
                    if repo not in waiting:
                        waiting.add(repo)
                        say(f"WAITING {repo}: no v3.0 tag yet (upload may be in progress)")
                    continue
                sha = api.dataset_info(repo).sha
                is_bi, desc = bimanual(repo, sha)
                if not is_bi:
                    state["handled"][repo] = {"sha": sha, "action": "skipped: not bimanual", "desc": desc}
                    say(f"NEW_NOT_BIMANUAL {repo}@{sha[:8]} {desc}")
                else:
                    job = launch(repo, sha)
                    state["handled"][repo] = {"sha": sha, "action": "sam3 job", "job": job, "desc": desc}
                    say(f"NEW_BIMANUAL {repo}@{sha[:8]} {desc} -> SAM3 job {job} -> {repo}_inpainted")
                STATE.write_text(json.dumps(state, indent=2))
        except Exception as e:  # keep watching through transient Hub/Slurm errors
            say(f"ERROR {type(e).__name__}: {str(e)[:200]}")
        time.sleep(POLL_S)


if __name__ == "__main__":
    sys.exit(main())
