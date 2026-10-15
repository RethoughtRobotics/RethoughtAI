"""Copy a LeRobot v3 dataset with measured-TCP actions relabelled to a new lead.

For measured labels the adapter sets the TCP-goal part of action[t] to the measured TCP pose at
min(t + lead, last frame of the episode); other action dims (gripper) are not led and are kept as they are. This rebuilds the action column for another lead from observation.state, recomputes the action stats
(meta/stats.json and per-episode stats) with LeRobot's own functions, and records the new lead in
meta/rethoughtai_sources.json. Videos and everything else are copied unchanged.

    python tools/relabel_lead.py --source DATASET --out NEW_DATASET --lead-steps 16
"""

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

from lerobot.datasets.compute_stats import get_feature_stats


def action_to_state_index(state_names: list[str], action_names: list[str]) -> list[int]:
    """left.goal.x -> left.tcp.x, goal.x -> tcp.x, gripper -> gripper."""
    idx = []
    for name in action_names:
        target = name.replace("goal.", "tcp.")
        if target not in state_names:
            raise ValueError(f"action {name!r} has no state counterpart {target!r}")
        idx.append(state_names.index(target))
    return idx


def lead_actions(action: np.ndarray, state: np.ndarray, episode: np.ndarray, pose: dict[int, int], lead: int) -> np.ndarray:
    """Copy of `action` whose TCP-goal dims (action dim -> state dim in `pose`) come from the state `lead` steps later."""
    out = action.astype(np.float32).copy()
    a_dims, s_dims = list(pose), list(pose.values())
    for ep in np.unique(episode):
        rows = np.where(episode == ep)[0]
        src = np.minimum(np.arange(len(rows)) + lead, len(rows) - 1)
        out[np.ix_(rows, a_dims)] = state[rows[src]][:, s_dims]
    return out


def to_jsonable(stats: dict) -> dict:
    return {k: v.tolist() for k, v in stats.items()}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--lead-steps", type=int, required=True)
    args = p.parse_args()
    if args.out.exists():
        raise SystemExit(f"Output exists: {args.out}")

    info = json.loads((args.source / "meta/info.json").read_text())
    state_names = info["features"]["observation.state"]["names"]
    action_names = info["features"]["action"]["names"]
    idx = action_to_state_index(state_names, action_names)
    pose = {a: s for a, s in enumerate(idx) if ".goal." in f".{action_names[a]}"}  # TCP goal dims only
    if not pose:
        raise SystemExit(f"No TCP goal dims in action names {action_names}")
    sources_path = args.source / "meta/rethoughtai_sources.json"
    sources = json.loads(sources_path.read_text())
    if sources.get("labels") != "measured":
        raise SystemExit(f"Only measured labels can be relabelled from state, got {sources.get('labels')!r}")

    shutil.copytree(args.source, args.out, ignore=shutil.ignore_patterns(".cache"))
    for f in args.out.rglob("*"):  # copies of read-only HF cache files
        f.chmod(f.stat().st_mode | 0o200)

    data_files = sorted((args.out / "data").rglob("*.parquet"))
    frames = pd.concat([pd.read_parquet(f) for f in data_files], ignore_index=True)
    state = np.stack(frames["observation.state"]).astype(np.float64)
    old = np.stack(frames["action"])
    old_lead = lead_actions(old, state, frames["episode_index"].values, pose, sources["lead_steps"])
    err = float(np.abs(old_lead - old).max())
    if err > 1e-5:
        raise SystemExit(f"Existing actions do not match lead_steps={sources['lead_steps']} from state (max err {err})")
    new = lead_actions(old, state, frames["episode_index"].values, pose, args.lead_steps)

    start = 0
    for f in data_files:  # write back file by file, preserving row order
        df = pd.read_parquet(f)
        df["action"] = list(new[start:start + len(df)])
        df.to_parquet(f, index=False)
        start += len(df)

    for ep_file in sorted((args.out / "meta/episodes").rglob("*.parquet")):
        eps = pd.read_parquet(ep_file)
        for i, row in eps.iterrows():
            rows = np.where(frames["episode_index"].values == row["episode_index"])[0]
            st = get_feature_stats(new[rows], axis=0, keepdims=False)
            for k, v in st.items():
                col = f"stats/action/{k}"
                if col in eps.columns:
                    eps.at[i, col] = np.asarray(v).reshape(-1).tolist()  # stored as flat lists, like the source
        eps.to_parquet(ep_file, index=False)

    stats_path = args.out / "meta/stats.json"
    stats = json.loads(stats_path.read_text())
    # Exact dataset-level stats over all frames. (LeRobot's aggregate_stats merges per-episode quantiles
    # approximately -- up to 0.16 off on these datasets; min/max/mean/std agree. MIN_MAX normalization is exact.)
    x = new.astype(np.float64)
    stats["action"] = {"min": x.min(0).tolist(), "max": x.max(0).tolist(), "mean": x.mean(0).tolist(),
                       "std": x.std(0).tolist(), "count": [len(x)]}
    for q in (1, 10, 50, 90, 99):
        stats["action"][f"q{q:02d}"] = np.quantile(x, q / 100, axis=0).tolist()
    stats_path.write_text(json.dumps(stats, indent=4))

    sources["relabelled_from"] = {"lead_steps": sources["lead_steps"], "tool": "tools/relabel_lead.py"}
    sources["lead_steps"] = args.lead_steps
    (args.out / "meta/rethoughtai_sources.json").write_text(json.dumps(sources, indent=2))
    print(f"Relabelled {len(new)} frames to lead {args.lead_steps} ({args.lead_steps / info['fps'] * 1000:.0f} ms): {args.out}")


if __name__ == "__main__":
    main()
