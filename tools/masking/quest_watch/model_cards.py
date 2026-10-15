"""Add a "Training data" section to each andnetdeboer model card, from the run's own train_config.json.

Facts only: dataset + revision from train_config.json; inpainting method from the dataset's report files;
action lead from the dataset's meta/rethoughtai_sources.json; policy/steps/augmentation from the config.
The section sits between markers, so re-running replaces it. Repos without a train_config.json are skipped.

  python model_cards.py            dry run (prints the first card)
  python model_cards.py --apply    push the updated cards
"""
import json
import re
import sys

from huggingface_hub import HfApi, ModelCard, hf_hub_download
from huggingface_hub.utils import EntryNotFoundError

api = HfApi()
BEGIN, END = "<!-- training-data:begin -->", "<!-- training-data:end -->"
METHODS = {
    "meta/video_remove_effecterase_report.json": "SAM3 person masks + EffectErase video inpainting (CVPR 2026, 20 steps)",
    "meta/video_remove_minimax_report.json": "SAM3 person masks + MiniMax-Remover video inpainting (NeurIPS 2025, 12 steps)",
    "meta/sam3_inpaint_report.json": "SAM3 person masks + OpenCV Telea inpainting",
}


def train_config(repo: str, files: list[str]) -> dict | None:
    for name in ("train_config.json", "pretrained_model/train_config.json"):
        if name in files:
            return json.load(open(hf_hub_download(repo, name)))
    return None


def dataset_facts(ds: str, rev: str | None) -> dict:
    out = {"method": "none (raw camera video)", "lead": "unknown", "sha": None}
    try:
        info = api.dataset_info(ds, revision=rev)
    except Exception as e:  # deleted or inaccessible dataset
        out["error"] = type(e).__name__
        return out
    out["sha"] = info.sha
    files = {s.rfilename for s in info.siblings}
    for f, m in METHODS.items():  # most specific first
        if f in files:
            out["method"] = m
            break
    if "meta/rethoughtai_sources.json" in files:
        src = json.load(open(hf_hub_download(ds, "meta/rethoughtai_sources.json", repo_type="dataset", revision=info.sha)))
        lead = src.get("lead_steps")
        fps = src.get("fps") or 25
        if lead is not None:
            out["lead"] = f"{lead} steps ({lead * 1000 // fps} ms)" + (" — relabelled" if "relabelled_from" in src else "")
    return out


def augmentation(cfg: dict) -> str:
    it = cfg.get("dataset", {}).get("image_transforms", {})
    if not it.get("enable"):
        return "none"
    tfs = it.get("tfs") or {}
    parts = ["LeRobot image augmentation"]
    if any(t.get("type") == "RandomErasing" for t in tfs.values()):
        parts.append("random spatial masking")
    drop = cfg.get("policy", {}).get("camera_dropout") or {}
    if drop:
        parts.append("camera dropout " + ", ".join(f"{k.split('.')[-1]} {v:.0%}" for k, v in drop.items()))
    return " + ".join(parts)


def section(repo: str, cfg: dict) -> tuple[str, str]:
    d = cfg["dataset"]
    ds, rev = d["repo_id"], d.get("revision")
    f = dataset_facts(ds, rev) if "/" in ds else {"method": "unknown", "lead": "unknown", "sha": None}
    pol = cfg.get("policy", {})
    rows = [
        ("Dataset", f"[`{ds}`](https://huggingface.co/datasets/{ds})" if "/" in ds else f"`{ds}` (local)"),
        ("Dataset revision", f"`{rev}`" if rev else "not pinned (latest at training time)"),
        ("Person removal", f["method"]),
        ("Action labels (lead)", f["lead"]),
        ("Policy", f"`{pol.get('type')}`"),
        ("Training steps", f"{cfg.get('steps'):,}" if cfg.get("steps") else "unknown"),
        ("Batch size", str(cfg.get("batch_size", "unknown"))),
        ("Image augmentation", augmentation(cfg)),
        ("Seed", str(cfg.get("seed", "unknown"))),
    ]
    if repo.endswith("-ema"):
        rows.append(("Weights", "exponential moving average (EMA) of the training weights"))
    body = "\n".join(f"| {k} | {v} |" for k, v in rows)
    text = (f"{BEGIN}\n## Training details\n\nRead from this run's `train_config.json` and the dataset's metadata.\n\n"
            f"| | |\n|---|---|\n{body}\n{END}")
    return text, ds


def update(repo: str, apply: bool, show: bool) -> str:
    files = [s.rfilename for s in api.model_info(repo).siblings]
    cfg = train_config(repo, files)
    if not cfg or not cfg.get("dataset", {}).get("repo_id"):
        return "SKIP (no train_config.json with a dataset)"
    text, ds = section(repo, cfg)
    try:
        card = ModelCard.load(repo)
    except (EntryNotFoundError, Exception):
        card = ModelCard(f"---\n---\n\n# {repo.split('/')[1]}\n")
    content = card.text
    if BEGIN in content:
        content = re.sub(re.escape(BEGIN) + r".*?" + re.escape(END), lambda m: text, content, flags=re.S)
    else:
        content = content.rstrip() + "\n\n" + text + "\n"
    card.text = content
    if "/" in ds:
        existing = card.data.datasets or []
        existing = [existing] if isinstance(existing, str) else list(existing)  # LeRobot writes a bare string
        if ds not in existing:
            card.data.datasets = existing + [ds]
    if show:
        full = str(card)
        print(full[:full.index("library_name")] + "...")
        print("...\n" + full[full.index(BEGIN):])
    if apply:
        card.push_to_hub(repo, commit_message="Model card: add training dataset")
        return f"updated ({ds})"
    return f"would update ({ds})"


if __name__ == "__main__":
    apply = "--apply" in sys.argv
    shown = False
    for m in sorted(api.list_models(author="andnetdeboer"), key=lambda m: m.id):
        res = update(m.id, apply, show=not apply and not shown and "effecterase" in m.id)
        shown = shown or ("effecterase" in m.id)
        print(f"{m.id}: {res}", flush=True)
