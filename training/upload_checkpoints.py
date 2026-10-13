"""
Upload every checkpoint of a training run to one HF model repo: each step on branch
`step-XXXXXX`, the final step on `main`, optionally adding the repo to a collection.

Works with rethought-lerobot tasks/*/train.py runs and lerobot-train runs, including PEFT/LoRA ones (both save
checkpoints/<step>/pretrained_model/). Optimizer/training state is not uploaded. Branches that
already exist are skipped, so it is safe to re-run.

Usage:
    python training/upload_checkpoints.py outputs/act_mytask me/act_MyTask [--collection SLUG] [--public]
"""
import argparse
from pathlib import Path

from huggingface_hub import HfApi


def main():
    p = argparse.ArgumentParser()
    p.add_argument("run_dir", type=Path)
    p.add_argument("repo_id")
    p.add_argument("--collection", default=None, help="HF collection slug to add the repo to")
    p.add_argument("--public", action="store_true", help="Create the repo public (default: private)")
    p.add_argument("--only_steps", default=None,
                   help="Comma-separated steps to upload (e.g. 10000,20000); the final step always goes to main")
    p.add_argument("--dry_run", action="store_true")
    args = p.parse_args()

    ckpt_root = args.run_dir.resolve() / "checkpoints"
    # Full checkpoints hold model.safetensors; PEFT (LoRA) checkpoints hold adapter_model.safetensors.
    steps = sorted(d for d in ckpt_root.iterdir()
                   if d.is_dir() and d.name.isdigit() and any(
                       (d / "pretrained_model" / f).exists() for f in ("model.safetensors", "adapter_model.safetensors")))
    if not steps:
        raise SystemExit(f"No complete checkpoints under {ckpt_root}")
    final = steps[-1]
    if args.only_steps:
        keep = {int(x) for x in args.only_steps.split(",")}
        steps = [s for s in steps if int(s.name) in keep]
    print(f"{args.repo_id}: {len(steps)} checkpoints, final = {final.name}", flush=True)
    if args.dry_run:
        for s in steps:
            print(f"  step-{s.name}  <- {s}")
        return

    api = HfApi()
    api.create_repo(args.repo_id, repo_type="model", private=not args.public, exist_ok=True)
    refs = api.list_repo_refs(args.repo_id)
    existing = {b.name for b in refs.branches}
    tags = {t.name for t in refs.tags}

    def is_complete(ckpt: Path, revision: str) -> bool:
        remote = {s.rfilename: s.size for s in
                  api.model_info(args.repo_id, revision=revision, files_metadata=True).siblings}
        files = [f for f in (ckpt / "pretrained_model").iterdir() if f.is_file()]
        if (ckpt / "training_step.pt").exists():
            files.append(ckpt / "training_step.pt")
        return all(remote.get(f.name) == f.stat().st_size for f in files)

    def upload(ckpt: Path, revision: str, message: str) -> None:
        api.upload_folder(folder_path=ckpt / "pretrained_model", repo_id=args.repo_id,
                          revision=revision, commit_message=message)
        if (ckpt / "training_step.pt").exists():
            api.upload_file(path_or_fileobj=ckpt / "training_step.pt", path_in_repo="training_step.pt",
                            repo_id=args.repo_id, revision=revision, commit_message=message)
        remote = {s.rfilename: s.size for s in
                  api.model_info(args.repo_id, revision=revision, files_metadata=True).siblings}
        for f in (ckpt / "pretrained_model").iterdir():
            if f.is_file():
                assert remote.get(f.name) == f.stat().st_size, f"{revision}/{f.name}: size mismatch"
        print(f"  {revision}: OK ({len(remote)} files)", flush=True)

    for s in steps:
        branch = f"step-{s.name}"
        if branch in tags:  # a same-named tag blocks commits to the branch
            api.delete_tag(args.repo_id, tag=branch)
            print(f"  {branch}: removed conflicting tag", flush=True)
        if branch in existing and is_complete(s, branch):
            print(f"  {branch}: already uploaded, skipping", flush=True)
            continue
        api.create_branch(args.repo_id, branch=branch, exist_ok=True)
        upload(s, branch, f"Checkpoint at step {int(s.name)}")
    upload(final, "main", f"Final checkpoint (step {int(final.name)})")

    if args.collection:
        api.add_collection_item(args.collection, item_id=args.repo_id, item_type="model", exists_ok=True)
        print(f"  added to collection {args.collection}", flush=True)
    print(f"https://huggingface.co/{args.repo_id}", flush=True)


if __name__ == "__main__":
    main()
