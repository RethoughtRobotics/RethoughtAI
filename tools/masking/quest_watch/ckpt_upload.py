"""Upload local LeRobot checkpoints to their Hub model repos and verify every file byte-for-byte.

  upload  make each run's repo public, push outputs/<run>/checkpoints/<step>/ to checkpoints/<step>/ (resumable)
  verify  for every local checkpoint file: present on the Hub, same size, same hash (LFS sha256 or git blob sha1)
          -> writes ckpt_verify.json; nothing is deleted
"""
import hashlib
import json
import sys
from pathlib import Path

from huggingface_hub import HfApi

OUT = Path("/projects/e52938/kmy2091/outputs")
REPORT = Path(__file__).with_name("ckpt_verify.json")
api = HfApi()


def runs():
    for d in sorted(OUT.iterdir()):
        ck = d / "checkpoints"
        if ck.is_dir() and any(p.name[0].isdigit() for p in ck.iterdir()):
            yield d


def upload():
    for d in runs():
        repo = f"andnetdeboer/{d.name}"
        api.update_repo_settings(repo, private=False)
        print(f"{repo}: public; uploading {sorted(p.name for p in (d / 'checkpoints').iterdir() if p.name != 'last')}", flush=True)
        api.upload_large_folder(repo_id=repo, repo_type="model", folder_path=str(d),
                                allow_patterns=["checkpoints/0*/**"], print_report=False)
        print(f"{repo}: upload done", flush=True)


def local_hashes(p: Path):
    sha256, sha1 = hashlib.sha256(), hashlib.sha1()
    sha1.update(f"blob {p.stat().st_size}\0".encode())
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 24), b""):
            sha256.update(b); sha1.update(b)
    return sha256.hexdigest(), sha1.hexdigest()


def verify():
    report = {}
    for d in runs():
        repo = f"andnetdeboer/{d.name}"
        info = api.model_info(repo, files_metadata=True)
        remote = {s.rfilename: s for s in info.siblings}
        steps = {}
        for step in sorted(p for p in (d / "checkpoints").iterdir() if p.name[0].isdigit() and not p.is_symlink()):
            files, bad = 0, []
            for f in sorted(step.rglob("*")):
                if not f.is_file():
                    continue
                rel = f.relative_to(d).as_posix()
                files += 1
                s = remote.get(rel)
                if s is None:
                    bad.append(f"{rel}: missing on Hub"); continue
                size = f.stat().st_size
                if s.size != size:
                    bad.append(f"{rel}: size {s.size} != local {size}"); continue
                h256, h1 = local_hashes(f)
                ok = (s.lfs.sha256 == h256) if s.lfs else (s.blob_id == h1)
                if not ok:
                    bad.append(f"{rel}: hash mismatch")
            steps[step.name] = {"files": files, "verified": files - len(bad), "problems": bad}
            print(f"{repo} {step.name}: {files - len(bad)}/{files} files verified" + (f"  PROBLEMS {bad[:3]}" if bad else ""), flush=True)
        report[d.name] = {"repo": repo, "private": info.private, "steps": steps}
    REPORT.write_text(json.dumps(report, indent=2))
    total = sum(s["files"] for r in report.values() for s in r["steps"].values())
    ok = sum(s["verified"] for r in report.values() for s in r["steps"].values())
    print(f"VERIFIED {ok}/{total} files across {len(report)} runs -> {REPORT}", flush=True)


if __name__ == "__main__":
    {"upload": upload, "verify": verify}[sys.argv[1]]()
