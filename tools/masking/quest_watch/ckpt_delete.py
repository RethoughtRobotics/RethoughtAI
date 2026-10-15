"""Delete local checkpoint steps that ckpt_verify.json shows fully verified on the Hub.

A step is deleted only if: it is in the report, files > 0, verified == files, problems is empty, and
outputs/<run>/checkpoints/<step> is still a real directory. checkpoints/last is removed only if it is a
symlink that is dangling or points at a deleted step. Everything else is left alone.

  python ckpt_delete.py          dry run: print what would be deleted
  python ckpt_delete.py --apply  delete and append to ckpt_deleted.txt
"""
import json
import shutil
import sys
import time
from pathlib import Path

OUT = Path("/projects/e52938/kmy2091/outputs")
HERE = Path(__file__).parent
REPORT = HERE / "ckpt_verify.json"
LOG = HERE / "ckpt_deleted.txt"


def size(p: Path) -> int:
    return sum(f.stat().st_size for f in p.rglob("*") if f.is_file() and not f.is_symlink())


def main(apply: bool) -> None:
    report = json.loads(REPORT.read_text())
    lines, freed, kept = [], 0, []
    for run, r in sorted(report.items()):
        ck = OUT / run / "checkpoints"
        deleted = set()
        for step, s in sorted(r["steps"].items()):
            d = ck / step
            ok = s["files"] > 0 and s["verified"] == s["files"] and not s["problems"]
            if not ok:
                kept.append(f"KEEP {run}/{step}: verified {s['verified']}/{s['files']} problems={s['problems'][:2]}")
                continue
            if d.is_symlink() or not d.is_dir():
                kept.append(f"KEEP {run}/{step}: not a real directory now")
                continue
            b = size(d)
            lines.append(f"{run}/checkpoints/{step}\t{s['files']} files\t{b / 1e9:.2f} GB\t{r['repo']}")
            freed += b
            deleted.add(step)
            if apply:
                shutil.rmtree(d)
        last = ck / "last"
        if last.is_symlink():
            target = last.readlink().name
            if target in deleted or not last.exists():
                lines.append(f"{run}/checkpoints/last\tsymlink -> {target}\t(dangling)")
                if apply:
                    last.unlink()
    for k in kept:
        print(k)
    for l in lines:
        print(("DELETED " if apply else "WOULD DELETE ") + l)
    print(f"{'freed' if apply else 'would free'} {freed / 1e9:.1f} GB in {sum('files' in l for l in lines)} steps")
    if apply:
        with open(LOG, "a") as f:
            f.write(f"# {time.strftime('%F %T')} ckpt_delete.py --apply (report {REPORT.name})\n")
            f.writelines(l + "\n" for l in lines)
            f.write(f"# freed {freed / 1e9:.1f} GB; kept: {len(kept)}\n")
            f.writelines(f"# {k}\n" for k in kept)


if __name__ == "__main__":
    main("--apply" in sys.argv)
