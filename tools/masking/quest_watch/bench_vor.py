"""Compare person-removal methods on one real clip: speed and a side-by-side sheet.

  masks   (sam3-venv)        clip frames + SAM3 masks -> clip.npz
  run     (method's venv)    clip.npz -> out_<method>_<steps>.npz with timing
  sheet   (any venv)         raw | telea (v1) | each method, a few frames -> sheet.jpg
"""
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
SRC = Path("/projects/e52938/kmy2091/datasets/SweepTrash_Kin-Full/videos/observation.images.third_eye/chunk-000/file-000.mp4")
V1 = Path("/projects/e52938/kmy2091/datasets/h100/SweepTrash_Kin-Full_inpainted/videos/observation.images.third_eye/chunk-000/file-000.mp4")
START, N, W, H = 3000, 161, 640, 480  # 161 = two overlapping 81-frame clips


def decode(path, start, n):
    raw = subprocess.check_output(["ffmpeg", "-v", "error", "-i", str(path), "-vf", f"select='between(n\\,{start}\\,{start + n - 1})'",
                                   "-vsync", "0", "-f", "rawvideo", "-pix_fmt", "rgb24", "-"])
    return np.frombuffer(raw, np.uint8).reshape(-1, H, W, 3)[:n]


def masks():
    sys.path.insert(0, "/projects/e52938/kmy2091/RethoughtAI-adapter/tools/masking")
    from sam3_inpaint_dataset import load_sam3, person_union_masks
    frames = decode(SRC, START, N)
    proc, model = load_sam3("cuda")
    m = []
    for i in range(0, len(frames), 8):
        m += [mk for mk, _ in person_union_masks(proc, model, list(frames[i:i + 8]), "person", 0.3, "cuda", True)]
    np.savez_compressed(HERE / "bench/clip.npz", frames=frames, masks=np.stack(m))
    print("clip", frames.shape, "person fraction", float(np.mean(m)))


def run(method, steps):
    sys.path.insert(0, "/projects/e52938/kmy2091/RethoughtAI-adapter/tools/masking")
    import cv2
    import torch
    from video_remove import LOADERS
    d = np.load(HERE / "bench/clip.npz")
    frames, m = d["frames"], d["masks"]
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (13, 13))
    reg = np.stack([cv2.dilate(x.astype(np.uint8), k) > 0 for x in m])
    t0 = time.time(); fn = LOADERS[method](HERE); load_s = time.time() - t0
    outs, times = [], []
    for s in (0, 80):  # the two clips the runner would use (overlap 1 frame here; timing only)
        torch.cuda.synchronize(); t = time.time()
        outs.append(fn(frames[s:s + 81], reg[s:s + 81], steps))
        torch.cuda.synchronize(); times.append(time.time() - t)
    out = np.concatenate([outs[0], outs[1][1:]])
    np.savez_compressed(HERE / f"bench/out_{method}_{steps}.npz", out=out)
    r = {"method": method, "steps": steps, "load_s": round(load_s, 1), "clip_s": [round(x, 1) for x in times],
         "fps": round(81 / times[-1], 2), "gpu_mem_gb": round(torch.cuda.max_memory_allocated() / 1e9, 1),
         "est_hours_81k_frames_1gpu": round(81017 / (81 / times[-1]) / 3600, 2)}
    print(json.dumps(r)); (HERE / f"bench/out_{method}_{steps}.json").write_text(json.dumps(r))


def sheet():
    import cv2
    from PIL import Image, ImageDraw
    d = np.load(HERE / "bench/clip.npz"); frames, m = d["frames"], d["masks"]
    v1 = decode(V1, START, N) if V1.exists() else None
    cols = [("raw", frames)] + ([("telea (v1)", v1)] if v1 is not None else [])
    for f in sorted((HERE / "bench").glob("out_*.npz")):
        o = np.load(f)["out"]
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (13, 13))
        a = np.stack([cv2.GaussianBlur(cv2.dilate(x.astype(np.uint8), k).astype(np.float32), (0, 0), 3) for x in m])[..., None]
        comp = (frames * (1 - a) + o * a).astype(np.uint8)  # what the runner writes: composited inside the mask
        cols.append((f.stem[4:], comp))
    picks = sorted(set(int(i) for i in np.linspace(5, N - 5, 4)) | {int(m.reshape(N, -1).sum(1).argmax())})
    rows = []
    for i in picks:
        tiles = []
        for name, v in cols:
            im = Image.fromarray(v[i]).resize((400, 300)); ImageDraw.Draw(im).text((6, 4), f"{name} #{START + i}", fill=(255, 0, 0))
            tiles.append(np.asarray(im))
        rows.append(np.concatenate(tiles, 1))
    Image.fromarray(np.concatenate(rows, 0)).save(HERE / "bench/sheet.jpg", quality=88)
    print("sheet", [c[0] for c in cols], "frames", [START + i for i in picks])


if __name__ == "__main__":
    {"masks": masks, "run": lambda: run(sys.argv[2], int(sys.argv[3])), "sheet": sheet}[sys.argv[1]]()
