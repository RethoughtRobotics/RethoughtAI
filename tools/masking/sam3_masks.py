"""Save SAM3 person masks for a LeRobot dataset's camera videos, for video-inpainting models to consume.

One lossless grayscale mask video per source video (same frame count and order, 0 = keep, 255 = person),
written to <out>/<videos/...>/file-XXX.mkv (FFV1). Masks are the raw SAM3 union, not dilated: each remover
applies its own dilation. Uses the same SAM3 settings and multi-GPU workers as sam3_inpaint_dataset.py.

    python tools/masking/sam3_masks.py --source DATASET --out MASK_DIR --camera observation.images.third_eye --workers 4
"""

import argparse
import json
import multiprocessing as mp
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
from sam3_inpaint_dataset import load_sam3, person_union_masks, video_paths  # noqa: E402

_worker = {}


def _init_worker(device: str, gpu_ids):
    if device == "cuda":
        torch.cuda.set_device(gpu_ids.get())
    _worker["processor"], _worker["model"] = load_sam3(device)


def masks_for_video(path: Path, source: Path, out_root: Path, args) -> dict:
    rel = path.relative_to(source)
    out = (out_root / rel).with_suffix(".mkv")
    out.parent.mkdir(parents=True, exist_ok=True)
    s = json.loads(subprocess.check_output(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
                                            "stream=width,height,r_frame_rate", "-of", "json", str(path)], text=True))["streams"][0]
    width, height = int(s["width"]), int(s["height"])
    num, den = s["r_frame_rate"].split("/")
    decoder = subprocess.Popen(["ffmpeg", "-v", "error", "-i", str(path), "-f", "rawvideo", "-pix_fmt", "rgb24", "-"],
                               stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    encoder = subprocess.Popen(["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "gray", "-s", f"{width}x{height}",
                                "-r", f"{num}/{den}", "-i", "-", "-c:v", "ffv1", "-pix_fmt", "gray", str(out)],
                               stdin=subprocess.PIPE, stderr=subprocess.DEVNULL)
    frame_bytes = width * height * 3
    frames, area, started = 0, 0.0, time.time()
    proc, model = _worker["processor"], _worker["model"]
    while True:
        batch = []
        while len(batch) < args.batch_size:
            raw = decoder.stdout.read(frame_bytes)
            if len(raw) != frame_bytes:
                break
            batch.append(np.frombuffer(raw, np.uint8).reshape(height, width, 3).copy())
        if not batch:
            break
        for mask, _ in person_union_masks(proc, model, batch, args.prompt, args.threshold, args.device, True):
            encoder.stdin.write((mask.astype(np.uint8) * 255).tobytes())
            area += float(mask.mean())
            frames += 1
        if frames % (args.batch_size * 250) < args.batch_size:
            print(f"  {rel}: {frames} frames", flush=True)
    decoder.stdout.close(); decoder.wait(); encoder.stdin.close(); encoder.wait()
    if encoder.returncode != 0 or decoder.returncode != 0:
        raise RuntimeError(f"ffmpeg failed for {path}")
    n = int(subprocess.check_output(["ffprobe", "-v", "error", "-count_packets", "-select_streams", "v:0", "-show_entries",
                                     "stream=nb_read_packets", "-of", "csv=p=0", str(out)], text=True).strip())
    if n != frames:
        raise RuntimeError(f"mask frame count {n} != {frames} for {out}")
    return {"video": str(rel), "masks": str(out.relative_to(out_root)), "frames": frames,
            "mean_mask_fraction": area / max(frames, 1), "seconds": round(time.time() - started, 1)}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--camera", action="append", dest="cameras", required=True)
    p.add_argument("--prompt", default="person")
    p.add_argument("--threshold", type=float, default=0.3)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--device", default="cuda")
    args = p.parse_args()
    source, out = args.source.resolve(), args.out.resolve()
    files = video_paths(source, args.cameras)
    if not files:
        raise SystemExit("No videos found")
    ctx = mp.get_context("spawn")
    gpu_ids = ctx.Queue()
    n_gpus = max(torch.cuda.device_count(), 1)
    workers = min(args.workers, len(files))
    for w in range(workers):
        gpu_ids.put(w % n_gpus)
    print(f"{len(files)} videos, {workers} workers over {n_gpus} GPU(s)", flush=True)
    results, started = {}, time.time()
    with ProcessPoolExecutor(workers, mp_context=ctx, initializer=_init_worker, initargs=(args.device, gpu_ids)) as pool:
        futs = {pool.submit(masks_for_video, f, source, out, args): f for f in files}
        for i, fut in enumerate(as_completed(futs), 1):
            r = fut.result(); results[futs[fut]] = r
            print(f"[{i}/{len(files)}] {r['video']}: {r['frames']} masks, {r['mean_mask_fraction']:.2%} person, "
                  f"{r['frames'] / max(r['seconds'], 1e-6):.1f} fps", flush=True)
    out.mkdir(parents=True, exist_ok=True)
    (out / "sam3_masks_report.json").write_text(json.dumps(
        {"source": str(source), "prompt": args.prompt, "threshold": args.threshold, "dilation": 0,
         "videos": [results[f] for f in files]}, indent=2) + "\n")
    print(f"done in {time.time() - started:.0f}s", flush=True)


if __name__ == "__main__":
    main()
