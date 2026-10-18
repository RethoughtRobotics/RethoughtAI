"""Remove people from a LeRobot dataset's camera videos with a video-inpainting model and saved SAM3 masks.

Methods (both built on Wan2.1-1.3B, run at native resolution in 81-frame clips):
  minimax      MiniMax-Remover (NeurIPS 2025), distilled, no CFG, 12 steps by default.
  effecterase  EffectErase (CVPR 2026), also removes the person's shadows/reflections; CC BY-NC 4.0.

Per video file: frames and masks (from sam3_masks.py) are streamed in clips of --clip-len frames overlapping by
--overlap. Masks are unioned over +-2 frames (SAM3 is per-frame and flickers) and dilated. Clips with no person are
copied unchanged. The model output is composited back only inside a feathered mask, so pixels away from the person
(robot, objects) stay identical to the source, and overlapping clips are cross-faded. Output videos are re-encoded
with the source camera's LeRobot settings; other cameras and all metadata are copied unchanged.
"""

import argparse
import json
import multiprocessing as mp
import shutil
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import cv2
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
from sam3_inpaint_dataset import copy_tree_without_videos, encoder_args, video_paths  # noqa: E402

_w = {}


# --------------------------------------------------------------------------- models
def load_minimax(root: Path):
    sys.path.insert(0, str(root / "MiniMax-Remover"))
    from diffusers.models import AutoencoderKLWan
    from diffusers.schedulers import UniPCMultistepScheduler
    from pipeline_minimax_remover import Minimax_Remover_Pipeline
    from transformer_minimax_remover import Transformer3DModel

    w = root / "weights/minimax-remover"
    pipe = Minimax_Remover_Pipeline(
        transformer=Transformer3DModel.from_pretrained(w / "transformer", torch_dtype=torch.float16),
        vae=AutoencoderKLWan.from_pretrained(w / "vae", torch_dtype=torch.float16),
        scheduler=UniPCMultistepScheduler.from_pretrained(w / "scheduler"),
    ).to("cuda")

    def run(frames: np.ndarray, masks: np.ndarray, steps: int) -> np.ndarray:
        t, h, w_ = masks.shape
        out = pipe(images=torch.from_numpy(frames).float() / 127.5 - 1.0,
                   masks=torch.from_numpy(masks[..., None].astype(np.float32)),
                   num_frames=t, height=h, width=w_, num_inference_steps=steps,
                   generator=torch.Generator(device="cuda").manual_seed(42), iterations=1).frames[0]
        return (np.clip(out, 0, 1) * 255 + 0.5).astype(np.uint8)

    return run


def load_effecterase(root: Path):
    sys.path.insert(0, str(root / "EffectErase"))
    sys.path.insert(0, str(root / "EffectErase/examples/remove_wan"))
    from diffsynth import ModelManager, WanRemovePipeline
    from infer_remove_wan import crop_square_from_pil
    from PIL import Image

    wan = root / "weights/Wan2.1-Fun-1.3B-InP"
    mm = ModelManager(device="cuda")
    mm.load_models([str(wan / "diffusion_pytorch_model.safetensors"), str(wan / "models_t5_umt5-xxl-enc-bf16.pth"),
                    str(wan / "Wan2.1_VAE.pth"), str(wan / "models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth")],
                   torch_dtype=torch.bfloat16)
    mm.load_lora_v2(str(root / "weights/effecterase/EffectErase.ckpt"), lora_alpha=1.0)
    pipe = WanRemovePipeline.from_model_manager(mm, torch_dtype=torch.bfloat16, device="cuda")
    pipe.enable_vram_management(num_persistent_param_in_dit=6 * 10**9)
    prompt = "Remove the specified object and all related effects, then restore a clean background."
    negative = ("细节模糊不清，字幕，作品，画作，画面，静止，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，"
                "画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，杂乱的背景，三条腿，背景人很多，倒着走")

    def run(frames: np.ndarray, masks: np.ndarray, steps: int) -> np.ndarray:
        t, h, w_ = masks.shape
        to_t = lambda x: torch.from_numpy(x).float().permute(3, 0, 1, 2).cuda() / 127.5 - 1.0  # T H W C -> C T H W
        m3 = np.repeat((masks[..., None] * 255).astype(np.uint8), 3, axis=-1)
        k = int(masks.reshape(t, -1).sum(1).argmax())  # reference crop from the frame with the most person
        ref = crop_square_from_pil(Image.fromarray(m3[k]), Image.fromarray(frames[k]), target_size=224).cuda()
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            video, _ = pipe(video_mask=to_t(m3), video_fg_bg=to_t(frames), video_bg=None, task="remove",
                            fg_first_img=ref, prompt_remove=prompt, negative_prompt=negative,
                            num_inference_steps=steps, cfg_scale=1.0, seed=2025, tiled=False, height=h, width=w_)
        return np.stack([np.asarray(f.convert("RGB")) for f in video])

    return run


LOADERS = {"minimax": load_minimax, "effecterase": load_effecterase}


def _init(method: str, root: str, gpu_ids):
    # Expose only this worker's GPU (as cuda:0) before CUDA starts in this process: MiniMax-Remover's pipeline
    # hard-codes "cuda:0", so torch.cuda.set_device(k) alone breaks a worker on any other GPU.
    import os
    k = gpu_ids.get()
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    os.environ["CUDA_VISIBLE_DEVICES"] = visible.split(",")[k] if visible else str(k)
    _w["run"] = LOADERS[method](Path(root))


# --------------------------------------------------------------------------- video io
def probe(path: Path):
    s = json.loads(subprocess.check_output(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
                                            "stream=width,height,r_frame_rate", "-of", "json", str(path)], text=True))["streams"][0]
    return int(s["width"]), int(s["height"]), s["r_frame_rate"]


def reader(path: Path, pix: str, w: int, h: int, start: int = 0):
    ch = 3 if pix == "rgb24" else 1
    sel = ["-vf", f"select=gte(n\\,{start})", "-fps_mode", "passthrough"] if start else []
    p = subprocess.Popen(["ffmpeg", "-v", "error", "-i", str(path), *sel, "-f", "rawvideo", "-pix_fmt", pix, "-"],
                         stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    n = w * h * ch
    while True:
        b = p.stdout.read(n)
        if len(b) != n:
            break
        a = np.frombuffer(b, np.uint8).reshape(h, w, ch)
        yield a if ch == 3 else a[..., 0] > 127
    p.stdout.close(); p.wait()


# --------------------------------------------------------------------------- per video
def count_frames(path: Path) -> int:
    return int(subprocess.check_output(["ffprobe", "-v", "error", "-count_packets", "-select_streams", "v:0", "-show_entries",
                                        "stream=nb_read_packets", "-of", "csv=p=0", str(path)], text=True).strip())


def clip_starts(n: int, L: int, step: int) -> list[int]:
    starts = list(range(0, max(n - L, 0) + 1, step))
    if starts[-1] + L < n:
        starts.append(n - L)
    return starts


def remove_video(path: Path, mask_path: Path, source: Path, out_root: Path, args,
                 seg: tuple[int, int] | None = None, part: Path | None = None) -> dict:
    """Streams the video: only the current clip window (+ temporal mask margin) is held in memory.

    With seg=(start, end) only those frames are processed and written losslessly to `part` (FFV1) for
    assemble(); segments are cut at episode boundaries, so no temporal blending is needed across them."""
    rel = path.relative_to(source)
    out = part if part is not None else out_root / rel
    out.parent.mkdir(parents=True, exist_ok=True)
    w, h, rate = probe(path)
    if seg is None:
        n, nm = count_frames(path), count_frames(mask_path)
        if n != nm:
            raise RuntimeError(f"{rel}: {n} frames but {nm} masks")
        seg = (0, n)
    first = seg[0]
    n = seg[1] - seg[0]
    if args.max_frames:
        n = min(n, args.max_frames)
    L, step, ov, mt = args.clip_len, args.clip_len - args.overlap, args.overlap, args.mask_temporal
    starts = clip_starts(n, L, step)
    fr_it, mk_it = reader(path, "rgb24", w, h, first), reader(mask_path, "gray", w, h, first)
    frames, masks = {}, {}             # index -> uint8 frame / bool raw mask, for frames still needed
    filled, wsum = {}, {}               # index -> float32 accumulated model output / cross-fade weight
    read_to = 0                         # frames [0, read_to) have been read
    written = 0                         # frames [0, written) have been encoded
    enc = subprocess.Popen(["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{w}x{h}",
                            "-r", rate, "-i", "-", *(LOSSLESS if part is not None else encoder_args(source, rel)), str(out)],
                           stdin=subprocess.PIPE, stderr=subprocess.DEVNULL)
    kf = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * args.feather + 1, 2 * args.feather + 1)) if args.feather else None

    def region_at(i):  # temporal union over +-mt frames, then spatial dilation
        m = masks[i].copy()
        for j in range(max(0, i - mt), min(n, i + mt + 1)):
            m |= masks[j]
        if args.dilation > 0:
            k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * args.dilation + 1, 2 * args.dilation + 1))
            m = cv2.dilate(m.astype(np.uint8), k) > 0
        return m

    def read_until(stop):
        nonlocal read_to
        while read_to < min(stop, n):
            frames[read_to] = next(fr_it); masks[read_to] = next(mk_it); read_to += 1

    def flush(upto):  # frames < upto can no longer receive contributions: composite and encode them
        nonlocal written
        while written < upto:
            i = written
            f = frames[i]
            if i in filled and wsum[i] > 0:
                reg = region_at(i)
                if kf is not None:
                    a = np.maximum(cv2.GaussianBlur(cv2.dilate(reg.astype(np.uint8), kf).astype(np.float32), (0, 0),
                                                    args.feather / 2), reg.astype(np.float32))
                else:
                    a = reg.astype(np.float32)
                a = np.clip(a, 0, 1)[..., None]
                f = (f.astype(np.float32) * (1 - a) + (filled[i] / wsum[i]) * a + 0.5).clip(0, 255).astype(np.uint8)
                del filled[i], wsum[i]
            enc.stdin.write(f.tobytes())
            written += 1
            for j in [k for k in frames if k < written - mt]:  # keep mt frames back for the temporal union
                del frames[j], masks[j]

    started, model_clips = time.time(), 0
    for ci, s in enumerate(starts):
        e = min(s + L, n)
        read_until(e + mt)
        region = np.stack([region_at(i) for i in range(s, e)])
        if region.any():
            clip = np.stack([frames[i] for i in range(s, e)])
            cm = region
            if len(clip) < L:  # short video: pad by repeating the last frame (model wants 4k+1 frames)
                pad = L - len(clip)
                clip = np.concatenate([clip, np.repeat(clip[-1:], pad, 0)])
                cm = np.concatenate([cm, np.repeat(cm[-1:], pad, 0)])
            res = _w["run"](clip, cm, args.steps)[: e - s].astype(np.float32)
            model_clips += 1
            ramp = np.ones(e - s, np.float32)
            nxt = starts[ci + 1] if ci + 1 < len(starts) else n
            if s > 0:  # fade in over the overlap with the previous clip
                k = min(ov, e - s)
                ramp[:k] = np.linspace(0, 1, k + 2)[1:-1]
            if e < n:  # fade out over the overlap with the next clip
                k = min(e - nxt, e - s)
                if k > 0:
                    ramp[-k:] = np.minimum(ramp[-k:], np.linspace(1, 0, k + 2)[1:-1])
            for t, i in enumerate(range(s, e)):
                filled[i] = filled.get(i, 0) + res[t] * ramp[t]
                wsum[i] = wsum.get(i, 0) + ramp[t]
        nxt = starts[ci + 1] if ci + 1 < len(starts) else n
        flush(nxt)
        if ci % 25 == 0:
            print(f"  {rel}[{first}:{first + n}]: clip {ci + 1}/{len(starts)}, {written} frames written, "
                  f"{written / max(time.time() - started, 1e-6):.1f} fps", flush=True)
    flush(n)
    enc.stdin.close(); enc.wait()
    got = count_frames(out)
    if enc.returncode != 0 or got != n:
        raise RuntimeError(f"{rel}: encoder rc={enc.returncode}, wrote {n} frames, probed {got}")
    return {"video": str(rel), "start": first, "frames": n, "clips": len(starts), "model_clips": model_clips,
            "seconds": round(time.time() - started, 1)}


# --------------------------------------------------------------------------- segments
LOSSLESS = ["-c:v", "ffv1", "-pix_fmt", "gbrp"]


def episode_segments(source: Path, files: list[Path], target: int) -> list[tuple[Path, int, int]]:
    """Split each video at episode boundaries into segments of about `target` frames."""
    import pandas as pd
    info = json.loads((source / "meta/info.json").read_text())
    eps = pd.concat([pd.read_parquet(f) for f in sorted((source / "meta/episodes").rglob("*.parquet"))])
    segs = []
    for f in files:
        cam, chunk, fi = f.parent.parent.name, int(f.parent.name.split("-")[1]), int(f.stem.split("-")[1])
        e = eps[(eps[f"videos/{cam}/chunk_index"] == chunk) & (eps[f"videos/{cam}/file_index"] == fi)]
        starts = sorted(int(round(t * info["fps"])) for t in e[f"videos/{cam}/from_timestamp"])
        n = count_frames(f)
        cuts, last = [0], 0
        for s0 in starts[1:]:
            if s0 - last >= target:
                cuts.append(s0); last = s0
        cuts.append(n)
        segs += [(f, a, b) for a, b in zip(cuts, cuts[1:]) if b > a]
    return segs


def part_path(parts: Path, rel: Path, a: int, b: int) -> Path:
    return parts / rel.with_suffix("") / f"part_{a:07d}_{b:07d}.mkv"


def assemble(source: Path, out: Path, parts: Path, files: list[Path]) -> list[dict]:
    """Concatenate each video's lossless parts in order and encode with the source's settings."""
    results = []
    for f in files:
        rel = f.relative_to(source)
        ps = sorted((parts / rel.with_suffix("")).glob("part_*.mkv"))
        spans = [tuple(int(x) for x in p.stem.split("_")[1:]) for p in ps]
        n = count_frames(f)
        if not spans or spans[0][0] != 0 or spans[-1][1] != n or any(b != c for (_, b), (c, _) in zip(spans, spans[1:])):
            raise RuntimeError(f"{rel}: parts {spans} do not cover [0, {n})")
        w, h, rate = probe(f)
        dst = out / rel; dst.parent.mkdir(parents=True, exist_ok=True)
        enc = subprocess.Popen(["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{w}x{h}",
                                "-r", rate, "-i", "-", *encoder_args(source, rel), str(dst)], stdin=subprocess.PIPE,
                               stderr=subprocess.DEVNULL)
        for p, (a, b) in zip(ps, spans):
            k = 0
            for fr in reader(p, "rgb24", w, h):
                enc.stdin.write(fr.tobytes()); k += 1
            if k != b - a:
                raise RuntimeError(f"{p}: {k} frames, expected {b - a}")
        enc.stdin.close(); enc.wait()
        got = count_frames(dst)
        if enc.returncode != 0 or got != n:
            raise RuntimeError(f"{rel}: assembled {got} frames, expected {n}")
        results.append({"video": str(rel), "frames": n, "parts": len(ps)})
        print(f"assembled {rel}: {n} frames from {len(ps)} parts", flush=True)
    return results


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--method", choices=sorted(LOADERS), required=True)
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--masks", type=Path, required=True, help="sam3_masks.py output directory")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--camera", action="append", dest="cameras", required=True)
    p.add_argument("--root", type=Path, default=Path.home() / ".cache" / "vor", help="repos + weights")
    p.add_argument("--steps", type=int, default=None, help="default: minimax 12, effecterase 50")
    p.add_argument("--clip-len", type=int, default=81)
    p.add_argument("--overlap", type=int, default=16)
    p.add_argument("--dilation", type=int, default=6)
    p.add_argument("--mask-temporal", type=int, default=2)
    p.add_argument("--feather", type=int, default=6)
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--max-videos", type=int, default=0)
    p.add_argument("--max-frames", type=int, default=0, help="smoke tests: only the first N frames of each video")
    p.add_argument("--segment-frames", type=int, default=0,
                   help="split videos at episode boundaries into ~N-frame segments (parallel across jobs/GPUs)")
    p.add_argument("--shard", default="0/1", help="K/M: process segments with index %% M == K")
    p.add_argument("--parts-dir", type=Path, help="where segment parts go (with --segment-frames / --assemble)")
    p.add_argument("--max-segments", type=int, default=0, help="smoke tests: only the first N segments")
    p.add_argument("--assemble", action="store_true", help="join parts into the output dataset (no model)")
    args = p.parse_args()
    args.steps = args.steps or {"minimax": 12, "effecterase": 50}[args.method]
    source, out = args.source.resolve(), args.out.resolve()
    files = video_paths(source, args.cameras)
    if args.max_videos:
        files = files[: args.max_videos]

    if args.assemble:
        copy_tree_without_videos(source, out)
        for v in sorted(set(video_paths(source, None)) - set(video_paths(source, args.cameras))):
            d = out / v.relative_to(source); d.parent.mkdir(parents=True, exist_ok=True); shutil.copy2(v, d)
        res = assemble(source, out, args.parts_dir.resolve(), files)
        reports = [json.loads(p.read_text()) for p in sorted(args.parts_dir.glob("shard_*.json"))]
        report = {"method": args.method, "steps": args.steps, "clip_len": args.clip_len, "overlap": args.overlap,
                  "dilation": args.dilation, "mask_temporal": args.mask_temporal, "feather": args.feather,
                  "masks": str(args.masks), "videos": res, "segments": [r for rep in reports for r in rep]}
        (out / "meta" / f"video_remove_{args.method}_report.json").write_text(json.dumps(report, indent=2) + "\n")
        print(f"assembled {len(res)} videos: {out}", flush=True)
        return

    if args.segment_frames:
        k, m = (int(x) for x in args.shard.split("/"))
        segs = episode_segments(source, files, args.segment_frames)
        if args.max_segments:
            segs = segs[: args.max_segments]
        tasks = [(f, a, b) for i, (f, a, b) in enumerate(segs) if i % m == k]
        tasks.sort(key=lambda t: t[1] - t[2])  # longest first
        print(f"{len(segs)} segments in total; shard {k}/{m} has {len(tasks)} "
              f"({sum(b - a for _, a, b in tasks)} frames)", flush=True)
    else:
        copy_tree_without_videos(source, out)
        for v in sorted(set(video_paths(source, None)) - set(files)):
            d = out / v.relative_to(source); d.parent.mkdir(parents=True, exist_ok=True); shutil.copy2(v, d)
        tasks = [(f, None, None) for f in files]

    ctx = mp.get_context("spawn")
    gpu_ids = ctx.Queue()
    n_gpus = max(torch.cuda.device_count(), 1)
    workers = min(args.workers, len(tasks))
    for i in range(workers):
        gpu_ids.put(i % n_gpus)
    print(f"{args.method}: {len(tasks)} tasks, {workers} workers over {n_gpus} GPU(s), steps={args.steps}", flush=True)
    results, t0 = [], time.time()
    with ProcessPoolExecutor(workers, mp_context=ctx, initializer=_init, initargs=(args.method, str(args.root), gpu_ids)) as pool:
        futs = {}
        for f, a, b in tasks:
            mask = (args.masks / f.relative_to(source)).with_suffix(".mkv")
            if a is None:
                futs[pool.submit(remove_video, f, mask, source, out, args)] = f
            else:
                part = part_path(args.parts_dir.resolve(), f.relative_to(source), a, b)
                futs[pool.submit(remove_video, f, mask, source, out, args, (a, b), part)] = f
        for i, fut in enumerate(as_completed(futs), 1):
            r = fut.result(); results.append(r)
            print(f"[{i}/{len(tasks)}] {r['video']}@{r['start']}: {r['frames']} frames, {r['model_clips']}/{r['clips']} "
                  f"clips inpainted, {r['frames'] / max(r['seconds'], 1e-6):.1f} fps, "
                  f"{time.time() - t0:.0f}s elapsed", flush=True)
    if args.segment_frames:
        args.parts_dir.mkdir(parents=True, exist_ok=True)
        (args.parts_dir / f"shard_{args.shard.replace('/', '_of_')}.json").write_text(json.dumps(results, indent=2))
        print(f"shard {args.shard} done in {time.time() - t0:.0f}s", flush=True)
        return
    report = {"method": args.method, "steps": args.steps, "clip_len": args.clip_len, "overlap": args.overlap,
              "dilation": args.dilation, "mask_temporal": args.mask_temporal, "feather": args.feather,
              "masks": str(args.masks), "videos": results}
    (out / "meta" / f"video_remove_{args.method}_report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(f"done in {time.time() - t0:.0f}s: {out}", flush=True)


if __name__ == "__main__":
    main()
