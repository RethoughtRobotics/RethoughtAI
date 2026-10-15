"""Build a full LeRobot dataset variant using SAM3 person masks + inpainting.

Copies all non-video files unchanged and rewrites selected camera videos with
person regions inpainted. The source dataset is never modified.
"""

import argparse
import json
import shutil
import subprocess
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image


def copy_tree_without_videos(source: Path, output: Path):
    if output.exists():
        raise ValueError(f"Output exists: {output}")
    output.mkdir(parents=True)
    for path in source.rglob("*"):
        rel = path.relative_to(source)
        dest = output / rel
        if path.is_dir():
            dest.mkdir(exist_ok=True)
            continue
        if rel.parts and rel.parts[0] == "videos":
            continue
        shutil.copy2(path, dest)


def video_paths(source: Path, cameras: list[str] | None):
    roots = [source / "videos" / c for c in cameras] if cameras else sorted((source / "videos").glob("observation.images.*"))
    files = []
    for root in roots:
        if root.exists():
            files.extend(sorted(root.glob("chunk-*/file-*.mp4")))
    return files


def load_sam3(device: str):
    from transformers import Sam3Model, Sam3Processor

    processor = Sam3Processor.from_pretrained("facebook/sam3")
    model = Sam3Model.from_pretrained("facebook/sam3").eval().to(device)
    return processor, model


def person_union_masks(processor, model, rgbs: list[np.ndarray], prompt: str, threshold: float, device: str,
                       bf16: bool):
    """Union of the prompt's instance masks for a batch of frames. bf16 autocast is ~4x faster than fp32 on an
    A100 at batch 1 and ~7x at batch 8, with mask IoU vs fp32 of 0.997 mean on this dataset's frames."""
    images = [Image.fromarray(rgb) for rgb in rgbs]
    inputs = processor(images=images, text=[prompt] * len(images), return_tensors="pt").to(device)
    use_bf16 = bf16 and device == "cuda"
    if use_bf16:
        inputs["pixel_values"] = inputs["pixel_values"].to(torch.bfloat16)
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_bf16):
        outputs = model(**inputs)
    results = processor.post_process_instance_segmentation(
        outputs, threshold=threshold, mask_threshold=0.5, target_sizes=[(im.height, im.width) for im in images]
    )
    out = []
    for im, result in zip(images, results):
        masks = result["masks"]
        if masks is None or len(masks) == 0:
            out.append((np.zeros((im.height, im.width), dtype=bool), 0))
        else:
            arr = masks.detach().cpu().numpy().astype(bool)
            out.append((arr.any(0), int(len(arr))))
    return out


def dilate(mask: np.ndarray, radius: int):
    if radius <= 0:
        return mask
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1))
    return cv2.dilate(mask.astype(np.uint8), kernel) > 0


def encoder_args(source_root: Path, rel: Path):
    """Re-encode with the source camera's own LeRobot settings so meta/info.json stays true and training
    keeps fast random access (LeRobot default: AV1, GOP 2, CRF 30, preset 12, yuv420p)."""
    info = json.loads((source_root / "meta" / "info.json").read_text())
    v = info["features"][rel.parts[1]]["info"]
    codec = {"av1": "libsvtav1", "h264": "libx264", "hevc": "libx265"}[v["video.codec"]]
    args = ["-c:v", codec, "-pix_fmt", v.get("video.pix_fmt", "yuv420p"), "-g", str(v.get("video.g", 2)),
            "-crf", str(v.get("video.crf", 30))]
    if v.get("video.preset") is not None:
        args += ["-preset", str(v["video.preset"])]
    return args


def rewrite_video(path: Path, source_root: Path, output_root: Path, processor, model, args):
    rel = path.relative_to(source_root)
    out = output_root / rel
    out.parent.mkdir(parents=True, exist_ok=True)

    info = json.loads(subprocess.check_output([
        "ffprobe", "-v", "error", "-select_streams", "v:0",
        "-show_entries", "stream=width,height,r_frame_rate,nb_frames",
        "-of", "json", str(path)
    ], text=True))
    stream = info["streams"][0]
    width = int(stream["width"])
    height = int(stream["height"])
    rate = stream.get("r_frame_rate", "25/1")
    num, den = rate.split("/")
    fps = float(num) / float(den) if float(den) else 25.0
    src_count = int(stream["nb_frames"]) if str(stream.get("nb_frames", "")).isdigit() else None

    decoder = subprocess.Popen([
        "ffmpeg", "-v", "error", "-i", str(path), "-f", "rawvideo", "-pix_fmt", "rgb24", "-"
    ], stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    encoder = subprocess.Popen([
        "ffmpeg", "-v", "error", "-y",
        "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{width}x{height}", "-r", str(fps), "-i", "-",
        *encoder_args(source_root, rel), str(out)
    ], stdin=subprocess.PIPE, stderr=subprocess.DEVNULL)

    frame_bytes = width * height * 3
    frames = 0
    total_mask_fraction = 0.0
    total_instances = 0
    early_stop = False
    done = False
    while not done:
        batch = []
        while len(batch) < args.batch_size:
            if args.max_frames > 0 and frames + len(batch) >= args.max_frames:
                early_stop = done = True
                break
            raw = decoder.stdout.read(frame_bytes)
            if len(raw) != frame_bytes:
                done = True
                break
            batch.append(np.frombuffer(raw, dtype=np.uint8).reshape((height, width, 3)).copy())
        if not batch:
            break
        results = person_union_masks(processor, model, batch, args.prompt, args.threshold, args.device, args.bf16)
        for rgb, (mask, instances) in zip(batch, results):  # frames stay in order
            mask = dilate(mask, args.dilation)
            bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            inpainted = cv2.inpaint(bgr, (mask.astype(np.uint8) * 255), args.inpaint_radius, cv2.INPAINT_TELEA)
            out_rgb = cv2.cvtColor(inpainted, cv2.COLOR_BGR2RGB)
            encoder.stdin.write(out_rgb.tobytes())
            total_mask_fraction += float(mask.mean())
            total_instances += instances
            frames += 1
        if frames % (args.batch_size * 250) < args.batch_size:
            print(f"  {rel}: {frames} frames", flush=True)

    decoder.stdout.close()
    if early_stop:
        decoder.terminate()
    decoder.wait()
    encoder.stdin.close()
    encoder.wait()
    if encoder.returncode != 0 or (not early_stop and decoder.returncode != 0):
        raise ValueError(f"ffmpeg failed for {path} -> {out}")

    out_info = json.loads(subprocess.check_output([
        "ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=nb_frames",
        "-of", "json", str(out)
    ], text=True))
    out_count = out_info["streams"][0].get("nb_frames")
    out_count = int(out_count) if str(out_count).isdigit() else frames
    if out_count != frames:
        raise ValueError(f"Frame count mismatch for {out}: wrote {frames}, probed {out_count}")

    return {
        "source": str(rel),
        "output": str(rel),
        "frames": frames,
        "source_reported_frames": src_count,
        "fps": fps,
        "mean_mask_fraction": (total_mask_fraction / frames) if frames else 0.0,
        "mean_instances_per_frame": (total_instances / frames) if frames else 0.0,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", choices=["cpu", "cuda"], default=("cuda" if torch.cuda.is_available() else "cpu"))
    parser.add_argument("--prompt", default="person")
    parser.add_argument("--threshold", type=float, default=0.3)
    parser.add_argument("--dilation", type=int, default=3)
    parser.add_argument("--inpaint-radius", type=int, default=6)
    parser.add_argument("--camera", action="append", dest="cameras", help="videos/<camera>, default all")
    parser.add_argument("--max-videos", type=int, default=0)
    parser.add_argument("--max-frames", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=8, help="frames per SAM3 forward pass")
    parser.add_argument("--no-bf16", dest="bf16", action="store_false", help="run SAM3 in fp32 (~7x slower)")
    args = parser.parse_args()

    source = args.source.resolve()
    output = args.out.resolve()
    copy_tree_without_videos(source, output)

    processor, model = load_sam3(args.device)
    files = video_paths(source, args.cameras)
    # Cameras that are not masked are copied unchanged; otherwise the output would lack their videos.
    for video in sorted(set(video_paths(source, None)) - set(files)):
        dest = output / video.relative_to(source)
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(video, dest)
        print(f"copied unmasked {video.relative_to(source)}", flush=True)
    if args.max_videos > 0:
        files = files[:args.max_videos]
    if not files:
        raise ValueError("No source videos found")

    stats = []
    for i, video in enumerate(files, start=1):
        record = rewrite_video(video, source, output, processor, model, args)
        stats.append(record)
        print(f"[{i}/{len(files)}] {record['source']} mask={record['mean_mask_fraction']:.2%}", flush=True)

    report = {
        "source": str(source),
        "output": str(output),
        "backend": "sam3+inpaint",
        "device": args.device,
        "prompt": args.prompt,
        "threshold": args.threshold,
        "dilation": args.dilation,
        "inpaint_radius": args.inpaint_radius,
        "batch_size": args.batch_size,
        "bf16": args.bf16,
        "videos": stats,
    }
    (output / "meta" / "sam3_inpaint_report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(f"Inpainted dataset written: {output}")


if __name__ == "__main__":
    main()