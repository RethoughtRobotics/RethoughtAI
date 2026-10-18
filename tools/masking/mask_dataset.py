"""Build a masked LeRobot dataset variant by replacing person pixels in camera videos.

The source dataset is never modified. Non-video files are copied as-is.
"""

import argparse
import importlib.util
import json
import subprocess
import shutil
from pathlib import Path

import numpy as np
from PIL import Image
import cv2


def load_backend(name: str, device: str):
    evaluate_path = Path(__file__).with_name("evaluate.py")
    spec = importlib.util.spec_from_file_location("masking_evaluate", evaluate_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    if name == "deeplab":
        return module.DeepLab(device)
    if name == "maskrcnn":
        return module.MaskRCNN(device)
    raise ValueError(f"Unsupported backend: {name}")


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
        if not root.exists():
            continue
        files.extend(sorted(root.glob("chunk-*/file-*.mp4")))
    return files


def dilate(mask: np.ndarray, radius: int) -> np.ndarray:
    if radius <= 0:
        return mask
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1))
    return cv2.dilate(mask.astype(np.uint8), kernel) > 0


def mask_video(path: Path, source_root: Path, output_root: Path, model, threshold: float, fill: int, prompts: list[str],
               dilation: int, max_frames: int) -> dict:
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
        "ffmpeg", "-v", "error", "-i", str(path),
        "-f", "rawvideo", "-pix_fmt", "rgb24", "-"
    ], stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    encoder = subprocess.Popen([
        "ffmpeg", "-v", "error", "-y",
        "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{width}x{height}", "-r", str(fps), "-i", "-",
        "-c:v", "libx264", "-pix_fmt", "yuv420p", str(out)
    ], stdin=subprocess.PIPE, stderr=subprocess.DEVNULL)

    frames = 0
    fractions = []
    frame_bytes = width * height * 3
    early_stop = False
    while True:
        if max_frames > 0 and frames >= max_frames:
            early_stop = True
            break
        raw = decoder.stdout.read(frame_bytes)
        if len(raw) != frame_bytes:
            break
        rgb = np.frombuffer(raw, dtype=np.uint8).reshape((height, width, 3)).copy()
        image = Image.fromarray(rgb)
        outputs = model.predict(image, prompts, threshold)
        union = np.zeros((height, width), dtype=bool)
        for _, mask, _ in outputs:
            union |= np.asarray(mask, dtype=bool)
        union = dilate(union, dilation)
        rgb[union] = fill
        encoder.stdin.write(rgb.tobytes())
        fractions.append(float(union.mean()))
        frames += 1

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
        "mask_fraction_p50": float(np.percentile(fractions, 50)) if fractions else 0.0,
        "mask_fraction_p95": float(np.percentile(fractions, 95)) if fractions else 0.0,
        "mask_fraction_mean": float(np.mean(fractions)) if fractions else 0.0,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True, help="Input LeRobot dataset folder")
    parser.add_argument("--out", type=Path, required=True, help="Output masked dataset folder")
    parser.add_argument("--backend", choices=["deeplab", "maskrcnn"], default="maskrcnn")
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--threshold", type=float, default=0.5, help="Person threshold")
    parser.add_argument("--fill", type=int, default=127, help="Gray fill value 0..255")
    parser.add_argument("--dilation", type=int, default=2, help="Mask dilation radius in pixels")
    parser.add_argument("--camera", action="append", dest="cameras",
                        help="Camera key under videos/, e.g. observation.images.third_eye")
    parser.add_argument("--max-videos", type=int, default=0, help="Debug: limit processed videos")
    parser.add_argument("--max-frames", type=int, default=0, help="Debug: limit frames per video")
    parser.add_argument("--prompts", nargs="+", default=["person", "human hand", "human forearm"])
    args = parser.parse_args()

    if not (0 <= args.fill <= 255):
        parser.error("--fill must be between 0 and 255")
    if not (0.0 <= args.threshold <= 1.0):
        parser.error("--threshold must be between 0 and 1")

    source = args.source.resolve()
    output = args.out.resolve()
    copy_tree_without_videos(source, output)

    model = load_backend(args.backend, args.device)
    files = video_paths(source, args.cameras)
    if args.max_videos > 0:
        files = files[:args.max_videos]
    if not files:
        raise ValueError("No source videos found")

    stats = []
    for i, video in enumerate(files, start=1):
        record = mask_video(video, source, output, model, args.threshold, args.fill, args.prompts, args.dilation,
                    args.max_frames)
        stats.append(record)
        print(f"[{i}/{len(files)}] {record['source']} mean={record['mask_fraction_mean']:.2%}", flush=True)

    report = {
        "source": str(source),
        "output": str(output),
        "backend": args.backend,
        "device": args.device,
        "threshold": args.threshold,
        "fill": args.fill,
        "dilation": args.dilation,
        "prompts": args.prompts,
        "videos": stats,
    }
    (output / "meta" / "masking_report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(f"Masked dataset written: {output}")


if __name__ == "__main__":
    main()