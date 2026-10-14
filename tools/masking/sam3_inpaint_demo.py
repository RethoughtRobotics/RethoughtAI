"""SAM3 text-prompted person removal + inpainting demo on sampled frames.

Offline only: reads PNG samples, writes comparison panels. No ROS, no dataset edits.
Loads facebook/sam3 via transformers (gated; requires accepted access).
"""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw


def load_sam3(device: str):
    from transformers import Sam3Model, Sam3Processor

    processor = Sam3Processor.from_pretrained("facebook/sam3")
    model = Sam3Model.from_pretrained("facebook/sam3").eval().to(device)
    return processor, model


def person_mask(processor, model, image: Image.Image, prompt: str, threshold: float, device: str) -> tuple[np.ndarray, int]:
    inputs = processor(images=image, text=prompt, return_tensors="pt").to(device)
    with torch.inference_mode():
        outputs = model(**inputs)
    results = processor.post_process_instance_segmentation(
        outputs, threshold=threshold, mask_threshold=0.5, target_sizes=[(image.height, image.width)]
    )[0]
    masks = results["masks"]
    if masks is None or len(masks) == 0:
        return np.zeros((image.height, image.width), dtype=bool), 0
    arr = masks.cpu().numpy().astype(bool)
    return arr.any(0), int(len(arr))


def dilate(mask: np.ndarray, radius: int) -> np.ndarray:
    if radius <= 0:
        return mask
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1))
    return cv2.dilate(mask.astype(np.uint8), kernel) > 0


def panel(images_titles, pad_top=24) -> Image.Image:
    h = images_titles[0][0].shape[0]
    w = images_titles[0][0].shape[1]
    canvas = Image.new("RGB", (w * len(images_titles), h + pad_top), "#161b22")
    draw = ImageDraw.Draw(canvas)
    for i, (pixels, title) in enumerate(images_titles):
        canvas.paste(Image.fromarray(pixels), (w * i, pad_top))
        draw.text((w * i + 5, 5), title, fill="white")
    return canvas


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--glob", default="*third_eye.png")
    parser.add_argument("--prompt", default="person")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--threshold", type=float, default=0.3)
    parser.add_argument("--dilation", type=int, default=3)
    parser.add_argument("--inpaint-radius", type=int, default=5)
    args = parser.parse_args()

    if args.out.exists():
        raise ValueError(f"Output exists; choose a new directory: {args.out}")
    args.out.mkdir(parents=True)

    frames = sorted(args.samples_dir.glob(args.glob))
    if not frames:
        raise ValueError(f"No frames matching {args.glob} in {args.samples_dir}")

    processor, model = load_sam3(args.device)
    rows = []
    for path in frames:
        with Image.open(path) as src:
            image = src.convert("RGB")
        rgb = np.asarray(image)
        mask, n = person_mask(processor, model, image, args.prompt, args.threshold, args.device)
        mask = dilate(mask, args.dilation)

        overlay = rgb.copy()
        overlay[mask] = np.round(0.5 * rgb[mask] + 0.5 * np.array([255, 0, 180])).astype(np.uint8)
        gray = rgb.copy()
        gray[mask] = 127
        inpainted = cv2.inpaint(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), mask.astype(np.uint8) * 255,
                                args.inpaint_radius, cv2.INPAINT_TELEA)
        inpainted = cv2.cvtColor(inpainted, cv2.COLOR_BGR2RGB)

        stem = path.stem
        comparison = panel([(rgb, "Original (person visible)"), (overlay, f"SAM3 '{args.prompt}' ({n} inst)"),
                            (gray, "Gray-fill (artifact)"), (inpainted, "Inpainted (person-free)")])
        comparison.save(args.out / f"{stem}_compare.jpg", quality=92)
        Image.fromarray(mask.astype(np.uint8) * 255).save(args.out / f"{stem}_mask.png")
        Image.fromarray(inpainted).save(args.out / f"{stem}_inpainted.png")
        rows.append({"frame": path.name, "instances": n, "mask_fraction": float(mask.mean())})
        print(f"{stem} instances={n} mask={mask.mean():.2%}", flush=True)

    (args.out / "results.json").write_text(json.dumps(
        {"prompt": args.prompt, "threshold": args.threshold, "dilation": args.dilation,
         "inpaint_radius": args.inpaint_radius, "device": args.device, "frames": rows}, indent=2) + "\n")
    print(f"Wrote {len(rows)} comparisons to {args.out}")


if __name__ == "__main__":
    main()
