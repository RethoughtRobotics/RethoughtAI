"""Person masks: SAM3 per frame (step 1 of the pipeline), and the clean-up every removal method applies.

SAM3 (facebook/sam3, transformers >= 5) is prompted with text ('person') and returns instance masks; their
union is the frame's mask. One lossless gray mask video is written per camera video (same frame count and
order, 255 = person), raw: no dilation, so each removal method can apply its own.

SAM3 segments every frame on its own and flickers at the person's edges, so before inpainting a mask is
unioned with its neighbours' (+-`temporal` frames) and dilated (`region`); the model's output is then pasted
back through a feathered version of it (`alpha`), so pixels away from the person stay exactly as recorded.
"""

import json
import multiprocessing as mp
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import cv2
import dataset
import numpy as np
from video import LOSSLESS_GRAY, Writer, probe, read_frames


def load_sam3(device: str = 'cuda'):
    """(processor, model) of SAM3."""
    from transformers import Sam3Model, Sam3Processor

    return Sam3Processor.from_pretrained('facebook/sam3'), Sam3Model.from_pretrained('facebook/sam3').eval().to(device)


def person_masks(sam3, frames: list[np.ndarray], prompt='person', threshold=0.3, device='cuda') -> list[np.ndarray]:
    """(H, W) bool union of the prompt's instances, per frame of a batch. bf16 on the GPU: ~7x faster than
    fp32 at batch 8 on an A100, mask IoU 0.997 against fp32."""
    import torch
    from PIL import Image

    processor, model = sam3
    images = [Image.fromarray(f) for f in frames]
    inputs = processor(images=images, text=[prompt] * len(images), return_tensors='pt').to(device)
    bf16 = device == 'cuda'
    if bf16:
        inputs['pixel_values'] = inputs['pixel_values'].to(torch.bfloat16)
    with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16, enabled=bf16):
        outputs = model(**inputs)
    results = processor.post_process_instance_segmentation(
        outputs, threshold=threshold, mask_threshold=0.5, target_sizes=[(im.height, im.width) for im in images]
    )
    out = []
    for im, r in zip(images, results):
        m = r['masks']
        out.append(np.zeros((im.height, im.width), bool) if m is None or len(m) == 0 else m.cpu().numpy().any(0))
    return out


def mask_video(sam3, video: Path, out: Path, prompt='person', threshold=0.3, batch=8, device='cuda') -> dict:
    """Write the person masks of `video` to `out` (a lossless gray .mkv)."""
    width, height, rate = probe(video)
    started, area, pending = time.time(), 0.0, []
    with Writer(out, width, height, rate, LOSSLESS_GRAY, pix_fmt='gray') as writer:

        def flush():
            nonlocal area
            for m in person_masks(sam3, pending, prompt, threshold, device):
                writer.write(m.astype(np.uint8) * 255)
                area += float(m.mean())
            pending.clear()

        for frame in read_frames(video, width, height):
            pending.append(frame.copy())
            if len(pending) == batch:
                flush()
        if pending:
            flush()
        frames = writer.frames
    seconds = time.time() - started
    return {'frames': frames, 'person_fraction': area / max(frames, 1), 'seconds': round(seconds, 1),
            'seconds_per_frame': round(seconds / max(frames, 1), 4)}  # fmt: skip


_worker = {}


def _init_worker(gpus):
    k = gpus.get()  # this worker's GPU only, as cuda:0
    visible = os.environ.get('CUDA_VISIBLE_DEVICES')
    os.environ['CUDA_VISIBLE_DEVICES'] = visible.split(',')[k] if visible else str(k)
    _worker['sam3'] = load_sam3('cuda')


def _mask_in_worker(*args, **kwargs):
    return mask_video(_worker['sam3'], *args, **kwargs)


def run(source: Path, out: Path, cameras: list[str], workers: int = 1, prompt='person', threshold=0.3, batch=8):
    """Masks of every video of `cameras` in the dataset at `source`, as out/videos/<camera>/chunk-*/file-*.mkv,
    and out/masks_report.json. SAM3 runs at about 8 frames/s per A100 worker; spread workers over GPUs."""
    import torch

    source, out = source.resolve(), out.resolve()
    videos = dataset.video_paths(source, cameras)
    if not videos:
        raise FileNotFoundError(f'no videos of {cameras} in {source}')
    gpu_count = max(torch.cuda.device_count(), 1)
    ctx = mp.get_context('spawn')
    gpus = ctx.Queue()
    workers = max(1, min(workers, len(videos)))
    for i in range(workers):
        gpus.put(i % gpu_count)
    print(f'SAM3 {prompt!r}: {len(videos)} videos, {workers} workers on {gpu_count} GPU(s)', flush=True)
    results, started = {}, time.time()
    with ProcessPoolExecutor(workers, mp_context=ctx, initializer=_init_worker, initargs=(gpus,)) as pool:
        futures = {}
        for v in videos:
            rel = v.relative_to(source)
            futures[pool.submit(_mask_in_worker, v, (out / rel).with_suffix('.mkv'), prompt, threshold, batch)] = rel
        for i, future in enumerate(as_completed(futures), 1):
            r = results[futures[future]] = {'video': str(futures[future]), **future.result()}
            print(f'[{i}/{len(videos)}] {r["video"]}: {r["frames"]} masks, {r["person_fraction"]:.2%} person, '
                  f'{r["seconds_per_frame"]:.3f} s/frame', flush=True)  # fmt: skip
    report = {'source': str(source), 'prompt': prompt, 'threshold': threshold,
              'videos': [results[v.relative_to(source)] for v in videos]}  # fmt: skip
    (out / 'masks_report.json').write_text(json.dumps(report, indent=2) + '\n')
    print(f'masks done in {time.time() - started:.0f} s: {out}')
    return report


def dilate(mask: np.ndarray, radius: int) -> np.ndarray:
    """Grow a bool mask by `radius` pixels (elliptical)."""
    if radius <= 0:
        return mask
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1))
    return cv2.dilate(mask.astype(np.uint8), kernel) > 0


def region(window: list[np.ndarray], dilation: int) -> np.ndarray:
    """The region to inpaint in a frame: the union of the raw masks of the frame and its neighbours
    (`window`, the frames +-temporal around it), dilated."""
    m = window[0].copy()
    for other in window[1:]:
        m |= other
    return dilate(m, dilation)


def alpha(region_mask: np.ndarray, feather: int) -> np.ndarray:
    """(H, W, 1) float weight of the inpainted pixels: 1 inside the region, falling off over `feather` pixels
    outside it, 0 elsewhere (a hard edge with feather 0)."""
    a = region_mask.astype(np.float32)
    if feather > 0:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * feather + 1, 2 * feather + 1))
        soft = cv2.GaussianBlur(
            cv2.dilate(region_mask.astype(np.uint8), kernel).astype(np.float32), (0, 0), feather / 2
        )
        a = np.maximum(soft, a)
    return np.clip(a, 0, 1)[..., None]
