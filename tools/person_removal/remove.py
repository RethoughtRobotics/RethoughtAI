"""Step 2 of the pipeline: remove the masked person from a dataset's camera videos with one method.

Per video, frames and masks are streamed in clips of `clip_len` frames, consecutive clips sharing `overlap`
frames. A clip with no person in it is copied as it is. Otherwise the method inpaints it, and every frame
gets the method's output only through its feathered region (masks.alpha): pixels away from the person
(the robot, the objects) stay identical to the recording. Where two clips overlap, their outputs are
cross-faded with linear ramps. Only the current clip and a few frames around it are held in memory.

The written videos use the source camera's LeRobot encoding; other cameras and all metadata are copied.

Work splits two ways. `workers` processes run side by side, one model each, spread over the visible GPUs.
For more GPUs than one machine has, `segment_frames` cuts the videos at episode starts into spans of about
that many frames; `shard K/M` then runs every M-th span from K, as a separate run, writing lossless parts,
and `assemble` joins all parts into the output dataset once every shard is done.
"""

import json
import multiprocessing as mp
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path

import dataset
import masks as masks_mod
import methods
import numpy as np
from video import LOSSLESS_RGB, Writer, count_frames, probe, read_frames, read_masks


@dataclass
class Settings:
    method: str
    steps: int
    clip_len: int
    overlap: int
    dilation: int
    temporal: int
    feather: int
    max_frames: int = 0  # smoke tests: only the first frames of each video

    @classmethod
    def for_method(cls, method: str, **overrides):
        """The method's DEFAULTS, with the `overrides` that are not None."""
        values = {**methods.get(method).DEFAULTS, **{k: v for k, v in overrides.items() if v is not None}}
        return cls(method=method, **values)


def clip_starts(n: int, clip_len: int, step: int) -> list[int]:
    """First frame of each clip covering frames [0, n), `step` apart; the last clip ends at n."""
    starts = list(range(0, max(n - clip_len, 0) + 1, step))
    if starts[-1] + clip_len < n:
        starts.append(n - clip_len)
    return starts


def ramp(s: int, e: int, n: int, overlap: int, next_start: int) -> np.ndarray:
    """Cross-fade weights of the clip [s, e) of a video of n frames: rising over the overlap with the clip
    before it, falling from `next_start`, where the next clip begins."""
    r = np.ones(e - s, np.float32)
    if s > 0:
        k = min(overlap, e - s)
        r[:k] = np.linspace(0, 1, k + 2)[1:-1]
    if e < n:
        k = min(e - next_start, e - s)
        if k > 0:
            r[-k:] = np.minimum(r[-k:], np.linspace(1, 0, k + 2)[1:-1])
    return r


def remove_video(inpaint, video: Path, mask: Path, out: Path, codec: list[str], cfg: Settings, span=None) -> dict:
    """Write `video` with the person removed to `out`, encoded with `codec`; with span=(first, end), only
    those frames (a segment, written to a lossless part)."""
    width, height, rate = probe(video)
    if span is None:
        n, n_masks = count_frames(video), count_frames(mask)
        if n != n_masks:
            raise RuntimeError(f'{video}: {n} frames but {n_masks} masks')
        span = (0, n)
    first, n = span[0], span[1] - span[0]
    if cfg.max_frames:
        n = min(n, cfg.max_frames)
    clip_len, overlap, mt = cfg.clip_len, cfg.overlap, cfg.temporal
    starts = clip_starts(n, clip_len, clip_len - overlap)
    frame_it, mask_it = read_frames(video, width, height, first), read_masks(mask, width, height, first)
    frames, raw = {}, {}  # index: frame and raw mask, for the frames still needed
    filled, weight = {}, {}  # index: accumulated method output and its cross-fade weight
    read_to = written = 0

    def region_at(i):
        return masks_mod.region([raw[i]] + [raw[j] for j in range(max(0, i - mt), min(n, i + mt + 1))], cfg.dilation)

    def read_until(stop):
        nonlocal read_to
        while read_to < min(stop, n):
            frames[read_to], raw[read_to] = next(frame_it), next(mask_it)
            read_to += 1

    started, model_clips = time.time(), 0
    with Writer(out, width, height, rate, codec) as writer:

        def flush(upto):  # frames before `upto` get no more clips: composite and encode them
            nonlocal written
            while written < upto:
                i, f = written, frames[written]
                if i in filled and weight[i] > 0:
                    a = masks_mod.alpha(region_at(i), cfg.feather)
                    f = f.astype(np.float32) * (1 - a) + (filled.pop(i) / weight.pop(i)) * a + 0.5
                    f = f.clip(0, 255).astype(np.uint8)
                writer.write(f)
                written += 1
                for j in [k for k in frames if k < written - mt]:  # keep mt frames back for the temporal union
                    del frames[j], raw[j]

        for c, s in enumerate(starts):
            e = min(s + clip_len, n)
            next_start = starts[c + 1] if c + 1 < len(starts) else n
            read_until(e + mt)
            regions = np.stack([region_at(i) for i in range(s, e)])
            if regions.any():
                clip = np.stack([frames[i] for i in range(s, e)])
                if len(clip) < clip_len:  # a short video: repeat its last frame (the video models want 4k+1)
                    pad = clip_len - len(clip)
                    clip = np.concatenate([clip, np.repeat(clip[-1:], pad, 0)])
                    regions = np.concatenate([regions, np.repeat(regions[-1:], pad, 0)])
                result = inpaint(clip, regions, cfg.steps)[: e - s].astype(np.float32)
                model_clips += 1
                w = ramp(s, e, n, overlap, next_start)
                for t, i in enumerate(range(s, e)):
                    filled[i] = filled.get(i, 0) + result[t] * w[t]
                    weight[i] = weight.get(i, 0) + w[t]
            flush(next_start)
            if c % 25 == 0:
                per_frame = (time.time() - started) / max(written, 1)
                print(f'  {video.name}[{first}:{first + n}]: clip {c + 1}/{len(starts)}, {written} frames, '
                      f'{per_frame:.3f} s/frame', flush=True)  # fmt: skip
        flush(n)
    seconds = time.time() - started
    return {'start': first, 'frames': n, 'clips': len(starts), 'model_clips': model_clips,
            'seconds': round(seconds, 1), 'seconds_per_frame': round(seconds / max(n, 1), 4)}  # fmt: skip


# One model per worker process, loaded once
_worker = {}


def _init_worker(method: str, models: str | None, gpus):
    # Show this worker only its own GPU, as cuda:0, before anything starts CUDA in the process:
    # MiniMax-Remover's pipeline hard-codes cuda:0.
    k = gpus.get()
    visible = os.environ.get('CUDA_VISIBLE_DEVICES')
    os.environ['CUDA_VISIBLE_DEVICES'] = visible.split(',')[k] if visible else str(k)
    _worker['inpaint'] = methods.get(method).load(Path(models) if models else None)


def _remove_in_worker(*args, **kwargs):
    return remove_video(_worker['inpaint'], *args, **kwargs)


def part_path(parts: Path, rel: Path, first: int, end: int) -> Path:
    return parts / rel.with_suffix('') / f'part_{first:07d}_{end:07d}.mkv'


def run(
    source: Path,
    mask_dir: Path,
    out: Path,
    cameras: list[str],
    cfg: Settings,
    models: Path | None = None,
    workers: int = 1,
    segment_frames: int = 0,
    shard: str = '0/1',
    parts: Path | None = None,
) -> list[dict]:
    """Process a dataset (see the module docstring). Without segment_frames, writes the whole output dataset;
    with it, writes this shard's parts (then `assemble`)."""  # fmt: skip
    source, out = source.resolve(), out.resolve()
    videos = dataset.video_paths(source, cameras)
    if not videos:
        raise FileNotFoundError(f'no videos of {cameras} in {source}')
    if segment_frames:
        if parts is None:
            raise ValueError('segments are written as parts: give a parts directory')
        k, m = (int(x) for x in shard.split('/'))
        spans = dataset.segments(source, videos, segment_frames)
        tasks = sorted([t for i, t in enumerate(spans) if i % m == k], key=lambda t: t[1] - t[2])  # longest first
        print(f'{len(spans)} segments; shard {k}/{m} has {len(tasks)} ({sum(b - a for _, a, b in tasks)} frames)')
    else:
        dataset.copy_without_videos(source, out)
        dataset.copy_videos(source, out, sorted(set(dataset.video_paths(source)) - set(videos)))
        tasks = [(v, None, None) for v in videos]

    try:
        import torch

        gpu_count = max(torch.cuda.device_count(), 1)
    except ImportError:  # Telea needs no torch
        gpu_count = 1
    ctx = mp.get_context('spawn')
    gpus = ctx.Queue()
    workers = max(1, min(workers, len(tasks)))
    for i in range(workers):
        gpus.put(i % gpu_count)
    print(f'{cfg.method}: {len(tasks)} videos or segments, {workers} workers on {gpu_count} GPU(s), {cfg}', flush=True)
    results, started = [], time.time()
    init = (cfg.method, str(models) if models else None, gpus)
    with ProcessPoolExecutor(workers, mp_context=ctx, initializer=_init_worker, initargs=init) as pool:
        futures = {}
        for video, a, b in tasks:
            rel = video.relative_to(source)
            mask = (mask_dir / rel).with_suffix('.mkv')
            if a is None:
                job = (video, mask, out / rel, dataset.encoder_args(source, video), cfg)
                futures[pool.submit(_remove_in_worker, *job)] = rel
            else:
                job = (video, mask, part_path(parts.resolve(), rel, a, b), LOSSLESS_RGB, cfg, (a, b))
                futures[pool.submit(_remove_in_worker, *job)] = rel
        for i, future in enumerate(as_completed(futures), 1):
            r = {'video': str(futures[future]), **future.result()}
            results.append(r)
            done = sum(x['frames'] for x in results)
            print(f'[{i}/{len(tasks)}] {r["video"]}@{r["start"]}: {r["frames"]} frames, {r["model_clips"]}/{r["clips"]} '
                  f'clips had a person, {r["seconds_per_frame"]:.3f} s/frame; {done} frames in '
                  f'{time.time() - started:.0f} s', flush=True)  # fmt: skip

    if segment_frames:
        parts.mkdir(parents=True, exist_ok=True)
        (parts / f'shard_{k}_of_{m}.json').write_text(json.dumps(results, indent=2))
    else:
        write_report(out, cfg, mask_dir, results)
    return results


def assemble(source: Path, mask_dir: Path, out: Path, cameras: list[str], cfg: Settings, parts: Path) -> list[dict]:
    """Join the shards' lossless parts into the output dataset, each video encoded as its source."""
    source, out, parts = source.resolve(), out.resolve(), parts.resolve()
    videos = dataset.video_paths(source, cameras)
    dataset.copy_without_videos(source, out)
    dataset.copy_videos(source, out, sorted(set(dataset.video_paths(source)) - set(videos)))
    results = []
    for video in videos:
        rel = video.relative_to(source)
        files = sorted((parts / rel.with_suffix('')).glob('part_*.mkv'))
        spans = [tuple(int(x) for x in p.stem.split('_')[1:]) for p in files]
        n = count_frames(video)
        if (
            not spans
            or spans[0][0] != 0
            or spans[-1][1] != n
            or any(b != c for (_, b), (c, _) in zip(spans, spans[1:]))
        ):
            raise RuntimeError(f'{rel}: parts {spans} do not cover [0, {n}); is a shard missing?')
        width, height, rate = probe(video)
        with Writer(out / rel, width, height, rate, dataset.encoder_args(source, video)) as writer:
            for part, (a, b) in zip(files, spans):
                before = writer.frames
                for frame in read_frames(part, width, height):
                    writer.write(frame)
                if writer.frames - before != b - a:
                    raise RuntimeError(f'{part}: {writer.frames - before} frames, expected {b - a}')
        results.append({'video': str(rel), 'frames': n, 'parts': len(files)})
        print(f'assembled {rel}: {n} frames from {len(files)} parts', flush=True)
    segments = [r for p in sorted(parts.glob('shard_*.json')) for r in json.loads(p.read_text())]
    write_report(out, cfg, mask_dir, results, segments)
    return results


def write_report(out: Path, cfg: Settings, mask_dir: Path, videos: list[dict], segments=None):
    """meta/person_removal_<method>.json: how the dataset was made."""
    report = {**asdict(cfg), 'masks': str(mask_dir), 'videos': videos}
    if segments is not None:
        report['segments'] = segments
    (out / 'meta' / f'person_removal_{cfg.method}.json').write_text(json.dumps(report, indent=2) + '\n')
