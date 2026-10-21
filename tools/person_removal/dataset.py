"""A local LeRobot v3 dataset's layout: its camera videos, a copy without them, and how they were encoded.

videos/<camera>/chunk-XXX/file-XXX.mp4 hold many episodes each; meta/episodes/*.parquet says where every
episode starts in its file. Only video files are ever rewritten: states, actions, timestamps and all other
metadata are copied unchanged, so the output is the same dataset with different pixels.
"""

import json
import shutil
from pathlib import Path

from video import count_frames


def video_paths(root: Path, cameras: list[str] | None = None) -> list[Path]:
    """The camera videos of the dataset at `root`, of `cameras` (feature keys) or of every camera."""
    dirs = [root / 'videos' / c for c in cameras] if cameras else sorted((root / 'videos').glob('observation.images.*'))
    return [f for d in dirs if d.exists() for f in sorted(d.glob('chunk-*/file-*.mp4'))]


def copy_without_videos(source: Path, out: Path):
    """Copy everything except videos/ to a new directory `out`."""
    if out.exists():
        raise FileExistsError(f'{out} exists; pick another output')
    out.mkdir(parents=True)
    for path in source.rglob('*'):
        rel = path.relative_to(source)
        if rel.parts[0] == 'videos':
            continue
        if path.is_dir():
            (out / rel).mkdir(exist_ok=True)
        else:
            shutil.copy2(path, out / rel)


def copy_videos(source: Path, out: Path, videos: list[Path]):
    """Copy `videos` (paths under `source`) unchanged, e.g. the cameras that are not processed."""
    for v in videos:
        dest = out / v.relative_to(source)
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(v, dest)


def encoder_args(source: Path, video: Path) -> list[str]:
    """ffmpeg arguments that re-encode `video` (a path under `source`) as LeRobot encoded its camera, so
    meta/info.json stays true and training decodes it as fast as before (LeRobot's default: AV1, GOP 2,
    CRF 30, yuv420p)."""
    camera = video.relative_to(source).parts[1]
    v = json.loads((source / 'meta/info.json').read_text())['features'][camera]['info']
    codec = {'av1': 'libsvtav1', 'h264': 'libx264', 'hevc': 'libx265'}[v['video.codec']]
    args = ['-c:v', codec, '-pix_fmt', v.get('video.pix_fmt', 'yuv420p'), '-g', str(v.get('video.g', 2)),
            '-crf', str(v.get('video.crf', 30))]  # fmt: skip
    if v.get('video.preset') is not None:
        args += ['-preset', str(v['video.preset'])]
    return args


def segment_cuts(episode_starts: list[int], frames: int, target: int) -> list[tuple[int, int]]:
    """Split a video of `frames` frames into spans of about `target` frames, cut only where an episode
    starts, so no span shares an episode with another and they need no blending when joined."""
    cuts, last = [0], 0
    for s in sorted(episode_starts)[1:]:
        if s - last >= target:
            cuts.append(s)
            last = s
    cuts.append(frames)
    return [(a, b) for a, b in zip(cuts, cuts[1:]) if b > a]


def segments(source: Path, videos: list[Path], target: int) -> list[tuple[Path, int, int]]:
    """(video, first frame, end frame) spans of about `target` frames, over all `videos`."""
    import pandas as pd

    fps = json.loads((source / 'meta/info.json').read_text())['fps']
    episodes = pd.concat([pd.read_parquet(f) for f in sorted((source / 'meta/episodes').rglob('*.parquet'))])
    out = []
    for v in videos:
        camera, chunk, file = v.parent.parent.name, int(v.parent.name.split('-')[1]), int(v.stem.split('-')[1])
        here = episodes[
            (episodes[f'videos/{camera}/chunk_index'] == chunk) & (episodes[f'videos/{camera}/file_index'] == file)
        ]
        starts = [int(round(t * fps)) for t in here[f'videos/{camera}/from_timestamp']]
        out += [(v, a, b) for a, b in segment_cuts(starts, count_frames(v), target)]
    return out
