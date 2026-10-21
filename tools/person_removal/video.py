"""Video files through ffmpeg: probing, streaming frames out, and encoding frames in.

Frames are streamed as raw pixels through pipes, so a video never has to fit in memory.
"""

import json
import subprocess
import tempfile
from collections.abc import Iterator
from pathlib import Path

import numpy as np

# Lossless intermediate (segment parts, masks): FFV1 keeps every pixel exact.
LOSSLESS_RGB = ['-c:v', 'ffv1', '-pix_fmt', 'gbrp']
LOSSLESS_GRAY = ['-c:v', 'ffv1', '-pix_fmt', 'gray']


def probe(path: Path) -> tuple[int, int, str]:
    """(width, height, frame rate as ffmpeg's fraction string, e.g. '25/1')."""
    out = subprocess.check_output(
        ['ffprobe', '-v', 'error', '-select_streams', 'v:0', '-show_entries', 'stream=width,height,r_frame_rate',
         '-of', 'json', str(path)],
        text=True,
    )  # fmt: skip
    s = json.loads(out)['streams'][0]
    return int(s['width']), int(s['height']), s['r_frame_rate']


def count_frames(path: Path) -> int:
    """Frames in the video, counted from its packets (exact, unlike the container's nb_frames)."""
    out = subprocess.check_output(
        ['ffprobe', '-v', 'error', '-count_packets', '-select_streams', 'v:0', '-show_entries',
         'stream=nb_read_packets', '-of', 'csv=p=0', str(path)],
        text=True,
    )  # fmt: skip
    return int(out.strip())


def read_frames(path: Path, width: int, height: int, start: int = 0) -> Iterator[np.ndarray]:
    """(H, W, 3) uint8 RGB frames from frame `start` on."""
    yield from _read(path, 'rgb24', width, height, start)


def read_masks(path: Path, width: int, height: int, start: int = 0) -> Iterator[np.ndarray]:
    """(H, W) bool masks (a gray video, > 127 is the person) from frame `start` on."""
    for gray in _read(path, 'gray', width, height, start):
        yield gray[..., 0] > 127


def _read(path, pix_fmt, width, height, start):
    channels = 3 if pix_fmt == 'rgb24' else 1
    select = ['-vf', f'select=gte(n\\,{start})', '-fps_mode', 'passthrough'] if start else []
    proc = subprocess.Popen(
        ['ffmpeg', '-v', 'error', '-i', str(path), *select, '-f', 'rawvideo', '-pix_fmt', pix_fmt, '-'],
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
    )
    size = width * height * channels
    try:
        while True:
            buf = proc.stdout.read(size)
            if len(buf) != size:
                break
            yield np.frombuffer(buf, np.uint8).reshape(height, width, channels)
    finally:
        proc.stdout.close()
        proc.wait()


class Writer:
    """Encodes raw frames into `path` with the given ffmpeg codec arguments; checks the frame count on close.

    with Writer(path, width, height, rate, codec_args, pix_fmt='rgb24') as w:
        w.write(frame)
    """

    def __init__(self, path: Path, width: int, height: int, rate: str, codec_args: list[str], pix_fmt: str = 'rgb24'):
        self.path, self.frames = Path(path), 0
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.log = tempfile.TemporaryFile()  # ffmpeg's messages, for the error if it fails
        self.proc = subprocess.Popen(
            ['ffmpeg', '-v', 'error', '-y', '-f', 'rawvideo', '-pix_fmt', pix_fmt, '-s', f'{width}x{height}',
             '-r', rate, '-i', '-', *codec_args, str(self.path)],
            stdin=subprocess.PIPE,
            stderr=self.log,
        )  # fmt: skip

    def write(self, frame: np.ndarray):
        try:
            self.proc.stdin.write(np.ascontiguousarray(frame).tobytes())
        except BrokenPipeError:
            self.proc.wait()
            raise RuntimeError(f'{self.path}: ffmpeg stopped (exit {self.proc.returncode}): {self.errors()}') from None
        self.frames += 1

    def errors(self) -> str:
        """The last lines ffmpeg wrote."""
        self.log.seek(0)
        return ' | '.join(self.log.read().decode(errors='replace').strip().splitlines()[-3:])

    def __enter__(self):
        return self

    def __exit__(self, exc_type, *_):
        try:
            self.proc.stdin.close()
        except BrokenPipeError:
            pass
        self.proc.wait()
        try:
            if exc_type is None:
                if self.proc.returncode != 0:
                    raise RuntimeError(f'{self.path}: ffmpeg exit {self.proc.returncode}: {self.errors()}')
                got = count_frames(self.path)
                if got != self.frames:
                    raise RuntimeError(f'{self.path}: wrote {self.frames} frames, the file has {got}')
        finally:
            self.log.close()
