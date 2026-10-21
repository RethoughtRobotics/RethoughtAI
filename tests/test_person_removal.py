"""tools/person_removal: clip layout, cross-fades, episode segments and mask clean-up. No models, GPU or ffmpeg."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / 'tools/person_removal'))

pytest.importorskip('cv2')
import dataset  # noqa: E402
import masks  # noqa: E402
import methods  # noqa: E402
import remove  # noqa: E402


@pytest.mark.parametrize('n', [1, 50, 81, 82, 145, 146, 1000])
def test_clips_cover_every_frame(n):
    starts = remove.clip_starts(n, 81, 65)
    assert starts[0] == 0
    assert min(starts[-1] + 81, n) == n
    covered = np.zeros(n, bool)
    for s in starts:
        covered[s : s + 81] = True
    assert covered.all()


@pytest.mark.parametrize('n', [81, 146, 300, 1000])
def test_cross_fade_weights_sum_to_one(n):
    clip_len, overlap = 81, 16
    starts = remove.clip_starts(n, clip_len, clip_len - overlap)
    total = np.zeros(n, np.float32)
    for c, s in enumerate(starts):
        e = min(s + clip_len, n)
        total[s:e] += remove.ramp(s, e, n, overlap, starts[c + 1] if c + 1 < len(starts) else n)
    assert np.all(total > 0)
    # one clip: weight 1; two overlapping: ramps that add to 1, except the last clip, aligned to the end
    assert np.allclose(total[: starts[-1]], 1.0, atol=1e-6)


def test_segments_cut_only_at_episode_starts():
    starts = [0, 400, 900, 1300, 2100, 2600]
    spans = dataset.segment_cuts(starts, 3000, 1000)
    assert spans[0][0] == 0 and spans[-1][1] == 3000
    assert all(b == c for (_, b), (c, _) in zip(spans, spans[1:]))
    assert all(a in starts for a, _ in spans)
    assert dataset.segment_cuts([0], 500, 1000) == [(0, 500)]


def test_region_unions_neighbours_then_dilates():
    a, b = np.zeros((20, 20), bool), np.zeros((20, 20), bool)
    a[5, 5], b[15, 15] = True, True
    r = masks.region([a, b], dilation=0)
    assert r[5, 5] and r[15, 15] and r.sum() == 2
    grown = masks.region([a], dilation=2)
    assert grown[5, 7] and grown[7, 5] and not grown[8, 8]


def test_alpha_is_one_inside_and_falls_off_outside():
    m = np.zeros((40, 40), bool)
    m[15:25, 15:25] = True
    hard, soft = masks.alpha(m, 0), masks.alpha(m, 6)
    assert hard.shape == (40, 40, 1) and set(np.unique(hard)) == {0.0, 1.0}
    assert np.all(soft[m] == 1.0)
    assert 0 < soft[20, 27, 0] < 1 and soft[0, 0, 0] < 1e-3  # a Gaussian tail, below a uint8 step


def test_telea_changes_only_masked_pixels():
    rng = np.random.default_rng(0)
    frames = rng.integers(0, 255, (2, 32, 32, 3), dtype=np.uint8)
    m = np.zeros((2, 32, 32), bool)
    m[:, 10:20, 10:20] = True
    out = methods.get('telea').load()(frames, m, 0)
    assert out.shape == frames.shape and out.dtype == np.uint8
    assert np.array_equal(out[~m], frames[~m])
    assert not np.array_equal(out[m], frames[m])


def test_settings_take_method_defaults_and_overrides():
    cfg = remove.Settings.for_method('minimax', steps=None, dilation=4)
    assert (cfg.steps, cfg.clip_len, cfg.dilation, cfg.max_frames) == (12, 81, 4, 0)
    assert remove.Settings.for_method('telea').clip_len == 1
    with pytest.raises(ValueError):
        methods.get('nope')
