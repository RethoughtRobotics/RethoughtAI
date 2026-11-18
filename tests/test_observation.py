"""inference.observation: one clock for the policy's inputs."""

import numpy as np

from inference.observation import Sampled, Stamps


def test_own_clock_stamp_is_used_as_is():
    stamps = Stamps()
    assert stamps.time('third_eye', stamp=100.0, receive=100.035) == 100.0
    assert not stamps.foreign('third_eye')


def test_foreign_clock_gap_is_the_smallest_seen():
    stamps = Stamps()
    gap = 439_642_498.37  # Baxter's clock behind the PC's, as in the 2026-11-05 bags
    t1 = stamps.time('joints', stamp=10.0, receive=10.0 + gap + 0.012)
    assert t1 == 10.0 + gap + 0.012  # one sample: the gap includes that sample's transport
    t2 = stamps.time('joints', stamp=10.01, receive=10.01 + gap + 0.005)
    t3 = stamps.time('joints', stamp=10.02, receive=10.02 + gap + 0.020)
    assert np.isclose(t2, 10.01 + gap + 0.005)
    assert np.isclose(t3, 10.02 + gap + 0.005)  # a slow sample keeps the minimum: its age is the jitter
    assert stamps.foreign('joints')


def test_gap_window_forgets_old_minima():
    stamps = Stamps(window=2)
    gap = 1000.0
    stamps.time('x', 0.0, gap + 0.001)
    stamps.time('x', 1.0, 1.0 + gap + 0.010)
    t = stamps.time('x', 2.0, 2.0 + gap + 0.010)
    assert np.isclose(t, 2.0 + gap + 0.010)


def test_sampled_nearest():
    s = Sampled(maxlen=3)
    assert s.at(0.0) is None
    for t in (1.0, 2.0, 3.0, 4.0):
        s.add(t, t * 10)
    assert len(s) == 3  # the oldest dropped
    assert s.at(2.4) == 20
    assert s.at(2.6) == 30
    assert s.at(9.0) == 40
