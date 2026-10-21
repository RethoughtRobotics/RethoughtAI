import numpy as np
import pytest

from adapter import labels
from adapter.labels import Episode
from inference import policies


class Clock:
    def __init__(self):
        self.t = 100.0

    def __call__(self):
        return self.t


@pytest.fixture
def replay(monkeypatch):
    """A 50-step recording: the left arm moving 1 mm a step along x, the right 2 mm along y."""
    n, q = 50, np.tile([0.0, 0.0, 0.0, 1.0], (50, 1))
    paths = {
        'left': np.c_[0.5 + 0.001 * np.arange(n), np.full(n, 0.3), np.full(n, 0.2)],
        'right': np.c_[np.full(n, 0.5), -0.3 + 0.002 * np.arange(n), np.full(n, 0.2)],
    }

    def load_measured(path, arms, fps, lead):
        assert arms == ('left', 'right') and lead == 0
        return {a: Episode(p, q, p, q, np.zeros(n), np.arange(n) * 40_000_000, fps) for a, p in paths.items()}

    monkeypatch.setattr(labels, 'load_measured', load_measured)  # imported where it is used
    clock = Clock()
    return policies.BimanualReplayPolicy('bag', 25.0, 31, clock=clock), clock


def observation(left, right):
    return {'observation.state': np.r_[left, 0, 0, 0, 1, np.zeros(7), right, 0, 0, 0, 1, np.zeros(7)]}


def test_chunks_follow_the_recording_on_the_clock_and_hold_at_its_end(replay):
    policy, clock = replay
    first = policy(observation([0.5, 0.3, 0.2], [0.5, -0.3, 0.2]))
    assert first.shape == (31, 14)
    assert np.allclose(first[:, 0], 0.5 + 0.001 * np.arange(31))
    clock.t += 0.4  # 10 steps later
    later = policy(observation([0.51, 0.3, 0.2], [0.5, -0.28, 0.2]))
    assert np.allclose(later[0, [0, 8]], [0.51, -0.28])
    clock.t += 10.0
    assert np.allclose(policy(observation([0.55, 0.3, 0.2], [0.5, -0.2, 0.2]))[:, 0], 0.549)


def test_starts_only_near_the_recording_start_and_restarts_after_a_reset(replay):
    policy, clock = replay
    with pytest.raises(ValueError, match='from the recording start'):
        policy(observation([0.5, 0.3, 0.2], [0.5, -0.1, 0.2]))  # right arm 20 cm off
    policy(observation([0.5, 0.3, 0.2], [0.5, -0.3, 0.2]))
    clock.t += 0.4
    policy.reset()
    assert np.isclose(policy(observation([0.5, 0.3, 0.2], [0.5, -0.3, 0.2]))[0, 0], 0.5)


def test_state_is_the_bimanual_layout(replay):
    policy, _ = replay
    q = {'left': np.array([0, 0, 0, 1.0]), 'right': np.array([0, 0, 0, 1.0])}
    state = policy.state({'left': np.ones(3), 'right': 2 * np.ones(3)}, q, None, {a: np.zeros(7) for a in q})
    assert state.shape == (28,) and state[0] == 1 and state[14] == 2
