"""inference.episodes: the stick held starts, stops and labels policy episodes, once per hold."""

import json
import types

import pytest

pytest.importorskip('rclpy')
from inference.episodes import Episodes  # noqa: E402


class Node:
    def __init__(self):
        self.sent, self.lines = [], []

    def create_publisher(self, kind, topic, qos):
        return types.SimpleNamespace(publish=lambda msg: self.sent.append(json.loads(msg.data)))

    def create_subscription(self, *args):
        pass

    def create_timer(self, *args):
        pass

    def get_logger(self):
        return types.SimpleNamespace(info=self.lines.append)


@pytest.fixture
def run():
    runtime = types.SimpleNamespace(node=Node(), requested=False, engaged=False, correcting=False)
    episodes = Episodes(runtime, {'stick': '/stick', 'topic': '/episode', 'hold': 1.0})
    return runtime, episodes


def hold(episodes, direction, at, held=1.0):
    episodes.on_stick(direction, now=at)
    episodes.check(now=at + held)
    episodes.on_stick('', now=at + held)


def test_a_short_push_does_nothing(run):
    runtime, episodes = run
    hold(episodes, 'north', 0.0, held=0.9)
    assert not runtime.requested


def test_start_stop_save_in_the_stick_layout(run):
    runtime, episodes = run
    hold(episodes, 'north', 0.0)
    assert runtime.requested
    runtime.engaged = True
    episodes.check(now=2.0)
    runtime.correcting = True
    episodes.check(now=3.0)
    runtime.correcting = False
    episodes.check(now=4.0)
    hold(episodes, 'south', 5.0)
    assert not runtime.requested
    runtime.engaged = False
    episodes.check(now=7.0)
    hold(episodes, 'east', 8.0)
    hold(episodes, 'east', 10.0)  # nothing left to save
    assert runtime.node.sent == [
        {'episode': 1, 'event': 'start'},
        {'episode': 1, 'event': 'stop'},
        {'episode': 1, 'event': 'save'},
    ]
    assert any('STOPPED (5.0 s, 1 corrections)' in line for line in runtime.node.lines)


def test_labels_wait_for_the_episode_to_stop(run):
    runtime, episodes = run
    hold(episodes, 'north', 0.0)
    runtime.engaged = True
    episodes.check(now=2.0)
    hold(episodes, 'west', 3.0)  # still running: not discarded
    runtime.requested = runtime.engaged = False
    episodes.check(now=5.0)
    hold(episodes, 'west', 6.0)
    assert [m['event'] for m in runtime.node.sent] == ['start', 'stop', 'discard']


def test_a_hold_fires_once(run):
    runtime, episodes = run
    episodes.on_stick('north', now=0.0)
    episodes.check(now=1.0)
    runtime.requested = False  # e.g. scripts/stop while the stick is still held
    episodes.check(now=2.0)
    assert not runtime.requested


def test_a_nudge_counts_as_a_correction_while_the_policy_drives_on(run):
    runtime, episodes = run
    hold(episodes, 'north', 0.0)
    runtime.engaged = True
    episodes.check(now=2.0)
    episodes.nudging = True  # the teleop's --takeover nudge: the runtime itself is not correcting
    episodes.check(now=3.0)
    episodes.nudging = False
    episodes.check(now=4.0)
    runtime.requested = runtime.engaged = False
    episodes.check(now=5.0)
    assert any('STOPPED (3.0 s, 1 corrections)' in line for line in runtime.node.lines)
