"""adapter.corrections: each takeover or cuff squeeze of a rollout bag becomes its own episode."""

import json

import pytest

rosbag2_py = pytest.importorskip('rosbag2_py')
from rclpy.serialization import serialize_message  # noqa: E402
from std_msgs.msg import Bool, Float32, String  # noqa: E402

from adapter import corrections  # noqa: E402

S = 1_000_000_000


def test_spans_run_from_the_first_true_sample_to_the_first_false_one():
    times = [0, 1, 2, 3, 4, 5, 6]
    assert corrections.spans(times, [0, 1, 1, 0, 0, 1, 0]) == [(1, 3), (5, 6)]


def test_a_span_still_on_at_the_end_ends_at_the_last_sample():
    assert corrections.spans([0, 1, 2], [0, 1, 1]) == [(1, 2)]
    assert corrections.spans([0, 1, 2], [0, 0, 0]) == []


def write_bag(path, takeover, engaged=None, events=()):
    """A rollout: the takeover flag at 10 Hz from `takeover` (one bool per 0.1 s), a 100 Hz signal, the
    teleop's config published once at the start; with `engaged` (one bool per 0.1 s) the runtime's
    engaged flag on change, and the stick's episode `events` ((s, event) pairs)."""
    writer = rosbag2_py.SequentialWriter()
    writer.open(rosbag2_py.StorageOptions(uri=str(path), storage_id='mcap'), rosbag2_py.ConverterOptions('', ''))
    topics = [
        ('/teleop/left/takeover', 'std_msgs/msg/Bool'),
        ('/signal', 'std_msgs/msg/Float32'),
        ('/teleop/left/config', 'std_msgs/msg/String'),
        ('/policy/left/engaged', 'std_msgs/msg/Bool'),
        ('/policy/left/episode', 'std_msgs/msg/String'),
    ]
    for i, (name, kind) in enumerate(topics):
        writer.create_topic(rosbag2_py.TopicMetadata(i, name, kind, 'cdr'))
    writer.write('/teleop/left/config', serialize_message(String(data='{}')), 1)
    events = sorted((S + int(s * S), e) for s, e in events)
    for k in range(len(takeover) * 10):
        t = S + k * S // 100
        writer.write('/signal', serialize_message(Float32(data=float(k))), t)
        while events and events[0][0] <= t:
            stamp, event = events.pop(0)
            writer.write('/policy/left/episode', serialize_message(String(data=json.dumps({'event': event}))), stamp)
        if k % 10 == 0:
            writer.write('/teleop/left/takeover', serialize_message(Bool(data=bool(takeover[k // 10]))), t)
            i = k // 10
            if engaged is not None and (i == 0 or engaged[i] != engaged[i - 1]):
                writer.write('/policy/left/engaged', serialize_message(Bool(data=bool(engaged[i]))), t)
    writer.close()


def read(path, topic):
    reader = corrections._reader(path, [topic])
    stamps = []
    while reader.has_next():
        stamps.append(reader.read_next()[2])
    return stamps


def test_each_takeover_is_an_episode_cut_to_its_span(tmp_path):
    # From the flags' start (1 s into the bag): 3 s on at 1 s, 0.5 s at 5 s (too short), on to the end at 8 s
    flags = [0] * 10 + [1] * 30 + [0] * 10 + [1] * 5 + [0] * 25 + [1] * 10
    write_bag(tmp_path / 'rollout', flags)
    rows = corrections.extract([tmp_path / 'rollout'], tmp_path / 'session', min_duration=0.8)

    assert [(r['episode'], r['kind'], r['start_s'], r['duration_s']) for r in rows] == [
        ('episode_000', 'takeover', 2.0, 3.0),
        ('episode_001', 'takeover', 9.0, 0.9),  # on until the last sample, 0.9 s later
    ]
    signal = read(tmp_path / 'session' / 'episode_000', '/signal')
    assert signal[0] == 2 * S and signal[-1] == 5 * S
    assert read(tmp_path / 'session' / 'episode_000', '/teleop/left/config') == [2 * S]  # latched, at its start
    assert json.loads((tmp_path / 'session' / 'corrections.json').read_text()) == rows


def test_a_second_extraction_numbers_on(tmp_path):
    write_bag(tmp_path / 'a', [0] * 5 + [1] * 30)
    write_bag(tmp_path / 'b', [0] * 5 + [1] * 30)
    corrections.extract([tmp_path / 'a'], tmp_path / 'session')
    rows = corrections.extract([tmp_path / 'b'], tmp_path / 'session')
    assert [r['episode'] for r in rows] == ['episode_001']
    assert len(json.loads((tmp_path / 'session' / 'corrections.json').read_text())) == 2


def test_whole_rollouts_skip_the_discarded_and_carry_their_label(tmp_path):
    # Three engaged runs (1 to 3 s, 4 to 6 s, 7 to 9 s from the flags' start), a takeover in each;
    # the first saved, the second discarded, the third never labelled. A label counts for the run
    # that ended before it: the save given during the second run is not the second's.
    engaged = [0] * 10 + ([1] * 20 + [0] * 10) * 3
    takeover = [0] * 10 + ([0] * 5 + [1] * 5 + [0] * 20) * 3
    events = [(3.5, 'save'), (4.5, 'save'), (6.5, 'discard')]
    write_bag(tmp_path / 'rollout', takeover, engaged, events)
    rows = corrections.extract([tmp_path / 'rollout'], tmp_path / 'session', min_duration=0.3, whole=True)

    assert [(r['episode'], r['start_s'], r['duration_s'], r['outcome']) for r in rows] == [
        ('episode_000', 2.0, 2.0, 'success'),
        ('episode_001', 8.0, 2.0, 'unlabelled'),
    ]
