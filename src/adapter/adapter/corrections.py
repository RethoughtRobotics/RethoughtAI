"""Corrections in a policy rollout, each cut into its own episode.

A correction is a span in which a person drove the arm instead of the policy: a VR takeover
(/teleop/{arm}/takeover true, BaxterSDK run_teleop.sh --takeover) or a cuff squeeze (the arm guided
by hand in zero-g); or nudged it while the policy drove on (/teleop/{arm}/nudging, --takeover nudge). Each becomes episode_NNN, every topic of the rollout cut to exactly that span, in
the session layout `adapt lerobot` exports (--labels goal for a takeover: the teleop's goals are the
operator's; a cuff correction has none, --labels measured) and recap browses one episode at a time.
corrections.json lists each one: the rollout it came from, its kind, arm, start and end.

With `whole`, an episode is instead a whole policy rollout (/policy/{arm}/engaged true to false) that
holds at least one correction: the policy's own stretches around the corrections are kept, the frames
a corrector learns to leave alone (export with adapt lerobot --intervention to mark which is which).
Its row lists the corrections inside it, in s from the episode's start, and its outcome: the label
given after it on /policy/{arm}/episode (inference.episodes, the VR stick): a discarded rollout is
skipped, a saved one is 'success', one never labelled 'unlabelled'.
"""

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import rosbag2_py
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message

# kind: (topic, field that is true while the person drives)
SOURCES = {
    'takeover': ('/teleop/{arm}/takeover', 'data'),
    'nudge': ('/teleop/{arm}/nudging', 'data'),  # run_teleop.sh --takeover nudge: the policy drives on
    'cuff': ('/robot/digital_io/{arm}_lower_cuff/state', 'state'),
}
# Published once (transient local), so an episode starting later would lack them: each episode gets
# their last message before its start. Named, because rosbag2_py exposes no durability getter.
LATCHED = ('/tf_static', '/robot_description', '/teleop/{arm}/config')
INDEX = 'corrections.json'
ENGAGED = '/policy/{arm}/engaged'
EPISODE = '/policy/{arm}/episode'
OUTCOMES = {'save': 'success', 'discard': 'discard'}


@dataclass(frozen=True)
class Correction:
    kind: str
    arm: str
    start_ns: int
    end_ns: int

    @property
    def duration(self):
        return (self.end_ns - self.start_ns) / 1e9


def spans(times, on):
    """(start, end) of each run of `on`: from its first true sample to the first false one after it,
    or to the last sample when it is still on there."""
    on = np.asarray(on, bool)
    edges = np.diff(np.concatenate([[False], on, [False]]).astype(int))
    starts, ends = np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)
    return [(int(times[a]), int(times[min(b, len(times) - 1)])) for a, b in zip(starts, ends)]


def _reader(bag, topics=None):
    reader = rosbag2_py.SequentialReader()
    reader.open(rosbag2_py.StorageOptions(uri=str(bag), storage_id='mcap'), rosbag2_py.ConverterOptions('', ''))
    if topics is not None:
        reader.set_filter(rosbag2_py.StorageFilter(topics=list(topics)))
    return reader


def find(bag, arm='left', min_duration=0.0):
    """The bag's corrections of `arm`, in time order, at least `min_duration` s long."""
    reader = _reader(bag)
    types = {t.name: t.type for t in reader.get_all_topics_and_types()}
    topics = {SOURCES[k][0].format(arm=arm): k for k in SOURCES}
    reader = _reader(bag, [t for t in topics if t in types])
    samples = {k: ([], []) for k in SOURCES}
    while reader.has_next():
        topic, raw, stamp = reader.read_next()
        kind = topics[topic]
        times, on = samples[kind]
        times.append(stamp)
        on.append(bool(getattr(deserialize_message(raw, get_message(types[topic])), SOURCES[kind][1])))
    found = [Correction(k, arm, a, b) for k, (times, on) in samples.items() if times for a, b in spans(times, on)]
    return sorted((c for c in found if c.duration >= min_duration), key=lambda c: c.start_ns)


def rollouts(bag, arm='left'):
    """(start, end) ns of each stretch the policy was engaged; one still engaged at the bag's end ends
    there. The topic is published on change only."""
    topic = ENGAGED.format(arm=arm)
    reader = _reader(bag)
    if topic not in {t.name for t in reader.get_all_topics_and_types()}:
        return []
    reader = _reader(bag, [topic])
    times, on = [], []
    while reader.has_next():
        _, raw, stamp = reader.read_next()
        times.append(stamp)
        on.append(bool(deserialize_message(raw, get_message('std_msgs/msg/Bool')).data))
    info = rosbag2_py.Info().read_metadata(str(bag), 'mcap')
    end = info.starting_time.nanoseconds + info.duration.nanoseconds
    return spans([*times, end], [*on, False])


def labels(bag, arm='left'):
    """(stamp ns, 'success' or 'discard') of each save or discard the bag holds, in time order."""
    topic = EPISODE.format(arm=arm)
    reader = _reader(bag)
    if topic not in {t.name for t in reader.get_all_topics_and_types()}:
        return []
    reader = _reader(bag, [topic])
    found = []
    while reader.has_next():
        _, raw, stamp = reader.read_next()
        event = json.loads(deserialize_message(raw, get_message('std_msgs/msg/String')).data).get('event')
        if event in OUTCOMES:
            found.append((stamp, OUTCOMES[event]))
    return found


def outcome(span, next_start, marks):
    """The last label given after the rollout `span` ended and before the next one started (None: no
    next one), or 'unlabelled'."""
    given = [o for t, o in marks if t >= span[1] and (next_start is None or t < next_start)]
    return given[-1] if given else 'unlabelled'


def _episodes(out):
    return sorted(p for p in Path(out).glob('episode_*') if p.is_dir())


def extract(bags, out, arm='left', min_duration=2.0, whole=False):
    """Cut every correction of `bags` (with `whole`, every policy rollout holding one) into
    out/episode_NNN, numbered on from the last episode already there, and add them to
    out/corrections.json. Returns the new rows."""
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    index = out / INDEX
    rows = json.loads(index.read_text()) if index.exists() else []
    n = max((int(p.name.split('_')[1]) + 1 for p in _episodes(out)), default=0)  # after the last, gaps kept
    added = []
    for bag in bags:
        corrections = find(bag, arm, min_duration)
        found, outcomes = corrections, {}
        if whole:
            runs, marks = rollouts(bag, arm), labels(bag, arm)
            for i, (a, b) in enumerate(runs):
                result = outcome((a, b), runs[i + 1][0] if i + 1 < len(runs) else None, marks)
                if result != 'discard' and any(a <= c.start_ns <= b for c in corrections):
                    outcomes[a] = result
            found = [Correction('rollout', arm, a, b) for a, b in runs if a in outcomes]
        if not found:
            continue
        reader = _reader(bag)
        metadata = reader.get_all_topics_and_types()
        latched = {t.format(arm=arm) for t in LATCHED}
        names = [f'episode_{n + i:03d}' for i in range(len(found))]
        n += len(found)
        writers, last = {}, {}  # open episode index: writer; latched topic: (raw, stamp)
        start = None
        while reader.has_next():
            topic, raw, stamp = reader.read_next()
            start = stamp if start is None else start
            for i, c in enumerate(found):
                if c.start_ns <= stamp <= c.end_ns and i not in writers:
                    writers[i] = _writer(out / names[i], metadata)
                    for tp, (data, _) in last.items():  # the latched state at the start
                        writers[i].write(tp, data, c.start_ns)
                if i in writers and stamp > c.end_ns:
                    writers.pop(i).close()
            if topic in latched:
                last[topic] = (raw, stamp)
            for i in writers:
                writers[i].write(topic, raw, stamp)
        for writer in writers.values():  # spans still open at the bag's end
            writer.close()
        for name, c in zip(names, found):
            row = {
                'episode': name,
                'bag': str(Path(bag).resolve()),
                'kind': c.kind,
                'arm': c.arm,
                'start_s': round((c.start_ns - start) / 1e9, 3),
                'end_s': round((c.end_ns - start) / 1e9, 3),
                'duration_s': round(c.duration, 3),
            }
            if whole:
                row['corrections'] = [
                    [round((k.start_ns - c.start_ns) / 1e9, 3), round((min(k.end_ns, c.end_ns) - c.start_ns) / 1e9, 3)]
                    for k in corrections
                    if c.start_ns <= k.start_ns <= c.end_ns
                ]
                row['outcome'] = outcomes[c.start_ns]
            rows.append(row)
            added.append(row)
    index.write_text(json.dumps(rows, indent=1) + '\n')
    return added


def _writer(path, metadata):
    writer = rosbag2_py.SequentialWriter()
    writer.open(rosbag2_py.StorageOptions(uri=str(path), storage_id='mcap'), rosbag2_py.ConverterOptions('', ''))
    for topic in metadata:
        writer.create_topic(topic)
    return writer
