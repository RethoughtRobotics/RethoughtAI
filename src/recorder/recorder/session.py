"""Session management: episode bag lifecycle and the main recording loop."""

import shutil
import subprocess
import threading
import time
from datetime import datetime
from pathlib import Path

import click

from recorder.images import compressed_recording, recording_topic


def make_session_dir(output: str, config_path: Path, name: str | None = None) -> Path:
    """Create the session directory (output/<name>, or a timestamped one) and copy the config into it.
    A named session that already exists is reused: its new episodes continue the numbering."""
    name = name or 'session_' + datetime.now().strftime('%Y_%m_%d_%H_%M_%S')
    session_dir = Path(output) / name
    session_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy(config_path, session_dir / config_path.name)
    return session_dir


class EpisodeBag:
    """One ros2 bag record process per episode.

    SIGTERM triggers a clean shutdown (cache flush + metadata write).
    A 5-second timeout followed by SIGKILL guards against hung processes.
    """

    def __init__(self, episode_dir: Path, topics: list[str]) -> None:
        self._dir = episode_dir
        # Lossless MCAP compression for all messages; colour images can already be JPEG
        # (recording.image_transport). Raw-image compression ratios do not apply to JPEG.
        self._proc = subprocess.Popen([
            'ros2',
            'bag',
            'record',
            '--storage',
            'mcap',
            '--storage-preset-profile',
            'zstd_fast',
            '--output',
            str(episode_dir),
            '--topics',
            *topics,
        ])

    def _terminate(self):
        """SIGTERM with 5 s timeout, then SIGKILL."""
        self._proc.terminate()
        try:
            self._proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            self._proc.kill()
            self._proc.wait()

    def stop(self):
        self._terminate()

    def discard(self):
        """Stop recording (if still running) and delete the bag directory."""
        self._terminate()
        if self._dir.exists():
            shutil.rmtree(self._dir)

    def frames(self, image_topic: str | None) -> int | None:
        """Messages recorded on image_topic, from the stopped bag's metadata; None if unknown."""
        import yaml

        try:
            info = yaml.safe_load((self._dir / 'metadata.yaml').read_text())['rosbag2_bagfile_information']
        except (OSError, KeyError, TypeError, yaml.YAMLError):
            return None
        return next(
            (
                t['message_count']
                for t in info.get('topics_with_message_count', [])
                if t['topic_metadata']['name'] == image_topic
            ),
            None,
        )


def run_session(session_dir: Path, config: dict, topics: list[str], viz: bool) -> None:
    """Run an interactive session until Ctrl+C, recording one bag per episode.

    The active bag is always terminated on exit, even mid-episode, to
    prevent orphaned ros2 bag record processes from leaking memory.
    """
    trig = config['trigger']
    episode = _next_episode(session_dir)
    saved = 0
    images = config['topics'].get('images') or [{}]
    image_topic = images[0].get('topic')  # its message count is the episode's frame count
    if image_topic:
        image_topic = recording_topic(config, image_topic)
    start_evt, stop_evt, success_evt, discard_evt = (threading.Event() for _ in range(4))

    ros_started, trigger, screen = _setup(
        trig, config, start_evt, stop_evt, success_evt, discard_evt, viz, session_dir.name
    )
    trigger.start()

    click.echo(f'Session: {session_dir}')
    click.echo('Press record to begin. Ctrl+C to end session.\n')
    if screen:
        screen.episode = episode
        screen.idle()
    _set_title(episode, 'idle')

    bag = None
    try:
        while True:
            # Clear stale signals from the previous episode before waiting.
            for e in (stop_evt, success_evt, discard_evt):
                e.clear()
            start_evt.wait()
            start_evt.clear()

            episode = _next_episode(session_dir)  # rescan: episodes may have been removed or renumbered
            bag = EpisodeBag(session_dir / f'episode_{episode:03d}', topics)
            started = time.monotonic()
            if screen:
                screen.episode = episode
                screen.recording()
            _set_title(episode, 'recording')
            click.echo(f'[episode {episode:03d}] Recording. Press record again to stop.')

            stop_evt.wait()
            # Only a save/discard made after stopping counts, not one pressed while recording.
            success_evt.clear()
            discard_evt.clear()
            length = time.monotonic() - started
            bag.stop()  # the episode ends at stop, not when it is saved or discarded
            if screen:
                screen.decide(length, bag.frames(image_topic))
            _set_title(episode, 'save or discard?')
            click.echo(f'[episode {episode:03d}] Stopped: save or discard.')

            while not success_evt.is_set() and not discard_evt.is_set():
                time.sleep(0.05)

            done = episode
            if success_evt.is_set():
                if screen:
                    screen.saving()
                _set_title(done, 'saving')
                bag = None
                saved += 1
                click.echo(f'[episode {done:03d}] Saved.\n')
            else:
                if screen:
                    screen.discarding()
                _set_title(done, 'discarding')
                bag.discard()
                bag = None
                click.echo(f'[episode {done:03d}] Discarded.\n')
            time.sleep(1.0)  # long enough to see the choice on the screen
            # Only a record press after the decision starts the next episode: drop one made while
            # deciding (or a button bounce), which would otherwise start recording at once.
            trigger.idle()
            start_evt.clear()
            episode = _next_episode(session_dir)
            if screen:
                screen.episode = episode
                screen.idle()
            _set_title(episode, 'idle')

    except KeyboardInterrupt:
        click.echo(f'\nSession ended: {saved} episode(s) saved, {episode} in {session_dir.name}.')
    finally:
        if bag is not None:
            bag.discard()
        if ros_started:
            import rclpy

            rclpy.shutdown()


def _setup(trig, config, start_evt, stop_evt, success_evt, discard_evt, viz, session=None):
    """Init trigger backend, ROS2 nodes, and optional viz. Returns (ros_started, trigger, screen)."""
    from recorder.trigger import KeyboardTrigger, RecorderTriggerNode, _spin_node

    compress = compressed_recording(config) and bool(config['topics'].get('images'))
    needs_ros = trig['type'] == 'digital_io' or viz or compress
    if needs_ros:
        try:
            import rclpy
        except ImportError:
            raise RuntimeError('rclpy not found - source your ROS2 workspace first.')
        rclpy.init()

    if compress:
        from recorder.images import make_compression_node

        node = make_compression_node(config)
        threading.Thread(target=_spin_node, args=(node,), daemon=True).start()
        click.echo('Recording JPEG images on <camera topic>/compressed (raw live topics unchanged).')

    def start_viz():
        from recorder.viz import init_rerun, make_viz_node

        init_rerun()
        threading.Thread(target=_spin_node, args=(make_viz_node(config),), daemon=True).start()

    if trig['type'] == 'keyboard':
        if viz:
            start_viz()
        return needs_ros, KeyboardTrigger(trig, start_evt, stop_evt, success_evt, discard_evt), None

    trigger = RecorderTriggerNode(trig, start_evt, stop_evt, success_evt, discard_evt)
    screen = None
    if trig.get('screen'):  # "session": drawn screens; true: the image screens in assets/
        from recorder.screen import BaxterScreen, SessionScreen

        screen = (SessionScreen if trig['screen'] == 'session' else BaxterScreen)(trigger.node, session)
    if viz:
        start_viz()
    return True, trigger, screen


def _next_episode(session_dir: Path) -> int:
    """One past the highest existing episode_NNN, so a reused session resumes after its last episode."""
    return 1 + max(
        (int(d.name.split('_')[-1]) for d in session_dir.glob('episode_*') if d.name.split('_')[-1].isdigit()),
        default=-1,
    )


def _set_title(episode: int, state: str) -> None:
    """Update the terminal window title with current episode and state."""
    click.echo(f'\033]0;RAI: Ep {episode:03d} | {state}\007', nl=False)
