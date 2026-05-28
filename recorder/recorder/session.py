import shutil
import subprocess
import threading
import time
from datetime import datetime
from pathlib import Path

import click


def make_session_dir(output: str, config_path: Path) -> Path:
    ts = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    session_dir = Path(output) / f"session_{ts}"
    session_dir.mkdir(parents=True)
    shutil.copy(config_path, session_dir / config_path.name)
    return session_dir


class EpisodeBag:
    def __init__(self, episode_dir: Path, topics: list[str], max_cache_duration: int):
        self._dir = episode_dir
        self._proc = subprocess.Popen([
            "ros2", "bag", "record",
            "--snapshot-mode",
            "--max-cache-duration", str(max_cache_duration),
            "--repeat-all-transient-local",
            "--output", str(episode_dir),
            "--topics", *topics,
        ])

    def commit(self):
        subprocess.run([
            "ros2", "service", "call",
            "/rosbag2_recorder/snapshot",
            "rosbag2_interfaces/srv/Snapshot", "{}",
        ], check=True, capture_output=True)
        time.sleep(0.5)
        self._stop()

    def discard(self):
        self._stop()
        if self._dir.exists():
            shutil.rmtree(self._dir)

    def _stop(self):
        self._proc.terminate()
        self._proc.wait()


def run_session(session_dir: Path, config: dict, topics: list[str], viz: bool) -> None:
    max_cache = config["recording"].get("max_cache_duration", 0)
    trig = config["trigger"]
    episode = 0

    start_evt = threading.Event()
    success_evt = threading.Event()
    discard_evt = threading.Event()

    _start_ros(trig, config, start_evt, success_evt, discard_evt, viz)

    if trig["type"] == "keyboard":
        _start_keyboard(trig, start_evt, success_evt, discard_evt)

    click.echo(f"Session: {session_dir}")
    click.echo("Press start to begin an episode. Ctrl+C to end session.\n")

    try:
        while True:
            start_evt.wait()
            start_evt.clear()
            success_evt.clear()
            discard_evt.clear()

            episode_dir = session_dir / f"episode_{episode:03d}"
            click.echo(f"[episode {episode:03d}] Buffering — commit or discard when done.")
            bag = EpisodeBag(episode_dir, topics, max_cache)

            while not success_evt.is_set() and not discard_evt.is_set():
                time.sleep(0.05)

            if success_evt.is_set():
                bag.commit()
                click.echo(f"[episode {episode:03d}] Committed.\n")
                episode += 1
            else:
                bag.discard()
                click.echo(f"[episode {episode:03d}] Discarded.\n")

    except KeyboardInterrupt:
        click.echo(f"\nSession ended — {episode} episode(s) recorded.")


def _start_ros(trig: dict, config: dict, start_evt, success_evt, discard_evt, viz: bool) -> None:
    needs_ros = trig["type"] == "digital_io" or viz
    if not needs_ros:
        return

    try:
        import rclpy
        from rclpy.executors import MultiThreadedExecutor
    except ImportError:
        raise RuntimeError("rclpy not found — source your ROS2 workspace first.")

    rclpy.init()
    nodes = []

    if trig["type"] == "digital_io":
        nodes.append(_make_trigger_node(trig, start_evt, success_evt, discard_evt))

    if viz:
        from recorder.viz import init_rerun, make_viz_node
        init_rerun()
        nodes.append(make_viz_node(config))

    executor = MultiThreadedExecutor()
    for node in nodes:
        executor.add_node(node)

    threading.Thread(target=executor.spin, daemon=True).start()


def _make_trigger_node(trig: dict, start_evt, success_evt, discard_evt):
    try:
        from baxter_core_msgs.msg import DigitalIOState
    except ImportError:
        raise RuntimeError("baxter_core_msgs not found — source your Baxter workspace.")

    from rclpy.node import Node

    node = Node("rai_recorder_trigger")

    def make_cb(evt):
        def cb(msg):
            if msg.state != 0:
                evt.set()
        return cb

    node.create_subscription(DigitalIOState, trig["start"],   make_cb(start_evt),   10)
    node.create_subscription(DigitalIOState, trig["success"], make_cb(success_evt), 10)
    node.create_subscription(DigitalIOState, trig["discard"], make_cb(discard_evt), 10)
    return node


def _start_keyboard(trig: dict, start_evt, success_evt, discard_evt) -> None:
    key_map = {
        trig["start"]: start_evt,
        trig["success"]: success_evt,
        trig["discard"]: discard_evt,
    }
    click.echo(f"Keys — start: [{trig['start']}]  commit: [{trig['success']}]  discard: [{trig['discard']}]\n")

    def loop():
        while True:
            ch = click.getchar()
            if ch in key_map:
                key_map[ch].set()

    threading.Thread(target=loop, daemon=True).start()
