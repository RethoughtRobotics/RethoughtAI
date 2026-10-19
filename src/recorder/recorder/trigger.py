"""Trigger backends: keyboard or ROS2 digital I/O -> recording events."""

import importlib
import threading
import time

import click


class KeyboardTrigger:
    """Maps configurable keys to start/stop, save, and discard actions."""

    def __init__(self, trig, start_evt, stop_evt, success_evt, discard_evt):
        self._recording = False
        self._lock = threading.Lock()
        self._start_evt, self._stop_evt = start_evt, stop_evt
        self._actions = {
            trig['record']: self._toggle,
            trig['success']: success_evt.set,
            trig['discard']: discard_evt.set,
        }
        click.echo(f'Keys -- record: [{trig["record"]}]  save: [{trig["success"]}]  discard: [{trig["discard"]}]\n')

    def _toggle(self):
        """Flip recording state and fire start or stop event."""
        with self._lock:
            self._recording = not self._recording
            (self._start_evt if self._recording else self._stop_evt).set()

    def idle(self):
        """Back to not recording once an episode is saved or discarded."""
        with self._lock:
            self._recording = False

    def start(self):
        """Spawn a daemon thread that dispatches keystrokes to action callbacks."""

        def _loop():
            while True:
                ch = click.getchar()
                if ch in self._actions:
                    self._actions[ch]()

        threading.Thread(target=_loop, daemon=True).start()


class RecorderTriggerNode:
    """Maps any ROS2 digital IO topics to recording events.

    Message types, topic names, state field, and optional LED feedback are
    all driven by the trigger section of the robot YAML -- no robot-specific
    code here.

    Presses are debounced: one counts only after its button was released for RELEASED s (Baxter's
    navigator buttons bounce on release, and the bounce restarted recording), and none within
    LOCKOUT s of the last one that counted.
    """

    RELEASED = 0.2  # s
    LOCKOUT = 0.5  # s

    def __init__(self, trig, start_evt, stop_evt, success_evt, discard_evt):
        try:
            from rclpy.node import Node
        except ImportError as exc:
            raise RuntimeError('ROS2 not found -- source your workspace first.') from exc

        IOState = _import_msg(trig['msg_type'])
        state_field = trig.get('state_field', 'state')

        self._start_evt, self._stop_evt = start_evt, stop_evt
        self._recording = False
        self._lock = threading.Lock()

        self._node = Node('rai_recorder_trigger')

        # Optional LED feedback -- only wired up if the config has a 'light' section.
        light_cfg = trig.get('light')
        if light_cfg:
            OutputCmd = _import_msg(light_cfg['msg_type'])
            self._light = self._node.create_publisher(OutputCmd, light_cfg['topic'], 10)
            self._light_name = light_cfg['name']
            self._OutputCmd = OutputCmd
        else:
            self._light = self._light_name = self._OutputCmd = None

        self._last_press = -float('inf')

        def sub(topic, fn):
            self._node.create_subscription(IOState, topic, self._press(fn, state_field), 10)

        sub(trig['record'], self._toggle)
        sub(trig['success'], success_evt.set)
        sub(trig['discard'], discard_evt.set)

        # Optional stick gestures -- every topic (a std_msgs/String direction, e.g. libsurvive_ros2
        # /vive/{hand}/thumbstick) at the same direction for `hold` seconds fires its action once.
        pad = trig.get('thumbstick')
        if pad:
            from rclpy.qos import DurabilityPolicy, QoSProfile
            from std_msgs.msg import String

            actions = {
                'start': lambda: self._set_recording(True),
                'stop': lambda: self._set_recording(False),
                'success': success_evt.set,
                'discard': discard_evt.set,
            }
            self._pad_actions = {pad[k]: fn for k, fn in actions.items() if k in pad}
            self._pad_hold = float(pad.get('hold', 2.0))
            self._pad = {topic: ('', 0.0) for topic in pad['topics']}  # topic -> (direction, since)
            self._pad_fired = False
            qos = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)
            for topic in pad['topics']:
                self._node.create_subscription(String, topic, lambda msg, t=topic: self._on_pad(t, msg.data), qos)
            self._node.create_timer(0.05, self._check_pad)

    @property
    def node(self):
        """Underlying rclpy Node -- attach additional publishers here (e.g. BaxterScreen)."""
        return self._node

    def _press(self, fn, state_field):
        """Return a callback that fires fn on a debounced press: a 0 -> non-zero transition after the
        button was released for RELEASED s, and LOCKOUT s after the last press of any button."""
        prev, released = [0], [-float('inf')]  # last state; when it last went 0

        def _cb(msg):
            val, now = getattr(msg, state_field), time.monotonic()
            if prev[0] != 0 and val == 0:
                released[0] = now
            elif (
                prev[0] == 0
                and val != 0
                and now - released[0] >= self.RELEASED
                and now - self._last_press >= self.LOCKOUT
            ):
                self._last_press = now
                fn()
            prev[0] = val

        return _cb

    def _set_light(self, on):
        if self._light is None:
            return
        msg = self._OutputCmd()
        msg.name, msg.value = self._light_name, on
        self._light.publish(msg)

    def _set_recording(self, on):
        """Start or stop recording (no-op if already there), update LED, and fire the event."""
        with self._lock:
            if self._recording == on:
                return
            self._recording = on
            self._set_light(on)
            (self._start_evt if on else self._stop_evt).set()

    def _toggle(self):
        self._set_recording(not self._recording)

    def idle(self):
        """Back to not recording once an episode is saved or discarded: a stray record press while
        deciding must not start the next episode."""
        with self._lock:
            self._recording = False
            self._set_light(False)

    def _on_pad(self, topic, direction):
        if direction != self._pad[topic][0]:
            self._pad[topic] = (direction, time.monotonic())
            self._pad_fired = False

    def _check_pad(self):
        """Fire once when every stick has held the same direction for the hold time."""
        directions = {direction for direction, _ in self._pad.values()}
        held = time.monotonic() - max(since for _, since in self._pad.values())
        if len(directions) == 1 and (direction := directions.pop()) and held >= self._pad_hold and not self._pad_fired:
            self._pad_fired = True
            if direction in self._pad_actions:
                self._pad_actions[direction]()

    def start(self):
        """Spin the ROS2 node in a daemon thread so callbacks can fire."""
        threading.Thread(target=_spin_node, args=(self._node,), daemon=True).start()


def _import_msg(type_str):
    """Import a ROS2 message class from a 'package/MsgName' string."""
    pkg, name = type_str.split('/')
    return getattr(importlib.import_module(f'{pkg}.msg'), name)


def _spin_node(node):
    """Spin a node until context shutdown, then release the executor."""
    from rclpy.executors import SingleThreadedExecutor

    ex = SingleThreadedExecutor()
    ex.add_node(node)
    try:
        ex.spin()
    except Exception:
        pass
    finally:
        ex.shutdown(timeout_sec=0)
