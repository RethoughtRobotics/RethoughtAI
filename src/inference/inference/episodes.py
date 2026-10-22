"""Policy episodes from one VR thumbstick, labelled for the corrections pipeline.

An episode is one engaged stretch of the runtime (/policy/{arm}/engaged true to false), however it
was engaged. The stick drives it as scripts/start and scripts/stop do, a direction held for `hold` s
firing once, in the recorder's layout (configs/baxter.yaml trigger.thumbstick):
  north  start: request engage
  south  stop: release the request
  east   save the last stopped episode (a success)
  west   discard it
Corrections are counted from the runtime (a takeover or cuff squeeze) and, with `nudging`, from the
teleop's nudges (run_teleop.sh --takeover nudge), during which the policy keeps driving.
Each event goes out on `topic` as JSON ({"episode": N, "event": "start" | "stop" | "save" |
"discard"}), so the bag carries it: adapt corrections --whole skips discarded rollouts and marks
saved ones. The terminal and Baxter's head display (the recorder's session screens) show the state.
"""

import json
import time

from rclpy.qos import DurabilityPolicy, QoSProfile
from std_msgs.msg import String

ACTIONS = ('start', 'stop', 'save', 'discard')
CHECK = 0.05  # s between checks of the held direction and the runtime's state


class Episodes:
    def __init__(self, runtime, cfg, screen=None):
        """`runtime` engages through its `requested` flag and reports `engaged` and `correcting`;
        `cfg` is the inference config's `episodes` section; `screen` a recorder.screen screen."""
        self.runtime, self.screen = runtime, screen
        self.hold = float(cfg.get('hold', 1.0))
        self.actions = {cfg.get(a, d): a for a, d in zip(ACTIONS, ('north', 'south', 'east', 'west'))}
        self.episode = 0
        self.running = False
        self.pending = None  # a stopped episode awaiting save or discard
        self.started = None  # monotonic time the running episode started
        self.corrections = 0
        self.direction, self.since, self.fired = '', 0.0, False
        self.correcting = False
        self.nudging = False
        node = runtime.node
        if cfg.get('nudging'):
            from std_msgs.msg import Bool

            node.create_subscription(Bool, cfg['nudging'], lambda m: setattr(self, 'nudging', bool(m.data)), 10)
        self.pub = node.create_publisher(String, cfg['topic'], 10)
        stick = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)  # published on change
        node.create_subscription(String, cfg['stick'], lambda m: self.on_stick(m.data), stick)
        node.create_timer(CHECK, self.check)
        if self.screen is not None:
            self.screen.idle()
        self._log(f'Episodes: right stick held {self.hold:g} s, {self._layout()}')

    def on_stick(self, direction, now=None):
        if direction != self.direction:
            self.direction, self.since, self.fired = direction, time.monotonic() if now is None else now, False
            action = self.actions.get(direction)
            if action is not None and self._allowed(action):
                self._log(f'{direction} held: {action} in {self.hold:g} s')

    def check(self, now=None):
        """Fire a held direction once; follow the runtime's engaged and correcting state."""
        now = time.monotonic() if now is None else now
        action = self.actions.get(self.direction)
        if action is not None and not self.fired and now - self.since >= self.hold:
            self.fired = True
            if self._allowed(action):
                getattr(self, f'_{action}')()
        engaged = self.runtime.engaged
        if engaged and not self.running:
            self._began(now)
        elif not engaged and self.running:
            self._ended(now)
        correcting = self.runtime.correcting or self.nudging
        if self.running and correcting != self.correcting:
            self.correcting = correcting
            if self.correcting:
                self.corrections += 1
            self._log(f'  correction {self.corrections} {"started" if self.correcting else "ended"}')

    def _allowed(self, action):
        if action == 'start':
            return not self.runtime.requested
        if action == 'stop':
            return self.runtime.requested or self.running
        return self.pending is not None and not self.running

    def _start(self):
        self.runtime.requested = True

    def _stop(self):
        self.runtime.requested = False

    def _save(self):
        self._label('save', 'SAVED', 'saving')

    def _discard(self):
        self._label('discard', 'DISCARDED', 'discarding')

    def _label(self, event, word, show):
        self._publish(self.pending, event)
        self._log(f'{"✓" if event == "save" else "✗"} EPISODE {self.pending} {word}')
        if self.screen is not None:
            getattr(self.screen, show)()
        self.pending = None

    def _began(self, now):
        if self.pending is not None:
            self._log(f'  episode {self.pending} left unlabelled')
        self.episode += 1
        self.running, self.pending, self.started, self.corrections = True, None, now, 0
        self.correcting = False
        self._publish(self.episode, 'start')
        self._log(f'▶ EPISODE {self.episode} STARTED')
        if self.screen is not None:
            self.screen.episode = self.episode
            self.screen.recording()

    def _ended(self, now):
        self.running, self.pending = False, self.episode
        length = now - self.started
        self._publish(self.episode, 'stop')
        self._log(
            f'■ EPISODE {self.episode} STOPPED ({length:.1f} s, {self.corrections} corrections): '
            f'{self._key("save")} save, {self._key("discard")} discard'
        )
        if self.screen is not None:
            self.screen.decide(length)

    def _publish(self, episode, event):
        self.pub.publish(String(data=json.dumps({'episode': episode, 'event': event})))

    def _key(self, action):
        return next(d for d, a in self.actions.items() if a == action)

    def _layout(self):
        return ', '.join(f'{d} {a}' for d, a in self.actions.items())

    def _log(self, text):
        self.runtime.node.get_logger().info(text)
