"""Policy runtime: drives a robot's teleop the way its VR controller does, from action chunks.

The teleop is left as is and pointed at this node instead of the controller (bringbackbaxter:
run_teleop.sh --policy), so a policy's actions go through the same servo, IK, gripper filter and
home button as a human's. It publishes what libsurvive_ros2 publishes for a hand:
  target   PoseStamped, the desired TCP pose in the base frame, streamed at publish_rate while
           engaged and exactly frozen while not, which the teleop reads as a released clutch (the
           arm holds). It freezes at the measured TCP, so the next engage starts without an offset.
  trigger  Float32, 0 open .. 1 closed.
Engaged = the deadman button held, its controller tracked, and a fresh TCP from the teleop (which
stops publishing it while the arm moves home).

Each policy step (1/fps, counted only while engaged) the policy maps the observation to an action
chunk (adapter.labels: targets relative to the TCP at observation time, plus the trigger), which
is anchored on the TCP measured then. As in ACT: with temporal_ensemble_coeff set, a chunk is
predicted every step and a step's target is the exp(-coeff * i) weighted mean of every chunk's
prediction for it (i = 0 the oldest); without it, n_action_steps of each chunk run before the
next prediction. Between steps the target moves linearly to the next step's.

A policy with a `start` pose (a replay) is first approached: the target moves there from the TCP
at APPROACH_SPEED, and step 0 runs once the TCP is within START_TOLERANCE of it.
"""

import time

import numpy as np
from geometry_msgs.msg import PoseStamped
from scipy.spatial.transform import Rotation, Slerp
from sensor_msgs.msg import Joy
from std_msgs.msg import Float32

from adapter.labels import apply_actions

TCP_STALE = 0.2  # s, as the teleop's stale target
REFREEZE = (0.002, np.radians(1.0))  # re-freeze on the TCP when it moved this far while released (m, rad)
APPROACH_SPEED = (0.05, np.radians(15.0))  # m/s, rad/s toward a policy's start pose
START_TOLERANCE = (0.005, np.radians(2.0))  # m, rad
# The teleop reads an unchanged pose as a released clutch. A human holding still while clutched never
# repeats one (tracker noise), so an engaged target that would repeat alternates by this much (m).
ENGAGED_JITTER = 1e-6


class Runtime:
    def __init__(self, node, cfg, policy):
        self.node, self.policy = node, policy
        self.fps = float(cfg["fps"])
        self.chunk_size = int(cfg["chunk_size"])
        self.n_action_steps = int(cfg.get("n_action_steps") or self.chunk_size)
        self.coeff = cfg.get("temporal_ensemble_coeff")
        deadman = cfg["deadman"]
        self.deadman_id, self.deadman_button = deadman["frame_id"], int(deadman["button"])
        self.held = False
        self.alive = -np.inf if deadman.get("alive") else np.inf  # last pose of the deadman's controller
        self.tcp = None  # (position, quaternion, receive time)
        self.frozen = None  # (position, quaternion) published while released
        self.trigger = None
        self.engaged = False
        self.done = False
        self.step = 0
        self.chunks = []  # (start step, positions (K, 3), Rotations, triggers (K,)), oldest first
        self.segment = None  # (start time, (p, R, trigger) at this step, (p, R) at the next)
        self.start = getattr(policy, "start", None)  # (position, quaternion) to reach before step 0
        self.approach = None  # (time, position, Rotation) the approach target was last at
        self.published = None  # last position sent
        self.jitter = ENGAGED_JITTER

        node.create_subscription(PoseStamped, cfg["tcp"], self._on_tcp, 10)
        node.create_subscription(Joy, deadman["topic"], self._on_joy, 10)
        if deadman.get("alive"):
            node.create_subscription(PoseStamped, deadman["alive"], lambda m: setattr(self, "alive", time.monotonic()), 10)
        self.target_pub = node.create_publisher(PoseStamped, cfg["target"], 10)
        self.trigger_pub = node.create_publisher(Float32, cfg["trigger"], 10)
        node.create_timer(1.0 / float(cfg["publish_rate"]), self._tick)

    def _on_tcp(self, msg):
        p, o = msg.pose.position, msg.pose.orientation
        self.tcp = (np.array([p.x, p.y, p.z]), np.array([o.x, o.y, o.z, o.w]), time.monotonic())

    def _on_joy(self, msg):
        if msg.header.frame_id == self.deadman_id and len(msg.buttons) > self.deadman_button:
            self.held = bool(msg.buttons[self.deadman_button])

    def _log(self, text):
        self.node.get_logger().info(text)

    def _tick(self):
        now = time.monotonic()
        tcp_fresh = self.tcp is not None and now - self.tcp[2] < TCP_STALE
        engage = self.held and tcp_fresh and now - self.alive < TCP_STALE and not self.done
        if not engage:
            self._released(tcp_fresh)
            return
        if not self.engaged:
            self._log(f"Engaged at step {self.step}")
            self.engaged, self.chunks, self.segment, self.approach = True, [], None, None
        if self.start is not None:
            self._approach(now)
            return
        if self.segment is None or now - self.segment[0] >= 1.0 / self.fps:
            if not self._policy_step(now):
                return
        start, (p0, R0, trigger), (p1, R1) = self.segment
        a = min((now - start) * self.fps, 1.0)
        rotation = Slerp([0.0, 1.0], Rotation.concatenate([R0, R1]))(a)
        self._publish(p0 + a * (p1 - p0), rotation.as_quat(), trigger, engaged=True)

    def _approach(self, now):
        """Move the target from the TCP toward the start pose; clear `start` once the TCP is there."""
        p_goal, R_goal = self.start[0], Rotation.from_quat(self.start[1])
        p_tcp, q_tcp, _ = self.tcp
        if self.approach is None:
            self.approach = (now, p_tcp, Rotation.from_quat(q_tcp))
            self._log(f"Moving to the start pose, {1000 * np.linalg.norm(p_goal - p_tcp):.0f} mm away")
        t, p, R = self.approach
        dt = now - t
        step = p_goal - p
        p = p + step * min(1.0, APPROACH_SPEED[0] * dt / max(np.linalg.norm(step), 1e-9))
        turn = (R_goal * R.inv()).as_rotvec()
        R = Rotation.from_rotvec(turn * min(1.0, APPROACH_SPEED[1] * dt / max(np.linalg.norm(turn), 1e-9))) * R
        self.approach = (now, p, R)
        self._publish(p, R.as_quat(), None, engaged=True)
        if (np.linalg.norm(p_tcp - p_goal) < START_TOLERANCE[0]
                and (Rotation.from_quat(q_tcp).inv() * R_goal).magnitude() < START_TOLERANCE[1]):
            self._log("At the start pose: running the policy")
            self.start = None

    def _released(self, tcp_fresh):
        """Hold: freeze the target at the TCP (again only if the arm moved, e.g. homed)."""
        if self.engaged:
            self._log(f"Released at step {self.step}" + (": episode finished, B sends the arm home" if self.done else ""))
            self.engaged, self.frozen = False, None
        if tcp_fresh:
            p, q, _ = self.tcp
            if self.frozen is None or (
                np.linalg.norm(p - self.frozen[0]) > REFREEZE[0]
                or (Rotation.from_quat(q).inv() * Rotation.from_quat(self.frozen[1])).magnitude() > REFREEZE[1]
            ):
                self.frozen = (p, q)
        if self.frozen is not None:
            self._publish(*self.frozen, self.trigger)

    def _policy_step(self, now):
        if hasattr(self.policy, "__len__") and self.step >= len(self.policy):
            self.done = True
            self._released(True)
            return False
        if not self.chunks or self.coeff is not None or self.step - self.chunks[-1][0] >= self.n_action_steps:
            p_obs, q_obs, _ = self.tcp
            actions = self.policy({"step": self.step, "tcp_position": p_obs, "tcp_orientation": q_obs})
            positions, rotations = apply_actions(actions, p_obs, q_obs)
            self.chunks.append((self.step, positions, rotations, actions[:, 6]))
            self.chunks = [c for c in self.chunks if c[0] + len(c[1]) > self.step]
            if self.coeff is None:
                self.chunks = self.chunks[-1:]
        current = self._combine(self.step)
        following = self._combine(self.step + 1) or current
        start = now if self.segment is None else self.segment[0] + 1.0 / self.fps
        self.segment = (start, current, following[:2])
        self.step += 1
        return True

    def _combine(self, step):
        """(position, Rotation, trigger) for `step` from the chunks covering it, or None."""
        covering = [(c[1][step - c[0]], c[2][step - c[0]], c[3][step - c[0]])
                    for c in self.chunks if c[0] <= step < c[0] + len(c[1])]
        if not covering:
            return None
        if self.coeff is None:
            return covering[-1]
        w = np.exp(-float(self.coeff) * np.arange(len(covering)))
        w /= w.sum()
        positions = np.array([c[0] for c in covering])
        rotations = Rotation.concatenate([c[1] for c in covering])
        triggers = np.array([c[2] for c in covering])
        return w @ positions, rotations.mean(weights=w), float(w @ triggers)

    def _publish(self, position, quaternion, trigger, engaged=False):
        position = np.asarray(position, dtype=float)
        if engaged and self.published is not None and np.array_equal(position, self.published):
            self.jitter = -self.jitter
            position = position + [self.jitter, 0.0, 0.0]
        self.published = position
        msg = PoseStamped()
        msg.header.stamp, msg.header.frame_id = self.node.get_clock().now().to_msg(), "base"
        p, o = msg.pose.position, msg.pose.orientation
        p.x, p.y, p.z = (float(x) for x in position)
        o.x, o.y, o.z, o.w = (float(x) for x in quaternion)
        self.target_pub.publish(msg)
        if trigger is not None:
            self.trigger = float(trigger)
            self.trigger_pub.publish(Float32(data=self.trigger))
