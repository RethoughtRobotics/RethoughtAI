"""Policy runtime: drives a robot's teleop the way its VR controller does, from action chunks.

The teleop is pointed at this node instead of the controller (bringbackbaxter: run_teleop.sh
--policy), so policy actions go through the same servo, IK, gripper filter and home button as a
human's. It publishes what libsurvive_ros2 publishes for a hand:
  target   PoseStamped in the base frame; unchanged while released, which the teleop reads as a
           released clutch
  trigger  Float32, 0 open .. 1 closed
The teleop turns a target change into a goal relative to where it clutched (goal = TCP at clutch +
target change, vr_axes_policy.yaml's identity map). So the released target never moves, not even
to follow a sagging or homed arm (that would be a stroke), and engaged goals are mapped through
that rule from the TCP and target at engage.
Engaged: the deadman held (or the `engage` service) and a fresh TCP. Published on `engaged`.
B (`home`, the teleop's home button) clears the run and opens the gripper; while still engaged it
resumes fresh after homing.

A chunk is a trajectory of TCP targets timed from its observation: from a policy called here (a
replay) or the `chunk` topic (policy_node). Each policy step merges the chunks covering it:
  ensemble  ACT temporal ensembling, exp(-coeff * i), i = 0 the oldest
  latest    the newest chunk (RTC chunks already continue the last)
  ramp      each new chunk crossfades in over `ramp` steps, finishing within the overlap
After a hold, chunks crossfade in over `entry_ramp` s: a policy's first targets lead the TCP.
With a `start` pose, the arm moves there first, then `on_start` runs.
"""

from dataclasses import dataclass

import numpy as np
from geometry_msgs.msg import PoseStamped
from moveit_msgs.msg import RobotTrajectory
from scipy.spatial.transform import Rotation, Slerp
from rclpy.qos import DurabilityPolicy, QoSProfile
from sensor_msgs.msg import Joy
from std_msgs.msg import Bool, Empty, Float32
from std_srvs.srv import SetBool

from adapter.labels import apply_actions

TCP_STALE = 0.2  # s, as the teleop's stale target
APPROACH_SPEED = (0.05, np.radians(15.0))  # m/s, rad/s toward a policy's start pose
START_TOLERANCE = (0.005, np.radians(2.0))  # m, rad
# The teleop reads an unchanged pose as a released clutch. A human holding still while clutched never
# repeats one (tracker noise), so an engaged target that would repeat alternates by this much (m).
ENGAGED_JITTER = 1e-6
# Engage only once the target and a fresh TCP have held this long, longer than the teleop's
# release_after (0.2 s), so the teleop has re-referenced at the current TCP.
SETTLE = 0.3  # s
MERGES = ("ensemble", "latest", "ramp")
ENTRY_RAMP = 1.0  # s


def continuous(q):
    """Quaternions with consecutive ones in the same hemisphere, for interpolation."""
    q = np.array(q, dtype=float)
    flips = np.cumsum(np.r_[0, np.einsum("ij,ij->i", q[1:], q[:-1]) < 0]) % 2 == 1
    q[flips] *= -1
    return q


@dataclass
class Chunk:
    """TCP targets and triggers at times t (s, ROS clock)."""
    t: np.ndarray
    positions: np.ndarray
    rotations: Rotation
    triggers: np.ndarray | None
    arrival: float  # executed from here on

    def __post_init__(self):
        self.quats = continuous(self.rotations.as_quat())

    @property
    def end(self):
        return self.t[-1]

    def sample(self, t):
        """(position, quaternion, trigger) at t. nlerp: steps are close, and slerp is too slow here."""
        i = int(np.clip(np.searchsorted(self.t, t, side="right") - 1, 0, len(self.t) - 2))
        a = float(np.clip((t - self.t[i]) / (self.t[i + 1] - self.t[i]), 0.0, 1.0))
        q = (1 - a) * self.quats[i] + a * self.quats[i + 1]
        trigger = None if self.triggers is None else (1 - a) * self.triggers[i] + a * self.triggers[i + 1]
        return (1 - a) * self.positions[i] + a * self.positions[i + 1], q / np.linalg.norm(q), trigger

    def at(self, t):
        """(position, Rotation, trigger) at time t."""
        p, q, trigger = self.sample(t)
        return p, Rotation.from_quat(q), trigger


def blend(x, y, w):
    """(position, Rotation, trigger) w of the way from x to y."""
    R = x[1] * Rotation.from_rotvec(w * (x[1].inv() * y[1]).as_rotvec())
    trigger = y[2] if x[2] is None else x[2] if y[2] is None else (1 - w) * x[2] + w * y[2]
    return (1 - w) * x[0] + w * y[0], R, trigger


def check_merge(merge, chunk_size, every=1):
    """Reject merge settings that leave gaps or jumps; return a warning, if any."""
    kind = merge.get("type", "ensemble")
    if kind not in MERGES:
        raise ValueError(f"merge type {kind!r}: one of {', '.join(MERGES)}")
    if every >= chunk_size:
        raise ValueError(f"every ({every}) must be below the chunk size ({chunk_size}), or the chunks leave gaps")
    if kind == "ramp":
        ramp = int(merge["ramp"])
        if not 0 < ramp < chunk_size:
            raise ValueError(f"ramp ({ramp}) must be in 1 .. chunk size - 1 ({chunk_size - 1})")
        if ramp > chunk_size - every:
            return (f"ramp ({ramp}) is longer than a chunk's overlap with the next ({chunk_size - every}): "
                    "crossfades are cut short")
    return None


class Runtime:
    def __init__(self, node, cfg, policy=None, merge=None, start=None, on_start=None):
        """Chunks from `policy` (e.g. a replay) or, without one, from cfg["chunk"]."""
        self.node, self.policy = node, policy
        self.fps = float(cfg["fps"])
        self.dt = 1.0 / self.fps
        self.chunk_size = int(cfg["chunk_size"])
        self.n_action_steps = int(cfg.get("n_action_steps") or self.chunk_size)
        self.merge = merge or {"type": "ensemble" if cfg.get("temporal_ensemble_coeff") is not None else "latest",
                               "coeff": cfg.get("temporal_ensemble_coeff")}
        deadman = cfg["deadman"]
        self.deadman_id, self.deadman_button = deadman["frame_id"], int(deadman["button"])
        self.held = False
        self.requested = False  # engage service
        self.alive = -np.inf if deadman.get("alive") else np.inf  # last pose of the deadman's controller
        self.tcp = None  # (position, quaternion, receive time)
        self.frozen = None  # (position, quaternion) target published while released
        self.settled = -np.inf  # since when the teleop can have re-referenced (release, B, TCP back)
        self.fresh = False  # TCP fresh last tick
        self.ref = None  # (TCP p, TCP Rotation, target p, target Rotation) at engage: goal -> target
        self.trigger = None
        self.engaged = False
        self.done = False
        self.step = 0
        self.chunks = []  # Chunk, oldest first
        self.accept_after = np.inf  # older chunks predate this run
        self.segment = None  # (start time, (p, R, trigger) at this step, (p, R) at the next)
        self.hold = None  # (p, R) when no chunk covers a step
        self.entry = None  # (time, (p, R, trigger)) chunks resumed from a hold
        self.entry_ramp = float(cfg.get("entry_ramp", ENTRY_RAMP))
        self.start = start if start is not None else getattr(policy, "start", None)  # pose to reach before step 0
        self.on_start = on_start
        self.approach = None  # (time, position, Rotation) the approach target was last at
        self.published = None  # last (position, quaternion) sent
        self.jitter = ENGAGED_JITTER

        node.create_subscription(PoseStamped, cfg["tcp"], self._on_tcp, 10)
        node.create_subscription(Joy, deadman["topic"], self._on_joy, 10)
        if deadman.get("alive"):
            node.create_subscription(PoseStamped, deadman["alive"], lambda m: setattr(self, "alive", self._now()), 10)
        if policy is None:
            node.create_subscription(RobotTrajectory, cfg["chunk"], self._on_chunk, 10)
        node.create_service(SetBool, cfg["engage"], self._on_engage)
        if cfg.get("home"):
            node.create_subscription(Empty, cfg["home"], self._on_home, 10)
        latched = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.engaged_pub = node.create_publisher(Bool, cfg["engaged"], latched) if cfg.get("engaged") else None
        self.target_pub = node.create_publisher(PoseStamped, cfg["target"], 10)
        self.trigger_pub = node.create_publisher(Float32, cfg["trigger"], 10)
        node.create_timer(1.0 / float(cfg["publish_rate"]), self._tick)
        self._announce()

    def _now(self):
        return self.node.get_clock().now().nanoseconds * 1e-9

    def _on_tcp(self, msg):
        p, o = msg.pose.position, msg.pose.orientation
        self.tcp = (np.array([p.x, p.y, p.z]), np.array([o.x, o.y, o.z, o.w]), self._now())

    def _on_joy(self, msg):
        if msg.header.frame_id == self.deadman_id and len(msg.buttons) > self.deadman_button:
            self.held = bool(msg.buttons[self.deadman_button])

    def _on_engage(self, request, response):
        self.requested = bool(request.data)
        response.success, response.message = True, "engage requested" if self.requested else "released"
        return response

    def _on_home(self, msg):
        """B: the teleop homes the arm. Release once, so nothing carries over: the policy server
        resets, and re-engaging (./start still on, or the clutch held) starts from fresh chunks."""
        self.chunks, self.segment, self.entry, self.approach = [], None, None, None
        self.trigger = 0.0  # open: the teleop opens it to home, and follows this trigger after
        self.trigger_pub.publish(Float32(data=self.trigger))
        self.settled = self._now()  # re-engage only after homing
        if self.engaged:
            self._released(self.tcp is not None)
        if self.settled - getattr(self, "_home_logged", -np.inf) > 1.0:  # B repeats while held
            self._home_logged = self.settled
            self._log("B: cleared, gripper open; resumes fresh after homing while ./start is on or the clutch is held")

    def _on_chunk(self, msg):
        t0 = msg.joint_trajectory.header.stamp
        t0 = t0.sec + t0.nanosec * 1e-9
        points = msg.multi_dof_joint_trajectory.points
        if not self.engaged or self.start is not None or t0 < self.accept_after or len(points) < 2:
            return
        t = t0 + np.array([p.time_from_start.sec + p.time_from_start.nanosec * 1e-9 for p in points])
        tf = [p.transforms[0] for p in points]
        positions = np.array([[x.translation.x, x.translation.y, x.translation.z] for x in tf])
        rotations = Rotation.from_quat([[x.rotation.x, x.rotation.y, x.rotation.z, x.rotation.w] for x in tf])
        joints = msg.joint_trajectory.points
        triggers = np.array([p.positions[0] for p in joints]) if len(joints) == len(points) else None
        self.chunks.append(Chunk(t, positions, rotations, triggers, self._now()))

    def _log(self, text):
        self.node.get_logger().info(text)

    def _tick(self):
        now = self._now()
        tcp_fresh = self.tcp is not None and now - self.tcp[2] < TCP_STALE
        deadman = self.held and now - self.alive < TCP_STALE
        if tcp_fresh and not self.fresh:  # teleop back (started, or done homing)
            self.settled = max(self.settled, now)
        self.fresh = tcp_fresh
        engage = (deadman or self.requested) and tcp_fresh and not self.done
        if engage and not self.engaged and (self.frozen is None or now - self.settled < SETTLE):
            engage = False
        if not engage:
            self._released(tcp_fresh)
            return
        if not self.engaged:
            self._log(f"Engaged at step {self.step}")
            self.engaged, self.chunks, self.segment, self.approach = True, [], None, None
            self._announce()
            p, R = self.tcp[0], Rotation.from_quat(self.tcp[1])
            self.ref = (p, R, self.frozen[0], Rotation.from_quat(self.frozen[1]))
            self.hold, self.accept_after = (p, R), now
        if self.start is not None:
            self._approach(now)
            return
        if self.segment is None or now - self.segment[0] >= self.dt:
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
            self.start, self.hold, self.accept_after = None, (p, R), now
            if self.on_start is not None:
                self.on_start()

    def _released(self, tcp_fresh):
        """Hold: keep publishing the last target unchanged (at first, the TCP)."""
        if self.engaged:
            self._log(f"Released at step {self.step}" + (": episode finished, B sends the arm home" if self.done else ""))
            self.engaged, self.frozen, self.settled = False, self.published, self._now()
            self._announce()
        if self.frozen is None and tcp_fresh:
            self.frozen, self.settled = (self.tcp[0], self.tcp[1]), self._now()
        if self.frozen is not None:
            self._publish(*self.frozen, self.trigger)

    def _policy_step(self, now):
        t = now if self.segment is None else self.segment[0] + self.dt  # this step's time
        if self.policy is not None:
            if hasattr(self.policy, "__len__") and self.step >= len(self.policy):
                self.done = True
                self._released(True)
                return False
            if (not self.chunks or self.merge["type"] == "ensemble"
                    or t - self.chunks[-1].t[0] >= (self.n_action_steps - 0.5) * self.dt):
                self._call_policy(t)
        self.chunks = [c for c in self.chunks if c.end >= t]
        if self.merge["type"] != "ensemble":  # drop superseded chunks
            settle = 0.0 if self.merge["type"] == "latest" else (int(self.merge["ramp"]) + 1) * self.dt
            done = [i for i, c in enumerate(self.chunks) if c.arrival + settle <= t]
            self.chunks = self.chunks[done[-1]:] if done else self.chunks
        merged, merged_next = self._combine(t), self._combine(t + self.dt)
        if merged is None:
            self.entry, current = None, (*self.hold, self.trigger)
        else:
            if self.entry is None:
                self.entry = (t, (*self.hold, self.trigger))
            current = self._entered(t, merged)
        following = current if merged is None or merged_next is None else self._entered(t + self.dt, merged_next)
        self.hold = current[:2]
        self.segment = (t, current, following[:2])
        self.step += 1
        return True

    def _entered(self, t, merged):
        """`merged`, crossfaded in from the hold."""
        w = (t - self.entry[0]) / self.entry_ramp if self.entry_ramp > 0 else 1.0
        return merged if w >= 1.0 else blend(self.entry[1], merged, max(w, 0.0))

    def _call_policy(self, t):
        p_obs, q_obs, _ = self.tcp
        actions = self.policy({"step": self.step, "tcp_position": p_obs, "tcp_orientation": q_obs})
        positions, rotations = apply_actions(actions, p_obs, q_obs)
        self.chunks.append(Chunk(t + self.dt * np.arange(len(actions)), positions, rotations, actions[:, 6], t))

    def _combine(self, t):
        """(position, Rotation, trigger) at t from the chunks covering it, or None."""
        covering = [c for c in self.chunks if c.t[0] <= t <= c.end and c.arrival <= t + 1e-9]
        if not covering:
            return None
        kind = self.merge["type"]
        if kind == "latest":
            return covering[-1].at(t)
        if kind == "ensemble":
            samples = [c.sample(t) for c in covering]
            w = np.exp(-float(self.merge["coeff"]) * np.arange(len(samples)))
            w /= w.sum()
            trigger = None if samples[0][2] is None else float(w @ [s[2] for s in samples])
            return (w @ np.array([s[0] for s in samples]),
                    Rotation.from_quat([s[1] for s in samples]).mean(weights=w), trigger)
        samples = [c.at(t) for c in covering]
        ramp = int(self.merge["ramp"])
        merged = samples[0]
        for older, chunk, sample in zip(covering, covering[1:], samples[1:]):
            overlap = (older.end - chunk.arrival) * self.fps  # the ramp must finish within it
            span = max(1.0, min(ramp + 1.0, overlap))
            merged = blend(merged, sample, min(1.0, ((t - chunk.arrival) * self.fps + 1.0) / span))
        return merged

    def _announce(self):
        if self.engaged_pub is not None:
            self.engaged_pub.publish(Bool(data=self.engaged))

    def _publish(self, position, quaternion, trigger, engaged=False):
        """Publish a target: engaged, `position, quaternion` is the TCP goal, mapped to the target
        that makes the teleop servo there; released, the target itself."""
        position = np.asarray(position, dtype=float)
        if engaged:
            p0, R0, fp, fR = self.ref
            position = position - p0 + fp
            quaternion = (Rotation.from_quat(quaternion) * R0.inv() * fR).as_quat()
            if self.published is not None and np.array_equal(position, self.published[0]):
                self.jitter = -self.jitter
                position = position + [self.jitter, 0.0, 0.0]
        self.published = (position, np.asarray(quaternion, dtype=float))
        msg = PoseStamped()
        msg.header.stamp, msg.header.frame_id = self.node.get_clock().now().to_msg(), "base"
        p, o = msg.pose.position, msg.pose.orientation
        p.x, p.y, p.z = (float(x) for x in position)
        o.x, o.y, o.z, o.w = (float(x) for x in quaternion)
        self.target_pub.publish(msg)
        if trigger is not None:
            self.trigger = float(trigger)
            self.trigger_pub.publish(Float32(data=self.trigger))
