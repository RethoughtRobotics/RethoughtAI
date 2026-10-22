"""Atomic, pose-only two-arm execution using Runtime's merge and teleop mapping.

Only the coordinator owns ROS callbacks and a timer. Arm runtimes never create
services, subscriptions, timers, or gripper publishers. A cuff correction is a
release of the whole run, not Runtime's independent single-arm re-anchoring.
"""

import numpy as np
from baxter_core_msgs.msg import DigitalIOState
from geometry_msgs.msg import PoseStamped
from moveit_msgs.msg import RobotTrajectory
from nav_msgs.msg import Path
from rclpy.qos import DurabilityPolicy, QoSProfile
from scipy.spatial.transform import Rotation, Slerp
from sensor_msgs.msg import Joy
from std_msgs.msg import Bool, Empty
from std_srvs.srv import SetBool

from .runtime import SETTLE, TCP_STALE, Chunk, Runtime


class _SilentPublisher:
    def publish(self, msg):
        pass


class _ArmNode:
    """Expose only target output (the pose stream and the timed path) and the coordinator's sampled tick clock."""

    def __init__(self, owner):
        self.owner = owner

    def create_subscription(self, *args, **kwargs):
        pass

    def create_service(self, *args, **kwargs):
        pass

    def create_timer(self, *args, **kwargs):
        pass

    def create_publisher(self, msg_type, topic, qos):
        if msg_type in (PoseStamped, Path):
            return self.owner.node.create_publisher(msg_type, topic, qos)
        return _SilentPublisher()

    def get_logger(self):
        return self.owner.node.get_logger()

    def get_clock(self):
        return self

    def now(self):
        sample = self.owner._clock_sample
        return sample if sample is not None else self.owner.node.get_clock().now()


class _ArmRuntime(Runtime):
    def __init__(self, owner, cfg, merge):
        self.owner = owner
        self.last_goal = None
        super().__init__(_ArmNode(owner), cfg, merge=merge)

    def _combine(self, t):
        # Coverage is an AND, even if an arm's state accidentally loses a chunk.
        return super()._combine(t) if self.owner._covered(t) else None

    def _publish(self, position, quaternion, trigger, engaged=False):
        if engaged:
            self.last_goal = (np.array(position, dtype=float), Rotation.from_quat(quaternion))
        # Never write a gripper, even if inherited state acquires a trigger.
        super()._publish(position, quaternion, None, engaged=engaged)


class BimanualRuntime:
    """One engage/deadman gate for ['left_tcp', 'right_tcp'] pose chunks.

    Observation stamps use joint_trajectory.header, as in Runtime. An optional
    nonzero multi-DOF header stamp must agree. Joint points are rejected: this
    runtime has no gripper command path. All poses (including TCP input) must
    have finite coordinates and unit quaternions, within 1e-5.
    """

    def __init__(self, node, cfg, merge=None):
        if cfg.get('arm') != 'both' or set(cfg['arms']) != {'left', 'right'}:
            raise ValueError('bimanual runtime requires arm: both and left/right arms')
        if any(not np.isfinite(float(cfg[key])) or float(cfg[key]) <= 0 for key in ('fps', 'publish_rate')):
            raise ValueError('fps and publish_rate must be finite and positive')
        self.node = node
        self._clock_sample = None
        self.engaged = self.requested = self.held = self.fresh = False
        self.settled, self.accept_after = -np.inf, np.inf
        deadman = cfg['deadman']
        self.deadman_id = deadman['frame_id']
        self.deadman_button = int(deadman['button'])
        if self.deadman_button < 0:
            raise ValueError('deadman button must be nonnegative')
        self.alive = -np.inf if deadman.get('alive') else np.inf
        self.cuffs = {'left': False, 'right': False}
        self.arms = {}
        for name in self.cuffs:
            topics = {key: cfg['arms'][name].get(key) for key in ('tcp', 'target', 'path', 'trigger', 'home', 'cuff')}
            arm_cfg = {**cfg, **topics, 'engaged': None, 'episodes': None}
            self.arms[name] = _ArmRuntime(self, arm_cfg, merge)
        self.dt = self.arms['left'].dt
        if any(arm.horizon is not None and arm.horizon < 2 for arm in self.arms.values()):
            raise ValueError('bimanual chunk horizon must contain at least two points')
        latched = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.engaged_pub = node.create_publisher(Bool, cfg['engaged'], latched)
        for name, arm_cfg in cfg['arms'].items():
            node.create_subscription(PoseStamped, arm_cfg['tcp'], lambda msg, name=name: self._on_tcp(name, msg), 10)
            if arm_cfg.get('home'):
                node.create_subscription(Empty, arm_cfg['home'], self._on_home, 10)
            if arm_cfg.get('cuff'):
                node.create_subscription(
                    DigitalIOState, arm_cfg['cuff'], lambda msg, name=name: self._on_cuff(name, msg), 10
                )
        node.create_subscription(Joy, deadman['topic'], self._on_joy, 10)
        if deadman.get('alive'):
            node.create_subscription(PoseStamped, deadman['alive'], lambda msg: setattr(self, 'alive', self._now()), 10)
        node.create_subscription(RobotTrajectory, cfg['chunk'], self._on_chunk, 10)
        node.create_service(SetBool, cfg['engage'], self._on_engage)
        node.create_timer(1.0 / float(cfg['publish_rate']), self._tick)
        self._announce()

    def _now(self):
        sample = self._clock_sample
        return (sample if sample is not None else self.node.get_clock().now()).nanoseconds * 1e-9

    @staticmethod
    def _pose(position, quaternion):
        p, q = np.array(position, dtype=float), np.array(quaternion, dtype=float)
        if not np.isfinite(p).all() or not np.isfinite(q).all():
            raise ValueError('nonfinite pose')
        if abs(np.linalg.norm(q) - 1.0) > 1e-5:
            raise ValueError('quaternion must be normalized')
        return p, q

    @staticmethod
    def _seconds(stamp):
        if stamp.sec < 0 or not 0 <= stamp.nanosec < 1_000_000_000:
            raise ValueError('invalid timestamp')
        # Match _now's conversion exactly at the accept_after boundary.
        return (stamp.sec * 1_000_000_000 + stamp.nanosec) * 1e-9

    def _on_tcp(self, name, msg):
        p, q = msg.pose.position, msg.pose.orientation
        arm = self.arms[name]
        try:
            self._pose([p.x, p.y, p.z], [q.x, q.y, q.z, q.w])
        except ValueError:
            arm.tcp = None  # Invalid input must not refresh the safety gate.
            return
        arm._on_tcp(msg)

    def _tcp_fresh(self, now):
        return all(arm.tcp is not None and 0 <= now - arm.tcp[2] < TCP_STALE for arm in self.arms.values())

    def _wanted(self, now):
        return self.requested or (self.held and now - self.alive < TCP_STALE)

    def _on_joy(self, msg):
        if msg.header.frame_id == self.deadman_id and len(msg.buttons) > self.deadman_button:
            self.held = bool(msg.buttons[self.deadman_button])
            if not self._wanted(self._now()):
                self._release(self._now())

    def _on_engage(self, request, response):
        self.requested = bool(request.data)
        if not self._wanted(self._now()):
            self._release(self._now())
        response.success = True
        response.message = 'engage requested' if self.requested else 'released'
        return response

    def _on_home(self, msg):
        # Unlike Runtime._on_home, this never sends a gripper-open command.
        self._release(self._now(), reset_settle=True)

    def _on_cuff(self, name, msg):
        previous = any(self.cuffs.values())
        self.cuffs[name] = bool(msg.state)
        if previous or any(self.cuffs.values()):
            self._release(self._now(), reset_settle=True)

    def _announce(self):
        self.engaged_pub.publish(Bool(data=self.engaged))

    def _release(self, now, reset_settle=False):
        was_engaged = self.engaged
        self.engaged, self.accept_after = False, np.inf
        if was_engaged or reset_settle:
            self.settled = now
        for arm in self.arms.values():
            # _released preserves the last target, never follows a sagging TCP.
            arm._released(arm.tcp is not None and 0 <= now - arm.tcp[2] < TCP_STALE)
            arm.chunks, arm.segment, arm.entry, arm.approach = [], None, None, None
            arm.accept_after, arm.ref, arm.last_goal = np.inf, None, None
            arm.correcting = arm.squeezed = False
            arm.settled = self.settled
        if was_engaged:
            self._announce()

    def _on_chunk(self, msg):
        now = self._now()
        if not self.engaged or not self._wanted(now) or not self._tcp_fresh(now) or any(self.cuffs.values()):
            return
        trajectory = msg.multi_dof_joint_trajectory
        if (
            list(trajectory.joint_names) != ['left_tcp', 'right_tcp']
            or len(trajectory.points) < 2
            or msg.joint_trajectory.points
        ):
            return
        try:
            t0 = self._seconds(msg.joint_trajectory.header.stamp)
            other_stamp = self._seconds(trajectory.header.stamp)
            if other_stamp != 0 and other_stamp != t0:
                return
            if t0 < self.accept_after or t0 > now:
                return
            times = t0 + np.array([self._seconds(p.time_from_start) for p in trajectory.points])
            if not np.isfinite(times).all() or np.any(np.diff(times) <= 0):
                return
            poses = {name: ([], []) for name in self.arms}
            for point in trajectory.points:
                if len(point.transforms) != 2:
                    return
                for name, transform in zip(self.arms, point.transforms):
                    p, q = transform.translation, transform.rotation
                    p, q = self._pose([p.x, p.y, p.z], [q.x, q.y, q.z, q.w])
                    poses[name][0].append(p)
                    poses[name][1].append(q)
            chunks = {}
            for name, arm in self.arms.items():
                n = arm.horizon or len(times)
                if times[:n][-1] < now:
                    return
                p, q = poses[name]
                chunks[name] = Chunk(times[:n].copy(), np.array(p[:n]), Rotation.from_quat(q[:n]), None, now)
        except (ValueError, TypeError, OverflowError):
            return
        # No arm mutates until the entire message, including beyond horizon, validates.
        for name, arm in self.arms.items():
            arm.chunks.append(chunks[name])

    def _covered(self, t):
        return all(
            any(c.t[0] <= t <= c.end and c.arrival <= t + 1e-9 for c in arm.chunks) for arm in self.arms.values()
        )

    def _tick(self):
        self._clock_sample = self.node.get_clock().now()
        try:
            self._tick_at(self._now())
        finally:
            self._clock_sample = None

    def _tick_at(self, now):
        fresh = self._tcp_fresh(now)
        if fresh and not self.fresh:
            self.settled = max(self.settled, now)
        self.fresh = fresh
        if not fresh or not self._wanted(now) or any(self.cuffs.values()):
            self._release(now)
            return
        if not self.engaged:
            if any(arm.frozen is None for arm in self.arms.values()) or now - self.settled < SETTLE:
                self._release(now)
                return
            self.engaged, self.accept_after = True, now
            for arm in self.arms.values():
                p, q, _ = arm.tcp
                R = Rotation.from_quat(q)
                arm.engaged, arm.accept_after, arm.settled = True, now, self.settled
                arm.ref = (p, R, arm.frozen[0], Rotation.from_quat(arm.frozen[1]))
                arm.hold, arm.last_goal = (p, R), (p, R)
            self._announce()
        if not self._covered(now):
            # Stop an already interpolating segment too, for both on this tick.
            for arm in self.arms.values():
                arm.hold = arm.last_goal
                arm.segment, arm.entry = None, None
        first = self.arms['left']
        if first.segment is None or now - first.segment[0] >= self.dt:
            for arm in self.arms.values():
                arm._policy_step(now)
        for arm in self.arms.values():
            start, (p0, R0, trigger), (p1, R1) = arm.segment
            a = min((now - start) * arm.fps, 1.0)
            R = Slerp([0.0, 1.0], Rotation.concatenate([R0, R1]))(a)
            arm._publish(p0 + a * (p1 - p0), R.as_quat(), None, engaged=True)
