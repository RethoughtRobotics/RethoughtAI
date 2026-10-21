"""Nonblocking, generation-gated inference for absolute, pose-only two-arm policies.

Receive times (ROS clock), not potentially unsynchronized message headers, gate
freshness and TCP skew. Quaternion hemispheres are tracked on *every* valid TCP
input, independently for each arm, starting with the first incoming sign.
Only the worker calls/reset the model; callbacks never wait for inference.
"""

import os
import threading
import time

import numpy as np
from baxter_core_msgs.msg import DigitalIOState
from builtin_interfaces.msg import Duration
from geometry_msgs.msg import PoseStamped, Transform
from moveit_msgs.msg import RobotTrajectory
from rclpy.qos import DurabilityPolicy, QoSProfile
from rclpy.time import Time
from sensor_msgs.msg import Image, JointState
from std_msgs.msg import Bool, Empty
from trajectory_msgs.msg import MultiDOFJointTrajectoryPoint

from adapter.lerobot_export import JOINTS, image_rgb

STALE = 0.5
MAX_TCP_SKEW = 0.1
ARMS = ('left', 'right')


def device_lost(exc):
    """True for errors after which the GPU context is unusable (unlike a bad frame or shape):
    CUDA launch failures, illegal memory accesses, ECC errors and the like."""
    text = str(exc)
    return 'CUDA error' in text or 'cudaError' in text or type(exc).__name__ == 'AcceleratorError'


class BimanualPolicyNode:
    """Interface: ``(node, cfg, images, policy, every=1)``.

    ``cfg['max_tcp_skew']`` overrides the default 0.1 s receive-time skew.
    Home or cuff press locally releases the gate until a shared engaged=True
    arrives again. Every control transition invalidates in-flight work, even a
    release/re-engage pair entirely inside one model call.
    """

    def __init__(self, node, cfg, images, policy, every=1):
        if set(cfg['arms']) != set(ARMS):
            raise ValueError('bimanual policy requires left/right arms')
        if not getattr(policy, 'bimanual', False) or policy.rtc is not None:
            raise ValueError('bimanual absolute policy with rtc=None required')
        self.fps, self.every = float(cfg['fps']), int(every)
        self.max_tcp_skew = float(cfg.get('max_tcp_skew', MAX_TCP_SKEW))
        if not np.isfinite(self.fps) or self.fps <= 0:
            raise ValueError('fps must be finite and positive')
        if self.every < 1 or self.every != every:
            raise ValueError('every must be a positive integer')
        if not np.isfinite(self.max_tcp_skew) or self.max_tcp_skew < 0:
            raise ValueError('max_tcp_skew must be finite and nonnegative')
        if not cfg.get('engaged'):
            raise ValueError('a shared engaged topic is required')
        self.cameras = {c['key']: c['topic'] for c in images if c['key'] in policy.features}
        missing = [k for k in policy.features if k.startswith('observation.images.') and k not in self.cameras]
        if missing:
            raise ValueError(f'no camera topic configured for {", ".join(missing)}')
        self.node, self.policy = node, policy
        self.dt = 1.0 / self.fps
        self.joint_names = {arm: [f'{arm}_{j}' for j in JOINTS] for arm in ARMS}
        self.latest = {}
        self.quaternions = {arm: None for arm in ARMS}
        self.cuffs = {arm: False for arm in ARMS}
        self.engaged = False
        self.generation = 0
        self._reset_generation = -1
        self.lock = threading.Lock()
        self._wake = threading.Event()
        self._stop = threading.Event()
        for key, topic in self.cameras.items():
            node.create_subscription(Image, topic, lambda msg, key=key: self._on_image(key, msg), 2)
        for arm in ARMS:
            arm_cfg = cfg['arms'][arm]
            node.create_subscription(PoseStamped, arm_cfg['tcp'], lambda msg, arm=arm: self._on_tcp(arm, msg), 10)
            if arm_cfg.get('cuff'):
                node.create_subscription(
                    DigitalIOState, arm_cfg['cuff'], lambda msg, arm=arm: self._on_cuff(arm, msg), 10
                )
        # A shared home topic must not register duplicate callbacks.
        homes = {cfg['arms'][arm].get('home') for arm in ARMS}
        homes.add(cfg.get('home'))
        for topic in sorted(homes - {None, ''}):
            node.create_subscription(Empty, topic, self._on_home, 10)
        node.create_subscription(JointState, cfg['joint_states'], self._on_joints, 10)
        latched = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        node.create_subscription(Bool, cfg['engaged'], self._on_engaged, latched)
        self.chunk_pub = node.create_publisher(RobotTrajectory, cfg['chunk'], 10)
        self.worker = threading.Thread(target=self._run, daemon=True, name='bimanual-policy')
        self.worker.start()

    def _now(self):
        return self.node.get_clock().now().nanoseconds * 1e-9

    def _invalidate_locked(self):
        self.generation += 1
        self._wake.set()

    def _on_engaged(self, msg):
        with self.lock:
            value = bool(msg.data) and not any(self.cuffs.values())
            if value != self.engaged:
                self.engaged = value
                self._invalidate_locked()

    def _on_home(self, msg):
        with self.lock:
            self.engaged = False
            self._invalidate_locked()

    def _on_cuff(self, arm, msg):
        with self.lock:
            value = bool(msg.state)
            if value != self.cuffs[arm]:
                self.cuffs[arm] = value
                if value:
                    self.engaged = False
                self._invalidate_locked()

    def _store(self, name, value):
        with self.lock:
            self.latest[name] = (value, self._now())

    def _on_image(self, key, msg):
        try:
            image = np.array(image_rgb(msg), copy=True)
        except (ValueError, TypeError, IndexError):
            with self.lock:
                self.latest.pop(key, None)
            return
        self._store(key, image)

    def _on_tcp(self, arm, msg):
        p, q = msg.pose.position, msg.pose.orientation
        p = np.array([p.x, p.y, p.z], dtype=float)
        q = np.array([q.x, q.y, q.z, q.w], dtype=float)
        norm = np.linalg.norm(q)
        with self.lock:
            key = f'{arm}_tcp'
            if not np.isfinite(p).all() or not np.isfinite(q).all() or not np.isfinite(norm) or abs(norm - 1.0) > 1e-5:
                self.latest.pop(key, None)
                return
            q /= norm
            previous = self.quaternions[arm]
            if previous is not None and np.dot(previous, q) < 0:
                q = -q
            self.quaternions[arm] = q.copy()
            self.latest[key] = ((p, q), self._now())

    def _on_joints(self, msg):
        try:
            names = list(msg.name)
            required = [name for arm in ARMS for name in self.joint_names[arm]]
            if not set(required) & set(names):
                return  # another publisher on the topic (each gripper's finger joint at 20 Hz)
            if any(names.count(name) != 1 for name in required):
                raise ValueError('all fourteen joints required exactly once')
            joints = {
                arm: np.array([msg.position[names.index(name)] for name in self.joint_names[arm]], dtype=float)
                for arm in ARMS
            }
            if not all(np.isfinite(value).all() for value in joints.values()):
                raise ValueError('nonfinite joints')
        except (ValueError, TypeError, IndexError):
            with self.lock:
                self.latest.pop('joints', None)
            return
        self._store('joints', joints)

    def _ready_locked(self, now):
        names = [*self.cameras, 'left_tcp', 'right_tcp', 'joints']
        return (
            self.engaged
            and not self._stop.is_set()
            and not any(self.cuffs.values())
            and all(name in self.latest and 0 <= now - self.latest[name][1] <= STALE for name in names)
            and abs(self.latest['left_tcp'][1] - self.latest['right_tcp'][1]) <= self.max_tcp_skew
        )

    def _observation(self):
        """Return (observation time, observation, generation), or None.

        Copies isolate the model from callback updates and model-side mutation.
        The wrapper orders state as [p3, q4, joints7] left, then right.
        """
        with self.lock:
            t = self._now()
            if not self._ready_locked(t):
                return None
            generation = self.generation
            tcp_p = {arm: self.latest[f'{arm}_tcp'][0][0].copy() for arm in ARMS}
            tcp_q = {arm: self.latest[f'{arm}_tcp'][0][1].copy() for arm in ARMS}
            joints = {arm: self.latest['joints'][0][arm].copy() for arm in ARMS}
            obs = {key: self.latest[key][0].copy() for key in self.cameras}
        state = np.asarray(self.policy.state(tcp_p, tcp_q, None, joints), dtype=np.float32)
        if state.shape != (28,) or not np.isfinite(state).all():
            return None
        obs['observation.state'] = state
        with self.lock:
            if generation != self.generation or not self._ready_locked(self._now()):
                return None
        return t, obs, generation

    def _step(self):
        """One worker iteration; never called from a ROS callback."""
        with self.lock:
            generation = self.generation
        if self._reset_generation != generation:
            self.policy.reset()
            self._reset_generation = generation
        observed = self._observation()
        if observed is None:
            return False
        t, obs, generation = observed
        with self.lock:
            if (
                generation != self.generation
                or self._reset_generation != generation
                or not self._ready_locked(self._now())
            ):
                return False
        actions = self.policy(obs)  # RAW absolute [left p3,q4, right p3,q4]. No RTC.
        return self._publish(t, actions, generation)

    def _run(self):
        next_start = 0.0
        while not self._stop.is_set():
            self._wake.wait(max(0.0, next_start - time.monotonic()))
            self._wake.clear()
            if self._stop.is_set():
                return
            started = time.monotonic()
            try:
                self._step()
            except Exception as exc:
                if device_lost(exc):
                    # The CUDA context is gone (e.g. "unspecified launch failure", NVIDIA Xid 45): every
                    # later call fails too. Exit so policy.launch.py respawns a fresh server; the
                    # runtime holds the arm meanwhile and resumes on the new server's chunks.
                    self.node.get_logger().fatal(f'GPU lost, exiting for a respawn: {exc}')
                    os._exit(75)
                # A failed model must not kill the worker or preserve its history.
                self._reset_generation = -1
                self.node.get_logger().warning(f'bimanual inference rejected: {exc}')
            next_start = started + self.every * self.dt

    def close(self):
        """Request worker exit without joining/blocking a ROS callback."""
        with self.lock:
            self.engaged = False
            self._stop.set()
            self._invalidate_locked()

    def _publish(self, t, actions, generation):
        """Validate the ENTIRE chunk before a single atomic, gated publication."""
        try:
            actions = np.array(actions, dtype=float, copy=True)
            if (
                actions.ndim != 2
                or actions.shape[1] != 14
                or len(actions) < 2
                or not np.isfinite(actions).all()
                or not np.isfinite(t)
                or t < 0
            ):
                return False
            for start in (3, 10):
                q = actions[:, start : start + 4]
                scale = np.max(np.abs(q), axis=1)
                if np.any(scale == 0):
                    return False
                # Scaling avoids overflow/underflow for finite nonzero model quats.
                q /= scale[:, None]
                q /= np.linalg.norm(q, axis=1)[:, None]
            stamp = Time(nanoseconds=round(t * 1e9)).to_msg()
            msg = RobotTrajectory()
            for header in (msg.joint_trajectory.header, msg.multi_dof_joint_trajectory.header):
                header.stamp, header.frame_id = stamp, 'base'
            msg.multi_dof_joint_trajectory.joint_names = ['left_tcp', 'right_tcp']
            for k, action in enumerate(actions):
                ns = round(k * self.dt * 1e9)
                duration = Duration(sec=ns // 1_000_000_000, nanosec=ns % 1_000_000_000)
                transforms = []
                for start in (0, 7):
                    tf = Transform()
                    tf.translation.x, tf.translation.y, tf.translation.z = map(float, action[start : start + 3])
                    tf.rotation.x, tf.rotation.y, tf.rotation.z, tf.rotation.w = map(
                        float, action[start + 3 : start + 7]
                    )
                    transforms.append(tf)
                msg.multi_dof_joint_trajectory.points.append(
                    MultiDOFJointTrajectoryPoint(transforms=transforms, time_from_start=duration)
                )
        except (ValueError, TypeError, OverflowError):
            return False
        # Control callbacks and the actual publish share this lock. A release
        # that has returned can never be followed by this generation's output.
        with self.lock:
            now = self._now()
            if (
                generation != self.generation
                or not self._ready_locked(now)
                or not 0 <= now - t <= STALE
                or self._stop.is_set()
            ):
                return False
            self.chunk_pub.publish(msg)
        return True
