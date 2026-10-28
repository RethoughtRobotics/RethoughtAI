"""Policy node: a LeRobot policy on the robot's topics, publishing action chunks for the runtime.

Observations are built as the LeRobot export built training frames: the latest message of each
camera, the TCP, joints and gripper. A worker thread runs the policy every `every` steps, so
inference never blocks ROS callbacks, and publishes:
  chunk       moveit_msgs/RobotTrajectory stamped at observation time: `{arm}_tcp` targets
              (base frame) and the `{arm}_gripper` trigger, at k / fps
  chunk_path  nav_msgs/Path of the targets, for RViz
With RTC, each chunk continues the last one over the measured inference delay.
A policy that sees past steps (n_obs_steps > 1, MultiTask DiT) gets them one step apart, as in
training, from observations sampled every half step, not from its previous inference (~100 ms back).
When the runtime engages or releases (`engaged`), the policy starts fresh: no chunk carries over.
A cuff squeeze (`cuff`) is a correction by hand, for any policy: it restarts when one starts and
ends; plans during it follow the hand (no previous chunk is continued), and on release the newest is
re-anchored on the corrected pose, as the runtime executes it, so the next one continues from there.
A VR takeover (`takeover`, the teleop's --takeover) is the same correction.
"""

import math
import threading
import time
from collections import deque

import numpy as np
from baxter_core_msgs.msg import DigitalIOState, EndEffectorState
from builtin_interfaces.msg import Duration
from geometry_msgs.msg import PoseStamped, Transform, TwistStamped, Vector3Stamped
from moveit_msgs.msg import RobotTrajectory
from nav_msgs.msg import Path
from rclpy.qos import DurabilityPolicy, QoSProfile
from rclpy.time import Time
from scipy.spatial.transform import Rotation
from sensor_msgs.msg import Image, JointState
from std_msgs.msg import Bool
from trajectory_msgs.msg import JointTrajectoryPoint, MultiDOFJointTrajectoryPoint

from adapter.labels import apply_actions, relative_actions
from adapter.lerobot_export import JOINTS, image_rgb
from inference.runtime import Chunk
from inference.tensor_image import to_image

STALE = 0.5  # s: older inputs pause inference (camera stopped, bag paused)
LOG_EVERY = 10.0  # s
WARMUP = 2  # inferences left out of the latency (CUDA init)
LATENCY_WINDOW = 10  # recent inferences the RTC delay is taken from; one spike must not pin it
# RTC continues the last chunk only while the arm follows it: a plan this far ahead of the TCP (m)
# starts fresh instead. Bag real_wrist_bugs: up to 203 mm ahead while fine, then 274 .. 591 mm.
MAX_LEAD = 0.25


class PolicyNode:
    def __init__(self, node, cfg, images, policy, every=1):
        """`images`: the config's topics.images; the policy's cameras are used."""
        self.node, self.policy = node, policy
        self.fps, self.every = float(cfg['fps']), int(every)
        self.dt = 1.0 / self.fps
        self.arm = cfg['arm']
        self.joint_names = [f'{self.arm}_{j}' for j in JOINTS]
        self.latest = {}  # name -> (value, receive time)
        self.lock = threading.Lock()
        self.last = None  # last chunk published, for RTC
        from lerobot.policies.rtc import LatencyTracker

        self.latency = LatencyTracker(maxlen=LATENCY_WINDOW)
        self.count = 0
        self.warned = False
        self.fresh = threading.Event()  # set: drop the last chunk and the policy's state
        self.anchor = None  # TCP (p, R) the last chunk was planned from
        self.squeezed = self.correcting = False  # by hand now (cuff, VR takeover); as the worker sees it

        self.cameras = {c['key']: c['topic'] for c in images if c['key'] in policy.features}
        missing = [k for k in policy.features if k.startswith('observation.images.') and k not in self.cameras]
        if missing:
            raise ValueError(f'no camera topic configured for {", ".join(missing)}')
        for key, topic in self.cameras.items():
            node.create_subscription(Image, topic, lambda m, k=key: self._store(k, np.array(image_rgb(m))), 2)
        node.create_subscription(PoseStamped, cfg['tcp'], self._on_tcp, 10)
        node.create_subscription(JointState, cfg['joint_states'], self._on_joints, 10)
        node.create_subscription(
            EndEffectorState, cfg['gripper'], lambda m: self._store('gripper', 1.0 - float(m.position) / 100.0), 10
        )
        if cfg.get('engaged'):
            latched = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)
            node.create_subscription(Bool, cfg['engaged'], self._on_engaged, latched)
        self.by_hand = {}  # source: correcting by hand now (the cuff squeezed, the teleop's VR takeover)
        if cfg.get('cuff'):
            node.create_subscription(DigitalIOState, cfg['cuff'], lambda m: self._by_hand('cuff', m.state), 10)
        if cfg.get('takeover'):
            node.create_subscription(Bool, cfg['takeover'], lambda m: self._by_hand('takeover', m.data), 10)
        self.chunk_pub = node.create_publisher(RobotTrajectory, cfg['chunk'], 10)
        # A nudge adapter's offset of each plan (linear: position m, angular: rotation vector rad, base axes)
        self.nudge_pub = (
            node.create_publisher(TwistStamped, cfg['nudge'], 10)
            if cfg.get('nudge') and getattr(policy, 'nudge', None)
            else None
        )
        # A kernel adapter's evidence for each plan's offset: x the highest cosine similarity to a stored
        # key, y s_min, z the keys within it; x <= y means the offset is exactly 0 (ttt.correctors.kernel)
        self.similarity_pub = (
            node.create_publisher(Vector3Stamped, cfg['nudge'] + '_similarity', 10)
            if self.nudge_pub is not None and policy.nudge.kind == 'kernel'
            else None
        )
        # The policy's observation encoding of each plan, for correctors fitted from the bag (inference.tensor_image)
        self.z_pub = node.create_publisher(Image, cfg['z'], 10) if cfg.get('z') and hasattr(policy, 'z') else None
        self.path_pub = node.create_publisher(Path, cfg['chunk'] + '_path', 10)
        self.samples = deque(maxlen=2 * policy.n_obs_steps + 2)  # (time, observation), every half step
        if policy.n_obs_steps > 1:
            node.create_timer(self.dt / 2, self._sample)
        threading.Thread(target=self._run, daemon=True, name='policy').start()

    def _by_hand(self, source, on):
        self.by_hand[source] = bool(on)
        self.squeezed = any(self.by_hand.values())

    def _now(self):
        return self.node.get_clock().now().nanoseconds * 1e-9

    def _store(self, name, value):
        with self.lock:
            self.latest[name] = (value, self._now())

    def _on_engaged(self, msg):
        self.fresh.set()
        explorer = getattr(self.policy, 'explorer', None)
        if explorer is not None:  # its record marks the plans of the attempt
            explorer.engaged = bool(msg.data)

    def _on_tcp(self, msg):
        p, o = msg.pose.position, msg.pose.orientation
        self._store('tcp', (np.array([p.x, p.y, p.z]), np.array([o.x, o.y, o.z, o.w])))

    def _on_joints(self, msg):
        if self.joint_names[0] in msg.name:
            self._store('joints', np.array([msg.position[msg.name.index(n)] for n in self.joint_names]))

    def _observation(self):
        """(time, observation, TCP) from the latest inputs, or None while any is missing or stale."""
        now = self._now()
        with self.lock:
            latest = dict(self.latest)
        names = [*self.cameras, 'tcp', 'joints', 'gripper']
        if any(n not in latest or now - latest[n][1] > STALE for n in names):
            return None
        tcp = latest['tcp'][0]
        obs = {key: latest[key][0] for key in self.cameras}
        obs['observation.state'] = self.policy.state(*tcp, latest['gripper'][0], latest['joints'][0])
        return now, obs, tcp

    def _sample(self):
        observed = self._observation()
        if observed is not None:
            with self.lock:
                self.samples.append(observed[:2])

    def _history(self, t):
        """The observations 1 .. n_obs_steps - 1 steps before t, oldest first: the sample nearest each,
        if one is within half a step of it (none yet after a pause: the policy repeats the oldest)."""
        with self.lock:
            samples = list(self.samples)
        history = []
        for j in range(self.policy.n_obs_steps - 1, 0, -1):
            nearest = min(samples, key=lambda s: abs(s[0] - (t - j * self.dt)), default=None)
            if nearest is not None and abs(nearest[0] - (t - j * self.dt)) <= self.dt / 2:
                history.append(nearest[1])
        return history

    def _rtc_args(self, t, tcp):
        """(delay steps, the last chunk's actions from t on, relative to `tcp`) for Real-Time Chunking;
        no last chunk (a fresh one from this observation) when RTC can't continue it safely:
          - the delay reaches execution_horizon: RTC pins the whole horizon to the last chunk and the
            runtime executes only past the delay, the part nothing anchors. Chunk after chunk the plan
            ran ahead of the arm (bag real_wrist_bugs: latency 0.5 -> 1.6 s, delay 36 of 25 steps,
            the plan 591 mm ahead of the TCP and the arm at 0.8 m/s).
          - the last chunk is more than MAX_LEAD ahead of the TCP: the arm isn't following it."""
        delay = math.ceil(self.latency.percentile(0.9) / self.dt) if len(self.latency) else 0
        horizon = self.policy.policy.config.rtc_config.execution_horizon
        if self.last is None or self.last.end < t:
            return delay, None
        if delay >= horizon:
            self._warn_once(
                f"inference delay ({delay} steps) reaches RTC's execution_horizon ({horizon}): "
                'fresh chunks, not continued, until it is below'
            )
            return delay, None
        ts = t + self.dt * np.arange(int((self.last.end - t) * self.fps) + 1)
        samples = [self.last.at(x) for x in ts]
        positions, rotations = np.array([s[0] for s in samples]), Rotation.concatenate([s[1] for s in samples])
        lead = np.linalg.norm(positions[0] - tcp[0])
        if lead > MAX_LEAD:
            self._warn_once(f'the last chunk is {1000 * lead:.0f} mm ahead of the TCP: a fresh chunk, not continued')
            return delay, None
        self.warned = False
        return delay, relative_actions(positions, rotations, np.array([s[2] for s in samples]), *tcp)

    def _warn_once(self, text):
        """Warn when RTC stops continuing chunks, once until it continues one again."""
        if not self.warned:
            self.warned = True
            self.node.get_logger().warning(text)

    def _run(self):
        next_start = 0.0
        while True:
            wait = next_start - time.monotonic()
            if wait > 0:
                time.sleep(wait)
            if self.fresh.is_set():
                self.fresh.clear()
                self.last = None
                self.policy.reset()
            if self.squeezed != self.correcting:  # a correction starts or ends
                self.correcting = self.squeezed
                self.policy.reset()
                with self.lock:
                    tcp = self.latest.get('tcp')
                if not self.correcting and self.last is not None and tcp is not None:
                    self.last = self.last.reanchored(*self.anchor, tcp[0][0], Rotation.from_quat(tcp[0][1]))
            if self.correcting:
                self.last = None  # plan from what the hand did, not the plan it overrode
            observed = self._observation()
            if observed is None:
                self.last = None
                time.sleep(0.01)
                continue
            next_start = time.monotonic() + self.every * self.dt
            t, obs, (tcp_p, tcp_q) = observed
            started = time.perf_counter()
            delay, prev = self._rtc_args(t, (tcp_p, tcp_q)) if self.policy.rtc else (0, None)
            history = self._history(t) if self.policy.n_obs_steps > 1 else ()
            actions = self.policy(obs, delay, prev, tcp=(tcp_p, tcp_q), history=history)
            latency = time.perf_counter() - started
            self.count += 1
            if self.count > WARMUP:
                self.latency.add(latency)
            positions, rotations = apply_actions(actions, tcp_p, tcp_q)
            self.last = Chunk(t + self.dt * np.arange(len(actions)), positions, rotations, actions[:, 6], t)
            self.anchor = (tcp_p, Rotation.from_quat(tcp_q))
            self._publish(t, positions, rotations, actions[:, 6])
            if self.nudge_pub is not None:
                self._publish_nudge(t, self.policy.nudge.offset)
            if self.similarity_pub is not None:
                self._publish_similarity(t, self.policy.nudge.model)
            if self.z_pub is not None and self.policy.z is not None:
                self.z_pub.publish(to_image(self.policy.z, Time(nanoseconds=round(t * 1e9)).to_msg()))
            if self.count % max(1, round(LOG_EVERY * self.fps / self.every)) == 0:
                self.node.get_logger().info(
                    f'{self.count} chunks; inference p50 {1000 * self.latency.percentile(0.5):.0f} ms, '
                    f'recent max {1000 * self.latency.percentile(0.99):.0f} ms ({self.dt * 1000:.0f} ms per step)'
                )

    def _publish_nudge(self, t, offset):
        msg = TwistStamped()
        msg.header.stamp, msg.header.frame_id = Time(nanoseconds=round(t * 1e9)).to_msg(), 'base'
        msg.twist.linear.x, msg.twist.linear.y, msg.twist.linear.z = (float(x) for x in offset[:3])
        msg.twist.angular.x, msg.twist.angular.y, msg.twist.angular.z = (float(x) for x in offset[3:6])
        self.nudge_pub.publish(msg)

    def _publish_similarity(self, t, kernel):
        msg = Vector3Stamped()
        msg.header.stamp, msg.header.frame_id = Time(nanoseconds=round(t * 1e9)).to_msg(), 'base'
        msg.vector.x, msg.vector.y, msg.vector.z = kernel.s_max, kernel.s_min, float(kernel.active)
        self.similarity_pub.publish(msg)

    def _publish(self, t, positions, rotations, triggers):
        stamp = Time(nanoseconds=round(t * 1e9)).to_msg()
        msg, path = RobotTrajectory(), Path()
        for header in (msg.joint_trajectory.header, msg.multi_dof_joint_trajectory.header, path.header):
            header.stamp, header.frame_id = stamp, 'base'
        msg.multi_dof_joint_trajectory.joint_names = [f'{self.arm}_tcp']
        msg.joint_trajectory.joint_names = [f'{self.arm}_gripper']
        for k, (p, q, trigger) in enumerate(zip(positions, rotations.as_quat(), triggers)):
            offset = k * self.dt
            duration = Duration(sec=int(offset), nanosec=int(round((offset - int(offset)) * 1e9)))
            tf = Transform()
            tf.translation.x, tf.translation.y, tf.translation.z = (float(x) for x in p)
            tf.rotation.x, tf.rotation.y, tf.rotation.z, tf.rotation.w = (float(x) for x in q)
            msg.multi_dof_joint_trajectory.points.append(
                MultiDOFJointTrajectoryPoint(transforms=[tf], time_from_start=duration)
            )
            msg.joint_trajectory.points.append(
                JointTrajectoryPoint(positions=[float(trigger)], time_from_start=duration)
            )
            pose = PoseStamped()
            pose.header = path.header
            pose.pose.position.x, pose.pose.position.y, pose.pose.position.z = (
                tf.translation.x,
                tf.translation.y,
                tf.translation.z,
            )
            pose.pose.orientation = tf.rotation
            path.poses.append(pose)
        self.chunk_pub.publish(msg)
        self.path_pub.publish(path)


class RecordingLight:
    """The light the recorder keeps on while it records an episode (config trigger.light, the left
    navigator's inner light), so every training frame shows it lit: on while the policy serves, as the
    cameras saw it. Re-sent every second in case anything else switched it off; off() at exit."""

    def __init__(self, node, light):
        import importlib

        package, name = light['msg_type'].split('/')
        self.msg_type = getattr(importlib.import_module(f'{package}.msg'), name)
        self.name = light['name']
        self.pub = node.create_publisher(self.msg_type, light['topic'], 10)
        self._set(True)
        node.create_timer(1.0, lambda: self._set(True))

    def _set(self, on):
        msg = self.msg_type()
        msg.name, msg.value = self.name, on
        self.pub.publish(msg)

    def off(self):
        self._set(False)
        time.sleep(0.3)  # let it go out before the node does
