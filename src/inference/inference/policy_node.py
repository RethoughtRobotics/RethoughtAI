"""Policy node: a LeRobot policy on the robot's topics, publishing action chunks for the runtime.

Observations are built as the LeRobot export built training frames: the latest message of each
camera, the TCP, joints and gripper. A worker thread runs the policy every `every` steps, so
inference never blocks ROS callbacks, and publishes:
  chunk       moveit_msgs/RobotTrajectory stamped at observation time: `{arm}_tcp` targets
              (base frame) and the `{arm}_gripper` trigger, at k / fps
  chunk_path  nav_msgs/Path of the targets, for RViz
With RTC, each chunk continues the last one over the measured inference delay.
"""

import math
import threading
import time

import numpy as np
from builtin_interfaces.msg import Duration
from baxter_core_msgs.msg import EndEffectorState
from geometry_msgs.msg import PoseStamped, Transform
from moveit_msgs.msg import RobotTrajectory
from nav_msgs.msg import Path
from rclpy.time import Time
from scipy.spatial.transform import Rotation
from sensor_msgs.msg import Image, JointState
from trajectory_msgs.msg import JointTrajectoryPoint, MultiDOFJointTrajectoryPoint

from adapter.labels import apply_actions, model_state, relative_actions
from adapter.lerobot_export import JOINTS, image_rgb
from inference.runtime import Chunk

STALE = 0.5  # s: older inputs pause inference (camera stopped, bag paused)
LOG_EVERY = 10.0  # s
WARMUP = 2  # inferences left out of the latency (CUDA init)


class PolicyNode:
    def __init__(self, node, cfg, images, policy, every=1):
        """`images`: the config's topics.images; the policy's cameras are used."""
        self.node, self.policy = node, policy
        self.fps, self.every = float(cfg["fps"]), int(every)
        self.dt = 1.0 / self.fps
        self.arm = cfg["arm"]
        self.joint_names = [f"{self.arm}_{j}" for j in JOINTS]
        self.latest = {}  # name -> (value, receive time)
        self.lock = threading.Lock()
        self.last = None  # last chunk published, for RTC
        from lerobot.policies.rtc import LatencyTracker

        self.latency = LatencyTracker()
        self.count = 0
        self.warned = False

        self.cameras = {c["key"]: c["topic"] for c in images if c["key"] in policy.features}
        missing = [k for k in policy.features if k.startswith("observation.images.") and k not in self.cameras]
        if missing:
            raise ValueError(f"no camera topic configured for {', '.join(missing)}")
        for key, topic in self.cameras.items():
            node.create_subscription(Image, topic, lambda m, k=key: self._store(k, np.array(image_rgb(m))), 2)
        node.create_subscription(PoseStamped, cfg["tcp"], self._on_tcp, 10)
        node.create_subscription(JointState, cfg["joint_states"], self._on_joints, 10)
        node.create_subscription(EndEffectorState, cfg["gripper"],
                                 lambda m: self._store("gripper", 1.0 - float(m.position) / 100.0), 10)
        self.chunk_pub = node.create_publisher(RobotTrajectory, cfg["chunk"], 10)
        self.path_pub = node.create_publisher(Path, cfg["chunk"] + "_path", 10)
        threading.Thread(target=self._run, daemon=True, name="policy").start()

    def _now(self):
        return self.node.get_clock().now().nanoseconds * 1e-9

    def _store(self, name, value):
        with self.lock:
            self.latest[name] = (value, self._now())

    def _on_tcp(self, msg):
        p, o = msg.pose.position, msg.pose.orientation
        self._store("tcp", (np.array([p.x, p.y, p.z]), np.array([o.x, o.y, o.z, o.w])))

    def _on_joints(self, msg):
        if self.joint_names[0] in msg.name:
            self._store("joints", np.array([msg.position[msg.name.index(n)] for n in self.joint_names]))

    def _observation(self):
        """(time, observation, TCP) from the latest inputs, or None while any is missing or stale."""
        now = self._now()
        with self.lock:
            latest = dict(self.latest)
        names = [*self.cameras, "tcp", "joints", "gripper"]
        if any(n not in latest or now - latest[n][1] > STALE for n in names):
            return None
        tcp = latest["tcp"][0]
        obs = {key: latest[key][0] for key in self.cameras}
        obs["observation.state"] = model_state(*tcp, latest["gripper"][0], latest["joints"][0])
        return now, obs, tcp

    def _rtc_args(self, t, tcp):
        """(delay steps, the last chunk's actions from t on, relative to `tcp`) for Real-Time Chunking."""
        delay = math.ceil(self.latency.max() / self.dt) if len(self.latency) else 0
        horizon = self.policy.policy.config.rtc_config.execution_horizon
        if delay >= horizon and not self.warned:
            self.warned = True
            self.node.get_logger().warning(
                f"inference delay ({delay} steps) reaches RTC's execution_horizon ({horizon}): chunks will switch "
                "where RTC did not constrain them. Raise execution_horizon above the delay.")
        if self.last is None or self.last.end < t:
            return delay, None
        ts = t + self.dt * np.arange(int((self.last.end - t) * self.fps) + 1)
        samples = [self.last.at(x) for x in ts]
        positions, rotations = np.array([s[0] for s in samples]), Rotation.concatenate([s[1] for s in samples])
        return delay, relative_actions(positions, rotations, np.array([s[2] for s in samples]), *tcp)

    def _run(self):
        next_start = 0.0
        while True:
            wait = next_start - time.monotonic()
            if wait > 0:
                time.sleep(wait)
            observed = self._observation()
            if observed is None:
                self.last = None
                time.sleep(0.01)
                continue
            next_start = time.monotonic() + self.every * self.dt
            t, obs, (tcp_p, tcp_q) = observed
            started = time.perf_counter()
            delay, prev = self._rtc_args(t, (tcp_p, tcp_q)) if self.policy.rtc else (0, None)
            actions = self.policy(obs, delay, prev)
            latency = time.perf_counter() - started
            self.count += 1
            if self.count > WARMUP:
                self.latency.add(latency)
            positions, rotations = apply_actions(actions, tcp_p, tcp_q)
            self.last = Chunk(t + self.dt * np.arange(len(actions)), positions, rotations, actions[:, 6], t)
            self._publish(t, positions, rotations, actions[:, 6])
            if self.count % max(1, round(LOG_EVERY * self.fps / self.every)) == 0:
                self.node.get_logger().info(
                    f"{self.count} chunks; inference p50 {1000 * self.latency.percentile(0.5):.0f} ms, "
                    f"max {1000 * self.latency.max():.0f} ms ({self.dt * 1000:.0f} ms per step)")

    def _publish(self, t, positions, rotations, triggers):
        stamp = Time(nanoseconds=round(t * 1e9)).to_msg()
        msg, path = RobotTrajectory(), Path()
        for header in (msg.joint_trajectory.header, msg.multi_dof_joint_trajectory.header, path.header):
            header.stamp, header.frame_id = stamp, "base"
        msg.multi_dof_joint_trajectory.joint_names = [f"{self.arm}_tcp"]
        msg.joint_trajectory.joint_names = [f"{self.arm}_gripper"]
        for k, (p, q, trigger) in enumerate(zip(positions, rotations.as_quat(), triggers)):
            offset = k * self.dt
            duration = Duration(sec=int(offset), nanosec=int(round((offset - int(offset)) * 1e9)))
            tf = Transform()
            tf.translation.x, tf.translation.y, tf.translation.z = (float(x) for x in p)
            tf.rotation.x, tf.rotation.y, tf.rotation.z, tf.rotation.w = (float(x) for x in q)
            msg.multi_dof_joint_trajectory.points.append(MultiDOFJointTrajectoryPoint(transforms=[tf], time_from_start=duration))
            msg.joint_trajectory.points.append(JointTrajectoryPoint(positions=[float(trigger)], time_from_start=duration))
            pose = PoseStamped()
            pose.header = path.header
            pose.pose.position.x, pose.pose.position.y, pose.pose.position.z = tf.translation.x, tf.translation.y, tf.translation.z
            pose.pose.orientation = tf.rotation
            path.poses.append(pose)
        self.chunk_pub.publish(msg)
        self.path_pub.publish(path)
