import numpy as np
import rerun as rr


def init_rerun() -> None:
    rr.init("rai_recorder", spawn=True)


def make_viz_node(config: dict):
    from rclpy.node import Node
    from sensor_msgs.msg import Image, JointState

    node = Node("rai_recorder_viz")

    for img_cfg in config["topics"]["images"]:
        node.create_subscription(Image, img_cfg["topic"], _image_cb(img_cfg["key"]), 10)

    for state_cfg in config["topics"]["state"]:
        node.create_subscription(JointState, state_cfg["topic"], _state_cb(state_cfg["key"]), 10)

    return node


def _image_cb(key: str):
    def cb(msg):
        img = _to_numpy(msg)
        if img is not None:
            rr.log(key, rr.Image(img))
    return cb


def _state_cb(key: str):
    def cb(msg):
        for name, pos in zip(msg.name, msg.position):
            rr.log(f"{key}/{name}", rr.Scalar(pos))
    return cb


def _to_numpy(msg) -> np.ndarray | None:
    channels = {"rgb8": 3, "bgr8": 3, "mono8": 1, "rgba8": 4}.get(msg.encoding)
    if channels is None:
        return None
    arr = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, channels)
    if msg.encoding == "bgr8":
        arr = arr[:, :, ::-1]
    return arr
