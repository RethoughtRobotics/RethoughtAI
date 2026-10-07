import numpy as np
import rerun as rr


def init_rerun() -> None:
    rr.init("rai_recorder", spawn=True)


def make_viz_node(config: dict):
    from rclpy.node import Node
    from sensor_msgs.msg import Image, JointState

    node = Node("rai_recorder_viz")
    node.create_timer(2.0, lambda: print("[viz] executor alive", flush=True))

    for img_cfg in config["topics"]["images"]:
        print(f"[viz] subscribing to image: {img_cfg['topic']}", flush=True)
        node.create_subscription(Image, img_cfg["topic"], _image_cb(img_cfg["key"]), 100)

    for state_cfg in config["topics"]["state"]:
        print(f"[viz] subscribing to state: {state_cfg['topic']}", flush=True)
        node.create_subscription(JointState, state_cfg["topic"], _state_cb(state_cfg["key"]), 10)

    return node


def _image_cb(key: str):
    _logged = [False]
    def cb(msg):
        if not _logged[0]:
            print(f"[viz] receiving {key}: {msg.width}x{msg.height} encoding={msg.encoding!r}")
            _logged[0] = True
        img = _to_numpy(msg)
        if img is not None:
            rr.log(key, rr.Image(img))
        else:
            print(f"[viz] unsupported encoding {msg.encoding!r} on {key}")
    return cb


def _state_cb(key: str):
    _logged = [False]
    def cb(msg):
        if not _logged[0]:
            print(f"[viz] receiving {key}: {len(msg.name)} joints", flush=True)
            _logged[0] = True
        for name, pos in zip(msg.name, msg.position):
            rr.log(f"{key}/{name}", rr.Scalars([pos]))
    return cb


def _to_numpy(msg) -> np.ndarray | None:
    channels = {"rgb8": 3, "bgr8": 3, "mono8": 1, "rgba8": 4}.get(msg.encoding)
    if channels is None:
        return None
    arr = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, channels)
    if msg.encoding == "bgr8":
        arr = arr[:, :, ::-1]
    return arr
