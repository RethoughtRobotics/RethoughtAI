import numpy as np
import rerun as rr


def init_rerun() -> None:
    rr.init('rai_recorder')
    # The viewer keeps every streamed frame; its default limit (75% of RAM) let it grow past 20 GB in
    # one session and swap-stall the teleop and recording. At the limit it drops the oldest frames
    # from the viewer only (the bags are unaffected).
    rr.spawn(memory_limit='2GB')


def make_viz_node(config: dict):
    from rclpy.node import Node
    from sensor_msgs.msg import Image, JointState

    node = Node('rai_recorder_viz')
    for img_cfg in config['topics']['images']:
        node.create_subscription(Image, img_cfg['topic'], _image_cb(img_cfg['key']), 100)

    for state_cfg in config['topics']['state']:
        node.create_subscription(JointState, state_cfg['topic'], _state_cb(state_cfg['key']), 10)

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
            rr.log(f'{key}/{name}', rr.Scalars([pos]))

    return cb


def _to_numpy(msg) -> np.ndarray | None:
    channels = {'rgb8': 3, 'bgr8': 3, 'mono8': 1, 'rgba8': 4, 'bgra8': 4}.get(msg.encoding)
    if channels is None:
        return None
    arr = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, channels)
    if msg.encoding == 'bgr8':
        arr = arr[:, :, ::-1]  # BGR to RGB
    elif msg.encoding == 'bgra8':
        arr = arr[:, :, [2, 1, 0, 3]]  # BGRA to RGBA
    return arr
