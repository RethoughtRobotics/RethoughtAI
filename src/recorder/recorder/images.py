"""Colour image transport for recording; live consumers keep using the raw topics."""
import numpy as np


def compressed_recording(config: dict) -> bool:
    transport = config.get("recording", {}).get("image_transport", "raw")
    if transport not in ("raw", "compressed"):
        raise ValueError("recording.image_transport must be raw or compressed")
    return transport == "compressed"


def recording_topic(config: dict, topic: str) -> str:
    if compressed_recording(config) and not topic.endswith("/compressed"):
        return topic + "/compressed"
    return topic


def jpeg_bytes(msg, quality: int) -> bytes:
    """Encode an 8-bit colour/mono ROS Image, respecting row stride and channel order.

    Depth and other unsupported encodings are rejected, never silently JPEG-compressed.
    """
    import cv2

    channels = {"rgb8": 3, "bgr8": 3, "rgba8": 4, "bgra8": 4, "mono8": 1}.get(msg.encoding)
    if channels is None:
        raise ValueError(f"JPEG recording does not support {msg.encoding}")
    if not 1 <= quality <= 100:
        raise ValueError("recording.jpeg_quality must be between 1 and 100")
    if msg.step < msg.width * channels or len(msg.data) != msg.step * msg.height:
        raise ValueError("Invalid image row stride or payload size")
    image = np.ndarray((msg.height, msg.width, channels), dtype=np.uint8, buffer=bytes(msg.data),
                       strides=(msg.step, channels, 1))
    image = image[..., :3] if channels > 1 else image[..., 0]
    if msg.encoding.startswith("rgb"):
        image = image[..., ::-1]
    try:
        ok, encoded = cv2.imencode(".jpg", np.ascontiguousarray(image), [cv2.IMWRITE_JPEG_QUALITY, quality])
    except cv2.error as exc:
        raise ValueError(f"JPEG encoding failed: {exc}") from exc
    if not ok:
        raise ValueError("JPEG encoding failed")
    return encoded.tobytes()


def make_compression_node(config: dict):
    from rclpy.node import Node
    from rclpy.qos import qos_profile_sensor_data
    from sensor_msgs.msg import CompressedImage, Image

    quality = int(config.get("recording", {}).get("jpeg_quality", 90))
    if not 1 <= quality <= 100:
        raise ValueError("recording.jpeg_quality must be between 1 and 100")
    node = Node("rai_recorder_compression")
    for camera in config["topics"]["images"]:
        topic = camera["topic"]
        if topic.endswith("/compressed"):
            continue  # already supplied by an external publisher
        publisher = node.create_publisher(CompressedImage, recording_topic(config, topic), 1)

        def on_image(msg, pub=publisher, source=topic):
            try:
                data = jpeg_bytes(msg, quality)
            except (ValueError, RuntimeError) as exc:
                node.get_logger().error(f"Cannot compress {source}: {exc}", throttle_duration_sec=5.0)
                return
            image = CompressedImage()
            image.header = msg.header  # preserve original acquisition timestamp and frame
            encoding = "mono8" if msg.encoding == "mono8" else "bgr8"
            image.format = f"{msg.encoding}; jpeg compressed {encoding}"
            image.data = data
            pub.publish(image)

        node.create_subscription(Image, topic, on_image, qos_profile_sensor_data)
    return node