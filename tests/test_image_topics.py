import numpy as np
import pytest

pytest.importorskip('rclpy')
cv2 = pytest.importorskip('cv2')

from sensor_msgs.msg import CompressedImage, Image  # noqa: E402

from adapter.lerobot_export import image_rgb  # noqa: E402
from inference.policy_node import image_type  # noqa: E402


def test_compressed_topics_use_the_compressed_message():
    assert image_type('/cameras/third_eye/color/image_raw_inpainted_lamb/compressed') is CompressedImage
    assert image_type('/cameras/left_hand_camera/image') is Image


def test_a_compressed_frame_decodes_to_rgb():
    red = np.zeros((8, 8, 3), np.uint8)
    red[..., 0] = 255
    jpeg = cv2.imencode('.jpg', red[..., ::-1])[1].tobytes()
    image = image_rgb(CompressedImage(format='jpeg', data=jpeg))
    assert image.shape == (8, 8, 3) and image[..., 0].min() > 240 and image[..., 2].max() < 15
