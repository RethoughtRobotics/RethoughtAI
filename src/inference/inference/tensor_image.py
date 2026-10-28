"""A 1-D float32 array (a policy's observation encoding) as a stamped ROS message, and back.

ROS 2 has no stamped tensor message from Jazzy to Lyrical (tensor_msgs/ExperimentalTensor exists only on
Lyrical and Rolling, and is experimental), so the array travels as a sensor_msgs/Image of one row,
encoding 32FC1: in every distribution, stamped, recorded by rosbag2 and read back with one
np.frombuffer. This module is the only place that knows the layout.
"""

import sys

import numpy as np
from sensor_msgs.msg import Image

ENCODING = '32FC1'


def to_image(array, stamp):
    """`array` (any shape, flattened) as a 1 x N 32FC1 Image stamped `stamp` (builtin_interfaces/Time)."""
    a = np.ascontiguousarray(np.asarray(array, np.float32).reshape(-1))
    msg = Image()
    msg.header.stamp = stamp
    msg.height, msg.width, msg.encoding = 1, a.size, ENCODING
    msg.is_bigendian = int(sys.byteorder == 'big')
    msg.step = 4 * a.size
    msg.data = a.tobytes()
    return msg


def from_image(msg):
    """The float32 array (N,) an Image from to_image holds."""
    if msg.encoding != ENCODING or msg.height != 1:
        raise ValueError(f'not a 1 x N {ENCODING} array image: {msg.height} x {msg.width} {msg.encoding}')
    dtype = np.dtype('>f4' if msg.is_bigendian else '<f4')
    return np.frombuffer(bytes(msg.data), dtype).astype(np.float32)
