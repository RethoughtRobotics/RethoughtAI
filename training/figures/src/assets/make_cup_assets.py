"""Frames and one planned action chunk of the cup Multi-Task DiT for dit_cup.typ, from the robot bag
20261028_014141_real_Kernel_Corrector_bluecup_samelayout (blue cup, the base policy driving), at the
plan nearest AT seconds into the episode: both cameras as the policy saw them (stretched to 240 x 240,
centre 224), and that plan's 24 TCP waypoints relative to the first.

    source /opt/ros/kilted/setup.bash && source ~/FinalProject/baxter_sdk/install/setup.bash
    /usr/bin/python3 training/figures/src/assets/make_cup_assets.py
"""

import json
from pathlib import Path

import cv2
import numpy as np
import rosbag2_py
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message

HERE = Path(__file__).parent
BAG = HERE / '../../../../../data/rollouts/20261028_014141_real_Kernel_Corrector_bluecup_samelayout'
AT = 7.0  # s after the first plan: the approach, the cup in both views
TOPICS = {
    'left': '/cameras/left_hand_camera/image/compressed',
    'third_eye': '/cameras/third_eye/color/image_raw/compressed',
}


def view(bgr):
    return cv2.resize(bgr, (240, 240), interpolation=cv2.INTER_AREA)[8:232, 8:232]


def main():
    r = rosbag2_py.SequentialReader()
    r.open(rosbag2_py.StorageOptions(uri=str(BAG), storage_id='mcap'), rosbag2_py.ConverterOptions('cdr', 'cdr'))
    types = {t.name: t.type for t in r.get_all_topics_and_types()}
    images, plans = {k: [] for k in TOPICS}, []
    names = {v: k for k, v in TOPICS.items()}
    while r.has_next():
        topic, raw, ts = r.read_next()
        if topic in names:
            images[names[topic]].append((ts * 1e-9, raw))
        elif topic == '/policy/left/chunk_path':
            m = deserialize_message(raw, get_message(types[topic]))
            if m.poses:
                plans.append((
                    m.header.stamp.sec + m.header.stamp.nanosec * 1e-9,
                    [[p.pose.position.x, p.pose.position.y, p.pose.position.z] for p in m.poses],
                ))
    t, path = min(plans, key=lambda p: abs(p[0] - (plans[0][0] + AT)))
    msg = get_message('sensor_msgs/msg/CompressedImage')
    for k, frames in images.items():
        raw = min(frames, key=lambda f: abs(f[0] - t))[1]
        bgr = cv2.imdecode(np.frombuffer(bytes(deserialize_message(raw, msg).data), np.uint8), cv2.IMREAD_COLOR)
        cv2.imwrite(str(HERE / f'cup_{k}.jpg'), view(bgr), [cv2.IMWRITE_JPEG_QUALITY, 92])
    p = np.array(path)
    (HERE / 'cup_chunk.json').write_text(
        json.dumps({
            'bag': BAG.name,
            'plan_time_s': round(t - plans[0][0], 2),
            'xyz_mm': np.round((p - p[0]) * 1000, 1).tolist(),
        })
    )
    print('chunk span (mm):', np.round(np.ptp(p, axis=0) * 1000, 1), len(p), 'waypoints')


if __name__ == '__main__':
    main()
