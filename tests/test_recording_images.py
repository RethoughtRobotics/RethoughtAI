"""ROS-sourced unittest tests; no ROS graph or robot connection is started."""

import copy
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np
import rosbag2_py
import yaml
from baxter_core_msgs.msg import EndEffectorState
from rclpy.serialization import serialize_message
from sensor_msgs.msg import CompressedImage, Image, JointState

from adapter.lerobot_export import JOINTS, image_rgb, read_frames
from recorder.cli import all_topics, exclude_topics
from recorder.images import compressed_recording, jpeg_bytes, make_compression_node, recording_topic
from recorder.session import EpisodeBag, _setup


def colour_image(encoding='bgra8', padding=0):
    pixel = {
        'bgra8': [0, 0, 255, 255],
        'rgba8': [255, 0, 0, 255],
        'bgr8': [0, 0, 255],
        'rgb8': [255, 0, 0],
        'mono8': [128],
    }[encoding]
    width, height = 32, 24
    row = bytes(pixel) * width + bytes(padding)
    return Image(height=height, width=width, encoding=encoding, step=len(row), data=row * height)


class RecordingImagesTest(unittest.TestCase):
    def test_raw_is_generic_default(self):
        self.assertFalse(compressed_recording({}))
        self.assertEqual(recording_topic({}, '/camera/image'), '/camera/image')

    def test_compressed_topic_and_existing_suffix(self):
        cfg = {'recording': {'image_transport': 'compressed'}}
        self.assertEqual(recording_topic(cfg, '/camera/image'), '/camera/image/compressed')
        self.assertEqual(recording_topic(cfg, '/camera/image/compressed'), '/camera/image/compressed')
        with self.assertRaises(ValueError):
            compressed_recording({'recording': {'image_transport': 'typo'}})

    def test_baxter_selection_excludes_raw_images_and_keeps_operator(self):
        cfg = yaml.safe_load((Path(__file__).parents[1] / 'configs/baxter.yaml').read_text())
        self.assertTrue(compressed_recording(cfg))
        topics = all_topics(cfg)
        for camera in cfg['topics']['images']:
            self.assertNotIn(camera['topic'], topics)
            self.assertIn(camera['topic'] + '/compressed', topics)
        self.assertEqual(len(topics), len(set(topics)))
        filtered = exclude_topics(copy.deepcopy(cfg), ('right',))
        self.assertNotIn('/cameras/right_hand_camera/image/compressed', all_topics(filtered))
        self.assertIn('/vive/right/target', all_topics(filtered))
        self.assertIn('/diagnostics', topics)

    def test_colour_round_trips_and_padded_rows(self):
        for encoding in ('rgb8', 'bgr8', 'rgba8', 'bgra8'):
            for padding in (0, 8):
                with self.subTest(encoding=encoding, padding=padding):
                    raw = colour_image(encoding, padding)
                    compressed = CompressedImage(format=f'{encoding}; jpeg compressed bgr8', data=jpeg_bytes(raw, 90))
                    rgb = image_rgb(compressed)
                    self.assertEqual(rgb.shape, (24, 32, 3))
                    self.assertTrue(rgb.flags.c_contiguous)
                    np.testing.assert_allclose(rgb[12, 16], [255, 0, 0], atol=3)

    def test_mono_round_trip(self):
        rgb = image_rgb(
            CompressedImage(format='mono8; jpeg compressed mono8', data=jpeg_bytes(colour_image('mono8', 4), 90))
        )
        self.assertEqual(rgb.shape, (24, 32, 3))
        np.testing.assert_allclose(rgb, 128, atol=2)

    def test_unsupported_depth_invalid_stride_quality_and_corrupt_jpeg(self):
        raw = colour_image()
        raw.encoding = '16UC1'
        with self.assertRaises(ValueError):
            jpeg_bytes(raw, 90)
        raw = colour_image()
        raw.step = 1
        with self.assertRaises(ValueError):
            jpeg_bytes(raw, 90)
        for quality in (0, 101):
            with self.assertRaises(ValueError):
                jpeg_bytes(colour_image(), quality)
        with self.assertRaises(ValueError):
            image_rgb(CompressedImage(format='jpeg', data=b'not a jpeg'))

    def test_compression_node_preserves_header_and_skips_precompressed(self):
        fake_node = Mock()
        cfg = {
            'topics': {'images': [{'topic': '/left/image'}, {'topic': '/right/image/compressed'}]},
            'recording': {'image_transport': 'compressed', 'jpeg_quality': 90},
        }
        with patch('rclpy.node.Node', return_value=fake_node):
            self.assertIs(make_compression_node(cfg), fake_node)
        fake_node.create_publisher.assert_called_once_with(CompressedImage, '/left/image/compressed', 1)
        self.assertEqual(fake_node.create_subscription.call_count, 1)
        cb = fake_node.create_subscription.call_args.args[2]
        raw = colour_image()
        raw.header.stamp.sec, raw.header.stamp.nanosec = 123, 456
        raw.header.frame_id = 'camera_frame'
        cb(raw)
        published = fake_node.create_publisher.return_value.publish.call_args.args[0]
        self.assertEqual(published.header, raw.header)
        self.assertEqual(published.format, 'bgra8; jpeg compressed bgr8')
        np.testing.assert_allclose(image_rgb(published)[0, 0], [255, 0, 0], atol=3)

    def test_bad_compression_quality_fails_before_node_creation(self):
        with patch('rclpy.node.Node') as node:
            with self.assertRaises(ValueError):
                make_compression_node({'recording': {'jpeg_quality': 101}})
            node.assert_not_called()

    def test_bag_frame_count_uses_compressed_topic(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)
            (path / 'metadata.yaml').write_text(
                yaml.safe_dump({
                    'rosbag2_bagfile_information': {
                        'topics_with_message_count': [
                            {'topic_metadata': {'name': '/camera/image/compressed'}, 'message_count': 42}
                        ]
                    }
                })
            )
            with patch('recorder.session.subprocess.Popen'):
                bag = EpisodeBag(path, ['/camera/image/compressed'])
            self.assertEqual(
                bag.frames(recording_topic({'recording': {'image_transport': 'compressed'}}, '/camera/image')), 42
            )

    def test_no_viz_keyboard_still_starts_compressor(self):
        cfg = {'topics': {'images': [{'topic': '/camera/image'}]}, 'recording': {'image_transport': 'compressed'}}
        trigger = {'type': 'keyboard', 'record': 'r', 'success': 's', 'discard': 'd'}
        with (
            patch('rclpy.init') as init,
            patch('recorder.images.make_compression_node') as make_node,
            patch('recorder.session.threading.Thread') as thread,
        ):
            started, _, screen = _setup(trigger, cfg, Mock(), Mock(), Mock(), Mock(), False)
            self.assertTrue(started)
            self.assertIsNone(screen)
            init.assert_called_once()
            make_node.assert_called_once_with(cfg)
            thread.return_value.start.assert_called_once()

    def test_export_reads_old_raw_and_new_compressed_mcap(self):
        # Real serialization + rosbag2 MCAP writer/reader, not mocked message payloads.
        for compressed in (False, True):
            with self.subTest(compressed=compressed), tempfile.TemporaryDirectory() as tmp:
                path = Path(tmp) / 'episode'
                writer = rosbag2_py.SequentialWriter()
                writer.open(
                    rosbag2_py.StorageOptions(uri=str(path), storage_id='mcap'), rosbag2_py.ConverterOptions('', '')
                )
                camera = '/camera/image' + ('/compressed' if compressed else '')
                raw = colour_image()
                image = (
                    CompressedImage(format='bgra8; jpeg compressed bgr8', data=jpeg_bytes(raw, 90))
                    if compressed
                    else raw
                )
                entries = [
                    (
                        '/robot/joint_states',
                        'sensor_msgs/msg/JointState',
                        JointState(name=[f'left_{j}' for j in JOINTS], position=[0.1] * 7),
                    ),
                    (
                        '/robot/end_effector/left_gripper/state',
                        'baxter_core_msgs/msg/EndEffectorState',
                        EndEffectorState(position=25.0),
                    ),
                    (camera, 'sensor_msgs/msg/CompressedImage' if compressed else 'sensor_msgs/msg/Image', image),
                ]
                for index, (topic, kind, msg) in enumerate(entries):
                    writer.create_topic(
                        rosbag2_py.TopicMetadata(id=index, name=topic, type=kind, serialization_format='cdr')
                    )
                    writer.write(topic, serialize_message(msg), 1_000_000_000)
                del writer  # finalize MCAP and metadata before reading
                frames = read_frames(path, 'left', {'/camera/image': 'left'}, np.array([1_100_000_000]))
                cameras, joints, grip, age = frames[0]
                np.testing.assert_allclose(cameras['left'][0, 0], [255, 0, 0], atol=3)
                np.testing.assert_allclose(joints, [0.1] * 7)
                self.assertEqual(grip, 0.75)
                self.assertAlmostEqual(age, 0.1)
                # An explicit compressed camera name must work, too.
                if compressed:
                    self.assertIsNotNone(read_frames(path, 'left', {camera: 'left'}, np.array([1_100_000_000]))[0])
                with self.assertRaisesRegex(ValueError, 'missing camera'):
                    read_frames(path, 'left', {'/missing/image': 'missing'}, np.array([1_100_000_000]))


if __name__ == '__main__':
    unittest.main()
