#!/usr/bin/env python3
"""Offline inpainting: two-pass to fit on the RTX 500 (3.8 GB VRAM).

Pass 1: load SAM3, segment every frame, store dilated masks on CPU, unload SAM3.
Pass 2: load LaMa, fill masked regions, write inpainted + mask into a new mcap
        alongside the original episode bag.

The output topics:
  /cameras/third_eye/color/image_raw_inpainted/compressed   (JPEG)
  /cameras/third_eye/color/image_raw_inpainted_mask/compressed  (PNG)

Usage (from rethought_ai, sourced ROS):
  ../ros2_inpaint/.venv/bin/python scripts/offline_inpaint_session.py SESSION_DIR

Runs fully offline (HF_HUB_OFFLINE=1).
"""

import argparse
import gc
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import rosbag2_py
import torch
import torch.nn.functional as F
from rclpy.serialization import deserialize_message, serialize_message
from rosidl_runtime_py.utilities import get_message
from sensor_msgs.msg import CompressedImage

os.environ['HF_HUB_OFFLINE'] = '1'

RAW_TOPIC = '/cameras/third_eye/color/image_raw/compressed'
OUT_TOPIC = '/cameras/third_eye/color/image_raw_inpainted/compressed'
MASK_TOPIC = '/cameras/third_eye/color/image_raw_inpainted_mask/compressed'
MSG_TYPE = 'sensor_msgs/msg/CompressedImage'
DILATION = 6


def decode_compressed(msg):
    buf = np.frombuffer(bytes(msg.data), np.uint8)
    bgr = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    if bgr is None:
        return None
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def encode_jpeg(rgb, quality=90):
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    ok, buf = cv2.imencode('.jpg', bgr, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return bytes(buf) if ok else None


def encode_png_mask(mask_bool):
    m = mask_bool.astype(np.uint8) * 255
    ok, buf = cv2.imencode('.png', m)
    return bytes(buf) if ok else None


def dilate(mask, radius, device):
    if radius <= 0:
        return mask
    r = torch.arange(-radius, radius + 1, device=device)
    disk = ((r[:, None] ** 2 + r[None, :] ** 2) <= radius**2).float()[None, None]
    return F.conv2d(mask[None, None].float(), disk, padding=radius)[0, 0] > 0


def read_raw_frames(episode_dir):
    reader = rosbag2_py.SequentialReader()
    reader.open(
        rosbag2_py.StorageOptions(uri=str(episode_dir), storage_id='mcap'),
        rosbag2_py.ConverterOptions('', ''),
    )
    types = {x.name: x.type for x in reader.get_all_topics_and_types()}
    if RAW_TOPIC not in types:
        return [], types
    if OUT_TOPIC in types:
        return None, types

    reader.set_filter(rosbag2_py.StorageFilter(topics=[RAW_TOPIC]))
    frames = []
    while reader.has_next():
        topic, raw, stamp = reader.read_next()
        msg = deserialize_message(raw, get_message(types[topic]))
        frames.append((stamp, msg))
    del reader
    return frames, types


def segment_all(frames, device):
    """Load SAM3, segment every frame, return dilated masks (CPU numpy). Unloads SAM3 after."""
    from ros2_inpaint import Sam3Masker
    from ros2_inpaint.temporal import SpeckleFilter

    print('    loading SAM3...', flush=True)
    masker = Sam3Masker('person', device=device, compile=False)
    speckle = SpeckleFilter(0.02)
    masks = []
    t0 = time.perf_counter()
    for i, (stamp, msg) in enumerate(frames):
        rgb = decode_compressed(msg)
        if rgb is None:
            masks.append(None)
            continue
        x = torch.from_numpy(rgb).to(device)
        raw_mask = masker(x)
        raw_mask = dilate(raw_mask, DILATION, device)
        raw_mask = speckle(raw_mask)
        masks.append(raw_mask.cpu().numpy())
        del x, raw_mask
        if (i + 1) % 50 == 0:
            elapsed = time.perf_counter() - t0
            hz = (i + 1) / elapsed
            print(f'    segmented {i + 1}/{len(frames)} ({hz:.1f} Hz)', flush=True)

    elapsed = time.perf_counter() - t0
    print(f'    segmented {len(frames)} frames in {elapsed:.0f}s ({len(frames) / elapsed:.1f} Hz)', flush=True)

    del masker, speckle
    gc.collect()
    torch.cuda.empty_cache()
    return masks


def fill_all(frames, masks, device, jpeg_quality=90):
    """Load LaMa, fill every masked frame, return encoded results. Unloads LaMa after."""
    from ros2_inpaint import LamaFiller

    print('    loading LaMa...', flush=True)
    filler = LamaFiller.from_pretrained(device)
    results = []
    t0 = time.perf_counter()
    for i, ((stamp, msg), mask) in enumerate(zip(frames, masks)):
        if mask is None:
            continue
        rgb = decode_compressed(msg)
        if rgb is None:
            continue
        mask_any = mask.any()
        if mask_any:
            x = torch.from_numpy(rgb).to(device)
            m = torch.from_numpy(mask).to(device)
            filled = filler(x, m)
            out = torch.where(m[..., None], filled, x).cpu().numpy()
            del x, m, filled
        else:
            out = rgb

        out_data = encode_jpeg(out, jpeg_quality)
        mask_data = encode_png_mask(mask)
        if out_data and mask_data:
            results.append((stamp, msg.header, out_data, mask_data))

        if (i + 1) % 100 == 0:
            elapsed = time.perf_counter() - t0
            hz = (i + 1) / elapsed
            print(f'    filled {i + 1}/{len(frames)} ({hz:.1f} Hz)', flush=True)

    elapsed = time.perf_counter() - t0
    print(f'    filled {len(frames)} frames in {elapsed:.0f}s ({len(frames) / elapsed:.1f} Hz)', flush=True)

    del filler
    gc.collect()
    torch.cuda.empty_cache()
    return results


def write_results(episode_dir, results):
    """Merge inpainted frames into a single mcap with all original data.

    rosbag2_py SequentialReader reads multi-file bags sequentially (file 1
    then file 2), not interleaved by timestamp, so a separate inpainted mcap
    would make iter_frames miss the inpainted camera entirely.  Writing one
    merged file avoids that.
    """
    import heapq
    import shutil
    import tempfile

    import yaml
    from mcap.reader import make_reader
    from mcap.writer import Writer

    episode_dir = Path(episode_dir)
    merged_name = f'{episode_dir.name}_merged_0.mcap'
    final_mcap = episode_dir / merged_name

    if final_mcap.exists():
        print(f'  skip write: {merged_name} already exists')
        return final_mcap

    orig_mcap = episode_dir / f'{episode_dir.name}_0.mcap'
    if not orig_mcap.exists():
        candidates = sorted(episode_dir.glob('*.mcap'))
        if not candidates:
            print('  error: no original mcap found')
            return None
        orig_mcap = candidates[0]

    orig_msgs = []
    img_schema = None
    with open(orig_mcap, 'rb') as f:
        reader = make_reader(f)
        summary = reader.get_summary()
        for ch in summary.channels.values():
            s = summary.schemas[ch.schema_id]
            if s.name == MSG_TYPE:
                img_schema = (s.encoding, s.data, ch.message_encoding)
                break
        for schema, channel, message in reader.iter_messages():
            orig_msgs.append((
                message.log_time,
                message.publish_time,
                channel.topic,
                channel.message_encoding,
                schema.name,
                schema.encoding,
                schema.data,
                message.data,
                message.sequence,
            ))

    s_enc, s_data, m_enc = img_schema or ('ros2msg', b'', 'cdr')

    inpainted_msgs = []
    for stamp, header, out_data, mask_data in results:
        out_msg = CompressedImage()
        out_msg.header.stamp = header.stamp
        out_msg.header.frame_id = header.frame_id
        out_msg.format = 'jpeg'
        out_msg.data = list(out_data)

        mask_msg = CompressedImage()
        mask_msg.header.stamp = header.stamp
        mask_msg.header.frame_id = header.frame_id
        mask_msg.format = 'png'
        mask_msg.data = list(mask_data)

        out_raw = serialize_message(out_msg)
        mask_raw = serialize_message(mask_msg)
        inpainted_msgs.append((stamp, stamp, OUT_TOPIC, m_enc, MSG_TYPE, s_enc, s_data, out_raw, 0))
        inpainted_msgs.append((stamp, stamp, MASK_TOPIC, m_enc, MSG_TYPE, s_enc, s_data, mask_raw, 0))

    all_msgs = sorted(heapq.merge(orig_msgs, inpainted_msgs, key=lambda x: x[0]), key=lambda x: x[0])

    with tempfile.NamedTemporaryFile(dir=episode_dir, suffix='.mcap', delete=False) as tmp:
        tmp_path = Path(tmp.name)
        writer = Writer(tmp)
        writer.start()
        schema_map = {}
        channel_map = {}
        for log_time, pub_time, topic, msg_enc, schema_name, schema_enc, schema_data, data, seq in all_msgs:
            sk = (schema_name, schema_enc)
            if sk not in schema_map:
                schema_map[sk] = writer.register_schema(name=schema_name, encoding=schema_enc, data=schema_data)
            if topic not in channel_map:
                channel_map[topic] = writer.register_channel(
                    topic=topic, message_encoding=msg_enc, schema_id=schema_map[sk]
                )
            writer.add_message(
                channel_id=channel_map[topic], log_time=log_time, publish_time=pub_time, data=data, sequence=seq
            )
        writer.finish()
    shutil.move(str(tmp_path), str(final_mcap))

    metadata_path = episode_dir / 'metadata.yaml'
    if metadata_path.exists():
        meta = yaml.safe_load(metadata_path.read_text())
        info = meta['rosbag2_bagfile_information']
        info['relative_file_paths'] = [merged_name]
        info['message_count'] += 2 * len(results)
        info['files'] = [
            {
                'path': merged_name,
                'starting_time': {'nanoseconds_since_epoch': all_msgs[0][0]},
                'duration': {'nanoseconds': all_msgs[-1][0] - all_msgs[0][0]},
                'message_count': len(all_msgs),
            }
        ]
        for tp, count in [(OUT_TOPIC, len(results)), (MASK_TOPIC, len(results))]:
            info['topics_with_message_count'].append({
                'topic_metadata': {
                    'name': tp,
                    'type': MSG_TYPE,
                    'serialization_format': 'cdr',
                    'offered_qos_profiles': [],
                    'type_description_hash': '',
                },
                'message_count': count,
            })
        metadata_path.write_text(yaml.dump(meta, default_flow_style=False, sort_keys=False))

    return final_mcap


def process_episode(episode_dir, device):
    episode_dir = Path(episode_dir)
    frames, types = read_raw_frames(episode_dir)
    if frames is None:
        print(f'  skip: {OUT_TOPIC} already exists')
        return 0
    if not frames:
        print(f'  skip: no {RAW_TOPIC}')
        return 0

    print(f'  {len(frames)} raw frames', flush=True)

    masks = segment_all(frames, device)
    results = fill_all(frames, masks, device)
    del masks

    out_mcap = write_results(episode_dir, results)
    dur = frames[-1][0] - frames[0][0]
    hz = len(results) / (dur / 1e9) if dur > 0 else 0
    print(f'  wrote {len(results)} inpainted frames ({hz:.1f} Hz) to {out_mcap.name}')
    return len(results)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('session', help='Session directory with episode_* subdirs')
    p.add_argument('--device', default='cuda')
    a = p.parse_args()

    session = Path(a.session)
    episodes = sorted(session.glob('episode_*'))
    if not episodes:
        print(f'No episodes in {session}')
        sys.exit(1)

    print(f'Processing {len(episodes)} episodes in {session.name}')
    total = 0
    for ep in episodes:
        print(f'{ep.name}:')
        total += process_episode(ep, a.device)
    print(f'\nDone: {total} inpainted frames across {len(episodes)} episodes')


if __name__ == '__main__':
    main()
