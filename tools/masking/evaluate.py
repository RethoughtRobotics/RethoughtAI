"""Offline pretrained segmentation review. Does not export datasets or use ROS."""

import argparse
import hashlib
import html
import io
import json
import subprocess
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2) + '\n')


def file_sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def fresh_directory(path):
    if path.exists():
        raise ValueError(f'Output already exists; choose a new directory: {path}')
    path.mkdir(parents=True)


def sample_indices(length, count):
    if length < 1 or count < 1:
        raise ValueError('Frame count and sample count must be positive')
    return np.unique(np.linspace(0, length - 1, min(count, length), dtype=int)).tolist()


def extract(args):
    import pyarrow.parquet as pq

    dataset = args.dataset.resolve()
    info = json.loads((dataset / 'meta/info.json').read_text())
    cameras = args.cameras or [k for k, v in info['features'].items() if v['dtype'] == 'video']
    episode_files = sorted(dataset.glob('meta/episodes/**/*.parquet'))
    if not episode_files:
        raise ValueError('Expected LeRobot v3 episode parquet metadata')
    episodes = [r for f in episode_files for r in pq.read_table(f).to_pylist()]
    selected = [r for r in episodes if args.episodes is None or r['episode_index'] in args.episodes]
    selected = selected[: args.max_episodes]
    if not selected:
        raise ValueError('No requested episodes found')
    fresh_directory(args.out)
    rows = []
    for episode in selected:
        data_path = (
            dataset / f'data/chunk-{episode["data/chunk_index"]:03d}/file-{episode["data/file_index"]:03d}.parquet'
        )
        data = pq.read_table(data_path, columns=['episode_index', 'frame_index', 'timestamp', 'index']).to_pylist()
        frames = sorted(
            (r for r in data if r['episode_index'] == episode['episode_index']), key=lambda r: r['frame_index']
        )
        if len(frames) != episode['length']:
            raise ValueError('Episode length differs from data; cannot guarantee alignment')
        for i in sample_indices(len(frames), args.samples):
            frame = frames[i]
            for camera in cameras:
                prefix = f'videos/{camera}'
                video = (
                    dataset
                    / f'{prefix}/chunk-{episode[prefix + "/chunk_index"]:03d}/file-{episode[prefix + "/file_index"]:03d}.mp4'
                )
                offset = episode[prefix + '/from_timestamp'] + frame['timestamp']
                if offset >= episode[prefix + '/to_timestamp']:
                    raise ValueError('Frame exceeds episode video interval')
                raw = subprocess.check_output([
                    'ffmpeg',
                    '-v',
                    'error',
                    '-threads',
                    '1',
                    '-c:v',
                    args.decoder,
                    '-ss',
                    str(offset),
                    '-i',
                    str(video),
                    '-frames:v',
                    '1',
                    '-f',
                    'image2pipe',
                    '-vcodec',
                    'png',
                    'pipe:1',
                ])
                with Image.open(io.BytesIO(raw)) as decoded:
                    image = decoded.convert('RGB')
                name = f'ep{frame["episode_index"]:04d}_f{frame["frame_index"]:06d}_{camera.removeprefix("observation.images.")}.png'
                image.save(args.out / name)
                rows.append({
                    **frame,
                    'camera': camera,
                    'image': name,
                    'video': str(video.relative_to(dataset)),
                    'requested_video_time_s': offset,
                    'rgb_sha256': hashlib.sha256(image.tobytes()).hexdigest(),
                })
    write_json(
        args.out / 'manifest.json',
        {
            'dataset': str(dataset),
            'fps': info['fps'],
            'sampling': 'Episode frame timestamps + per-camera video offsets; FFmpeg seek, not exact training decoder parity',
            'frames': rows,
        },
    )
    print(f'Extracted {len(rows)} images to {args.out}')


def checked_mask(mask, shape):
    mask = np.asarray(mask)
    if mask.shape != shape:
        raise ValueError(f'Mask shape {mask.shape} != original image {shape}')
    return mask.astype(bool)


def panels(image, mask):
    rgb = np.asarray(image)
    mask = checked_mask(mask, rgb.shape[:2])
    overlay = rgb.copy()
    overlay[mask] = np.round(0.5 * rgb[mask] + 0.5 * np.array([255, 0, 180])).astype(np.uint8)
    removed = rgb.copy()
    removed[mask] = 127
    canvas = Image.new('RGB', (image.width * 3, image.height + 24), '#161b22')
    draw = ImageDraw.Draw(canvas)
    for i, (pixels, title) in enumerate([
        (rgb, 'Original'),
        (overlay, 'Predicted removal'),
        (removed, 'Neutral fill preview'),
    ]):
        canvas.paste(Image.fromarray(pixels), (image.width * i, 24))
        draw.text((image.width * i + 5, 4), title, fill='white')
    return canvas


class DeepLab:
    """Ungated semantic person baseline, not an isolated-hand specialist."""

    def __init__(self, device):
        import torch
        import torchvision
        from torchvision.models.segmentation import DeepLabV3_ResNet50_Weights, deeplabv3_resnet50

        self.torch = torch
        self.device = device
        weights = DeepLabV3_ResNet50_Weights.COCO_WITH_VOC_LABELS_V1
        self.model = deeplabv3_resnet50(weights=weights).eval().to(device)
        self.transform = weights.transforms()
        self.person = weights.meta['categories'].index('person')
        self.meta = {
            'backend': 'deeplab',
            'weights': str(weights),
            'torch': torch.__version__,
            'torchvision': torchvision.__version__,
            'device': device,
            'weights_url': weights.url,
            'checkpoint_sha256': file_sha256(Path(torch.hub.get_dir()) / 'checkpoints' / Path(weights.url).name),
        }

    def predict(self, image, prompts, threshold):
        torch = self.torch
        with torch.inference_mode():
            output = self.model(self.transform(image).unsqueeze(0).to(self.device))['out']
            output = torch.nn.functional.interpolate(
                output, size=(image.height, image.width), mode='bilinear', align_corners=False
            )
            probabilities = output.softmax(1)[0, self.person].cpu().numpy()
        return [('person', probabilities >= threshold, {'max_person_probability': float(probabilities.max())})]


class MaskRCNN:
    """Ungated instance segmentation baseline for person class."""

    def __init__(self, device):
        import torch
        import torchvision
        from torchvision.models.detection import MaskRCNN_ResNet50_FPN_V2_Weights, maskrcnn_resnet50_fpn_v2

        self.torch = torch
        self.device = device
        weights = MaskRCNN_ResNet50_FPN_V2_Weights.COCO_V1
        self.model = maskrcnn_resnet50_fpn_v2(weights=weights).eval().to(device)
        self.transform = weights.transforms()
        self.person = weights.meta['categories'].index('person')
        self.meta = {
            'backend': 'maskrcnn',
            'weights': str(weights),
            'torch': torch.__version__,
            'torchvision': torchvision.__version__,
            'device': device,
            'weights_url': weights.url,
            'checkpoint_sha256': file_sha256(Path(torch.hub.get_dir()) / 'checkpoints' / Path(weights.url).name),
        }

    def predict(self, image, prompts, threshold):
        torch = self.torch
        with torch.inference_mode():
            batch = self.transform(image).unsqueeze(0).to(self.device)
            output = self.model(batch)[0]
            labels = output['labels'].detach().cpu().numpy()
            scores = output['scores'].detach().cpu().numpy()
            masks = (
                output['masks'].detach().cpu().numpy()[:, 0]
                if len(output['masks'])
                else np.empty((0, image.height, image.width))
            )
        keep = (labels == self.person) & (scores >= threshold)
        selected = masks[keep] if keep.any() else np.empty((0, image.height, image.width))
        union = selected >= 0.5
        combined = union.any(0) if len(union) else np.zeros((image.height, image.width), dtype=bool)
        return [
            (
                'person',
                combined,
                {
                    'person_instances': int(keep.sum()),
                    'max_person_score': float(scores[keep].max()) if keep.any() else 0.0,
                },
            )
        ]


class SAM3:
    def __init__(self, device, checkpoint, threshold):
        import torch
        from sam3.model.sam3_image_processor import Sam3Processor
        from sam3.model_builder import build_sam3_image_model

        if checkpoint is None or not checkpoint.is_file():
            raise ValueError('SAM3 requires --checkpoint pointing to authorized pretrained sam3.pt')
        self.torch = torch
        self.device = device
        model = build_sam3_image_model(device=device, checkpoint_path=str(checkpoint), load_from_HF=False)
        self.processor = Sam3Processor(model, device=device, confidence_threshold=threshold)
        self.meta = {
            'backend': 'sam3',
            'checkpoint': str(checkpoint.resolve()),
            'checkpoint_sha256': file_sha256(checkpoint),
            'torch': torch.__version__,
            'device': device,
        }

    def predict(self, image, prompts, threshold):
        torch = self.torch
        outputs = []
        with torch.inference_mode():
            state = self.processor.set_image(image)
            for prompt in prompts:
                self.processor.reset_all_prompts(state)
                result = self.processor.set_text_prompt(prompt=prompt, state=state)
                masks = result['masks'].detach().cpu().numpy()
                union = masks[:, 0].any(0) if len(masks) else np.zeros((image.height, image.width), dtype=bool)
                outputs.append((prompt, union, {'instance_scores': result['scores'].detach().cpu().tolist()}))
        return outputs


def run(args):
    import torch

    torch.set_num_threads(args.threads)
    if not 0 <= args.threshold <= 1:
        raise ValueError('Threshold must be between zero and one')
    manifest = json.loads((args.samples_dir / 'manifest.json').read_text())
    if args.out.resolve() == args.samples_dir.resolve():
        raise ValueError('Results must be separate from samples')
    if args.out.exists():
        raise ValueError('Choose a new output directory; existing runs are never overwritten')
    if args.backend == 'deeplab':
        model = DeepLab(args.device)
    elif args.backend == 'maskrcnn':
        model = MaskRCNN(args.device)
    else:
        model = SAM3(args.device, args.checkpoint, args.threshold)
    fresh_directory(args.out)
    prompts = args.prompts or ['person', 'human hand', 'human forearm']
    rows = []
    cards = []
    for row in manifest['frames']:
        with Image.open(args.samples_dir / row['image']) as source:
            image = source.convert('RGB')
        actual_hash = hashlib.sha256(image.tobytes()).hexdigest()
        if actual_hash != row['rgb_sha256']:
            raise ValueError(f'Sample content changed since extraction: {row["image"]}')
        started = time.perf_counter()
        outputs = model.predict(image, prompts, args.threshold)
        elapsed = time.perf_counter() - started
        union = np.zeros((image.height, image.width), dtype=bool)
        details = []
        stem = Path(row['image']).stem
        for i, (prompt, mask, confidence) in enumerate(outputs):
            mask = checked_mask(mask, union.shape)
            union |= mask
            filename = f'{stem}_prompt{i}.png'
            Image.fromarray(mask.astype(np.uint8) * 255).save(args.out / filename)
            details.append({'prompt': prompt, 'mask': filename, 'fraction': float(mask.mean()), **confidence})
        mask_name, preview = f'{stem}_mask.png', f'{stem}_preview.jpg'
        Image.fromarray(union.astype(np.uint8) * 255).save(args.out / mask_name)
        panels(image, union).save(args.out / preview, quality=90)
        record = {
            **row,
            'mask': mask_name,
            'preview': preview,
            'mask_fraction': float(union.mean()),
            'inference_s': elapsed,
            'prompts': details,
            'review_status': 'unreviewed',
        }
        rows.append(record)
        cards.append(
            f'<h3>{html.escape(stem)}</h3><p>removed {union.mean():.1%}; {elapsed:.2f}s</p><img loading="lazy" src="{preview}" width="100%">'
        )
        write_json(
            args.out / 'results.json',
            {
                'model': model.meta,
                'threshold': args.threshold,
                'threshold_meaning': 'person pixel probability'
                if args.backend == 'deeplab'
                else 'instance confidence; mask probability fixed at 0.5',
                'sample_manifest': str((args.samples_dir / 'manifest.json').resolve()),
                'frames': rows,
            },
        )
        print(stem, f'mask={union.mean():.1%}', flush=True)
    (args.out / 'index.html').write_text(
        '<!doctype html><meta charset="utf-8"><title>Mask review</title>'
        '<body style="background:#161b22;color:white;font-family:sans-serif"><h1>Unreviewed pretrained masks</h1>'
        '<p>Inspect missed hands and false removal of robot/cup/container. Area and confidence are not accuracy.</p>'
        + '\n'.join(cards)
    )


def score(args):
    result = json.loads((args.run / 'results.json').read_text())
    rows = []
    for frame in result['frames']:
        truth = args.truth / frame['mask']
        if not truth.exists():
            continue
        with Image.open(args.run / frame['mask']) as image:
            prediction = np.asarray(image.convert('L')) > 0
        with Image.open(truth) as image:
            target = checked_mask(np.asarray(image.convert('L')) > 0, prediction.shape)
        tp, fp, fn = (
            int((prediction & target).sum()),
            int((prediction & ~target).sum()),
            int((~prediction & target).sum()),
        )
        rows.append({
            'image': frame['image'],
            'camera': frame['camera'],
            'tp': tp,
            'fp': fp,
            'fn': fn,
            'iou': tp / (tp + fp + fn) if tp + fp + fn else 1.0,
        })
    if not rows:
        raise ValueError('No matching manually labelled truth masks')
    tp, fp, fn = (sum(r[k] for r in rows) for k in ['tp', 'fp', 'fn'])
    per_camera = {}
    for camera in sorted({r['camera'] for r in rows}):
        subset = [r for r in rows if r['camera'] == camera]
        ct, cf, cn = (sum(r[k] for r in subset) for k in ['tp', 'fp', 'fn'])
        per_camera[camera] = {
            'labelled_frames': len(subset),
            'micro_iou': ct / (ct + cf + cn) if ct + cf + cn else 1.0,
            'pixel_precision': ct / (ct + cf) if ct + cf else None,
            'pixel_recall': ct / (ct + cn) if ct + cn else None,
        }
    report = {
        'labelled_frames': len(rows),
        'total_run_frames': len(result['frames']),
        'micro_iou': tp / (tp + fp + fn) if tp + fp + fn else 1.0,
        'pixel_precision': tp / (tp + fp) if tp + fp else None,
        'pixel_recall': tp / (tp + fn) if tp + fn else None,
        'per_camera': per_camera,
        'frames': rows,
    }
    if args.out.exists():
        raise ValueError('Metrics output already exists')
    write_json(args.out, report)
    print(json.dumps({k: v for k, v in report.items() if k != 'frames'}, indent=2))


def contact_sheet(args):
    """Compact overview; full-resolution previews remain the source for boundary review."""
    result = json.loads((args.run / 'results.json').read_text())
    frames = result['frames']
    if not frames or args.out.exists():
        raise ValueError('Need at least one result and a new output path')
    width, height = 960, 270
    canvas = Image.new('RGB', (width, height * len(frames)), '#161b22')
    draw = ImageDraw.Draw(canvas)
    for index, row in enumerate(frames):
        with Image.open(args.run / row['preview']) as image:
            image.thumbnail((width, height - 30))
            canvas.paste(image, (0, height * index + 30))
        draw.text(
            (5, height * index + 5),
            f'{row["image"]} removal={row["mask_fraction"]:.1%} — UNREVIEWED prediction',
            fill='white',
        )
    canvas.save(args.out)
    print(f'Saved {len(frames)} preview rows to {args.out}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    extract_parser = sub.add_parser('extract')
    extract_parser.add_argument('--dataset', type=Path, required=True)
    extract_parser.add_argument('--out', type=Path, required=True)
    extract_parser.add_argument('--episodes', type=int, nargs='+')
    extract_parser.add_argument('--cameras', nargs='+')
    extract_parser.add_argument('--max-episodes', type=int, default=3)
    extract_parser.add_argument('--samples', type=int, default=6)
    extract_parser.add_argument('--decoder', default='libdav1d')
    extract_parser.set_defaults(func=extract)
    run_parser = sub.add_parser('run')
    run_parser.add_argument('--samples-dir', type=Path, required=True)
    run_parser.add_argument('--out', type=Path, required=True)
    run_parser.add_argument('--backend', choices=['sam3', 'deeplab', 'maskrcnn'], required=True)
    run_parser.add_argument('--device', choices=['cpu', 'cuda'], default='cpu')
    run_parser.add_argument('--checkpoint', type=Path)
    run_parser.add_argument('--threshold', type=float, default=0.5)
    run_parser.add_argument('--threads', type=int, default=4)
    run_parser.add_argument('--prompts', nargs='+')
    run_parser.set_defaults(func=run)
    score_parser = sub.add_parser('score')
    score_parser.add_argument('--run', type=Path, required=True)
    score_parser.add_argument('--truth', type=Path, required=True)
    score_parser.add_argument('--out', type=Path, required=True)
    score_parser.set_defaults(func=score)
    sheet_parser = sub.add_parser('contact-sheet')
    sheet_parser.add_argument('--run', type=Path, required=True)
    sheet_parser.add_argument('--out', type=Path, required=True)
    sheet_parser.set_defaults(func=contact_sheet)
    args = parser.parse_args()
    if args.command == 'extract' and (args.samples < 1 or args.max_episodes < 1):
        parser.error('Sample and episode limits must be positive')
    if args.command == 'run' and args.threads < 1:
        parser.error('Thread limit must be positive')
    args.func(args)


if __name__ == '__main__':
    main()
