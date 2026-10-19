"""Offline mask utilities; no model downloads, ROS or GPU required."""

import hashlib
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

spec = importlib.util.spec_from_file_location(
    'masking_evaluate', Path(__file__).parents[1] / 'tools/masking/evaluate.py'
)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_sampling():
    assert module.sample_indices(3, 6) == [0, 1, 2]
    assert module.sample_indices(10, 3) == [0, 4, 9]
    with pytest.raises(ValueError):
        module.sample_indices(0, 3)


def test_no_overwrite(tmp_path):
    with pytest.raises(ValueError):
        module.fresh_directory(tmp_path)


def test_mask_and_preview():
    rgb = np.full((4, 5, 3), 220, dtype=np.uint8)
    original = rgb.copy()
    mask = np.zeros((4, 5), dtype=bool)
    mask[1, 2] = True
    preview = np.asarray(module.panels(Image.fromarray(rgb), mask))
    assert np.array_equal(rgb, original)
    assert np.array_equal(preview[24:, :5], original)
    assert np.array_equal(preview[25, 12], [127, 127, 127])
    with pytest.raises(ValueError):
        module.checked_mask(np.zeros((5, 4)), (4, 5))


def test_truth_metrics(tmp_path):
    run, truth = tmp_path / 'run', tmp_path / 'truth'
    run.mkdir()
    truth.mkdir()
    module.write_json(run / 'results.json', {'frames': [{'image': 'frame.png', 'camera': 'left', 'mask': 'mask.png'}]})
    Image.fromarray(np.array([[255, 255], [0, 0]], dtype=np.uint8)).save(run / 'mask.png')
    Image.fromarray(np.array([[255, 0], [255, 0]], dtype=np.uint8)).save(truth / 'mask.png')
    out = tmp_path / 'metrics.json'
    module.score(SimpleNamespace(run=run, truth=truth, out=out))
    result = json.loads(out.read_text())
    assert result['micro_iou'] == pytest.approx(1 / 3)
    assert result['pixel_precision'] == 0.5
    assert result['pixel_recall'] == 0.5
    assert result['per_camera']['left']['micro_iou'] == pytest.approx(1 / 3)
    with pytest.raises(ValueError):
        module.score(SimpleNamespace(run=run, truth=truth, out=out))


def test_run_writes_review_artifacts_and_preserves_samples(tmp_path, monkeypatch):
    samples = tmp_path / 'samples'
    samples.mkdir()
    image = Image.new('RGB', (8, 6), (100, 150, 200))
    image.save(samples / 'frame.png')
    digest = hashlib.sha256(image.tobytes()).hexdigest()
    module.write_json(
        samples / 'manifest.json',
        {'frames': [{'image': 'frame.png', 'camera': 'observation.images.left', 'rgb_sha256': digest}]},
    )

    class FakeModel:
        meta = {'backend': 'test'}

        def __init__(self, device):
            pass

        def predict(self, image, prompts, threshold):
            mask = np.zeros((image.height, image.width), dtype=bool)
            mask[0, 0] = True
            return [('person', mask, {})]

    monkeypatch.setattr(module, 'DeepLab', FakeModel)
    args = SimpleNamespace(
        samples_dir=samples,
        out=tmp_path / 'result',
        backend='deeplab',
        device='cpu',
        threads=1,
        threshold=0.5,
        prompts=None,
    )
    module.run(args)
    results = json.loads((args.out / 'results.json').read_text())
    assert results['frames'][0]['mask_fraction'] == pytest.approx(1 / 48)
    assert (args.out / 'index.html').exists()
    assert (args.out / 'frame_preview.jpg').exists()
    sheet = tmp_path / 'sheet.jpg'
    module.contact_sheet(SimpleNamespace(run=args.out, out=sheet))
    with Image.open(sheet) as overview:
        assert overview.size == (960, 270)
    with pytest.raises(ValueError):
        module.contact_sheet(SimpleNamespace(run=args.out, out=sheet))
    with Image.open(samples / 'frame.png') as original:
        assert hashlib.sha256(original.tobytes()).hexdigest() == digest
    with pytest.raises(ValueError, match='never overwritten'):
        module.run(args)
    image.putpixel((0, 0), (1, 2, 3))
    image.save(samples / 'frame.png')
    args.out = tmp_path / 'changed'
    with pytest.raises(ValueError, match='Sample content changed'):
        module.run(args)
