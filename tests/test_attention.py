from types import SimpleNamespace

import numpy as np

from inference import attention


def test_frame_extent_of_a_padded_canvas():
    # SmolVLA: a 400 x 640 frame fits a 512 canvas as 320 x 512, padded 192 on top
    ratio = 640 / 512
    box = (-192 * ratio, 0.0, 320 * ratio, 512 * ratio)
    assert np.allclose(attention._to_frame(box, (400, 640), (400, 640)), (-0.6, 0.0, 1.0, 1.0))
    # letterboxed to a square input first: the frame sits in the middle band of it
    top, left, h, w = attention._letterbox(400, 640, 224, 224)
    assert (left, w) == (0, 224) and top == (224 - h) // 2


class LooksAtTopLeft:
    """A stand-in policy whose chunk moves with the brightness of the left camera's top-left corner."""

    type, features = 'toy', ['observation.images.left', 'observation.state']

    def reset(self):
        pass

    def __call__(self, observation, tcp=None):
        corner = observation['observation.images.left'][:20, :20].mean() / 255
        chunk = np.zeros((5, 7))
        chunk[:, 0] = corner
        return chunk


def test_occlusion_finds_the_region_the_policy_uses():
    frame = np.zeros((120, 160, 3), np.uint8)
    frame[:20, :20] = 255
    maps = attention.occlusion(
        LooksAtTopLeft(), {'observation.images.left': frame, 'observation.state': None}, tcp=None, grid=(6, 8)
    )
    position = maps.scores[0]
    assert maps.dims == ['target', 'token'] and position.shape == (48,)
    assert position.argmax() == 0 and np.count_nonzero(position) == 1
    assert maps.segments[0].grid == (6, 8) and maps.segments[0].extent == (0.0, 0.0, 1.0, 1.0)


def test_averaging_out_a_dim_keeps_the_rest_selectable():
    maps = attention.Maps(
        'attention',
        ['step', 'layer', 'head', 'token'],
        np.random.rand(2, 3, 4, 5).astype(np.float32),
        [],
        {'layer': ['a', 'b', 'c']},
    )
    small = maps.averaged(['head'])
    assert small.dims == ['step', 'layer', 'token'] and small.scores.shape == (2, 3, 5)
    assert np.allclose(small.scores, maps.scores.mean(axis=2)) and small.meta()['means'] == ['step', 'layer']
    occluded = attention.Maps('occlusion', ['target', 'token'], np.ones((3, 4)), [], means=[])
    assert occluded.averaged(['target']) is occluded  # units differ: never averaged


def test_cls_attention_is_clips_own_last_layer_attention():
    # a small random CLIP, run as the DiT runs it: 2 steps x 2 cameras, the current step's images last
    import torch
    from transformers import CLIPVisionConfig, CLIPVisionModel

    torch.manual_seed(0)
    config = CLIPVisionConfig(
        hidden_size=32, intermediate_size=64, num_hidden_layers=2, num_attention_heads=4, image_size=32, patch_size=8
    )
    config._attn_implementation = 'eager'
    clip = CLIPVisionModel(config).eval()
    encoder = SimpleNamespace(
        camera_names=['observation.images.left', 'observation.images.third_eye'],
        vision_encoder=SimpleNamespace(model=clip),
    )
    seen = attention.ClsAttention(encoder)
    assert seen.maps() == {}
    pixels = torch.randn(4, 3, 32, 32)
    with torch.no_grad():
        clip(pixel_values=pixels)
        own = clip(pixel_values=pixels, output_attentions=True).attentions[-1][-2:, :, 0, 1:].mean(1)
    maps = seen.maps()
    assert list(maps) == ['left', 'third_eye']
    for i, (weights, view) in enumerate(maps.values()):
        assert weights.shape == (4, 4) and 0 < weights.sum() < 1
        assert np.allclose(weights.ravel(), own[i].numpy(), atol=1e-5)
        assert view.shape == (32, 32, 3) and view.dtype == np.uint8
    image = attention.overlay(*maps['left'])
    assert image.shape == (32, 32, 3) and image.dtype == np.uint8


def test_cls_attention_pools_the_patch_tokens_as_the_sim_key_screen_does():
    # 2026-10-27-key-scope-sim score_keys.py: sum over patches of attention x tokens, entering (L11) and
    # leaving (L12) the last layer, per camera of the current step
    import torch
    from transformers import CLIPVisionConfig, CLIPVisionModel

    torch.manual_seed(1)
    config = CLIPVisionConfig(
        hidden_size=32, intermediate_size=64, num_hidden_layers=2, num_attention_heads=4, image_size=32, patch_size=8
    )
    config._attn_implementation = 'eager'
    clip = CLIPVisionModel(config).eval()
    encoder = SimpleNamespace(
        camera_names=['observation.images.left', 'observation.images.third_eye'],
        vision_encoder=SimpleNamespace(model=clip),
    )
    seen = attention.ClsAttention(encoder)
    assert seen.pooled() == {}
    pixels = torch.randn(4, 3, 32, 32)
    with torch.no_grad():
        clip(pixel_values=pixels)
        own = clip(pixel_values=pixels, output_attentions=True, output_hidden_states=True)
    w = own.attentions[-1][-2:, :, 0, 1:].mean(1)
    parts = seen.pooled()
    assert list(parts) == ['left', 'third_eye']
    for i, camera in enumerate(parts.values()):
        assert set(camera) == {'weights', 'pooled_l11', 'pooled_l12'}
        assert all(v.dtype == np.float32 for v in camera.values())
        np.testing.assert_allclose(camera['weights'], w[i].numpy(), atol=1e-5)
        for name, hs in (('pooled_l11', own.hidden_states[-2]), ('pooled_l12', own.hidden_states[-1])):
            expected = (w[i][:, None] * hs[2 + i, 1:]).sum(0).numpy()
            np.testing.assert_allclose(camera[name], expected, atol=1e-5)
