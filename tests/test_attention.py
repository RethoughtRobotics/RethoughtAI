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
