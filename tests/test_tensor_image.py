"""inference.tensor_image and the policy's recorded encoding: the z published with each plan is the
encoder's output, bit for bit, and reads back as the same float32 array."""

import types

import numpy as np
import pytest

pytest.importorskip('sensor_msgs')
from builtin_interfaces.msg import Time  # noqa: E402

from inference import tensor_image  # noqa: E402
from inference.policies import record_encoding  # noqa: E402


def test_an_array_round_trips_bit_for_bit_with_its_stamp():
    z = np.random.default_rng(0).normal(size=4130).astype(np.float32)
    msg = tensor_image.to_image(z, Time(sec=12, nanosec=345))
    assert (msg.height, msg.width, msg.encoding, msg.step) == (1, 4130, '32FC1', 4 * 4130)
    assert (msg.header.stamp.sec, msg.header.stamp.nanosec) == (12, 345)
    np.testing.assert_array_equal(tensor_image.from_image(msg), z)


def test_an_image_that_is_not_a_float_row_is_refused():
    msg = tensor_image.to_image(np.zeros(3), Time())
    msg.encoding = 'rgb8'
    with pytest.raises(ValueError):
        tensor_image.from_image(msg)


class Encoder:
    """Stands in for the MultiTask DiT's observation encoder: returns the batch, a (B, d) tensor."""

    def encode(self, batch):
        return batch


def test_the_policy_keeps_the_encoders_output_of_its_last_plan():
    torch = pytest.importorskip('torch')
    encoder, sink = Encoder(), types.SimpleNamespace(z=None)
    record_encoding(encoder, sink)
    batch = torch.arange(6, dtype=torch.float16).reshape(2, 3)
    out = encoder.encode(batch)
    assert out is batch  # the policy still gets exactly what the encoder returned
    np.testing.assert_array_equal(sink.z, np.array([0.0, 1.0, 2.0], np.float32))
    assert sink.z.dtype == np.float32
