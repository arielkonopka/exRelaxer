"""8-bit weights and outputs (doc/precision.md)."""
import numpy as np

import exrelaxer as exr


def build(precision):
    exr.reseed(2)
    net = exr.Network()
    spec = exr.LayerSpec.dense(8, False, True)
    spec.precision = precision
    hid = net.add_layer("h", spec)
    net.add_inputs(hid, 4)
    net.add_output(hid)
    return net, hid


def test_outputs_and_weights_are_eight_bit_and_saved():
    net, hid = build(exr.Precision.int8())
    rng = np.random.default_rng(0)
    for _ in range(100):
        net.set_inputs(rng.uniform(-1, 1, 4).astype(np.float32))
        net.step()
        net.apply_reward(float(rng.uniform(-1, 1)), 0.01)
    codes = net.outputs() * 16
    assert np.array_equal(codes, np.round(codes)) and np.abs(codes).max() <= 127
    loaded = exr.Network.from_bytes(net.to_bytes())
    assert loaded.layer_spec(hid).precision == exr.Precision.int8()


def test_default_is_float():
    assert exr.LayerSpec.dense(4).precision == exr.Precision()
    p = exr.Precision(int8_weights=True)
    assert p.int8_weights and not p.int8_values and p.value_step == 1 / 16
