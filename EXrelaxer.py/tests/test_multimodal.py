"""Multimodal and stereo inputs from Python: named input sources, a cochlea
with several microphones, Resize2D and Disparity."""
import numpy as np
import pytest

import exrelaxer as exr
from exrelaxer import audio


def test_named_sources_feed_layers_by_name():
    net = exr.Network()
    eye = net.add_layer("eye", exr.LayerSpec.retina(exr.RetinaSpec(exr.Shape(1, 2, 3)), False, False))
    both = net.add_layer("both", exr.LayerSpec.dense(1, False, False))
    ear = net.add_layer("ear", exr.LayerSpec.dense(2, False, False))
    assert net.add_inputs(eye, exr.Shape(1, 2, 3), name="camera") == 0
    assert net.add_inputs(ear, 4, name="microphone") == 6
    net.connect_inputs("camera", both)
    image = np.arange(6, dtype=np.float32).reshape(2, 3)
    net.set_inputs("camera", image)
    net.set_inputs("microphone", [1.0, 2.0, 3.0, 4.0])
    assert np.array_equal(net.input_values("camera"), image.reshape(-1))
    sources = net.input_sources
    assert [s["name"] for s in sources] == ["camera", "microphone"]
    assert sources[0]["targets"] == [eye, both]
    assert sources[0]["shape"] == exr.Shape(1, 2, 3)
    with pytest.raises(ValueError):
        net.set_inputs("camera", [1.0])
    with pytest.raises(IndexError):
        net.set_inputs("nose", [1.0])
    copy = exr.Network.from_bytes(net.to_bytes())
    assert copy.input_sources[0]["targets"] == [eye, both]
    assert np.array_equal(copy.input_values("microphone"), [1, 2, 3, 4])


def test_stereo_cochlea_hears_each_microphone():
    spec = exr.CochleaSpec(sample_rate=8000, hop=64, window=256, bands=16, min_frequency=100,
                           compression=exr.Compression.Linear, gain=1.0, channels=2)
    net = exr.Network()
    ear = net.add_layer("ears", exr.LayerSpec.cochlea(spec, False, False))
    net.add_inputs(ear, 2 * spec.hop, name="microphones")
    net.add_output(ear)
    assert net.layer_shape(ear) == exr.Shape(2, 16, 1)
    sound = np.stack([audio.tone(300, 0.2, 8000), audio.tone(2500, 0.2, 8000)])
    ys = net.run(audio.frames(sound, spec.hop))
    left, right = ys[-1].reshape(2, 16)
    bands = net.cochlea_bands(ear)
    assert bands[np.argmax(left)][0] <= 300 <= bands[np.argmax(left)][2]
    assert bands[np.argmax(right)][0] <= 2500 <= bands[np.argmax(right)][2]
    assert net.cochlea_power(ear).shape == (2, spec.window // 2 + 1)
    assert exr.Network.from_bytes(net.to_bytes()).layer_spec(ear).cochlea_spec.channels == 2


def test_resize_brings_vision_and_hearing_to_one_grid():
    net = exr.Network()
    eye = net.add_layer("eye", exr.LayerSpec.retina(exr.RetinaSpec(exr.Shape(1, 12, 16)), False, False))
    ear = net.add_layer("ear", exr.LayerSpec.retina(exr.RetinaSpec(exr.Shape(1, 20, 5)), False, False))
    small_eye = net.add_layer("small_eye", exr.LayerSpec.resize2d(8, 8, exr.Interpolation.Area))
    small_ear = net.add_layer("small_ear", exr.LayerSpec.resize2d(8, 8))
    fuse = net.add_layer("fuse", exr.LayerSpec.conv2d(4, exr.Window2D.square(3, 1, 1), False, False))
    net.add_inputs(eye, exr.Shape(1, 12, 16), name="camera")
    net.add_inputs(ear, exr.Shape(1, 20, 5), name="spectrogram")
    net.connect(eye, small_eye)
    net.connect(ear, small_ear)
    net.connect(small_eye, fuse)
    net.connect(small_ear, fuse)
    assert net.layer_shape(fuse) == exr.Shape(4, 8, 8)
    net.set_inputs("camera", np.ones((12, 16), np.float32))
    net.step()
    assert np.allclose(net.layer_output(small_eye), 1.0)
    spec = exr.Network.from_bytes(net.to_bytes()).layer_spec(small_eye).resize_spec
    assert spec == exr.ResizeSpec(8, 8, exr.Interpolation.Area)


def test_disparity_finds_depth_in_a_random_dot_stereogram():
    rng = np.random.default_rng(1)
    scene = rng.choice([-1.0, 1.0], size=(10, 40)).astype(np.float32)
    net = exr.Network()
    shape = exr.Shape(1, 10, 24)
    left = net.add_layer("left", exr.LayerSpec.retina(exr.RetinaSpec(shape), False, False))
    right = net.add_layer("right", exr.LayerSpec.retina(exr.RetinaSpec(shape), False, False))
    depth = net.add_layer("depth", exr.LayerSpec.disparity(
        exr.DisparitySpec(min_disparity=-2, max_disparity=4, window=3, measure=exr.DisparityMeasure.Normalized)))
    net.add_inputs(left, shape, name="left_eye")
    net.add_inputs(right, shape, name="right_eye")
    net.connect(left, depth)
    net.connect(right, depth)
    assert net.layer_shape(depth) == exr.Shape(7, 10, 24)
    for d in (-2, 0, 3):
        net.set_inputs("left_eye", scene[:, 8:32])
        net.set_inputs("right_eye", scene[:, 8 + d:32 + d])  # right[x] = left[x + d]
        net.step()
        match = net.layer_output(depth).reshape(7, 10, 24)
        assert int(np.argmax(match[:, 5, 12])) - 2 == d
    with pytest.raises(ValueError):
        net.add_layer("bad", exr.LayerSpec.disparity(exr.DisparitySpec(window=2)))
