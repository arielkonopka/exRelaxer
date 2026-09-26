"""Vision layers, filter banks, and a small learning task driven from Python."""
import math

import numpy as np
import pytest

import exrelaxer as exr


def test_spiral_retina_matches_the_documented_sample_count():
    net = exr.Network()
    image = exr.Shape(1, 200, 320)
    eye = net.add_layer("eye", exr.LayerSpec.retina(exr.RetinaSpec(image, exr.Sampling.Spiral), False, False))
    net.add_inputs(eye, image)
    assert net.layer_shape(eye) == exr.Shape(1, 1, 31099)  # NNtesting/README.md: spiral.py info
    points = net.retina_points(eye)
    assert points.shape == (31099, 2)
    assert points[0].tolist() == [159.5, 99.5]  # sample 0 at the image centre


def test_grid_retina_passes_the_image_through():
    net = exr.Network()
    image = exr.Shape(1, 4, 5)
    eye = net.add_layer("eye", exr.LayerSpec.retina(exr.RetinaSpec(image), False, False))
    net.add_inputs(eye, image)
    net.add_output(eye)
    pixels = np.linspace(0, 1, 20, dtype=np.float32).reshape(1, 4, 5)
    net.set_inputs(pixels)
    net.step()
    assert np.array_equal(net.layer_output(eye), pixels)


def test_filter_banks():
    bank = exr.filters.gabor_bank(7, 4, 5.0, 2.0)
    assert len(bank) == 8  # 4 orientations x 2 phases
    for f in bank:
        assert f.weights.shape == (7, 7)
        assert abs(float(f.weights.sum())) < 1e-4  # zero mean
    custom = exr.filters.gabor_bank(7, 4, 5.0, 2.0, phases=[0.0, math.pi / 2])
    assert all(np.array_equal(a.weights, b.weights) for a, b in zip(bank, custom))
    assert len(exr.filters.gabor_bank(7, 3, 5.0, 2.0, phases=[0.0])) == 3
    on, off = exr.filters.centre_surround_bank(9, 1.0, 3.0)
    assert np.allclose(on.weights, -off.weights)
    assert exr.filters.gaussian(5, 1.0).weights.sum() == pytest.approx(1.0, abs=1e-5)


def test_conv2d_kernels_and_filters():
    net = exr.Network()
    image = exr.Shape(1, 12, 12)
    eye = net.add_layer("eye", exr.LayerSpec.retina(exr.RetinaSpec(image), False, False))
    v1 = net.add_layer("v1", exr.LayerSpec.conv2d(8, exr.Window2D.square(7, 1, 3), False, False))
    net.add_inputs(eye, image)
    net.connect(eye, v1)
    bank = exr.filters.gabor_bank(7, 4, 5.0, 2.0)
    net.load_filters(v1, bank)
    for k, f in enumerate(bank):
        assert np.array_equal(net.kernel(v1, k), f.weights.ravel())
    assert net.layer_shape(v1) == exr.Shape(8, 12, 12)
    net.set_kernel(v1, 0, np.ones(49, np.float32))
    assert net.kernel(v1, 0).tolist() == [1.0] * 49
    with pytest.raises(ValueError):
        net.set_kernel(v1, 0, [1.0])


# --- Bar orientation: the nntest experiment bar_orientation, from Python ---------

SIDE, LENGTH, WIDTH = 24, 12, 2


def draw_bar(rng, noise=0.1):
    """A bright bar at a random position on a noisy dark background; +1 vertical, -1 horizontal."""
    image = 0.1 + rng.uniform(-noise, noise, (SIDE, SIDE))
    vertical = rng.integers(2) == 1
    margin = LENGTH // 2 - 2
    cy, cx = rng.integers(margin, SIDE - margin, 2)
    for along in range(LENGTH):
        for across in range(WIDTH):
            y = cy - LENGTH // 2 + along if vertical else cy + across
            x = cx + across if vertical else cx - LENGTH // 2 + along
            if 0 <= y < SIDE and 0 <= x < SIDE:
                image[y, x] = 0.9 + rng.uniform(-noise, noise)
    return image.astype(np.float32).ravel(), 1.0 if vertical else -1.0


def bar_network(seed):
    """retina -> Gabor Conv2D (frozen) -> max pool -> random mix (frozen) -> learned readout."""
    exr.reseed(seed)
    net = exr.Network()
    image = exr.Shape(1, SIDE, SIDE)
    eye = net.add_layer("eye", exr.LayerSpec.retina(exr.RetinaSpec(image), False, False))
    v1 = net.add_layer("v1", exr.LayerSpec.conv2d(8, exr.Window2D.square(7, 1, 3), False, False))
    pool = net.add_layer("pool", exr.LayerSpec.pool2d(exr.Window2D.square(6, 6)))
    mix = net.add_layer("mix", exr.LayerSpec.dense(64, False, False, frozen=True))
    out = net.add_layer("out", exr.LayerSpec.dense(1, False, False))
    net.add_inputs(eye, image)
    for a, b in ((eye, v1), (v1, pool), (pool, mix), (mix, out)):
        net.connect(a, b)
    net.add_output(out)
    net.load_filters(v1, exr.filters.gabor_bank(7, 4, 5.0, 2.0))
    net.freeze(v1)
    return net


def classify(net, image):
    net.set_inputs(image)
    net.step()
    return float(net.outputs()[0])


def accuracy(net, seed, count=200):
    rng = np.random.default_rng(seed)
    hits = 0
    for _ in range(count):
        image, label = draw_bar(rng)
        hits += classify(net, image) * label > 0
    return hits / count


def train(net, seed, count, learning_rate):
    rng = np.random.default_rng(seed)
    for _ in range(count):
        image, label = draw_bar(rng)
        if classify(net, image) * label <= 0:  # error-driven: reward only wrong answers
            net.apply_reward(label, learning_rate)


def test_bar_orientation_learns():
    learned, control = [], []
    for seed in range(3):
        net = bar_network(seed)
        train(net, 2000 + seed, 1500, 0.01)
        learned.append(accuracy(net, 1000 + seed))
        net = bar_network(seed)  # the same network, without learning
        train(net, 2000 + seed, 1500, 0.0)
        control.append(accuracy(net, 1000 + seed))
    assert np.mean(learned) >= 0.9, learned
    assert np.mean(control) <= 0.65, control
