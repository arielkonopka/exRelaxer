"""Vertical vs horizontal bars from Python: the C++ experiment bar_orientation
(frozen Gabor bank + frozen random mix + learned readout), written against the
exrelaxer package. A single-file experiment.

Images come from numpy's generator, not the C++ one, so the numbers are close
to bar_orientation's but not identical.
"""
import numpy as np

import exrelaxer as exr
from exrelaxer import harness as nnt


def draw(rng, side, noise, length=12, width=2):
    """A bright bar at a random position on a noisy dark background: (image, +1 vertical / -1 horizontal)."""
    image = 0.1 + rng.uniform(-noise, noise, (side, side))
    vertical = rng.integers(2) == 1
    margin = length // 2 - 2
    cy, cx = rng.integers(margin, side - margin, 2)
    for along in range(length):
        for across in range(width):
            y = cy - length // 2 + along if vertical else cy + across
            x = cx + across if vertical else cx - length // 2 + along
            if 0 <= y < side and 0 <= x < side:
                image[y, x] = 0.9 + rng.uniform(-noise, noise)
    return image.astype(np.float32).ravel(), 1.0 if vertical else -1.0


def build(p):
    net = exr.Network()
    image = exr.Shape(1, p["side"], p["side"])
    k = p["kernel"]
    eye = net.add_layer("eye", exr.LayerSpec.retina(exr.RetinaSpec(image), False, False))
    v1 = net.add_layer("v1", exr.LayerSpec.conv2d(2 * p["orientations"], exr.Window2D.square(k, 1, k // 2), False, False))
    pool = net.add_layer("pool", exr.LayerSpec.pool2d(exr.Window2D.square(p["pool"], p["pool"])))
    mix = net.add_layer("mix", exr.LayerSpec.dense(p["mix"], False, False, frozen=True))
    out = net.add_layer("out", exr.LayerSpec.dense(1, False, p["readout_er"]))
    net.add_inputs(eye, image)
    for a, b in ((eye, v1), (v1, pool), (pool, mix), (mix, out)):
        net.connect(a, b)
    net.add_output(out)
    net.load_filters(v1, exr.filters.gabor_bank(k, p["orientations"], p["wavelength"], p["sigma"]))
    net.freeze(v1)
    return net


def classify(net, image):
    net.set_inputs(image)
    net.step()
    return float(net.outputs()[0])


def accuracy(net, p, seed):
    rng = np.random.default_rng(seed)
    hits = 0
    for _ in range(p["test"]):
        image, label = draw(rng, p["side"], p["noise"])
        hits += classify(net, image) * label > 0
    return hits / p["test"]


def train(net, p, seed, learning_rate):
    rng = np.random.default_rng(seed)
    for _ in range(p["train"]):
        image, label = draw(rng, p["side"], p["noise"])
        wrong = classify(net, image) * label <= 0
        if wrong or p["reward"] == "target":
            net.apply_reward(label, learning_rate)


@nnt.experiment(
    description="bar_orientation from Python: frozen Gabor bank + frozen random mix + learned readout",
    tags=["vision", "learning", "quick"],
    params={
        "side": (24, "image side in pixels"),
        "noise": (0.1, "uniform pixel noise amplitude"),
        "kernel": (7, "Gabor kernel size"),
        "orientations": (4, "Gabor orientations (2 phases each)"),
        "wavelength": (5.0, "Gabor wavelength in pixels"),
        "sigma": (2.0, "Gabor envelope sigma in pixels"),
        "pool": (6, "max-pooling window and stride"),
        "mix": (64, "frozen random mixing neurons"),
        "readout_er": (False, "E-R in the learned readout"),
        "train": (1500, "training images"),
        "test": (200, "test images"),
        "lr": (0.01, "learning rate"),
        "reward": ("error", "error: reward only wrong answers; target: reward every image"),
    },
    trials=10,
    expect={"accuracy": (0.95, None), "accuracy_control": (None, 0.65)},
)
def run(t):
    p = t.params
    if p["reward"] not in ("error", "target"):
        raise ValueError("reward must be error or target")
    test_seed, train_seed = 1000 + t.seed, 2000 + t.seed

    net = build(p)
    t.record("accuracy_before", accuracy(net, p, test_seed))
    train(net, p, train_seed, p["lr"])
    after = accuracy(net, p, test_seed)
    t.record("accuracy", after)

    exr.reseed(t.seed)  # the same network, trained without learning
    control = build(p)
    train(control, p, train_seed, 0.0)
    control_accuracy = accuracy(control, p, test_seed)
    t.record("accuracy_control", control_accuracy)
    t.record("gain", after - control_accuracy)
