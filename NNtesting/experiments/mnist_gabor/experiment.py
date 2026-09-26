"""Handwritten digits (MNIST, or any 28 x 28-like set from fetch.py) with
frozen Gabor features and learned one-vs-rest readouts. See README.md here.

Needs the dataset: NNtesting/datasets/fetch.py get mnist
"""
import numpy as np

import exrelaxer as exr
from exrelaxer import harness as nnt
from network import build  # this folder's own module


def predict(net, image):
    net.set_inputs(image)
    net.step()
    y = net.outputs()
    return y, int(np.argmax(y))


def accuracy(net, data):
    hits = 0
    for image, label in zip(data.images, data.labels):
        hits += predict(net, image)[1] == label
    return hits / len(data)


def train(net, readouts, data, order, p, learning_rate):
    """One pass over data in `order`; returns the accuracy while training."""
    hits = 0
    for i in order:
        y, guess = predict(net, data.images[i])
        label = data.labels[i]
        hits += guess == label
        for c, out in enumerate(readouts):
            target = 1.0 if c == label else -1.0
            if p["reward"] == "target" or y[c] * target <= 0:  # error-driven: only wrong readouts learn
                net.apply_reward_to(out, target, learning_rate)
    return hits / len(order)


@nnt.experiment(
    description="MNIST: frozen Gabor bank + max pool + frozen random mix + one learned readout per class",
    tags=["vision", "learning", "dataset"],
    params={
        "dataset": ("mnist", "prepared dataset (NNtesting/datasets/fetch.py get NAME)"),
        "train": (10000, "training images per epoch (a random subset per trial)"),
        "test": (2000, "test images (the first of the test split)"),
        "epochs": (1, "passes over the training images"),
        "kernel": (7, "Gabor kernel size"),
        "orientations": (4, "Gabor orientations (2 phases each)"),
        "wavelength": (4.0, "Gabor wavelength in pixels"),
        "sigma": (2.0, "Gabor envelope sigma in pixels"),
        "pool": (4, "max-pooling window and stride"),
        "mix": (512, "frozen random mixing neurons (0: readouts read the pooled features)"),
        "readout_er": (False, "E-R in the readouts"),
        "lr": (0.01, "learning rate"),
        "reward": ("error", "error: only wrong readouts learn; target: every readout every image"),
    },
    trials=3,
    expect={"accuracy": (0.85, None)},
)
def run(t):
    p = t.params
    if p["reward"] not in ("error", "target"):
        raise ValueError("reward must be error or target")
    train_set = t.dataset(p["dataset"], "train")
    test_set = t.dataset(p["dataset"], "test", limit=p["test"])
    classes = len(train_set.classes)
    subset = t.rng.permutation(len(train_set))[:p["train"]]

    net, readouts = build(p, train_set.shape, classes)
    for epoch in range(p["epochs"]):
        t.record("train_accuracy", train(net, readouts, train_set, t.rng.permutation(subset), p, p["lr"]))
        t.log(f"trial {t.index} epoch {epoch}: train accuracy {t.metrics['train_accuracy']:.4f}")
    t.record("accuracy", accuracy(net, test_set))

    exr.reseed(t.seed)  # the same network without learning
    control, _ = build(p, train_set.shape, classes)
    t.record("accuracy_control", accuracy(control, test_set))
    t.record("gain", t.metrics["accuracy"] - t.metrics["accuracy_control"])
