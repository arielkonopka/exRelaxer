"""State component ablation: which parts of the hidden neurons' state carry
the direction of an occluded object, and are they complementary? See
README.md and doc/state_output.md.

    clip -> 128 hidden E-R neurons (frozen, with habituation) -> State layer
         -> 2 readouts trained online (library delta rule) -> LEFT / RIGHT

The task is video_memory's `side` task, unchanged: the object moves, the
screen is blank for `gap` frames (video_memory's definition of the gap),
then the object reappears at the centre, the same for both classes (see
video_memory/clips.py). Classes are exactly balanced in every split.

The State layer exposes, per hidden neuron, its output, its threshold minus
the resting threshold and its habituation streak (min(streak / onset, 1)),
with no processing. Each configuration lets the readouts read only some of
those columns:

    A output                 D output + threshold
    B threshold              E threshold + habituation
    C habituation            F output + threshold + habituation

Everything else is shared by the six configurations of a trial: the clips,
the hidden network and its weights, the recordings (the hidden layer runs
once per clip and every configuration reads the same recorded values), the
readout's initial weights (zero), its learning rate, epochs, clip order and
input gain rule. So the only difference is the set of columns.

Readout: the video_memory_learned readout, immediate (no trace): two plain
neurons with a bias, trained online by Network.apply_error once per clip on
the error of the prediction (the mean readout output over the readout
ticks). Each input column is scaled by 1 / its sd over the training readout
ticks (the same rule for every column and configuration; a readout-side
gain, the State layer itself passes raw values).

Separately from the readouts, the state itself is analysed at two moments:
the end of the blank (before the reappearance) and the last readout tick
(immediately before the prediction): per component, the population means for
LEFT and RIGHT, per-neuron class separation |d| (Cohen's d), and a ridge
probe (the best linear classifier, fitted offline) on that component alone.
The probe says whether the information is there; the online readout
whether the library's learning rule can use it.
"""
import importlib.util
import math
import os
import sys
from pathlib import Path

import numpy as np

import exrelaxer as exr
from exrelaxer import harness as nnt

HERE = os.path.dirname(os.path.abspath(__file__))
VM_DIR = os.path.join(HERE, "..", "video_memory")


def _video_memory():
    """video_memory's module: the one nntest already imported, or imported here
    under the same name (and marked loaded, so it registers once)."""
    name = "nntest_experiment_video_memory"
    if name in sys.modules:
        return sys.modules[name]
    source = os.path.realpath(os.path.join(VM_DIR, "experiment.py"))
    if VM_DIR not in sys.path:
        sys.path.insert(0, VM_DIR)
    spec = importlib.util.spec_from_file_location(name, source)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    nnt._loaded.add(Path(source).resolve())
    previous, nnt._loading = nnt._loading, (Path(source), Path(VM_DIR).resolve())
    try:
        spec.loader.exec_module(module)
    finally:
        nnt._loading = previous
    return module


vm = _video_memory()
clips = vm.clips
LEFT, RIGHT = clips.LEFT, clips.RIGHT
COMPONENTS = ("output", "threshold", "habituation")   # the State layer's field order
CONFIGS = {
    "A": ("output",),
    "B": ("threshold",),
    "C": ("habituation",),
    "D": ("output", "threshold"),
    "E": ("threshold", "habituation"),
    "F": ("output", "threshold", "habituation"),
}


def parse_list(value, cast=float):
    if isinstance(value, (list, tuple)):
        return [cast(v) for v in value]
    return [cast(v) for v in str(value).replace(";", ",").split(",") if str(v).strip()]


class Recorder:
    """The hidden layer (video_memory's E0 with habituation) and its State
    layer. For each clip: the State values of every readout tick, and the
    state at the end of the blank and at the last readout tick."""

    def __init__(self, p):
        self.ticks, self.width = p["ticks"], p["width"]
        self.net = exr.Network()
        rj = (exr.Jitter.uniform(1e-6) if p["recovery"] != 0.9 else exr.Jitter.none()).around(p["recovery"])
        spec = exr.LayerSpec.dense(self.width, True, True, frozen=True, recovery_jitter=rj)
        spec.normalize = p["normalize"]
        spec.resting_threshold = p["resting_threshold"]
        spec.habituation_rule = exr.Habituation(steps=p["hab_steps"], tolerance=p["hab_tolerance"],
                                                decay=p["hab_decay"], fade_after=p["hab_fade_after"])
        self.h = self.net.add_layer("h", spec)
        self.tap = self.net.add_layer("h_state", exr.LayerSpec.state())   # output, threshold, habituation
        self.net.add_inputs(self.h, clips.PIXELS, "x")
        self.net.connect(self.h, self.tap)
        self.net.add_output(self.tap)
        for i in range(self.width):   # video_memory's initialisation: variance 1 / fan-in
            w = np.asarray(self.net.weights(self.h, i), dtype=np.float32)
            self.net.set_weights(self.h, i, (w * math.sqrt(3.0 / len(w))).astype(np.float32))

    def split(self, rows):
        """[ticks, 3 * width] interleaved -> dict component -> [ticks, width]."""
        return {c: rows[:, k::3] for k, c in enumerate(COMPONENTS)}

    def clip(self, frames, info):
        x = frames.reshape(len(frames), -1).astype(np.float32)
        lead = info["visible"] + info["gap"]
        self.net.reset_state(self.h)
        before = np.asarray(self.net.run(np.repeat(x[:lead], self.ticks, axis=0)))[-1]
        read = np.asarray(self.net.run(np.repeat(x[lead:], self.ticks, axis=0)))
        return dict(read=self.split(read), before={c: before[k::3] for k, c in enumerate(COMPONENTS)},
                    final={c: read[-1, k::3] for k, c in enumerate(COMPONENTS)})


def train_online(feats, labels, lr, epochs, order_seed):
    """feats: per clip [readout ticks, n] (already scaled). Two readouts, zero
    initial weights, trained once per clip on the error of the mean output."""
    n = feats[0].shape[1]
    net = exr.Network()
    out = net.add_layer("out", vm.readout_spec(exr.LearningRule.feedback_alignment().with_bias()))
    net.add_inputs(out, n, "s")
    net.add_output(out)
    for j in range(2):
        net.set_weights(out, j, np.zeros(n, np.float32))
    order = np.random.default_rng(order_seed)
    for _ in range(epochs):
        for i in order.permutation(len(feats)):
            y = np.asarray(net.run(feats[i])).mean(0)
            target = np.array([1.0, -1.0] if labels[i] == LEFT else [-1.0, 1.0], np.float32)
            net.apply_error((target - y).astype(np.float32), lr)
    return net


def predict(net, feats):
    return np.array([int(np.argmax(np.asarray(net.run(f)).mean(0))) for f in feats])


def scores(pred, y):
    """RIGHT (1) as positive: accuracy, balanced accuracy, confusion."""
    tp = int(np.sum((pred == 1) & (y == 1)))
    fn = int(np.sum((pred == 0) & (y == 1)))
    fp = int(np.sum((pred == 1) & (y == 0)))
    tn = int(np.sum((pred == 0) & (y == 0)))
    bal = 0.5 * (tp / max(tp + fn, 1) + tn / max(tn + fp, 1))
    return (tp + tn) / len(y), bal, (tp, fn, fp, tn)


def cohen_d(a, b):
    sd = np.sqrt(0.5 * (a.var(0) + b.var(0)))
    diff = a.mean(0) - b.mean(0)
    return np.where(sd > 1e-9, np.abs(diff) / np.maximum(sd, 1e-9), np.where(np.abs(diff) > 1e-9, np.inf, 0.0))


@nnt.experiment(
    name="state_ablation",
    description="which State components (output, threshold, habituation) carry the direction of an occluded "
                "object: six column sets, paired seeds, online readout vs probes",
    tags=["temporal", "dynamic", "learning"],
    params={
        "gap": (1, "blank frames (video_memory's definition)"),
        "recovery": (0.9, "E-R recovery of the hidden neurons"),
        "configs": ("A,B,C,D,E,F", "configurations to run (';' separates in --set)"),
        "width": (128, "hidden neurons"),
        "ticks": (4, "network ticks per frame"),
        "train": (1000, "training clips"),
        "val": (500, "validation clips (probe penalties)"),
        "test": (1000, "test clips"),
        "normalize": (True, "hidden layer: weighted sum / |w| (video_memory default)"),
        "resting_threshold": (0.2, "E-R resting threshold (video_memory default)"),
        "hab_steps": (100, "habituation: repeats before the input is cut (library default)"),
        "hab_tolerance": (0.0, "habituation: relative tolerance of a repeat (library default)"),
        "hab_decay": (0.0, "habituation: 0 cuts (library default); > 0 fades"),
        "hab_fade_after": (2, "habituation: repeats before fading (fade mode)"),
        "lr": (0.0003, "readout learning rate, the same for every configuration (tuned once: sweep.sh)"),
        "epochs": (10, "readout epochs"),
    },
    trials=10,
    expect={},
)
def run(t):
    p = dict(t.params)
    exr.set_threads(1)
    gap = int(p["gap"])
    configs = [c.strip() for c in str(p["configs"]).replace(";", ",").split(",") if c.strip()]
    for c in configs:
        if c not in CONFIGS:
            raise ValueError("configs: letters A-F")
    rec = Recorder(p)
    data = {}
    for s, name in enumerate(("train", "val", "test")):
        frames, infos, labels = clips.make_set(p[name], gap, clips.split_seed(t.seed, s), "side")
        n_left, n_right = int(np.sum(labels == LEFT)), int(np.sum(labels == RIGHT))
        if abs(n_left - n_right) > 1:
            raise RuntimeError(f"{name}: classes not balanced ({n_left}/{n_right})")
        t.record(f"{name}_left", n_left)
        t.record(f"{name}_right", n_right)
        recs = [rec.clip(f, dict(i, gap=gap)) for f, i in zip(frames, infos)]
        data[name] = (recs, labels)
    t.record("chance", 0.5)
    ytr, yva, yte = (data[n][1] for n in ("train", "val", "test"))

    # --- The state itself, independent of any readout -----------------------
    for moment in ("before", "final"):
        for c in COMPONENTS:
            X = {n: np.array([r[moment][c] for r in data[n][0]], np.float64) for n in data}
            a, b = X["test"][yte == LEFT], X["test"][yte == RIGHT]
            d = cohen_d(a, b)
            t.record(f"state_{moment}_{c}_mean_left", float(a.mean()))
            t.record(f"state_{moment}_{c}_mean_right", float(b.mean()))
            t.record(f"state_{moment}_{c}_sd", float(X["test"].std()))
            t.record(f"state_{moment}_{c}_d_median", float(np.median(d)))
            t.record(f"state_{moment}_{c}_d_max", float(np.max(np.where(np.isfinite(d), d, 0.0))))
            t.record(f"state_{moment}_{c}_neurons_d_over_0.5", int(np.sum(d > 0.5)))
            t.record(f"state_{moment}_{c}_constant", int(X["test"].std() < 1e-9))
            t.record(f"probe_{moment}_{c}", vm.probe(X["train"], ytr, X["val"], yva, X["test"], yte))

    # --- Readouts per configuration: identical conditions, only the columns differ
    # gain per column from the training readout ticks (the same rule for all)
    gain = {}
    for c in COMPONENTS:
        sd = np.concatenate([r["read"][c] for r in data["train"][0]]).std(0)
        gain[c] = np.where(sd > 1e-6, 1.0 / np.maximum(sd, 1e-6), 1.0).astype(np.float32)
    order_seed = [int(t.seed), 101]
    for key in configs:
        comps = CONFIGS[key]

        def feats(name):
            return [np.concatenate([r["read"][c] * gain[c] for c in comps], axis=1).astype(np.float32)
                    for r in data[name][0]]

        ftr, fva, fte = feats("train"), feats("val"), feats("test")
        net = train_online(ftr, ytr, p["lr"], p["epochs"], order_seed)
        acc, bal, (tp, fn, fp, tn) = scores(predict(net, fte), yte)
        t.record(f"{key}_accuracy", acc)
        t.record(f"{key}_balanced", bal)
        for name, v in (("tp", tp), ("fn", fn), ("fp", fp), ("tn", tn)):
            t.record(f"{key}_{name}", v)
        t.record(f"{key}_train_accuracy", scores(predict(net, ftr), ytr)[0])
        t.record(f"{key}_val_balanced", scores(predict(net, fva), yva)[1])  # for tune.sh only
        t.record(f"{key}_inputs", ftr[0].shape[1])
        # the same columns, best linear readout (offline): is the information there?
        mean = {n: np.array([f.mean(0) for f in feats(n)], np.float64) for n in data}
        t.record(f"{key}_ridge", vm.probe(mean["train"], ytr, mean["val"], yva, mean["test"], yte))
