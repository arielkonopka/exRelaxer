"""Video temporal memory: can a feed-forward E-R layer keep the direction of a
moving object across blank frames in its neurons' own state? See README.md.

    clip (20 x 15 frames, each held `ticks` ticks) -> 128 hidden (frozen, random) -> 2 readouts (LEFT, RIGHT)

A clip is visible motion, `gap` blank frames (input exactly zero, the
network keeps ticking), then the object reappears standing still for two
frames (clips.py). The readouts are read over those two frames (8 ticks)
and their mean decides. Every frame taken alone is the same for both
classes, so only history can answer.

Models (hidden layer, what it is shown each frame):
    C0   relu, current frame                      (no memory: expect chance)
    C1   E-R reset to rest at every frame, current frame
         (no memory across frames; the same weights as E0: the reset ablation)
    E0   E-R, current frame                       (the primary condition)
    R1   relu, current + previous frame
    R4   relu, current + previous 3 frames
    E1   E-R, current + previous frame
    Rg   relu, the shortest window that reaches back past the blank: current +
         previous gap + 1 frames (side) or gap + 2 (order)
    D2   E-R, then a second E-R layer (readouts read the second), feed-forward
    D2R  as D2, the second layer also reads its own previous output (recurrence)

Hidden layers are frozen after the random initialization (uniform, variance
1 / fan-in, from the trial seed) and use the local weight normalisation
(sum = x.w / |w|); E-R as in the library: no habituation, linear threshold
growth, recovery 0.9, resting threshold 0.2. Only the two readouts are
trained: ridge regression of the targets (+1 for the true class's readout,
-1 for the other) on the mean hidden output over the readout ticks, the penalty
chosen on a validation split, then loaded into a library readout layer
(raw weighted sum plus bias, clamped at +-max_output, as every readout in
the repository) that is run on the test clips. The hidden layer does not
depend on the readouts, so each clip's hidden activity is recorded once.

Probes (the network unchanged): the same ridge classifier on the hidden
state (E-R thresholds and outputs) right after the last visible frame and
right before the reappearance.
"""
import math
import os
import sys

import numpy as np

import exrelaxer as exr
from exrelaxer import harness as nnt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import clips  # noqa: E402

MODELS = {
    #       neuron, window (-1: see Recorder), reset every frame, depth, recurrent
    "C0": ("relu", 0, False, 1, False),
    "C1": ("er", 0, True, 1, False),
    "E0": ("er", 0, False, 1, False),
    "R1": ("relu", 1, False, 1, False),
    "R4": ("relu", 3, False, 1, False),
    "E1": ("er", 1, False, 1, False),
    "Rg": ("relu", -1, False, 1, False),     # window gap + 1 (side) or gap + 2 (order)
    "D2": ("er", 0, False, 2, False),
    "D2R": ("er", 0, False, 2, True),
}
EPS = exr.constants.firing_epsilon
LAMBDAS = (1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0)  # ridge penalties per row (standardised features)


def build(p, inputs, trainable=False):
    """The hidden network; every hidden layer is an output, so run() returns
    them all. trainable: hidden layers learn by feedback alignment with a
    bias (hidden_rule=fa) instead of being frozen."""
    neuron, _, _, depth, recurrent = MODELS[p["model"]]
    er = neuron == "er"
    net = exr.Network()
    width = p["width"]

    def spec():
        # recovery other than 0.9 (the library default when this was written):
        # a jitter of negligible width around it
        rj = (exr.Jitter.uniform(1e-6) if p["recovery"] != 0.9 else exr.Jitter.none()).around(p["recovery"])
        s = exr.LayerSpec.dense(width, False, er, frozen=not trainable, recovery_jitter=rj)
        if trainable or p["hidden_rule"] == "fa":
            s.learning_rule = exr.LearningRule.feedback_alignment().with_bias()
        s.rectify = not er
        s.normalize = p["normalize"]
        s.resting_threshold = p["resting_threshold"]
        return s

    layers = []
    for l in range(depth):
        h = net.add_layer(f"h{l + 1}", spec())
        if l == 0:
            net.add_inputs(h, inputs, "x")
        else:
            net.connect(layers[-1], h)
        layers.append(h)
    if recurrent:
        net.connect(layers[-1], layers[-1])
    if not trainable:
        for h in layers:
            net.add_output(h)
    rs = p["recurrent_scale"]
    for h in layers:
        for i in range(width):
            w = np.asarray(net.weights(h, i), dtype=np.float32)
            if recurrent and h == layers[-1]:
                w[:-width] *= math.sqrt(3.0 / (len(w) - width))
                w[-width:] *= rs * math.sqrt(3.0 / width)
            else:
                w *= math.sqrt(3.0 / len(w))
            net.set_weights(h, i, w)
    return net, layers


def fa_train(p, rec, train, rng):
    """hidden_rule=fa (a separate experiment): the hidden layers and two
    readouts learn together online by feedback alignment, from the error of
    every readout tick (the other ticks run without learning; nothing
    propagates the error back through time). The learned hidden weights and
    biases are then copied into the recorder's network, which is run and
    read out like every frozen model. Returns the online readouts' network
    and the per-clip runner for scoring it."""
    net, layers = build(p, clips.PIXELS * (rec.window + 1), trainable=True)
    out = net.add_layer("out", readout_spec(exr.LearningRule.feedback_alignment().with_bias()))
    net.connect(layers[-1], out)
    net.add_output(out)
    for j in range(2):
        net.set_weights(out, j, (np.asarray(net.weights(out, j)) * math.sqrt(3.0 / p["width"])).astype(np.float32))
    t = rec.ticks

    def run_clip(frames, info, target=None):
        x = windowed(frames, rec.window).astype(np.float32)
        lead = info["visible"] + rec.gap
        for h in layers:
            net.reset_state(h)
        if lead:
            net.run(np.repeat(x[:lead], t, axis=0))
        ys = []
        for row in np.repeat(x[lead:], t, axis=0):
            net.set_inputs("x", row)
            net.step()
            y = np.asarray(net.outputs())
            ys.append(y)
            if target is not None:
                net.apply_error(target - y, p["fa_lr"])
        return np.mean(ys, axis=0)

    frames, infos, labels = train
    for _ in range(p["fa_epochs"]):
        for i in rng.permutation(len(frames)):
            target = np.array([1.0, -1.0] if labels[i] == clips.LEFT else [-1.0, 1.0], np.float32)
            run_clip(frames[i], infos[i], target)
    for l, h in enumerate(layers):
        bias = np.asarray(net.bias(h))
        for i in range(p["width"]):
            rec.net.set_weights(rec.layers[l], i, np.asarray(net.weights(h, i), dtype=np.float32))
            rec.net.set_bias(rec.layers[l], i, float(bias[i]))
    return run_clip


def windowed(frames, window):
    """Frame k shown with the `window` frames before it (zeros before the clip)."""
    flat = frames.reshape(len(frames), -1)
    if window == 0:
        return flat
    padded = np.concatenate([np.zeros((window, flat.shape[1]), np.float32), flat])
    return np.concatenate([padded[window - j:window - j + len(flat)] for j in range(window + 1)], axis=1)


class Recorder:
    """Runs clips through the hidden network and keeps what the metrics need."""

    def __init__(self, p, gap):
        neuron, window, self.reset, self.depth, _ = MODELS[p["model"]]
        # Rg: the shortest window whose first readout frame still shows the
        # last visible frame (side) or the last two (order).
        self.window = gap + (1 if p["task"] == "side" else 2) if window < 0 else window
        self.er = neuron == "er"
        self.ticks, self.width, self.gap = p["ticks"], p["width"], gap
        self.net, self.layers = build(p, clips.PIXELS * (self.window + 1))

    def reset_all(self):
        for h in self.layers:
            self.net.reset_state(h)

    def thresholds(self):
        return np.concatenate([np.asarray(self.net.neuron_state(h)["threshold"]) for h in self.layers])

    def run_frames(self, x):
        """x: frames x inputs. Returns ticks x (depth * width) hidden outputs."""
        rows = np.repeat(x, self.ticks, axis=0)
        if not self.reset:
            return np.asarray(self.net.run(rows))
        out = []
        for f in range(len(x)):
            self.reset_all()
            out.append(np.asarray(self.net.run(rows[f * self.ticks:(f + 1) * self.ticks])))
        return np.concatenate(out)

    def clip(self, frames, info):
        x = windowed(frames, self.window).astype(np.float32)
        v, g, t = info["visible"], self.gap, self.ticks
        self.reset_all()
        vis = self.run_frames(x[:v])
        pre = np.concatenate([self.thresholds(), vis[-1]]) if self.er else vis[-1].copy()
        blank, thr_blank = [], []
        for f in range(v, v + g):
            blank.append(self.run_frames(x[f:f + 1]))
            if self.er:
                thr_blank.append(self.thresholds())
        blank = np.concatenate(blank) if blank else np.zeros((0, vis.shape[1]), np.float32)
        before = np.concatenate([thr_blank[-1], blank[-1]]) if self.er else blank[-1].copy()
        read = self.run_frames(x[v + g:])
        spikes = lambda a: np.count_nonzero(np.abs(a) > EPS) / (len(a) / t)
        rec = dict(
            readout=read[:, -self.width:],  # the readouts read the top hidden layer
            pre=pre, before=before,
            spikes_visible=spikes(vis), spikes_blank=spikes(blank) if g else 0.0, spikes_readout=spikes(read),
            spikes_first_readout_tick=np.count_nonzero(np.abs(read[0]) > EPS),
            frames=v + g + len(read) // t,
            spikes_total=np.count_nonzero(np.abs(np.concatenate([vis, blank, read])) > EPS),
        )
        if self.er:
            rec["thr_blank_mean"] = float(np.mean(thr_blank))
            rec["thr_before"] = thr_blank[-1]
            rec["thr_pre_mean"] = float(np.mean(pre[:self.depth * self.width]))
        return rec


def ridge_fit(X, Y, lam):
    """Least squares with an unpenalised bias on standardised features (the
    penalty lam per row), returned for the raw features: (W [features x
    outputs], b)."""
    mx, sx = X.mean(0), X.std(0)
    sx[sx < 1e-12] = 1.0
    my = Y.mean(0)
    Xs, Yc = (X - mx) / sx, Y - my
    A = Xs.T @ Xs + lam * len(X) * np.eye(X.shape[1])
    W = np.linalg.solve(A, Xs.T @ Yc) / sx[:, None]
    return W, my - mx @ W


def probe(train, ytr, val, yva, test, yte):
    """Ridge classifier (targets +-1); penalty by validation accuracy."""
    best = None
    for lam in LAMBDAS:
        W, b = ridge_fit(train, (2.0 * ytr - 1.0)[:, None], lam)
        acc = np.mean(((val @ W + b)[:, 0] > 0) == (yva == 1))
        if best is None or acc > best[0]:
            best = (acc, W, b)
    _, W, b = best
    return float(np.mean(((test @ W + b)[:, 0] > 0) == (yte == 1)))


def readout_spec(rule):
    """Two plain readout neurons: raw sum + bias. The Dense builder normalises
    by default since 2026-09-28, which would rescale the fitted weights."""
    s = exr.LayerSpec.dense(2, False, False, learning_rule=rule)
    s.normalize = False
    return s


def readout_network(W, b):
    """The trained readouts as a library layer: 2 plain neurons (raw sum + bias, clamped)."""
    net = exr.Network()
    out = net.add_layer("out", readout_spec(exr.LearningRule.feedback_alignment().with_bias()))
    net.add_inputs(out, W.shape[0], "h")
    net.add_output(out)
    for j in range(2):
        net.set_weights(out, j, W[:, j].astype(np.float32))
        net.set_bias(out, j, float(b[j]))
    return net


def readout_accuracy(net, feats, labels, ticks):
    """Mean output over each clip's readout ticks, then the larger readout wins."""
    y = np.asarray(net.run(np.concatenate(feats).astype(np.float32)))
    mean = y.reshape(len(feats), ticks, 2).mean(axis=1)
    return float(np.mean(np.argmax(mean, axis=1) == labels))


def lms_readout(train_feats, ytr, lr, epochs, rng):
    """The library's own error-driven readout training (for comparison with ridge)."""
    width = train_feats[0].shape[1]
    net = exr.Network()
    out = net.add_layer("out", readout_spec(exr.LearningRule.feedback_alignment().with_bias()))
    net.add_inputs(out, width, "h")
    net.add_output(out)
    for j in range(2):
        net.set_weights(out, j, (np.asarray(net.weights(out, j)) * math.sqrt(3.0 / width)).astype(np.float32))
    for _ in range(epochs):
        for i in rng.permutation(len(train_feats)):
            target = np.array([1.0, -1.0] if ytr[i] == clips.LEFT else [-1.0, 1.0], np.float32)
            for h in train_feats[i]:
                net.set_inputs("h", h.astype(np.float32))
                net.step()
                net.apply_error(target - np.asarray(net.outputs()), lr)
    return net


@nnt.experiment(
    name="video_memory",
    description="occluded moving object: does feed-forward E-R keep the direction across blank frames?",
    tags=["temporal", "dynamic"],
    params={
        "model": ("E0", "C0, C1, E0, R1, R4, E1, Rg, D2 or D2R (see the module docstring)"),
        "task": ("side", "side (where it was is enough) or order (only the order of frames tells); see clips.py"),
        "gap": (1, "blank frames between the last visible frame and the reappearance"),
        "width": (128, "hidden neurons per layer"),
        "ticks": (4, "network ticks per frame"),
        "train": (1000, "training clips"),
        "val": (500, "validation clips (readout penalty, probes)"),
        "test": (1000, "test clips"),
        "normalize": (True, "hidden layers: weighted sum / |w|"),
        "resting_threshold": (0.2, "E-R resting threshold"),
        "recovery": (0.9, "E-R recovery (threshold decay per silent tick); sensitivity study only"),
        "recurrent_scale": (0.5, "D2R: scale of the recurrent weights"),
        "hidden_rule": ("frozen", "frozen (the experiment) or fa: hidden layers trained by feedback alignment first"),
        "fa_lr": (0.01, "hidden_rule=fa: learning rate"),
        "fa_epochs": (10, "hidden_rule=fa: epochs over the training clips"),
        "lms_lr": (0.0, "if > 0, also train the readouts with the library's error rule at this rate"),
        "lms_epochs": (10, "epochs of that training"),
        "dump": ("", "if set, a directory to save per-clip states and labels (npz) for later analysis"),
    },
    trials=20,
    expect={},
)
def run(t):
    p = dict(t.params)
    if p["model"] not in MODELS:
        raise ValueError("model must be one of " + ", ".join(MODELS))
    exr.set_threads(1)
    gap = int(p["gap"])
    rec = Recorder(p, gap)
    data = {name: clips.make_set(p[name], gap, clips.split_seed(t.seed, s), p["task"])
            for s, name in enumerate(("train", "val", "test"))}
    if p["hidden_rule"] == "fa":
        run_clip = fa_train(p, rec, data["train"], t.rng)
        frames, info, labels = data["test"]
        mean = np.array([run_clip(f, i) for f, i in zip(frames, info)])
        t.record("fa_online_accuracy", np.mean(np.argmax(mean, axis=1) == labels))
    elif p["hidden_rule"] != "frozen":
        raise ValueError("hidden_rule must be frozen or fa")
    splits = {name: ([rec.clip(f, i) for f, i in zip(frames, info)], labels)
              for name, (frames, info, labels) in data.items()}

    # Readouts: ridge regression of the targets on the mean hidden output of
    # the readout interval (the mean readout output is what decides), the
    # penalty by validation accuracy of the library readout layer.
    def readout_set(name):
        rs, labels = splits[name]
        return [r["readout"] for r in rs], labels

    ftr, ytr = readout_set("train")
    fva, yva = readout_set("val")
    fte, yte = readout_set("test")
    rt = len(ftr[0])
    onehot = lambda y: np.where(y[:, None] == np.array([[clips.LEFT, clips.RIGHT]]), 1.0, -1.0)

    def fit(X, Y):
        best = None
        for lam in LAMBDAS:
            W, b = ridge_fit(X, Y, lam)
            acc = readout_accuracy(readout_network(W, b), fva, yva, rt)
            if best is None or acc > best[0]:
                best = (acc, lam, readout_network(W, b))
        return best

    val_acc, lam, net = fit(np.array([f.mean(0) for f in ftr]), onehot(ytr))
    t.record("accuracy", readout_accuracy(net, fte, yte, rt))
    t.record("train_accuracy", readout_accuracy(net, ftr, ytr, rt))
    t.record("val_accuracy", val_acc)
    t.record("ridge_lambda", lam)
    # The same fitted on every readout tick separately (as online training would).
    _, _, tick_net = fit(np.concatenate(ftr), np.repeat(onehot(ytr), rt, axis=0))
    t.record("accuracy_tick_fit", readout_accuracy(tick_net, fte, yte, rt))
    if p["lms_lr"] > 0:
        lnet = lms_readout(ftr, ytr, p["lms_lr"], p["lms_epochs"], t.rng)
        t.record("lms_accuracy", readout_accuracy(lnet, fte, yte, rt))

    # Probes on the hidden state.
    def stack(name, key):
        return np.array([r[key] for r in splits[name][0]])

    for key, metric in (("pre", "probe_pre_occlusion"), ("before", "probe_before_reappearance")):
        t.record(metric, probe(stack("train", key), ytr, stack("val", key), yva, stack("test", key), yte))

    # Activity on the test clips (spikes: hidden outputs with |y| > firing_epsilon, per frame).
    test = splits["test"][0]
    for key in ("spikes_visible", "spikes_blank", "spikes_readout", "spikes_first_readout_tick"):
        t.record(key, np.mean([r[key] for r in test]))
    t.record("spikes_per_frame", np.sum([r["spikes_total"] for r in test]) / np.sum([r["frames"] for r in test]))
    t.record("window", rec.window)
    t.record("inputs", clips.PIXELS * (rec.window + 1))
    if rec.er:
        thr = np.concatenate([r["thr_before"] for r in test])
        t.record("threshold_blank_mean", np.mean([r["thr_blank_mean"] for r in test]))
        t.record("threshold_pre_occlusion_mean", np.mean([r["thr_pre_mean"] for r in test]))
        for q in (10, 50, 90):
            t.record(f"threshold_reappearance_p{q}", np.percentile(thr, q))
    if p["dump"]:
        os.makedirs(p["dump"], exist_ok=True)
        np.savez_compressed(os.path.join(p["dump"], f"{p['task']}_{p['model']}_gap{gap}_seed{t.seed}.npz"),
                            **{f"{n}_{k}": stack(n, k) for n in ("train", "test") for k in ("pre", "before")},
                            train_labels=ytr, test_labels=yte,
                            test_readout=np.array([r["readout"] for r in test]))
