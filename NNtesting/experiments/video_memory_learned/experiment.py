"""Learned video memory: can exRelaxer learn, online, to use the memory trace
that its E-R state already keeps? See README.md.

    clip -> 128 hidden (the video_memory network) -> 2 readouts, learned online -> LEFT / RIGHT

The clips, the hidden network and the E-R settings are video_memory's,
unchanged (the same seeds give the same clips and the same weights). What
changes is how the two readouts are trained: not by ridge regression on
recorded states, but online, clip by clip and tick by tick, by the
library's own error rule (Network.apply_error on an output layer with the
feedback_alignment rule: the delta rule w += rate * error * pre). The
prediction error exists only on the readout ticks, after the blank.

Two ways to assign that delayed error to inputs:

    trace = 0      immediate: pre = the hidden output at this tick
    trace = d > 0  eligibility trace: pre = X, X <- d * X + x on every tick of
                   the clip (the rule's own input trace, reset at the start of
                   each clip), so activity from before the blank is still
                   eligible when the error arrives. d is per tick; a frame is
                   4 ticks.

The forward pass is the same in both: the readout computes w . x + b from
the current hidden output only. Each trial records the hidden network once
and trains every trace value from the same recordings and initial weights
(paired), each with its learning rate chosen on validation clips.

mode=hidden (a separate experiment, condition D): the hidden layer learns
too, by feedback alignment from the same readout-tick errors, with its own
input trace (hidden_trace) over the pixels. That is the only path by which a
delayed error can change what the E-R state holds at the reappearance.

Controls in every trial: the ridge readout of video_memory (reproduced
exactly), a constant predictor, and probes (ridge classifiers, network
unchanged) on the hidden state before the reappearance and at the moment of
prediction, and on each eligibility trace.
"""
import importlib.util
import math
import os
import sys

import numpy as np

import exrelaxer as exr
from exrelaxer import harness as nnt

HERE = os.path.dirname(os.path.abspath(__file__))
VM_DIR = os.path.join(HERE, "..", "video_memory")


def _video_memory():
    """video_memory's experiment module: the one nntest already imported, or
    imported here under the same name (and marked loaded, so it registers once)."""
    name = "nntest_experiment_video_memory"
    if name in sys.modules:
        return sys.modules[name]
    source = os.path.realpath(os.path.join(VM_DIR, "experiment.py"))
    if VM_DIR not in sys.path:
        sys.path.insert(0, VM_DIR)
    spec = importlib.util.spec_from_file_location(name, source)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    from pathlib import Path
    nnt._loaded.add(Path(source).resolve())
    previous, nnt._loading = nnt._loading, (Path(source), Path(VM_DIR).resolve())
    try:
        spec.loader.exec_module(module)
    finally:
        nnt._loading = previous
    return module


vm = _video_memory()
clips = vm.clips
EPS = vm.EPS
LEFT, RIGHT = clips.LEFT, clips.RIGHT


def parse_list(value, cast=float):
    if isinstance(value, (list, tuple)):
        return [cast(v) for v in value]
    return [cast(v) for v in str(value).replace(";", ",").split(",") if str(v).strip()] if isinstance(value, str) \
        else [cast(value)]


def targets(label):
    return np.array([1.0, -1.0] if label == LEFT else [-1.0, 1.0], np.float32)


class FullRecorder(vm.Recorder):
    """video_memory's Recorder, keeping every tick of the clip (the eligibility
    trace needs the activity before the blank) and the state at the end of the
    readout interval.

    readout_inputs=output_state: a State layer on the top hidden layer
    (LayerSpec.state, doc/model.md#state-as-output) is added as the last
    output, and the readouts read the top layer's outputs followed by its
    (threshold - rest, habituation streak) per neuron: 3 * width inputs."""

    def __init__(self, p, gap):
        super().__init__(p, gap)
        self.features = self.width
        if p["readout_inputs"] == "output_state":
            tap = self.net.add_layer("top_state", exr.LayerSpec.state())
            self.net.connect(self.layers[-1], tap)
            self.net.add_output(tap)
            self.features = 3 * self.width
        elif p["readout_inputs"] != "output":
            raise ValueError("readout_inputs must be output or output_state")

    def clip(self, frames, info):
        x = vm.windowed(frames, self.window).astype(np.float32)
        v, g, t = info["visible"], self.gap, self.ticks
        self.reset_all()
        vis = self.run_frames(x[:v])
        thr_pre = self.thresholds() if self.er else None
        blank, thr_before = [], None
        for f in range(v, v + g):
            blank.append(self.run_frames(x[f:f + 1]))
        if self.er:
            thr_before = self.thresholds()
        blank = np.concatenate(blank) if blank else np.zeros((0, vis.shape[1]), np.float32)
        read = self.run_frames(x[v + g:])
        top = slice(-self.features, None)
        rec = dict(
            ticks=np.concatenate([vis, blank, read])[:, top].astype(np.float32),
            lead=(v + g) * t,
            readout=read[:, top],
            visible_mean=vis[:, top].mean(0),
            before=np.concatenate([thr_before, blank[-1]]) if self.er else blank[-1].copy(),
            spikes_visible=np.count_nonzero(np.abs(vis) > EPS) / v,
        )
        if self.er:
            last = slice(-self.width, None)  # the top layer's thresholds
            rec["thr_before"] = thr_before[last]
            rec["thr_pre"] = thr_pre[last]
            rec["thr_end"] = self.thresholds()[last]
            rec["prediction_state"] = np.concatenate([rec["thr_end"], read[:, last].mean(0)])
        else:
            rec["prediction_state"] = read[:, top].mean(0)
        return rec


def eligibility(ticks, trace):
    """X after every tick: X <- trace * X + x (what the rule keeps), float64."""
    X = np.zeros(ticks.shape[1])
    out = np.empty(ticks.shape)
    for k, row in enumerate(ticks):
        X = trace * X + row
        out[k] = X
    return out


def eligible(r, trace, error_at, from_readout=False):
    """The eligibility the rule multiplies the error with: the trace at the
    last readout tick (error_at=end) or its mean over the readout ticks (tick).
    from_readout: the part built during the readout interval alone."""
    ticks = r["ticks"][r["lead"]:] if from_readout else r["ticks"]
    X = eligibility(ticks, trace)[-(len(r["ticks"]) - r["lead"]):]
    return X[-1] if error_at == "end" else X.mean(0)


def online_net(width, trace, seed):
    """Two library readouts learning by the delta rule, with an input trace."""
    exr.reseed(seed)
    net = exr.Network()
    rule = exr.LearningRule.feedback_alignment(trace).with_bias()
    out = net.add_layer("out", vm.readout_spec(rule))
    net.add_inputs(out, width, "h")
    net.add_output(out)
    for j in range(2):
        net.set_weights(out, j, (np.asarray(net.weights(out, j)) * math.sqrt(3.0 / width)).astype(np.float32))
    return net, out


def weights(net, layer, width):
    W = np.stack([np.asarray(net.weights(layer, j)) for j in range(2)], axis=1)
    return W.astype(np.float64), np.asarray(net.bias(layer)).astype(np.float64)


def train_online(recs, labels, trace, lr, epochs, width, seed, order_rng, val=None, error_at="end"):
    """Online training, one clip at a time. The whole clip is run through the
    readout layer, which keeps its input trace from the first tick.

    error_at=end (the experiment): one prediction per clip, as in the test:
    the mean readout output over the readout ticks. Its error (target - mean)
    is applied once, after the last readout tick, to the eligibility the
    rule holds then: the hidden output of that tick (trace 0) or the trace of
    the whole clip.
    error_at=tick: the error of every readout tick is applied at that tick.

    Returns (W, b, curve); curve holds per epoch the mean squared error of the
    predictions, the accuracy of the predictions made during training, and,
    with val, the validation accuracy after the epoch."""
    net, out = online_net(width, trace, seed)
    curve = []
    for _ in range(epochs):
        sq, hits = 0.0, 0
        for i in order_rng.permutation(len(recs)):
            r = recs[i]
            target = targets(labels[i])
            net.reset_traces(out)   # a new clip: an empty eligibility trace
            if error_at == "end":
                y = np.asarray(net.run(r["ticks"]))[r["lead"]:].mean(0)
                err = target - y
                net.apply_error(err.astype(np.float32), lr)
            elif error_at == "tick":
                if r["lead"]:
                    net.run(r["ticks"][:r["lead"]])
                ys = []
                for row in r["ticks"][r["lead"]:]:
                    net.set_inputs("h", row)
                    net.step()
                    yt = np.asarray(net.outputs())
                    net.apply_error(target - yt, lr)
                    ys.append(yt)
                y = np.mean(ys, axis=0)
                err = target - y
            else:
                raise ValueError("error_at must be end or tick")
            sq += float(err @ err) / 2
            hits += int(np.argmax(y) == labels[i])
        entry = dict(mse=sq / len(recs), online_accuracy=hits / len(recs))
        if val is not None:
            W, b = weights(net, out, width)
            entry["val_accuracy"] = scores(W, b, *val)["accuracy"]
        curve.append(entry)
    W, b = weights(net, out, width)
    return W, b, curve


def scores(W, b, recs, labels):
    """The readouts' decision (the larger mean output over the readout ticks),
    through a library readout layer, and its confusion matrix."""
    feats = [r["readout"] for r in recs]
    net = vm.readout_network(W, b)
    y = np.asarray(net.run(np.concatenate(feats).astype(np.float32)))
    pred = np.argmax(y.reshape(len(feats), -1, 2).mean(axis=1), axis=1)
    cm = np.zeros((2, 2), int)
    for true, guess in zip(labels, pred):
        cm[true, guess] += 1
    recall = cm.diagonal() / np.maximum(cm.sum(1), 1)
    return dict(accuracy=float(np.mean(pred == labels)), left_accuracy=float(recall[LEFT]),
                right_accuracy=float(recall[RIGHT]), balanced_accuracy=float(recall.mean()),
                cm_LL=int(cm[LEFT, LEFT]), cm_LR=int(cm[LEFT, RIGHT]), cm_RL=int(cm[RIGHT, LEFT]),
                cm_RR=int(cm[RIGHT, RIGHT]))


def ridge_readout(splits):
    """video_memory's readout, exactly: ridge on the mean readout-tick output,
    penalty by validation accuracy of the library readout layer."""
    (ftr, ytr), (fva, yva) = [([r["readout"] for r in splits[s][0]], splits[s][1]) for s in ("train", "val")]
    rt = len(ftr[0])
    onehot = lambda y: np.where(y[:, None] == np.array([[LEFT, RIGHT]]), 1.0, -1.0)
    X, Y = np.array([f.mean(0) for f in ftr]), onehot(ytr)
    best = None
    for lam in vm.LAMBDAS:
        W, b = vm.ridge_fit(X, Y, lam)
        acc = vm.readout_accuracy(vm.readout_network(W, b), fva, yva, rt)
        if best is None or acc > best[0]:
            best = (acc, lam, W, b)
    return best


def selectivity(a, labels):
    """Per feature: (mean RIGHT - mean LEFT) / pooled sd (0 where constant)."""
    a = np.asarray(a, np.float64)
    r, l = a[labels == RIGHT], a[labels == LEFT]
    sd = np.sqrt((r.var(0) + l.var(0)) / 2)
    return np.where(sd > 1e-12, (r.mean(0) - l.mean(0)) / np.where(sd > 1e-12, sd, 1.0), 0.0)


def corr(a, b):
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    if a.std() < 1e-12 or b.std() < 1e-12:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def credit(t, prefix, W0, W, recs, labels, trace, ridge_w, dump, error_at):
    """Credit-assignment diagnostics on the training clips: which hidden
    neurons carried the direction before the blank, kept it in their state,
    accumulated eligibility, and got the weight change. dw is the change of
    (w_RIGHT - w_LEFT): positive means the neuron now votes RIGHT."""
    dw = (W[:, RIGHT] - W[:, LEFT]) - (W0[:, RIGHT] - W0[:, LEFT])
    sel = dict(visible=selectivity([r["visible_mean"] for r in recs], labels),
               readout=selectivity([r["readout"].mean(0) for r in recs], labels))
    if "thr_before" in recs[0]:
        sel["state"] = selectivity([r["thr_before"] for r in recs], labels)
    # The eligibility the rule multiplied the error with.
    elig = np.array([eligible(r, trace, error_at) for r in recs])
    sel["eligibility"] = selectivity(elig, labels)
    for name, s in sel.items():
        # readout_inputs=output_state: the per-neuron selectivities cover the
        # outputs only, the first third of the weights
        t.record(f"{prefix}_corr_dw_{name}", corr(dw[:len(s)], s))
        if prefix == "t0":   # once per trial: how many neurons carry the direction, and when
            t.record(f"neurons_selective_{name}", int(np.sum(np.abs(s) > 0.5)))
    t.record(f"{prefix}_corr_dw_ridge", corr(dw, ridge_w))
    # Share of the weight change on the 10% most direction-selective neurons before the blank.
    top = np.argsort(-np.abs(sel["visible"]))[:max(1, len(dw) // 10)]
    t.record(f"{prefix}_dw_share_top_visible", float(np.abs(dw[top]).sum() / max(np.abs(dw).sum(), 1e-12)))
    t.record(f"{prefix}_dw_norm", float(np.linalg.norm(dw)))
    # Share of that eligibility left by activity before the readout interval.
    own = np.array([eligible(r, trace, error_at, from_readout=True) for r in recs])
    t.record(f"{prefix}_eligibility_pre_share", float(np.abs(elig - own).sum() / max(np.abs(elig).sum(), 1e-12)))
    if dump is not None:
        dump.update({f"{prefix}_dw": dw, f"{prefix}_sel_eligibility": sel["eligibility"]})
        for name in ("visible", "readout", "state"):
            if name in sel:
                dump[f"sel_{name}"] = sel[name]


def hidden_train(p, rec, train, rng, gap):
    """mode=hidden: the hidden layer and the readouts learn together online by
    feedback alignment from the error of each clip's prediction (applied once,
    after the last readout tick, as error_at=end); the hidden layer's input
    trace (hidden_trace, per tick) keeps the pixels of the visible frames
    eligible. The learned hidden weights and biases are copied into the
    recorder's network, which is then recorded like the frozen one. Returns
    the readouts' weights and the training curve."""
    q = dict(p, hidden_rule="fa")
    net, layers = vm.build(q, clips.PIXELS * (rec.window + 1), trainable=True)
    for l, h in enumerate(layers):
        net.set_learning_rule(h, exr.LearningRule.feedback_alignment(p["hidden_trace"]).with_bias())
        for i in range(p["width"]):   # start from the frozen network's weights (paired with mode=readout)
            net.set_weights(h, i, np.asarray(rec.net.weights(rec.layers[l], i), dtype=np.float32))
    orule = exr.LearningRule.feedback_alignment(0.0).with_bias()
    out = net.add_layer("out", vm.readout_spec(orule))
    net.connect(layers[-1], out)
    net.add_output(out)
    for j in range(2):
        net.set_weights(out, j, (np.asarray(net.weights(out, j)) * math.sqrt(3.0 / p["width"])).astype(np.float32))
    t = rec.ticks
    frames, infos, labels = train
    curve = []
    for _ in range(p["hidden_epochs"]):
        sq, hits, n = 0.0, 0, 0
        for i in rng.permutation(len(frames)):
            x = vm.windowed(frames[i], rec.window).astype(np.float32)
            lead = infos[i]["visible"] + gap
            target = targets(labels[i])
            for h in layers:
                net.reset_state(h)
            for h in layers + [out]:
                net.reset_traces(h)   # a new clip: empty eligibility traces
            # One prediction per clip (the mean readout output over the readout
            # ticks); its error is applied once, after the last tick.
            y = np.asarray(net.run(np.repeat(x, t, axis=0)))[lead * t:].mean(0)
            err = target - y
            net.apply_error(err.astype(np.float32), p["hidden_lr"])
            sq += float(err @ err) / 2
            n += 1
            hits += int(np.argmax(y) == labels[i])
        curve.append(dict(mse=sq / n, online_accuracy=hits / len(frames)))
    for l, h in enumerate(layers):
        bias = np.asarray(net.bias(h))
        for i in range(p["width"]):
            rec.net.set_weights(rec.layers[l], i, np.asarray(net.weights(h, i), dtype=np.float32))
            rec.net.set_bias(rec.layers[l], i, float(bias[i]))
    W, b = weights(net, out, p["width"])
    return W, b, curve


@nnt.experiment(
    name="video_memory_learned",
    description="video_memory with online readouts: immediate delta rule vs eligibility traces, probes, credit",
    tags=["temporal", "dynamic", "learning"],
    params={
        "model": ("E0", "the video_memory hidden network: E0 (E-R), C0 (ReLU), C1 (E-R reset every frame), ..."),
        "task": ("side", "video_memory task"),
        "gap": (1, "blank frames"),
        "width": (128, "hidden neurons"),
        "ticks": (4, "ticks per frame"),
        "train": (1000, "training clips"),
        "val": (500, "validation clips (learning rate, ridge penalty, probes)"),
        "test": (1000, "test clips"),
        "normalize": (True, "hidden layers: weighted sum / |w| (video_memory default)"),
        "resting_threshold": (0.2, "E-R resting threshold (video_memory default)"),
        "recovery": (0.9, "E-R recovery (video_memory default 0.9; separate sweep only)"),
        "recurrent_scale": (0.5, "D2R only"),
        "mode": ("readout", "readout (hidden frozen, readouts learn online) or hidden (hidden learns too)"),
        "traces": ("0,0.5,0.8,0.9,0.95,0.99", "readout eligibility-trace decays per tick (0 = immediate)"),
        "lrs": ("0.0001,0.0003,0.001,0.003,0.01,0.03", "learning rates tried, chosen per trace on validation"),
        "epochs": (10, "passes over the training clips"),
        "error_at": ("end", "end: one error per clip on the mean readout output (the prediction); tick: every readout tick"),
        "input_gain": ("sd", "sd: readout inputs scaled by 1 / sd of each hidden output (training clips); none: raw"),
        "hidden_trace": (0.0, "mode=hidden: the hidden layer's input-trace decay per tick"),
        "hidden_lr": (0.01, "mode=hidden: learning rate"),
        "hidden_epochs": (10, "mode=hidden: epochs of the joint training"),
        "readout_inputs": ("output", "output, or output_state: the readouts also read the top layer's thresholds "
                                     "and habituation streaks through a State layer"),
        "credit": (True, "record credit-assignment diagnostics"),
        "dump": ("", "if set, a directory for per-neuron diagnostics (npz)"),
    },
    trials=10,
    expect={},
)
def run(t):
    p = dict(t.params)
    exr.set_threads(1)
    gap = int(p["gap"])
    # mode=hidden: the recorder's layers get a bias (to receive the learned one);
    # their weights are drawn the same way.
    p["hidden_rule"] = "fa" if p["mode"] == "hidden" else "frozen"
    rec = FullRecorder(p, gap)   # built right after the runner's reseed: video_memory's weights
    data = {name: clips.make_set(p[name], gap, clips.split_seed(t.seed, s), p["task"])
            for s, name in enumerate(("train", "val", "test"))}

    # Balanced classes, checked and reported before any training.
    for name, (_, _, labels) in data.items():
        n_left, n_right = int(np.sum(labels == LEFT)), int(np.sum(labels == RIGHT))
        t.record(f"{name}_left", n_left)
        t.record(f"{name}_right", n_right)
        if abs(n_left - n_right) > 1:
            raise RuntimeError(f"{name}: classes not balanced ({n_left} LEFT, {n_right} RIGHT)")
    print("  classes (LEFT/RIGHT): " + ", ".join(
        f"{n} {int(np.sum(l == LEFT))}/{int(np.sum(l == RIGHT))}" for n, (_, _, l) in data.items()), flush=True)
    yte = data["test"][2]
    t.record("constant_accuracy", float(max(np.mean(yte == LEFT), np.mean(yte == RIGHT))))
    t.record("constant_balanced_accuracy", 0.5)

    mode = p["mode"]
    online = {}
    if mode == "hidden":
        hW, hb, hcurve = hidden_train(p, rec, data["train"], t.rng, gap)
        t.record("hidden_joint_online_accuracy_last", hcurve[-1]["online_accuracy"])
        t.record("hidden_joint_mse_first", hcurve[0]["mse"])
        t.record("hidden_joint_mse_last", hcurve[-1]["mse"])
    elif mode != "readout":
        raise ValueError("mode must be readout or hidden")
    splits = {name: ([rec.clip(f, i) for f, i in zip(frames, info)], labels)
              for name, (frames, info, labels) in data.items()}

    # The readouts' input gain: each hidden neuron's output divided by its
    # standard deviation over the training readout ticks (a fixed gain per
    # synapse, from the training clips only), or 1. The ridge readout and the
    # probes standardise their features anyway and do not change.
    width = rec.features
    gain = np.ones(width, np.float32)
    if p["input_gain"] == "sd":
        sd = np.concatenate([r["readout"] for r in splits["train"][0]]).std(0)
        gain = np.where(sd > 1e-6, 1.0 / np.maximum(sd, 1e-6), 1.0).astype(np.float32)
    elif p["input_gain"] != "none":
        raise ValueError("input_gain must be sd or none")
    for recs, _ in splits.values():
        for r in recs:
            r["ticks"] = r["ticks"] * gain
            r["readout"] = r["readout"] * gain

    # 1. Ridge readout (video_memory's, the offline ceiling for a linear readout of the outputs).
    val_acc, lam, rW, rb = ridge_readout(splits)
    for k, v in scores(rW, rb, *splits["test"]).items():
        t.record(f"ridge_{k}", v)
    t.record("ridge_lambda", lam)
    ridge_w = rW[:, RIGHT] - rW[:, LEFT]

    # 2, 3. Online readouts: immediate (trace 0) and eligibility traces, from
    # the same recordings and initial weights; the learning rate per trace by
    # validation accuracy.
    init_seed = (t.seed * 7919 + 17) & 0xFFFFFFFF
    for trace in parse_list(p["traces"]):
        best = None
        for lr in parse_list(p["lrs"]):
            W, b, curve = train_online(*splits["train"], trace, lr, p["epochs"], width, init_seed,
                                       np.random.default_rng([t.seed, 101]), val=splits["val"],
                                       error_at=p["error_at"])
            acc = scores(W, b, *splits["val"])["accuracy"]
            if best is None or acc > best[0]:
                best = (acc, W, b, curve, lr)
        online[f"t{trace:g}"] = (best[1], best[2], best[3], best[4], trace)

    W0, _ = weights(*online_net(width, 0.0, init_seed), width)
    dump = {} if p["dump"] else None
    for key, (W, b, curve, lr, trace) in online.items():
        for k, v in scores(W, b, *splits["test"]).items():
            t.record(f"{key}_{k}", v)
        t.record(f"{key}_train_accuracy", scores(W, b, *splits["train"])["accuracy"])
        t.record(f"{key}_val_accuracy", scores(W, b, *splits["val"])["accuracy"])
        t.record(f"{key}_lr", lr)
        t.record(f"{key}_mse_first", curve[0]["mse"])
        t.record(f"{key}_mse_last", curve[-1]["mse"])
        t.record(f"{key}_online_accuracy_last", curve[-1]["online_accuracy"])
        for e, c in enumerate(curve):
            t.record(f"{key}_mse_epoch{e + 1}", c["mse"])
        if p["credit"]:
            credit(t, key, W0, W, *splits["train"], trace, ridge_w, dump, p["error_at"])

    # 4. Probes (network unchanged): what the state holds, independent of the readouts.
    def stack(name, key):
        return np.array([r[key] for r in splits[name][0]])

    labels = {n: splits[n][1] for n in splits}

    def probe(key):
        return vm.probe(stack("train", key), labels["train"], stack("val", key), labels["val"],
                        stack("test", key), labels["test"])

    t.record("probe_before_reappearance", probe("before"))
    t.record("probe_prediction_state", probe("prediction_state"))
    for r_list in (s[0] for s in splits.values()):
        for r in r_list:
            r["readout_mean"] = r["readout"].mean(0)
    t.record("probe_prediction_outputs", probe("readout_mean"))
    for trace in parse_list(p["traces"]):
        if trace == 0:
            continue
        for r_list in (s[0] for s in splits.values()):
            for r in r_list:
                r["elig"] = eligible(r, trace, p["error_at"])
        t.record(f"probe_eligibility_t{trace:g}", probe("elig"))
    t.record("spikes_visible", np.mean([r["spikes_visible"] for r in splits["test"][0]]))
    t.record("train_clips", len(splits["train"][1]))
    if dump is not None:
        os.makedirs(p["dump"], exist_ok=True)
        np.savez_compressed(os.path.join(p["dump"], f"{p['model']}_r{p['recovery']}_gap{gap}_seed{t.seed}.npz"),
                            ridge_w=ridge_w, **dump)
