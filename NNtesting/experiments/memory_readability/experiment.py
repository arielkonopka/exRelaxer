"""Memory readability: after a LEFT or RIGHT event and a blank period, is the
direction (1) stored in the hidden layer's state, (2) readable from what
downstream neurons can see (the outputs), and (3) usable by a readout that
learns online with the library's own rule? See README.md.

    inputs [L, R, Q] -> hidden (width neurons, frozen random weights) -> readouts

One episode (the hidden layer is reset to rest before it):

    event   1 tick    L = 1 (LEFT) or R = 1 (RIGHT), Q = 0
    blank   `blank` ticks, L = R = Q = 0
    query   1 tick    Q = 1, L = R = 0: the same for both classes
    (every input also gets Gaussian noise of sd `noise` on every tick)

The query input carries no information about the class, so the class can
only come from the state the event left behind. Classes are exactly
balanced in every split (half LEFT, half RIGHT, shuffled).

Readouts, all at the query tick, from the same episodes (same seed):

    A  output         the hidden outputs at the query tick (what a downstream neuron reads)
    B  state          the hidden E-R thresholds after the query tick
    C  output+state   both
    E  eligibility    the Sign rule's eligibility (threshold / rest - 1, 0 at or below rest)
    S  state_before   the thresholds at the end of the blank, before the query

Each is fitted two ways: ridge regression (the best linear readout, fitted
offline; penalty on a validation split) and an online readout layer trained
by the library's own error rule (apply_error, feedback alignment on an
output layer = the delta rule, with a bias) over standardised features.
The state readouts (B, C, E, S) are an experimental probe: in the
library, downstream neurons only ever read outputs.

    stored    = ridge on S or B (the information is in the state)
    readable  = ridge on A (a downstream linear neuron could read it)
    usable    = online readout on A (the library's own learning finds it)

Controls: model=reset (the hidden layer is reset to rest after the event:
nothing can be stored, all readouts must be at chance) and model=relu
(stateless rectified neurons: no threshold state).
"""
import numpy as np

import exrelaxer as exr
from exrelaxer import harness as nnt

LAMBDAS = (1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0)
READOUTS = ("output", "state", "output_state", "eligibility", "state_before")


def build(p):
    er = p["model"] in ("er", "reset")
    spec = exr.LayerSpec.dense(p["width"], False, er, frozen=True)
    spec.normalize = False
    if er and p["recovery"] != exr.constants.recovery_factor:
        spec.recovery_jitter = exr.Jitter.uniform(1e-7).around(p["recovery"])
    if not er:
        spec.rectify = True
    net = exr.Network()
    h = net.add_layer("h", spec)
    net.add_inputs(h, 3, "x")
    net.add_output(h)
    return net, h


def ridge_fit(X, Y, lam):
    mx, sx = X.mean(0), X.std(0)
    sx[sx < 1e-12] = 1.0
    my = Y.mean(0)
    Xs, Yc = (X - mx) / sx, Y - my
    A = Xs.T @ Xs + lam * len(X) * np.eye(X.shape[1])
    W = np.linalg.solve(A, Xs.T @ Yc) / sx[:, None]
    return W, my - mx @ W


def scores(pred, y):
    """accuracy, balanced accuracy, confusion (LEFT=1 as positive)."""
    tp = int(np.sum((pred == 1) & (y == 1)))
    fn = int(np.sum((pred == 0) & (y == 1)))
    fp = int(np.sum((pred == 1) & (y == 0)))
    tn = int(np.sum((pred == 0) & (y == 0)))
    acc = (tp + tn) / len(y)
    bal = 0.5 * (tp / max(tp + fn, 1) + tn / max(tn + fp, 1))
    return acc, bal, (tp, fn, fp, tn)


def ridge_readout(tr, ytr, va, yva, te):
    best = None
    for lam in LAMBDAS:
        W, b = ridge_fit(tr, (2.0 * ytr - 1.0)[:, None], lam)
        acc = np.mean(((va @ W + b)[:, 0] > 0) == (yva == 1))
        if best is None or acc > best[0]:
            best = (acc, W, b)
    _, W, b = best
    return ((te @ W + b)[:, 0] > 0).astype(int)


def online_readout(tr, ytr, te, p, rng):
    """Two plain readout neurons (LEFT, RIGHT; raw sum + bias) trained online
    with apply_error on standardised features; the larger output wins."""
    mx, sx = tr.mean(0), tr.std(0)
    sx[sx < 1e-12] = 1.0
    f = lambda X: ((X - mx) / sx).astype(np.float32)
    trs, tes = f(tr), f(te)
    spec = exr.LayerSpec.dense(2, False, False, learning_rule=exr.LearningRule.feedback_alignment().with_bias())
    spec.normalize = False
    net = exr.Network()
    out = net.add_layer("out", spec)
    net.add_inputs(out, tr.shape[1], "f")
    net.add_output(out)
    for j in range(2):
        net.set_weights(out, j, np.zeros(tr.shape[1], np.float32))
    lr = p["online_lr"] / tr.shape[1]
    for _ in range(p["online_epochs"]):
        for i in rng.permutation(len(trs)):
            net.set_inputs("f", trs[i])
            net.step()
            target = np.array([1.0, -1.0] if ytr[i] == 1 else [-1.0, 1.0], np.float32)
            net.apply_error(target - np.asarray(net.outputs()), lr)
    y = np.asarray(net.run(tes))
    return (y[:, 0] > y[:, 1]).astype(int)


def episodes(net, h, p, n, rng):
    """n balanced episodes; returns labels (1 = LEFT) and the features per readout."""
    labels = rng.permutation(np.resize([1, 0], n))
    feats = {k: [] for k in READOUTS}
    fired_event = []
    for lab in labels:
        net.reset_state(h)
        rows = np.zeros((p["blank"] + 1, 3), np.float32)
        rows[0, 0 if lab == 1 else 1] = 1.0
        rows += rng.normal(0.0, p["noise"], rows.shape).astype(np.float32)
        net.set_inputs("x", rows[0])
        net.step()
        fired_event.append(np.mean(np.abs(net.state_probe(h)["output"]) > exr.constants.firing_epsilon))
        if p["model"] == "reset":
            net.reset_state(h)
        if p["blank"] > 0:
            net.run(rows[1:])
        before = np.asarray(net.state_probe(h)["threshold"], np.float64)
        q = np.array([0.0, 0.0, 1.0], np.float32) + rng.normal(0.0, p["noise"], 3).astype(np.float32)
        net.set_inputs("x", q)
        net.step()
        s = net.state_probe(h)
        out = np.asarray(s["output"], np.float64)
        thr = np.asarray(s["threshold"], np.float64)
        feats["output"].append(out)
        feats["state"].append(thr)
        feats["output_state"].append(np.concatenate([out, thr]))
        feats["eligibility"].append(np.asarray(s["eligibility"], np.float64))
        feats["state_before"].append(before)
    return labels, {k: np.array(v) for k, v in feats.items()}, float(np.mean(fired_event))


@nnt.experiment(
    name="memory_readability",
    description="LEFT/RIGHT event, blank, ambiguous query: is the direction stored, readable from outputs, usable?",
    tags=["temporal", "dynamic"],
    params={
        "model": ("er", "er (E-R hidden layer), reset (E-R reset after the event: control), relu (stateless control)"),
        "blank": (4, "blank ticks between the event and the query"),
        "width": (64, "hidden neurons"),
        "recovery": (0.9, "E-R recovery (threshold decay per silent tick)"),
        "noise": (0.05, "Gaussian input noise (sd) on every input, every tick"),
        "train": (400, "training episodes"),
        "val": (200, "validation episodes (ridge penalty)"),
        "test": (400, "test episodes"),
        "online_lr": (0.5, "online readout learning rate (divided by the feature count)"),
        "online_epochs": (10, "online readout epochs over the training episodes"),
    },
    trials=20,
    expect={},
)
def run(t):
    p = dict(t.params)
    if p["model"] not in ("er", "reset", "relu"):
        raise ValueError("model must be er, reset or relu")
    exr.set_threads(1)
    net, h = build(p)
    rng = t.rng
    # Frozen random weights, uniform in [-1, 1] (from the trial seed).
    for i in range(p["width"]):
        net.set_weights(h, i, rng.uniform(-1.0, 1.0, 3).astype(np.float32))
    ytr, ftr, fired = episodes(net, h, p, p["train"], rng)
    yva, fva, _ = episodes(net, h, p, p["val"], rng)
    yte, fte, _ = episodes(net, h, p, p["test"], rng)
    t.record("chance", max(np.mean(yte), 1 - np.mean(yte)))  # 0.5 by construction
    t.record("fired_at_event", fired)
    t.record("query_output_active", np.mean(np.abs(fte["output"]) > exr.constants.firing_epsilon))
    # How different the two classes' states are (mean over neurons of |mean LEFT - mean RIGHT| / pooled sd).
    for key in ("output", "state"):
        a, b = fte[key][yte == 1], fte[key][yte == 0]
        sd = np.sqrt(0.5 * (a.var(0) + b.var(0))) + 1e-3  # floor: noiseless states have sd 0
        t.record(f"separation_{key}", np.mean(np.abs(a.mean(0) - b.mean(0)) / sd))
    for key in READOUTS:
        for how in ("ridge", "online"):
            if how == "ridge":
                pred = ridge_readout(ftr[key], ytr, fva[key], yva, fte[key])
            else:
                pred = online_readout(ftr[key], ytr, fte[key], p, rng)
            acc, bal, (tp, fn, fp, tn) = scores(pred, yte)
            t.record(f"{how}_{key}_accuracy", acc)
            t.record(f"{how}_{key}_balanced", bal)
            for name, v in (("tp", tp), ("fn", fn), ("fp", fp), ("tn", tn)):
                t.record(f"{how}_{key}_{name}", v)
