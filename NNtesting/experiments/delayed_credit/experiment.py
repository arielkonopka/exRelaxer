"""Delayed credit assignment: can one neuron's learning rule link a reward to
the synapse that was active `delay` ticks before it? See README.md.

    K cue inputs + M distractor inputs -> 1 neuron (the only learning layer)

One episode (the neuron's state and traces are reset before it):

    t0            one cue input is 1 (the others 0)
    t0+1..t0+d    cue inputs 0; each distractor input is +1 or -1 with
                  probability `distract` per tick, else 0
    t0+d          after the step: apply_reward(m, lr), the only learning call

    cue 0 -> m = +1          (the relevant synapse to strengthen)
    cue 1 -> m = -1          (the relevant synapse to weaken)
    cues 2..K-1 -> m = +1 or -1 at random, half each  (irrelevant cues)

Cues are presented equally often (a shuffled balanced list) and the reward
is +1 in exactly half the episodes. The distractors are irrelevant too, but
active right up to the reward, so a rule that credits whatever is active
when the reward arrives credits them.

The network is the simplest possible: the inputs feed the neuron directly,
every weight starts at w0 (plus a small jitter from the seed), there is no
hidden layer, no recurrence, no habituation and no weight normalisation.
Rules (library, unchanged): Sign (the E-R threshold is the eligibility; the
input factor is sign(x) at the reward tick) and Trace (P and X traces with
decay `trace`; see doc/model.md). The neuron is E-R (recovery `recovery`) or
plain linear.

After training: fresh episodes without learning, the neuron's output at t0
(its response to the cue) against the reward that followed.

state=tap (the State layer, doc/model.md#state-as-output): every input also
drives its own frozen E-R relay neuron (weight 1, no habituation, recovery
`relay_recovery`), and the learner reads, besides the inputs, the relays'
thresholds above rest (a State layer). A cue then leaves a decaying trace in
an input that is still non-zero when the reward arrives. The learner's
weights are [inputs..., relay thresholds...]; the success and weight metrics
are reported for both halves (``*_state`` for the second).
"""
import numpy as np

import exrelaxer as exr
from exrelaxer import harness as nnt

EPS = exr.constants.firing_epsilon


def build(p):
    """Returns the network, the learner, and the layers reset per episode."""
    spec = exr.LayerSpec.dense(1, False, p["neuron"] == "er")
    spec.normalize = False
    if p["rule"] == "trace":
        spec.learning_rule = exr.LearningRule.traced(p["trace"], 0.0)
    elif p["rule"] == "eligibility":
        spec.learning_rule = exr.LearningRule.eligibility(p["trace"], 0.0)
    elif p["rule"] == "eprop":
        spec.learning_rule = exr.LearningRule.eprop(p["trace"])
    elif p["rule"] != "sign":
        raise ValueError("rule must be sign, trace, eligibility or eprop")
    if p["neuron"] == "er" and p["recovery"] != exr.constants.recovery_factor:
        spec.recovery_jitter = exr.Jitter.uniform(1e-7).around(p["recovery"])
    if p["neuron"] not in ("er", "linear"):
        raise ValueError("neuron must be er or linear")
    net = exr.Network()
    size = p["cues"] + p["distractors"]
    if p["state"] == "none":
        n = net.add_layer("n", spec)
        net.add_inputs(n, size, "x")
        net.add_output(n)
        return net, n, [n]
    if p["state"] != "tap":
        raise ValueError("state must be none or tap")
    rspec = exr.LayerSpec.dense(size, False, True)
    rspec.normalize = False
    rspec.frozen = True
    if p["relay_recovery"] != exr.constants.recovery_factor:
        rspec.recovery_jitter = exr.Jitter.uniform(1e-7).around(p["relay_recovery"])
    relay = net.add_layer("relay", rspec)
    tap = net.add_layer("relay_state", exr.LayerSpec.state(output=False, threshold=True, habituation=False))
    n = net.add_layer("n", spec)
    net.add_inputs(relay, size, "x")
    net.connect(relay, tap)
    net.connect_inputs("x", n)
    net.connect(tap, n)
    net.add_output(n)
    for i in range(size):
        net.set_weights(relay, i, np.eye(size, dtype=np.float32)[i])
    return net, n, [relay, n]


def rewards_for(cues, rng):
    """+1 for cue 0, -1 for cue 1, a balanced random +-1 for the others."""
    m = np.where(cues == 0, 1.0, np.where(cues == 1, -1.0, 0.0))
    for c in np.unique(cues[cues >= 2]):
        idx = np.flatnonzero(cues == c)
        signs = np.resize([1.0, -1.0], len(idx))
        m[idx] = rng.permutation(signs)
    return m


def balanced_cues(n, k, rng):
    return rng.permutation(np.resize(np.arange(k), n))


def episode(net, n, reset, p, cue, rng):
    """Runs one episode; returns (output at t0, output trace |P| at the reward
    tick, input trace of the cue synapse at the reward tick, eligibility)."""
    k, m = p["cues"], p["distractors"]
    for layer in reset:
        net.reset_state(layer)
    net.reset_traces(n)
    x = np.zeros(k + m, np.float32)
    x[cue] = 1.0
    net.set_inputs("x", x)
    net.step()
    y0 = float(net.outputs()[0])
    for _ in range(p["delay"]):
        x = np.zeros(k + m, np.float32)
        on = rng.random(m) < p["distract"]
        x[k:] = np.where(on, rng.choice([-1.0, 1.0], m), 0.0)
        net.set_inputs("x", x)
        net.step()
    probe = net.state_probe(n)
    tr = net.input_trace(n, 0)
    x_cue = float(tr[cue]) if len(tr) else 0.0
    return y0, float(abs(probe["output_trace"][0])), x_cue, float(probe["eligibility"][0])


@nnt.experiment(
    name="delayed_credit",
    description="one neuron, a cue, `delay` distractor ticks, then a reward: does the cue's synapse get the credit?",
    tags=["temporal", "learning"],
    params={
        "rule": ("trace", "sign, trace, eligibility or eprop (library rules, unchanged)"),
        "trace": (0.9, "trace, eligibility: trace decay lambda; eprop: eligibility filter kappa"),
        "neuron": ("er", "er or linear"),
        "recovery": (0.9, "E-R recovery (threshold decay per silent tick)"),
        "delay": (4, "ticks between the cue (t0) and the reward (t0 + delay)"),
        "cues": (4, "cue inputs: cue 0 rewarded +1, cue 1 -1, the rest +-1 at random"),
        "distractors": (4, "distractor inputs, active between the cue and the reward"),
        "distract": (0.25, "probability per tick that a distractor is on (+1 or -1)"),
        "w0": (0.5, "initial weight of every synapse"),
        "jitter": (0.05, "initial weights: w0 + uniform(-jitter, jitter) from the seed"),
        "episodes": (400, "training episodes"),
        "test": (200, "test episodes (no learning)"),
        "lr": (0.01, "learning rate"),
        "state": ("none", "none, or tap: the learner also reads E-R relay thresholds (a State layer)"),
        "relay_recovery": (0.99, "state=tap: recovery of the relay neurons"),
    },
    trials=10,
    expect={},
)
def run(t):
    p = dict(t.params)
    exr.set_threads(1)
    net, n, reset = build(p)
    k, m = p["cues"], p["distractors"]
    halves = 1 if p["state"] == "none" else 2
    rng = t.rng
    w_init = (p["w0"] + rng.uniform(-p["jitter"], p["jitter"], halves * (k + m))).astype(np.float32)
    net.set_weights(n, 0, w_init)

    cues = balanced_cues(p["episodes"], k, rng)
    rewards = rewards_for(cues, rng)
    credit, post, elig = [], [], []
    for cue, r in zip(cues, rewards):
        _, P, X, e = episode(net, n, reset, p, int(cue), rng)
        credit.append(P * X)
        post.append(P)
        elig.append(e)
        net.apply_reward(float(r), p["lr"])
    w_all = np.asarray(net.weights(n, 0), dtype=np.float64)
    for half, suffix in enumerate(["", "_state"][:halves]):
        w = w_all[half * (k + m):(half + 1) * (k + m)]
        dw = w - w_init[half * (k + m):(half + 1) * (k + m)]
        irrelevant = np.concatenate([dw[2:k], dw[k:]])
        t.record("dw_relevant_pos" + suffix, dw[0])
        t.record("dw_relevant_neg" + suffix, dw[1])
        t.record("dw_irrelevant_cue_mean" + suffix, np.mean(dw[2:k]) if k > 2 else 0.0)
        t.record("dw_irrelevant_cue_abs" + suffix, np.mean(np.abs(dw[2:k])) if k > 2 else 0.0)
        t.record("dw_distractor_mean" + suffix, np.mean(dw[k:]) if m else 0.0)
        t.record("dw_distractor_abs" + suffix, np.mean(np.abs(dw[k:])) if m else 0.0)
        sel = 0.5 * (dw[0] - dw[1])
        spread = np.sqrt(np.mean(irrelevant ** 2)) if len(irrelevant) else 0.0
        t.record("selectivity" + suffix, sel)
        t.record("selectivity_snr" + suffix, sel / spread if spread > 0 else (np.inf if sel > 0 else 0.0))
        # Success: the rewarded cue's weight above every irrelevant weight, the
        # punished cue's below every one.
        t.record("success" + suffix,
                 float(w[0] > np.max(np.r_[w[2:k], w[k:]]) and w[1] < np.min(np.r_[w[2:k], w[k:]])))
    t.record("credit_mean", np.mean(credit))  # |P| * X_cue at the reward tick (trace rule)
    t.record("post_trace_mean", np.mean(post))
    t.record("eligibility_mean", np.mean(elig))  # Sign rule's E-R eligibility at the reward tick

    # Test: no learning. The response at t0 against the reward that follows.
    tcues = balanced_cues(p["test"], k, rng)
    trew = rewards_for(tcues, rng)
    y0 = np.array([episode(net, n, reset, p, int(c), rng)[0] for c in tcues])
    rel = tcues < 2

    def corr(a, b):
        return float(np.corrcoef(a, b)[0, 1]) if np.std(a) > 0 and np.std(b) > 0 else 0.0

    t.record("reward_corr", corr(y0, trew))
    t.record("reward_corr_relevant", corr(y0[rel], trew[rel]))
    # Balanced accuracy on the relevant cues: does the rewarded cue get the larger response?
    t.record("response_cue0", np.mean(y0[tcues == 0]))
    t.record("response_cue1", np.mean(y0[tcues == 1]))
    t.record("fires_cue0", np.mean(np.abs(y0[tcues == 0]) > EPS))
    t.record("fires_cue1", np.mean(np.abs(y0[tcues == 1]) > EPS))
