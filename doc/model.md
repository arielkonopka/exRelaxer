# The model

A formal description of what exRelaxer computes: one neuron's dynamics,
the learning rules, and the timing of one network tick. It describes the
code as it is (`core/neuron.cpp`, `core/layers/neuron_layer.cpp`,
`core/layers/dense.cpp`, `core/network.cpp`), nothing more. The other pages
explain the API and the options and refer here for the definitions:
[neuron](neuron.md), [learning](learning.md), [network](network.md),
[dense](dense.md), [kernels](kernels.md).

Timing claims on this page are checked by `tests/temporal.cpp` and the
trace formulas by `tests/traces.cpp`.

## Notation and time

| Symbol | Meaning |
|---|---|
| *t* | one network tick: one call of `network::step()` |
| *t*−1 | the previous tick |
| x_j(*t*) | input *j* of a neuron at tick *t*: the value it summed in its forward pass at *t* (a sensor, or another neuron's output; see [which value a neuron reads](#which-value-a-neuron-reads)) |
| w_ij | weight from input *j* to neuron *i*; w(*t*−1) before the learning call that follows step *t*, w(*t*) after it |
| s_i(*t*) | neuron *i*'s effective sum at *t* (after normalisation, bias, noise, clamp, habituation) |
| y_i(*t*) | neuron *i*'s **output** at *t*: the value readers see |
| θ_i(*t*) | E-R **threshold** after tick *t* |
| ρ | resting (**baseline**) threshold, `LayerSpec::restingThreshold`, default 0.2 |
| r | E-R recovery, per neuron, default 0.9 |
| c_i(*t*), p_i(*t*) | habituation streak and the previous raw sum it compares against |
| P_i(*t*), X_ij(*t*) | **eligibility traces**: output trace and input trace |
| e_i(*t*) | the Sign rule's **eligibility** (from θ, see below) |
| m_i | the **modulator** of a learning call: the reward, or neuron *i*'s share of an error |
| η, γ_i | learning rate (per call) and the neuron's learning gain (default 2) |

**Pre-state** of tick *t*: everything a neuron holds before step *t* runs:
θ(*t*−1), c(*t*−1), p(*t*−1), P(*t*−1), X(*t*−1), y(*t*−1) and w(*t*−1).
**Post-state**: the same after step *t*: θ(*t*), c(*t*), p(*t*), P(*t*),
X(*t*), y(*t*). Weights are not changed by `step()`; they change only in a
learning call.

**Reward / error at *t***: the scalar r(*t*) or vector e(*t*) the caller
passes to `applyReward` / `applyError` after step *t* (and before step
*t*+1). The library does not store rewards: a reward always acts on the
post-state of the tick it follows. A "delayed" reward is simply a reward
passed after a later step, and it reaches an earlier event only through the
traces or the E-R threshold.

**Feedback** has two meanings in the code: a *feedback edge*
(`addFeedback(from, to, width)`: new neurons in `to` read `from`'s outputs;
it is a wiring operation with no built-in delay, see below) and
*feedback alignment* (a learning rule: a fixed random matrix projects the
output error onto a hidden layer). They are unrelated.

## One neuron

### Weighted input

For a neuron of a layer made of neurons (`dense`, `conv2d`,
`locally_connected2d`, `retina`, `cochlea`), with its input pool
x(*t*) and weight row w(*t*−1):

    a_i(t) = Σ_j w_ij(t-1) · x_j(t)                        summed in index order from 0
    a_i(t) ← a_i(t) / ‖w_i‖       if the layer is normalised and ‖w_i‖ > 1e-6
    a_i(t) ← a_i(t) + b_i        if the rule has a bias (b_i learned, input 1)
    a_i(t) ← a_i(t) + ξ_i(t)     Perturbation rule: exploration noise, sd σ, drawn at the start of the layer's forward
    s̃_i(t) = clamp(a_i(t), −10, 10)                        (max_output)

The sum order is fixed, so the SIMD and scalar paths agree bit for bit
([kernels](kernels.md#determinism)).

### Habituation (neurons with habituation)

With tolerance τ (default 0), ε = 1e-10:

    same(t)  = |s̃(t) − p(t−1)| ≤ max(ε, τ · max(|s̃(t)|, |p(t−1)|))
    c(t)     = same(t) ? c(t−1) + 1 : 0
    p(t)     = s̃(t)
    h(t)     = c(t) − onset + 1        onset = steps (cut mode) or fadeAfter (fade mode)
    s(t)     = s̃(t)                     if h(t) ≤ 0
             = 0                        cut mode (decay = 0), h(t) > 0
             = s̃(t) · decay^h(t)       fade mode (decay > 0), h(t) > 0

Without habituation, s(t) = s̃(t).

### Activation and E-R threshold dynamics (neurons with E-R)

θ(0) = ρ. The firing decision at *t* uses the threshold left by *t*−1:

    if |s(t)| > θ(t−1):                                   (fires)
        y(t) = s(t)
        θ(t) = max(θ(t−1), 2ρ, G(θ(t−1), |s(t)|))
    else:                                                  (silent)
        y(t) = 0
        θ'   = r · θ(t−1)                                  (relaxation, towards 0, not towards ρ)
        if θ' ≤ β  or  u(t) < q:                           (spontaneous activation)
            y(t) = v(t),   v(t) ~ Uniform(−A, A)
            θ(t) = max(θ', 2ρ, G(θ', |v(t)|))
        else:
            θ(t) = θ'

The output of a firing neuron is its full sum, not the sum minus the
threshold. Growth rules G(θ, s) (`ThresholdGrowth`, default Linear with
a = 0.5): Linear θ + a(s − θ); Log θ + α ln(s/θ); Fixed θ + a;
Multiplicative θ(1 + a).

**Spontaneous activation** (`Spontaneous`): β = `below` (default 1e-10),
A = `amplitude` (default 0.1), q = `rate` (default 0: no draw is made).
u(*t*) and v(*t*) come from the neuron's own generator (`minstd_rand`,
seeded from the spontaneous stream when the neuron is created, saved with
the neuron).

The threshold is the only E-R state. It carries history in two ways: a
neuron that fired recently needs a stronger sum to fire again, and every
threshold relaxes by the same factor r per silent tick, so after k silent
ticks θ = θ₀ rᵏ.

### Activation without E-R

    y(t) = s(t)                  linear (gate 0, not rectified; the default)
    y(t) = s(t) if |s(t)| > g, else 0        fixed gate g > 0
    y(t) = s(t) if s(t) > g, else 0          rectified (ReLU; g = 0 by default)

These neurons keep no state besides habituation's c and p.

### Eligibility traces

Every rule except Sign keeps, per neuron and per input of its group,
updated in the forward pass at *t* (λ = `LearningRule::trace`, in [0, 1);
all traces start at 0):

    P_i(t)  = λ · P_i(t−1) + y_i(t)                 output trace
    X_ij(t) = λ · X_ij(t−1) + x_j(t)                input trace (per input, in the group's pool order)
    Z_i(t)  = λ · Z_i(t−1) + ξ_i(t)                 noise trace (Perturbation)
    θM_i   ← θM_i + κ · (P_i(t)² − θM_i)            BCM sliding threshold, κ = bcmRate

With λ = 0 the traces are this tick's values. For an isolated input at
t₀, X(t₀ + k) = λᵏ; under a constant input c, X → c / (1 − λ). Inputs are
not clamped; outputs are, so |P| ≤ 10 / (1 − λ). Measured on the float
implementation (`TraceTest.LongSilenceDecaysIntoDenormalsAndCanStickThere`):
after a long silence a trace decays into the denormal range and, for λ
close to 1, stops at a few denormal steps above 0 (7.0e-44 for λ = 0.99)
instead of reaching exactly 0, because λ·X rounds back to X there.
`clearTraces()` (Python `reset_traces`) sets P, X and Z to 0 and a rule
change resets them. Learning calls read traces and never advance them.

### The Sign rule's eligibility

The Sign rule keeps no traces. Its eligibility is read from the post-state
of the last tick:

    E-R:       eligible(t) = θ(t) > ρ,           e(t) = θ(t)/ρ − 1
    no E-R:    eligible(t) = |y(t)| > 1e-6,      e(t) = 1

After one firing of magnitude s at t₀ (Linear growth), θ(t₀) = max(2ρ,
ρ + ½(s − ρ)), and the neuron stays eligible for k silent ticks while
θ(t₀) rᵏ > ρ. For s = 1, ρ = 0.2: θ = 0.6, eligible for 10 ticks at
r = 0.9, 36 at r = 0.97, 109 at r = 0.99
(`TraceTest.SignEligibilityWindowFollowsRecovery`). This window is on the
neuron side only: see the input factor below.

## Learning update

A learning call after step *t* computes, for each neuron *i* of each
unfrozen layer, a step δ_i, a keep factor κ_i (1 unless noted) and whether
the neuron learns, then

    w_ij(t) = clamp(w_ij(t−1) · κ_i + δ_i · pre_ij(t), −10, 10)      (max_weight)
    b_i     = clamp(b_i · κ_i + δ_i, −10, 10)                         (bias: its input is 1)

| Rule | δ_i | pre_ij(*t*) | learns when |
|---|---|---|---|
| Sign | η γ_i m_i e_i(*t*) | sign(x_j(*t*)) | eligible(*t*) |
| Trace | η γ_i (m_i − b̄_i) \|P_i(*t*)\| | X_ij(*t*) | δ ≠ 0 |
| Feedback alignment | η γ_i m_i | X_ij(*t*) | surrogate: E-R eligible(*t*) (with E-R), y(*t*) ≠ 0 (gate or ReLU), always (linear); not while y is at ±10 and m pushes further out |
| Perturbation | η γ_i (m_i − b̄_i) Z_i(*t*) / σ | X_ij(*t*) | δ ≠ 0 |
| Oja | η γ_i P_i(*t*), κ_i = 1 − η γ_i P_i(*t*)² | X_ij(*t*) | all, or the `winners` largest \|P\| |
| BCM | η γ_i P_i(*t*) (P_i(*t*) − θM_i) | X_ij(*t*) | all, or the `winners` largest \|P\| |

Weight decay multiplies κ_i by max(0, 1 − η · decay) for neurons that
learn. The reward baseline follows the modulator after every call:
b̄_i ← b̄_i + baseline · (m_i − b̄_i). `conv2d` shares a kernel among
positions and applies the mean of its neurons' updates:
w ← w · mean(κ) + mean(δ · pre) over the neurons that learn.

x_j(*t*) in the Sign rule is the value the neuron summed at *t* (the
group's input snapshot from its last forward pass), whatever the source
holds by the time learning runs. Before 2026-09-30 the Sign rule re-read
the sources at learning time; see
[learning reads the forward snapshot](#learning-reads-the-forward-snapshot).

### Modulators

- `applyReward(r)`: m_i = r for every neuron of every unfrozen layer.
- `applyError(e)`, with e = desired − actual, one entry per output neuron:
  an output layer gets m_i = e_i (its own entries); a hidden
  FeedbackAlignment layer m_i = (B e)_i / √dim(e), with B a fixed random
  matrix (uniform in [−1, 1], drawn once from the learning stream); a
  Perturbation layer the scalar m = −½ Σ e²; Oja and BCM ignore the error;
  hidden Sign and Trace layers do not learn from errors.
- `applyModulators(m)` (per layer): one modulator per neuron.

Layers learn in layer-id order, and a learning call reads only post-state
and writes only weights, biases and baselines, so the order does not
matter. Nothing propagates an error through time: a learning call at *t*
sees earlier ticks only through the traces and the thresholds.

### Feedback propagation

A feedback edge `addFeedback(from, to, width)` adds `width` neurons to
`to` that read `from`'s output buffer. The value they read is set by the
update order, like every other connection (next section). Errors are never
sent back through feedback edges; feedback alignment is the only way an
error reaches a hidden layer, through B, within the same learning call.

## One tick

`network::step()` followed by a learning call, in this order:

1. The caller sets the sensors: x_sensor(*t*) (`setInputs`). Sensors keep
   their value until set again.
2. `step()`: every layer's `forward()` once, in **update order**
   (default: a topological order of the `connect` edges, the layer created
   first breaking ties; or `setUpdateOrder`). For each layer:
   1. draw this tick's exploration noise ξ(*t*) (Perturbation) and refresh
      cached weight norms if the weights changed;
   2. for each wiring group, in creation order: copy the group's whole input
      pool into its snapshot (the values the source buffers hold *now*),
      advance the input traces X(*t*), compute every row's sum with
      w(*t*−1), and for every neuron compute y(*t*), θ(*t*), c(*t*), p(*t*),
      P(*t*) (Z, θM) and write y(*t*) to the layer's output buffer.
3. The caller reads y(*t*) (`outputs()`), acts, and observes the reward
   r(*t*) or error e(*t*) produced by that output.
4. `applyReward(r(t))` / `applyError(e(t))`: w(*t*−1) → w(*t*) from the
   post-state of step *t*. Outputs, thresholds, habituation state and
   traces are unchanged.
5. Step *t*+1 uses w(*t*).

A learning call is optional; several learning calls after one step all use
the same post-state and snapshot.

### Which value a neuron reads

A neuron's input x_j(*t*) is its source's buffer at the moment the neuron's
group copies its pool:

| Source | Runs | x_j(*t*) is |
|---|---|---|
| a sensor | set before the step | x_sensor(*t*) |
| another layer | earlier in the update order | y(*t*) (same tick) |
| another layer | later in the update order | y(*t*−1) |
| the same layer (`connect(x, x)`, `addFeedback(x, x, n)`), its own group | now | y(*t*−1) |
| the same layer, an earlier group | now | y(*t*) |
| the same layer, a later group | now | y(*t*−1) |

So with the default order a forward edge is same-tick and a feedback edge
from a later layer is one tick late; a loop of *k* layers adds exactly one
tick per turn. The delay belongs to the update order, not to the edge kind:
a feedback edge whose source runs first is same-tick
(`TemporalTest.TheUpdateOrderDecidesWhichEdgeCarriesTheDelay`). A neuron
reading both earlier and later layers mixes *t* and *t*−1 values in one sum.
Within one group the update is synchronous: no neuron sees a partial update
of its own group.

### Learning reads the forward snapshot

Every rule learns from what the neuron summed at *t* (the snapshot, or
traces advanced from it), and from its post-state at *t*. A neuron is
therefore never trained on a value that was produced later in the same
tick, even when its source runs after it or it reads itself, and changing
the sensors between `step()` and `applyReward()` does not change what it
learns (`TemporalTest.SignRuleLearnsFromTheInputsTheForwardPassSummed`,
`SignRuleIgnoresSensorsChangedAfterTheStep`,
`RecurrentSignLearningUsesItsOwnPreviousOutput`).

This was not true of the Sign rule before 2026-09-30: it re-read each
group's sources at learning time, so a group reading a later layer or
itself learned from sign(y(*t*)) while its sum had used y(*t*−1), and a
change of sensors between step and reward was learned as well
([appendix 1](appendix1.md), Q1). The fix changes results only in those
two cases; feed-forward Sign networks driven as set inputs → step →
reward are bit-identical (the regression tests and the nntest quick
suites pass unchanged; `KernelsTest.DenseMatchesPerNeuronReferenceBitForBit`,
whose reference re-read a self-connected layer after the step, now reads
the pre-step values). Inputs added to a group after its last forward pass
count as 0 in the next learning call.

### Delayed reward

A reward passed after step *t*₀ + d credits the event at *t*₀ only through
the state that survives d ticks:

- Trace rules: δ ∝ |P(*t*₀+d)| and pre = X(*t*₀+d), so an isolated event
  contributes λᵈ · λᵈ y(*t*₀) x(*t*₀) (λ²ᵈ, both traces decay), plus
  whatever the traces picked up in between
  (`TemporalTest.DelayedRewardWithTracesCreditsTheEventThroughLambdaPowers`).
- Sign: e(*t*₀+d) can still be positive from a firing at *t*₀ (the
  threshold window above), but pre = sign(x(*t*₀+d)): the credit goes to
  the inputs active at the reward tick, not to the event's inputs
  (`TemporalTest.DelayedRewardWithTheSignRuleCreditsOnlyThePresentInputs`).

How far this reaches in practice is measured by the
[delayed-credit benchmark](research.md#25-temporal-semantics-state-readability-and-delayed-credit).

## Inspecting the state

Read-only probes, for experiments; nothing reads them back into the
network, and the default outputs do not include them:

| C++ | Python | Value |
|---|---|---|
| `neuron::output()` | `state_probe(l)["output"]` | y(*t*) |
| `neuron::threshold()` | `["threshold"]`, `neuron_state(l)["threshold"]` | θ(*t*) |
| `neuron::restingThreshold()` | `["resting_threshold"]` | ρ |
| `neuron::eligibility()` | `["eligibility"]` | e(*t*) (Sign) |
| `neuron_layer::outputTrace(i)` | `["output_trace"]` | P(*t*) |
| `neuron::habituationStreak()`, `previousInput()` | `["habituation_streak"]`, `["previous_input"]` | c(*t*), p(*t*) |
| `dense::lastInputs(i)` | `last_inputs(l, i)` | x(*t*) as summed |
| `dense::inputTrace(i)` | `input_trace(l, i)` | X(*t*) |

## Determinism

Given the same seed (`reseed`, or nntest's per-trial seed), the same
build and the same call sequence, every value above is reproduced bit for
bit, whatever the OpenMP thread count and whichever SIMD width the kernels
use ([kernels](kernels.md#determinism); `build_info()["simd"]` names the
variant). Known limits, unchanged: random streams are process-global
(reseed before building each network), `std` distributions and libm may
differ between standard libraries, and network files are native-endian.
The rules for experiments are in [protocol](protocol.md).

## Not implemented

For clarity, none of these exist in the code: synaptic or axonal delays
other than those the update order creates; backpropagation, through layers
or through time; a refractory period separate from the E-R threshold; a
threshold that relaxes towards ρ (it relaxes towards 0); reward storage or
reward prediction inside the library; any coupling between neurons other
than through weighted sums of outputs.
