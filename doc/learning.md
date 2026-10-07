# learning rules

`core/learning.hpp`, `core/learning.cpp`; the state and the update live in
[neuron_layer](layer.md#neuron_layer) (`core/layers/neuron_layer.*`), and
each layer type with weights applies them (`dense`, `conv2d`,
`locally_connected2d`).

Every layer made of neurons chooses how its weights learn. One network can
mix them: a layer of Oja feature detectors, a hidden layer trained by
feedback alignment and sign-rule readouts, for example. The default is the
original sign rule, and with it every result is the same, bit for bit, as
before the rules existed.

```cpp
LayerSpec hidden = LayerSpec::Dense(64, false, false);
hidden.learningRule = LearningRule::feedbackAlignment().withBias();
const auto h = net.addLayer("hidden", hidden);
...
net.setLearningRule(readout, LearningRule::traced(0.5f));  // later, on a built network
```

```python
spec = exr.LayerSpec.dense(64, False, False, learning_rule=exr.LearningRule.oja(winners=4))
net.set_learning_rule(readout, exr.LearningRule.traced(0.5))
```

## The update

Formal definitions (traces, eligibility, when a learning call reads what)
are in [model](model.md#learning-update).

Every rule changes the weights of each neuron `i` that learns in one call as

```
w[j] = clamp(w[j] * keep[i] + delta[i] * pre[j], ±max_weight)
```

with a per-neuron step `delta`, a per-input factor `pre` and a per-neuron
shrink factor `keep` (1 unless the rule or weight decay shrinks weights).
Rules differ in how they compute them from the neuron's **modulator** `m`:
the reward, or its own share of an error.

| Rule | `LearningRule::` | delta | pre | Who learns |
|------|------------------|-------|-----|-----------|
| Sign (default) | `sign()` | `rate · gain · m · eligibility` | `sign(x)` of the last forward (t) | E-R-eligible neurons |
| Trace | `traced(trace, baseline)` | `rate · gain · (m − b) · \|P\|` | `X` | neurons with a non-zero step |
| Feedback alignment | `feedbackAlignment(trace)` | `rate · gain · m` | `X` | active, not clamped outward |
| Perturbation | `perturbation(noise, trace, baseline)` | `rate · gain · (m − b) · Z / noise` | `X` | neurons with a non-zero step |
| Oja | `oja(winners)` | `rate · gain · P`, keep `1 − rate · gain · P²` | `X` | all, or the `winners` most active |
| BCM | `bcm(bcmRate, winners, decay)` | `rate · gain · P · (P − θ)` | `X` | all, or the `winners` most active |
| Eligibility | `eligibility(trace, baseline)` | `rate · gain · (m − b)` | `e[i][j]`, a trace per synapse | every neuron (dense only) |
| E-prop | `eprop(trace, width, baseline)` | `rate · gain · (m − b)` | `E[i][j]`, e-prop's eligibility | every neuron (dense only) |
| Surrogate | `surrogate(window, width)` | gradient descent through the last `window` ticks | (see below) | every neuron (dense only) |

- **Traces.** Every rule but Sign keeps a trace of each neuron's output
  `P ← trace · P + y` and of each input `X ← trace · X + x`, updated on
  every `forward()`. `trace = 0` (the default) means this tick's values.
  Sign uses the inputs as they are when learning happens, like before.
  `neuron_layer::clearTraces()` (Python `Network.reset_traces(layer)`)
  empties them, e.g. at the start of an episode, and keeps weights, bias,
  baselines and the feedback matrix.
- **Baseline** `b` (Trace, Perturbation): a running average of each
  neuron's modulator, `b ← b + baseline · (m − b)`, updated on every call. A
  constant reward stops teaching; only surprises do.
- **Gain** is the neuron's learning gain (`default_learning_gain`, or drawn
  from `learningJitter`).
- **Eligibility** (Sign) is `threshold / baseline_threshold − 1` with E-R and 1
  without, unchanged ([neuron](neuron.md)).

What each rule is for:

- **Sign** is the original rule: the reward's sign is the direction to move
  the output in, and only the sign of each input counts. A 0 input never
  teaches.
- **Trace** is a graded three-factor rule. Like Sign, the modulator is the
  direction to move the output in, but input magnitudes count and `|P|`
  grades how much the neuron took part. With `trace > 0` a reward can arrive
  some ticks after the activity it rewards.
- **Feedback alignment** gives hidden layers their own credit. It is driven
  by `network::applyError` (below): output layers learn their own error
  (the delta rule), and a hidden layer gets `m = B · errors / √outputs`
  through a fixed random matrix `B`, one row per neuron, drawn from the
  learning random stream the first time and saved with the network. No
  weight transport and no backward pass: it stays local. The surrogate
  derivative is 1 while the neuron takes part (with E-R: eligible; with a
  fixed threshold, `gate`, or rectification, `rectify`: firing; otherwise
  always) and 0 when it is silent
  or when it is held at `±max_output` in the direction it would
  be pushed.
- **Perturbation** (node perturbation) gives per-neuron credit from a single
  scalar reward. `forward()` adds noise `ξ` to each neuron's sum, and the
  update correlates the noise with how much better than usual the reward
  was. Noise is approximately normal (the sum of four 16-bit uniforms from
  one xorshift64* draw, within ±3.46 sd) from the layer's own generator,
  drawn serially in neuron order: deterministic and independent of the
  thread count. `m` is a fitness here: higher is better. It learns slowly in
  large layers.
- **Oja** is unsupervised normalised Hebbian learning: a neuron's weights
  converge to the first principal component of its inputs, with unit norm,
  so they stay bounded by themselves. The reward is ignored.
- **BCM** is unsupervised, with a sliding threshold `θ ← θ + bcmRate · (P² − θ)`
  updated every tick (starting at 1). Activity above θ strengthens the
  weights, and activity below weakens them. Its sliding threshold plays the
  same part as E-R's adaptive one.
- **Eligibility** is the reward-modulated (three-factor) rule with a trace
  per synapse instead of one per neuron and one per input: see below. A
  reward that arrives later teaches the synapses that were active together,
  not every pairing of an active neuron with an active input.
- **E-prop** follows a neuron's real dependence on each weight through its
  own state (the E-R threshold), so a weight that fired the neuron some
  ticks ago still gets credit while the threshold remembers it.
- **Surrogate** is truncated backpropagation through time with a
  pseudo-derivative for the threshold: exact gradients through the network
  and its recurrence, for the last `window` ticks. It is not local, and it
  is the reference the local rules can be measured against.

## Per-synapse rules

Eligibility, E-prop and Surrogate keep state per synapse, so they cost a
second (and E-prop a third) matrix the size of the weights. Only `dense`
supports them; `addLayer` and `setLearningRule` refuse them elsewhere.

**Eligibility.** On every `forward()`, each synapse updates

```
e[i][j] = trace · e[i][j] + |y_i| · x_j
```

and a learning call applies `w[i][j] += rate · gain · (m − b) · e[i][j]`.
`|y|` is the neuron's output, so only synapses whose neuron took part
collect credit; `x` keeps its sign. With `trace = 0` it is the Trace rule
with no traces. The default trace is 0.9 and the baseline 0.05.

**E-prop** (Bellec et al., 2020) uses the neuron's local derivatives
(`neuron::activate(sum, width, Derivatives&)`, probe
`network.derivatives(layer)`):

| Derivative | Meaning |
|------------|---------|
| `dyds` | output by the weighted sum, at this tick |
| `dydthr` | output by the E-R threshold |
| `dthrdthr` | the next threshold by this one (growth or recovery) |
| `dthrds` | the next threshold by the sum (it grows when the neuron fires) |

Firing is a step, so a triangular pseudo-derivative stands in for it:
height 1 at the threshold, falling to 0 at `width · threshold` either side
(`width`, default 0.5). Each synapse keeps the threshold's dependence on
the weight, `a`, and the eligibility `e`:

```
e[i][j] = dyds · x_j + dydthr · a[i][j]
a[i][j] = dthrdthr · a[i][j] + dthrds · x_j
E[i][j] = trace · E[i][j] + e[i][j]           (trace = e-prop's κ, default 0)
w[i][j] += rate · gain · (m − b) · E[i][j]
```

Without E-R (`dydthr = 0`) `e` is `dyds · x`: the delta rule with a
pseudo-derivative. Driven by `applyError`, an output layer gets its own
error and a hidden E-prop layer the feedback-alignment projection
(random e-prop); `applyReward` gives every neuron the reward (reward-based
e-prop, with the critic's TD error as the reward, below).

**Surrogate** keeps the last `window` (default 8, up to 1024) forward
passes of each Surrogate layer: inputs, outputs, thresholds and
derivatives. `network::applyError` (only it; `applyReward` does nothing
for this rule) takes the loss `½ Σ errors²` at the last tick and
propagates it back through every Surrogate layer, through each neuron's
threshold, through recurrent connections and back through time, then
takes one step `w −= rate · gain · ∂L/∂w` per layer. A source that runs
earlier in the update order passes the gradient to the same tick; one that
runs later, or the layer itself, passes it to the tick before. Layers with
another rule are a wall: gradients stop there. A frozen Surrogate layer
passes gradients on and does not change. With one layer and `window = 1`
it gives the same update as E-prop with `trace = 0`
(`tests/plasticity.cpp`).

The gradients treat normalisation (`LayerSpec::normalize`) as a constant
factor `1 / |w|`; set `normalize = false` for exact gradients.

## Actor-critic and curiosity

`core/reinforcement.hpp`. A network can hold one critic and one curiosity
model. Both read **features**: the outputs of chosen layers and the values
of named input sources, concatenated in the order given. They are linear,
learn on every call, are saved with the network (format 18) and follow
their layers when they grow or are pruned (new features start with weight
0).

**Critic** (`network::setCritic(CriticSpec{layers, inputs, gamma, lambda, rate})`):
a linear value `V(φ) = w · φ + b` learned by TD(λ). Each call of
`temporalDifference(r, terminal)`, after the step that received the reward
`r`, computes

```
δ = r + γ · V(φ_now) − V(φ_before)      (V(φ_now) = 0 when terminal)
z = γλ · z + φ_before
w += rate / (1 + |φ_before|²) · δ · z
```

The step is normalised by the feature norm, so `rate` (default 0.05) does
not depend on how many features there are. The first call after a reset
returns 0; a terminal call resets the traces and the previous state.
`applyRewardTD(r, rate, terminal)` passes `δ` to `applyReward`: the actor
(the layers, with any rule that takes a reward: Trace, Eligibility,
E-prop...) learns from the critic's surprise, not from the raw reward.
Defaults: γ 0.95, λ 0.8. `criticValue()` reads `V` of the current features.

**Curiosity** (`network::setCuriosity(CuriositySpec{predictLayers, predictInputs, fromLayers, fromInputs, rate, scale})`):
a linear forward model that predicts this tick's targets (e.g. the next
sensor values) from the previous tick's inputs (e.g. the sensors and the
hidden layer). `curiosityReward()`, called once per tick, learns from the
prediction error (normalised LMS, rate 0.1) and returns
`scale · mean(error²)`: an intrinsic reward that is high where the world
is still unpredictable and falls as the model learns it. Add it to the
game's reward:

```python
net.set_critic(layers=[h], inputs=["eye"], gamma=0.95)
net.set_curiosity(predict_inputs=["eye"], from_layers=[h], from_inputs=["eye"])
for t in range(steps):
    net.set_inputs("eye", observation)
    net.step()
    r = game_reward + 0.1 * net.curiosity_reward()
    net.apply_reward_td(r, 0.01, terminal=done)
```

## Options for every rule

| Field | Default | Effect |
|-------|---------|--------|
| `bias` | false | a learned bias per neuron, added to its weighted sum; learns like a weight whose input is always 1 (`pre = 1`) |
| `decay` | 0 | weight decay: `keep *= 1 − rate · decay` for neurons that learn (and their bias) |
| `trace` | 0 (`eligibility`: 0.9) | trace decay per tick, in [0, 1) |
| `baseline` | 0 (`traced`, `perturbation`, `eligibility`: 0.05) | reward-baseline rate, in [0, 1] |
| `noise` | 0.1 | Perturbation: noise standard deviation |
| `bcmRate` | 0.01 | BCM: sliding-threshold rate, in (0, 1] |
| `width` | 0.5 | E-prop, Surrogate: half-width of the pseudo-derivative, as a fraction of the threshold; > 0 |
| `window` | 8 | Surrogate: ticks of history to backpropagate through, 1–1024 |
| `winners` | 0 | Oja, BCM: only this many most active neurons learn (by `\|P\|`, ties to the lower index): in the whole layer for `dense`, per position across channels for spatial layers; 0 = all |

`LearningRule::validate()` throws `std::invalid_argument` for values outside
these ranges; `setLearningRule` and `network::addLayer` validate.

## Driving learning

| Call | Modulator |
|------|-----------|
| `network::applyReward(r, rate)` / `layer.applyReward(r, rate)` | `r` for every neuron |
| `neuron_layer::applyModulators(m, rate)` | one per neuron, e.g. each readout's own reward or error |
| `neuron_layer::applyFeedback(errors, rate)` | feedback alignment's projection of an error vector |
| `network::applyError(errors, rate)` | per layer, by its rule (below) |
| `network::applyRewardTD(r, rate, terminal)` | the critic's TD error for every layer ([above](#actor-critic-and-curiosity)) |

`network::applyError(errors, rate)` takes one error per output
(`errors.size() == outputs().size()`, error = desired − actual). Every
unfrozen layer that learns gets:

| The layer | Gets |
|-----------|------|
| Oja, BCM | an unsupervised update (the errors are ignored) |
| Perturbation | the reward `−½ Σ errors²` |
| an output layer (other rules) | its own errors, one per neuron (the first time it is marked as output) |
| Surrogate (output or hidden) | its gradient, backpropagated through the Surrogate layers and time |
| a hidden feedback-alignment or E-prop layer | its random projection of the errors |
| a hidden Sign, Trace or Eligibility layer | nothing: they need a scalar reward (use `applyReward`) |

So a network can use both: `applyError` for the layers that learn from
errors and `applyReward` on the layers that learn from rewards.

## Layer types

- `dense`: each group's neurons learn against the group's inputs (their
  signs as its last `forward()` saw them, or the group's input trace). The trace is kept per group, in
  pool order.
- `conv2d`: shared kernels learn the mean of their channel's updates:
  `w = w · mean(keep) + mean(delta · pre)` over the neurons that learn.
  Biases stay per neuron (per position).
- `locally_connected2d`: every neuron learns against its own window, like a
  dense neuron.
- Eligibility, E-prop and Surrogate: `dense` only (per-synapse state).
- `retina`, `cochlea` (neurons without weights) and `pool2d`, `history` do
  not learn. Only `sign()` is accepted for them.

## Serialization

Network format 9 saves each layer's rule with its `LayerSpec`, and neuron
format 3 appends the rule's state to every layer of neurons: bias, output
traces, noise, noise traces, baselines, BCM thresholds, the feedback matrix,
the input traces and the noise generator. `FullState` loads all of it, so a
loaded network continues exactly. `WeightsOnly` keeps the bias and the
feedback matrix and resets the rest. Older files load with the Sign rule.

Neuron format 4 (network format 18) adds `width`, `window`, the bias
eligibility, the frozen-neuron mask and, for `dense`, the frozen input
columns and the per-synapse eligibility matrices. Surrogate's history is
not saved: a loaded network starts with an empty window.

## Cost

With the default rule, forward and learning run at the same speed as
before (`dense_throughput`, `vision_throughput`: within ±3 %, the run-to-run
noise). The other rules keep traces on every forward pass and use a scaled
update. On a 1000 × 1000 dense pair with 4 threads (`nntest run
dense_throughput --set rule=...`):

| Rule | step | step + reward | step + error |
|------|------|---------------|--------------|
| sign | 0.036 ms | 0.134 ms | 0.122 ms |
| trace, oja, bcm | 0.030–0.044 ms | 0.13–0.16 ms | 0.13 ms |
| feedback alignment | 0.044 ms | 0.076 ms | 0.153 ms |
| perturbation | 0.044 ms | 0.142 ms | 0.143 ms |

The per-synapse rules update a matrix per synapse on every forward pass
(scalar loops). Measured on a different machine from the table above (4 threads), where learning
calls are slower for every rule, so read them against `sign` on the same
machine:

| Rule | step | step + reward | step + error |
|------|------|---------------|--------------|
| sign | 0.049 ms | 2.43 ms | 2.28 ms |
| trace | 0.048 ms | 2.40 ms | 2.29 ms |
| eligibility | 0.161 ms | 2.58 ms | 2.40 ms |
| eprop | 0.467 ms | 2.91 ms | 3.00 ms |
| surrogate (window 8) | 0.078 ms | 0.083 ms (no-op) | 5.85 ms |

Feedback alignment's error projection is a SIMD matrix-vector product (the
same kernel as the forward pass). Perturbation noise costs one xorshift draw
per neuron and tick.

## Results: snake

`nntest run snake_rules` plays [snake](../NNtesting/experiments/snake/README.md)
with a rule for the three readouts (`readout=`) and for the 64-neuron mixing
layer (`mix=`, or `frozen`). The results are in the
[research log](research.md#11-learning-rules).
