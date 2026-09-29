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
| Sign (default) | `sign()` | `rate · gain · m · eligibility` | `sign(x)` now | E-R-eligible neurons |
| Trace | `traced(trace, baseline)` | `rate · gain · (m − b) · \|P\|` | `X` | neurons with a non-zero step |
| Feedback alignment | `feedbackAlignment(trace)` | `rate · gain · m` | `X` | active, not clamped outward |
| Perturbation | `perturbation(noise, trace, baseline)` | `rate · gain · (m − b) · Z / noise` | `X` | neurons with a non-zero step |
| Oja | `oja(winners)` | `rate · gain · P`, keep `1 − rate · gain · P²` | `X` | all, or the `winners` most active |
| BCM | `bcm(bcmRate, winners, decay)` | `rate · gain · P · (P − θ)` | `X` | all, or the `winners` most active |

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

## Options for every rule

| Field | Default | Effect |
|-------|---------|--------|
| `bias` | false | a learned bias per neuron, added to its weighted sum; learns like a weight whose input is always 1 (`pre = 1`) |
| `decay` | 0 | weight decay: `keep *= 1 − rate · decay` for neurons that learn (and their bias) |
| `trace` | 0 | trace decay per tick, in [0, 1) |
| `baseline` | 0 (`traced`, `perturbation`: 0.05) | reward-baseline rate, in [0, 1] |
| `noise` | 0.1 | Perturbation: noise standard deviation |
| `bcmRate` | 0.01 | BCM: sliding-threshold rate, in (0, 1] |
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

`network::applyError(errors, rate)` takes one error per output
(`errors.size() == outputs().size()`, error = desired − actual). Every
unfrozen layer that learns gets:

| The layer | Gets |
|-----------|------|
| Oja, BCM | an unsupervised update (the errors are ignored) |
| Perturbation | the reward `−½ Σ errors²` |
| an output layer (other rules) | its own errors, one per neuron (the first time it is marked as output) |
| a hidden feedback-alignment layer | its random projection of the errors |
| a hidden Sign or Trace layer | nothing: they need a scalar reward (use `applyReward`) |

So a network can use both: `applyError` for the layers that learn from
errors and `applyReward` on the layers that learn from rewards.

## Layer types

- `dense`: each group's neurons learn against the group's inputs (their
  signs now, or the group's input trace). The trace is kept per group, in
  pool order.
- `conv2d`: shared kernels learn the mean of their channel's updates:
  `w = w · mean(keep) + mean(delta · pre)` over the neurons that learn.
  Biases stay per neuron (per position).
- `locally_connected2d`: every neuron learns against its own window, like a
  dense neuron.
- `retina`, `cochlea` (neurons without weights) and `pool2d`, `history` do
  not learn. Only `sign()` is accepted for them.

## Serialization

Network format 9 saves each layer's rule with its `LayerSpec`, and neuron
format 3 appends the rule's state to every layer of neurons: bias, output
traces, noise, noise traces, baselines, BCM thresholds, the feedback matrix,
the input traces and the noise generator. `FullState` loads all of it, so a
loaded network continues exactly. `WeightsOnly` keeps the bias and the
feedback matrix and resets the rest. Older files load with the Sign rule.

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

Feedback alignment's error projection is a SIMD matrix-vector product (the
same kernel as the forward pass). Perturbation noise costs one xorshift draw
per neuron and tick.

## Results: snake

`nntest run snake_rules` plays [snake](../NNtesting/experiments/snake/README.md)
with a rule for the three readouts (`readout=`) and for the 64-neuron mixing
layer (`mix=`, or `frozen`). The results are in the
[research log](research.md#11-learning-rules).
