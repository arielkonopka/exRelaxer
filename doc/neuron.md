# neuron

`core/neuron.hpp`, `core/neuron.cpp`

A single unit's **dynamics**: what happens to a weighted sum, through two
optional, independently switchable adaptation mechanisms, **habituation** and
**excitation–relaxation (E-R)**, plus the eligibility and step size of the
reward-modulated learning rule. Everything is in `namespace exr`.

A neuron does **not** own weights. Layers own them, laid out for fast SIMD
weighted sums ([kernels](kernels.md)), compute each neuron's sum and hand it
to `activate()`. Neurons are normally created and owned by a layer
([neuron_layer](layer.md#neuron_layer)); you rarely construct one yourself
except in tests and experiments, where `step()` and `learn()` run one neuron
against weights you keep yourself.

## Construction

```cpp
explicit neuron(bool hasHabituation = true, bool hasER = true, float alpha = default_alpha);
```

| Parameter | Meaning |
|-----------|---------|
| `hasHabituation` | enables habituation (see below) |
| `hasER` | enables excitation–relaxation (see below) |
| `alpha` | E-R threshold growth rate of the `Log` [growth rule](#excitationrelaxation-e-r); higher means the threshold climbs faster after a strong signal. Ignored by the other rules, including the default `Linear` |

A new neuron has output `0`, threshold `baseline_threshold`, and its own
spontaneous-firing random generator, seeded from a shared stream (see
[Randomness](#randomness-and-reseed)).

## One tick: `activate()`

```cpp
float activate(float weightedSum);   // returns the output
float output() const;
void setOutput(float value);         // drive the output by hand
```

A layer computes the neuron's weighted sum, `Σ inputs[i] × weights[i]` summed
in index order (so results are bit-reproducible, see
[kernels](kernels.md#determinism)), and passes it to `activate()`, which
runs these stages in order:

1. **Output clamp.** `sum` is clamped to `[-max_output, max_output]` (±10).
   Neurons have no other bounded activation; without the clamp, networks
   with feedback loops diverge within a few dozen ticks. The clamp is also
   the only non-adaptive nonlinearity, which is why hand-wired detectors use
   it (a high-gain neuron saturates at ±10).
2. **Habituation** (if enabled). If the sum is "the same" as the previous
   one, a streak counter increments; otherwise it resets to 0. Once the
   streak reaches the rule's onset, the sum is suppressed until the
   signal changes. The previous sum stored is the raw one, from before
   suppression. The rule (`Habituation`, set per layer through
   `LayerSpec::habituationRule` or with `neuron::setHabituation`) has two
   modes, cut (`decay` 0, the default) and fade (`decay` > 0), and four
   fields:
   - `steps`: cut mode, the streak length before the sum is cut to 0.
     Default `habituation_steps`, 100.
   - `tolerance`: "the same" means `|sum − previous| ≤ max(habituation_epsilon,
     tolerance × max(|sum|, |previous|))`. Default 0, i.e. exact; a small
     tolerance lets a flickering sensor habituate too.
   - `decay`: 0 (the default) is cut mode. A value in (0, 1] selects fade
     mode: a suppressed sum is scaled by `decay^(ticks habituated)`, so
     closer to 1 fades it more slowly.
   - `fadeAfter`: fade mode, the streak length before fading starts.
     Default 2: a repeated input starts fading on its second repeat.
     Networks saved before format 15 load with `fadeAfter = steps`, which
     is when fading started then.

   The defaults are the original behaviour. Note that the clamp comes
   first, so a neuron held at ±`max_output` sees an identical sum even when
   its input changes, and habituates.
3. **E-R** (if enabled):
   - if `|sum| > threshold`: the neuron **fires**. Output = `sum`, and the
     threshold is raised (see [E-R](#excitationrelaxation-e-r));
   - otherwise output = 0 and the threshold **relaxes**:
     `threshold ×= recovery`, the neuron's own recovery factor (default
     `recovery_factor`, 0.9; see [per-neuron dynamics](#per-neuron-dynamics)). If it falls to the
     spontaneous-firing level (`min_threshold` by default) or below, or a
     random chance comes up, the neuron fires **spontaneously** with a
     random value in `[-amplitude, +amplitude]`
     (`spontaneous_min_amplitude`, 0.01, by default), which raises the
     threshold again through the same rule as a real firing (a spontaneous
     firing weaker than the threshold leaves it unchanged). `Spontaneous`,
     set per layer through `LayerSpec::spontaneous` or with
     `neuron::setSpontaneous`, holds the three settings:
     - `below`: the threshold at or below which a silent neuron fires.
       Default `min_threshold`, 1e-10: about 200 silent ticks at recovery
       0.9. A higher level gives faster cycles: roughly
       `ln(0.4 / below) / ln(1 / recovery)` ticks between firings.
     - `amplitude`: the range of the random output. Default 0.01, too weak
       to drive other neurons; around the resting threshold (0.2) and above,
       spontaneous firings propagate.
     - `rate`: an extra probability of firing on any silent tick. Default 0
       (none); the random draw happens only when a rate is set, so the
       default keeps the generator's sequence.

4. **Gate** (only without E-R, if set). With `setGate(g)`, `g > 0`, the
   output is 0 whenever `|sum| ≤ g`: the same all-or-nothing firing as E-R,
   with a fixed threshold that never adapts. It is a control for
   experiments that separate thresholding from adaptation
   ([activity](activity.md)); layers set it through `LayerSpec::gate`.
5. **Rectification** (only without E-R, if set). With `setRectified(true)`
   the neuron is a ReLU: the output is 0 whenever `sum ≤ gate` (0 by
   default), otherwise `sum`. Off by default; layers set it through
   `LayerSpec::rectify`. It is the conventional baseline of the
   [nonlinearity experiments](nonlinearity.md).

Without E-R (and without a gate or rectification) the output is simply the (clamped,
possibly habituated) sum.

For one neuron with caller-owned weights, `step(inputs, weights)` is
`activate(dot(inputs, weights))`.

## Excitation–relaxation (E-R)

E-R models fatigue / spike-frequency adaptation: a neuron that fires becomes
harder to fire, and recovers while silent.

On every firing with value `v` (real or spontaneous), by default:

```
threshold = max(2 × baseline_threshold, threshold + amount × (|v| − threshold))
```

with `amount` 0.5: the threshold moves halfway towards the firing's
magnitude, so a firing far above the threshold raises it a lot, a marginal
one barely.

- The floor `2 × baseline_threshold` guarantees that right after any firing
  the threshold is above `baseline_threshold`, which is what makes the
  neuron eligible to learn (see below).
- `ThresholdGrowth` (per layer through `LayerSpec::thresholdGrowth`, or
  `neuron::setThresholdGrowth`) selects the rule, always with the same
  floor:

  | Rule | New threshold | Note |
  |------|---------------|------|
  | `Linear` (default) | `threshold + amount × (abs(v) − threshold)` | moves part of the way towards the firing's magnitude |
  | `Log` (the original) | `threshold + alpha × ln(abs(v) / threshold)` | the neuron's `alpha` sets the rate; the jump grows without bound as the threshold falls: after a long silence one firing makes the neuron refractory |
  | `Fixed` | `threshold + amount` | the same jump for every firing |
  | `Multiplicative` | `threshold × (1 + amount)` | in proportion to the threshold; ignores the magnitude |

  `amount` defaults to 0.5. `Log` and `Linear` keep a trace of how strong
  the firing was; `Fixed` and `Multiplicative` only that it happened.
  Linear became the default on 2026-09-27 after the temporal and silence
  experiments ([research log §16](research.md#16-temporal-tasks-and-the-threshold-growth-rule)).
  Networks saved in format 13 or earlier load with `Log`, the only rule
  they knew; `alpha` affects only `Log`.
- While silent the threshold decays geometrically (`× recovery` per tick,
  0.9 by default), so a neuron that fired strongly stays refractory for a
  while. Because the threshold keeps a trace of recent firing, E-R neurons
  carry some memory even with no connections between them (measured in
  [pattern_benchmark](pattern_benchmark.md#memory-without-hand-built-delay-lines)).

With a small `baseline_threshold` (0.2) and inputs in a normal range, most
neurons fire most of the time; E-R mainly adds refractoriness after strong
firings and spontaneous activity after long silence.

## Learning

```cpp
bool eligible() const;                                        // does this neuron learn now?
float learningDelta(float reward, float learningRate) const;  // its step size, when eligible
void learn(std::span<float> weights, std::span<const float> inputs,
           float reward, float learningRate) const;           // one neuron, caller-owned weights
```

Layers ask each neuron whether it is eligible and for its step size, then
update all weights of a group at once with the SIMD kernel. The rule (the
default `Sign` [learning rule](learning.md); the others are described
there):

```
weights[i] = clamp(weights[i] + learningRate × gain × reward × eligibility × sign(inputs[i]),
                   -max_weight, +max_weight)
```

`gain` is the neuron's learning gain (default `default_learning_gain`, 2.0; see
[per-neuron dynamics](#per-neuron-dynamics)).

- **Only the sign of each input** is used, not its magnitude. An input of
  exactly 0 never changes its weight. (Rationale: by the time a reward
  arrives, the current input magnitude may no longer reflect what caused the
  firing; its direction is more trustworthy.) Consequence: encode features as
  ±values rather than 0/value if absence should be learnable.
- **Positive reward** moves the output toward positive for this input
  pattern, **negative reward** toward negative. The reward is therefore a
  *desired direction*, not a right/wrong score.
- **Eligibility** decides whether and how strongly a neuron learns:
  - with E-R: eligible only while `threshold > baseline_threshold` (it fired
    recently), with `eligibility = threshold / baseline_threshold − 1`.
    Because the threshold after a firing is at least `2 × baseline`,
    eligibility is at least 1 and often much larger, so E-R neurons take big
    steps;
  - without E-R: eligible if `|output| > firing_epsilon`, with
    `eligibility = 1`.
- Each weight is clamped to `[-max_weight, max_weight]` (±10). There is no
  weight decay.

`learningDelta` is `learningRate × gain × reward × eligibility`; the kernel
adds `delta × sign(inputs[i])` to each weight and clamps. Layers apply it
against the inputs as they are when the reward arrives; call it right after
the step.

## Per-neuron dynamics

Three parameters can differ between neurons, each with its own optional
jitter:

| Parameter | Default | Meaning |
|-----------|---------|---------|
| recovery (`recovery()` / `setRecovery`) | `recovery_factor` (0.9) | per-tick E-R threshold decay while silent; larger means slower relaxation, i.e. a longer memory of past firing |
| learning gain (`learningGain()` / `setLearningGain`) | `default_learning_gain` (2.0) | multiplies this neuron's weight updates |
| alpha (`alpha()` / `setAlpha`) | `default_alpha` (1.2) | E-R threshold growth on firing with the `Log` rule (ignored by the others); larger means a longer refractory period and a longer memory trace, ≥ 0 |

Each is drawn from a `Jitter`, a description of a random distribution:

```cpp
struct Jitter {
    enum class Distribution : uint8_t { None, Uniform, Normal };
    Distribution distribution = None;
    float spread = 0;            // Uniform: half-width; Normal: standard deviation
    bool relative = false;       // spread is a fraction of the parameter's scale
    std::optional<float> mean;   // centre; unset = the parameter's default
    float min = -inf, max = +inf;
    bool enabled() const;        // distribution != None && spread > 0
};

Jitter::uniform(0.05f)                      // default ± 0.05, uniform
Jitter::normal(0.02f).around(0.95f)         // normal, mean 0.95, sd 0.02
Jitter::uniform(0.3f).within(0.5f, 2.0f)    // only values in [0.5, 2.0]
Jitter::uniformRelative()                   // ±50% of the parameter's scale
Jitter::normalRelative(0.2f)                // sd = 20% of the scale
Jitter::none()                              // off
```

**Relative spreads** are a fraction of the parameter's *scale*, so the same
jitter keeps its meaning when the default changes:

| Parameter | Scale | `uniformRelative()` (±50%) at the defaults |
|-----------|-------|---------------------------------------------|
| learning gain | the value itself | 2 → 1 … 3 |
| alpha | the value itself | 1.2 → 0.6 … 1.8 (alpha 2 → 1 … 3) |
| recovery | its distance from 1 (the relaxation speed) | 0.9 → 0.85 … 0.95 |

Recovery uses the distance from 1 because it must stay below 1: ±50% of the
value itself (0.45 … 1.35) would be mostly invalid and lopsided.

```cpp
void randomizeRecovery(const Jitter& jitter);
void randomizeLearningGain(const Jitter& jitter);
void randomizeAlpha(const Jitter& jitter);
void randomizeDynamics(const Jitter& recovery, const Jitter& learning, const Jitter& alpha = {});
```

- Values outside the jitter's limits, or outside the parameter's own valid
  range (recovery [0.01, 0.999], gain ≥ 0, alpha ≥ 0), are **redrawn**, not clamped, so
  no probability piles up at the edges.
- Draws come from a dedicated random stream (reset by `reseed`).
- A disabled jitter sets the default and draws nothing, so without jitter
  no random numbers are consumed. Learning-gain jitter is centred on
  `default_learning_gain` unless `around()` gives another centre.

Layers apply the jitter from their `LayerSpec` to every neuron they create,
and `network::setRecoveryJitter` / `setLearningJitter` change it later (see
[network](network.md#per-neuron-dynamics)).

Measured effects (gapped-pattern benchmark, 50 paired trials, all with the
original `Log` threshold growth):

- **learning jitter** sometimes helps a learned hidden layer with E-R, but
  **not robustly**: the effect appeared and vanished with every change of
  the E-R constants (e.g. +0.166, t 5.0 at gain 1.0 / baseline 0.1;
  +0.028, t 1.4 at gain 1.0 / baseline 0.2; +0.203, t 6.5 at gain 2.0 /
  baseline 0.2). `LearningJitterEffectDependsOnBaseGain` reports it without
  asserting it;
- **recovery jitter** gave a small, not significant improvement of the memory
  in unconnected E-R neurons (+0.036 ± 0.022 at ±0.05, gain 1.0);
- **alpha jitter** (±25%, ±50%, normal sd 25%) did not help in either test
  of `AlphaJitterTest`: Pavlovian sign inversion learned equally fast
  (5.7 epochs, all differences noise), and 3-number sequence detection with a
  15-tick delayed response (memory in unconnected E-R neurons) stayed at
  or slightly below the baseline (0.847; −0.008 to −0.082).

## Randomness and `reseed`

```cpp
void exr::reseed(std::uint32_t seed);   // core/random.hpp
```

Process-wide `mt19937` streams, one per purpose (`core/random.hpp`): a new
group's initial weights, weights added by growth, weights for attached
sensors, per-neuron jitter, learning (feedback-alignment matrices, the
perturbation noise seed), and the seeds of new neurons' spontaneous-firing
generators. Each draws U(−1, 1) weights row after row, so adding draws for
one purpose never shifts another. Each neuron then draws spontaneous values
from its **own** `minstd_rand`, so neurons can step in parallel without
sharing state and results do not depend on thread count.

Because the streams are shared, a network's initial weights depend on
everything constructed before it in the process. Call `reseed` before
building a network to make it reproducible on its own. `reseed` is not
thread-safe with respect to concurrent network construction.

## Serialization

```cpp
void serialize(std::ostream& os, std::span<const float> weights) const;
std::vector<float> deserialize(std::istream& is, DeserializeMode mode = DeserializeMode::FullState,
                               std::uint32_t format = NEURON_FORMAT_VERSION);   // returns the weights
```

A record carries the neuron's weights, which the layer passes in and gets
back, so the file layout is the same as when neurons owned their weights.
Binary, native endianness, in this order:

| Field | Type |
|-------|------|
| hasHabituation, hasER | `bool`, `bool` |
| alpha | `float` |
| weight count, weights | `size_t`, `float × count` |
| threshold, previous sum, habituation streak | `float`, `float`, `int` |
| current output | `float` |
| spontaneous generator state | `uint32` |
| recovery, learning gain | `float`, `float` |

`DeserializeMode`:

- `FullState`: restores everything, so the neuron continues exactly as the
  saved one would, including future spontaneous firings.
- `WeightsOnly`: restores flags, alpha, recovery, learning gain and weights; resets threshold to
  `baseline_threshold`, habituation state and output to 0, and keeps the
  neuron's current (fresh) random generator.

`deserialize` also takes the neuron data `format` (`NEURON_FORMAT_VERSION`,
currently 3). Format 1, written by network files of versions 1–2, has no
recovery or learning gain; they then keep their current values. Format 3
has the same neuron record as format 2; it marks that layers of neurons
append their [learning rule](learning.md#serialization) state.

The gate, rectification, habituation rule and threshold growth rule are
layer-level settings and are not in the record: the layer sets them on its
neurons, and a saved network restores them from each layer's `LayerSpec`
(see [network](network.md#file-format)).

Everything is read and checked before the neuron changes. A weight count
above `max_serialized_weights` (2^26) or an impossible generator state throws
`std::runtime_error` instead of allocating or corrupting state.

## Tunable constants

Constants in `neuron.hpp` (`inline constexpr`), shared by all neurons:

| Constant | Value | Meaning |
|----------|-------|---------|
| `habituation_epsilon` | 1e-10 | max change in the sum still counted as "the same signal" (the floor of the habituation tolerance) |
| `habituation_steps` | 100 | default `Habituation::steps`: identical steps before the input is suppressed |
| `recovery_factor` | 0.9 | default per-tick threshold decay while not firing (per-neuron value: recovery) |
| `min_threshold` | 1e-10 | threshold at or below which spontaneous firing starts |
| `spontaneous_min_amplitude` | 0.01 | amplitude of spontaneous firing |
| `firing_epsilon` | 1e-6 | output magnitude counted as "fired" (neurons without E-R) |
| `baseline_threshold` | 0.2 | resting E-R threshold and eligibility boundary |
| `max_weight` | 10 | learning clamps each weight to ±this |
| `max_output` | 10 | each weighted sum is clamped to ±this |
| `default_learning_gain` | 2.0 | default per-neuron learning gain (multiplies every weight update) |
| `default_alpha` | 1.2 | default E-R threshold growth rate of the `Log` rule (`neuron` constructor) |

## Notes and pitfalls

- `step` and `learn` use the shorter of `inputs` and `weights`.
- A neuron has **no bias term**. Use an input held at a constant value (e.g.
  a sensor at 1.0) when a threshold at a chosen value is needed.
- With E-R on, a neuron may output exactly 0 on a tick simply because it is
  refractory. Tests and benchmarks that read single ticks should account for
  that (e.g. average over several ticks).
