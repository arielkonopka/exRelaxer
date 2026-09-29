# Research log: parameters, mechanisms and topologies

A summary of the experiments run on exrelaxer so far: what was changed, under
which settings it was measured, and what it showed. Numbers come from the
test suite and from screening programs; where a result depends on the E-R
constants in force at the time, the constants are given.

**Threshold growth rule.** Every section here (§1–§16) was run with the
original logarithmic growth on firing, thr + alpha × ln(|v| / thr). Since
2026-09-27 the default is linear, thr + 0.5 × (|v| − thr) (§16,
`ThresholdGrowth`); results that depend on E-R may differ under it.

- [How results were measured](#how-results-were-measured)
- [Current state](#current-state)
- [1. Making E-R networks learn at all](#1-making-e-r-networks-learn-at-all)
- [2. Stability: bounding weights and outputs](#2-stability-bounding-weights-and-outputs)
- [3. Reward protocol](#3-reward-protocol)
- [4. Default constants: manual changes](#4-default-constants-manual-changes)
- [5. Systematic E-R parameter search](#5-systematic-e-r-parameter-search)
- [6. Topologies for gapped-pattern detection](#6-topologies-for-gapped-pattern-detection)
- [7. Memory without hand-built delay lines](#7-memory-without-hand-built-delay-lines)
- [8. Per-neuron jitter](#8-per-neuron-jitter)
- [9. E-R firing frequency](#9-e-r-firing-frequency)
- [10. Performance parameters](#10-performance-parameters)
- [11. Learning rules](#11-learning-rules)
- [12. Several senses and stereo vision](#12-several-senses-and-stereo-vision)
- [13. Activity economy and path selection](#13-activity-economy-and-path-selection)
- [14. Dynamic nonlinearity substitution: static tasks](#14-dynamic-nonlinearity-substitution-static-tasks)
- [15. How E-R behaves: learning, silence, state, habituation](#15-how-e-r-behaves-learning-silence-state-habituation)
- [16. Temporal tasks and the threshold growth rule](#16-temporal-tasks-and-the-threshold-growth-rule)
- [17. Training E-R, spontaneous cycles and early fading](#17-training-e-r-spontaneous-cycles-and-early-fading)
- [18. Rerun with linear growth and three habituation variants](#18-rerun-with-linear-growth-and-three-habituation-variants)
- [19. Normalised weighted sum](#19-normalised-weighted-sum)
- [20. Dynamic ladder: time-varying input and Doom](#20-dynamic-ladder-time-varying-input-and-doom)
- [21. Video temporal memory](#21-video-temporal-memory)
- [Conclusions](#conclusions)
- [Open questions and next steps](#open-questions-and-next-steps)

## How results were measured

**Many seeded trials, not single runs.** Each result is averaged over 20–200
independent networks (`neuron::reseed(seed)` per trial, now `exr::reseed`). Early single runs
turned out to depend on which tests ran before them in the same process, and
E-R dynamics make single runs noisy.

**Controls.** Every learning result is compared with a *no-learning control*:
the same seed, network and input stream with learning rate 0. This caught one
misleading check early: the original Pavlovian test's "response grew after
training" held in 38/50 seeds *without* learning (vs 39/50 with).

**Paired comparisons and noise verdicts.** Two settings are compared on the
same seeds; the per-seed differences give a mean ± standard error and
t = mean / SE. |t| < 2 is reported as noise (about 95% confidence).

**Benchmarks**

| Benchmark | Task | Main measure |
|-----------|------|--------------|
| Pavlovian sign inversion | a negative stimulus must produce a positive response and vice versa; stimulus ±5 × `baseline_threshold` | correct out of 50 seeds, or epochs to learn |
| Sequence order | respond positively to "1 then 2", negatively to "2 then 1" | correct out of 100 seeds |
| Gapped pattern | `A,{0..1},B,{0..1},C` in a random scalar stream with 7 kinds of decoy; respond on the tick after C | **accuracy after C** (valid pattern vs decoy, 0.5 = chance) |
| 3-number sequences | detect one trigger sequence among three, response delayed 15 ticks | balanced accuracy |

For the gapped pattern, *balanced accuracy over all ticks* turned out to be
misleading: "respond after any C" scores 0.958 on it while knowing nothing
about the sequence. Accuracy after C scores that rule at exactly 0.5; the best
shortcut rule ("C with B 1–2 ticks before") reaches 0.81, the bar a real
sequence detector has to beat.

## Current state

| Constant | Value | Notes |
|----------|-------|-------|
| `recovery_factor` | 0.9 | per-tick threshold decay while silent |
| `baseline_threshold` | 0.2 | resting threshold, eligibility boundary |
| `default_alpha` | 1.2 | threshold growth on firing (log rule only) |
| `default_learning_gain` | 2.0 | multiplies every weight update |
| `max_weight`, `max_output` | 10, 10 | clamps |
| threshold rule on firing | `max(2 × baseline, threshold + 0.5 × (\|v\| − threshold))` | linear, the default since 2026-09-27; §1–§16 used `threshold + alpha × ln(\|v\| / threshold)` |
| learning rule | `w += rate × gain × reward × eligibility × sign(input)` | eligibility = threshold / baseline − 1 with E-R |

## 1. Making E-R networks learn at all

**Problem.** With the original constants (`baseline_threshold` 1.0,
threshold growth `threshold *= 1 + alpha × ln(ratio)`), networks with E-R did
not learn the Pavlovian task at all: trained and control networks scored
identically (23/50) and layers A and B never changed a weight.

**Cause.** An E-R neuron may learn only while `threshold > baseline_threshold`.
Weak inputs (±0.5 through weights ≤ 1) only fire a neuron after its threshold
has relaxed below the input, and a firing then lifts it only slightly, never
back above 1.0. The neurons that respond are exactly the ones never allowed to
learn.

| Change | Pavlovian, E-R on | Sequence, E-R on all layers |
|--------|-------------------|-----------------------------|
| baseline 1.0 (original) | = control (23/50) | — |
| baseline 0.1 | 32/50 (rate 0.005), 45/50 (rate 0.05) | 43/100 |
| + additive growth `max(baseline, t + α ln r)` | 50/50 | **100/100** |
| + floor `max(2 × baseline, …)` | 50/50 | 100/100 |

With E-R off, the same Pavlovian network learned 50/50 throughout; E-R was
the only obstacle.

## 2. Stability: bounding weights and outputs

- **Weight clamp ±10** (`max_weight`): no effect on the tests at the time
  (no weight reached ±10), but bounds long training.
- **Output clamp ±10** (`max_output`): essential. Without it, every network
  with feedback loops diverged within ~30 ticks (outputs ~1e36), controls
  included; with E-R the divergence was hidden, because infinite thresholds
  silenced every neuron and the output sat at exactly 0. Cause: weights in
  [−1, 1] regardless of fan-in, so a loop with 56 inputs per neuron
  amplifies activity ~4× per pass.
- The clamp is also the only non-adaptive nonlinearity, which is what makes
  hand-wired band detectors possible (a high-gain neuron saturates at ±10).

## 3. Reward protocol

The reward sign acts as the desired *direction* of the output
(`updateWeights` moves every eligible neuron's output toward it), not as a
right/wrong score. Two ways to deliver it:

| Mode | Reward | Window + readout, after C |
|------|--------|---------------------------|
| target | the desired sign on every tick | **0.500** |
| error-driven | the desired sign only when the output's sign is wrong | **0.976** |

With the target mode the readout learned exactly the right weight *pattern*
but every weight ran into the ±10 clamp and the bias never learned (balanced
rewards cancel), so every tick after a C scored positive. Error-driven reward
(perceptron-style, a reward prediction error) learns the threshold. All later
sequence results use it.

## 4. Default constants: manual changes

Each change of a default shifted E-R results; E-R-off results barely moved.

**Learning gain 1.0 → 1.5** (recovery 0.9, baseline 0.1):

| Test | gain 1.0 | gain 1.5 |
|------|----------|----------|
| Sequence order, E-R off | 98/100 | 99/100 |
| Window + readout, E-R off (after C) | 0.976 | 0.964 |
| Window + readout, E-R on | 0.882 | 0.778 |
| Random reservoir 40+60 | 0.824 | 0.814 |
| Unconnected E-R neurons | 0.735 | 0.740 |

**Recovery 0.9 → 0.8** (gain 1.5, baseline 0.1):

| Test | recovery 0.9 | recovery 0.8 |
|------|--------------|--------------|
| Sequence, E-R on hidden+readout | 29/100 | 43/100 |
| Sequence, E-R on all layers | 100/100 | 87/100 |
| Window + readout, E-R on | 0.778 | 0.794 |
| Unconnected E-R neurons | 0.740 | 0.713 |

Faster relaxation changes how far thresholds recover in the 3 rest ticks
between sequence presentations.

**Test robustness.** Tests now express E-R-sensitive inputs relative to the
constants (a weak stimulus is 5 × `baseline_threshold`; waits for spontaneous
firing are computed from `recovery_factor`). With that, the suite passes for
`baseline_threshold` from 0.05 to 1.0. Before, a baseline of 0.5 collapsed
the Pavlovian E-R test to 10–14/50 because the ±0.5 stimulus became too weak
to ever fire a neuron.

## 5. Systematic E-R parameter search

About 250 settings of recovery, alpha, baseline and the post-firing floor,
screened on three roles of E-R (learning gain 1.5 at the time):

- **memory**: 100 unconnected E-R neurons → 1-neuron readout (gapped pattern, after C)
- **learning**: E-R in the learned readout (window + 8-neuron readout, after C)
- **whole network**: sequence order with E-R on all layers

**Effect of each parameter** (coarse grid, averaged over the others):

| Parameter | Memory | Learning | Sequence |
|-----------|--------|----------|----------|
| `baseline_threshold` 0.03 → 0.1 → 0.3 | ≈ same | **0.59 → 0.84 → 0.96** | mixed |
| alpha 0.3 → 2.4 | **0.63 → 0.76** | slightly down | best around 1.2 |
| `recovery_factor` 0.6 → 0.95 | best at 0.95 | ≈ same | **0.40 → 0.96** |
| floor 1.5 / 2 / 4 × baseline | none | none | none |

- **The resting threshold is the biggest lever for learning**: eligibility is
  `threshold / baseline − 1`, so a higher baseline shrinks E-R's oversized
  learning steps.
- **Alpha around 2 helps memory; alpha ≥ 3.5 is a cliff**: the sequence
  network collapses from 200/200 to 0–4/100.
- **A broad plateau**, not a sharp optimum: recovery 0.93–0.97, alpha
  1.8–2.4, baseline 0.3–1.0 all score about the same (average of the three
  roles ≈ 0.93–0.94, vs ≈ 0.79 for the constants then in use).

**The best baseline depends on signal strength.** With strong ±5 features a
baseline of 0.5–1.0 gives learning ≈ 0.98–0.99, but the Pavlovian test with
±0.5 stimuli breaks above ≈ 0.2 (neurons stay refractory through the short
stimuli). Full-suite check of candidates (40–200 trials):

| recovery / alpha / baseline | Memory | Learning | Sequence | Full suite |
|-----------------------------|--------|----------|----------|------------|
| 0.8 / 1.2 / 0.1 (then current) | 0.742 | 0.794 | 172/200 | passes |
| 0.95 / 2.0 / 0.5 | 0.813 | 0.968 | 200/200 | Pavlovian E-R 10/50 |
| 0.93 / 2.0 / 0.5 | 0.805 | 0.988 | 200/200 | Pavlovian E-R 14/50 |
| **0.9 / 2.0 / 0.2 (recommended)** | **0.815** | **0.904** | **200/200** | passes |
| 0.93 / 2.0 / 0.2 | 0.815 | 0.913 | 200/200 | passes |

Recovery 0.9 and baseline 0.2 were adopted; **alpha is still 1.2** (the
recommendation's 2.0 was applied to the learning gain instead). The
Pavlovian failures were later traced to its fixed ±0.5 stimulus (see
section 4, test robustness).

## 6. Topologies for gapped-pattern detection

Accuracy after C, error-driven reward unless noted; 0.5 = chance, shortcut
rules reach 0.81.

| Topology | After C | Notes |
|----------|---------|-------|
| fully learned feedback network, raw sensor | 0.50 | every variant tried, E-R on or off |
| frozen value detectors + learned feedback network | 0.50–0.51 | no usable memory |
| detectors + frozen delay window + learned readout, E-R off | **0.976** | gain 1.0; 0.964 at gain 1.5 |
| … same, 1-neuron readout, 40,000 training ticks | **1.000** | gain 1.0 |
| … + learned hidden layer, E-R off | 0.969 | no gain over the readout alone |
| … + frozen random mix layer, E-R off | 0.952 | |
| … readout with E-R | 0.882 | gain 1.0; 0.778 at gain 1.5 |
| … learned hidden layer with E-R | 0.50 | E-R in hidden layers breaks learning |

- **Value detection is the hard part.** Distractors fill [0, 10] with only a
  ±0.5 exclusion zone around each symbol; learning with sign-only updates and
  no bias cannot build sharp band detectors, so they are hand-wired and frozen.
- **One learning neuron is enough** once features and memory are provided: the
  task is nearly linearly separable in the windowed features, and the
  error-driven rule acts as a perceptron.
- **A widened exclusion zone** (±0.05 → ±0.5) was needed for the detectors;
  ±1.0 would merge the zones of A, C and B (1.6–1.7 apart).
- A bug in an intermediate benchmark version (the event draw shared the
  0–10 distractor distribution, making patterns 3.5% instead of 30% of events)
  produced a misleading 0.598; it was fixed before these results.

## 7. Memory without hand-built delay lines

Replacing the hand-built delay window (1-neuron readout, 40,000 training ticks):

| Memory | E-R off | E-R on |
|--------|---------|--------|
| none | 0.500 | — |
| unconnected neurons (no recurrence) | **0.500** | **0.692–0.740** |
| frozen random reservoir 40 + 60 | 0.824–0.872 | 0.779 |
| frozen random reservoir 100 + 200 | 0.938 | 0.887 |
| hand-built delay window | 0.999 | — |

- **E-R carries memory by itself**: neurons with no connections between them
  have no memory without E-R (exactly 0.500) but reach ~0.72 with it; the
  adaptive threshold records recent firing. This is the clearest positive
  result for the E-R mechanism. It needs clean input features: with random
  ramps instead of the hand-built detectors it disappeared (0.506).
- **Recurrence carries more**, and E-R on top of a reservoir slightly hurts.
- **E-R in the readout hurts** (0.735 → 0.527 in the unconnected setup).
- **Fully undesigned** (random frozen ramps + random reservoir): 0.728; a
  perfect window over the random ramps reaches only 0.776, so value
  detection, not memory, is what still needs design.

## 8. Per-neuron jitter

Recovery, learning gain and alpha can each be drawn per neuron from a chosen
distribution (absolute or relative spread).

**Learning-gain jitter ±0.1**, window + learned hidden layer + readout with
E-R, 50 paired trials:

| Constants | Base gain | Effect of jitter |
|-----------|-----------|------------------|
| recovery 0.9, baseline 0.1 | 1.0 | **+0.166 ± 0.033 (t 5.0)**, 0.538 → 0.705 |
| recovery 0.9, baseline 0.1 | 1.5 | −0.019 ± 0.026 (t −0.7) |
| recovery 0.8, baseline 0.1 | 1.0 | **+0.239 ± 0.033 (t 7.2)** |
| recovery 0.8, baseline 0.1 | 1.5 | −0.017 (t −0.6) |
| recovery 0.9, baseline 0.2 | 1.0 | +0.028 (t 1.4) |
| recovery 0.9, baseline 0.2 | 2.0 | **+0.203 (t 6.5)** |

Strong when present, but it appeared and vanished with every change of the
constants: **not a robust property**. The corresponding test only reports it.

**Recovery jitter** (100 unconnected E-R neurons, 50 paired trials, gain 1.0):
±0.05 → +0.036 ± 0.022 (t 1.7), ±0.08 → +0.029 (t 1.2): noise level. No
effect on a random reservoir.

**Alpha jitter** (current constants, 50 paired trials):

| Alpha jitter | Pavlovian: epochs to learn | 3-number sequences, balanced |
|--------------|----------------------------|------------------------------|
| none | 5.68 | 0.847 |
| uniform ±25% | 6.10 (t 1.4) | 0.765 (t −2.1) |
| uniform ±50% | 5.86 (t 0.5) | 0.839 (t −0.2) |
| normal sd 25% | 5.68 (t 0.0) | 0.806 (t −1.2) |

No setting helps; the single t = −2.1 is most likely noise (one of three
comparisons, the larger ±50% jitter shows nothing). The sequence task needed
a 15-tick delayed response: with an immediate response every setting scored
1.000, a ceiling that cannot show an effect.

## 9. E-R firing frequency

Jitter was meant to give neurons different operating frequencies. Measured
under constant input (1.0, weight 1), the period between firings once
settled:

| | recovery 0.80 | 0.85 | 0.90 | 0.95 |
|---|---|---|---|---|
| alpha 0.6 | 2 | 1 | 1 | 1 |
| alpha 1.2 | 2 | 2 | 2 | 2 |
| alpha 1.8 | 2 | 2 | 2 | 2 |

Input strength moves it only slightly (4 ticks at input 0.3, 1 tick at 3.0).

**Cause.** Threshold growth is `alpha × ln(output / threshold)`; under steady
input the threshold settles just below the input, so the ratio is ≈ 1 and the
growth ≈ 0. The rule regulates itself toward "fire as often as possible", so
alpha and recovery hardly change the rhythm, and jittering them cannot create
frequency diversity. This explains the null results of section 8 for
sustained inputs.

A growth rule whose jump does not vanish at steady state (e.g.
`threshold += alpha` or `threshold += alpha × |output|`, as in
spike-frequency adaptation models) would give a period of about
`ln((input + jump) / input) / ln(1 / recovery)` ticks, set directly by alpha
and recovery. Not implemented yet. (Since §16, `ThresholdGrowth` offers
such rules, e.g. `fixed` thr + a; their firing periods are not measured.)

## 10. Performance parameters

| Change | Effect |
|--------|--------|
| Release build by default (was no build type) | ~8× faster |
| inputs gathered into contiguous floats; hoisted constants; min/max instead of `std::clamp` | bit-identical results; gapped-pattern tests 13.5 s → 6.3 s; both hot loops vectorized at `-O3` |
| OpenMP threads per group = work / 16,384 (was: all threads above 32,768) | reservoir workload: serial 2.40 s wall / 2.39 s CPU; old 0.91 s / 6.32 s; new **1.22 s / 2.86 s** |
| `forward` on large layers, 20 threads | 1000 × 1000 group ≈ 70 µs vs 890 µs serial |

Intermediate thread counts (7–14) were slower than both 3 and 20 threads on
mid-sized groups.

## 11. Learning rules

Six learning rules, chosen per layer ([learning](learning.md)): sign (the
original), trace, feedback alignment, node perturbation, Oja and BCM.

**Credit assignment beyond the last layer.** A 2-neuron linear bottleneck
between 6 inputs and 4 outputs, where the target depends on two directions
of the input (`LearningRuleTest.FeedbackAlignmentTrainsAHiddenLayer`):
with the hidden layer frozen at its random start the mean squared error
stays at 0.49. With feedback alignment training it through `applyError`,
the error falls to 7 × 10⁻⁷ at rate 0.03, or 0.009 at rate 0.01
(5000 samples). It is the first mechanism here that trains a hidden layer
towards a task.

XOR through 16 hidden neurons does not learn with any rule: without E-R the
neurons are linear up to the ±10 clamp, and with E-R the threshold's own
dynamics make the evaluation noisy (loss 0.8–1.2 for frozen and trained
hidden layers alike). Nonlinear hidden features need a nonlinearity the
learning rules can use; E-R's threshold is a poor one.

**Snake** (`nntest run snake_rules`, 10 × 10, 200 training games, 50 test
games, 10 trials; control: the same network untrained, 0.06 apples in 64
steps). Mix = the 64-neuron mixing layer, 64 → 3 readouts. Mix learning
rate 0.0003.

| Readouts | Mix | Apples / game | Steps / game | Apples / 100 steps | Best | µs / training step |
|----------|-----|---------------|--------------|--------------------|------|--------------------|
| sign (lr 0.03) | frozen | **13.6 ± 0.6** | 103 ± 5 | 13.2 | 25.0 | 1.4 |
| sign | sign | 13.6 ± 0.4 | 102 ± 4 | 13.3 | 27.5 | 1.7 |
| sign | trace | 13.7 ± 0.7 | 103 ± 5 | 13.2 | 24.7 | 1.6 |
| sign | feedback alignment | 13.0 ± 0.4 | 98 ± 4 | 13.3 | 25.0 | 1.6 |
| sign | perturbation | 10.9 ± 1.4 | 86 ± 9 | 11.7 | 22.2 | 1.8 |
| sign | Oja | 10.8 ± 1.1 | 96 ± 11 | 12.5 | 23.2 | 1.8 |
| sign | BCM | 12.6 ± 0.5 | 95 ± 4 | 13.3 | 25.8 | 1.8 |
| trace (lr 0.03) | frozen | 12.9 ± 1.1 | 99 ± 8 | 13.0 | 25.1 | 1.7 |
| feedback alignment (lr 0.003, bias) | frozen | 11.0 ± 0.4 | 79 ± 3 | 13.9 | 23.6 | 1.6 |
| feedback alignment | feedback alignment | 9.9 ± 0.4 | 71 ± 3 | 13.9 | 21.9 | 1.7 |
| feedback alignment | BCM | 10.2 ± 0.5 | 73 ± 4 | 14.0 | 23.9 | 2.1 |
| feedback alignment | Oja | 6.4 ± 1.3 | 97 ± 16 | 8.9 | 16.8 | 1.6 |
| perturbation (lr 0.03, noise 1) | frozen | 0.1 | 26 ± 17 | 1.1 | 1.5 | 1.1 |

(± is the standard error over trials.)

- **Snake does not need hidden learning.** No mix rule beats the frozen
  random mix; the sign-rule readouts already use its features well, and
  games end when the snake traps itself, which one-step values cannot
  foresee.
- **Feedback-alignment readouts are the most efficient** (13.9 apples per
  100 steps against 13.2) but die sooner: they regress the reward's
  size, not its sign. Error-driven gating matters: with `reward=target`
  they fail (0.1–0.2 apples).
- **Trace readouts** match sign readouts at the same rate (12.9 vs 13.6,
  within noise) when the reward baseline is off. With a baseline, the
  error-driven gating biases the baseline and they fall to about 5.
- **Perturbation readouts do not learn snake**: each readout learns only
  when its action is chosen, and the noise's effect on one step's squared
  error is small next to the variance between states. In the mix, it
  costs apples at rates above 0.0003.
- **Oja** in the mix loses apples and makes results vary more: unit-norm
  principal components throw away the directions the readouts used.
  **BCM** is close to frozen.

Tuning (sweeps in the same experiment): sign readouts 0.03; trace readouts
best at 0.03 (4.0 at 0.1, 0.3 at 1.0); feedback-alignment readouts
0.001–0.003 (0.5 at 0.1); mix rules best at 0.0003, sign and trace mixes
flat up to 0.003, feedback alignment down to 9.9 at 0.003, perturbation to
0.5 at 0.003. A 16-neuron mix gives the same ranking (frozen 13.7, every
learned mix 12–13).

## 12. Several senses and stereo vision

Named input sources, a cochlea with several microphones, `Resize2D` and
`Disparity` ([multimodal](multimodal.md)) were tested with two experiments.
Result files: [`results/multimodal/`](../results/multimodal/).

**Stereo depth** (`nntest run stereo_depth`, 5 trials). Julesz random-dot
stereograms, 16 × 32: each eye sees only ±1 dots; an 8 × 8 square floats in
front (disparity 3) in the left or the right half. A single learned
readout (sign rule, error-driven) answers which half.

| Model | Accuracy |
|-------|----------|
| left eye → average pool → readout | 0.49 |
| both eyes → average pool → readout (no matching) | 0.50 |
| both eyes → Disparity (0..5, window 3) → average pool → readout | **0.98** (correlation 0.98, difference 0.98, normalized 0.98) |

With 20 % of the right eye's dots redrawn, correlation and normalized
matching keep 0.97, but absolute difference drops to 0.62: every
mismatched dot adds a full unit of difference at every disparity, which
buries the small margin at the true one. Pooling 4 × 4 learned faster than
8 × 8 (0.93 vs 0.89 after 400 stereograms); 1000 stereograms are the
default. The depth is invisible to one eye and to a linear readout of both,
so the Disparity layer is doing the work: matching is a product (or
difference) of the two views, which no weighted sum of them computes.

**Sight and sound** (`nntest run audiovisual`, 5 trials). Four objects,
each with its own look (a bar at its own orientation, at a random place on
a 16 × 16 image with uniform ±0.5 pixel noise) and sound (a tone at its own
pitch, 250 Hz to 3.5 kHz with ±15 % detuning, ±0.5 noise). Sight is a
frozen Gabor bank and max pooling; sound is a 16-band cochlea. One readout
per object, one vs rest, error-driven.

| Readouts on | Accuracy |
|-------------|----------|
| sight | 0.79 |
| sound | 0.95 |
| both | **0.96** |

With both senses noisier (±0.8 pixels, ±1.0 sound) the gain is clearer:
sight 0.71, sound 0.75, both **0.86**.

The sign rule could not use the sound at all (0.09, worse than chance): the
cochlea's bands are all positive and the sign rule sees only each input's
sign, so every weight moves together. The graded trace rule is the default
for this experiment.

**Learning what things sound like by watching them.** Readouts on sight
learn the objects from labels, in silence. Then, without labels, the
network sees and hears 1000 objects, and a second set of readouts, on
sound only, learns to agree with what sight chooses (sight's choice is its
target). Tested by sound alone, in the dark, it names the object **0.66**
of the time (0.52–0.76 over trials; chance 0.25, 0.29 without the watching
phase), although it was never told what any object sounds like. Its
teacher was right 0.79 of the time; the sound readouts do worse than their
teacher, not better, so they learn its mistakes too. The gap to supervised
sound readouts (0.95) is the cost of a noisy teacher.

An unsupervised shared layer did not work as well: a 16–32-neuron Oja or
BCM layer (winners 1–2) reading sight and sound, trained on paired objects,
then frozen, with readouts taught by sight and tested by sound, reached
0.2–0.4 (controls 0.2–0.3). The layer's units were driven mostly by sight's
128 inputs, and uncentred inputs make Oja's first component the mean
rather than the object.

## 13. Activity economy and path selection

Does E-R make a network use less activity, prefer cheaper routes, or
answer differently depending on its recent past, **with no activity
penalty anywhere**? Four experiments ([activity](activity.md)) compare
production E-R hidden neurons with linear neurons and with a fixed
threshold (`gate`) calibrated to be exactly as sparse as E-R. Three paths
of 12, 20 and 35 neurons connect 32 inputs to 4 linear readouts; 4 noisy
prototypes, each held for 4 ticks. 10 trials each. Result files:
[`results/er-activity/`](../results/er-activity/).

**Economy** (`nntest run er_economy`, Exp 1 and 6).

| Learning | Model | Accuracy | Active fraction | Spikes per decision | Accuracy per 100 spikes |
|----------|-------|----------|-----------------|---------------------|-------------------------|
| paths FA + readouts | er | 0.995 | 0.50 | 133 | 0.75 |
| | gate (matched) | 0.762 | 0.49 | 132 | 0.58 |
| | linear | 0.987 | 1.00 | 268 | 0.37 |
| readouts only | er | 0.941 | 0.49 | 133 | 0.71 |
| | gate (matched) | 0.953 | 0.50 | 133 | 0.72 |
| | linear | 0.980 | 1.00 | 268 | 0.37 |

- E-R uses half the activity of the linear network, but so does a fixed
  threshold of the same sparsity: the saving comes from **thresholding**,
  not from adaptation.
- Activity does **not** fall with training (E-R 0.484 → 0.496 active):
  the sparsity is static, present from the first sample, not learned
  economy.
- E-R's active runs are shorter (5.2 vs 8.6 ticks for the gate): it
  alternates neurons more.
- With the paths learning (FA), E-R was the only sparse model that
  learned well (0.995 vs 0.762); with frozen paths the matched gate is as
  good as E-R. Silent gated neurons take no FA step, just as silent E-R
  neurons don't (the surrogate derivative is 0 for both). So E-R's
  adaptive threshold helps FA credit assignment through a thresholded
  layer; it does not make inference cheaper than a
  fixed threshold.

**Path preference** (`nntest run er_paths`, Exp 2). Drive per neuron,
relative to a path's size, is ≈ 1.0 for all three paths and all three
models, early and late in training, with and without online learning. No
model prefers the small path.

**Switching over long runs** (Exp 7, 3000 samples). E-R changes the
leading path while the input stays the same (0.08 per tick); the fixed
threshold and linear models never do (0, or ≈ 0.001 when learning
online). Under one input held for 400 ticks, E-R stays ≈ 82 % active,
the leading path changes 0.04–0.22 times per tick and holds for
110–250 ticks on average, and the answer stays right (≈ 1.0). E-R does not
fall silent under a constant input: its threshold approaches the input
and it keeps firing.

**Fatigue and recovery** (`nntest run er_fatigue`, Exp 3 and 5). One path
stimulated (amplitude 10, 20 ticks), then 40 test samples:

| Recovery | Fatigued | Its share vs rest | Other paths' share | First-sample accuracy | Recovery ticks |
|----------|----------|-------------------|--------------------|-----------------------|----------------|
| 0.9 | A | 0.40 | 1.15 | 0.97 | 10 |
| | B | 0.35 | 1.23 | 0.97 | 10 |
| | C | 0.44 | 1.69 | 0.93 | 9 |
| | all | 0.20 | – | 0.80 | 12 |
| 0.97 | A | 0.13 | 1.23 | 1.00 | 25 |
| | B | 0.05 | 1.37 | 0.97 | 30 |
| | C | 0.21 | 1.86 | 0.97 | 24 |
| | all | 0.06 | – | 0.93 | 34 |

Fatiguing one path moves activity to the others and the answer survives;
fatiguing all of them costs accuracy until thresholds relax. Recovery time
follows the recovery factor (≈ 10 ticks at 0.9, ≈ 30 at 0.97); later
samples are back at 0.995. The fixed threshold and linear models are
unaffected (share 1.0). This is redundancy with state-dependent
recruitment, on the time scale of threshold recovery, not a lasting
reorganization.

**History** (`nntest run er_history`, Exp 4). The same 20 inputs after
each condition, from the same trained network; changes relative to rest
(recovery 0.9 / 0.97):

| Before | Pattern change | Evidence change | Decisions changed | Spikes vs rest | Latency (ticks) |
|--------|----------------|-----------------|-------------------|----------------|-----------------|
| busy | 0.28 / 0.27 | 0.28 / 0.28 | 0 / 0 | 0.77 / 0.54 | 1.06 / 1.00 |
| A | 0.09 / 0.10 | 0.25 / 0.23 | 0.01 / 0.005 | 0.86 / 0.83 | 1.02 / 1.00 |
| B | 0.16 / 0.17 | 0.35 / 0.31 | 0.03 / 0.03 | 0.76 / 0.71 | 1.07 / 1.01 |
| C | 0.28 / 0.29 | 0.50 / 0.48 | 0.06 / 0.02 | 0.56 / 0.52 | 1.20 / 1.02 |
| all | 0.53 / 0.56 | 0.64 / 0.80 | 0.11 / 0.20 | 0.17 / 0.06 | 1.55 / 1.27 |

For the fixed threshold and linear models every change is exactly 0. E-R's
response to an input depends on what it did just before: which neurons
fire, how many and how strongly; ordinary activity (`busy`) changes the
pattern and spike count but not the decisions, heavy stimulation of
everything changes one decision in 5–10.

**In short.** E-R networks are sparse without a penalty, but no sparser
than a fixed threshold; they do not prefer cheaper paths; they do route
activity around recently used neurons and respond differently depending
on recent history, which fixed nonlinearities cannot. The fixed-threshold
control (`LayerSpec::gate`, network format 11) was added for this.

## 14. Dynamic nonlinearity substitution: static tasks

Can E-R dynamics replace network size? Milestone 1 of the
[nonlinearity suite](nonlinearity.md): how large must a network be to
reach test MSE ≤ 1e-3 (in ≥ 80 % of seeds) on known static functions,
with ReLU, E-R, fixed-threshold (`gate` 0.2) or plain clamped hidden
neurons? The runs covered:

- 9 tasks: l0–l3, and l4 with K = 1, 2, 4, 8, 16;
- depth {1, 2, 3, 4, 6, 8} × width {4, …, 128};
- 4 learning rates, the best chosen per model and architecture on
  validation;
- 5 seeds, for 25 920 trials.

Every model uses the same data, initial weights, feedback-alignment
learning and stopping rule. Result files are in
[`results/nonlinearity/`](../results/nonlinearity/).

**Minimum architecture.**

- ReLU solves every task. Its smallest solving networks:
  - l0: 1 × 4;
  - l1: 2 × 8;
  - l2: 1 × 16;
  - l3: 1 × 64;
  - l4: 1 × 32 for every K except K = 2 (1 × 128).
- The plain clamped neuron solves only l0, the linear task.
- The fixed threshold solves l0 at 1 × 32.
- **E-R solves none, not even l0.**

So in this setting E-R does not reduce the topology needed. It does not
reach the predefined error at any size.

**Best test MSE on any architecture** (median over seeds; 1 tick after the
input arrives):

| Task | relu | er | gate | clamp |
|------|------|----|------|-------|
| l0 x1+x2 | 3.7e-4 | 0.045 | 7.7e-4 | 3e-15 |
| l1 x1·x2 | 6.6e-4 | 0.088 | 0.088 | 0.11 |
| l2 sin(x1·x2) | 6.7e-4 | 0.077 | 0.077 | 0.099 |
| l3 + exp(−x3²) | 8.5e-4 | 0.055 | 0.11 | 0.14 |
| l4 K=1 | 8.3e-4 | 0.30 | 0.24 | 0.33 |
| l4 K=2 | 8.9e-4 | 0.33 | 0.23 | 0.30 |
| l4 K=4 | 7.2e-4 | 0.22 | 0.14 | 0.20 |
| l4 K=8 | 9.1e-4 | 0.22 | 0.19 | 0.25 |
| l4 K=16 | 8.8e-4 | 0.17 | 0.10 | 0.13 |

The target variances are 0.11 (l1), about 0.5 (l4) and 0.67 (l0), so
E-R's l1 error (0.088) is about 80 % of the variance: it learns little of
the product. On l1 it matches the fixed threshold.

**More ticks per sample** (E-R only; the other models' outputs do not
change with extra ticks, which was checked). Holding each sample longer
lowers E-R's error by 2–4×:

| Task | 1 extra tick | 7 | 15 |
|------|--------------|---|----|
| l0 | 0.045 | 0.012 | 0.0064 |
| l1 | 0.088 | 0.029 | 0.033 |
| l3 | 0.055 | 0.047 | 0.068 |
| l4 K=4 | 0.22 | 0.094 | 0.074 |
| l4 K=16 | 0.17 | 0.072 | 0.060 |

With 7 ticks E-R represents x1·x2 better than any static threshold
network (0.029 vs 0.088) and better than every model but ReLU on l3 and
l4. That is nonlinear capacity bought with **more time**, not more
neurons. It still never reaches 1e-3; at a relaxed 1e-2 only l0 is
solved (1 × 4, 15 extra ticks).

**Activity** at 2 × 32, l1:

| Model | Active fraction | Spikes per sample | Event synaptic operations | Test MSE |
|-------|-----------------|-------------------|--------------------------|----------|
| ReLU | 0.40 | 102 | 2200 | 6.5e-4 |
| E-R | 0.16 | 41 | 1000 | 0.093 |
| gate | 0.77 | 198 | 3400 | 0.091 |

With 7 or 15 extra ticks, E-R's spikes per sample rise to 121 and 260.
E-R is the sparsest per tick, but at the error it reaches it spends
comparable or more spikes per answer once it needs more ticks. Wall time
per inference is similar (3–4 µs at this size) and grows with the ticks.

**Divergence.** With linear-looking units (clamp, gate, E-R) at rates
≥ 0.01, and at depth 6–8, feedback alignment often diverges to the ±10
clamp. The per-model rate choice avoids most of this; ReLU tolerates the
largest rates (0.03).

**Reading.** For static functions, trained this way (feedback alignment,
one error per sample on the last tick), E-R does not substitute for
topology. It supplies some nonlinearity that grows with the time it is
given, beyond a fixed threshold's. Following the spec, the next step is
the temporal tasks, with the memoryless E-R control. Two open causes
remain to test:

- the readout-on-the-last-tick protocol, since E-R's output fluctuates
  from tick to tick;
- the learning signal, since an E-R neuron only learns on the ticks it
  fires.

## 15. How E-R behaves: learning, silence, state, habituation

Four questions that followed §13 and §14. There is no activity term in any
of these experiments. Result files are in [`results/`](../results/):
`nonlinearity/curves`, `nonlinearity/state`, `er-silence` and
`er-habituation`.

**Learning curves** (`nl_static`, `early_stop=false`, 100 000 samples,
5 seeds, the learning rate chosen on validation). Validation MSE after
2k / 10k / 20k / 50k / 100k samples, 1 × 32:

| Task | Model | 2k | 10k | 20k | 50k | 100k |
|------|-------|----|-----|-----|-----|------|
| x1·x2 | relu | 5.2e-3 | 1.3e-3 | 1.7e-3 | 2.4e-4 | 1.9e-4 |
| | er, 1 extra tick | 0.119 | 0.109 | 0.107 | 0.082 | 0.072 |
| | er, 7 extra ticks | 0.111 | 0.051 | 0.044 | 0.034 | 0.034 |
| | gate | 0.116 | 0.108 | 0.112 | 0.106 | 0.122 |
| sin(x1·x2) | relu | 3.5e-3 | 8.9e-4 | 4.5e-4 | 1.6e-4 | 5.6e-5 |
| | er, 7 extra ticks | 0.063 | 0.033 | 0.033 | 0.030 | 0.028 |

The curves have different shapes, not just different levels:

- ReLU keeps improving through 100k samples.
- E-R with extra ticks drops quickly, then plateaus from about 20k.
- The fixed threshold barely learns.

At two hidden layers, E-R is unstable. On l4 (K = 4) and on some l1/l2
runs it diverges to the ±10 clamp after 5–50k samples, even at lr 0.001.
ReLU does not.

**Same stimulus, different state** (`nl_static`, `state_probes=200`).
Each stimulus is shown after 20 different random histories of 10 samples,
10 seeds, stable networks only. For each stimulus, the error splits into
the squared bias of its mean output and the variance across histories,
which is the part that depends on state. For the static models that
variance is exactly 0.

- E-R, 1 extra tick: the state accounts for 6–14 % of the error. Outputs
  for the same stimulus span 0.3–0.5.
- E-R, 7 extra ticks: the state accounts for **30–57 %** of the error. The
  span is 0.5–0.7, and spikes for the same stimulus vary with an SD of
  12–42.

More ticks remove bias but add dependence on the past. With enough time,
about half of what E-R gets "wrong" on a static task is its history, not
its mapping.

**Zeroed inputs** (`er_silence`, 3000 ticks of zero input after training,
10 seeds):

- **Without recurrence**, every model is silent from the first tick. After
  about 214 ticks (recovery 0.9) E-R starts firing spontaneously, and it is
  the only model that does. The firing is a slow, self-paced rhythm, and
  every neuron takes part:
  - recovery 0.9: 268 firings per 1000 ticks, bursts every 73–110 ticks;
  - recovery 0.97: 67 per 1000 ticks, bursts every 250–340 ticks.
- The firings are tiny (±0.01) and never grow into ordinary activity. With
  recurrence at gain 1, 3 or 10, E-R activity dies within 3–10 ticks.
  Linear recurrent networks, by contrast, ring for 600–1300 ticks before
  decaying. **E-R does not sustain its own activity**, but it keeps
  "idling" at a low, rhythmic rate.
- The silence changes the next answers:
  - During the silence, thresholds decay to about 1e-5. The first input
    then fires every neuron, and the logarithmic rule raises their
    thresholds by about 1.2 × ln(|s| / 1e-5) ≈ 14.
  - At recovery 0.97 that leaves the network refractory for about 90
    ticks. Accuracy on the first 5 samples drops from 0.92–1.0 to
    **0.44–0.52**.
  - At recovery 0.9 it recovers within a sample or two (0.98).

  This comes straight from the logarithmic threshold growth, because the
  ratio to a near-zero threshold is huge.

**Habituation** (`er_habituation`, stimuli held for 4–500 ticks, 10
seeds). New options: `steps`, `tolerance` and `decay` (§ neuron).
Results at a 500-tick hold for E-R, reported as spikes per stimulus, then
recognition at onset / summed / at the end:

| Rule | Exact input | With ±0.001 flicker |
|------|-------------|---------------------|
| off | 27 077; 0.98 / 1.00 / 1.00 | 25 789; 0.99 / 1.00 / 0.99 |
| cut after 100 (original) | 4 533; 1.00 / 1.00 / **0.00** | 25 678: never triggers |
| cut after 5 | **296**; 0.98 / 0.99 / 0.00 | 25 652: never triggers |
| cut after 5, 1 % tolerance | 295; 0.99 / 0.99 / 0.00 | **607**; 0.98 / 0.96 / 0.15 |
| fade 0.99 after 5 | 8 585; 0.98 / 1.00 / 0.28 | 25 715: never triggers |

- Habituation is an effective activity saver, with no penalty:
  - E-R uses up to **90× fewer** spikes on a held stimulus;
  - onset and summed recognition stay at about 0.98.
- The cost is that the stimulus is no longer represented at the end of the
  hold.
- Fading instead of cutting keeps a weak trace. For linear neurons it keeps
  end accuracy at 0.99 while saving 2–3×, because a scaled pattern keeps
  its winner. E-R's thresholds drop a faded input below firing, so it gains
  little (0.28 at the end).
- The original exact rule never triggers on a flickering sensor. Only a
  tolerance restores the saving (607 spikes).
- The clamp comes before habituation, so neurons saturated at ±10 always
  see the same sum and habituate across samples. That silenced part of the
  linear network and cost it accuracy (0.82 with flicker; 0.28–0.6 with a
  1 % tolerance). E-R's thresholds keep it out of saturation, and it did
  not suffer.

## 16. Temporal tasks and the threshold growth rule

**Question.** Milestone 1 found no topology advantage on static functions.
The spec's next step: when the target depends on earlier inputs, does E-R's
state substitute for memory the network does not otherwise have, and is
that effect due to the state or to E-R's transfer function? A second
question comes from the user: should the threshold rule be logarithmic?

**Setup.** `nntest run nl_temporal` ([nonlinearity](nonlinearity.md#temporal-tasks-nl_temporal)):
delayed XOR `t1` (x(t) XOR x(t−1)), `t2` x(t) AND NOT x(t−3), parity of the
last n `t3` (n = 4, 8, 16), `t4` sin(x(t)·x(t−2)). Each step is held for
depth + 2 ticks, so a feed-forward network without neuron state sees only
x(t). Models: `relu`, `er`, `er_memoryless` (the same E-R neurons reset to
rest before every step: a test-only wrapper), `gate`, `clamp`. Depth
{1, 2, 3, 4, 6, 8} × width {4 … 128}, 5 seeds, lr {0.001, 0.003, 0.01,
0.03} chosen per architecture on validation, 20 000 training steps.
Success, fixed in advance: accuracy ≥ 0.95 (MSE ≤ 1e-3 on `t4`) in ≥ 80 %
of seeds. `input_ceiling_accuracy` is the best any function of x(t) alone
can do. 21 600 trials; raw results in
[`results/nonlinearity/temporal/`](../results/nonlinearity/temporal/).

**Results** (median test accuracy at each model's best architecture):

| Task | Ceiling without memory | relu | gate | clamp | er_memoryless | er |
|------|------|------|------|------|------|------|
| t1 delayed XOR | 0.51 | 0.51 | 0.51 | 0.51 | 0.51 | **0.952** (1×64; solved) |
| t2 x(t) ∧ ¬x(t−3) | 0.748 | 0.748 | 0.748 | 0.743 | 0.743 | 0.844 (1×32) |
| t3 parity 4 | 0.511 | 0.49 | 0.511 | 0.511 | 0.511 | 0.621 (1×64) |
| t3 parity 8 / 16 | 0.52 / 0.51 | chance | chance | chance | chance | chance |
| t4 sin(x(t)·x(t−2)), NMSE | 1 | 0.99 | 1.0 | 1.0 | 0.99 | 1.0 |

- Only E-R with state goes above the memoryless ceiling, on t1, t2 and
  parity 4. The memoryless E-R control stays at the ceiling everywhere, so
  the effect comes from the **state**, not from E-R's transfer function
  (spec §16: "E-R with state > memoryless E-R ≈ static").
- It solves only delayed XOR, and only with one hidden layer (64 or 128
  neurons; 1×32 gives 0.94). Deeper E-R networks lose it: depth 4 and more
  stay at chance. E-R's memory lasts about one step: x(t−3) is only
  partly available, parity beyond 4 not at all, and the continuous t4 not
  at all.
- Deeper or wider E-R networks on t2 and t4 often diverge (NMSE in the
  hundreds), like the 2-layer runs in §15.
- Activity: at its best t1 architecture E-R has 48 % of hidden neurons
  active per tick, 91 spikes per step.

**Threshold growth rules.** `ThresholdGrowth` (network format 14; log stayed
the default for these runs; linear is the default since) offers `linear` thr + a(s − thr), `fixed` thr + a and
`multiplicative` thr·(1 + a), a = 0.5. Swept on t1, t2, t3 n=4, t4 and the
static l1, l2 with depth {1, 2} × width {8 … 64}, the same lr grid and 5
seeds, with learning from the last tick or from every tick at lr / ticks
(`learn_ticks=all`). Raw results in [`results/nonlinearity/growth/`](../results/nonlinearity/growth/).

| Rule | t1 best (solved architectures) | t2 | t3 n=4 | l1 MSE (settle 7) | l2 MSE |
|------|------|------|------|------|------|
| log | 0.952 (1: 1×64) | 0.844 | 0.621 | 0.037 | 0.032 |
| linear | **0.9995 (5, from 1×8)** | 0.755 | 0.586 | 0.038 | **0.026** |
| fixed | 0.70 (0) | 0.784 | 0.617 | 0.082 | 0.068 |
| multiplicative | 0.69 (0) | 0.756 | 0.567 | 0.107 | 0.096 |

- The linear rule solves delayed XOR with 8 neurons, eight times fewer
  than the log rule needs; on the static tasks it is as good as log or
  slightly better. The rules that ignore the firing's magnitude (fixed,
  multiplicative) lose both the memory and the static accuracy: the
  information E-R carries is how strongly a neuron fired.
- Learning from every tick did not help under any rule.
- After a long silence (`er_silence`, recovery 0.97), the log rule's
  overshoot drops accuracy to 0.5; with linear, fixed or multiplicative
  growth it stays at 1.0. These rules instead fire 2–20× more on the first
  sample after the silence (their thresholds come back low).

**Takeaway.** E-R's state is a real, short memory that no stateless
control has: it solves delayed XOR, which no static model can. But it
reaches only about one step back, fails for deeper networks, and does not
help with continuous-valued history. A linear growth rule makes that memory
much cheaper (8 neurons) and removes the post-silence blindness, without
costing static accuracy.

## 17. Training E-R, spontaneous cycles and early fading

Three questions from the user after the switch to linear growth: can a
network be trained without E-R and switched to E-R afterwards, or should
E-R simply be trained longer? Which settings give spontaneous activation
cycles? And what does habituation that fades from the second repeat do?
All runs use linear threshold growth. Results in
[`results/er-training/`](../results/er-training/),
[`results/er-cycles/`](../results/er-cycles/) and
[`results/er-habituation/`](../results/er-habituation/) (`er_habituation_fade`).

**Pretraining without E-R** (`nl_static` l1, l2 with settle 7, depth
{1, 2} × width {16, 32}; `nl_temporal` t1, t2 with 1 × {16, 64}; lr grid
chosen on validation, 5 seeds; `pretrain_model` relu, gate or clamp for
20 000 samples, then E-R for 0, 20 000 or 80 000 more):

| Task (arch) | E-R 20k | E-R 80k | ReLU 20k → E-R, no more training | ReLU 20k → E-R 20k | clamp 20k → E-R 20k |
|------|------|------|------|------|------|
| l1 x1·x2 (1×32), MSE | 0.031 | 0.032 | 0.115 (ReLU itself: 0.005) | 0.028 | 0.032 |
| l2 sin(x1·x2) (1×32), MSE | 0.035 | 0.057 | 0.104 (ReLU: 0.003) | 0.025 | 0.027 |
| l1 (2×32), MSE | 0.061 | 0.069 | 0.22 | 0.057 | 0.062 |
| t1 delayed XOR (1×16), accuracy | 0.999 | 0.999 | 0.50 | 1.0 | 0.999 |
| t2 x(t) ∧ ¬x(t−3) (1×64), accuracy | 0.744 | **0.837** | 0.743 | 0.778 | 0.842 |

- A network trained without E-R does not keep its skill when E-R is
  switched on: ReLU's 0.005 becomes 0.1–0.3, worse than E-R trained on its
  own. The static neuron's solution relies on a transfer that E-R does not
  have (one-sided rectification, no threshold state).
- Pretraining followed by an E-R phase ends where E-R alone does, slightly
  better with one layer (l2 0.025 vs 0.035). It is a warm start, not a
  shortcut.
- Longer E-R training helps only where E-R has to learn to use its
  memory: t2 goes from 0.74 to 0.84 with 80 000 samples. On static tasks
  it does not help (E-R's static error plateaus at about 20 000 samples,
  §15).
- With linear growth, delayed XOR is solved with 16 neurons in 20 000
  samples (§16 needed 64 with the log rule).

**Spontaneous cycles** (`er_silence`, E-R, 3000 silent ticks after
training, 10 trials; `spontaneous_below` × `spontaneous_amplitude` ×
`spontaneous_rate` × recurrent):

| Setting | Active fraction, ticks > 1000 | Readout mass per tick | Accuracy before / after |
|------|------|------|------|
| default (1e-10, 0.01, 0) | 0.5 % | 1e-4 | 1.0 / 1.0 |
| level 0.001 | 1.75 % | 3e-4 | 1.0 / 1.0 |
| level 0.05 | 5 % | 9e-4 | 1.0 / 1.0 |
| rate 0.05 | 5 % | 1e-3 | 1.0 / 1.0 |
| amplitude 1, rate 0.05 | 5 % | 0.09 | 1.0 / 1.0 |
| amplitude 1, rate 0.05, recurrent | 12 % | 0.05 | 1.0 / 1.0 |

- The level sets a regular per-neuron cycle: after a spontaneous firing the
  threshold sits at the floor 0.4 and decays by `recovery` per tick, so a
  silent neuron fires every `ln(0.4 / below) / ln(1 / recovery)` ticks:
  about 200 at the default level, 57 at 0.001 and 20 at 0.05 (recovery
  0.9). The measured active fractions (0.5 %, 1.75 %, 5 %) match 1/period.
- The rate adds irregular (random) firing on top.
- The amplitude decides whether spontaneous firing is felt downstream: at
  0.01 it is below every threshold it reaches; at 0.5–1 the readouts carry
  it (100–1000× the readout mass), and with recurrent paths it recruits
  other neurons (12 % instead of 5 %).
- None of these settings changed accuracy before or after the silence. The
  cycles are per neuron; no synchronized, network-wide rhythm appeared.

**Fading from the second repeat** (`er_habituation`, a stimulus held for
500 ticks, 10 trials; `habituation_decay` × `habituation_fade_after`):

| Rule | E-R spikes per sample | accuracy (whole sample / last tick) |
|------|------|------|
| no habituation | ≈ 27 000 | 1.0 / 1.0 |
| fade 0.9 after 100 repeats | 6 144 | 1.0 / 0 |
| fade 0.9 after 5 | 446 | 1.0 / 0 |
| fade 0.9 after 2 | 321 | 1.0 / 0 |
| fade 0.99 after 2 | 10 596 | 1.0 / 0.32 |

- Fading from the second repeat saves most: 80× fewer spikes than none,
  20× fewer than fading from the 100th, with the stimulus still recognised
  over the sample. Only a slow fade (0.99) keeps it represented at the end.
- With ±0.001 sensor flicker the input never repeats exactly and no rule
  acts; with 1 % tolerance, fading after 2 gives 717 spikes but accuracy
  0.97 over the sample and 0.13 at its end.

## 18. Rerun with linear growth and three habituation variants

Every E-R experiment again, with the current defaults (linear threshold
growth) and three habituation settings on every E-R-capable layer:
**off**; **cut5**, the input is cut after 5 identical ticks (decay 0);
**fade2**, the input fades by 0.9 per tick from its 2nd identical tick.
The library's default habituation (cut after 100) is unchanged; cut5 is
an experiment setting. The "log, off" column is the earlier run (log
growth, no habituation) where one exists, with the same parameters.
`nl_static` and `nl_temporal` were reduced to depth {1, 2, 3} × width
{4 … 64}, lr {0.001 … 0.03}, 5 seeds, lr chosen per network on validation.
Full tables: [`results/rerun/summary.md`](../results/rerun/summary.md);
raw files and `run.sh` in [`results/rerun/`](../results/rerun/).

**Activity-economy experiments** (medians over trials, E-R unless noted):

| Metric | log, off | off | cut5 | fade2 |
|------|------|------|------|------|
| `er_economy` accuracy | 0.997 | 1.0 | 1.0 | 1.0 |
| `er_economy` active fraction | 0.50 | 0.47 | 0.47 | **0.33** |
| `er_economy` accuracy per 100 spikes | 0.75 | 0.80 | 0.80 | **1.13** |
| `er_economy` gate accuracy | 0.785 | 0.823 | 0.827 | **0.957** |
| `er_paths` gate accuracy | 0.785 | 0.831 | 0.869 | **0.962** |
| `er_fatigue` share of first answers on the fatigued path | 0.39 | 0.14 | 0.14 | 0 |
| `er_fatigue` accuracy, all paths fatigued | 0.667 | 1.0 | 1.0 | 1.0 |
| `er_history` pattern change after a different history | 0.53 | **0.86** | 0.86 | 0.66 |
| `er_history` accuracy after that history | 0.90 | 0.85 | 0.85 | 0.80 |
| `er_silence` largest threshold after silence | 0.46 | 0.017 | 0.017 | 0.014 |
| `er_habituation` spikes per 500-tick sample | 26 523 | 30 107 | 446 | 321 |
| `er_habituation` accuracy at the sample's end | 1.0 | 1.0 | 0 | 0 |

- Linear growth alone (log → off) keeps every result and improves some:
  E-R's history effect grows (0.53 → 0.86 of hidden patterns change after
  a different history), and a long silence no longer leaves high
  thresholds (0.46 → 0.017).
- cut5 changes nothing in these experiments except `er_habituation`,
  because their inputs change at least every 5 ticks.
- fade2 is the only variant that saves activity during ordinary
  inference: a third fewer active neurons and spikes in `er_economy` at the
  same accuracy. It also weakens the history effect (0.86 → 0.66).
- fade2 lifts the fixed-threshold gate from 0.82 to 0.96, close to E-R. So
  E-R's lead over the gate in §13 shrinks to 0.04 once inputs fade. This
  fits the caveat that part of E-R's advantage under feedback alignment
  may come from its longer eligibility after firing; the mechanism was not
  isolated here.
- On held stimuli both habituation variants keep the answer over the
  sample but lose it at its end (as in §15, §17); the linear readout model
  under fade2 keeps it (0.99), because its many small inputs rarely repeat
  exactly.

**Static tasks** (`nl_static`, best median test MSE; ReLU needs 4–32
neurons for MSE ≈ 7e-4 on every task in every variant):

| Task | E-R log, off | E-R off | E-R cut5 | E-R fade2 | gate off | gate fade2 |
|------|------|------|------|------|------|------|
| l0 linear | 0.045 | 0.046 | 0.046 | 0.255 | 0.0008 | 0.0007 |
| l1 x1·x2 | 0.088 | **0.034** | 0.037 | 0.045 | 0.091 | 0.045 |
| l2 sin(x1·x2) | 0.077 | **0.030** | 0.030 | 0.040 | 0.078 | 0.031 |
| l3 | 0.074 | 0.060 | 0.060 | 0.074 | 0.119 | 0.068 |
| l4 k=1 | 0.30 | 0.15 | 0.15 | 0.31 | 0.25 | **0.084** |
| l4 k=16 | 0.19 | 0.10 | 0.10 | 0.35 | 0.10 | **0.039** |

- Linear growth halves E-R's static error on most tasks (l1 0.088 →
  0.034, l4 k=1 0.30 → 0.15). E-R still never reaches 1e-3.
- fade2 hurts E-R on static tasks (l0 0.046 → 0.26, l4 0.15 → 0.31): a
  held sample fades while E-R is still settling. It helps the gate
  (l4 0.25 → 0.084), which then beats E-R on l4. The cause was not
  isolated.
- cut5 has no effect: samples are held for fewer than 5 ticks.

**Temporal tasks** (`nl_temporal`, best median test accuracy; in brackets
the smallest network that solves delayed XOR in ≥ 80 % of seeds):

| Task | model | log, off | off | cut5 | fade2 |
|------|------|------|------|------|------|
| t1 delayed XOR | E-R | 0.95 (64) | **1.0 (8)** | 0.97 (4) | 0.98 (24) |
| t1 | ReLU | 0.51 | 0.51 | **1.0 (4)** | **1.0 (4)** |
| t1 | gate / clamp | 0.51 | 0.51 | 1.0 (8) | 1.0 (16) |
| t1 | memoryless E-R | 0.50 | 0.51 | 0.51 | 0.51 |
| t2 x(t) ∧ ¬x(t−3) | E-R | 0.84 | 0.76 | 0.78 | 0.75 |
| t2 | ReLU | 0.75 | 0.75 | 0.75 | 0.80 |
| t3 parity 4 | E-R | 0.62 | 0.59 | 0.55 | 0.56 |
| t3 parity 4 | best other | 0.50 | 0.50 | 0.57 (gate) | 0.56 (ReLU) |
| t3 parity 8, t4 | all | chance / ceiling | same | same | same |

- **Habituation is itself a one-step memory.** With cut5 or fade2, ReLU,
  gate and clamp networks solve delayed XOR, which none of them could
  before (ReLU with 4 neurons). Each bit is held for 3 ticks, so a neuron
  whose input repeats from the previous step crosses the 5-tick cut, or
  starts fading, during the new step. Its output then encodes "same as
  before", which is exactly the XOR of the two bits. Resetting the state
  (memoryless E-R, which also resets habituation) removes it again.
- Conclusion 13 therefore needs a correction: E-R is not the only state
  that beats the no-memory ceiling. Any per-neuron history does, and
  habituation, a change detector, fits this task better than E-R.
- Neither habituation variant reaches further back: t2 stays at 0.75–0.80
  and parity 8 and t4 stay at chance for every model. E-R's best t2 result
  (0.84) came from the log rule and was not reproduced here (0.76). §17
  reached 0.84 with linear growth only after 80 000 samples.

**Other experiments** (`bar_orientation`, `chirp_direction`, `snake`,
`snake_rules`, `stereo_depth`, `gapped_pattern`, `audiovisual`, with their
own defaults) pass all their built-in checks under linear growth. Where
earlier files exist the numbers match them: stereo 0.98 (was 0.975),
audiovisual with both senses 0.965 (same), snake 13.4 apples (sweep
median 11.8).

**Caveats carried over from the audit:**
- `er_fatigue`'s "share on the fatigued path" is 1 minus the other path's
  share, and the paths never compete, so its drop is not evidence of
  rerouting.
- The default sign rule reads a neuron's inputs after the whole step. With
  feedback edges it learns from values the neuron never saw.
  Feed-forward experiments, which include everything here, are unaffected.
- Feedback-alignment training accuracy uses the label during the sample,
  so only test numbers are compared above.

## 19. Normalised weighted sum

The user asked for a renormalisation of weights and outputs and chose the
normalised weighted sum: each hidden neuron's sum is divided by the length
of its weight vector, `Σ x·w / |w|` (`LayerSpec::normalize`, off by
default; the readouts keep their raw sum). Only the weights' direction
matters, and the sum is at most the length of the inputs. The same grids as
§18 (habituation off) were run with it. Results in
[`results/normalize/`](../results/normalize/),
tables in [`summary.md`](../results/normalize/summary.md).

| Measure | raw sum | normalised |
|------|------|------|
| `er_economy` E-R accuracy | 1.0 | 0.87 |
| `er_paths` E-R accuracy | 0.995 | 0.85 |
| `er_fatigue` E-R accuracy, all paths fatigued | 1.0 | 0.33 |
| `er_history` E-R accuracy after a different history | 0.85 | 0.30 |
| `er_economy` gate / linear accuracy | 0.82 / 1.0 | 0.82 / 1.0 |
| `nl_static` E-R best MSE, l1 / l2 / l4 k=1 | 0.034 / 0.030 / 0.149 | 0.052 / 0.046 / 0.143 |
| `nl_static` ReLU best MSE, l1 | 6.6e-4 | 6.1e-4 |
| `nl_temporal` E-R t1 / t2 accuracy | 1.0 (8 neurons) / 0.755 | 1.0 (8) / 0.797 |
| `nl_static` runs diverging (test MSE > 1), E-R at lr 0.01 / 0.03 | 82 % / 99 % | 33 % / 57 % |
| the same, gate at lr 0.01 / 0.03 | 55 % / 91 % | 27 % / 56 % |

- **It stabilises learning.** At the larger learning rates, half as
  many E-R, gate and clamp runs diverge. The rest probably diverge through
  the readout, whose sum is not normalised (inferred, not tested).
- **It does not improve the best results.** With the learning rate chosen
  on validation, E-R's static error rises by about half on l1–l3 and is
  unchanged on l4. ReLU is unchanged or slightly better (it is scale-invariant), the gate
  slightly worse (l1 0.091 → 0.101) and clamp unchanged. On temporal tasks E-R is unchanged, slightly better on t2
  (0.80 vs 0.76).
- **It hurts E-R in the activity experiments** at their fixed learning
  rate (accuracy 1.0 → 0.85–0.87, and 0.33 once all paths are fatigued),
  while the gate and the linear network are unaffected. The recalibration
  below shows that this is a learning-rate effect, not a threshold one.

**Recalibrated thresholds.** The user asked to recalibrate the thresholds
for normalised sums. E-R's resting threshold (0.2, also the eligibility
boundary and half the floor after firing) is now a per-layer setting
(`LayerSpec::restingThreshold`, network format 17). The experiments'
`resting_threshold=auto` scales it by the layer's mean 1/|w| at
initialization: 0.055 in the activity networks (40 inputs of uniform ±1
weights, |w| ≈ 3.6), and 0.23 in `nl_static` / `nl_temporal`, whose
initialization already gives |w| ≈ 1 (so their grids were not rerun).
The activity experiments were run at three learning rates, raw and
normalised, with the resting threshold 0.2 or auto
([`calibration.md`](../results/normalize/calibration.md), 5–10 trials):

| E-R, medians | raw, lr 0.0003 / 0.001 / 0.003 | normalised, rest 0.2 | normalised, rest auto |
|------|------|------|------|
| `er_economy` accuracy | **1.0** / 0.06 / 0.24 | 0.87 / 0.99 / **1.0** | 0.86 / 0.99 / **1.0** |
| `er_paths` accuracy | **0.995** / 0.07 / 0.30 | 0.85 / 0.99 / **0.998** | 0.85 / 0.99 / **0.999** |
| `er_fatigue` accuracy, all paths fatigued | **1.0** / 0 / 0 | 0.33 / 0.67 / **1.0** | 0.33 / 0.5 / **1.0** |
| `er_history` accuracy after a different history | **0.85** / 0.10 / 0.40 | 0.30 / 0.40 / **0.85** | 0.30 / 0.35 / **0.90** |
| `er_history` pattern change | 0.86 / 0.57 / 0.58 | 0.77 / 0.79 / 0.78 | 0.77 / 0.79 / 0.78 |
| `er_economy` gate accuracy | 0.82 / 0.38 / 0.32 | 0.82 / 0.96 / **0.99** | 0.83 / 0.96 / **0.99** |

- **Recalibrating the resting threshold changes nothing.** E-R's
  thresholds follow the size of the sums they fire on (their mean is
  about 1.0 during training, far above either resting value), so where
  they rest hardly matters. The §19 explanation above (absolute thresholds
  too high for smaller sums) was wrong.
- **The learning rate was the cause.** A normalised sum changes by a
  factor 1/|w| less per weight update, so it needs a larger learning rate.
  At 10× the rate, normalised E-R matches raw E-R at its best on every
  activity experiment, while raw E-R collapses (0.06). The usable range
  moves up and gets wider.
- **Normalised, the fixed-threshold gate catches up**: 0.99 vs 0.82 raw
  on `er_economy`, 1.0 once all paths are fatigued. E-R keeps its
  history effect (0.78 of hidden patterns change after a different
  history, 0.86 raw), which the gate does not have.
## 20. Dynamic ladder: time-varying input and Doom

**Question** (the user's): is E-R better at dynamic tasks, and can it play
Doom? §16 compared E-R only with networks that have no memory at all. The
control that matters is the same stateless network shown the previous
frame too (a frame window, the usual trick in game-playing networks).

**The missing control on §16's tasks** (`nl_temporal --set window=1`, 3
seeds): ReLU given x(t−1) solves delayed XOR with 8 neurons in every seed
within 1000 steps, where E-R without a window needs 32 neurons and is
unreliable at 8. With `window=3` ReLU solves t2 (x(t) ∧ ¬x(t−3)) too. E-R
with `window=1` reaches 0.97 on t2: its state adds about one step to the
window.

**Setup.** `nntest run dyn_ladder` ([dynamic](dynamic.md)): motion
direction on a 16-pixel retina (`dir`), change detection between 4 random
patterns (`change`), velocity of a bump (`vel`, regression) and a closed-loop
catch game with bouncing balls (`catch`, imitation of an oracle). Models
relu, er, er_memoryless, gate; `window` 0 or 1; width 16, 64; lr {0.001,
0.003, 0.01, 0.03} chosen on validation; 5 seeds; linear growth. Raw
results for this section are in [`results/dynamic/`](../results/dynamic/).

**Results** (median test score at width 64: accuracy, R², catch rate):

| Task | Single-frame ceiling | relu | gate | er_memoryless | er | relu + 1 frame | er + 1 frame |
|------|------|------|------|------|------|------|------|
| dir | 0.56 | 0.52 | 0.52 | 0.52 | 0.62 (0.69 at 256) | 1.00 | 1.00 |
| change | 0.70 | 0.70 | 0.70 | 0.70 | **0.98** | 1.00 | 0.92 |
| vel (R²) | 0 | 0.00 | −0.01 | −0.01 | 0.01 (0.04 at 256) | 0.93 | 0.84 |
| catch | chase 0.04 | 0.32 | 0.30 | 0.32 | 0.32 | 1.00 | 0.60 |

- Without a window, only E-R goes above the single-frame ceiling, and
  resetting its state removes the gain: it is the state again.
- E-R's state is a **novelty detector**: thresholds adapt to the current
  pattern, so a change stands out (0.98). It carries little of the order
  of events (direction 0.62–0.69) and nothing graded (velocity), so it
  cannot tell where the ball is going.
- One past frame lets every stateless model solve all four tasks. Adding
  E-R to a frame window **hurts** (catch 0.60 vs 1.00 for relu, 0.88 for
  memoryless E-R): the state interferes with what the window provides.
- Activity: no saving. On `change` E-R fires 147 spikes per step, relu with
  a window 84; on `catch` E-R 32–41, relu with a window 87.
- The log growth rule does not help (width 256, log vs linear: `dir`
  0.71 vs 0.70, `vel` R² 0.02 vs 0.08).

**Doom, imitation** (`doom`, ViZDoom `predict_position`: lead a walking
monster with one slow rocket; 40 × 30 pixels, 64 hidden, 600 training
episodes, 3 seeds). The oracle (object positions, never shown) wins 0.58;
aiming at the monster's current position wins 0.10. Without a window no
model learns from pixels: win rate 0.07 (er), 0.00 (relu, gate), oracle
agreement ≤ 0.5.

**Doom, from reward** (`doom_rl`: the screen at ViZDoom's minimum plus
stereo sound through a two-ear cochlea, a frozen 256-neuron mix and 8
learned action readouts; reward from hurt, death, kills, ammo, armor,
items, keys, doors, level exit and idling; 200 training episodes, 3 seeds).
- First attempt (the reward as is, trace rule): every readout was driven
  to the −10 clamp by the steady penalties and the agent stood still.
  Error-driven learning (as on snake) and a running reward baseline fixed
  that.
- `defend_the_center`: relu and gate **without sound** learned to keep
  running (11 000–12 000 map units per episode vs ≈ 800 untrained),
  took 34–39 health points of damage instead of 100 and survived to the
  timeout (reward −0.3 vs −6.1). No model learned to kill (≤ 0.4 per
  episode; one exploratory run reached 5.2). E-R did not learn to survive
  (damage 100). With sound, no model learned: with 3 seeds this may be
  seed variance rather than an effect of hearing.
- `map01` (a whole level, one-minute episodes): nothing beyond the
  untrained network: no kills, doors, items or exits.

**Doom, the E-R + habituation agent** (the user's goal since 2026-09-27:
an E-R network with habituation that plays Doom, its topology found by a
long search).
- `search.py` (successive halving over topology, E-R, habituation, ticks,
  pooling and learning settings, 1558 evaluations on `defend_the_center`)
  found winners that were seed luck: `retest.py` replayed the top five on
  three fresh seeds with 30 test episodes each, and trained networks
  scored worse than the same networks untrained (reward −4.97 vs −4.21).
  The one-step readout rules do not assign credit in Doom.
- `es.py` skips credit assignment: an evolution strategy (OpenAI-ES,
  12 antithetic pairs, centred ranks, Adam) moves the readout weights
  towards the candidates that scored best on the same three episodes;
  the current weights are scored on ten fixed validation episodes the
  gradient never sees (two and six on `map01`). Five topologies, 500 generations each on
  `defend_the_center` (validation reward, mean of the last 100
  generations; spikes per step):

| Network | First 20 gens | Last 100 gens | Best | Kills | Spikes |
|---------|---------------|---------------|------|-------|--------|
| **E-R + habituation, 1 layer, no feedback** | −4.79 | **+0.95** | **4.74** | 5.1 | **145** |
| E-R + habituation, 2 layers, recurrent | −2.64 | −1.04 | 0.80 | 5.1 | 363 |
| E-R, no habituation, 2 layers, recurrent | −2.49 | −0.83 | 0.48 | 5.3 | 365 |
| E-R + habituation, 3 layers, feedback both ways, 256 reservoir | −4.10 | −2.25 | 0.29 | 3.8 | 796 |
| ReLU, 2 layers, recurrent | −6.11 | −6.11 | −6.11 | 0 | 763 |

- The simplest network is the best and the most frugal. Habituation makes
  no measurable difference in the 2-layer network; depth, feedback and a
  reservoir make things worse. The ReLU network never moves: every
  candidate plays the same game, so the ranks carry no signal. The
  untrained E-R network already kills about one monster per episode from
  its own activity, so part of every E-R score comes before evolution.
- `map01` (1-layer E-R + habituation, one-minute episodes, 400
  generations, from scratch and from the arena agent's weights; 12 fresh
  episodes):

| | Untrained | Evolved from scratch | Evolved from the arena agent |
|---|---|---|---|
| Reward | −2.68 | −1.54 | −1.57 |
| Distance walked | 1287 | 2694 | 2582 |
| Items | 0.75 | 1.83 | 2.08 |
| Doors | 0.42 | 0.75 | 0.75 |
| Kills | 0.08 | 0.17 | 0.25 |
| Exits | 0 | 0 | 0 |

- Evolution learns to explore (twice the distance, more items and doors)
  but not to fight or finish the level; the arena weights give no lasting
  head start. Three-minute episodes (600 more generations) do not
  change that: still no exits, and a flat validation curve. Every Doom
  experiment is recorded in [doom](doom.md).

**Takeaway.** E-R is not better at dynamic tasks in general. Its state is
a cheap change detector that stateless networks lack, but one frame of
history gives a plain ReLU network everything E-R gives and more (order,
speed, planning in catch), and combining the two is worse than the window
alone. Doom from reward alone, with a frozen random mix and one-step
reward on the chosen action, learns survival on a small arena at best; a
whole level needs credit over many steps and learned features, which this
setup does not have. Evolving the readouts instead gives the goal's agent:
one E-R layer with habituation, the fewest spikes of every topology tried.


## 21. Video temporal memory

**Question** (the user's): can a feed-forward E-R layer keep temporal
information from a video stream in its own state, with no recurrence and
no frame history, and how much explicit history is that worth? Details,
tables and the figure: [video_memory](video_memory.md); raw results in
[`results/video-memory/`](../results/video-memory/).

**Setup.** `video_memory` (Python): a square moves left or right on a
20 × 15 screen, the screen goes blank for 1, 2, 4, 8 or 16 frames (input
0, the network keeps ticking), then the object reappears at the centre and
stands still for 2 frames, which are read out. In the primary `side` task
the object vanishes on the side it came from. An audit on 20,000 clips
puts a classifier given the final frame at chance (0.50). Networks: input
→ 128 frozen random hidden neurons → 2 readouts, with 4 ticks per frame
and library-default E-R (normalised sum, no habituation, linear growth,
recovery 0.9, resting threshold 0.2). Readouts are fitted by ridge
regression on the mean hidden output of the 8 readout ticks. Models C0
(ReLU), C1 (E-R reset every frame, the ablation), E0 (E-R), R1/R4 (ReLU +
1 / 3 previous frames), E1 (E-R + 1 frame), Rg (ReLU with a window that
reaches past the blank), D2/D2R (two E-R layers, feed-forward and
recurrent). 20 seeds, with the same clips for every model.

**Results** (side, test accuracy, gaps 1/2/4/8/16):

| | 1 | 2 | 4 | 8 | 16 |
|---|---|---|---|---|---|
| C0, C1, R1 | 0.49–0.50 | = | = | = | = |
| E0 | **0.544** | **0.511** | 0.495 | 0.496 | 0.495 |
| E1 | 0.595 | 0.555 | 0.532 | 0.508 | 0.502 |
| R4 | 0.991 | 0.974 | 0.495 | 0.495 | 0.495 |
| Rg | 0.970 | 0.974 | 0.974 | 0.975 | 0.973 |
| D2 / D2R | 0.558 / 0.560 | 0.538 / 0.541 | 0.50 | 0.50 | 0.50 |

- **The state keeps the answer; the readout cannot get at it.** A probe
  on E0's thresholds reads the answer with 0.79 accuracy right before the
  reappearance, at every gap from 1 to 16 (C1: 0.50). A silent threshold
  decays by the same factor every tick with no floor, so the pattern is
  only rescaled. At the reappearance the thresholds are far below the sums
  (median 0.13 at gap 1, 0.0002 at gap 16), nearly every driven neuron
  fires (40 → 128 of 128 on the first tick), and firing resets the trace.
- **Memory horizon** (the largest gap with the 99% lower bound over seeds
  above 0.5): E0 2 frames, E1 4, R4 2, Rg 16, and 0 for C0, C1 and R1.
- **Recovery sets the readable horizon** (separate sensitivity study): at
  recovery 0.95, 0.97 and 0.99, E0's horizon is 4, 8 and 16 frames, with
  0.67, 0.73 and 0.72 at gap 1, 0.70 at gap 16 for 0.99, and fewer spikes
  (33 → 8 per frame).
- **Recurrence adds nothing**: D2R − D2 is within ±0.007 at every gap,
  and the recurrent layer is silent during the blank.
- **The order task** (every single frame alike for both classes) is at
  chance for every model; Rg reaches only 0.51–0.52. A 128-neuron random
  layer does not represent direction from two frames (the audit needs
  2048 random features to reach 0.99), so this task measures the
  representation, not memory.
- **Feedback alignment on the hidden layer** (separate, 10 seeds) does not
  help. The online readouts stay at 0.50 and activity grows 4–15×, because
  no error reaches the frames before the blank.

The Sign rule timing issue (appendix 1) does not apply: the hidden layers
are frozen, or learn by feedback alignment.

## 22. Learned video memory

**Question** (the user's): can exRelaxer learn, online, to use the memory
trace that its E-R state keeps (§21), and does an eligibility trace let a
prediction error that arrives after the blank reach the activity from
before it? Memory, readout and learning are measured separately. Details:
[video_memory_learned](video_memory_learned.md); raw results and every
table in [`results/video-memory-learned/`](../results/video-memory-learned/summary.md).

**Setup.** §21's side task, clips, hidden networks (E0, and the non-E-R
controls C0 and C1) and settings, unchanged (library defaults: recovery
0.9, resting threshold 0.2, alpha 1.2, linear growth, normalised sums, no
habituation); §21's ridge readout is reproduced exactly in every trial.
Classes are exactly balanced in every split, gap and seed (checked before
training). Two library readouts learn online by the delta rule
(`apply_error`, one error per clip on the mean readout output, applied
after the last readout tick) on each hidden output scaled by a fixed
per-synapse gain, with no trace (immediate) or an input eligibility trace
d = 0.5–0.99 per tick, emptied at each clip (new `Network.reset_traces`).
20 seeds, gaps 1–16; recovery 0.95–0.99 as a separate study; hidden
learning with its own trace (`mode=hidden`) as condition D.

**Results** (test accuracy, gaps 1/4/16, mean over 20 seeds; chance 0.500):

| | recovery 0.9 | recovery 0.99 |
|---|---|---|
| probe: state before the reappearance | 0.79 / 0.79 / 0.79 | 0.92 / 0.92 / 0.92 |
| probe: state at the prediction | 0.57 / 0.52 / 0.50 | 0.89 / 0.87 / 0.73 |
| ridge readout (offline) | 0.544 / 0.495 / 0.495 | 0.716 / 0.721 / 0.701 |
| online, immediate | 0.496 / 0.500 / 0.501 | 0.523 / 0.505 / 0.505 |
| online, trace 0.8 | 0.500 / 0.503 / 0.495 | **0.652 / 0.692 / 0.646** |
| online, trace 0.99 | 0.504 / 0.504 / 0.499 | 0.502 / 0.504 / 0.510 |
| hidden layer learns, hidden trace 0.99 (10 seeds) | ridge 0.71 / 0.68 / 0.70, online ≤ 0.61 | **online 0.77 / 0.71 / 0.75**, ridge 0.82 / 0.79 / 0.78 |

- **At the default recovery nothing learns online.** Every trace is at
  0.49–0.51; the signal ridge finds (0.54 at gap 1) is too weak. C0 and C1
  are at chance for every readout and probe.
- **At recovery 0.99 the readout learns**, 0.65–0.69 at every gap with
  trace 0.8 (ridge 0.70–0.72), LEFT and RIGHT within 0.03.
- **Accuracy against the trace is a peak, not a rise.** The best trace
  matches the readout interval (0.8), and moves up to 0.9–0.95 only when
  the blank is long enough to empty the trace. Trace 0.99 is at chance at
  every gap, also with learning rates small enough to stay stable.
- **Why: the trace credits the right neurons for the wrong synapses.**
  With trace 0.99 the weight change correlates +0.5 to +0.7 with the
  neurons' direction selectivity before the blank and +0.05 with their
  output at the prediction; with trace 0.8 it correlates +0.76 with the
  latter and +0.83 with ridge's weights. The readout reads only current
  outputs, and the neurons that saw the object are not the ones that carry
  it at the prediction.
- **The trace is itself a memory**: a probe on it reads 0.96–0.99, even for
  C0, which has no state; the readout never sees it in its forward pass.
- **Without the per-synapse gain the delta rule learns nothing** (0.50;
  E-R outputs are sparse and small, condition number ≈ 25,000). With an
  error at every readout tick: 0.60–0.64 at recovery 0.99.
- **Condition D works.** When the hidden layer learns with a trace of
  0.99, the delayed error reshapes what the E-R thresholds store (probe
  0.99 at every gap, 0.90 at the prediction at gap 16), and the online
  readout reaches 0.71–0.77 at every gap, the best learned result. Without
  a hidden trace, hidden learning hurts (0.52–0.60).

The Sign rule timing issue (appendix 1) does not apply: the readouts and
the trained hidden layers use the delta rule / feedback alignment.

## Conclusions

1. **E-R was the main obstacle to learning**, through its eligibility rule.
   Lowering the resting threshold, additive threshold growth and a floor
   after firing fixed it; the resting threshold still sets how large E-R
   learning steps are.
2. **Error-driven reward is essential** for anything beyond one-step tasks.
3. **Learning works in one layer.** Every learned hidden layer, with or
   without feedback, added nothing or failed; with a global reward and no
   credit assignment, the reliable design is fixed layers for features and
   memory plus one learned readout.
4. **E-R is useful as memory, not in learning layers.** Unconnected E-R
   neurons hold enough of the recent past to decode sequences (≈ 0.72 vs 0.50
   without E-R); E-R in learned layers consistently hurts.
5. **Frozen random recurrence is the best undesigned memory** (0.94 with
   100 + 200 neurons); value detection remains the part that needs design.
6. **Jitter results are fragile.** Learning-gain jitter effects came and went
   with the constants; recovery and alpha jitter showed no reliable effect.
7. **E-R, as implemented, cannot produce slow rhythms**; frequency diversity
   needs a different threshold-growth rule (selectable since §16; not yet
   measured for frequency).
8. **Feedback alignment trains hidden layers** from an error vector (a
   bottleneck task: 0.49 → 7 × 10⁻⁷). On snake no hidden rule beats a frozen
   random mix, and sign-rule readouts remain the best.
9. **Stereo and cross-modal learning work with fixed matching and learned
   readouts.** A Disparity layer finds depth that neither eye shows
   (0.98 vs 0.50); two senses beat either; and sight can teach readouts on
   sound without labels (0.66, chance 0.25). Graded inputs such as cochlea
   bands need a graded rule.
10. **E-R's sparsity is thresholding; its state is history.** With no
    activity penalty E-R halves activity, exactly as a fixed threshold of
    the same sparsity does, and activity does not fall with training. No
    path is preferred. What only E-R does: activity moves away from
    recently fatigued paths (recovering in ≈ 10–30 ticks), the leading
    path changes under a constant input, and the same input gets a
    different response after a different history.

11. **E-R does not replace topology on static functions.** No network
    size reached test MSE 1e-3 with E-R, where ReLU needs 5–129 neurons.
    More ticks per sample buy E-R some nonlinearity (x1·x2: 0.029, vs
    0.088 for a fixed threshold), paid in time and spikes.

12. **E-R idles but does not sustain itself; habituation is the activity
    saver.** Without input E-R keeps a slow spontaneous rhythm, but
    recurrence does not turn it into self-sustained activity. After a long
    silence the logarithmic threshold rule overshoots and blinds the
    network for a while. Habituation cuts spikes up to 90× on held stimuli,
    at the cost of the stimulus's representation; it needs a tolerance
    to work on noisy sensors. On static tasks, a third to a half of E-R's
    remaining error is state-dependent.
13. **E-R's state is a one-step memory; a linear threshold rule makes it
    cheap.** On temporal tasks only E-R with state beats the ceiling of a
    network without memory (delayed XOR solved; x(t−3) and parity 4
    partly); resetting its state removes the gain, so it is the state, not
    the transfer function. It does not reach further back, fails in deep
    networks, and does not help continuous history. With linear threshold
    growth, delayed XOR needs 8 neurons instead of 64, and a long silence
    no longer blinds the network.
14. **Habituation is a memory too, and early fading is the activity saver
    during normal inference** (§18). Habituation that cuts or fades
    repeated input gives even ReLU networks the one-step memory that
    delayed XOR needs, so correction to 13: E-R is one of several kinds
    of neuron state that beat the no-memory ceiling. Fading from the 2nd
    repeat cuts E-R's active neurons by a third at unchanged accuracy on
    feed-forward tasks, and brings a fixed threshold within 0.04 of E-R. It
    hurts E-R on static regression and weakens E-R's history effect.
15. **A normalised weighted sum stabilises learning; it needs a larger
    learning rate, not new thresholds** (§19). It halves divergence at
    large learning rates, and at 10× the rate normalised E-R matches its
    best raw results. E-R's thresholds follow the sums, so recalibrating
    their resting value changes nothing. Normalised, a fixed threshold is
    as accurate as E-R on the activity tasks; E-R's history effect
    remains its own.
16. **E-R is a change detector, not a better dynamic network.** Given one
    past frame, stateless ReLU solves motion direction, velocity and a
    catch game where E-R alone cannot, and E-R on top of a frame window
    hurts. Only on change detection does E-R's state beat the single-frame
    ceiling by itself (0.98 vs 0.70). Doom from pixels is not learned by
    imitation; from reward, one-step readout rules learn nothing beyond the
    untrained network. Evolving the readouts works: one E-R layer with
    habituation and no feedback is the best and most frugal Doom agent on
    the arena, and on a whole level it learns to explore but not to finish.

17. **E-R keeps what it saw for long; it can read it back only briefly.**
    After a stimulus disappears, a feed-forward E-R layer's thresholds
    keep it through 16 blank frames (a probe reads 0.79), but at the
    default recovery a readout can use it for about 2 frames (0.54 at one
    blank frame), less than one explicit frame of history (0.97). The
    readable horizon follows the recovery time constant (16 frames at
    0.99, 0.70 accuracy, 4–8 spikes per frame), and recurrence adds
    nothing (§21).

18. **An eligibility trace helps one layer earlier than expected.** A
    readout can learn online to use E-R memory when the state still holds
    it at the prediction (recovery 0.99: 0.65–0.69, ridge 0.70–0.72), but
    an eligibility trace on the readout does not extend that: it credits
    the neurons active before the blank, whose current output does not
    carry the answer. The same trace on the hidden layer lets the delayed
    error reshape what E-R stores, and the online readout then reaches
    0.71–0.77 at every gap. The limit is reading the state at the
    prediction, not keeping it (§22).

## Open questions and next steps

- **Learned readable memory** (§22): hidden-layer learning with a long
  eligibility trace is the one mechanism that made E-R memory usable
  online. Next: the same on the order task and the dynamic ladder, the
  threshold state as a readout input, and a local input-gain rule
  (running variance) in place of the fixed per-synapse gain.
- **Readable E-R memory** (§21): recovery 0.97–0.99 makes E-R's state
  readable for 8–16 blank frames at a quarter of the activity. Try a slow
  recovery on the dynamic ladder and Doom, and a readout of the thresholds
  themselves (the probe reads 0.79–0.92 where the outputs read 0.5–0.7).

- **Alpha 2.0** (the search's recommendation) has not been applied; alpha is
  still 1.2. Alpha now matters only under the log growth rule.
- **Per-layer `baseline_threshold` and alpha**: the best values depend on
  signal strength (strong features want ≈ 0.5, weak inputs ≤ 0.2); only
  recovery, learning gain and alpha are per neuron today, and baseline is
  global.
- **Linear threshold growth is the default** since 2026-09-27 (§16, the
  user's decision). §1–§16 were run with the log rule; §18 repeated the
  E-R experiments with it. Next: sweep its amount and recovery.
- **Why fading helps the fixed threshold** (§18: 0.82 → 0.96 on
  `er_economy`, 3× lower error on l4): isolate it, for example by
  measuring learning eligibility per tick with and without fading.
- **Normalised sums at larger learning rates**: repeat the `nl_static` /
  `nl_temporal` grids with lr up to 0.1 (§19 showed the usable range moves
  up), and normalise the readouts too, to remove the remaining divergence.
- **Habituation versus E-R as memory**: compare them on change-detection
  and lag tasks at equal activity, and combine them on t2 and parity.
- **Deserialization of older files** (next version): read everything a file
  contains and default only what is missing; version-3 jitter widths become
  uniform jitter settings.
- **Undesigned value detection**: more or steeper random ramps, or a random
  layer on top, to push the fully undesigned network past the shortcut rules.
