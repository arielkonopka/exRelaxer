# Research log: parameters, mechanisms and topologies

A summary of the experiments run on exrelaxer so far: what was changed, under
which settings it was measured, and what it showed. Numbers come from the
test suite and from screening programs; where a result depends on the E-R
constants in force at the time, the constants are given.

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
- [Conclusions](#conclusions)
- [Open questions and next steps](#open-questions-and-next-steps)

## How results were measured

**Many seeded trials, not single runs.** Each result is averaged over 20–200
independent networks (`neuron::reseed(seed)` per trial). Early single runs
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
| `default_alpha` | 1.2 | threshold growth on firing |
| `default_learning_gain` | 2.0 | multiplies every weight update |
| `max_weight`, `max_output` | 10, 10 | clamps |
| threshold rule on firing | `max(2 × baseline, threshold + alpha × ln(|v| / threshold))` | |
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
and recovery. Not implemented yet.

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
Result files: `/mnt/project-files/reports/multimodal/` in the project.

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
`/mnt/project-files/reports/er-activity/` in the project.

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
`/mnt/project-files/reports/nonlinearity/` in the project.

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
   needs a different threshold-growth rule.
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

## Open questions and next steps

- **Alpha 2.0** (the search's recommendation) has not been applied; alpha is
  still 1.2.
- **Per-layer `baseline_threshold` and alpha**: the best values depend on
  signal strength (strong features want ≈ 0.5, weak inputs ≤ 0.2); only
  recovery, learning gain and alpha are per neuron today, and baseline is
  global.
- **A threshold-growth rule with non-vanishing jumps** as an option, to test
  whether frequency diversity helps (e.g. telling apart pulse trains with
  different periods).
- **Deserialization of older files** (next version): read everything a file
  contains and default only what is missing; version-3 jitter widths become
  uniform jitter settings.
- **Undesigned value detection**: more or steeper random ramps, or a random
  layer on top, to push the fully undesigned network past the shortcut rules.
