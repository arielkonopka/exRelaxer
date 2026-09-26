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
