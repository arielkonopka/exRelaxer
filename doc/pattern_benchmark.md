# pattern_benchmark (test support)

`tests/pattern_benchmark.hpp`, namespace `pattern_benchmark`

Not part of the library: a header shared by the tests (`tests/network.cpp`)
and by quick experiment programs. It defines the **gapped pattern detection**
benchmark, a statistics/reporting layer, and two frozen building blocks
(value detectors and a delay window).

## The task

A single scalar sensor receives a random stream. The network must output a
positive value for `PATTERN_HOLD` ticks after the sequence

```
A, {0..n values}, B, {0..n values}, C
```

and a negative value everywhere else. With the current constants:

| Constant | Value | Meaning |
|----------|-------|---------|
| `PATTERN_A`, `PATTERN_B`, `PATTERN_C` | 2.2, 5.5, 3.8 | symbol values |
| `DISTRACTOR_MAX` | 10 | distractors are uniform in [0, 10] ... |
| `SYMBOL_MARGIN` | 0.5 | ... but never within 0.5 of a symbol |
| `PATTERN_MAX_GAP` | 1 | n: 0..1 ignored values between A–B and B–C |
| `PATTERN_HOLD` | 1 | positive ticks after C |

`SYMBOL_MARGIN` cannot grow much further: A, C and B are only 1.6–1.7 apart,
and zones that merge would make "any value in 1.2–6.5" a symbol detector.

### The stream (`makePatternStream(seed, length)`)

Each event is, with probability 0.3, a full pattern; with 0.5, one of seven
decoys; otherwise nothing; each event is followed by 1–4 distractors.

| Decoy | Sequence |
|-------|----------|
| no C | A g B |
| wrong order | B g A g C |
| first gap too long | A G B g C |
| second gap too long | A g B G C |
| no B | A g C |
| no A | B g C |
| C alone | C |

(g = 0..n distractors, G = n+1..n+2 distractors.) Six of seven decoys end
in C, so "respond after any C" is not enough.

Labels are computed from the pattern definition over the whole stream, so a
pattern that forms by chance (e.g. a decoy next to a real pattern) counts.
`PatternStream` holds `values`, `target` (+1 / −1) and `afterC` (the tick is
within `PATTERN_HOLD` ticks after any C).

## Measures

Per trial (`scorePredictions`), a prediction is "positive" if the network's
mean output is > 0:

- **balanced accuracy**: 0.5 × (hits on positive ticks + hits on negative
  ticks) over all ticks. Mostly rewards reacting to C.
- **accuracy after C**: the same, restricted to ticks after a C, i.e.
  valid pattern vs. decoy. **This is the sequence measure**: responding to C
  alone scores exactly 0.5 here.

`printShortcutBaselines` scores two rules that ignore part of the sequence on
the same test streams, as reference points:

| Rule | Balanced | After C |
|------|----------|---------|
| respond after any C | ~0.96 | 0.50 |
| respond after C with B 1..n+1 ticks before | ~0.98 | ~0.81 |

A network has learned the sequence only if it beats ~0.81 after C.

## Running a trial

```cpp
using NetworkBuilder = std::function<std::unique_ptr<network>(bool hasER)>;

PatternScore runPatternTrial(const NetworkBuilder& build, std::uint32_t seed, bool hasER,
                             float learningRate, size_t trainTicks = PATTERN_TRAIN_TICKS,
                             RewardMode mode = RewardMode::Target);
```

1. `exr::reseed(seed)`, build the network.
2. Train on stream `1000 + seed` for `trainTicks` ticks: `patternStep`,
   then `applyReward`. Positive ticks are rarer, so their reward is scaled by
   (negatives / positives) to balance the classes.
3. Test on stream `2000 + seed` with learning off.

`patternStep(net, x)` sets input 0 to `x` and every further input to 1.0
(**bias inputs**), steps, and returns the mean of `net.outputs()`.

Divergence (non-finite or |output| > 1e6) is flagged on any tick.

### Reward modes

| Mode | Reward on each training tick |
|------|------------------------------|
| `RewardMode::Target` | always the desired sign |
| `RewardMode::Error` | the desired sign only when the output's sign is wrong, else nothing |

This matters a lot. With `Target`, weights keep moving even when the answer
is already right, until they hit the ±10 clamp, and with balanced rewards the
bias never learns: a linear readout learned the right weight *pattern* but
had no threshold and scored exactly 0.5 after C. With `Error` (a reward
prediction error, perceptron-style), the same readout reaches 0.964 (0.976
with learning gain 1.0).

## Statistics

`measurePattern(build, hasER, learningRate, trials = 50, trainTicks, mode)`
runs each seed twice, trained and as a **control** (learning rate 0, same
seed and streams), and summarises both measures (`Measure`: mean, control
mean, standard error, paired trained − control difference and its standard
error).

`printPatternStats` prints both measures with two verdicts each, based on
t = effect / standard error:

- vs control: LEARNING (t ≥ 2), HARMFUL (t ≤ −2), else NOISE;
- vs chance (0.5): ABOVE CHANCE, BELOW CHANCE, else NOISE.

|t| < 2 is treated as noise (about 95% confidence at 50 trials).

## Showing the topology

Results are only comparable when it is clear which network produced them:

- `printPatternHeader(title, build, hasER)` prints the full
  `network::describe()` output of the topology above the results.
- `topologySummary(net)` gives a one-line form for tables: layers in update
  order as `name(neurons + flags)`, with `*` frozen, `E` E-R on,
  `h` habituation on, `r` recovery jitter, `l` learning jitter; runs like `tap5..tap1` collapsed; then feedback edges
  as `fb from->to(new neurons)`. For example:

  ```
  tap5..tap1(4*) > ramps(7*) > bands(4*) > window(24*) > out(8)
  in(16h) > h1(56h) > h2(37h) > h3(25h) > out(8h); fb h3->h2(5); fb h2->h1(16); fb h1->h1(8)
  ```

### Comparing topologies

The test `NetworkPatternTest.TopologyComparison` runs every candidate
builder with E-R off and on, under identical conditions (error-driven reward,
20 trials each, same streams), and prints them ranked by accuracy after C
with their topology summaries. To evaluate a new topology, write a builder
and add it to the `candidates` list in that test.

## Frozen building blocks

### `addValueDetectors(network&) -> ValueDetectors{ramps, bands}`

Hand-wired, frozen symbol detectors fed by the network's sensor and a bias
input (it calls `addInputs(ramps, 2)`, so it should create the network's
first inputs):

- `ramps` (7): for each symbol s, `k·(x − (s − h))` and `k·(x − (s + h))`,
  saturating at ±10 through the neuron output clamp (h = `SYMBOL_MARGIN`/2,
  k = 20/h), plus a constant +10.
- `bands` (`BAND_FEATURES` = 4): per symbol, half the difference of its two
  ramps minus 5: **+5 inside the band, −5 outside**; plus a constant +5
  (bias feature).

Features are ±5 rather than 0/10 because the learning rule uses only the
sign of each input and never learns from a 0. E-R and habituation are off
(habituation would silence the constant).

### `addDelayWindow(net, source, width, depth, tapOrder) -> window`

A frozen delay line: copy layers `tap1..tapN` (each copies the previous
one) and a `window` layer of `(depth + 1) × width` neurons holding `source`
at lags 0..depth, lag-major. Each lag is its own wiring group (built with
`addFeedback`), so every window neuron reads one source only. The taps are
appended to `tapOrder` oldest first; the caller must build the update order
as `tapOrder`, then the source's layers, then `window`, then the readers.

### `addReservoir(net, source, sourceWidth, inputNeurons, recurrentNeurons, inputScale, recurrentScale, hasER, recoveryJitter = Jitter::none()) -> reservoir`

A frozen random recurrent layer (echo-state style), nothing in it designed
for the task. It has two populations, because a `dense` neuron sums one
wiring group only:

- **input neurons** read `source`, weights scaled by
  `inputScale / sqrt(sourceWidth)`;
- **recurrent neurons** (added with `addFeedback(res, res, n)`) read the
  whole reservoir including themselves, weights scaled by
  `recurrentScale / sqrt(inputNeurons + recurrentNeurons)`.

The input neurons bring in the current tick, the recurrent neurons carry
the past forward. With `recurrentNeurons = 0` there is no recurrence at all,
so any memory must come from the neurons' own state (E-R thresholds). Place
the returned layer after `source` in the update order. Weights come from
the library's random streams, so `exr::reseed` controls them.
`recoveryJitter` spreads the neurons' E-R relaxation rates, i.e. their
memory timescales.

## Memory without hand-built delay lines

Two tests check whether memory can come from something that is not designed
for the task (hand-built detectors in front, 1-neuron readout, 40,000
training ticks, 20 trials):

| Test | Memory | After C |
|------|--------|---------|
| `ReservoirLearnsTheSequence` | random reservoir 40 + 60 (recurrent scale 2), E-R off | 0.814 |
| `ERNeuronsCarryMemoryWithoutRecurrence` | 100 unconnected neurons, E-R off | 0.500 |
| | the same, E-R on (readout E-R off) | **0.740** |

- A frozen **random reservoir** replaces the delay window and still beats the
  shortcut rules; screening with a larger one (100 + 200) reached 0.938.
- **E-R carries memory by itself**: neurons with no connections between them
  have no memory without E-R (exactly 0.500), but with E-R the adaptive
  threshold records recent firing and a readout can decode part of the
  sequence. This depends on clean input features: with random frozen ramps
  instead of the hand-built detectors the effect disappeared in screening.
- With recurrence available, E-R in the reservoir slightly hurts
  (0.887 vs 0.938 at 100 + 200, screening).
- E-R in the **readout** hurts learning (0.735 → 0.527 in the E-R memory
  setup, gain 1.0), so the reservoir builders keep the readout E-R off.

**Per-neuron jitter** ([neuron](neuron.md#per-neuron-dynamics)), 50 paired
trials, measured with base learning gain 1.0 (the default was 1.0 then; it
is now 1.5):

| Change | Setup | Effect on after C |
|--------|-------|-------------------|
| recovery jitter ±0.05 | 100 unconnected E-R neurons | +0.036 ± 0.022 (t 1.7): noise level |
| recovery jitter ±0.08 | same | +0.029 ± 0.025 (t 1.2): noise level |
| recovery jitter ±0.08 | reservoir 40 + 60, E-R (20 trials) | none |
| learning jitter ±0.1 | window + hidden(32) + readout, E-R on | **+0.166 ± 0.033 (t 5.0)**, 0.538 → 0.705 |
| learning jitter ±0.2 | same | **+0.195 ± 0.036 (t 5.4)** |
| learning jitter ±0.1 | window + 8-neuron readout, E-R on | −0.024 ± 0.030: noise level |

The learning-jitter effect is not robust: it appeared and vanished with
every change of the E-R constants (at gain 1.5 it was gone; with baseline
0.2 it moved to gain 2.0). `LearningJitterEffectDependsOnBaseGain` reports it
at two base gains without asserting it.

Fully undesigned (random frozen ramps + random reservoir, screening):
0.728, below the shortcut rules; a perfect window over the random ramps
reaches only 0.776, so value detection, not memory, is what still needs
design.

## Results so far

See `TopologyComparison` for the current ranking (20 trials each). Error-driven
reward unless noted. Trials: 50, or 10 for screening runs. Unless a row says
otherwise, results were measured with recovery 0.9; the current default is
0.8, which mainly moves E-R results (the test comments give both where
measured). Learning gain: 1.5 for the test results, 1.0 for screening runs.

| Topology | Trials | Gain | After C | Balanced |
|----------|--------|------|---------|----------|
| detectors + window + learned readout, E-R off | 50 | 1.5 | **0.964** | 0.99 |
| detectors + window + learned readout, E-R on | 50 | 1.5 | 0.778 | — |
| detectors + window + hidden(32) + readout, E-R off (comparison) | 20 | 1.5 | 0.943 | 0.993 |
| detectors + window + frozen mix + readout, E-R off (comparison) | 20 | 1.5 | 0.974 | 0.995 |
| detectors + window + hidden(32) + readout, E-R on (comparison) | 20 | 1.5 | 0.559 | 0.578 |
| detectors + fully learned feedback network, no window | 10 | 1.0 | 0.51 | 0.50 |
| fully learned feedback network, raw sensor (comparison) | 20 | 1.5 | 0.49 | 0.50 |
| detectors + random reservoir 40+60 + 1-neuron readout, E-R off, 40k ticks | 20 | 1.5 | 0.814 | — |
| detectors + unconnected E-R neurons + 1-neuron readout, 40k ticks | 20 | 1.5 | 0.740 (0.713 with recovery 0.8) | 0.941 |
| detectors + random reservoir 100+200 + readout, E-R off, 40k ticks | 10 | 1.0 | 0.938 | 0.989 |

With gain 1.0 the window readout reached 0.976 (E-R off) and 0.882 (E-R on).

The working topology is in `buildWindowReadoutNetwork` (tests/network.cpp).
Frozen detectors and an explicit delay window make the task nearly linearly
separable; only the readout has to learn.
