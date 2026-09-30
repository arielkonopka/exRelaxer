# State output: three values per neuron

Experimental, opt-in. A **State layer** makes a neuron layer's internal
state readable by the rest of the network: for every neuron of its source
it outputs **three values**, the neuron's output, its E-R threshold and its
habituation streak. Layers and readouts connected to it read them like any
other output. Nothing else changes: the source neurons behave exactly as
before, and a network without a State layer is bit-identical to one built
before the layer existed (`StateTapTest.AddingATapChangesNothingElse`).

Why: the outputs alone hide most of what E-R keeps. The threshold still
holds an event long after the neuron has gone silent, and a downstream
neuron that reads only outputs cannot see it ([research log §25](research.md#25-temporal-semantics-state-readability-and-delayed-credit)).
The hypothesis tested here (§26): *reading the threshold and the
habituation streak as inputs lets downstream neurons, readouts and the
existing learning rules use that stored information, without changing the
neuron.*

## The three values

Per neuron *i* of the source layer, after tick *t*, in this order:

| # | Field | Value | Range | Without the mechanism |
|---|---|---|---|---|
| 1 | output | yᵢ(*t*), the source's own output | [−max_output, max_output] | (always present) |
| 2 | threshold | θᵢ(*t*) − ρ: the E-R threshold above rest | [−ρ, max_output − ρ] | 0 without E-R |
| 3 | habituation | min(cᵢ(*t*) / onset, 1) | [0, 1] | 0 without habituation |

- **output** is the value the source layer itself outputs, so a reader of
  the State layer needs no second connection to the source.
- **threshold** is 0 at rest, positive after a firing (it grows with the
  firing's size; see [model](model.md#activation-and-e-r-threshold-dynamics-neurons-with-e-r)),
  and turns negative while the threshold relaxes below rest during a long
  silence (it relaxes towards 0, then fires spontaneously). Subtracting ρ
  makes "nothing happened lately" 0 and puts the value in the same units
  as outputs and sums.
- **habituation**: cᵢ is the number of consecutive ticks whose raw sum was
  "the same" (`neuron::habituationStreak`), onset the streak length at
  which suppression starts (`Habituation::onset`: `steps` in cut mode,
  `fadeAfter` in fade mode). 0 = the input just changed, 1 = the input is
  being suppressed.

The layer's output is flat and interleaved per neuron,
`[y₀, θ₀−ρ, h₀, y₁, θ₁−ρ, h₁, ...]`, so it has 3 × N values for a source
of N neurons, and a source that grows (feedback neurons) only appends.
Any of the three fields can be switched off; the others keep their order.

## Using it

C++:

```cpp
auto h = net.addLayer("h", LayerSpec::Dense(64));
auto s = net.addLayer("h_state", LayerSpec::State());  // output, threshold, habituation
net.connect(h, s);          // exactly one source, a layer made of neurons
net.connect(s, readout);    // the readout reads 3 x 64 values
// LayerSpec::State(output, threshold, habituation) picks fields, e.g.
// State(false, true, false): thresholds only (next to a connection to h itself).
```

Python:

```python
s = net.add_layer("h_state", exr.LayerSpec.state())            # three values per neuron
s = net.add_layer("h_state", exr.LayerSpec.state(output=False)) # threshold and habituation only
net.connect(h, s)
```

`connect` throws for a second source or a source without neurons; `State`
with every field off throws. The layer has no neurons, weights or learning,
and nothing to save: saved networks replay it and it recomputes its values
from the source on the next tick. In a `LayerSpec` the fields are the bits
of `size` (1 output, 2 threshold, 4 habituation).

## Timing

The State layer runs after its source in the update order (a `connect`
edge). A layer reading it forward sees the three values of tick *t* in
the same tick, exactly as it would see y(*t*); a feedback neuron reading it
sees tick *t*−1; before the first tick all values are 0
(`StateTapTest.ReadersSeeTheStateOfTickTAndFeedbackSeesTMinusOne`). A
learning rule on the reader treats the values like any other input: the
Sign rule uses their sign, the trace rules their trace (see
[model](model.md#learning-update)).

## What it changes: the evaluation

Every controlled benchmark except Doom was run with and without a State
layer, from the same seeds (research log
[§26](research.md#26-state-as-output); data and tables in
[`results/state-output/`](../results/state-output/), commands in
`NNtesting/tools/state_output_sweep.sh`).

| Benchmark | Readout or learner | Without | With State | Where it helps |
|---|---|---|---|---|
| `memory_readability`, recovery 0.9, blank 16 | online delta rule | 0.74 | 0.92 | the event is gone from the outputs but not from the thresholds |
| `memory_readability`, recovery 0.99, blank 128 | online delta rule | 0.78 | 0.95 | |
| `video_memory_learned`, recovery 0.99, gap 4 | online, immediate | 0.51 | 0.83 | the online readout finally uses E-R memory, with no trace |
| `video_memory_learned`, recovery 0.99, gap 4 | ridge | 0.72 | 0.88 | |
| `delayed_credit`, Sign rule, delay 1–64 | one learning neuron | success 0 | success 1.0 (to d = 8 at recovery 0.9, to 64 at 0.99) | the relay threshold is still above rest when the reward comes |
| `delayed_credit`, Trace λ 0.9, delay 32 | one learning neuron | response gap 0.1 | 16.2 | |
| `nl_temporal` t1, x(t) XOR x(t−1), with habituation | end-to-end (FA) | 0.50 | 1.00 | habituation cuts the outputs, the thresholds keep the last step |
| `nl_temporal` t1, no habituation | end-to-end | 0.95 ± 0.09 | 0.98 ± 0.02 | none: at lr 0.03 both reach 0.99–1.00 |
| `dyn_ladder` change | end-to-end | 0.91 | 0.96, 25% fewer spikes | |
| `dyn_ladder` catch / vel | end-to-end | 0.22 / R² −0.04 | 0.25 / R² 0.07 | small |
| `nl_temporal` t3 (parity of 3) | end-to-end | 0.75 | 0.55 | **worse**: more inputs, same learning rate grid |
| `dyn_ladder` dir | end-to-end | 0.57 | 0.53 | **worse** |
| `nl_temporal` t2, t4 | end-to-end | 0.74, R² < 0 | 0.74, R² < 0 | no change (not learned either way) |

In short: when the information is in the threshold and the outputs no
longer show it (a blank, a delay, a suppressed input), reading the
threshold helps a lot, most of all the learning rules that have no memory
of their own (the online delta rule, the Sign rule). Where the outputs
already carry the answer, the extra inputs change little or cost a few
points. **The habituation value never helped on its own**: in
`memory_readability` it is at chance (it records when the input changed,
which is the same for both classes), and in `nl_temporal` a ReLU network
with habituation, whose State layer carries only the streaks (no E-R, no
threshold), stays at chance on t1 (0.50) where E-R reaches 1.00 through the threshold.

**Which value carries it** ([§27](research.md#27-state-component-ablation),
[`results/state-ablation/`](../results/state-ablation/)): a paired
ablation over the six combinations of the three values, on the occluded
video direction, finds the threshold alone reaches what all three reach
(online readout 0.83 at gap 4, recovery 0.99, against 0.51 on outputs);
the outputs next to it add information only for an offline readout, and
the habituation streak carries no class information there and costs the
online readout 3–5 points.

## Limits

- The habituation value carries *when* the input last changed, not *what*
  it was; it is useful only where the timing of repeats is the signal.
- The threshold value mixes how strongly and how long ago a neuron fired,
  so a readout sees an event's age and size as one number.
- It adds 2 inputs per source neuron to every reader (3 when the reader
  reads only the State layer), and the readers' learning rate may need
  retuning (the Sign rule's large steps in `delayed_credit` needed a 10×
  smaller rate to keep weights off the ±10 clamp).
- Not implemented: the Sign rule's eligibility, the traces or the previous
  input as fields; spatial shapes (the output is flat, so conv layers
  cannot read it as an image).
