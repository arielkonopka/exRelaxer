# dense

`core/layers/dense.hpp`, `core/layers/dense.cpp`

The standard layer: a growable set of [neurons](neuron.md), each reading a
whole input pool (fully connected within its group). Derives from
[neuron_layer](layer.md#neuron_layer).

```cpp
explicit dense(size_t count = 0, bool hasHabituation = true, bool hasER = true,
               const Jitter& recoveryJitter = {}, const Jitter& learningJitter = {},
               const Jitter& alphaJitter = {});
```

Creates `count` neurons with the given mechanisms, not wired yet. Every
neuron the layer adds later (feedback growth) gets the same flags and jitter
([per-neuron dynamics](neuron.md#per-neuron-dynamics)). A layer may start
with 0 neurons and grow only through feedback.

## Wiring groups

A `dense` layer organises its neurons in **wiring groups**. A group is:

- an **input pool**: the concatenated outputs of its source layers and any
  attached sensors (as [input ranges](layer.md#outputs-and-inputs)), plus
- the **neurons** that read that pool, always a contiguous run of neurons,
  with one weight row each, one weight per pool entry in pool order,
- the **source** layers in the pool, used for growth propagation (sensors are
  not a source and never grow).

Every neuron belongs to at most one group (none until it is wired). A neuron
reading several sources integrates them in one weighted sum over the group's
pool. This is enforced: all groups are created in one place, which throws
`std::logic_error` if a neuron is already in a group.

Each call below creates or extends a group:

| Call | Effect |
|------|--------|
| `B.join(A)` | **all neurons B has now** read all of `A`'s outputs. `A`'s outputs are appended to the pool of each existing group of `B` (with matching random weights; skipped if the group already reads `A`); neurons of `B` in no group form a new group with fresh random weights. `B` becomes a reader of `A`. |
| `C.addNeurons(n, A)`, or `A.addFeedback(C, n)` | new group in `C`: **n new neurons**, reading all of `A`'s outputs. `C`'s existing neurons are untouched. `C` becomes a reader of `A`, and `C`'s own readers are told that `C` grew. |
| `A.attachInputs(sensors)` | like `join`: the sensors are appended to the pool of **every group of A** (its neurons get extra weights), and neurons in no group form a new group reading the sensors. |

Inspection: `groupCount()`, `groupNeurons(g)` (a `{first, count}` range) and
`groupOf(i)` (`dense::no_group` for a neuron not wired yet).

### Growth propagation

When a layer grows (neurons added to it), it notifies every reader of the
new output range. Each reader finds every group whose sources include the
grown layer, appends the new range to the pool and new random weights to
every row of the group. This cascades naturally: if the grown layer's growth
adds neurons elsewhere, those layers notify their readers in turn.

```
B.join(A); C.join(B);
X.addFeedback(B, 3);   // B: +3 neurons -> C's group reading B gets 3 more inputs,
                       // and every neuron in that group 3 more weights
```

![Feedback growth](diagrams/sequence_growth.svg)

### Caveats

- **`join` wires the neurons present now.** Neurons added later by feedback
  are in their own group and do not read sources joined before they existed.
- **Several sources share one sum.** Joining `A` and then `B` gives each
  neuron one weight row over `[A..., B...]`. To keep sources apart, give
  each its own neurons with `addNeurons` (`addDelayWindow` in the pattern
  benchmark does this).
- **Pool order is append order.** When a source grows, its new outputs go to
  the end of the pool, after any sources or sensors added since.
- **Self-feedback.** A group whose source is the layer itself (`A.join(A)`
  or `A.addFeedback(A, n)`) reads a copy taken just before the group runs,
  so all its neurons see the same values regardless of order.

## forward()

For each group, in creation order:

1. **Gather**: copy the group's pool into a contiguous `float` buffer (one
   `memcpy` per input range; adjacent ranges of one buffer are merged).
2. **Weighted sums** of every neuron at once with the SIMD kernel
   ([kernels](kernels.md)): the group's weights are stored in blocks of 8
   neurons, so one pass over the inputs computes 8 sums.
3. **Activate** each neuron with its sum and write its output.

![One tick](diagrams/sequence_step.svg)

Because inputs are copied first, a neuron in a group sees the values from
before the group ran. Groups run one after another, so a later group of the
same layer sees the updated outputs of earlier groups.

## applyReward()

For each group: gather its current pool, take the sign of every input, ask
every neuron whether it is eligible and for its step
([neuron](neuron.md#learning)), then update all rows of eligible neurons
with the SIMD kernel. That is the default `Sign` rule; the other
[learning rules](learning.md) use the group's input trace instead of the
signs (kept during `forward()`), and `applyModulators` / `applyFeedback`
(from [neuron_layer](layer.md#neuron_layer)) give each neuron its own
modulator. Rows of ineligible neurons stay exactly as they are.
Call it right after `forward()` so the inputs are the ones the neurons
stepped on.

## Parallelism

With OpenMP, a group with enough work (neurons × inputs) is split into one
contiguous run of weight blocks per thread, with `work / 32768` threads
capped at the OpenMP maximum; smaller groups run serially. `forward()` and
`applyReward()` split a group the same way, so each thread learns on the
weights it just multiplied, which are still in its cache. Results are
identical for any thread count. See
[kernels: parallelism](kernels.md#parallelism) for the measurements. To limit
threads, set `OMP_NUM_THREADS`.

## Weights

| Method | Purpose |
|--------|---------|
| `weights(i)` | neuron `i`'s weights in pool order (empty when not wired) |
| `setWeights(i, w)` | replace them; `w.size()` must equal `inputCount(i)`, else `std::invalid_argument`. Not clamped, so hand-wired neurons may use weights beyond ±10 |
| `inputCount(i)` | how many inputs neuron `i` reads |

Hand-wiring (fixed weights for detectors, copy layers, etc.) uses
`setWeights`.

## Serialization

See [neuron_layer: serialization](layer.md#serialization-1). `dense`
supplies each neuron's weight row for its record and checks, on load, that
every wired neuron's weight count equals its group's pool size. To save
wiring as well, use [network::save](network.md#serialization).
