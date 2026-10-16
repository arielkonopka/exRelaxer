# precision: 8-bit weights and values

`Precision` (`core/neuron.hpp`), `LayerSpec::precision`,
`neuron_layer::setPrecision`, `kernels::weight_matrix::quantize8`

A per-layer option, like the gate or the learning rule, that composes with
every other layer setting (E-R, habituation, normalisation, any learning
rule, growth, freezing, buses). The default is float; old networks load as
float.

```cpp
LayerSpec spec = LayerSpec::Dense(16, false, true);
spec.precision = Precision::Int8();          // 8-bit weights and outputs, output step 1/16
spec.precision.int8Values = false;           // or only one of them
```

```python
spec = exr.LayerSpec.dense(16, False, True)
spec.precision = exr.Precision.int8()        # exr.Precision(int8_weights=True, value_step=1/32), ...
```

Experiments: `nl_static --set precision=float,int8,int8w,int8v` (hidden
layers; `readout_precision` for the readout, `value_step` for the step) and
`snake_growth --precision int8 --readout-precision int8`.

## The number format

An 8-bit number is an integer code in [−127, 127] times a **power-of-two
step**. With power-of-two steps, every product of an 8-bit value and an 8-bit
weight is exact, and a weighted sum is integer arithmetic followed by one
scaling, which is what an int8 kernel (SIMD, RISC-V vector or a dedicated
accelerator) computes.

| Setting | What is 8-bit | Step | Layer types |
|---|---|---|---|
| `int8Weights` | every weight | per neuron (weight row): the smallest power of two with 127 · step ≥ the row's largest \|w\|, so it follows the row as it grows or shrinks | Dense |
| `int8Values` | every output, after E-R, habituation, gate and ReLU | `valueStep`, a fixed power of two in [2⁻¹², 2⁻³]; default 1/16, which covers ±7.94 (`max_output` is 10) | Dense, Conv2D, LocallyConnected2D |

Outputs are rounded to the nearest code (ties to even) and clamped to ±127
codes. What stays float: E-R thresholds, habituation state, the learned bias,
learning traces, eligibility, gradients, and sensor values (quantize those at
the source if an all-integer forward pass is wanted). A layer reading only
8-bit layers therefore sums 8-bit numbers only.

## Learning on the 8-bit grid

Learning computes each update in float as usual, then rounds every changed
weight back to its row's grid **stochastically**: up with probability equal to
the remainder (w = 10.3 steps becomes 11 with probability 0.3, else 10). So an
update smaller than one step still counts on average, and the weights never
hold more than 8 bits: there is no hidden float copy. Nearest rounding would
undo every update smaller than half a step.

- The uniform draw is a hash of a per-layer seed, the row and the column:
  the same result for any thread count. The seed comes from a SplitMix64
  counter drawn from the learning random stream when 8-bit weights are first
  turned on, and the counter is saved with the layer, so a saved and
  reloaded network continues exactly as one that never stopped.
- A weight already on the grid never moves when it is rounded again. When a
  row's largest weight crosses a power of two the row's step doubles (or
  halves), and the row is re-rounded.
- Wiring, growth, `setWeights` and loading round to the nearest code.
- Every rule works, including the per-synapse rules and surrogate gradients:
  rounding runs after each update (`applyReward`, `applyModulators`,
  `applyFeedback`, `applyGradients`). Frozen columns are put back before the
  rounding; a frozen weight changes only when its row's step grows.
- 8-bit outputs use straight-through derivatives (e-prop, surrogate): the
  rounding is ignored in the gradient.

## What this is not (yet)

The weights are kept in the layer's float matrix, restricted to 8-bit values,
so memory and speed are unchanged. The numbers are exactly what packed int8
storage would hold, so a packed int8 forward kernel (the 4× memory-bandwidth
saving for large layers) would give the same results bit for bit, and is a
separate step.

## Serialization

Network format 21 saves `int8Weights`, `int8Values` and `valueStep` per layer.
A layer with 8-bit weights ends its learning state with its rounding counter
(a layer setting, like the gate: a standalone layer must be given its
precision before `deserialize`). Files before format 21 load as float.

## Results

Research log §32, raw data in `results/precision/`. In short: on nl_static
8-bit makes no difference to E-R networks and costs ReLU networks about 2-3×
in error floor through the 8-bit outputs (weights alone nearly nothing); on
Snake an 8-bit ER(16) learns online as well as a float one (9.6-9.7 vs 9.8
apples), and 8-bit readouts too cost about 0.6 apples.
