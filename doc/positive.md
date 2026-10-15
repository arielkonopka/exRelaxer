# Positive-only values and weights

An option per layer that keeps neuron values non-negative (like firing
rates, 0..n) and weights non-negative (like excitatory synapses). It
composes with every other layer option: E-R or not, habituation, gate,
ReLU, binary, normalisation, any learning rule, growth, freezing, buses.
Off by default; files saved before network format 21 load with it off.

```cpp
net.addLayer("h", LayerSpec::Dense(32).positive());      // values in [0, 10], weights >= 0
net.addLayer("h", LayerSpec::Dense(32).positive(4.0f));  // values in [0, 4]

LayerSpec spec = LayerSpec::Dense(32);
spec.positiveValues = true;    // only the values
spec.positiveWeights = true;   // only the weights
spec.ceiling = 2.0f;           // the value bound, with or without positiveValues
```

```python
spec = exr.LayerSpec.dense(32).positive(4.0)
spec.positive_values, spec.positive_weights, spec.ceiling
```

## Values (`positiveValues`, `ceiling`)

The weighted sum is clamped to `[0, ceiling]` before habituation and E-R
(otherwise to `[-ceiling, ceiling]`; `ceiling` defaults to `max_output`,
10, the clamp every neuron always had). So:

- a negative sum is silence: output 0, and with E-R the threshold relaxes
  as on any silent tick;
- E-R firing, threshold growth and eligibility work as before, on sums >= 0;
- spontaneous firing draws its output from `[0, amplitude]` instead of
  `[-amplitude, amplitude]`;
- binary neurons output 1 or 0 (never -1);
- the pseudo-derivative used by e-prop and surrogate gradients is 0 below
  0, as for a ReLU.

Per neuron: `ValueRange` (`neuron::setValueRange`); per layer:
`neuron_layer::setValueRange`, for every neuron now and later (growth).

## Weights (`positiveWeights`)

Only for layers with learned weights (Dense, Conv2D, LocallyConnected2D);
other types throw. Every weight and every learned bias stays in
`[0, max_weight]`:

- new weights (wiring, growth, new inputs) are the usual random draw's
  magnitudes, `|w|`, so they keep their scale;
- every learning rule clamps at 0 instead of `-max_weight` (the kernels take
  a lower and an upper bound);
- `setWeights`, `setKernel` and `setBias` clamp negative values to 0, so an
  evolution strategy can perturb weights freely;
- turning it on for a wired layer (`neuron_layer::setPositiveWeights`) folds
  the existing weights to `|w|` and negative biases to 0; turning it off
  keeps the weights.

The inputs a layer reads are not changed: a positive-weight layer reading
signed sensors still sees their signs. Fixed matrices outside the layers
(the feedback-alignment matrix, critic and curiosity weights) are not
constrained.

`network::describe` lists each non-default layer as
`range h: values [0, 3], weights >= 0`.

## Results

See [research log](research.md) for the learning rules under the constraint.
