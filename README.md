# exrelaxer

A small C++20 library for experimenting with **biologically inspired neural networks**.
Instead of backpropagation, neurons adapt through two local mechanisms —
**excitation–relaxation** and **habituation** — and learn through a
**reward-modulated Hebbian** rule. Layers can grow at runtime and be wired
with arbitrary feedback (recurrent) connections; growth propagates
automatically to every downstream layer.

> Status: experimental / research code. APIs and the binary serialization
> format may change without notice.

**Documentation:** [doc/](doc/README.md) has a detailed page for each class:
[neuron](doc/neuron.md), [layer](doc/layer.md), [dense](doc/dense.md),
[layer_factory](doc/layer_factory.md), [network](doc/network.md), and the
test-support [pattern_benchmark](doc/pattern_benchmark.md). It also has
PlantUML composition, class and interaction diagrams, and a
[research log](doc/research.md) summarizing every parameter and topology
experiment so far.

## Features

- **Excitation–Relaxation (E-R)** – each neuron has an adaptive firing
  threshold. Firing raises it (fatigue / spike-frequency adaptation),
  silence lets it decay. When it decays to ~0 the neuron fires
  spontaneously at a small amplitude, which re-excites the threshold.
- **Habituation** – if a neuron's weighted input stays unchanged for
  `habituation_steps` consecutive steps, the input is treated as zero until
  the signal changes again.
- **Reward-modulated learning** – `applyReward(reward, learningRate)` moves
  the weights of recently active neurons toward the reward's sign.
- **Networks** – `network` owns a graph of layers with forward and feedback
  edges, input sensors and output layers, runs them in a well-defined order,
  and can freeze individual layers.
- **Pluggable layer types** – layers are created by `layer_factory` from a
  `LayerSpec`; new types plug in by registering a creator.
- **Dynamic topology** – layers can be joined, grown with feedback neurons,
  and fed from external sensors at any time.
- **Parallel forward pass** – with OpenMP, neuron groups are stepped in
  parallel with a thread count scaled to their size (about 12× faster for a
  1000 × 1000 group on 20 threads). Results do not depend on the thread
  count.
- **Serialization** – save and load a whole network, wiring included, either
  with full internal state (continues bit-identically) or weights only.

Both adaptation mechanisms can be toggled independently per layer.

## Requirements

- CMake ≥ 3.20
- A C++20 compiler (GCC, Clang or MSVC)
- Internet access on first configure — GoogleTest v1.15.2 is fetched via
  `FetchContent`
- Optional: OpenMP, for the parallel forward pass (detected automatically)

## Building and testing

```bash
cmake -S . -B build          # Release by default; add -DCMAKE_BUILD_TYPE=Debug to debug
cmake --build build -j
cd build && ctest --output-on-failure
```

| Target              | Description                          |
|---------------------|--------------------------------------|
| `exrelaxer_core`    | Static library (`libexrelaxer_core.a`) |
| `exrelaxer_tests`   | GoogleTest executable                |

Run selected tests directly:

```bash
./build/exrelaxer_tests --gtest_filter='NetworkTest.*'
```

The learning tests run many seeded trials and print statistics with noise
verdicts; the full suite takes under a minute in Release.

## Quick start

Link against `exrelaxer_core`; its public include directory is `core/`:

```cmake
add_subdirectory(exrelaxer)
target_link_libraries(my_app PRIVATE exrelaxer_core)
```

```cpp
#include "network.hpp"

network net;
auto in  = net.addLayer("in",  {LayerType::Dense, 16});
auto hid = net.addLayer("hid", {LayerType::Dense, 32});
auto out = net.addLayer("out", {LayerType::Dense, 4, /*hasHabituation*/ true, /*hasER*/ false});

net.addInputs(in, 2);            // two sensors owned by the network
net.connect(in, hid);            // hid reads in
net.connect(hid, out);
net.addFeedback(out, hid, 8);    // 8 new neurons in hid read out
net.addOutput(out);

for (int t = 0; t < 1000; ++t) {
    net.setInputs({0.5f, -0.2f});
    net.step();                  // forward() on every layer, in update order
    std::vector<float> y = net.outputs();
    net.applyReward(/* the sign y should have */ 1.0f, 0.005f);
}

std::ofstream file("net.bin", std::ios::binary);
net.save(file);                  // network::load(stream) restores it, wiring included
```

Key points (details in [doc/network.md](doc/network.md)):

- **Timing:** `step()` runs layers in topological order of the `connect`
  edges, so a value crosses the forward path in one tick and feedback
  arrives one tick late. `setUpdateOrder()` overrides this, e.g. to build a
  delay line.
- **Wiring:** `connect` wires only the neurons a layer has *now*; call it
  before adding feedback into that layer, and give each layer one `connect`
  source (see [doc/dense.md](doc/dense.md#caveats)).
- **Freezing:** `net.freeze(id)` or `LayerSpec{..., /*frozen*/ true}` keeps a
  layer running but stops it learning.
- **Reward:** the reward is the desired direction of the output. Rewarding
  only when the output is wrong (error-driven) usually learns much better
  than rewarding every tick (see
  [doc/pattern_benchmark.md](doc/pattern_benchmark.md#reward-modes)).
- **Reproducibility:** call `neuron::reseed(seed)` before building a network.

Layers can also be used directly without a network; see
[doc/dense.md](doc/dense.md).

## Tuning

Global constants in [core/neuron.hpp](core/neuron.hpp) (full list in
[doc/neuron.md](doc/neuron.md#tunable-constants)):

| Constant                    | Default  | Meaning |
|-----------------------------|----------|---------|
| `habituation_steps`         | `100`    | Identical steps before the input is suppressed |
| `recovery_factor`           | `0.9`    | Default per-step threshold decay while not firing |
| `baseline_threshold`        | `0.2`    | Resting E-R threshold and learning-eligibility boundary |
| `spontaneous_min_amplitude` | `0.01`   | Amplitude of spontaneous firing |
| `max_weight`                | `10.0`   | Learning clamps each weight to ±this |
| `max_output`                | `10.0`   | Each weighted sum is clamped to ±this |
| `default_learning_gain`     | `2.0`    | Default per-neuron learning gain (multiplies weight updates) |

The per-neuron threshold growth rate `alpha` (default `default_alpha`, 1.2)
is a `neuron` constructor argument. Each neuron also has its own recovery factor,
learning gain and alpha, each of which can be randomized per layer,
independently and optionally (decided at layer creation), with a chosen
distribution (uniform or normal, centre, limits, absolute or relative
spread):

```cpp
net.setRecoveryJitter(res, Jitter::normal(0.02f).around(0.95f));
net.setLearningJitter(hid, Jitter::uniformRelative());   // ±50% of the gain: 2 -> 1..3
net.addLayer("res", {LayerType::Dense, 100, false, true, true,
                     {}, {}, Jitter::uniformRelative()});  // alpha ±50% from creation
```

Relative spreads scale with the parameter (for recovery: with its distance
from 1, so 0.9 ± 50% = 0.85…0.95).

See [doc/network.md](doc/network.md#per-neuron-dynamics). The default
learning gain is `default_learning_gain` (2.0) and the default E-R threshold
growth `default_alpha` (1.2). A small learning-speed jitter sometimes helps
learned hidden layers with E-R, but the effect is not robust to the other
E-R constants.

## Project layout

```
core/
  neuron.hpp/.cpp                single neuron: E-R, habituation, learning, serialization
  network.hpp/.cpp               graph of layers, inputs, outputs, update order, freezing, save/load
  layers/layer.hpp               abstract layer interface
  layers/layer_factory.hpp/.cpp  creates layers by LayerType
  layers/dense.hpp/.cpp          dense layer: wiring groups, growth propagation, parallel forward
doc/                             detailed documentation, one page per class
  diagrams/                      PlantUML sources (*.puml) and rendered SVGs
tests/
  neuron.cpp                     neuron mechanisms and serialization
  dense.cpp                      topology, cascade growth, serialization, stress, Pavlovian
                                 and sequence-order learning
  network.cpp                    factory, graph building, update order, freezing,
                                 serialization, gapped-pattern tests, alpha-jitter experiments
  pattern_benchmark.hpp          gapped-pattern benchmark, statistics, frozen detectors
  er_scales.hpp                  test inputs and timings relative to the E-R constants
```

Tests express their E-R-sensitive inputs and timings relative to these
constants (`tests/er_scales.hpp`): a "weak" stimulus is 5 × `baseline_threshold`,
and waits for spontaneous firing are computed from `recovery_factor`. The
suite passes for `baseline_threshold` from 0.05 to 1.0.

## Known limitations

- **No bias term and a sign-only learning rule.** Neurons have no bias (use
  an input held at 1.0) and learning uses only the sign of each input, so a
  0 input never teaches anything.
- **Learning is weak beyond the last layer.** Every eligible neuron gets the
  same reward; there is no per-neuron credit assignment. Tasks work well when
  frozen or hand-wired layers provide the features and memory and only a
  readout learns. Gapped-pattern detection (`A,{0..1},B,{0..1},C` among
  decoys) reaches ~0.96 valid-vs-decoy accuracy that way, and ~0.81–0.94
  with a frozen *random* reservoir as memory; fully learned feedback
  networks stay at chance. E-R neurons carry some memory on their own
  (0.74 with no connections between them, vs 0.50 without E-R).
- **E-R makes learning steps very large.** Eligibility is
  `threshold / baseline_threshold − 1` and a firing lifts the threshold well
  above baseline, so E-R neurons take big steps and hit the weight clamp
  quickly. There is no weight decay.
- **Bounded, not squashed.** Weights and weighted sums are clamped at ±10;
  without the output clamp, networks with feedback loops diverge.
- **Shared random streams.** Initial weights depend on everything created
  earlier in the process unless `neuron::reseed` is called first.
- **Serialization is not portable** (raw binary, native endianness), and a
  saved network can only be loaded by a program that registers all its
  layer types. Files from older format versions load as weights only.
- **One source per `connect`ed layer.** A `dense` layer joined to two
  sources steps its neurons once per group and keeps only the last result.
