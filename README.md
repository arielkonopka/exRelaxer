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
[kernels](doc/kernels.md), [layer_factory](doc/layer_factory.md),
[network](doc/network.md), and the
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
- **Vision layers** – `Retina` reads camera images (every pixel, or samples
  on a tight spiral out from the centre), `Conv2D` (shared kernels),
  `LocallyConnected2D` (own weights per position) and `Pool2D`. A 320 × 200
  pipeline with about 2 million neurons steps in about 18 ms. Fixed filter
  banks (Gabor, centre-surround, Gaussian) turn a Conv2D into a frozen
  feature detector (see [doc/spatial.md](doc/spatial.md)).
- **Pluggable layer types** – layers are created by `layer_factory` from a
  `LayerSpec`; new types plug in by registering a creator.
- **Dynamic topology** – layers can be joined, grown with feedback neurons,
  and fed from external sensors at any time.
- **Fast, deterministic kernels** – layers own their weights in a SIMD
  layout (8 neurons per pass) and run large groups on several threads with
  OpenMP, for both the forward pass and learning. Results are bit-identical
  to plain scalar code whatever the SIMD width or thread count (see
  [doc/kernels.md](doc/kernels.md)).
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

`-DEXRELAXER_NATIVE=ON` builds for the build machine's CPU (e.g. AVX2);
results are the same either way.

The learning tests run many seeded trials and print statistics with noise
verdicts; the full suite takes under a minute in Release.

Experiments (how well and how fast networks learn, across architectures and
settings) run with the benchmark harness `nntest`; see
[NNtesting/README.md](NNtesting/README.md):

```bash
./build/NNtesting/nntest list
./build/NNtesting/nntest run gapped_pattern --set topology=window_readout,reservoir --out results.jsonl
NNtesting/tools/compare.py results.jsonl
```

## Quick start

Link against `exrelaxer_core`; its public include directory is `core/`:

```cmake
add_subdirectory(exrelaxer)
target_link_libraries(my_app PRIVATE exrelaxer_core)
```

```cpp
#include "network.hpp"
using namespace exr;   // everything is in namespace exr

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
- **Wiring:** `connect` wires only the neurons a layer has *now*, so call it
  before adding feedback into that layer. A layer connected from several
  sources (or sources plus `addInputs` sensors) sums them all in one
  weighted sum per neuron (see [doc/dense.md](doc/dense.md#caveats)).
- **Freezing:** `net.freeze(id)` or `LayerSpec{..., /*frozen*/ true}` keeps a
  layer running but stops it learning.
- **Reward:** the reward is the desired direction of the output. Rewarding
  only when the output is wrong (error-driven) usually learns much better
  than rewarding every tick (see
  [doc/pattern_benchmark.md](doc/pattern_benchmark.md#reward-modes)).
- **Reproducibility:** call `exr::reseed(seed)` before building a network.

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
  neuron.hpp/.cpp                single neuron's dynamics: E-R, habituation, eligibility, serialization
  kernels.hpp/.cpp               SIMD weight matrix, weighted sums, learning rule, thread sizing
  random.hpp/.cpp                the library's random streams, exr::reseed
  network.hpp/.cpp               graph of layers, inputs, outputs, update order, freezing, save/load
  layers/layer.hpp/.cpp          base class: output buffer, Shape, InputRange, growth notifications
  layers/neuron_layer.hpp/.cpp   base of neuron layers: neurons, jitter, neuron-record serialization
  layers/layer_factory.hpp/.cpp  creates layers by LayerType
  layers/dense.hpp/.cpp          dense layer: wiring groups, growth propagation, SIMD forward and learning
  layers/spatial.hpp/.cpp        Window2D, retina and pooling specs, the channels a spatial layer reads
  layers/spatial_neuron_layer.*  base of Conv2D and LocallyConnected2D
  layers/conv2d.hpp/.cpp         convolution with shared kernels
  layers/locally_connected2d.*   convolution geometry, own weights per position
  layers/pool2d.hpp/.cpp         max / average pooling
  layers/retina.hpp/.cpp         image input: grid or spiral sampling
  filters.hpp/.cpp               fixed filter banks for Conv2D: Gaussian, difference of Gaussians, Gabor
  parallel.hpp                   splitting work between OpenMP threads
  binary_io.hpp                  binary stream I/O for serialization
doc/                             detailed documentation, one page per class
  diagrams/                      PlantUML sources (*.puml) and rendered SVGs
tests/
  neuron.cpp                     neuron mechanisms and serialization
  dense.cpp                      topology, cascade growth, serialization, stress, Pavlovian
                                 and sequence-order learning
  network.cpp                    factory, graph building, update order, freezing,
                                 serialization, gapped-pattern tests, alpha-jitter experiments
  kernels.cpp                    SIMD kernels and dense layers bit-identical to scalar references
  regressions.cpp                one test per fixed bug; DISABLED_ tests for open ones
  spatial.cpp                    retina, Conv2D, LocallyConnected2D, Pool2D against scalar references
  filters.cpp                    filter banks, and bar-orientation learning with frozen Gabor features
  er_scales.hpp                  test inputs and timings relative to the E-R constants
NNtesting/                       benchmark harness nntest (see NNtesting/README.md)
  harness/                       runner: parameters, trials, statistics, result files
  tasks/                         task code shared with the unit tests (pattern_benchmark.hpp, bars.hpp)
  experiments/                   one file per experiment
  datasets/fetch.py              downloads public datasets (MNIST, CIFAR-10, Kaggle, ...) into one format
  tools/compare.py               tables and before/after comparisons of result files
  tools/spiral.py                spiral sampling of images, bit-identical to the Spiral retina
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
  earlier in the process unless `exr::reseed` is called first.
- **Serialization is not portable** (raw binary, native endianness), and a
  saved network can only be loaded by a program that registers all its
  layer types. Files from older format versions load as weights only.

## Changelog

### 2026-09-27: datasets and spiral tool

- `NNtesting/datasets/fetch.py`: downloads MNIST, Fashion-MNIST, CIFAR-10,
  the Kaggle casting-inspection set, Speech Commands and ESC-50 (the audio
  sets download only), or any Kaggle dataset (`fetch.py kaggle OWNER/SLUG`,
  with or without API credentials). Image sets are prepared as
  `SPLIT_images.npy` (uint8, N × C × H × W), `SPLIT_labels.npy` and
  `meta.json`; downloads are checked against pinned SHA-256 hashes where
  known and recorded otherwise.
- `NNtesting/tools/spiral.py`: spiral sampling of images and prepared
  datasets from the command line, with polar, reconstruction and path
  views. Bit-identical to the C++ Spiral retina, checked by the CTest test
  `spiral_matches_retina`.

### 2026-09-26: benchmark harness

- `NNtesting/`: the `nntest` runner, one file per experiment in
  `NNtesting/experiments/`, registered automatically. Seeded trials,
  parameter grids (`--set a=1,2 --set b=x,y`), per-metric statistics,
  checks on default runs, JSON Lines results with the machine, build and git
  commit, and `tools/compare.py` for tables and before/after comparisons.
- Experiments: `bar_orientation`, `gapped_pattern` (window, mixed window or
  reservoir topologies), `dense_throughput`, `vision_throughput`. The
  `quick` ones also run in CTest (`nntest_quick`).
- `tests/pattern_benchmark.hpp` moved to `NNtesting/tasks/`, with the
  detector front end formerly private to the unit tests; the bar images
  moved to `NNtesting/tasks/bars.hpp`. Unit tests use them from there.
- Corrected stale measurements in docs and test comments (reservoir 0.798,
  unconnected E-R neurons 0.757).
- Found with the harness: small dense layers run slower on the default
  20-thread pool than on one thread (see NNtesting/README.md).

### 2026-09-26: filter banks

- `core/filters.hpp` (`exr::filters`): `gaussian`, `differenceOfGaussians`
  (on- and off-centre), `gabor`, `gaborBank`, `centreSurroundBank`, and
  `load(conv2d&, bank)` to put a bank into a Conv2D's kernels. DoG and Gabor
  filters have zero mean and give exactly `gain` for their best
  full-contrast pattern.
- `network::layerAs<T>(id)`: a layer as its concrete type.
- New learning test: frozen Gabor features + a learned readout tell
  vertical from horizontal bars at 0.99 accuracy (0.52 without learning),
  10 seeded trials.
- Tests no longer leak `std::cout` formatting (`std::fixed`,
  `std::setprecision`) into later tests' output, which had made printed
  results depend on test order. The results themselves are unchanged.

### 2026-09-26: vision layers

New layer types for images (see [doc/spatial.md](doc/spatial.md)):

- **`Retina`** reads a camera image (`net.addInputs(eye, Shape{1, 200, 320})`)
  with one neuron per sample and channel, so habituation and E-R adapt per
  sample. Two samplings:
  - Grid: one sample per pixel. With habituation and E-R off this is a plain
    Input2D.
  - Spiral: sample 0 at the image centre, then outwards on a tight
    Archimedean spiral, `spacing` pixels between samples and between turns,
    each interpolated from its 4 nearest pixels. Output `channels × 1 × N`,
    in order of distance from the centre.
- **`Conv2D`**: one kernel per output channel shared by all positions, own
  neurons per position. Kernels learn the mean update of their eligible
  neurons, with the same result for any thread count.
- **`LocallyConnected2D`**: own weights per position; learns exactly like
  `dense`.
- **`Pool2D`**: max or average, padding skipped.
- Spatial layers read one or more sources of the same height × width and
  concatenate their channels, keeping one weighted sum per neuron; a flat
  source that grows adds channels.
- `LayerSpec` gains `window`, `pool` and `retina` fields and builders
  (`LayerSpec::Conv2D(16, Window2D::square(5, 1, 2))`, ...);
  `layer::shape()`, `learns()` and `hasNeurons()`;
  `network::addInputs(target, Shape)`; `describe()` shows spatial shapes.
- Network file format 7 (spatial parameters). Version 6 files still load
  with their full state.
- Performance, 320 × 200, i7-12700H: retina 0.45 ms; Conv2D 16 × 5 × 5
  (1 M neurons) 2.8 ms forward; Conv2D 32 × 5 × 5 on 16 × 100 × 160
  (205 M multiply-adds) 6.5 ms forward; the whole pipeline about 18 ms per
  step, 35 ms with learning.
- `tests/spatial.cpp`: each layer against a scalar reference (bit for bit
  for forward passes and LocallyConnected2D learning), save/load
  continuation of a vision network, thread-count independence of Conv2D
  learning.

### 2026-09-26: references instead of raw pointers

- Stored references are `std::reference_wrapper` (`InputRange::buffer`,
  a layer's readers and sources, a group's sources); "no source" is a
  `std::optional`. An `InputRange` can no longer be bound to a temporary
  vector (compile error).
- Kernels take `std::span` parameters (bounds-checked in debug builds);
  SIMD loads and stores go through fixed-size spans.
- Stream I/O goes through `binary_io.hpp`, the one place with the byte
  casts iostreams require. `network` uses reference `dynamic_cast` behind
  `layer::hasNeurons()`.
- No measurable performance change: single-thread kernels within 0.3% of the
  pointer version. Replacing the SIMD `memcpy` with `std::bit_cast` was tried
  and rejected (up to 4× slower).

### 2026-09-26: layer-owned SIMD weights, `exr` namespace

Breaking API changes. Every result is bit-identical to before, and saved
files keep the same format, so older networks still load. This was checked
with a harness that hashes every output and weight over seeded networks and
a save/load round trip, and against the learning statistics the test suite
prints.

**API**

- Everything is in `namespace exr`. `exr::reseed(seed)` replaces
  `neuron::reseed`.
- `neuron` holds only its dynamics: `activate(sum)`, `eligible()`,
  `learningDelta()`. It no longer owns weights. `step(inputs, weights)` and
  `learn(...)` run one neuron against weights you keep yourself.
- The layer hierarchy is `layer` → `neuron_layer` → `dense`:
  - `layer` owns one contiguous output buffer and a `Shape` (channels ×
    height × width), and passes growth notifications to its readers.
  - `neuron_layer` owns the neurons, the per-neuron jitter, and saving and
    loading neuron records.
  - `dense` adds only its wiring rule and the weights.
- Outputs: `output()` returns a span; `setOutput(i, v)` drives an output by
  hand. Readers refer to a source's buffer by range (`InputRange`), and
  sensors are plain `std::vector<float>`.
- Weights: `dense::weights(i)`, `setWeights(i, w)` (the size is checked) and
  `inputCount(i)`.
- Wiring calls (`join`, `attachInputs`, `addNeurons`) throw
  `std::logic_error` instead of returning `bool`.
- Every neuron is in at most one wiring group. A layer connected from
  several sources, or from sources and sensors, sums them all in one
  weighted sum per neuron.

**Performance**

Weights are stored in blocks of 8 neurons, so one pass over the inputs
computes 8 weighted sums with SIMD. Each sum still adds its inputs in the
same order as a scalar loop, which keeps results exact; on one thread this
is 9× faster than the scalar code. Learning now runs in parallel, and each
thread handles the same neurons in the forward pass and in learning, so
their weights are still in its cache. Best of 15 runs, i7-12700H:

| Workload | Before | After | Speed-up |
|---|---|---|---|
| dense 1000×1000, step + reward | 966 µs | 78 µs | 12× |
| dense 4096×4096, step + reward | 12.5 ms | 3.4 ms | 3.7× |
| dense 256×256, step | 44 µs | 12 µs | 3.6× |
| reservoir (500 neurons), step | 28 µs | 12.5 µs | 2.2× |
| 8 layers × 16 neurons, step + reward | 3.3 µs | 1.5 µs | 2.2× |

Link-time optimization, OpenMP's passive wait policy and dynamic thread
scheduling were tried and did not help. The 4096×4096 forward step is
limited by memory bandwidth. See [doc/kernels.md](doc/kernels.md).

**Build**

- `-ffp-contract=off`: results do not depend on the compiler fusing
  multiply-adds.
- New option `-DEXRELAXER_NATIVE=ON` builds for the build machine's CPU.
  It is off by default; the default build reaches about the same speed.

**Fixes**

- A layer connected from two sources re-initialized its neurons' weights
  for the second source and then read past the end of them.
- Sensors attached to a layer were extended with the layer's own new
  outputs when it grew.
- `attachInputs` reached only a layer's first wiring group.
- A neuron in two groups learned twice per reward.
- Loading no longer trusts the data: a weight count that does not match the
  wiring, a corrupt weight count, or another neuron count on a wired layer
  now throws `std::runtime_error` and leaves the layer unchanged.

**Tests**

- `tests/kernels.cpp` checks the SIMD kernels and whole `dense` layers bit
  for bit against scalar code, including partial blocks and the
  multi-threaded path.
- `tests/regressions.cpp` has one test per fixed bug. `DISABLED_` tests
  describe the two bugs still open.
- The full suite runs clean under AddressSanitizer, UndefinedBehaviorSanitizer
  and Valgrind memcheck.
