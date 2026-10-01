# exrelaxer

A C++20 library, with Python bindings, for building and studying
**biologically inspired neural networks** that see, hear and act.
Instead of backpropagation, neurons adapt through two local mechanisms,
**excitation–relaxation** and **habituation**, and each layer learns by a
local rule of its choice: reward-modulated Hebbian learning, eligibility
traces, feedback alignment, node perturbation, or the unsupervised Oja and
BCM rules.

Networks are graphs of layers that can grow at runtime and be wired with
arbitrary feedback (recurrent) connections. Vision layers (retina,
convolution, pooling, fixed filter banks), audio layers (a cochlea with one
or more microphones, spectrograms), stereo vision (binocular disparity) and
named input sources let one network combine several senses. Kernels are
SIMD and multithreaded, results are deterministic, and whole networks save
and load. A benchmark harness (`nntest`, in C++ and Python) runs
experiments such as MNIST, snake, chirp direction, stereograms and
audio-visual objects.

> Status: experimental / research code. APIs and the binary serialization
> format may change without notice.

**Documentation:** [doc/](doc/README.md) has a detailed page for each class:
[neuron](doc/neuron.md), [layer](doc/layer.md), [dense](doc/dense.md),
[learning](doc/learning.md), [kernels](doc/kernels.md), [spatial](doc/spatial.md), [audio](doc/audio.md),
[multimodal and stereo](doc/multimodal.md),
[activity economy](doc/activity.md), [nonlinearity](doc/nonlinearity.md), [dynamic ladder](doc/dynamic.md),
[layer_factory](doc/layer_factory.md),
[network](doc/network.md), and the
test-support [pattern_benchmark](doc/pattern_benchmark.md). It also has
PlantUML composition, class and interaction diagrams, and a
[research log](doc/research.md) summarizing every parameter and topology
experiment so far.

## Features

- **Excitation–Relaxation (E-R)** – each neuron has an adaptive firing
  threshold. Firing raises it (fatigue / spike-frequency adaptation; by
  default linearly, halfway towards the firing's magnitude, with log, fixed
  and multiplicative growth selectable per layer), silence lets it decay. When it decays to ~0 the neuron fires
  spontaneously at a small amplitude, which re-excites the threshold; the
  trigger level, the amplitude and an optional random firing rate are
  configurable per layer (`LayerSpec::spontaneous`).
- **Habituation** – if a neuron's weighted input stays unchanged for
  `habituation_steps` consecutive steps, the input is treated as zero until
  the signal changes again. The streak length, a tolerance for "unchanged"
  and a gradual fade instead of the cut are configurable per layer
  (`LayerSpec::habituationRule`); the fade starts after `fadeAfter`
  identical steps (default 2).
- **Normalised weighted sum** (optional, per layer: `LayerSpec::normalize`)
  – each neuron's sum is divided by the length of its weight vector, so
  only the weights' direction matters; the lengths are cached and
  recomputed only after the weights change.
- **Reward-modulated learning** – `applyReward(reward, learningRate)` moves
  the weights of recently active neurons toward the reward's sign.
- **Learning rules per layer** – besides that sign rule, each layer can learn
  with a graded trace rule, feedback alignment (per-neuron credit for hidden
  layers from an error vector, `applyError`), node perturbation, or the
  unsupervised Oja and BCM rules, with an optional learned bias and weight
  decay; one network can mix them (see [doc/learning.md](doc/learning.md)).
- **Networks** – `network` owns a graph of layers with forward and feedback
  edges, input sensors and output layers, runs them in a well-defined order,
  and can freeze individual layers.
- **Vision layers** – `Retina` reads camera images (every pixel, or samples
  on a tight spiral out from the centre), `Conv2D` (shared kernels),
  `LocallyConnected2D` (own weights per position) and `Pool2D`. A 320 × 200
  pipeline with about 2 million neurons steps in about 18 ms. Fixed filter
  banks (Gabor, centre-surround, Gaussian) turn a Conv2D into a frozen
  feature detector (see [doc/spatial.md](doc/spatial.md)).
- **Audio layers** – `Cochlea` reads sound a hop of samples per tick from
  one or more microphones and splits it (Hann window, FFT) into mel or
  linear frequency bands, one adapting neuron per band and microphone;
  `History` keeps the last ticks side by side, so a cochlea's bands become
  a spectrogram the vision layers read like an image (see
  [doc/audio.md](doc/audio.md)).
- **Several senses in one network** – named input sources
  (`addInputs(eye, shape, "camera")`, `setInputs("camera", image)`) can each
  feed several layers; `Resize2D` brings maps of different sizes (a camera
  image, a spectrogram) to one grid so a convolution reads them together;
  `Disparity` matches a left and a right view at a range of shifts for
  stereo depth (see [doc/multimodal.md](doc/multimodal.md)).
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

Both adaptation mechanisms can be toggled independently per layer. Neurons
without E-R can instead have a fixed firing threshold (`LayerSpec::gate`) or
be ReLUs (`LayerSpec::rectify`).

## Requirements

- CMake ≥ 3.20
- A C++20 compiler (GCC, Clang or MSVC)
- Internet access on first configure — GoogleTest v1.15.2 is fetched via
  `FetchContent`
- Optional: OpenMP, for the parallel forward pass and learning (detected automatically)

## Building and testing

One command builds everything, runs every test and installs the library:

```bash
./build.sh                   # library, unit tests, nntest; runs ctest; installs into ./install
./build.sh --python          # ... and the Python package, with its tests and quick experiments
./build.sh --vizdoom         # ... and ViZDoom from PyPI for the Doom experiments (implies --python)
./build.sh --goe             # ... and Gardens of Eris, cloned from its repository, for goe_rl (implies --python)
./build.sh --all             # everything: --python --vizdoom --goe
./build.sh --help            # --prefix DIR, --debug, --native, --no-tests, --no-install, --clean, -j N
```

It configures `build/` (Release), builds, runs `ctest` (the unit tests, the
quick experiments' checks, a program built against the installed package,
and with `--python` the pytest suite) and installs into `./install`
(`--prefix` to change). Afterwards:

| To | Use |
|----|-----|
| run the unit tests again | `./build/exrelaxer_tests` or `ctest --test-dir build` |
| run experiments | `./build/NNtesting/nntest list` (also `install/bin/nntest`) |
| run Python experiments (`--python`) | `PYTHONPATH=build/EXrelaxer.py/package NNtesting/nntest.py list` |
| watch the best Doom agent play (`--vizdoom`) | `PYTHONPATH=build/EXrelaxer.py/package python3 NNtesting/experiments/doom_rl/watch.py` writes a video (`replay.mp4`), subtitles and a replay page (`replay.html`); `--live` shows the game window ([doom_rl README](NNtesting/experiments/doom_rl/README.md#watching-an-agent-watchpy)) |
| use the library in another C++ program | `find_package(exrelaxer)` with `-DCMAKE_PREFIX_PATH=install` (see [below](#using-the-library)) |

Arguments after `--` go to CMake, e.g.
`./build.sh -- -DFETCHCONTENT_SOURCE_DIR_GOOGLETEST=/path/to/googletest`
to build offline. On Windows run it from Git Bash, or use the plain CMake
steps it runs:

```bash
cmake -S . -B build          # Release by default; add -DCMAKE_BUILD_TYPE=Debug to debug
cmake --build build -j
cd build && ctest --output-on-failure
cmake --install build --prefix install
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

| Option | Default | Builds |
|--------|---------|--------|
| `EXRELAXER_BUILD_TESTS` | on when top-level | the GoogleTest unit tests (fetches GoogleTest) |
| `EXRELAXER_BUILD_NNTESTING` | on when top-level | the `nntest` benchmark harness |
| `EXRELAXER_BUILD_PYTHON` | off | the Python extension, tested by CTest as `python_tests` |
| `EXRELAXER_INSTALL` | on when top-level | install rules: library, headers, CMake package, `nntest` |

A project that adds exrelaxer with `add_subdirectory` gets only the library.

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

## Python

`EXrelaxer.py/` is the Python package `exrelaxer` (nanobind bindings): build
networks, run them and reward them from Python, with numpy arrays in and
out. It also holds the Python experiment runner (`NNtesting/nntest.py`,
one folder or file per experiment) and a loader for the datasets `fetch.py`
prepares. See [EXrelaxer.py/README.md](EXrelaxer.py/README.md).

```bash
pip install ./EXrelaxer.py
NNtesting/datasets/fetch.py get mnist
NNtesting/nntest.py run mnist_gabor       # MNIST from Python: 0.90 accuracy
```

## Using the library

Either install it (`./build.sh --prefix DIR`) and find the package:

```cmake
find_package(exrelaxer 1.0 REQUIRED)       # configure with -DCMAKE_PREFIX_PATH=DIR
target_link_libraries(my_app PRIVATE exrelaxer::core)
```

or build it inside your project:

```cmake
add_subdirectory(exrelaxer)                # just the library, no tests
target_link_libraries(my_app PRIVATE exrelaxer::core)
```

Either way headers are included as `"network.hpp"`, `"layers/cochlea.hpp"`,
... (installed, they are in `DIR/include/exrelaxer`), and OpenMP comes along
when the library was built with it. [examples/consumer](examples/consumer)
is a complete program; the CTest test `installed_package_example` installs
the build and compiles it against the package. [examples/doom_agent](examples/doom_agent)
loads an agent evolved in Python ([doc/doom_guide.md](doc/doom_guide.md#c)) and
runs one game step.

## Quick start

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
| `habituation_steps`         | `100`    | Identical steps before the input is suppressed (default of `Habituation::steps`) |
| `recovery_factor`           | `0.9`    | Default per-step threshold decay while not firing |
| `baseline_threshold`        | `0.2`    | Resting E-R threshold and learning-eligibility boundary |
| `spontaneous_min_amplitude` | `0.1`    | Default amplitude of spontaneous firing (0.01 before 2026-09-28) |
| `max_weight`                | `10.0`   | Learning clamps each weight to ±this |
| `max_output`                | `10.0`   | Each weighted sum is clamped to ±this |
| `default_learning_gain`     | `2.0`    | Default per-neuron learning gain (multiplies weight updates) |

How E-R thresholds grow on firing is chosen per layer
(`LayerSpec::thresholdGrowth`, `ThresholdGrowth`): linear (the default,
`thr + 0.5 × (|v| − thr)`), log (the original, `thr + alpha × ln(|v| / thr)`),
fixed or multiplicative; after a firing the threshold is at least
2 × `baseline_threshold`. The per-neuron growth rate `alpha` (default
`default_alpha`, 2.0), a `neuron` constructor argument, affects only the
log rule. Each neuron also has its own recovery factor,
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
learning gain is `default_learning_gain` (2.0) and the default log-rule
`alpha` is `default_alpha` (2.0). A small learning-speed jitter sometimes helps
learned hidden layers with E-R, but the effect is not robust to the other
E-R constants.

## Project layout

```
core/
  neuron.hpp/.cpp                single neuron's dynamics: E-R, habituation, eligibility, serialization
  learning.hpp/.cpp              LearningRule: the per-layer learning rules and their update
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
  layers/audio.hpp               CochleaSpec, frequency scales, compression
  layers/cochlea.hpp/.cpp        sound input: FFT into frequency bands, one neuron per band
  layers/history.hpp/.cpp        the last ticks of its sources side by side (spectrograms)
  layers/resize2d.hpp/.cpp       resampling to a fixed height x width (nearest, bilinear, area)
  layers/disparity.hpp/.cpp      binocular disparity: left and right views matched at a range of shifts
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
  learning.cpp                   each learning rule's update, options, applyError, serialization
  kernels.cpp                    SIMD kernels and dense layers bit-identical to scalar references
  regressions.cpp                one test per fixed bug; DISABLED_ tests for open ones
  spatial.cpp                    retina, Conv2D, LocallyConnected2D, Pool2D against scalar references
  snake.cpp                      the headless snake game: rules, state vector, reference values shared with Python
  audio.cpp                      Cochlea against a double-precision DFT, History, save/load of a hearing network
  multimodal.cpp                 named input sources, multichannel cochlea, Resize2D, Disparity, save/load
  filters.cpp                    filter banks, and bar-orientation learning with frozen Gabor features
  er_scales.hpp                  test inputs and timings relative to the E-R constants
build.sh                         one command: build everything, run the tests, install
cmake/                           package config template, the installed-package test
examples/consumer/               a separate program using the installed library (find_package)
examples/doom_agent/             loads an evolved Doom agent (best.exr) in C++ and runs one game step
third_party/                     Gardens of Eris, cloned by ./build.sh --goe (not in git)
EXrelaxer.py/                    Python package exrelaxer: nanobind bindings, experiment runner,
                                 dataset loader, pytest suite
NNtesting/                       benchmark harness nntest (see NNtesting/README.md)
  harness/                       runner: parameters, trials, statistics, result files
  tasks/                         task code shared with the unit tests (pattern_benchmark.hpp, bars.hpp, snake.hpp)
                                 and the experiments (activity.hpp, nonlinearity.hpp, er_options.hpp)
  experiments/                   one per experiment: NAME.cpp, NAME/experiment.py or NAME.py
    _shared/                     Python helpers shared by experiments (wiring.py: what every weight reads)
  nntest.py                      runner for the Python experiments
  datasets/fetch.py              downloads public datasets (MNIST, CIFAR-10, Kaggle, ...) into one format
  tools/compare.py               tables and before/after comparisons of result files
  tools/spiral.py                spiral sampling of images, bit-identical to the Spiral retina
  tools/capacity.py              minimum architectures and capacity curves from nl_static results
```

Tests express their E-R-sensitive inputs and timings relative to these
constants (`tests/er_scales.hpp`): a "weak" stimulus is 5 × `baseline_threshold`,
and waits for spontaneous firing are computed from `recovery_factor`. The
suite passes for `baseline_threshold` from 0.05 to 1.0.

## Known limitations

- **The default rule has no bias and uses only input signs.** With the
  default sign rule neurons have no bias (use an input held at 1.0, or a
  rule with `withBias()`) and a 0 input never teaches anything. The other
  [learning rules](doc/learning.md) have graded updates and an optional
  learned bias.
- **Learning is weak beyond the last layer with a global reward.** With
  `applyReward`, every eligible neuron gets the same reward: there is no
  per-neuron credit assignment. Tasks work well when frozen or hand-wired
  layers provide the features and memory and only a readout learns.
  Gapped-pattern detection (`A,{0..1},B,{0..1},C` among decoys) reaches
  ~0.96 valid-vs-decoy accuracy that way, and ~0.81–0.94 with a frozen
  *random* reservoir as memory. Fully learned feedback networks stay at
  chance. E-R neurons carry some memory on their own (0.757 with no
  connections between them, vs 0.50 without E-R). Feedback alignment
  (`applyError`) now trains hidden layers from an error vector (a
  bottleneck task: error 0.49 → 7 × 10⁻⁷), but on snake no learned hidden
  layer beats a frozen random one yet ([research log §11](doc/research.md#11-learning-rules)).
- **E-R makes learning steps very large.** Eligibility is
  `threshold / baseline_threshold − 1` and a firing lifts the threshold well
  above baseline, so E-R neurons take big steps and hit the weight clamp
  quickly. The sign rule has no weight decay (every other rule, and the
  sign rule with `withDecay()`, can have it).
- **Bounded, not squashed.** Weights and weighted sums are clamped at ±10;
  without the output clamp, networks with feedback loops diverge.
- **Shared random streams.** Initial weights depend on everything created
  earlier in the process unless `exr::reseed` is called first.
- **Serialization is not portable** (raw binary, native endianness), and a
  saved network can only be loaded by a program that registers all its
  layer types. Files older than format 6 load as weights only.

## Changelog

### 2026-10-01: Gardens of Eris: designed networks, growing sight, build

- `./build.sh --goe` clones [Gardens of Eris](https://github.com/arielkonopka/Gardens-of-Eris)
  into `third_party/` (or updates the clone) and installs its Python
  package `goe`; `--all` builds everything (`--python --vizdoom --goe`).
- `goe_rl`: the eye covers the player's growing sight (`radius` `auto`,
  an optional `seen` channel); a frozen echo-state reservoir, `skip`, a
  feedback ladder with rung layers of their own neuron model, and
  `readout_from all`; `es.py` takes model files and grows the network
  (`grow_to`), carrying weights over by the input they read
  (`NNtesting/experiments/_shared/wiring.py`).
- Three designed networks in `goe_rl/models/` (frozen reservoir; growing
  E-R layers; three E-R layers with a relu feedback ladder), and the
  record of every GoE experiment, [doc/goe.md](doc/goe.md) (research log
  §28). G1: untrained, every network stands still; a random player
  scores 7–9 times more.

### 2026-10-01: Doom tooling, video replays, feedback ladder

- `./build.sh --vizdoom` installs ViZDoom (the official release, from
  PyPI) with the Python build requirements and builds the Python package;
  it warns when `libopenal1` (the game's sound) or `ffmpeg` (replays) is
  missing.
- `doom_rl/watch.py` writes a replay as a video (`replay.mp4`, H.264, one
  frame per game step; `--video mkv` or `mpg` for MPEG-2), subtitles
  (`replay.srt`: action, health, ammo, kills, reward) and a small page
  (`replay.html`) that plays the video with the network's readouts and
  firing beside it. Frames stream to `ffmpeg`, so long games no longer
  make one huge HTML file.
- The **feedback ladder** in `doom_rl` (`feedback_first`): every hidden
  layer above h1 comes back to h1 as input, one tick late, so the rungs
  remember over different spans. The rungs are neurons in h1 or layers of
  their own with another neuron model (`feedback_first_model`, e.g. relu
  with habituation). `es.py --grow-to` grows the ladder: each new layer
  brings its rung, and the weights carry over by the input they read.
- `es.py` fix: the workers now close their games; before, every run (and
  every growth) left ViZDoom engines running and `es.py` hung at exit.
- [doc/doom_guide.md](doc/doom_guide.md): how to install, watch, retest
  and train Doom agents with other parameters, and use them from C++;
  `examples/doom_agent` loads a saved agent in C++ and runs one game step.

### 2026-09-28: new defaults

- `default_alpha` 2.0 (was 1.2; log growth rule only; saved per neuron).
- `spontaneous_min_amplitude` 0.1 (was 0.01; files older than format 15
  load 0.01, `legacy_spontaneous_amplitude`).
- Normalised weighted sums in every layer made by `LayerSpec::Dense`,
  `Conv2D` or `LocallyConnected2D` (Python `LayerSpec.dense`, ...); a bare
  `LayerSpec` stays raw; files older than format 16 load raw sums. The
  experiments' `normalize` and `spontaneous_amplitude` parameters default
  to the new values; pass `normalize=false` or `spontaneous_amplitude=0.01`
  to reproduce older results (output layers are now normalised too).
- Finding ([research log §23](doc/research.md#23-new-defaults-and-spontaneous-firing-after-silence)):
  a reconnected network goes blind after a long silence only under the log
  growth rule; the spontaneous amplitude does not change that.

### 2026-09-28: Doom by evolution

- `doom_rl/es.py`: evolves the action readouts of the frozen doom_rl
  network (OpenAI-ES scored by the game reward; resumable; `--init` warm
  start). `doom_rl/search.py` and `retest.py` search topology and settings.
- Findings ([research log §20](doc/research.md#20-dynamic-ladder-time-varying-input-and-doom)):
  the searched readout-rule winners were seed luck (trained ≤ untrained).
  Evolution works: one E-R layer with fading habituation and no feedback
  is the best and most frugal agent on `defend_the_center` (validation
  reward −4.8 → +0.95); on MAP01 it learns to explore but not to finish.

### 2026-09-27: dynamic ladder and Doom

- `nntest run dyn_ladder`: motion direction, change detection, velocity
  and a closed-loop catch game, with relu, E-R, memoryless E-R or gate
  neurons and an optional window of past frames (the control §16 lacked).
  Only E-R beats the single-frame ceiling without a window, clearly only on
  change detection (0.98 vs 0.70); with one past frame ReLU solves all four
  and E-R on top of a window hurts.
- `doom` (Python, ViZDoom): `basic` and `predict_position` from 40 × 30
  pixels, imitating an oracle that reads object positions.
- `doom_rl` (Python, ViZDoom): Doom from reward, with the screen at the
  smallest settings and stereo sound through a two-ear cochlea; reward
  from hurt, death, kills, ammo, armor, items, keys, doors, level exit and
  idling.
- `NNtesting/tools/dyn_summary.py`; [doc/dynamic.md](doc/dynamic.md);
  [research log §20](doc/research.md#20-dynamic-ladder-time-varying-input-and-doom).
### 2026-09-27: per-layer E-R resting threshold

- **E-R resting threshold per layer** (`LayerSpec::restingThreshold`,
  `neuron_layer::setRestingThreshold`, `neuron::setRestingThreshold`,
  Python `LayerSpec.resting_threshold`): the threshold E-R relaxes to,
  its learning-eligibility boundary and half its floor after firing.
  Default 0.2 (`baseline_threshold`), as before. Network format 17 saves it.
- Experiments: `resting_threshold` (a value, or `auto`: scaled by the
  layer's mean 1/|w|, for normalised sums).
- Finding ([research log §19](doc/research.md#19-normalised-weighted-sum)):
  recalibrating the resting threshold changes nothing, because E-R's
  thresholds follow its sums. Normalised networks need a larger learning
  rate instead, and at 10× the rate they match the raw ones' best.

### 2026-09-27: normalised weighted sum

- **Normalised weighted sum** (`LayerSpec::normalize`,
  `neuron_layer::setNormalized`, Python `LayerSpec.normalize`): a layer
  divides each neuron's sum by the length of its weights. Off by default.
  Norms are cached and recomputed only after weights change. Network
  format 16 saves it.
- `nl_static`, `nl_temporal` and the activity experiments take
  `normalize=true` (hidden layers).
- Findings ([research log §19](doc/research.md#19-normalised-weighted-sum)):
  about half as many runs diverge at large learning rates, but the best
  results do not improve and E-R loses accuracy in networks with many
  inputs per neuron.

### 2026-09-27: early fading, spontaneous firing settings, rerun

- **Habituation fade starts after `fadeAfter` identical steps** (default 2,
  Python `Habituation(..., fade_after=2)`); the cut still happens after
  `steps`. The library's default habituation is unchanged (cut after 100).
- **Spontaneous firing is configurable** (`Spontaneous`,
  `LayerSpec::spontaneous`): the threshold level that triggers it, its
  amplitude and a random per-tick firing rate for silent neurons. A
  spontaneous firing no longer lowers a threshold.
- Network format 15 saves both; older files load as before (a saved fade
  starts at its `steps`).
- `nl_static` / `nl_temporal`: habituation options, spontaneous-firing
  options, and `pretrain_model` / `pretrain` to train without E-R first and
  switch to E-R.
- Result files now live in [`results/`](results/README.md) instead of
  outside the repository; the research log links to them.
- Every E-R experiment rerun with linear growth and three habituation
  variants (off, cut after 5, fade from the 2nd repeat): see [research log
  §17–18](doc/research.md#18-rerun-with-linear-growth-and-three-habituation-variants)
  and [`results/rerun/summary.md`](results/rerun/summary.md).

### 2026-09-27: temporal tasks, threshold growth rules

- **Threshold growth rule** (`ThresholdGrowth`, `LayerSpec::thresholdGrowth`,
  Python `ThresholdGrowth`): linear, log (the original), fixed or
  multiplicative growth on firing. Network format 14 saves it. Python now
  also exports `Habituation`.
- **Default changed: E-R thresholds now grow linearly**, halfway towards
  the firing's magnitude, instead of by `alpha × ln(ratio)`. Networks saved
  before (format ≤ 13) load with the log rule; `alpha` now affects only the
  log rule. Pass `growth=log` to the experiments to reproduce earlier
  results. The experiments' own checks all still pass; `er_economy` E-R
  accuracy 0.997 → 1.0.
- `nntest run nl_temporal`: delayed XOR, x(t) AND NOT x(t−3), parity, and
  sin(x(t)·x(t−2)), with a memoryless E-R control. Only E-R with state beats
  the no-memory ceiling; it solves delayed XOR (1×64 neurons, 8 with linear
  growth) but not longer lags or continuous history.
- `nl_static` / `nl_temporal`: `growth`, `growth_amount`, `learn_ticks=all`.
- See [research log §16](doc/research.md#16-temporal-tasks-and-the-threshold-growth-rule).

### 2026-09-27: E-R behaviour: learning curves, silence, state, habituation

- **Configurable habituation** (`Habituation`, `LayerSpec::habituationRule`,
  Python `Habituation`): the streak of identical ticks before it acts
  (default 100, as before), a relative tolerance for "identical" (default
  0), and fading by a factor per tick instead of cutting (default: cut).
  Network format 13 saves it. Defaults behave exactly as before.
- `nntest run er_silence`: inputs zeroed after training. All models go
  silent at once; E-R alone restarts, with spontaneous bursts every
  73–110 ticks from about tick 214 (recovery 0.9). Afterwards its thresholds
  jump and accuracy drops to 0.44–0.52 at recovery 0.97.
- `nntest run er_habituation`: long-held stimuli. Habituation after 5
  identical ticks cuts E-R's spikes 90× with onset accuracy 0.98, but the
  stimulus is gone by the end of the hold, and sensor flicker stops exact
  repeats unless a tolerance is set.
- `nl_static`: learning curves (`curve_<n>`, `early_stop=false`) and a
  state test (`state_probes`): with 7 extra ticks, 30–57 % of E-R's error
  comes from the state left by earlier samples.
- Fix: spontaneous E-R firings are drawn in ±0.01, not exactly ±0.01;
  the activity meters now detect them.
- See [research log §15](doc/research.md#15-how-e-r-behaves-learning-silence-state-habituation).

### 2026-09-27: nonlinearity substitution, milestone 1

- **ReLU neurons** (optional): `LayerSpec::rectify`, `neuron::setRectified`,
  `neuron_layer::setRectified`, Python `LayerSpec.rectify`. Only sums above
  the gate (0 by default) pass. Off by default, so the plain clamped neuron
  is unchanged. Network format 12 saves it.
- Feedback alignment: a silent gated or rectified neuron takes no step,
  like a silent E-R neuron.
- `nntest run nl_static` and `NNtesting/tools/capacity.py`: how large must
  a network be to reach test MSE 1e-3 on known static functions, with
  ReLU, E-R, fixed-threshold or clamped neurons? ReLU solves every task
  (5–129 neurons); E-R reaches none at any size. More ticks per sample
  lower E-R's error 2–4×. See [doc/nonlinearity.md](doc/nonlinearity.md)
  and [research log §14](doc/research.md#14-dynamic-nonlinearity-substitution-static-tasks).

### 2026-09-27: activity economy experiments

- **Fixed firing threshold** for neurons without E-R: `LayerSpec::gate`,
  `neuron::setGate`, `neuron_layer::setGate`. Output is 0 while
  |sum| ≤ gate; the threshold never adapts. Network format 11 saves it;
  `describe()` shows it in the E-R column. Python: `LayerSpec.gate`.
  Under feedback alignment a silent gated neuron does not learn (its
  surrogate derivative is 0, as for a silent E-R neuron).
- `nntest run er_economy`, `er_paths`, `er_fatigue`, `er_history`: does
  E-R use less activity, prefer cheaper paths or respond to its history,
  with no activity penalty? It is as sparse as a fixed threshold of the
  same sparsity (half of a linear network) and prefers no path, but shifts
  activity away from fatigued paths, changes paths under a constant input
  and answers the same input differently after different histories. See
  [doc/activity.md](doc/activity.md) and
  [research log §13](doc/research.md#13-activity-economy-and-path-selection).

### 2026-09-27: several senses and stereo vision

- **Named input sources**: `network::addInputs(target, count | shape, name)`,
  `setInputs(name, values)`, `connectInputs(name, layer)` (one camera or
  microphone feeding several layers), `inputSource(name)`, `inputs(name)`.
  `describe()` lists each source with its shape and the layers it feeds.
- **Several microphones**: `CochleaSpec::channels`; the cochlea's output is
  channels × bands × 1, each microphone analysed like a single one.
- **`Resize2D`** (`LayerType::Resize2D = 7`): every input channel resampled
  to a fixed height × width (nearest, bilinear or area), so maps of
  different sizes can be read together.
- **`Disparity`** (`LayerType::Disparity = 8`): binocular matching of a
  left and a right view at disparities `min..max` (correlation, absolute
  difference or normalized cross-correlation over a window).
- Network format 10 saves the new parameters, source names and shapes, and
  `connectInputs`; older files load as before.
- `nntest run stereo_depth`: where is the near square in a random-dot
  stereogram? 0.98 with the Disparity layer, 0.49–0.50 with one eye or both
  eyes without matching. `nntest run audiovisual`: objects that look and
  sound different; seeing and hearing (0.96) beat sight (0.79) or sound
  (0.95) alone, and readouts on sound that learn only from what sight says,
  without labels, name objects by sound 0.66 of the time (chance 0.25). See
  [research log §12](doc/research.md#12-several-senses-and-stereo-vision).
- Python: `add_inputs(..., name=)`, `set_inputs(name, values)`,
  `connect_inputs`, `input_values`, `input_sources`, `ResizeSpec`,
  `DisparitySpec`, `LayerSpec.resize2d` / `.disparity`,
  `CochleaSpec(channels=)`; `audio.frames` and `audio.read_wav(mono=False)`
  handle several channels.

### 2026-09-27: learning rules per layer

- `LearningRule` (`core/learning.hpp`): every layer of neurons chooses how
  it learns, through `LayerSpec::learningRule` or `network::setLearningRule`.
  The rules are sign (the default and unchanged), trace (graded
  three-factor with eligibility traces and a reward baseline), feedback
  alignment, node perturbation, Oja and BCM (unsupervised, with optional
  k-winners competition). Every rule can add a learned bias and weight
  decay. See [doc/learning.md](doc/learning.md).
- `network::applyError(errors, rate)`: output layers learn their own errors
  and hidden feedback-alignment layers a fixed random projection of them.
  `neuron_layer::applyModulators` gives each neuron its own reward.
- Network format 9 (neuron format 3) saves the rule and its state; loaded
  networks continue exactly. Older files load with the sign rule.
- With the default rule, results are bit-identical and speed is the same
  as before (within ±3 %).
- `nntest run snake_rules`: snake with a rule for the readouts and for the
  mix layer, reporting apples, steps and apples per 100 steps per game (see
  [research log §11](doc/research.md#11-learning-rules)).
  `dense_throughput` takes `--set rule=...` and times `applyError` too.
- Python: `LearningRule`, `LayerSpec.dense(..., learning_rule=...)`,
  `Network.set_learning_rule`, `apply_error`, `apply_modulators_to`, `bias`.

### 2026-09-27: snake

- `snake` (C++, game in `NNtesting/tasks/snake.hpp`) and `snake_py`
  (Python, `NNtesting/experiments/snake/`): a network learns to play snake
  without a screen. It sees 23 values: the 8 cells around the head, the
  apple's direction, 4 points it sees (forward, left, right, back) with their
  map positions and distances, and a bias. One readout per action learns
  from error-driven rewards. It eats 13.6 apples per game on a 10 × 10 field
  (0.06 untrained). Both versions play the same games and give the same
  numbers.
- Tests: `tests/snake.cpp` and `EXrelaxer.py/tests/test_snake.py` check the
  game rules and the same reference states in both languages.

### 2026-09-27: audio layers

- **`Cochlea`** (`LayerType::Cochlea = 5`): the entry point for sound. It
  reads `hop` samples per tick, keeps the last `window`, and each tick takes
  their power spectrum (Hann window, radix-2 FFT) and sums it into `bands`
  triangular mel or linear bands, compressed as `log(1 + gain · energy)`.
  One neuron per band, so habituation and E-R make a steady tone fade.
- **`History`** (`LayerType::History = 6`): the last `length` ticks of its
  sources side by side along the width; behind a cochlea, a bands × ticks
  spectrogram for the spatial layers.
- `LayerSpec` gains a `cochlea` field and the builders `Cochlea` and
  `History`; network format version 8 saves the cochlea spec (older files
  load as before).
- Python: `CochleaSpec`, `FrequencyScale`, `Compression`,
  `LayerSpec.cochlea` / `history`, `Network.cochlea_bands` /
  `cochlea_power`, and `exrelaxer.audio` (WAV reading, test tones and
  chirps, framing).
- New quick experiment `chirp_direction`: rising vs falling chirps through
  cochlea → history → frozen Gabor bank → pool → frozen mix → learned
  readout, accuracy 0.998 (0.56 without learning).
- Tests: `tests/audio.cpp` checks the spectrum and bands against a
  double-precision DFT, tones landing in their band, History ordering and
  growth, and a hearing network continuing bit-identically after save and
  load; `EXrelaxer.py/tests/test_audio.py` checks against numpy's FFT.
- Documentation: [doc/audio.md](doc/audio.md).

### 2026-09-27: one build command, installable package

- `./build.sh` builds the library, unit tests and `nntest` (and with
  `--python` the Python package), runs every test and installs the library.
- `cmake --install` installs `libexrelaxer_core`, its headers
  (`include/exrelaxer`), a CMake package (`find_package(exrelaxer)`, target
  `exrelaxer::core`) and `nntest`; the option `EXRELAXER_INSTALL` (on when
  top-level) controls it. `exrelaxer::core` also works after
  `add_subdirectory`. GoogleTest is no longer installed along with it.
- `examples/consumer`: a separate program built against the installed
  package, as the CTest test `installed_package_example`.

### 2026-09-27: Python package, Python experiments, build options

- `EXrelaxer.py/`: the Python package `exrelaxer` (nanobind). It binds
  networks, every layer type, jitter, filter banks, per-neuron state, weights
  and kernels, and save/load, with numpy in and out. `Network.run()` runs
  many ticks (with rewards) in one call. It installs with
  `pip install ./EXrelaxer.py`. Its pytest suite checks `run()` bit for bit
  against a Python step loop, save/load continuation, and bar-orientation
  learning from Python.
- New CMake options `EXRELAXER_BUILD_TESTS` and `EXRELAXER_BUILD_NNTESTING`
  (on only when exrelaxer is the top-level project) and
  `EXRELAXER_BUILD_PYTHON`. The core library is position-independent.
- Python experiments: `NNtesting/nntest.py` (`exrelaxer.harness`) runs
  experiments that are a folder (`NAME/experiment.py` plus their own files)
  or a single `NAME.py` in `NNtesting/experiments/`. It has the same
  commands, seeding, statistics, checks and JSON Lines as `nntest`.
  `exrelaxer.datasets.load` reads `fetch.py`'s output.
  `Network.apply_reward_to(layer, ...)` gives each readout its own reward.
- New experiments: `bar_orientation_py` (single file; 0.9875, as the C++
  one) and `mnist_gabor` (folder; MNIST at 0.90 with frozen Gabor features,
  a frozen random mix and one-vs-rest readouts).
- `fetch.py --data DIR` also works after the command.
- Docs: unconnected E-R neurons 0.757 (was still 0.74 in Known limitations);
  `doc/spatial.md` in the documentation list.

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
