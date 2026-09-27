# NNtesting: the benchmark harness

`nntest` runs **experiments**: a task on a kind of network, with parameters
for its architecture and settings. It measures how well networks learn and
how fast they run, across architectures, settings, tasks, machines and
commits. Unit tests (GoogleTest, in `tests/`) check that the code is correct;
experiments measure what the networks do.

```
NNtesting/
  harness/       the runner: parameters, trials, statistics, result files
  nntest.py      the same runner for Python experiments (exrelaxer.harness)
  tasks/         task code shared by experiments and unit tests
  experiments/   one file per experiment, found automatically: NAME.cpp (C++),
                 NAME/experiment.py or NAME.py (Python)
  datasets/      fetch.py: downloads public datasets into one format (see datasets/README.md)
  tools/         compare.py: tables and comparisons of result files
                 capacity.py: minimum architectures and capacity curves (nl_static)
                 spiral.py: spiral sampling of images, exactly as the Spiral retina
  data/          downloaded datasets (not in git)
```

Python tools need numpy and Pillow: `pip install -r NNtesting/requirements.txt`.

## Running

`nntest` is built with the rest of the project (option
`EXRELAXER_BUILD_NNTESTING`, on by default):

```bash
./build.sh                                        # from the repository root: builds and tests everything
./build/NNtesting/nntest list                     # every experiment
./build/NNtesting/nntest describe bar_orientation # its parameters, defaults, checks
./build/NNtesting/nntest run bar_orientation
./build/NNtesting/nntest run tag:performance --out results.jsonl
./build/NNtesting/nntest run gapped_pattern --set topology=window_readout,reservoir --set er=false,true
```

| Option | Meaning |
|--------|---------|
| `--trials N` | trials per parameter combination (default: the experiment's) |
| `--seed S` | trial `i` uses seed `S + i` (default 0) |
| `--set NAME=V1,V2,...` | parameter values; several `--set` run every combination |
| `--threads N` | OpenMP threads (default: all) |
| `--out FILE` | append every trial and summary to `FILE` (JSON Lines) |
| `--verbose` | show the experiments' diagnostics |

Selectors: an experiment name, a pattern with `*` (`'*_throughput'`),
`tag:TAG`, or `all`.

Every trial is seeded: the runner calls `exr::reseed(seed)` before it, so a
trial gives the same result on every run (bar orientation reproduces its
unit test exactly). For each parameter combination the runner prints the
mean, standard error, standard deviation, minimum and maximum of every
metric the experiment recorded.

**Checks**: an experiment can state expected results (e.g. accuracy ≥ 0.95).
They apply when it runs with its default parameters, since changed
parameters are meant to change results. `nntest` exits with 1 when a check
fails or a trial throws, and 2 for usage errors. The experiments tagged
`quick` also run as the CTest test `nntest_quick`, next to the unit tests.

## Result files

`--out` appends JSON Lines, one object per line:

- `"type": "trial"`: experiment, parameters, trial index, seed, seconds,
  every metric, and the error message if it threw;
- `"type": "summary"`: per parameter combination, every metric's `n`,
  `mean`, `sd`, `stderr`, `min`, `max`, and the checks with pass/fail.

Both carry `env`: run id, time (UTC), host, CPU, OpenMP threads, compiler,
build type, `native` (built with `EXRELAXER_NATIVE`), and the git commit
with a `dirty` flag. The commit is taken at build time, so a result always
names the code that produced it.

## Comparing

```bash
NNtesting/tools/compare.py results.jsonl                        # a table per experiment
NNtesting/tools/compare.py --baseline before.jsonl after.jsonl  # what changed
```

The table has a row per parameter combination, with only the parameters
that vary. `--baseline` matches the same combinations in both files and
shows every metric's change with a Welch t statistic, marking `|t| >= 2`
(set with `--t`). Use it for a code change (run before and after), two
machines, or thread counts:

```
experiment        params  metric          baseline  new     change   t
dense_throughput  n=256   step_ms         0.0137    0.02496 +82.2%   +5.71  *
dense_throughput  n=1000  step_reward_ms  0.3547    0.08427 -76.2%   -143.78 *
```

(1 thread vs 20: the 1000 × 1000 layer gains from the threads, the 256 × 256
one loses; see Findings.)

## Writing an experiment

Add a `.cpp` file to `experiments/` and re-run cmake. The file registers one
or more experiments with a static `nnt::Register` object:

```cpp
#include "experiment.hpp"
#include "network.hpp"

namespace {

nnt::Register experiment({
    .name = "my_task",                              // unique
    .description = "what it measures, in one line",
    .tags = {"learning"},                           // "quick": also runs in ctest
    .params = {
        {"hidden", "32", "hidden neurons"},        // name, default, help
        {"er", "false", "E-R in the hidden layer"},
    },
    .trials = 10,
    .expect = {{.metric = "accuracy", .min = 0.9}}, // checked with default parameters
    .run = [](nnt::Trial& t) {
        const auto hidden = t.params().getInt("hidden");
        const bool er = t.params().getBool("er");
        // exr::reseed(t.seed()) has been called: build, train, test...
        t.record("accuracy", accuracy);
    },
});

} // namespace
```

- **Parameters** are text with typed getters (`getInt`, `getDouble`,
  `getBool`, `getString`); a bad value throws, which fails the trial with a
  message.
- **Metrics** are any numbers the trial records; record the untrained or
  no-learning control too (e.g. `accuracy_control`) and their difference, so
  a run shows what learning added.
- **Timing**: `nnt::Trial::bestMs(repeats, iterations, f)` gives
  milliseconds per call, best of several repeats.
- **Task code** used by several experiments (and unit tests) goes to
  `tasks/`.

## Python experiments

Experiments can also be written in Python, against the `exrelaxer` package
(`pip install ./EXrelaxer.py`, see [EXrelaxer.py/README.md](../EXrelaxer.py/README.md)).
An experiment is a **folder** `experiments/NAME/` with an `experiment.py`,
next to whatever else it needs (helper modules it imports by name, notes,
configs). A small one can be a **single file** `experiments/NAME.py`.
`NNtesting/nntest.py` runs them with the same commands, options, seeding,
statistics, checks and exit codes as `nntest`. It also writes the same
JSON Lines, so `compare.py` compares C++ and Python results alike.

```bash
NNtesting/nntest.py list
NNtesting/nntest.py describe mnist_gabor
NNtesting/nntest.py run mnist_gabor --set mix=0,512 --trials 3 --out results.jsonl
NNtesting/nntest.py --data /big/disk run mnist_gabor   # datasets elsewhere (or EXR_DATA)
```

```python
import exrelaxer as exr
from exrelaxer import harness as nnt

@nnt.experiment(
    description="what it measures, in one line",
    tags=["learning"],                                 # "quick": also runs in ctest
    params={"hidden": (32, "hidden neurons"),          # name: (default, help); --set values
            "er": (False, "E-R in the hidden layer")},  # are parsed as the default's type
    trials=10,
    expect={"accuracy": (0.9, None)},                  # metric: (min, max), checked with defaults
)
def run(t):                                            # exr.reseed(t.seed) has been called
    data = t.dataset("mnist", "train")                 # prepared by datasets/fetch.py
    hidden = t.params["hidden"]
    ...                                                # t.rng: numpy generator seeded with t.seed
    t.record("accuracy", accuracy)
```

The name defaults to the folder or file name and must not clash with a C++
experiment. With `EXRELAXER_BUILD_PYTHON=ON`, the Python experiments tagged
`quick` run as the CTest test `nntest_py_quick`.

## Experiments

| Experiment | Tags | Measures |
|------------|------|----------|
| `bar_orientation` | vision, learning, quick | frozen Gabor bank + frozen random mix + learned readout, vertical vs horizontal bars; accuracy 0.9875 (0.523 without learning) |
| `gapped_pattern` | temporal, learning | A..B..C with gaps in a random stream, decoys; `topology` = `window_readout`, `window_mix` or `reservoir`; `er`, reward mode, sizes |
| `dense_throughput` | performance, quick | ms per step, per step + reward and per step + error, `n × n` dense layer; `rule` = any learning rule |
| `chirp_direction` | audio, learning, quick | cochlea → history (spectrogram) → frozen Gabor bank → pool → frozen random mix → learned readout, rising vs falling chirps; accuracy 0.998 (0.56 without learning) |
| `snake` | control, learning, quick | snake without a screen: 23-value state (neighbourhood, apple direction, 4 seen points and their distances) → frozen mix → one learned readout per action; 13.6 apples per game on 10 × 10 (0.06 untrained). See [experiments/snake/README.md](experiments/snake/README.md) |
| `stereo_depth` | vision, stereo, learning, quick | random-dot stereograms: is the near square in the left or right half? Disparity layer → average pool → learned readout 0.98; one eye 0.49, both eyes without matching 0.50 |
| `audiovisual` | audio, vision, multimodal, learning | objects with their own look (bar orientation) and sound (pitch), both noisy: readouts on sight 0.79, sound 0.95, both 0.96; readouts on sound taught only by sight's choices, no labels: 0.66 by sound alone (0.29 without) |
| `er_economy` | er, activity, learning, quick | E-R vs a fixed threshold of the same sparsity vs linear neurons, no activity penalty: accuracy, active fraction, spikes per decision, active runs, activity before and after training. E-R and the fixed threshold both halve activity; only E-R learns well with FA (0.995 vs 0.76). See [doc/activity.md](../doc/activity.md) |
| `er_paths` | er, activity, paths, learning | three paths (12, 20, 35 neurons): drive per neuron per path (≈ 1.0, no preference), switching of the leading path over 3000 samples and under one held input |
| `er_fatigue` | er, activity, paths | fatigue one path or all, then probe: the path's share drops to 0.05–0.44, others take over, recovery ≈ 10 ticks (recovery 0.9) or ≈ 30 (0.97) |
| `er_history` | er, activity, state | the same inputs after rest, ordinary activity or stimulation: change of pattern, evidence, decision, spikes and latency (always 0 for the controls) |
| `er_silence` | er, activity, state | zero the inputs of a trained network, optionally with recurrence: does activity continue (spontaneous E-R firing), and how does it answer afterwards? |
| `er_habituation` | er, activity, habituation | hold each stimulus for up to 500 ticks: spikes saved by habituation vs accuracy at onset, over the sample and at its end |
| `nl_temporal` | er, nonlinearity, temporal, learning | temporal tasks (delayed XOR, x(t) AND NOT x(t−3), parity of n, sin(x(t)·x(t−2))) with relu, E-R, memoryless E-R (state reset every step), fixed-threshold or clamped neurons: accuracy against the best possible without memory, size, activity, cost. See [doc/nonlinearity.md](../doc/nonlinearity.md) |
| `nl_static` | er, nonlinearity, learning | known static functions (l0 x1+x2 … l4 sum of K sines) with relu, E-R, fixed-threshold or plain clamped hidden neurons on a depth × width grid: test MSE, success at MSE ≤ 1e-3, size, activity, cost. Analyse with `tools/capacity.py`. See [doc/nonlinearity.md](../doc/nonlinearity.md) |
| `snake_rules` | control, learning, rules | `snake` with a learning rule per layer: `readout` = sign, trace, fa, perturbation; `mix` = frozen or any rule (also oja, bcm); apples, steps and apples per 100 steps per game. The defaults play exactly the games of `snake` |
| `snake_py` (Python folder) | control, learning, quick | `snake` written in Python, playing the same games with the same results |
| `vision_throughput` | performance, vision | ms per step, per step + reward and per layer of a 320 × 200 camera pipeline |
| `bar_orientation_py` (Python file) | vision, learning, quick | `bar_orientation` written in Python; accuracy 0.9875 (0.514 without learning) |
| `mnist_gabor` (Python folder) | vision, learning, dataset | MNIST (or `--set dataset=fashion_mnist`): frozen Gabor bank + pool + frozen random mix + one learned readout per class; accuracy 0.90 on 10 000 training images (0.08 without learning) |

## Datasets

`datasets/fetch.py` downloads public datasets (MNIST, Fashion-MNIST,
CIFAR-10, the Kaggle casting-inspection set, audio sets, or any Kaggle
dataset) and prepares them as `SPLIT_images.npy` (uint8, N × C × H × W) +
`SPLIT_labels.npy` + `meta.json` in `NNtesting/data/NAME/`. See
[datasets/README.md](datasets/README.md).

## Spiral sampling tool

`tools/spiral.py` samples images like the Spiral retina: sample 0 at the
image centre, then outwards along an Archimedean spiral, turns and samples
`--spacing` pixels apart, each sample interpolated from its 4 nearest
pixels. Its values are bit-identical to the C++ retina's sums; the CTest
test `spiral_matches_retina` checks this on 7 image sizes, spacings and
radii (through the small `retina_dump` program).

```bash
spiral.py info --width 320 --height 200                  # 31099 samples, radius 99.5, 100 turns
spiral.py points --width 28 --height 28 points.csv       # k, x, y, radius, theta per sample
spiral.py image photo.png --grey --out samples.npy       # C x samples (.npy or .csv)
spiral.py image photo.png --polar polar.png              # rows = turns (distance), columns = angle
spiral.py image photo.png --reconstruct back.png         # the image rebuilt from the samples
spiral.py image photo.png --overlay path.png             # the sample path, red (centre) to blue
spiral.py dataset data/mnist/train_images.npy train_spiral.npy   # N x C x H x W -> N x C x 1 x samples
```

Values are scaled to 0…1 (`--raw` keeps 0…255); `--radius R` limits the
spiral. The dataset command converts 60 000 MNIST images in about a second.
In a network, the retina does the same sampling itself
(`LayerSpec::Retina({shape, Sampling::Spiral})`); the tool is for
inspection, and for using spiral data elsewhere.

## Findings so far

- **gapped_pattern** (10 trials, after-C accuracy): window + readout 0.984,
  with E-R in the readout 0.908; window + frozen mix 0.994, with E-R 0.762;
  reservoir 40 + 60 with 10 000 training ticks 0.539, with E-R in the
  reservoir 0.716 (40 000 ticks and a 1-neuron readout: 0.798, as the unit
  test).
- **Threads on small layers**: a 256 × 256 dense step takes 25 µs with the
  default 20 threads against 14 µs on one: the 65 k multiply-adds are split
  over 2 threads while the rest of the pool spins. Larger layers gain (1000 ×
  1000 step + reward: 0.35 ms → 0.08 ms). The threshold in
  `kernels::threadsFor` needs another look.
