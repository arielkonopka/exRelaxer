# EXrelaxer.py: Python bindings

The `exrelaxer` Python package: the C++ library's networks, layers and
filter banks, bound with [nanobind](https://github.com/wjakob/nanobind).
Networks run at C++ speed. Python builds them, feeds them, rewards them and
reads their state, with numpy arrays in and out.

## Installing

From a checkout of the repository (the package builds the C++ library in the
parent directory, so it installs from the source tree, not from an sdist):

```bash
pip install ./EXrelaxer.py            # or: pip install -e ./EXrelaxer.py
pip install "./EXrelaxer.py[test]"    # with pytest
```

This needs a C++20 compiler and CMake ≥ 3.20. pip fetches the build tools
(scikit-build-core, nanobind) itself. OpenMP is used when it is found.

To build it in the main CMake tree instead, next to the unit tests, pass
`-DEXRELAXER_BUILD_PYTHON=ON` (nanobind must be installed in the Python
CMake finds). The package is then staged in
`build/EXrelaxer.py/package/`, and `ctest` runs its tests as `python_tests`.

## Example

```python
import numpy as np
import exrelaxer as exr

exr.reseed(1)                                   # reproducible weights
net = exr.Network()
inp = net.add_layer("in",  exr.LayerSpec.dense(16))
hid = net.add_layer("hid", exr.LayerSpec.dense(32))
out = net.add_layer("out", exr.LayerSpec.dense(1, er=False))
net.add_inputs(inp, 2)
net.connect(inp, hid)
net.connect(hid, out)
net.add_feedback(out, hid, 8)
net.add_output(out)

for t in range(1000):
    net.set_inputs([0.5, -0.2])
    net.step()
    y = net.outputs()                           # numpy float32
    if y[0] <= 0:                               # error-driven reward
        net.apply_reward(1.0, 0.005)

# Many ticks in one call, without Python in the loop:
ys = net.run(np.zeros((1000, 2), np.float32))                 # 1000 x 1 outputs
x = np.random.default_rng(0).uniform(-1, 1, (1000, 2)).astype(np.float32)
r = np.where(x[:, 0] > 0, 1.0, -1.0).astype(np.float32)
ys = net.run(x, rewards=r, learning_rate=0.005)   # apply_reward(r[t]) after each tick

net.save("net.bin")
copy = exr.Network.load("net.bin")              # continues bit for bit
```

Vision layers and filter banks:

```python
image = exr.Shape(1, 28, 28)
eye  = net.add_layer("eye", exr.LayerSpec.retina(exr.RetinaSpec(image, exr.Sampling.Spiral), False, False))
v1   = net.add_layer("v1",  exr.LayerSpec.conv2d(8, exr.Window2D.square(7, 1, 3), False, False))
pool = net.add_layer("pool", exr.LayerSpec.pool2d(exr.Window2D.square(4, 4)))
net.add_inputs(eye, image)
net.connect(eye, v1)
net.load_filters(v1, exr.filters.gabor_bank(7, 4, 5.0, 2.0))
net.freeze(v1)
```

## Experiments and datasets

`exrelaxer.harness` runs experiments written in Python: one folder (or one
file) per experiment in `NNtesting/experiments/`. It uses the C++ `nntest`
commands, options and result format (see
[NNtesting/README.md](../NNtesting/README.md#python-experiments)):

```bash
NNtesting/nntest.py list                 # or: python -m exrelaxer.nntest list, exr-nntest list
NNtesting/nntest.py run mnist_gabor --trials 3 --out results.jsonl
```

`exrelaxer.datasets` reads what `NNtesting/datasets/fetch.py` prepares:

```python
from exrelaxer import datasets
train = datasets.load("mnist", "train")              # images float32 N x C x H x W in 0..1
test = datasets.load("mnist", "test", limit=2000)
train.images.shape, train.labels, train.classes, train.shape   # shape: C x H x W of one image
datasets.load("fashion_mnist", fetch_missing=True)   # runs fetch.py first if needed
```

The data directory is `data_dir=`, then `$EXR_DATA`, then
`NNtesting/data` of the repository around the current directory.

## API

The names are the C++ names in snake_case. The pages in [doc/](../doc/README.md)
describe the behaviour.

| Python | C++ |
|---|---|
| `reseed(seed)` | `exr::reseed` |
| `Network()`, `add_layer`, `connect`, `add_feedback`, `add_inputs(target, count or Shape, name="")`, `connect_inputs(name, target)`, `add_output` | `network` building |
| `freeze`, `unfreeze`, `is_frozen`, `set_recovery_jitter`, `set_learning_jitter`, `set_alpha_jitter`, `set_learning_rule`, `set_update_order`, `use_default_update_order` | the same |
| `set_input`, `set_inputs(array or list)`, `set_inputs(name, values)`, `step()`, `apply_reward(reward, lr)`, `outputs()` | running |
| `run(inputs[T×N], rewards=None, learning_rate=0)` → `T×M` | a `step` loop in C++ |
| `apply_reward_to(layer, reward, lr)` | one layer learns (unless frozen): a reward per readout |
| `apply_modulators_to(layer, modulators, lr)`, `apply_error(errors, lr)`, `bias(id)`, `set_bias` | a reward per neuron; an error per output; learned biases |
| `layer_output(id)` (C×H×W), `layer_shape`, `layer_size`, `layer_type`, `layer_spec`, `layer_name`, `find_layer`, `neuron_state(id)` (threshold, recovery, learning gain, alpha per neuron) | inspection |
| `state_probe(id)` (output, threshold, resting threshold, Sign eligibility, output trace, habituation streak and previous sum per neuron), `last_inputs(id, i)` (what a Dense neuron summed in the last step), `input_trace(id, i)` | read-only state probes for experiments ([model](../doc/model.md#inspecting-the-state)) |
| `weights(id, neuron)`, `set_weights` (Dense); `kernel(id, channel)`, `set_kernel`, `load_filters(id, bank)` (Conv2D); `retina_points(id)`; `cochlea_bands(id)`, `cochlea_power(id)`; `set_output(id, i, v)` | layer access |
| `edges`, `output_layers`, `update_order`, `inputs`, `input_values(name)`, `input_sources`, `input_count`, `layer_count`, `describe()` | the same |
| `save(path)`, `Network.load(path, mode)`, `to_bytes()`, `Network.from_bytes(data, mode)` | `save`, `load` |
| `LayerSpec.dense / conv2d / locally_connected2d / pool2d / retina / cochlea / history / resize2d / disparity / state(..., frozen=, recovery_jitter=, learning_jitter=, alpha_jitter=, learning_rule=)` | `LayerSpec` builders |
| `LayerSpec` fields: `has_er`, `has_habituation`, `frozen`, `learning_rule`, `rectify`, `gate`, `habituation_rule`, `threshold_growth`, `spontaneous`, `resting_threshold`, `normalize`, ... | the same |
| `Shape`, `Window2D`, `RetinaSpec` (`Sampling`), `CochleaSpec` (`FrequencyScale`, `Compression`), `ResizeSpec` (`Interpolation`), `DisparitySpec` (`DisparityMeasure`), `PoolMode`, `Jitter` (`uniform`, `normal`, `*_relative`, `.around`, `.within`) | the same |
| `LearningRule` (`sign`, `traced`, `feedback_alignment`, `perturbation`, `oja`, `bcm`, `.with_bias`, `.with_decay`), `Habituation(steps, tolerance, decay, fade_after)`, `Spontaneous(below, amplitude, rate)`, `ThresholdGrowth(rule, amount)` (`ThresholdGrowth.Rule.LINEAR`, `LOG`, `FIXED`, `MULTIPLICATIVE`) | the same |
| `filters.gaussian`, `difference_of_gaussians`, `gabor`, `gabor_bank`, `centre_surround_bank` | `exr::filters` |
| `constants.max_weight`, `baseline_threshold`, ... | `core/neuron.hpp` |
| `threads()`, `set_threads(n)`, `build_info()` | OpenMP threads; compiler, build type, native, openmp |
| `datasets.load / available / fetch`, `harness.experiment`, `harness.main`, `audio` (`tone`, `chirp`, `read_wav`, ...) | Python only (above) |

C++ exceptions become Python ones: `std::invalid_argument` becomes
`ValueError`, `std::out_of_range` becomes `IndexError`, and the rest become
`RuntimeError`. `step`, `apply_reward` and `run` release the GIL. A network
must not be used from two threads at once.

## Tests

`pytest EXrelaxer.py/tests` covers the bindings and the tools around them:
- the bindings: building and inspection, determinism, `run` against a Python
  `step` loop (bit for bit), save/load continuation, errors, jitter, retina,
  filter banks, and per-layer rewards;
- the experiment runner: folder and file discovery, typed parameters, grids,
  seeding, checks, exit codes, and JSON Lines that `compare.py` reads;
- `datasets.load`;
- learning rules per layer, `apply_error` and biases, and multimodal inputs
  (named input sources, several microphones, `Resize2D`, `Disparity`);
- the Python snake game against the C++ game's reference values;
- the audio layers against numpy's FFT, spectrogram save/load, and
  `exrelaxer.audio` (framing, WAV reading at 8 to 32 bits);
- the bar-orientation task learned from Python, at 0.97 to 1.0 accuracy
  against about 0.5 without learning.
