# Spatial layers: Retina, Conv2D, LocallyConnected2D, Pool2D

`core/layers/spatial.hpp`, `spatial_neuron_layer.hpp`, `retina.hpp`,
`conv2d.hpp`, `locally_connected2d.hpp`, `pool2d.hpp` (and their `.cpp`)

Layer types for images, all in `namespace exr`. Their outputs have a
[`Shape`](layer.md#shape) (channels × height × width, channel-major), and they
read other layers by that shape. They plug into `network` like `dense`:

```cpp
const Shape image{1, 200, 320};   // a grey 320 x 200 camera frame
auto eye  = net.addLayer("eye",  LayerSpec::Retina({image}));                        // adapts per pixel
auto v1   = net.addLayer("v1",   LayerSpec::Conv2D(16, Window2D::square(5, 1, 2)));  // 16 x 200 x 320
auto pool = net.addLayer("pool", LayerSpec::Pool2D(Window2D::square(2, 2)));         // 16 x 100 x 160
auto v2   = net.addLayer("v2",   LayerSpec::Conv2D(32, Window2D::square(5, 1, 2)));  // 32 x 100 x 160
auto out  = net.addLayer("out",  LayerSpec::Dense(10));
net.addInputs(eye, image);        // 64000 sensors: channel after channel, row after row
net.connect(eye, v1); net.connect(v1, pool); net.connect(pool, v2); net.connect(v2, out);
```

| Type | Neurons | Weights | Learns |
|------|---------|---------|--------|
| Retina | one per sample and channel | fixed sampling weights | no |
| Conv2D | one per output channel and position | one kernel per output channel, shared by all positions | yes (mean update) |
| LocallyConnected2D | one per output channel and position | own weights per position | yes (like dense) |
| Pool2D | none | none | no |

## Window2D

```cpp
struct Window2D { size_t kernelHeight = 1, kernelWidth = 1, strideY = 1, strideX = 1, padY = 0, padX = 0; };
Window2D::square(kernel, stride = 1, padding = 0);
```

Output height = `(height + 2·padY − kernelHeight) / strideY + 1`, likewise for
the width. `square(5, 1, 2)` keeps the size ("same" padding); `square(2, 2)`
halves it. Padding reads as 0 for Conv2D and LocallyConnected2D, and is
skipped by Pool2D. A window of height 1 on a `1 × 1 × N` input is a 1D
window, e.g. along a spiral retina.

## Reading sources

A spatial layer reads the **channels** of its sources:

- The **first** `join` (or `connect`) fixes the input height × width, and so
  the output size; Conv2D and LocallyConnected2D create their neurons then.
  Before that, their size is 0.
- **Later** sources must have the same height × width; their channels are
  appended, and every neuron reads them in the same weighted sum (one wiring
  group per neuron, as in [dense](dense.md#wiring-groups)). This fuses maps,
  e.g. a colour retina and a depth map.
- A **flat** layer (dense, `n × 1 × 1`) is `n` channels of one pixel. When it
  grows, the new outputs become new input channels, and every kernel gets
  matching new weights at its end. Growth that is not whole channels throws
  `std::logic_error`.
- Spatial layers do not take feedback neurons or sensors (`addNeurons`,
  `attachInputs` throw `std::logic_error`), except Retina, which reads sensors.

A neuron's inputs (`windowSize()` of them) are its window over every input
channel: channel by channel, row by row within the window. Each forward pass
and each reward first copies the inputs into a snapshot, so every neuron sees
the values from before the layer ran, also when a layer reads itself.

## Retina

```cpp
struct RetinaSpec {
    Shape input;                       // the image: channels x height x width
    Sampling sampling = Sampling::Grid;
    float spacing = 1.0f;              // Spiral: pixels between samples and between turns
    float radius = 0.0f;               // Spiral: outermost radius; 0 = the largest circle inside the image
};
LayerSpec::Retina(spec, hasHabituation = true, hasER = true);
```

The entry point for images. It reads exactly `input.size()` sensors (use
`net.addInputs(eye, image)`), channel after channel, row after row, and has
one neuron per sample and channel. With habituation and E-R on, each sample
adapts like a photoreceptor: a static image fades and changes stand out.
With both off, it passes values through (Input2D). Like every neuron, it
clamps values to ±10 (`max_output`), so normalize pixel values (e.g. to
0…1) first.

- **Grid**: one sample per pixel. Output shape = the image shape.
- **Spiral**: sample 0 is the image centre (`(width − 1) / 2`,
  `(height − 1) / 2`); the following samples lie on an Archimedean spiral
  `r = spacing · θ / 2π` going outwards. Turns are `spacing` pixels apart and
  consecutive samples are `spacing` pixels apart along the curve, so the
  samples cover the disc about evenly, in order of distance from the centre.
  Each sample interpolates its 4 nearest pixels (bilinear; exact on linear
  images). Output shape: `channels × 1 × samples`. On a 320 × 200 image
  with spacing 1 that is 31 099 samples, within radius 99.5.

`samples()` and `samplePoint(k)` (pixel coordinates, x right, y down) show
where each sample is.

## Conv2D

```cpp
LayerSpec::Conv2D(channels, window, hasHabituation = true, hasER = true);
```

One kernel of `windowSize()` weights per output channel, shared by every
position; each position has its own neurons, so E-R and habituation adapt
per position. `kernel(c)` / `setKernel(c, w)` read and set a kernel, e.g.
for fixed filter banks (difference of Gaussians, Gabor, Sobel) in a frozen
layer.

**Forward**: positions are processed in tiles of 256. A tile's windows are
laid out for SIMD with positions as the lanes, and each kernel computes 8
positions per pass (see [kernels](kernels.md)). Every weighted sum is summed
in window order, exactly like a scalar dot product.

**Learning**: a kernel moves by the **mean** of the sign-rule updates of its
channel's eligible neurons:

```
w[j] = clamp(w[j] + mean over eligible positions p of delta(c, p) × sign(input_p[j]), ±max_weight)
```

A sum would move shared kernels by thousands of steps at once. With one
position this is exactly the dense rule. Tiles are laid out transposed (a row
per window input, a column per position), so each channel's sum over
positions is one multiply by its deltas. Partial sums use fixed chunks of
256 positions and are reduced in chunk order, so results do not depend on the
thread count (checked by a test).

Kernels are saved once, after the neuron records.

## Filter banks

`core/filters.hpp`, namespace `exr::filters`: classic fixed receptive fields
to load into a Conv2D.

| Builder | Receptive field |
|---------|-----------------|
| `gaussian(size, sigma, gain = 1)` | blur; weights sum to `gain` |
| `differenceOfGaussians(size, centreSigma, surroundSigma, polarity, gain = 1)` | centre-surround, like retinal ganglion cells; `OnCentre` prefers a bright spot on a dark surround, `OffCentre` the opposite |
| `gabor(size, orientation, wavelength, sigma, phase = 0, aspect = 1, gain = 1)` | oriented stripes and edges, like V1 simple cells; `orientation` is the stripes' direction from the x axis (0 horizontal, π/2 vertical); phase 0 prefers a bright bar on the centre line, π/2 an edge; `aspect` < 1 stretches the envelope along the stripes |
| `gaborBank(size, orientations, wavelength, sigma, phases = {0, π/2})` | evenly spaced orientations in [0, π), each with every phase, orientation-major |
| `centreSurroundBank(size, centreSigma, surroundSigma)` | {OnCentre, OffCentre} |

DoG and Gabor filters have **zero mean** (a uniform image gives 0) and are
scaled so the best-matching full-contrast pattern (pixels 0 where the filter
is negative, 1 where it is positive) gives exactly `gain`: responses stay
well inside the neurons' ±10 range.

```cpp
filters::load(net.layerAs<conv2d>(v1), filters::gaborBank(7, 4, 5.0f, 2.0f));  // 8 channels
filters::load(conv, channel, filter, inputChannel);                          // one filter, one input
net.freeze(v1);
```

`load(layer, bank)` puts `bank[k]` into output channel `k`, on every input
channel divided by their count (a grey colour image responds like a grey
single-channel one). The layer must already be wired, and each filter must
have the window's size.

**Frozen filters + a learned readout** is the setup that learns best with
this library's learning rule (see [pattern_benchmark](pattern_benchmark.md)).
`FiltersTest.FrozenGaborFeaturesLearnBarOrientation` tells vertical from
horizontal bars at random positions on a noisy 24 × 24 image:

```
retina (grid) -> Conv2D 8 Gabor 7 x 7, frozen -> Pool2D max 6 x 6 -> dense 64, frozen random -> dense 1, learns
```

Accuracy goes from 0.52 to 0.99 (worst of 10 trials 0.965) after 1500
error-driven training images. The frozen random dense layer matters: pooled
orientation energies are all ≥ 0, and the learning rule uses only the sign of
each input, so it could not tell them apart directly. Random mixing gives
features whose signs differ between the classes. With 32 mixing neurons and
400 images it reached 0.87; with 64 and 400, 0.94.

## LocallyConnected2D

```cpp
LayerSpec::LocallyConnected2D(channels, window, hasHabituation = true, hasER = true);
```

Like Conv2D, but every position has its own weights, as in biological
receptive fields. A position's neurons (one per output channel) read the same
window, so they are one block of 8 channels per SIMD pass and learn exactly
like dense neurons (bit for bit against a per-neuron reference). `weights(i)`
/ `setWeights(i, w)` give a neuron's weights. Memory is positions × channels
× `windowSize()` weights, so it suits small maps, e.g. after pooling.

## Pool2D

```cpp
LayerSpec::Pool2D(window, PoolMode::Max);   // or PoolMode::Average
```

The maximum or average of each window, per input channel. Padding positions
are skipped; a window entirely in the padding gives 0. No neurons, no
weights, no learning. It saves its current outputs, so a restored network
continues exactly.

## Performance

A 320 × 200 pipeline on an i7-12700H (20 threads), default build, best of
several runs:

| Layer | Output | Forward | Learning |
|-------|--------|---------|----------|
| Retina, grid, E-R + habituation | 1 × 200 × 320 | 0.45 ms | – |
| Retina, spiral (31 099 samples) | 1 × 1 × 31099 | 0.4 ms | – |
| Conv2D 16, 5 × 5, same padding | 16 × 200 × 320 (1 M neurons) | 2.8 ms | about 4 ms |
| Pool2D 2 × 2 | 16 × 100 × 160 | 0.8 ms | – |
| Conv2D 32, 5 × 5, 16 input channels | 32 × 100 × 160 | 6.5 ms | 9–13 ms |
| Pool2D 4 × 4 | 32 × 25 × 40 | 0.2 ms | – |
| LocallyConnected2D 16, 3 × 3 | 16 × 25 × 40 | 0.15 ms | 0.2 ms |

The whole pipeline (with a dense readout and a spiral retina) steps in about
17–19 ms and steps and learns in about 35 ms. The second Conv2D computes
205 million multiply-adds per step at about 30–40 GMAC/s; its learning pass
does the same work plus the transposed layout and is the most expensive part.
