# multimodal and stereo

`core/network.hpp` (input sources), `core/layers/cochlea.*` (microphones),
`core/layers/resize2d.*`, `core/layers/disparity.*`; the specs are in
`core/layers/spatial.hpp` and `core/layers/audio.hpp`.

One network can see and hear at once. The pieces:

| Piece | What it does |
|-------|--------------|
| [named input sources](network.md#input-sources) | `addInputs(layer, shape, "camera")`, then `setInputs("camera", image)`; `connectInputs("camera", other)` feeds the same sensors to more layers |
| [cochlea](audio.md) with `channels` | several microphones, analysed alike: channels × bands × 1 |
| `Resize2D` | every channel of its sources resampled to one height × width, so maps of different sizes line up |
| `Disparity` | a left and a right view matched at a range of horizontal shifts: one channel per disparity |

Layers of any shape can already be read together by a `dense` layer (it
reads everything flat). Spatial layers (`conv2d`, `pool2d`,
`locally_connected2d`, `history`) need all their sources on the same height
× width; that is what `Resize2D` is for.

```cpp
network net;
const Shape view{1, 48, 64};
auto left  = net.addLayer("left",  LayerSpec::Retina({view}, false, false));
auto right = net.addLayer("right", LayerSpec::Retina({view}, false, false));
auto ears  = net.addLayer("ears",  LayerSpec::Cochlea({.sampleRate = 16000, .hop = 160, .bands = 32, .channels = 2}));
net.addInputs(left, view, "left_eye");
net.addInputs(right, view, "right_eye");
net.addInputs(ears, 2 * 160, "microphones");

// Stereo depth: 8 disparities, then shrink to 12 x 16.
auto depth = net.addLayer("depth", LayerSpec::Disparity({.minDisparity = 0, .maxDisparity = 7,
                                                         .measure = DisparityMeasure::Normalized}));
net.connect(left, depth);   // first: the left view
net.connect(right, depth);  // second: the right view
auto depthSmall = net.addLayer("depth_small", LayerSpec::Resize2D(12, 16, Interpolation::Area));
net.connect(depth, depthSmall);

// A spectrogram of both ears (2 x 32 x 16), stretched to the same grid.
auto heard = net.addLayer("heard", LayerSpec::History(16));
auto heardSmall = net.addLayer("heard_small", LayerSpec::Resize2D(12, 16));
net.connect(ears, heard);
net.connect(heard, heardSmall);

// One convolution reads depth and sound as 8 + 2 channels.
auto fuse = net.addLayer("fuse", LayerSpec::Conv2D(16, Window2D::square(3, 1, 1)));
net.connect(depthSmall, fuse);
net.connect(heardSmall, fuse);
```

```python
net.add_inputs(left, exr.Shape(1, 48, 64), name="left_eye")
net.set_inputs("left_eye", image)                       # any array with 48 * 64 values
spec = exr.LayerSpec.disparity(exr.DisparitySpec(0, 7, 3, exr.DisparityMeasure.Normalized))
fit = exr.LayerSpec.resize2d(12, 16, exr.Interpolation.Area)
frames = exr.audio.frames(stereo_sound, 160)            # 2 x samples -> ticks x 320
```

## Resize2D

```cpp
enum class Interpolation : uint8_t { Nearest = 0, Bilinear = 1, Area = 2 };
struct ResizeSpec { size_t height = 0, width = 0; Interpolation interpolation = Interpolation::Bilinear; };
LayerSpec::Resize2D(height, width, interpolation);   // LayerType::Resize2D = 7
```

- Output: input channels × height × width. Channels follow the sources like
  `Pool2D`: every channel of every source (they share one height × width),
  and channels a source grows later.
- **Nearest**: the input pixel under the output pixel's centre.
- **Bilinear**: pixel centres aligned (`(o + 0.5) · in / out − 0.5`),
  clamped at the edges. The same size gives the input back exactly.
- **Area**: each output pixel is the mean of the input pixels it covers,
  weighted by overlap: the right choice for shrinking. When enlarging it is
  bilinear.
- The resampling is separable (rows, then columns) with weights computed
  once, when the first source is joined. No neurons, no weights; it saves
  its outputs like `Pool2D`.
- A zero height or width throws `std::invalid_argument`.

## Disparity

```cpp
enum class DisparityMeasure : uint8_t { Correlation = 0, Difference = 1, Normalized = 2 };
struct DisparitySpec {
    int minDisparity = 0;
    int maxDisparity = 4;   // inclusive
    size_t window = 3;      // odd box size
    DisparityMeasure measure = DisparityMeasure::Correlation;
};
LayerSpec::Disparity(spec);   // LayerType::Disparity = 8
```

The first source joined is the **left** view, the second the **right**;
both must have the same shape C × H × W. The output is D × H × W with
`D = maxDisparity − minDisparity + 1`. Channel `d` at (y, x) compares left
pixel x with right pixel `x − (minDisparity + d)`, over a `window × window`
box around (y, x) and over all C channels:

| Measure | Value | Best match |
|---------|-------|------------|
| Correlation | mean of `left · right` | highest |
| Difference | mean of `|left − right|` | 0 |
| Normalized | normalized cross-correlation, in [−1, 1] (0 where either view is flat) | 1 |

A near object appears further left in the right eye than in the left, so
its disparity is positive; objects beyond the fixation distance have
negative disparities (use a negative `minDisparity`). Pixels whose partner
is outside the right view are left out of the box; with none left the
value is 0.

The views can be anything with the same shape: retinas, Gabor or
centre-surround responses (`conv2d`), or several channels of each. Filtered
views make the match less sensitive to brightness differences between the
eyes; `Normalized` does too.

Errors: `maxDisparity < minDisparity`, an even window or an unknown measure
throw `std::invalid_argument`; a right view of another shape throws
`std::invalid_argument`, a third source `std::logic_error`, and views whose
shapes stop matching (one grew) `std::logic_error` on `forward()`. Cost:
`D · C · H · W` for matching plus `D · H · W · window` for the box, split
between threads by disparity.

## Experiments

- `nntest run stereo_depth`: random-dot stereograms (each eye sees only
  noise). Where is the near square? 0.98 with Disparity, chance without.
- `nntest run audiovisual`: objects that look and sound different. Fusion
  of both senses, and readouts on sound that learn what objects sound like
  from what sight says, without labels.

Results and discussion: [research log §12](research.md#12-several-senses-and-stereo-vision).
