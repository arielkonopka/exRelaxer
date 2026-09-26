# Audio layers

Two layer types bring sound into a network:

| Type | Header | What it does |
|------|--------|--------------|
| `Cochlea` | `core/layers/cochlea.hpp` | reads `hop` audio samples per tick and has one neuron per frequency band |
| `History` | `core/layers/history.hpp` | keeps the last `length` ticks of its sources side by side, e.g. a spectrogram |

Together they turn sound into an image (frequency × time) that the
[spatial](spatial.md) layers (`Conv2D`, `Pool2D`, `LocallyConnected2D`) read
like any other image:

```cpp
CochleaSpec spec;                 // 16 kHz, hop 160 (10 ms), window 512, 40 mel bands
auto ear  = net.addLayer("ear", LayerSpec::Cochlea(spec));           // 1 x 40 x 1
auto past = net.addLayer("spectrogram", LayerSpec::History(32));     // 1 x 40 x 32
auto v1   = net.addLayer("v1", LayerSpec::Conv2D(8, Window2D::square(5, 1, 2)));
net.addInputs(ear, spec.hop);     // hop sensors: this tick's samples
net.connect(ear, past);
net.connect(past, v1);
...
for (size_t at = 0; at + spec.hop <= sound.size(); at += spec.hop) {
    net.setInputs(std::span(sound).subspan(at, spec.hop));
    net.step();
}
```

`NNtesting/experiments/chirp_direction.cpp` is a complete example: it tells
rising from falling chirps with this pipeline, frozen Gabor filters and a
learned readout (accuracy about 0.998 against 0.56 without learning).

## Cochlea

```cpp
enum class FrequencyScale : uint8_t { Mel = 0, Linear = 1 };
enum class Compression : uint8_t { Log = 0, Linear = 1 };

struct CochleaSpec {
    float sampleRate = 16000.0f;  // samples per second
    size_t hop = 160;             // new samples per tick (the sensor count)
    size_t window = 512;          // samples analysed per tick, a power of two >= hop
    size_t bands = 40;            // frequency bands (neurons), lowest first
    float minFrequency = 50.0f;   // Hz, lower edge of the lowest band
    float maxFrequency = 0.0f;    // Hz, upper edge of the highest band; 0 = sampleRate / 2
    FrequencyScale scale = FrequencyScale::Mel;
    Compression compression = Compression::Log;
    float gain = 100.0f;          // band energy multiplier before compression
};

class cochlea final : public neuron_layer {
public:
    explicit cochlea(const CochleaSpec&, bool hasHabituation = true, bool hasER = true, jitters...);
    Shape shape() const;                            // 1 x bands x 1
    const CochleaSpec& spec() const;
    const std::vector<Band>& bands() const;         // each band's filter (Hz edges, bin weights)
    const std::vector<float>& samples() const;      // the last `window` samples, oldest first
    const std::vector<float>& power() const;        // last tick's power spectrum, window / 2 + 1 bins
    void attachInputs(const InputRange& sensors);   // exactly `hop` sensors, once
};
```

Each tick:

1. **Samples.** The window keeps the last `window` samples: the older ones
   shift by `hop` and this tick's `hop` sensors fill the end. Consecutive
   windows overlap by `window − hop` samples. Before `window / hop` ticks
   have passed, the start of the window is silence (0).
2. **Spectrum.** The window is multiplied by a (periodic) Hann window and
   transformed with an in-place radix-2 FFT. The power of bin *k*
   (`k · sampleRate / window` Hz) is normalised so that a sine of amplitude
   *A* centred on a bin has power *A²*.
3. **Bands.** `bands + 2` edges are spaced evenly between `minFrequency` and
   `maxFrequency`, in mel (`2595 · log10(1 + f / 700)`: narrow bands at low
   frequencies, like the cochlea) or in Hz. Band *b* is a triangle from
   edge *b* to edge *b + 2* with its peak (weight 1) at edge *b + 1*, so each
   band overlaps half of each neighbour. Its energy is the weighted sum of
   the power under the triangle. A band narrower than one bin hears its
   nearest bin.
4. **Compression.** `Log`: `log(1 + gain · energy)`, loudness-like and 0 for
   silence; `Linear`: `gain · energy`.
5. **Neurons.** The value is the weighted sum of band *b*'s neuron. With
   habituation and E-R on, each band adapts like a group of hair cells: a
   steady tone fades and onsets and changes stand out. With both off the
   band values pass through (clamped at ±`max_output`, like any neuron).

The filters are fixed: a cochlea has no weights and does not learn
(`applyReward` changes nothing; it has no eligible inputs). The constructor
throws `std::invalid_argument` for a spec that breaks the rules in the
comments above (window not a power of two, `window < hop`, `bands == 0`,
`minFrequency >= maxFrequency`, `maxFrequency` above Nyquist, `gain <= 0`).
`attachInputs` throws `std::invalid_argument` for a sensor count other than
`hop` and `std::logic_error` when called twice; a cochlea cannot `join`
another layer.

**Serialization:** the neuron records, then the sample window (`uint64`
count + `float`s). `FullState` restores the window, so a loaded network
continues bit-identically; `WeightsOnly` zeroes it.

**Cost:** one FFT of `window` points plus a pass over each band's bins per
tick: about 20 µs for the defaults.

## History

```cpp
class history final : public layer {
public:
    explicit history(size_t length);    // throws std::invalid_argument for 0
    Shape shape() const;                // C x H x (W * length); 0 x 1 x 1 until wired
    size_t length() const;
    void join(layer& source);           // any number of sources with the same H x W
};
```

A history remembers the last `length` ticks of its sources' outputs. Each
row of each channel of the sources becomes a row `W · length` wide: the
oldest tick first and this tick's values last. Every tick the rows shift one
tick towards the start. Behind a cochlea (1 × bands × 1) this is a
spectrogram: frequency as the height (lowest band at row 0), time as the
width (newest column last).

- **Timing.** A history copies what its sources hold when it runs; in a
  network's default (topological) order that is after them, so the newest
  column is this tick's. Until `length` ticks have passed, the older columns
  are 0.
- **Sources.** Like `Pool2D`, a history reads every channel of every source
  (see [spatial](spatial.md)); sources must share one height × width. A flat
  source (`dense`) is N × 1 × 1: N channels, one row each. When a source
  grows by whole channels (e.g. a dense layer getting feedback neurons) the
  new channels start with an empty past and every reader of the history is
  extended.
- **No neurons, no weights, no learning.**
- **Serialization:** its outputs (the remembered ticks), like `Pool2D`;
  `FullState` restores them, `WeightsOnly` zeroes them.

## In a network file

`LayerSpec` has a `cochlea` field and the builders
`LayerSpec::Cochlea(spec, hasHabituation, hasER)` and
`LayerSpec::History(length)` (the length is `LayerSpec::size`). Network
format version 8 saves the cochlea spec with every layer; see
[network](network.md#file-format).

## From Python

```python
import exrelaxer as exr
from exrelaxer import audio

spec = exr.CochleaSpec(sample_rate=16000, hop=160, bands=40)
net = exr.Network()
ear = net.add_layer("ear", exr.LayerSpec.cochlea(spec, habituation=False, er=False))
past = net.add_layer("spectrogram", exr.LayerSpec.history(32))
net.add_inputs(ear, spec.hop)
net.connect(ear, past)
net.add_output(past)

sound, rate = audio.read_wav("speech.wav", sample_rate=spec.sample_rate)
net.run(audio.frames(sound, spec.hop))      # one row per tick
image = net.layer_output(past)              # 1 x 40 x 32 spectrogram
net.cochlea_bands(ear)                      # bands x (low, centre, high) Hz
net.cochlea_power(ear)                      # last tick's power spectrum
```

`exrelaxer.audio` also has `tone`, `chirp` (exponential or linear glide),
`resample` (linear interpolation) and `frames`. `read_wav` reads 8/16/24/32-bit
PCM with the standard `wave` module and averages the channels.
