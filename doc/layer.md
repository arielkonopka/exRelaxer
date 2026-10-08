# layer and neuron_layer

`core/layers/layer.hpp`, `core/layers/layer.cpp`,
`core/layers/neuron_layer.hpp`, `core/layers/neuron_layer.cpp`

Two levels shared by every layer type, so each concrete type only adds what
is specific to it:

```
layer            output buffer, shape, growth notifications, wiring interface
└─ neuron_layer  neurons, per-neuron dynamics (jitter), neuron-record serialization
   └─ dense      its wiring rule, the weights, forward pass and learning
```

`network` and `layer_factory` use layers only through `layer`, so a new
layer type works with them once it derives from `layer` (or `neuron_layer`)
and is registered with the factory (see
[layer_factory](layer_factory.md#adding-a-layer-type)). Layer types without
neurons (e.g. pooling) derive from `layer` directly. Everything is in
`namespace exr`.

## LayerType

```cpp
enum class LayerType : uint8_t {
    Dense = 0, Conv2D = 1, Pool2D = 2, LocallyConnected2D = 3, Retina = 4,
    Cochlea = 5, History = 6, Resize2D = 7, Disparity = 8, State = 9
};
```

Identifies the concrete class. It is stored in `LayerSpec`, used by the
factory to pick a creator, and written into saved networks, so **existing
values must never be renumbered**. The spatial types are described in
[spatial](spatial.md), Cochlea and History in [audio](audio.md), Resize2D
and Disparity in [multimodal](multimodal.md), State (experimental: a
neuron layer's output, threshold and habituation streak per neuron) in
[state output](state_output.md).

## Shape

```cpp
struct Shape { size_t channels = 0, height = 1, width = 1; size_t size() const; static Shape flat(size_t); };
```

The layout of a layer's output: channels × height × width, **channel-major**
(all of channel 0, then channel 1, ...). A layer that grows by whole channels
therefore only ever appends to its output, which is what growth propagation
relies on. Flat layers (`dense`) are `{size, 1, 1}`.

## Outputs and inputs

Each layer owns **one contiguous output buffer** (`std::vector<float>`).

| Method | Contract |
|--------|----------|
| `LayerType type() const` | the concrete class (pure virtual) |
| `Shape shape() const` | the output's layout; `Shape::flat(size())` by default, spatial layers override it |
| `size_t size() const` | number of outputs (neurons) |
| `std::span<const float> output() const` | this tick's outputs; the span is invalidated when the layer grows |
| `const std::vector<float>& outputBuffer() const` | the buffer object itself, which is what readers keep |

A reader refers to what it reads with an `InputRange`: a reference
(`std::reference_wrapper`) to the buffer **object**, an offset and a count. It stays valid when the buffer
reallocates as its owner grows. A plain `std::vector<float>` converts to an
`InputRange` over the whole vector, which is how caller-owned sensors are
attached; binding it to a temporary vector does not compile. The buffer
must outlive every reader of it: a network owns and
destroys all its layers and sensors together; standalone, declare sources
(and sensor vectors) before the layers that read them.

## Interface

### Processing and learning

| Method | Contract |
|--------|----------|
| `void forward()` | compute this tick's outputs |
| `void applyReward(float reward, float learningRate)` | apply the learning rule; the default does nothing (layers that do not learn) |
| `bool learns() const` | whether `applyReward` can change the layer (false by default) |
| `bool hasNeurons() const` | whether the layer is a `neuron_layer` |

### Wiring

| Method | Contract |
|--------|----------|
| `void join(layer& source)` | every neuron this layer has *now* reads `source`'s whole output |
| `void attachInputs(const InputRange& sensors)` | every neuron this layer has now reads the sensors |
| `void addNeurons(size_t count, layer& source)` | `count` new neurons reading `source` |
| `void addFeedback(layer& target, size_t count)` | non-virtual convenience: `target.addNeurons(count, *this)` |

Layer types that do not support an operation throw `std::logic_error` (the
defaults do). Errors are exceptions; nothing returns a status flag.

### Growth notifications (for implementers)

| Member | Contract |
|--------|----------|
| `readFrom(layer& source)` (protected) | registers this layer as a reader of `source`, once |
| `outputGrew(size_t oldSize)` (protected) | call after appending to the output: tells every reader, in registration order, that entries `[oldSize, size())` are new |
| `sourceGrew(const layer& source, size_t offset, size_t count)` (protected virtual) | a source this layer reads grew; extend whatever reads it |
| `bool hasReaders() const` | whether other layers read this one |

The reader/source lists are private to `layer`. A destroyed layer
unregisters itself from both sides, so it is never notified afterwards.

### Serialization

| Method | Contract |
|--------|----------|
| `void serialize(std::ostream&) const` | write the layer's own state, not its wiring |
| `void deserialize(std::istream&, DeserializeMode = FullState, uint32_t neuronFormat = NEURON_FORMAT_VERSION)` | read what `serialize` wrote; `neuronFormat` is the neuron data layout of the file (older network files pass older formats) |

Wiring is not the layer's job to save: `network` records the construction
history and replays it (see [network](network.md#serialization)).

## neuron_layer

The base of every layer made of [neurons](neuron.md). It owns the neurons,
creates new ones with the layer's flags and jitter, and serializes them.
Derived classes decide how neurons are wired and own the weights.

| Member | Purpose |
|--------|---------|
| `hasHabituation()`, `hasER()` | flags of every neuron the layer creates, including later growth |
| `neurons()` | span over the neurons' dynamics state (threshold, recovery, gain, alpha, ...); invalidated by growth |
| `setOutput(i, value)` | drive an output by hand, e.g. a layer used as a fixed source; the next `forward()` overwrites wired neurons |
| `setRecoveryJitter(j)`, `setLearningJitter(j)`, `setAlphaJitter(j)` | redraw that parameter for every existing neuron, in neuron order (disabled: set to its centre, or the default if it has none), and keep `j` for later growth |
| `recoveryJitter()`, `learningJitter()`, `alphaJitter()` | the current settings |
| `setGate(g)` / `gate()`, `setRectified(b)` / `rectified()` | neurons without E-R: a fixed firing threshold, and ReLU ([neuron](neuron.md#one-tick-activate)), for every neuron now and later; `std::invalid_argument` for a negative or non-finite gate, or a non-zero gate / rectification on a layer with E-R |
| `setHabituationRule(h)` / `habituationRule()` | the [habituation](neuron.md#one-tick-activate) rule of every neuron, now and later; `std::invalid_argument` for an invalid rule |
| `setThresholdGrowth(g)` / `thresholdGrowth()` | the [E-R threshold growth](neuron.md#excitationrelaxation-e-r) rule of every neuron, now and later; `std::invalid_argument` for an invalid rule |
| `setSpontaneous(s)` / `spontaneous()` | the [spontaneous firing](neuron.md#one-tick-activate) setting (level, amplitude, rate) of every neuron, now and later; `std::invalid_argument` for an invalid setting |
| `setRestingThreshold(r)` / `restingThreshold()` | the E-R resting threshold (eligibility boundary, half the floor after firing) of every neuron, now and later; neurons at rest move to it; `std::invalid_argument` outside (0, max_output] |
| `setNormalized(b)` / `normalized()` | the [normalised weighted sum](neuron.md#one-tick-activate): each sum divided by the length of the neuron's weights; `std::invalid_argument` on layers with fixed filters (Retina, Cochlea). Derived layers call `weightsChanged()` whenever they change weights outside learning, so the cached norms are recomputed |
| `learningRule()`, `setLearningRule(rule)` | the layer's [learning rule](learning.md); setting it resets the rule's state, keeps the weights |
| `bias(i)`, `setBias(i, value)` | a neuron's learned bias (rules with `bias`) |
| `applyReward(r, rate)`, `applyModulators(m, rate)`, `applyFeedback(errors, rate)` | learning with one reward, one modulator per neuron, or feedback alignment's projection of an error vector |
| `feedbackRow(i)` | a neuron's row of the fixed feedback-alignment matrix (empty before `applyFeedback`) |
| `newNeuron()` (protected) | append a neuron and its output slot |
| `beginForward()`, `fire(i, sum)`, `traceInputs(...)` (protected) | forward-pass hooks derived layers call: draw perturbation noise, add bias and noise and update traces around `neuron::activate` |
| `updateWeights()` (protected, virtual) | apply the per-neuron steps (`step_delta_`, `step_keep_`, `step_active_`) to the layer's weights |

See [neuron: per-neuron dynamics](neuron.md#per-neuron-dynamics).

### Serialization

`serialize` writes `hasHabituation`, `hasER`, the neuron count (`size_t`),
then one [neuron record](neuron.md#serialization) per neuron, carrying that
neuron's weights (obtained from the derived class through `copyWeights`),
then, for layer types with weights shared by many neurons (Conv2D kernels),
their count (`uint64`) and values, then (neuron format 3) the learning rule and its state
([learning](learning.md#serialization)).

`deserialize`:

- with the same neuron count, every neuron is restored in place; each wired
  neuron's weight count must equal what it reads (`expectedWeights`), or
  `std::runtime_error` is thrown. This is the intended use: rebuild the same
  topology, then deserialize;
- with another count, only an **unwired** layer (no inputs, no readers) is
  rebuilt from the data. A wired layer throws `std::runtime_error`.

Everything is read and validated before the layer changes, so a refused or
malformed load leaves it as it was. A truncated stream also leaves it
unchanged; the caller checks the stream state (`network::load` reports it).

## Requirements for implementations

- **Append-only growth.** Readers refer to output positions, so outputs
  must only ever be appended, and `outputGrew` called after appending.
- **Stable address.** Readers hold references to layers and to their output
  buffers, so a layer must not move after wiring. Layers are neither
  copyable nor movable; `network` stores them in `unique_ptr`s.
- **Deterministic results.** Results must not depend on thread count or SIMD
  width. `dense` guarantees this by summing in input order and by copying a
  group's inputs before stepping it (see [kernels](kernels.md#determinism)).
