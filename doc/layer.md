# layer

`core/layers/layer.hpp`

The abstract interface every layer type implements. `network` and
`layer_factory` use layers only through this interface, so a new layer type
works with them once it implements `layer` and is registered with the
factory (see [layer_factory](layer_factory.md#adding-a-layer-type)).

`dense` ([dense](dense.md)) is the only implementation so far.

## LayerType

```cpp
enum class LayerType : uint8_t { Dense = 0, Conv2D = 1 };
```

Identifies the concrete class. It is stored in `LayerSpec`, used by the
factory to pick a creator, and written into saved networks, so **existing
values must never be renumbered**. `Conv2D` is reserved; no class implements
it yet.

## Interface

### Processing and learning

| Method | Contract |
|--------|----------|
| `void forward()` | step every neuron once against its inputs, updating the layer's output values |
| `void applyReward(float reward, float learningRate)` | apply the learning rule to every neuron (see [neuron](neuron.md#learning-updateweights)) |

### Inspection

| Method | Contract |
|--------|----------|
| `LayerType getType()` | the concrete type |
| `size_t size() const` | current number of neurons |
| `const std::vector<std::shared_ptr<float>>& getOutput() const` | one live output slot per neuron, in neuron order. Readers keep the pointers. |

### Wiring

| Method | Contract |
|--------|----------|
| `bool join(layer& source)` | every neuron this layer has *now* reads `source`'s output |
| `bool addFeedback(layer& target, size_t count)` | add `count` new neurons to `target`, reading *this* layer's output. Implemented by calling `target.addNeuronsWithGroup(count, getOutput(), *this)`. |
| `void addNeuronsWithGroup(size_t width, const std::vector<std::shared_ptr<float>>& sourceOutput, layer& source)` | add `width` new neurons reading `sourceOutput`, register as a listener of `source`, and notify this layer's own listeners that its output grew |
| `void notifySourceGrew(layer& source, const std::vector<std::shared_ptr<float>>& newOutputEntries)` | called by a source this layer listens to when the source's output grew; extend whatever reads that source |
| `bool attachInputs(const std::vector<std::shared_ptr<float>>& inputs)` | read external sensors |
| `std::vector<std::reference_wrapper<layer>> listeners` | layers that read this layer's output and must be notified when it grows |

Return values: `true` on success. Implementations should return `false` (or
throw) for requests they cannot honour; `network` turns `false` into a
`std::runtime_error`.

### Per-neuron dynamics

| Method | Contract |
|--------|----------|
| `void setRecoveryJitter(const Jitter&)` | redraw every neuron's E-R recovery from the jitter (disabled: reset to the default) and use it for neurons added later |
| `void setLearningJitter(const Jitter&)` | the same for the learning gain |
| `void setAlphaJitter(const Jitter&)` | the same for the E-R alpha |

See [neuron: per-neuron dynamics](neuron.md#per-neuron-dynamics).

### Serialization

| Method | Contract |
|--------|----------|
| `void serialize(std::ostream&) const` | write the layer's own state (its neurons), not its wiring |
| `void deserialize(std::istream&, DeserializeMode, uint32_t neuronFormat = NEURON_FORMAT_VERSION)` | read what `serialize` wrote; `neuronFormat` is the neuron data layout of the file (older network files pass older formats). When the layer was rebuilt with the same neurons, restore them **in place**, so output pointers held by other layers stay valid. |

Wiring is not the layer's job to save: `network` records the construction
history and replays it (see [network](network.md#serialization)).

## Requirements for implementations

- **Stable output pointers.** Other layers hold the pointers returned by
  `getOutput()`. A layer must never replace an existing neuron's output slot
  while wired; growth must only append.
- **Stable address.** Listeners hold references to layers, so a layer must
  not move after wiring. `network` stores layers in `unique_ptr`s for this.
- **Growth propagation.** When the layer's output grows, it must call
  `notifySourceGrew(*this, newEntries)` on every listener.
- **Deterministic results.** Results should not depend on thread count;
  `dense` guarantees this by copying each group's inputs before stepping.

## Known rough edges

- `listeners` is a public data member; anything can modify it. It is kept
  consistent by `join` / `addNeuronsWithGroup`.
- `getType()` is not `const`.
