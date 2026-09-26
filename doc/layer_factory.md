# layer_factory and LayerSpec

`core/layers/layer_factory.hpp`, `core/layers/layer_factory.cpp`

Creates [layers](layer.md) from a description (`LayerSpec`) without the
caller knowing the concrete class. [network](network.md) uses it for every
layer it creates and when loading a saved network.

## LayerSpec

```cpp
struct LayerSpec
{
    LayerType type = LayerType::Dense;
    size_t size = 0;             // number of neurons at construction
    bool hasHabituation = true;
    bool hasER = true;
    bool frozen = false;         // network::applyReward skips frozen layers
    Jitter recoveryJitter = {};  // per-neuron E-R recovery (default: none)
    Jitter learningJitter = {};  // per-neuron learning gain (default: none)
    Jitter alphaJitter = {};     // per-neuron E-R alpha (default: none)
};
```

Aggregate-initialized in field order:

```cpp
LayerSpec a{LayerType::Dense, 16};                        // habituation + E-R on
LayerSpec b{LayerType::Dense, 8, false, false};           // plain linear neurons
LayerSpec c{LayerType::Dense, 8, false, false, true};     // ... and frozen
LayerSpec d{LayerType::Dense, 32, false, true, false,
            Jitter::normal(0.02f).around(0.95f),   // recovery
            Jitter::uniform(0.1f),                 // learning gain
            Jitter::uniformRelative()};            // alpha +-50%
```

| Field | Used by |
|-------|---------|
| `type` | the factory, to select a creator |
| `size`, `hasHabituation`, `hasER`, `recoveryJitter`, `learningJitter`, `alphaJitter` | the creator (for `dense`: its constructor arguments; see [per-neuron dynamics](neuron.md#per-neuron-dynamics)) |
| `frozen` | the network only ([freezing](network.md#freezing)); creators ignore it |

`network::freeze` / `unfreeze` and `network::setRecoveryJitter` /
`setLearningJitter` change the network's copy of the spec, so
`network::layerSpec(id)` always shows the current settings.

When a layer type needs more parameters (e.g. a future `Conv2D` with
width/height/kernel), add fields to `LayerSpec` **at the end** so existing
aggregate initializers keep compiling, and extend the network's save format
(see [network](network.md#file-format)).

## layer_factory

```cpp
class layer_factory {
public:
    using Creator = std::function<std::unique_ptr<layer>(const LayerSpec&)>;

    layer_factory();                                   // registers built-in types
    static layer_factory& instance();                  // shared instance

    void registerType(LayerType type, Creator creator); // add or replace
    bool isRegistered(LayerType type) const;
    std::unique_ptr<layer> create(const LayerSpec& spec) const;
};
```

- `create` calls the creator registered for `spec.type` and throws
  `std::invalid_argument` if there is none.
- `instance()` is the process-wide factory, used by default by `network`.
  A separately constructed `layer_factory` starts with the built-in types
  and can be customised without affecting the shared one; pass it to the
  `network` constructor (the factory must outlive the network).

### Built-in types

| Type | Creator |
|------|---------|
| `LayerType::Dense` | `std::make_unique<dense>(spec.size, spec.hasHabituation, spec.hasER, spec.recoveryJitter, spec.learningJitter)` |

Built-in types are registered in the constructor, **not** by static
self-registering objects in their own source files: in a static library the
linker drops object files that nothing references, and a self-registered
type would silently be missing.

## Adding a layer type

1. Add a value to `LayerType` (never renumber existing ones; they are saved
   in network files).
2. Implement `layer` (see [requirements](layer.md#requirements-for-implementations)).
3. Register a creator, either in `layer_factory`'s constructor (built-in) or
   at startup:

```cpp
layer_factory::instance().registerType(LayerType::Conv2D,
    [](const LayerSpec& spec) -> std::unique_ptr<layer> {
        return std::make_unique<conv2d>(spec.size, spec.hasHabituation, spec.hasER);
    });
```

A saved network can only be loaded by a program whose factory knows all the
layer types it contains.
