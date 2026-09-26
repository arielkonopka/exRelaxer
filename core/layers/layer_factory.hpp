#pragma once
#include <cstddef>
#include <functional>
#include <map>
#include <memory>
#include "layer.hpp"

// Everything needed to construct a layer of any type. Layer types that need
// more parameters (e.g. Conv2D's width/height/kernel) add fields here.
struct LayerSpec
{
    LayerType type = LayerType::Dense;
    size_t size = 0;             // number of neurons at construction
    bool hasHabituation = true;
    bool hasER = true;
    bool frozen = false;         // network::applyReward skips frozen layers (see network::freeze)
    Jitter recoveryJitter = {};  // per-neuron E-R recovery (default: none, all recovery_factor)
    Jitter learningJitter = {};  // per-neuron learning gain (default: none, all default_learning_gain)
    Jitter alphaJitter = {};     // per-neuron E-R alpha (default: none, all default_alpha)
};

// Creates layers from a LayerSpec without the caller knowing the concrete
// class. Each LayerType maps to a creator function; new layer types are
// added with registerType().
//
// Built-in types are registered in the constructor rather than by static
// self-registering objects in their own .cpp files: in a static library the
// linker drops object files nothing references, and the type would silently
// be missing.
class layer_factory
{
public:
    using Creator = std::function<std::unique_ptr<layer>(const LayerSpec&)>;

    layer_factory(); // registers the built-in types

    // Shared instance with the built-in types registered.
    static layer_factory& instance();

    // Adds or replaces the creator for `type`.
    void registerType(LayerType type, Creator creator);
    bool isRegistered(LayerType type) const;

    // Throws std::invalid_argument if `spec.type` has no registered creator.
    std::unique_ptr<layer> create(const LayerSpec& spec) const;

private:
    std::map<LayerType, Creator> creators;
};
