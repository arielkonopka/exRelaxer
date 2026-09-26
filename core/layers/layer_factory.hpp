#pragma once
#include <cstddef>
#include <functional>
#include <map>
#include <memory>
#include "audio.hpp"
#include "layer.hpp"
#include "spatial.hpp"

namespace exr {

// Everything needed to construct a layer of any type. Fields a type does not
// use are ignored. The builders below fill in the fields each type uses:
//
//   net.addLayer("eye", LayerSpec::Retina({{1, 200, 320}, Sampling::Spiral}));
//   net.addLayer("v1", LayerSpec::Conv2D(16, Window2D::square(5, 1, 2)));
//   net.addLayer("pool", LayerSpec::Pool2D(Window2D::square(2, 2)));
//   net.addLayer("ear", LayerSpec::Cochlea({.sampleRate = 16000, .hop = 160, .bands = 40}));
//   net.addLayer("spectrogram", LayerSpec::History(32));
struct LayerSpec
{
    LayerType type = LayerType::Dense;
    size_t size = 0;             // Dense: neurons at construction; Conv2D, LocallyConnected2D: output channels;
                                 // History: ticks remembered
    bool hasHabituation = true;
    bool hasER = true;
    bool frozen = false;         // network::applyReward skips frozen layers (see network::freeze)
    Jitter recoveryJitter = {};  // per-neuron E-R recovery (default: none, all recovery_factor)
    Jitter learningJitter = {};  // per-neuron learning gain (default: none, all default_learning_gain)
    Jitter alphaJitter = {};     // per-neuron E-R alpha (default: none, all default_alpha)
    Window2D window = {};        // Conv2D, LocallyConnected2D, Pool2D
    PoolMode pool = PoolMode::Max;  // Pool2D
    RetinaSpec retina = {};      // Retina
    CochleaSpec cochlea = {};    // Cochlea

    static LayerSpec Dense(size_t size, bool hasHabituation = true, bool hasER = true)
    {
        return {LayerType::Dense, size, hasHabituation, hasER};
    }
    static LayerSpec Conv2D(size_t channels, const Window2D& window, bool hasHabituation = true, bool hasER = true)
    {
        LayerSpec s{LayerType::Conv2D, channels, hasHabituation, hasER};
        s.window = window;
        return s;
    }
    static LayerSpec LocallyConnected2D(size_t channels, const Window2D& window, bool hasHabituation = true,
                                        bool hasER = true)
    {
        LayerSpec s = Conv2D(channels, window, hasHabituation, hasER);
        s.type = LayerType::LocallyConnected2D;
        return s;
    }
    static LayerSpec Pool2D(const Window2D& window, PoolMode mode = PoolMode::Max)
    {
        LayerSpec s{LayerType::Pool2D, 0, false, false};
        s.window = window;
        s.pool = mode;
        return s;
    }
    static LayerSpec Retina(const RetinaSpec& retina, bool hasHabituation = true, bool hasER = true)
    {
        LayerSpec s{LayerType::Retina, 0, hasHabituation, hasER};
        s.retina = retina;
        return s;
    }
    static LayerSpec Cochlea(const CochleaSpec& cochlea, bool hasHabituation = true, bool hasER = true)
    {
        LayerSpec s{LayerType::Cochlea, 0, hasHabituation, hasER};
        s.cochlea = cochlea;
        return s;
    }
    static LayerSpec History(size_t length)
    {
        return {LayerType::History, length, false, false};
    }
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

} // namespace exr
