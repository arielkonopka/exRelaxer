#pragma once
#include "../learning.hpp"
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
//   net.addLayer("fit", LayerSpec::Resize2D(32, 32));
//   net.addLayer("depth", LayerSpec::Disparity({.minDisparity = 0, .maxDisparity = 8}));
//   net.addLayer("h_state", LayerSpec::State());  // then net.connect(h, h_state)
struct LayerSpec
{
    LayerType type = LayerType::Dense;
    size_t size = 0;             // Dense: neurons at construction; Conv2D, LocallyConnected2D: output channels;
                                 // History: ticks remembered
                                 // State: the fields, a bit mask (1 output, 2 threshold, 4 habituation)
    bool hasHabituation = true;
    bool hasER = true;
    bool frozen = false;         // network::applyReward skips frozen layers (see network::freeze)
    Jitter recoveryJitter = {};  // per-neuron E-R recovery (default: none, all recovery_factor)
    Jitter learningJitter = {};  // per-neuron learning gain (default: none, all default_learning_gain)
    Jitter alphaJitter = {};     // per-neuron E-R alpha (default: none, all default_alpha)
    LearningRule learningRule = {};  // how the layer learns (layers with weights; see learning.hpp)
    float gate = 0.0f;           // neurons without E-R: fixed firing threshold (see neuron::gate); 0 = linear
    ThresholdGrowth thresholdGrowth = {};  // how E-R thresholds grow on firing (default: linear, amount 0.5)
    Spontaneous spontaneous = {};  // when and how strongly E-R neurons fire on their own (default: the original)
    Habituation habituationRule = {};  // how habituation suppresses repeated inputs (default: cut after 100 exact repeats)
    bool rectify = false;        // neurons without E-R: ReLU, only sums above the gate pass (see neuron::rectified)
    float restingThreshold = baseline_threshold;  // E-R: resting threshold, eligibility boundary, half the floor after firing
    bool normalize = false;      // Dense, Conv2D, LocallyConnected2D: weighted sum / |w| (see neuron_layer::setNormalized);
                                 // their builders below turn it on
    Window2D window = {};        // Conv2D, LocallyConnected2D, Pool2D
    PoolMode pool = PoolMode::Max;  // Pool2D
    RetinaSpec retina = {};      // Retina
    CochleaSpec cochlea = {};    // Cochlea
    ResizeSpec resize = {};      // Resize2D
    DisparitySpec disparity = {};  // Disparity

    static LayerSpec Dense(size_t size, bool hasHabituation = true, bool hasER = true)
    {
        LayerSpec s{LayerType::Dense, size, hasHabituation, hasER};
        s.normalize = true;
        return s;
    }
    static LayerSpec Conv2D(size_t channels, const Window2D& window, bool hasHabituation = true, bool hasER = true)
    {
        LayerSpec s{LayerType::Conv2D, channels, hasHabituation, hasER};
        s.window = window;
        s.normalize = true;
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
    static LayerSpec Resize2D(size_t height, size_t width, Interpolation interpolation = Interpolation::Bilinear)
    {
        LayerSpec s{LayerType::Resize2D, 0, false, false};
        s.resize = {height, width, interpolation};
        return s;
    }
    // Experimental: a neuron layer's output, threshold and habituation
    // streak per neuron, three values by default (see state_tap.hpp and
    // doc/state_output.md).
    static LayerSpec State(bool output = true, bool threshold = true, bool habituation = true)
    {
        return {LayerType::State, (output ? 1u : 0u) | (threshold ? 2u : 0u) | (habituation ? 4u : 0u), false, false};
    }
    static LayerSpec Disparity(const DisparitySpec& disparity)
    {
        LayerSpec s{LayerType::Disparity, 0, false, false};
        s.disparity = disparity;
        return s;
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
