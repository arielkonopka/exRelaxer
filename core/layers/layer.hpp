#pragma once
#include <vector>
#include <memory>
#include <iostream>
#include "../neuron.hpp"

enum class LayerType : uint8_t {
    Dense = 0,
    Conv2D = 1
};

class layer {
public:
    virtual ~layer() = default;

    // Core processing
    virtual void forward() = 0;
    // Learning & Plasticity
    virtual void applyReward(float reward,float learningRate) = 0;

    // Inspection
    virtual LayerType getType()=0;
    virtual size_t size() const = 0;
    virtual const std::vector<std::shared_ptr<float>>& getOutput() const = 0;

    // Topology wiring
    virtual bool join(layer& source) = 0;
    virtual bool addFeedback(layer& target, size_t feedback_count) = 0;
    virtual void notifySourceGrew(layer& source ,const std::vector<std::shared_ptr<float>>& newOutputEntries)=0;

    // Per-neuron dynamics: redraw that parameter for every existing neuron
    // from `jitter` (a disabled jitter resets it to the default), and use
    // `jitter` for neurons added later. See Jitter in neuron.hpp.
    virtual void setRecoveryJitter(const Jitter& jitter) = 0;
    virtual void setLearningJitter(const Jitter& jitter) = 0;
    virtual void setAlphaJitter(const Jitter& jitter) = 0;

    // Polymorphic serialization. `neuronFormat`: the NEURON_FORMAT_VERSION the
    // data was written with (older network files pass older formats).
    virtual void serialize(std::ostream& os) const = 0;
    virtual void deserialize(std::istream& is, DeserializeMode mode,
                             std::uint32_t neuronFormat = NEURON_FORMAT_VERSION) = 0;
    std::vector<std::reference_wrapper<layer>> listeners; // layers with a group sourced from *this
    virtual void addNeuronsWithGroup(size_t width, const std::vector<std::shared_ptr<float>>& sourceOutput, layer& source)=0;

    virtual bool attachInputs(const std::vector<std::shared_ptr<float>>& input_pointers) = 0;
};
