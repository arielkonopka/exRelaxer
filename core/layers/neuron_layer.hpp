// What every layer made of neurons shares: the neurons and their output
// slots, per-neuron dynamics (jitter), and serialization of neuron records.
// Derived layers decide how neurons are wired and own the weights.
#pragma once
#include <span>
#include <vector>
#include "layer.hpp"

namespace exr {

class neuron_layer : public layer
{
public:
    bool hasNeurons() const override { return true; }
    bool hasHabituation() const { return has_habituation_; }
    bool hasER() const { return has_er_; }

    // Dynamics state of each neuron (threshold, recovery, gain, ...).
    std::span<const neuron> neurons() const { return neurons_; }
    std::span<neuron> neurons() { return neurons_; }

    // Drives an output by hand (e.g. a layer used as a fixed source). The
    // next forward() overwrites it for wired neurons.
    void setOutput(size_t index, float value);

    // Per-neuron dynamics: redraws that parameter for every existing neuron,
    // in neuron order (a disabled jitter resets it to the default), and uses
    // the jitter for neurons added later. See Jitter in neuron.hpp.
    void setRecoveryJitter(const Jitter& jitter);
    void setLearningJitter(const Jitter& jitter);
    void setAlphaJitter(const Jitter& jitter);
    const Jitter& recoveryJitter() const { return recovery_jitter_; }
    const Jitter& learningJitter() const { return learning_jitter_; }
    const Jitter& alphaJitter() const { return alpha_jitter_; }

    // Flags, neuron count, then one record per neuron (see neuron::serialize)
    // carrying that neuron's weights; then, for layer types with weights
    // shared by many neurons (e.g. Conv2D kernels), their count and values.
    void serialize(std::ostream& os) const override;
    // With the same neuron count, restores every neuron in place; the weight
    // counts must match the wiring. With another count, only an unwired layer
    // (no inputs, no readers) is rebuilt from the data. Throws
    // std::runtime_error otherwise; the layer is then left unchanged.
    void deserialize(std::istream& is, DeserializeMode mode = DeserializeMode::FullState,
                     std::uint32_t neuronFormat = NEURON_FORMAT_VERSION) override;

protected:
    // `count` neurons, not wired yet.
    neuron_layer(size_t count, bool hasHabituation, bool hasER, const Jitter& recoveryJitter,
                 const Jitter& learningJitter, const Jitter& alphaJitter);

    // Appends a neuron (with this layer's flags and jitter) and its output slot.
    neuron& newNeuron();

    // Weight hooks for serialization. The weights neuron `index` reads
    // (empty when it is not wired):
    virtual void copyWeights(size_t index, std::vector<float>& out) const = 0;
    // How many weights neuron `index` must have in loaded data:
    virtual size_t expectedWeights(size_t index) const = 0;
    virtual void storeWeights(size_t index, std::span<const float> weights) = 0;
    // True when the layer reads anything (so its neurons cannot be replaced).
    virtual bool wired() const = 0;
    // Weights shared by many neurons, saved after the neuron records. Only
    // layer types that return true from hasSharedWeights() write the block.
    virtual bool hasSharedWeights() const { return false; }
    virtual void copySharedWeights(std::vector<float>& out) const { out.clear(); }
    virtual void storeSharedWeights(std::span<const float>) {}
    // Called after deserialize() rebuilt an unwired layer with a new count.
    virtual void neuronsReplaced() {}

    std::vector<neuron> neurons_;

private:
    bool has_habituation_, has_er_;  // for every neuron this layer creates, including later growth
    Jitter recovery_jitter_, learning_jitter_, alpha_jitter_;  // likewise
};

} // namespace exr
