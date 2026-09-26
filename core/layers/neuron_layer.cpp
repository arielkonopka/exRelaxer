#include "neuron_layer.hpp"
#include "../binary_io.hpp"
#include <istream>
#include <ostream>
#include <stdexcept>
#include <string>

namespace exr {

neuron_layer::neuron_layer(size_t count, bool hasHabituation, bool hasER, const Jitter& recoveryJitter,
                           const Jitter& learningJitter, const Jitter& alphaJitter)
    : has_habituation_(hasHabituation), has_er_(hasER),
      recovery_jitter_(recoveryJitter), learning_jitter_(learningJitter), alpha_jitter_(alphaJitter)
{
    neurons_.reserve(count);
    output_.reserve(count);
    for (size_t i = 0; i < count; ++i)
        newNeuron();
}

neuron& neuron_layer::newNeuron()
{
    neuron& n = neurons_.emplace_back(has_habituation_, has_er_);
    n.randomizeDynamics(recovery_jitter_, learning_jitter_, alpha_jitter_);
    output_.push_back(n.output());
    return n;
}

void neuron_layer::setOutput(size_t index, float value)
{
    neurons_.at(index).setOutput(value);
    output_[index] = value;
}

void neuron_layer::setRecoveryJitter(const Jitter& jitter)
{
    recovery_jitter_ = jitter;
    for (neuron& n : neurons_)
        n.randomizeRecovery(jitter);
}

void neuron_layer::setLearningJitter(const Jitter& jitter)
{
    learning_jitter_ = jitter;
    for (neuron& n : neurons_)
        n.randomizeLearningGain(jitter);
}

void neuron_layer::setAlphaJitter(const Jitter& jitter)
{
    alpha_jitter_ = jitter;
    for (neuron& n : neurons_)
        n.randomizeAlpha(jitter);
}

void neuron_layer::serialize(std::ostream& os) const
{
    binary_io::write(os, has_habituation_);
    binary_io::write(os, has_er_);
    binary_io::write(os, neurons_.size());

    std::vector<float> weights;
    for (size_t i = 0; i < neurons_.size(); ++i) {
        copyWeights(i, weights);
        neurons_[i].serialize(os, weights);
    }
    if (hasSharedWeights()) {
        copySharedWeights(weights);
        binary_io::write(os, static_cast<std::uint64_t>(weights.size()));
        binary_io::write(os, std::span<const float>(weights));
    }
}

void neuron_layer::deserialize(std::istream& is, DeserializeMode mode, std::uint32_t neuronFormat)
{
    const auto has_habituation = binary_io::read<bool>(is);
    const auto has_er = binary_io::read<bool>(is);
    const auto count = binary_io::read<size_t>(is);
    if (!is)
        return;  // the caller reports the truncated stream
    if (count > max_serialized_weights)
        throw std::runtime_error("layer deserialize: implausible neuron count " + std::to_string(count));

    // Replacing the neurons of a wired layer would leave its wiring (and the
    // readers' view of its output) describing neurons that no longer exist.
    const bool in_place = count == neurons_.size();
    if (!in_place && (wired() || hasReaders()))
        throw std::runtime_error("layer deserialize: data has " + std::to_string(count) +
                                 " neurons but this wired layer has " + std::to_string(neurons_.size()));

    // Everything is read and checked before the layer changes.
    std::vector<neuron> loaded;
    loaded.reserve(count);
    std::vector<std::vector<float>> weights(count);
    for (size_t i = 0; i < count; ++i) {
        neuron n = in_place ? neurons_[i] : neuron(has_habituation, has_er);
        weights[i] = n.deserialize(is, mode, neuronFormat);
        if (!is)
            return;  // the caller reports the truncated stream
        // Every weighted sum reads one weight per input.
        if (in_place && weights[i].size() != expectedWeights(i))
            throw std::runtime_error("layer deserialize: neuron " + std::to_string(i) + " has " +
                                     std::to_string(weights[i].size()) + " weights but reads " +
                                     std::to_string(expectedWeights(i)) + " inputs");
        loaded.push_back(n);
    }

    std::vector<float> shared;
    if (hasSharedWeights()) {
        const auto shared_count = binary_io::read<std::uint64_t>(is);
        if (shared_count > max_serialized_weights)
            throw std::runtime_error("layer deserialize: implausible shared weight count " +
                                     std::to_string(shared_count));
        shared.resize(is ? static_cast<size_t>(shared_count) : 0);
        binary_io::read(is, std::span<float>(shared));
        if (!is)
            return;  // the caller reports the truncated stream
        std::vector<float> expected;
        copySharedWeights(expected);
        if (in_place && shared.size() != expected.size())
            throw std::runtime_error("layer deserialize: " + std::to_string(shared.size()) +
                                     " shared weights, the wiring needs " + std::to_string(expected.size()));
    }

    has_habituation_ = has_habituation;
    has_er_ = has_er;
    neurons_ = std::move(loaded);
    output_.resize(count);
    for (size_t i = 0; i < count; ++i)
        output_[i] = neurons_[i].output();
    if (in_place) {
        for (size_t i = 0; i < count; ++i)
            if (!weights[i].empty())
                storeWeights(i, weights[i]);
        if (hasSharedWeights())
            storeSharedWeights(shared);
    } else {
        neuronsReplaced();  // unwired: the weights have nothing to belong to
    }
}

} // namespace exr
