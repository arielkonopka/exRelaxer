// Like Conv2D, but every position has its own weights (no sharing), as in
// biological receptive fields. Each position's neurons (one per output
// channel) read the same window, so they form a block of 8 channels per
// SIMD pass, and learn exactly like dense neurons.
//
// Memory: positions x channels x windowSize() weights, so best for small
// maps (e.g. late in a pipeline, after pooling).
#pragma once
#include <vector>
#include "../kernels.hpp"
#include "spatial_neuron_layer.hpp"

namespace exr {

class locally_connected2d final : public spatial_neuron_layer
{
public:
    explicit locally_connected2d(size_t channels, const Window2D& window, bool hasHabituation = true,
                                 bool hasER = true, const Jitter& recoveryJitter = {},
                                 const Jitter& learningJitter = {}, const Jitter& alphaJitter = {});

    LayerType type() const override { return LayerType::LocallyConnected2D; }

    void forward() override;
    // Learning: every neuron that learns updates its weights against its
    // window, like a dense neuron (current signs, or the input trace).
    bool learns() const override { return true; }

    // Neuron `index`'s weights, in window order (empty when not wired).
    std::vector<float> weights(size_t index) const;
    // Size must equal windowSize(); throws std::invalid_argument otherwise.
    void setWeights(size_t index, const std::vector<float>& weights);

protected:
    void createWeights() override;
    void appendInputs(size_t count) override;
    void updateWeights() override;
    void copyWeights(size_t index, std::vector<float>& out) const override;
    size_t expectedWeights(size_t) const override { return wired() ? windowSize() : 0; }
    void storeWeights(size_t index, std::span<const float> weights) override;

private:
    std::vector<kernels::weight_matrix> weights_;  // per position: output channels x windowSize()
};

} // namespace exr
