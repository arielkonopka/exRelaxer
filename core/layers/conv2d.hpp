// 2D convolution: every output channel has one kernel (windowSize() weights)
// shared by all positions. Each output position has its own neurons, so
// E-R and habituation adapt per position (like retinotopic cells), while
// the feature they detect is the same everywhere.
#pragma once
#include <vector>
#include "spatial_neuron_layer.hpp"

namespace exr {

class conv2d final : public spatial_neuron_layer
{
public:
    explicit conv2d(size_t channels, const Window2D& window, bool hasHabituation = true, bool hasER = true,
                    const Jitter& recoveryJitter = {}, const Jitter& learningJitter = {},
                    const Jitter& alphaJitter = {});

    LayerType type() const override { return LayerType::Conv2D; }

    // Positions are cut into tiles of `tile_positions`; each tile's windows
    // are laid out for SIMD with positions as the lanes, and each kernel
    // computes 8 positions per pass. Every weighted sum is still summed in
    // window order, exactly like a scalar dot product.
    void forward() override;
    // Shared kernels learn the mean of the sign-rule updates of their
    // channel's eligible neurons: w += mean over them of delta * sign(input),
    // then clamped. Partial sums use fixed chunks of positions, reduced in
    // order, so the result does not depend on the thread count.
    void applyReward(float reward, float learningRate) override;
    bool learns() const override { return true; }

    // Kernel of an output channel: windowSize() weights in window order.
    std::vector<float> kernel(size_t channel) const;
    // Size must equal windowSize(); throws std::invalid_argument otherwise.
    void setKernel(size_t channel, const std::vector<float>& weights);

    static constexpr size_t tile_positions = 256;   // positions per SIMD tile
    // Positions per partial sum when learning: small enough to give every
    // thread work on mid-sized maps, fixed so results never depend on the
    // thread count.
    static constexpr size_t learning_chunk = tile_positions;

protected:
    void createWeights() override;
    void appendInputs(size_t count) override;

    // The kernels are saved once, after the neuron records.
    void copyWeights(size_t, std::vector<float>& out) const override { out.clear(); }
    size_t expectedWeights(size_t) const override { return 0; }
    void storeWeights(size_t, std::span<const float>) override {}
    bool hasSharedWeights() const override { return true; }
    void copySharedWeights(std::vector<float>& out) const override { out = kernels_; }
    void storeSharedWeights(std::span<const float> weights) override { kernels_.assign(weights.begin(), weights.end()); }

private:
    std::span<const float> kernelRow(size_t channel) const;

    std::vector<float> kernels_;  // output channels x windowSize(), row after row
};

} // namespace exr
