// Pooling: each output is the maximum or the average of its window, per
// input channel. No neurons, no weights, no learning. Positions of the window
// outside the input (padding) are ignored; a window entirely outside gives 0.
#pragma once
#include <vector>
#include "layer.hpp"
#include "spatial.hpp"

namespace exr {

class pool2d final : public layer
{
public:
    explicit pool2d(const Window2D& window, PoolMode mode = PoolMode::Max);

    LayerType type() const override { return LayerType::Pool2D; }
    // input channels x output height x output width; 0 x 1 x 1 until wired.
    Shape shape() const override;
    const Window2D& window() const { return window_; }
    PoolMode mode() const { return mode_; }

    // Every source's channels are pooled; sources share one height x width.
    void join(layer& source) override;
    void forward() override;

    // The current outputs (FullState restores them, WeightsOnly zeroes them),
    // so a restored network continues exactly.
    void serialize(std::ostream& os) const override;
    void deserialize(std::istream& is, DeserializeMode mode = DeserializeMode::FullState,
                     std::uint32_t neuronFormat = NEURON_FORMAT_VERSION) override;

protected:
    void sourceGrew(const layer& source, size_t offset, size_t count) override;

private:
    // Output channels follow the input channels: grows the output to match.
    void resizeOutput();

    spatial_inputs inputs_;
    Window2D window_;
    PoolMode mode_;
    size_t out_height_ = 0;
    size_t out_width_ = 0;
    std::vector<float> tensor_;  // input snapshot
};

} // namespace exr
