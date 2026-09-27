// Binocular disparity: compares a left and a right view of the same scene at
// a range of horizontal shifts, the way disparity-tuned cells in V1 do. Output
// channel d at (y, x) says how well left pixel x matches right pixel
// x - (minDisparity + d), over a window x window box and every channel (see
// DisparitySpec). A near object is shifted further left in the right view,
// so the channel that matches best tells its depth. No neurons, no weights,
// no learning: read it with conv2d, pool2d or dense layers.
//
//   auto left  = net.addLayer("left",  LayerSpec::Retina({{1, 32, 32}}));
//   auto right = net.addLayer("right", LayerSpec::Retina({{1, 32, 32}}));
//   auto depth = net.addLayer("depth", LayerSpec::Disparity({.minDisparity = 0, .maxDisparity = 6}));
//   net.connect(left, depth);   // the first source is the left view
//   net.connect(right, depth);  // the second the right view
#pragma once
#include <vector>
#include "layer.hpp"
#include "spatial.hpp"

namespace exr {

class disparity final : public layer
{
public:
    // Throws std::invalid_argument when maxDisparity < minDisparity, the
    // window is even, or the measure is unknown.
    explicit disparity(const DisparitySpec& spec);

    LayerType type() const override { return LayerType::Disparity; }
    // disparities x height x width; 0 x 1 x 1 until the left view is joined.
    Shape shape() const override;
    const DisparitySpec& spec() const { return spec_; }

    // The first source joined is the left view, the second the right; both
    // must have the same shape (checked again on every forward, as sources
    // can grow). A third source throws std::logic_error.
    void join(layer& source) override;
    // Zero until both views are joined.
    void forward() override;

    // The current outputs (FullState restores them, WeightsOnly zeroes them).
    void serialize(std::ostream& os) const override;
    void deserialize(std::istream& is, DeserializeMode mode = DeserializeMode::FullState,
                     std::uint32_t neuronFormat = NEURON_FORMAT_VERSION) override;

private:
    DisparitySpec spec_;
    const layer* left_ = nullptr;
    const layer* right_ = nullptr;
    size_t height_ = 0, width_ = 0;
};

} // namespace exr
