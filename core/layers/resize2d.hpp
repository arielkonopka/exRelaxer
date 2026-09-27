// Resizing: every input channel resampled to a fixed height x width, so maps
// of different sizes (a camera image, a spectrogram) can be brought to one
// grid and read together by a spatial layer. No neurons, no weights, no
// learning.
#pragma once
#include <vector>
#include "layer.hpp"
#include "spatial.hpp"

namespace exr {

class resize2d final : public layer
{
public:
    // Throws std::invalid_argument for a zero height or width.
    explicit resize2d(const ResizeSpec& spec);

    LayerType type() const override { return LayerType::Resize2D; }
    // input channels x spec height x spec width; 0 x 1 x 1 until wired.
    Shape shape() const override;
    const ResizeSpec& spec() const { return spec_; }

    // Every source's channels are resized; sources share one height x width.
    void join(layer& source) override;
    void forward() override;

    // The current outputs (FullState restores them, WeightsOnly zeroes them).
    void serialize(std::ostream& os) const override;
    void deserialize(std::istream& is, DeserializeMode mode = DeserializeMode::FullState,
                     std::uint32_t neuronFormat = NEURON_FORMAT_VERSION) override;

protected:
    void sourceGrew(const layer& source, size_t offset, size_t count) override;

private:
    // Output pixel o along one axis is the sum of weight * input pixel over
    // taps [first[o], first[o + 1]).
    struct Axis
    {
        std::vector<size_t> first;
        std::vector<size_t> index;
        std::vector<float> weight;
    };
    static Axis makeAxis(size_t in, size_t out, Interpolation interpolation);
    void resizeOutput();

    spatial_inputs inputs_;
    ResizeSpec spec_;
    Axis rows_, columns_;
    std::vector<float> tensor_;  // input snapshot
    std::vector<float> wide_;    // channels x input height x output width
};

} // namespace exr
