// A short memory of its sources: the last `length` ticks of their outputs,
// side by side along the width, oldest first and this tick's newest last.
// Behind a Cochlea (1 x bands x 1) it is a spectrogram, bands x length, that
// the spatial layers (Conv2D, Pool2D, ...) read like an image; behind any
// C x H x W source it is C x H x (W * length). No neurons, no weights, no
// learning.
//
// Each tick the columns shift one step towards the start and the sources'
// current outputs fill the last one. A history reads its sources when it
// runs: put it after them in the network so it sees this tick's outputs.
// Until `length` ticks have passed, the older columns are 0.
#pragma once
#include <vector>
#include "layer.hpp"
#include "spatial.hpp"

namespace exr {

class history final : public layer
{
public:
    // Throws std::invalid_argument when length is 0.
    explicit history(size_t length);

    LayerType type() const override { return LayerType::History; }
    // channels x height x (width * length); 0 x 1 x 1 until wired.
    Shape shape() const override;
    size_t length() const { return length_; }

    // Every source's channels are remembered; sources share one height x width.
    void join(layer& source) override;
    void forward() override;

    // The remembered ticks (FullState restores them, WeightsOnly zeroes them),
    // so a restored network continues exactly.
    void serialize(std::ostream& os) const override;
    void deserialize(std::istream& is, DeserializeMode mode = DeserializeMode::FullState,
                     std::uint32_t neuronFormat = NEURON_FORMAT_VERSION) override;

protected:
    // New source channels start with an empty (zero) past.
    void sourceGrew(const layer& source, size_t offset, size_t count) override;

private:
    void resizeOutput();

    spatial_inputs inputs_;
    size_t length_;
    std::vector<float> tensor_;  // this tick's input
};

} // namespace exr
