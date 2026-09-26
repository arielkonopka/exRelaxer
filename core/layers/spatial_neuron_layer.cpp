#include "spatial_neuron_layer.hpp"
#include <stdexcept>
#include <string>

namespace exr {

spatial_neuron_layer::spatial_neuron_layer(size_t channels, const Window2D& window, bool hasHabituation, bool hasER,
                                           const Jitter& recoveryJitter, const Jitter& learningJitter,
                                           const Jitter& alphaJitter)
    : neuron_layer(0, hasHabituation, hasER, recoveryJitter, learningJitter, alphaJitter),
      window_(window), channels_(channels)
{
    if (channels == 0)
        throw std::invalid_argument("spatial layer: needs at least one output channel");
    if (window.area() == 0 || window.strideY == 0 || window.strideX == 0)
        throw std::invalid_argument("spatial layer: window kernel and stride must be at least 1");
}

Shape spatial_neuron_layer::shape() const
{
    return wired() ? Shape{channels_, out_height_, out_width_} : Shape::flat(0);
}

void spatial_neuron_layer::join(layer& source)
{
    if (inputs_.reads(source))
        return;
    if (inputs_.empty()) {
        const Shape in = source.shape();
        const size_t height = window_.outputHeight(in.height), width = window_.outputWidth(in.width);
        if (height == 0 || width == 0)
            throw std::invalid_argument("spatial layer: the window does not fit a " + std::to_string(in.height) +
                                        " x " + std::to_string(in.width) + " input");
        inputs_.add(source);
        out_height_ = height;
        out_width_ = width;
        neurons_.reserve(channels_ * positions());
        output_.reserve(channels_ * positions());
        for (size_t i = 0; i < channels_ * positions(); ++i)
            newNeuron();
        createWeights();
    } else {
        appendInputs(inputs_.add(source) * window_.area());
    }
    readFrom(source);
}

void spatial_neuron_layer::sourceGrew(const layer& source, size_t offset, size_t count)
{
    appendInputs(inputs_.grew(source, offset, count) * window_.area());
}

void spatial_neuron_layer::windowAt(size_t position, std::span<float> out) const
{
    const Shape in{inputs_.channels(), inputs_.height(), inputs_.width()};
    gatherWindow(tensor_, in, window_, position / out_width_, position % out_width_, out);
}

} // namespace exr
