#include "layer.hpp"
#include <algorithm>
#include <stdexcept>

namespace exr {

layer::~layer()
{
    // A destroyed reader must not be notified later. Readers of this layer
    // keep references to its output buffer: a source must outlive its readers
    // (a network owns and destroys all of its layers together).
    const auto isThis = [this](const layer& other) { return same(other, *this); };
    for (layer& source : sources_)
        std::erase_if(source.readers_, isThis);
    for (layer& reader : readers_)
        std::erase_if(reader.sources_, isThis);
}

void layer::applyReward(float, float) {}

void layer::join(layer&)
{
    throw std::logic_error("layer: this layer type cannot join a source");
}

void layer::attachInputs(const InputRange&)
{
    throw std::logic_error("layer: this layer type cannot read sensors");
}

void layer::addNeurons(size_t, layer&)
{
    throw std::logic_error("layer: this layer type cannot add neurons");
}

void layer::sourceGrew(const layer&, size_t, size_t) {}

void layer::readFrom(layer& source)
{
    if (std::ranges::any_of(source.readers_, [this](const layer& reader) { return same(reader, *this); }))
        return;
    source.readers_.push_back(*this);
    sources_.push_back(source);
}

void layer::outputGrew(size_t oldSize)
{
    if (output_.size() == oldSize)
        return;
    // Indexed loop: a reader's reaction may register further readers.
    for (size_t i = 0; i < readers_.size(); ++i)
        readers_[i].get().sourceGrew(*this, oldSize, output_.size() - oldSize);
}

} // namespace exr
