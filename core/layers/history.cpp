#include "history.hpp"
#include <algorithm>
#include <stdexcept>
#include <string>
#include "../binary_io.hpp"

namespace exr {

history::history(size_t length) : length_(length)
{
    if (length == 0)
        throw std::invalid_argument("history: length must be at least 1");
}

Shape history::shape() const
{
    return inputs_.empty() ? Shape::flat(0) : Shape{inputs_.channels(), inputs_.height(), inputs_.width() * length_};
}

void history::join(layer& source)
{
    if (inputs_.reads(source))
        return;
    inputs_.add(source);
    readFrom(source);
    resizeOutput();
}

void history::sourceGrew(const layer& source, size_t offset, size_t count)
{
    inputs_.grew(source, offset, count);
    resizeOutput();
}

void history::resizeOutput()
{
    // Channels only ever append, so the existing channels keep their past.
    const size_t old_size = output_.size();
    output_.resize(inputs_.channels() * inputs_.height() * inputs_.width() * length_, 0.0f);
    outputGrew(old_size);
}

void history::forward()
{
    if (inputs_.empty())
        return;
    inputs_.gather(tensor_);
    const size_t W = inputs_.width(), rows = inputs_.channels() * inputs_.height(), row = W * length_;
    for (size_t r = 0; r < rows; ++r) {
        float* out = output_.data() + r * row;
        std::copy(out + W, out + row, out);
        std::copy_n(tensor_.data() + r * W, W, out + row - W);
    }
}

void history::serialize(std::ostream& os) const
{
    binary_io::write(os, static_cast<std::uint64_t>(output_.size()));
    binary_io::write(os, std::span<const float>(output_));
}

void history::deserialize(std::istream& is, DeserializeMode mode, std::uint32_t)
{
    const auto count = binary_io::read<std::uint64_t>(is);
    if (!is)
        return;  // the caller reports the truncated stream
    if (count != output_.size())
        throw std::runtime_error("history deserialize: data has " + std::to_string(count) +
                                 " outputs, the layer has " + std::to_string(output_.size()));
    std::vector<float> values(output_.size());
    binary_io::read(is, std::span<float>(values));
    if (!is)
        return;
    if (mode == DeserializeMode::FullState)
        output_ = std::move(values);
    else
        std::ranges::fill(output_, 0.0f);
}

} // namespace exr
