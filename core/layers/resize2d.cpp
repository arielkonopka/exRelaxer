#include "resize2d.hpp"
#include <algorithm>
#include <cmath>
#include <stdexcept>
#include "../binary_io.hpp"
#include "../kernels.hpp"
#include "../parallel.hpp"

namespace exr {

resize2d::resize2d(const ResizeSpec& spec) : spec_(spec)
{
    if (spec.height == 0 || spec.width == 0)
        throw std::invalid_argument("resize2d: the output height and width must be at least 1");
    if (spec.interpolation > Interpolation::Area)
        throw std::invalid_argument("resize2d: unknown interpolation");
}

Shape resize2d::shape() const
{
    return inputs_.empty() ? Shape::flat(0) : Shape{inputs_.channels(), spec_.height, spec_.width};
}

resize2d::Axis resize2d::makeAxis(size_t in, size_t out, Interpolation interpolation)
{
    Axis axis;
    const double scale = static_cast<double>(in) / static_cast<double>(out);
    if (interpolation == Interpolation::Area && out >= in)
        interpolation = Interpolation::Bilinear;  // enlarging: nothing to average
    for (size_t o = 0; o < out; ++o) {
        axis.first.push_back(axis.index.size());
        switch (interpolation) {
        case Interpolation::Nearest: {
            const auto i = static_cast<size_t>((static_cast<double>(o) + 0.5) * scale);
            axis.index.push_back(std::min(i, in - 1));
            axis.weight.push_back(1.0f);
            break;
        }
        case Interpolation::Bilinear: {
            const double s = std::clamp((static_cast<double>(o) + 0.5) * scale - 0.5, 0.0, static_cast<double>(in - 1));
            const auto i0 = static_cast<size_t>(s);
            const auto f = static_cast<float>(s - static_cast<double>(i0));
            axis.index.push_back(i0);
            axis.weight.push_back(1.0f - f);
            if (f > 0.0f) {
                axis.index.push_back(i0 + 1);
                axis.weight.push_back(f);
            }
            break;
        }
        case Interpolation::Area: {
            const double lo = static_cast<double>(o) * scale, hi = static_cast<double>(o + 1) * scale;
            for (auto i = static_cast<size_t>(lo); i < in && static_cast<double>(i) < hi; ++i) {
                const double covered = std::min(hi, static_cast<double>(i + 1)) - std::max(lo, static_cast<double>(i));
                if (covered <= 0.0)
                    continue;
                axis.index.push_back(i);
                axis.weight.push_back(static_cast<float>(covered / scale));
            }
            break;
        }
        }
    }
    axis.first.push_back(axis.index.size());
    return axis;
}

void resize2d::join(layer& source)
{
    if (inputs_.reads(source))
        return;
    const bool first = inputs_.empty();
    inputs_.add(source);
    if (first) {
        rows_ = makeAxis(inputs_.height(), spec_.height, spec_.interpolation);
        columns_ = makeAxis(inputs_.width(), spec_.width, spec_.interpolation);
    }
    readFrom(source);
    resizeOutput();
}

void resize2d::sourceGrew(const layer& source, size_t offset, size_t count)
{
    inputs_.grew(source, offset, count);
    resizeOutput();
}

void resize2d::resizeOutput()
{
    const size_t old_size = output_.size();
    output_.resize(inputs_.channels() * spec_.height * spec_.width, 0.0f);
    outputGrew(old_size);
}

void resize2d::forward()
{
    if (inputs_.empty())
        return;
    inputs_.gather(tensor_);
    const size_t H = inputs_.height(), W = inputs_.width(), OH = spec_.height, OW = spec_.width;
    wide_.resize(inputs_.channels() * H * OW);
    parallelChunks(inputs_.channels(), kernels::threadsFor(tensor_.size() + output_.size()), [&](size_t c0, size_t c1) {
        for (size_t c = c0; c < c1; ++c) {
            const float* in = tensor_.data() + c * H * W;
            float* wide = wide_.data() + c * H * OW;
            for (size_t y = 0; y < H; ++y)  // along each row first
                for (size_t ox = 0; ox < OW; ++ox) {
                    float sum = 0.0f;
                    for (size_t t = columns_.first[ox]; t < columns_.first[ox + 1]; ++t)
                        sum += columns_.weight[t] * in[y * W + columns_.index[t]];
                    wide[y * OW + ox] = sum;
                }
            float* out = output_.data() + c * OH * OW;
            for (size_t oy = 0; oy < OH; ++oy) {
                float* row = out + oy * OW;
                std::fill(row, row + OW, 0.0f);
                for (size_t t = rows_.first[oy]; t < rows_.first[oy + 1]; ++t) {
                    const float w = rows_.weight[t];
                    const float* src = wide + rows_.index[t] * OW;
                    for (size_t ox = 0; ox < OW; ++ox)
                        row[ox] += w * src[ox];
                }
            }
        }
    });
}

void resize2d::serialize(std::ostream& os) const
{
    binary_io::write(os, static_cast<std::uint64_t>(output_.size()));
    binary_io::write(os, std::span<const float>(output_));
}

void resize2d::deserialize(std::istream& is, DeserializeMode mode, std::uint32_t)
{
    const auto count = binary_io::read<std::uint64_t>(is);
    if (!is)
        return;  // the caller reports the truncated stream
    if (count != output_.size())
        throw std::runtime_error("resize2d deserialize: data has " + std::to_string(count) +
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
