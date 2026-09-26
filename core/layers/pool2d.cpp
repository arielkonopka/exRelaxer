#include "pool2d.hpp"
#include <algorithm>
#include <limits>
#include <stdexcept>
#include <string>
#include "../binary_io.hpp"
#include "../kernels.hpp"
#include "../parallel.hpp"

namespace exr {

pool2d::pool2d(const Window2D& window, PoolMode mode) : window_(window), mode_(mode)
{
    if (window.area() == 0 || window.strideY == 0 || window.strideX == 0)
        throw std::invalid_argument("pool2d: window kernel and stride must be at least 1");
}

Shape pool2d::shape() const
{
    return inputs_.empty() ? Shape::flat(0) : Shape{inputs_.channels(), out_height_, out_width_};
}

void pool2d::join(layer& source)
{
    if (inputs_.reads(source))
        return;
    if (inputs_.empty()) {
        const Shape in = source.shape();
        const size_t height = window_.outputHeight(in.height), width = window_.outputWidth(in.width);
        if (height == 0 || width == 0)
            throw std::invalid_argument("pool2d: the window does not fit a " + std::to_string(in.height) + " x " +
                                        std::to_string(in.width) + " input");
        inputs_.add(source);
        out_height_ = height;
        out_width_ = width;
    } else {
        inputs_.add(source);
    }
    readFrom(source);
    resizeOutput();
}

void pool2d::sourceGrew(const layer& source, size_t offset, size_t count)
{
    inputs_.grew(source, offset, count);
    resizeOutput();
}

void pool2d::resizeOutput()
{
    const size_t old_size = output_.size();
    output_.resize(inputs_.channels() * out_height_ * out_width_, 0.0f);
    outputGrew(old_size);
}

void pool2d::forward()
{
    if (inputs_.empty())
        return;
    inputs_.gather(tensor_);
    const size_t H = inputs_.height(), W = inputs_.width(), plane = out_height_ * out_width_;
    parallelChunks(inputs_.channels(), kernels::threadsFor(output_.size() * window_.area()), [&](size_t c0, size_t c1) {
        for (size_t c = c0; c < c1; ++c) {
            const std::span<const float> in = std::span<const float>(tensor_).subspan(c * H * W, H * W);
            for (size_t oy = 0; oy < out_height_; ++oy)
                for (size_t ox = 0; ox < out_width_; ++ox) {
                    float best = -std::numeric_limits<float>::infinity(), sum = 0.0f;
                    size_t valid = 0;
                    for (size_t ky = 0; ky < window_.kernelHeight; ++ky) {
                        const size_t y = oy * window_.strideY + ky - window_.padY;  // wraps when above the input
                        for (size_t kx = 0; kx < window_.kernelWidth; ++kx) {
                            const size_t x = ox * window_.strideX + kx - window_.padX;
                            if (y >= H || x >= W)
                                continue;
                            const float v = in[y * W + x];
                            best = std::max(best, v);
                            sum += v;
                            ++valid;
                        }
                    }
                    float value = 0.0f;
                    if (valid > 0)
                        value = mode_ == PoolMode::Max ? best : sum / static_cast<float>(valid);
                    output_[c * plane + oy * out_width_ + ox] = value;
                }
        }
    });
}

void pool2d::serialize(std::ostream& os) const
{
    binary_io::write(os, static_cast<std::uint64_t>(output_.size()));
    binary_io::write(os, std::span<const float>(output_));
}

void pool2d::deserialize(std::istream& is, DeserializeMode mode, std::uint32_t)
{
    const auto count = binary_io::read<std::uint64_t>(is);
    if (!is)
        return;  // the caller reports the truncated stream
    if (count != output_.size())
        throw std::runtime_error("pool2d deserialize: data has " + std::to_string(count) + " outputs, the layer has " +
                                 std::to_string(output_.size()));
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
