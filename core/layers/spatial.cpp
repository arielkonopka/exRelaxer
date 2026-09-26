#include "spatial.hpp"
#include <algorithm>
#include <stdexcept>
#include <string>

namespace exr {
namespace {

size_t outputExtent(size_t size, size_t kernel, size_t stride, size_t pad)
{
    if (kernel == 0 || stride == 0 || size + 2 * pad < kernel)
        return 0;
    return (size + 2 * pad - kernel) / stride + 1;
}

} // namespace

size_t Window2D::outputHeight(size_t height) const { return outputExtent(height, kernelHeight, strideY, padY); }
size_t Window2D::outputWidth(size_t width) const { return outputExtent(width, kernelWidth, strideX, padX); }

bool spatial_inputs::reads(const layer& source) const
{
    return std::ranges::any_of(sources_, [&source](const layer& s) { return same(s, source); });
}

size_t spatial_inputs::add(const layer& source)
{
    const Shape shape = source.shape();
    if (shape.size() == 0)
        throw std::invalid_argument("spatial layer: a source must have outputs before it is joined");
    if (!sources_.empty() && (shape.height != height_ || shape.width != width_))
        throw std::invalid_argument("spatial layer: source is " + std::to_string(shape.height) + " x " +
                                    std::to_string(shape.width) + ", earlier sources are " +
                                    std::to_string(height_) + " x " + std::to_string(width_));
    height_ = shape.height;
    width_ = shape.width;
    const size_t plane = height_ * width_;
    for (size_t c = 0; c < shape.channels; ++c)
        channels_.push_back({source, c * plane});
    sources_.push_back(source);
    return shape.channels;
}

size_t spatial_inputs::grew(const layer& source, size_t offset, size_t count)
{
    const size_t plane = height_ * width_;
    if (count % plane != 0)
        throw std::logic_error("spatial layer: a source grew by " + std::to_string(count) +
                               " outputs, not whole " + std::to_string(height_) + " x " +
                               std::to_string(width_) + " channels");
    for (size_t c = 0; c < count / plane; ++c)
        channels_.push_back({source, offset + c * plane});
    return count / plane;
}

void spatial_inputs::gather(std::vector<float>& tensor) const
{
    const size_t plane = height_ * width_;
    tensor.resize(channels_.size() * plane);
    auto out = tensor.begin();
    for (const Channel& channel : channels_)
        out = std::ranges::copy(channel.source.get().output().subspan(channel.offset, plane), out).out;
}

void gatherWindow(std::span<const float> tensor, const Shape& shape, const Window2D& window, size_t oy,
                  size_t ox, std::span<float> out)
{
    const size_t plane = shape.height * shape.width;
    size_t i = 0;
    for (size_t c = 0; c < shape.channels; ++c) {
        const std::span<const float> channel = tensor.subspan(c * plane, plane);
        for (size_t ky = 0; ky < window.kernelHeight; ++ky) {
            // Signed arithmetic via wrap-around: an input row "above" the
            // image becomes a huge unsigned value, rejected by the bound check.
            const size_t y = oy * window.strideY + ky - window.padY;
            for (size_t kx = 0; kx < window.kernelWidth; ++kx) {
                const size_t x = ox * window.strideX + kx - window.padX;
                out[i++] = (y < shape.height && x < shape.width) ? channel[y * shape.width + x] : 0.0f;
            }
        }
    }
}

} // namespace exr
