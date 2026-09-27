#include "disparity.hpp"
#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>
#include "../binary_io.hpp"
#include "../kernels.hpp"
#include "../parallel.hpp"

namespace exr {
namespace {

std::string describe(const Shape& s)
{
    return std::to_string(s.channels) + " x " + std::to_string(s.height) + " x " + std::to_string(s.width);
}

// Sums every plane of `planes` (count planes of H x W) over a box of radius r
// (zero outside), in place, using `row` as scratch.
void boxSum(std::vector<float>& planes, size_t count, size_t H, size_t W, size_t r, std::vector<float>& tmp)
{
    if (r == 0)
        return;
    tmp.resize(H * W);
    for (size_t p = 0; p < count; ++p) {
        float* plane = planes.data() + p * H * W;
        for (size_t y = 0; y < H; ++y)
            for (size_t x = 0; x < W; ++x) {
                float sum = 0.0f;
                for (size_t i = x > r ? x - r : 0; i <= std::min(x + r, W - 1); ++i)
                    sum += plane[y * W + i];
                tmp[y * W + x] = sum;
            }
        for (size_t y = 0; y < H; ++y)
            for (size_t x = 0; x < W; ++x) {
                float sum = 0.0f;
                for (size_t j = y > r ? y - r : 0; j <= std::min(y + r, H - 1); ++j)
                    sum += tmp[j * W + x];
                plane[y * W + x] = sum;
            }
    }
}

} // namespace

disparity::disparity(const DisparitySpec& spec) : spec_(spec)
{
    if (spec.maxDisparity < spec.minDisparity)
        throw std::invalid_argument("disparity: maxDisparity must be at least minDisparity");
    if (spec.window == 0 || spec.window % 2 == 0)
        throw std::invalid_argument("disparity: the window must be odd");
    if (spec.measure > DisparityMeasure::Normalized)
        throw std::invalid_argument("disparity: unknown measure");
}

Shape disparity::shape() const
{
    return left_ ? Shape{spec_.count(), height_, width_} : Shape::flat(0);
}

void disparity::join(layer& source)
{
    if (same(source, *this))
        throw std::invalid_argument("disparity: cannot read itself");
    const Shape s = source.shape();
    if (s.size() == 0)
        throw std::invalid_argument("disparity: a view must have outputs before it is joined");
    if (!left_) {
        left_ = &source;
        height_ = s.height;
        width_ = s.width;
        readFrom(source);
        const size_t old_size = output_.size();
        output_.resize(spec_.count() * height_ * width_, 0.0f);
        outputGrew(old_size);
        return;
    }
    if (right_ || same(source, *left_))
        throw std::logic_error("disparity: reads exactly two views, a left and a right");
    if (s != left_->shape())
        throw std::invalid_argument("disparity: the right view is " + describe(s) + ", the left view is " +
                                    describe(left_->shape()));
    right_ = &source;
    readFrom(source);
}

void disparity::forward()
{
    if (!right_)
        return;
    const Shape s = left_->shape();
    if (s != right_->shape())
        throw std::logic_error("disparity: the views no longer have the same shape (" + describe(s) + " and " +
                               describe(right_->shape()) + ")");
    const std::span<const float> L = left_->output(), R = right_->output();
    const size_t C = s.channels, H = height_, W = width_, plane = H * W, r = spec_.window / 2;
    const bool normalized = spec_.measure == DisparityMeasure::Normalized;
    // Per pixel: valid count and value (Correlation, Difference), or valid count
    // and sums of l, r, l^2, r^2, l*r (Normalized), each over the channels.
    const size_t sums = normalized ? 6 : 2;
    const size_t D = spec_.count();
    parallelChunks(D, kernels::threadsFor(D * C * plane * (1 + spec_.window)), [&](size_t d0, size_t d1) {
        std::vector<float> acc(sums * plane), tmp;
        for (size_t d = d0; d < d1; ++d) {
            const long shift = static_cast<long>(spec_.minDisparity) + static_cast<long>(d);
            std::ranges::fill(acc, 0.0f);
            for (size_t y = 0; y < H; ++y)
                for (size_t x = 0; x < W; ++x) {
                    const long xr = static_cast<long>(x) - shift;
                    if (xr < 0 || xr >= static_cast<long>(W))
                        continue;
                    const size_t p = y * W + x, q = y * W + static_cast<size_t>(xr);
                    acc[p] = static_cast<float>(C);
                    for (size_t c = 0; c < C; ++c) {
                        const float l = L[c * plane + p], rv = R[c * plane + q];
                        if (!normalized) {
                            acc[plane + p] += spec_.measure == DisparityMeasure::Correlation ? l * rv : std::abs(l - rv);
                        } else {
                            acc[plane + p] += l;
                            acc[2 * plane + p] += rv;
                            acc[3 * plane + p] += l * l;
                            acc[4 * plane + p] += rv * rv;
                            acc[5 * plane + p] += l * rv;
                        }
                    }
                }
            boxSum(acc, sums, H, W, r, tmp);
            float* out = output_.data() + d * plane;
            for (size_t p = 0; p < plane; ++p) {
                const float n = acc[p];
                if (n <= 0.0f) {
                    out[p] = 0.0f;
                    continue;
                }
                if (!normalized) {
                    out[p] = acc[plane + p] / n;
                    continue;
                }
                const float sl = acc[plane + p], sr = acc[2 * plane + p];
                const float vl = acc[3 * plane + p] - sl * sl / n, vr = acc[4 * plane + p] - sr * sr / n;
                const float cov = acc[5 * plane + p] - sl * sr / n;
                const float denom = std::sqrt(std::max(vl, 0.0f) * std::max(vr, 0.0f));
                out[p] = denom > 1e-6f * n ? std::clamp(cov / denom, -1.0f, 1.0f) : 0.0f;
            }
        }
    });
}

void disparity::serialize(std::ostream& os) const
{
    binary_io::write(os, static_cast<std::uint64_t>(output_.size()));
    binary_io::write(os, std::span<const float>(output_));
}

void disparity::deserialize(std::istream& is, DeserializeMode mode, std::uint32_t)
{
    const auto count = binary_io::read<std::uint64_t>(is);
    if (!is)
        return;  // the caller reports the truncated stream
    if (count != output_.size())
        throw std::runtime_error("disparity deserialize: data has " + std::to_string(count) +
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
