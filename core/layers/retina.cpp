#include "retina.hpp"
#include <algorithm>
#include <cmath>
#include <numbers>
#include <stdexcept>
#include <string>
#include "../kernels.hpp"
#include "../parallel.hpp"

namespace exr {

std::vector<retina::Point> retina::samplePoints(const RetinaSpec& spec)
{
    const Shape& in = spec.input;
    if (in.size() == 0)
        throw std::invalid_argument("retina: the image must not be empty");
    std::vector<Point> points;
    if (spec.sampling == Sampling::Grid) {
        points.reserve(in.height * in.width);
        for (size_t y = 0; y < in.height; ++y)
            for (size_t x = 0; x < in.width; ++x)
                points.push_back({static_cast<double>(x), static_cast<double>(y)});
        return points;
    }

    if (!(spec.spacing > 0.0f))
        throw std::invalid_argument("retina: spiral spacing must be > 0");
    const double cx = (static_cast<double>(in.width) - 1.0) / 2.0;
    const double cy = (static_cast<double>(in.height) - 1.0) / 2.0;
    // Samples stay inside the image, so every bilinear tap is a real pixel.
    double max_radius = std::min(cx, cy);
    if (spec.radius > 0.0f)
        max_radius = std::min(max_radius, static_cast<double>(spec.radius));

    const double spacing = spec.spacing;
    const double b = spacing / (2.0 * std::numbers::pi);  // r = b * theta: turns `spacing` apart
    points.push_back({cx, cy});
    double theta = 0.0;
    for (;;) {
        // Arc length grows by sqrt(r^2 + b^2) per radian: step `spacing` along it.
        const double r = b * theta;
        theta += spacing / std::sqrt(r * r + b * b);
        const double next = b * theta;
        if (next > max_radius)
            break;
        points.push_back({cx + next * std::cos(theta), cy + next * std::sin(theta)});
    }
    return points;
}

retina::retina(const RetinaSpec& spec, bool hasHabituation, bool hasER, const Jitter& recoveryJitter,
               const Jitter& learningJitter, const Jitter& alphaJitter)
    : retina(spec, samplePoints(spec), hasHabituation, hasER, recoveryJitter, learningJitter, alphaJitter)
{
}

retina::retina(const RetinaSpec& spec, std::vector<Point> points, bool hasHabituation, bool hasER,
               const Jitter& recoveryJitter, const Jitter& learningJitter, const Jitter& alphaJitter)
    : neuron_layer(spec.input.channels * points.size(), hasHabituation, hasER, recoveryJitter, learningJitter,
                   alphaJitter),
      spec_(spec), points_(std::move(points))
{
    const size_t H = spec.input.height, W = spec.input.width;
    tap_begin_.reserve(points_.size() + 1);
    for (const Point& p : points_) {
        tap_begin_.push_back(taps_.size());
        if (spec.sampling == Sampling::Grid) {
            taps_.push_back({static_cast<std::uint32_t>(static_cast<size_t>(p.y) * W + static_cast<size_t>(p.x)), 1.0f});
            continue;
        }
        // Bilinear: the 4 pixels around the sample, weighted by closeness.
        const size_t x0 = std::min(static_cast<size_t>(std::floor(p.x)), W > 1 ? W - 2 : 0);
        const size_t y0 = std::min(static_cast<size_t>(std::floor(p.y)), H > 1 ? H - 2 : 0);
        const double fx = W > 1 ? p.x - static_cast<double>(x0) : 0.0;
        const double fy = H > 1 ? p.y - static_cast<double>(y0) : 0.0;
        const double weights[4] = {(1 - fx) * (1 - fy), fx * (1 - fy), (1 - fx) * fy, fx * fy};
        const size_t pixels[4] = {y0 * W + x0, y0 * W + x0 + 1, (y0 + 1) * W + x0, (y0 + 1) * W + x0 + 1};
        for (int t = 0; t < 4; ++t)
            if (weights[t] > 0.0)
                taps_.push_back({static_cast<std::uint32_t>(pixels[t]), static_cast<float>(weights[t])});
    }
    tap_begin_.push_back(taps_.size());
}

Shape retina::shape() const
{
    return spec_.sampling == Sampling::Grid ? spec_.input : Shape{spec_.input.channels, 1, points_.size()};
}

void retina::attachInputs(const InputRange& sensors)
{
    if (sensors_)
        throw std::logic_error("retina: it already reads its image");
    if (sensors.count != spec_.input.size())
        throw std::invalid_argument("retina: the image has " + std::to_string(spec_.input.size()) +
                                    " values, got " + std::to_string(sensors.count) + " sensors");
    sensors_ = sensors;
}

void retina::forward()
{
    if (!sensors_)
        return;
    const std::span<const float> image = sensors_->values();
    const size_t N = points_.size(), plane = spec_.input.height * spec_.input.width;
    parallelChunks(neurons_.size(), kernels::threadsFor(taps_.size() * spec_.input.channels),
                   [&](size_t i0, size_t i1) {
        for (size_t i = i0; i < i1; ++i) {
            const size_t c = i / N, k = i % N;
            const std::span<const float> channel = image.subspan(c * plane, plane);
            float sum = 0.0f;
            for (size_t t = tap_begin_[k]; t < tap_begin_[k + 1]; ++t)
                sum += taps_[t].weight * channel[taps_[t].pixel];
            output_[i] = neurons_[i].activate(sum);
        }
    });
}

} // namespace exr
