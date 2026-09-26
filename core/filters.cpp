#include "filters.hpp"
#include <cmath>
#include <stdexcept>
#include <string>
#include "layers/conv2d.hpp"

namespace exr::filters {
namespace {

// Evaluates f(dx, dy) at every pixel of a size x size filter, with (0, 0) at
// the centre (dx right, dy down), in double precision.
template <typename F>
std::vector<double> sample(size_t size, F f)
{
    if (size == 0)
        throw std::invalid_argument("filters: size must be at least 1");
    const double centre = (static_cast<double>(size) - 1.0) / 2.0;
    std::vector<double> values;
    values.reserve(size * size);
    for (size_t y = 0; y < size; ++y)
        for (size_t x = 0; x < size; ++x)
            values.push_back(f(static_cast<double>(x) - centre, static_cast<double>(y) - centre));
    return values;
}

Filter toFilter(size_t size, const std::vector<double>& values, double scale)
{
    Filter filter{size, size, {}};
    filter.weights.reserve(values.size());
    for (double v : values)
        filter.weights.push_back(static_cast<float>(v * scale));
    return filter;
}

// Zero mean, then scaled so the sum of |w| is 2 * gain: the pattern that is
// 1 where w > 0 and 0 elsewhere gives sum(w * pattern) = gain.
Filter zeroMeanFilter(size_t size, std::vector<double> values, float gain)
{
    double mean = 0.0;
    for (double v : values)
        mean += v;
    mean /= static_cast<double>(values.size());
    double absolute = 0.0;
    for (double& v : values) {
        v -= mean;
        absolute += std::abs(v);
    }
    if (absolute == 0.0)
        throw std::invalid_argument("filters: the filter is flat at this size");
    return toFilter(size, values, 2.0 * gain / absolute);
}

void checkPositive(float value, const char* what)
{
    if (!(value > 0.0f))
        throw std::invalid_argument(std::string("filters: ") + what + " must be > 0");
}

} // namespace

Filter gaussian(size_t size, float sigma, float gain)
{
    checkPositive(sigma, "sigma");
    const double s2 = 2.0 * static_cast<double>(sigma) * sigma;
    const std::vector<double> values = sample(size, [&](double dx, double dy) { return std::exp(-(dx * dx + dy * dy) / s2); });
    double sum = 0.0;
    for (double v : values)
        sum += v;
    return toFilter(size, values, gain / sum);
}

Filter differenceOfGaussians(size_t size, float centreSigma, float surroundSigma, Polarity polarity, float gain)
{
    checkPositive(centreSigma, "centre sigma");
    checkPositive(surroundSigma, "surround sigma");
    if (!(surroundSigma > centreSigma))
        throw std::invalid_argument("filters: the surround must be wider than the centre");
    const double c2 = 2.0 * static_cast<double>(centreSigma) * centreSigma;
    const double s2 = 2.0 * static_cast<double>(surroundSigma) * surroundSigma;
    const double sign = polarity == Polarity::OnCentre ? 1.0 : -1.0;
    // Each Gaussian normalized to unit volume, so the difference is ~zero-sum.
    std::vector<double> values = sample(size, [&](double dx, double dy) {
        const double r2 = dx * dx + dy * dy;
        return sign * (std::exp(-r2 / c2) / c2 - std::exp(-r2 / s2) / s2);
    });
    return zeroMeanFilter(size, std::move(values), gain);
}

Filter gabor(size_t size, float orientation, float wavelength, float sigma, float phase, float aspect, float gain)
{
    checkPositive(wavelength, "wavelength");
    checkPositive(sigma, "sigma");
    checkPositive(aspect, "aspect");
    const double c = std::cos(orientation), s = std::sin(orientation);
    const double s2 = 2.0 * static_cast<double>(sigma) * sigma, g2 = static_cast<double>(aspect) * aspect;
    std::vector<double> values = sample(size, [&](double dx, double dy) {
        const double along = dx * c + dy * s;    // along the stripes
        const double across = -dx * s + dy * c;  // across them: the carrier varies here
        return std::exp(-(across * across + g2 * along * along) / s2) *
               std::cos(2.0 * std::numbers::pi * across / wavelength + phase);
    });
    return zeroMeanFilter(size, std::move(values), gain);
}

std::vector<Filter> gaborBank(size_t size, size_t orientations, float wavelength, float sigma,
                              std::initializer_list<float> phases, float aspect, float gain)
{
    std::vector<Filter> bank;
    for (size_t o = 0; o < orientations; ++o) {
        const float orientation = std::numbers::pi_v<float> * static_cast<float>(o) / static_cast<float>(orientations);
        for (float phase : phases)
            bank.push_back(gabor(size, orientation, wavelength, sigma, phase, aspect, gain));
    }
    return bank;
}

std::vector<Filter> centreSurroundBank(size_t size, float centreSigma, float surroundSigma, float gain)
{
    return {differenceOfGaussians(size, centreSigma, surroundSigma, Polarity::OnCentre, gain),
            differenceOfGaussians(size, centreSigma, surroundSigma, Polarity::OffCentre, gain)};
}

namespace {

void checkFits(const conv2d& layer, const Filter& filter)
{
    if (layer.inputChannels() == 0)
        throw std::invalid_argument("filters::load: the layer must be wired to its input first");
    if (filter.height != layer.window().kernelHeight || filter.width != layer.window().kernelWidth ||
        filter.weights.size() != filter.height * filter.width)
        throw std::invalid_argument("filters::load: a " + std::to_string(filter.height) + " x " +
                                    std::to_string(filter.width) + " filter does not fit a " +
                                    std::to_string(layer.window().kernelHeight) + " x " +
                                    std::to_string(layer.window().kernelWidth) + " window");
}

} // namespace

void load(conv2d& layer, std::span<const Filter> bank)
{
    if (bank.size() != layer.outputChannels())
        throw std::invalid_argument("filters::load: " + std::to_string(bank.size()) + " filters for " +
                                    std::to_string(layer.outputChannels()) + " output channels");
    for (const Filter& filter : bank)
        checkFits(layer, filter);
    const size_t inputs = layer.inputChannels(), area = layer.window().area();
    std::vector<float> kernel(inputs * area);
    for (size_t k = 0; k < bank.size(); ++k) {
        for (size_t ch = 0; ch < inputs; ++ch)
            for (size_t i = 0; i < area; ++i)
                kernel[ch * area + i] = bank[k].weights[i] / static_cast<float>(inputs);
        layer.setKernel(k, kernel);
    }
}

void load(conv2d& layer, size_t channel, const Filter& filter, size_t input)
{
    checkFits(layer, filter);
    if (input >= layer.inputChannels())
        throw std::invalid_argument("filters::load: no input channel " + std::to_string(input));
    const size_t area = layer.window().area();
    std::vector<float> kernel(layer.windowSize(), 0.0f);
    for (size_t i = 0; i < area; ++i)
        kernel[input * area + i] = filter.weights[i];
    layer.setKernel(channel, kernel);
}

} // namespace exr::filters
