#include "cochlea.hpp"
#include <algorithm>
#include <bit>
#include <cmath>
#include <numbers>
#include <stdexcept>
#include <string>
#include "../binary_io.hpp"

namespace exr {
namespace {

double toMel(double hz) { return 2595.0 * std::log10(1.0 + hz / 700.0); }
double fromMel(double mel) { return 700.0 * (std::pow(10.0, mel / 2595.0) - 1.0); }

CochleaSpec validated(CochleaSpec spec)
{
    if (!(spec.sampleRate > 0.0f))
        throw std::invalid_argument("cochlea: sampleRate must be > 0");
    if (spec.hop == 0)
        throw std::invalid_argument("cochlea: hop must be at least 1");
    if (spec.window < 2 || !std::has_single_bit(spec.window) || spec.window < spec.hop)
        throw std::invalid_argument("cochlea: window must be a power of two >= 2 and >= hop, got " +
                                    std::to_string(spec.window));
    if (spec.bands == 0)
        throw std::invalid_argument("cochlea: bands must be at least 1");
    const float nyquist = spec.sampleRate / 2.0f;
    if (spec.maxFrequency == 0.0f)
        spec.maxFrequency = nyquist;
    if (!(spec.minFrequency >= 0.0f && spec.minFrequency < spec.maxFrequency && spec.maxFrequency <= nyquist))
        throw std::invalid_argument("cochlea: need 0 <= minFrequency < maxFrequency <= sampleRate / 2");
    if (!(spec.gain > 0.0f))
        throw std::invalid_argument("cochlea: gain must be > 0");
    if (spec.scale > FrequencyScale::Linear)
        throw std::invalid_argument("cochlea: unknown frequency scale");
    if (spec.compression > Compression::Linear)
        throw std::invalid_argument("cochlea: unknown compression");
    return spec;
}

} // namespace

cochlea::cochlea(const CochleaSpec& spec, bool hasHabituation, bool hasER, const Jitter& recoveryJitter,
                 const Jitter& learningJitter, const Jitter& alphaJitter)
    : neuron_layer(validated(spec).bands, hasHabituation, hasER, recoveryJitter, learningJitter, alphaJitter),
      spec_(validated(spec))
{
    const size_t N = spec_.window, bins = N / 2 + 1;

    // Hann window, and the power normalisation: a sine of amplitude A on a
    // bin gives |X| = A * sum(w) / 2 there.
    hann_.resize(N);
    double sum = 0.0;
    for (size_t n = 0; n < N; ++n) {
        const double w = 0.5 - 0.5 * std::cos(2.0 * std::numbers::pi * static_cast<double>(n) / static_cast<double>(N));
        hann_[n] = static_cast<float>(w);
        sum += w;
    }
    power_scale_ = static_cast<float>((2.0 / sum) * (2.0 / sum));

    // FFT tables.
    const int bits = std::countr_zero(N);
    reversed_.resize(N);
    for (size_t n = 0; n < N; ++n) {
        size_t r = 0;
        for (int b = 0; b < bits; ++b)
            r |= ((n >> b) & 1u) << (bits - 1 - b);
        reversed_[n] = r;
    }
    cos_.resize(N / 2);
    sin_.resize(N / 2);
    for (size_t k = 0; k < N / 2; ++k) {
        const double angle = -2.0 * std::numbers::pi * static_cast<double>(k) / static_cast<double>(N);
        cos_[k] = static_cast<float>(std::cos(angle));
        sin_[k] = static_cast<float>(std::sin(angle));
    }

    // Band edges: bands + 2 points evenly spaced on the scale.
    const bool mel = spec_.scale == FrequencyScale::Mel;
    const double lo = mel ? toMel(spec_.minFrequency) : spec_.minFrequency;
    const double hi = mel ? toMel(spec_.maxFrequency) : spec_.maxFrequency;
    std::vector<double> edges(spec_.bands + 2);
    for (size_t i = 0; i < edges.size(); ++i) {
        const double v = lo + (hi - lo) * static_cast<double>(i) / static_cast<double>(spec_.bands + 1);
        edges[i] = mel ? fromMel(v) : v;
    }
    const double bin_hz = static_cast<double>(spec_.sampleRate) / static_cast<double>(N);
    for (size_t b = 0; b < spec_.bands; ++b) {
        const double left = edges[b], centre = edges[b + 1], right = edges[b + 2];
        Band band{static_cast<float>(left), static_cast<float>(centre), static_cast<float>(right), 0, {}};
        std::vector<float> weights(bins, 0.0f);
        bool any = false;
        for (size_t k = 0; k < bins; ++k) {
            const double f = static_cast<double>(k) * bin_hz;
            double w = 0.0;
            if (f > left && f <= centre)
                w = (f - left) / (centre - left);
            else if (f > centre && f < right)
                w = (right - f) / (right - centre);
            weights[k] = static_cast<float>(w);
            any = any || w > 0.0;
        }
        // A band narrower than a bin still hears its nearest bin.
        if (!any)
            weights[std::min(bins - 1, static_cast<size_t>(std::lround(centre / bin_hz)))] = 1.0f;
        const auto first = std::ranges::find_if(weights, [](float w) { return w > 0.0f; });
        const auto last = std::find_if(weights.rbegin(), weights.rend(), [](float w) { return w > 0.0f; }).base();
        band.firstBin = static_cast<size_t>(first - weights.begin());
        band.weights.assign(first, last);
        bands_.push_back(std::move(band));
    }

    samples_.assign(N, 0.0f);
    re_.resize(N);
    im_.resize(N);
    power_.assign(bins, 0.0f);
}

void cochlea::attachInputs(const InputRange& sensors)
{
    if (sensors_)
        throw std::logic_error("cochlea: it already reads its sound");
    if (sensors.count != spec_.hop)
        throw std::invalid_argument("cochlea: hop is " + std::to_string(spec_.hop) + " samples, got " +
                                    std::to_string(sensors.count) + " sensors");
    sensors_ = sensors;
}

void cochlea::fft()
{
    const size_t N = spec_.window;
    for (size_t n = 0; n < N; ++n)
        if (n < reversed_[n]) {
            std::swap(re_[n], re_[reversed_[n]]);
            std::swap(im_[n], im_[reversed_[n]]);
        }
    for (size_t len = 2; len <= N; len <<= 1) {
        const size_t half = len / 2, step = N / len;
        for (size_t start = 0; start < N; start += len)
            for (size_t j = 0; j < half; ++j) {
                const float wr = cos_[j * step], wi = sin_[j * step];
                const size_t a = start + j, b = a + half;
                const float tr = re_[b] * wr - im_[b] * wi;
                const float ti = re_[b] * wi + im_[b] * wr;
                re_[b] = re_[a] - tr;
                im_[b] = im_[a] - ti;
                re_[a] += tr;
                im_[a] += ti;
            }
    }
}

void cochlea::forward()
{
    if (!sensors_)
        return;
    const size_t N = spec_.window, hop = spec_.hop;
    std::copy(samples_.begin() + static_cast<std::ptrdiff_t>(hop), samples_.end(), samples_.begin());
    const std::span<const float> sound = sensors_->values();
    std::ranges::copy(sound, samples_.end() - static_cast<std::ptrdiff_t>(hop));

    for (size_t n = 0; n < N; ++n) {
        re_[n] = samples_[n] * hann_[n];
        im_[n] = 0.0f;
    }
    fft();
    for (size_t k = 0; k < power_.size(); ++k)
        power_[k] = (re_[k] * re_[k] + im_[k] * im_[k]) * power_scale_;

    for (size_t b = 0; b < bands_.size(); ++b) {
        const Band& band = bands_[b];
        float energy = 0.0f;
        for (size_t i = 0; i < band.weights.size(); ++i)
            energy += band.weights[i] * power_[band.firstBin + i];
        const float value = spec_.compression == Compression::Log ? std::log1p(spec_.gain * energy)
                                                                  : spec_.gain * energy;
        output_[b] = neurons_[b].activate(value);
    }
}

void cochlea::serialize(std::ostream& os) const
{
    neuron_layer::serialize(os);
    binary_io::write(os, static_cast<std::uint64_t>(samples_.size()));
    binary_io::write(os, std::span<const float>(samples_));
}

void cochlea::deserialize(std::istream& is, DeserializeMode mode, std::uint32_t neuronFormat)
{
    neuron_layer::deserialize(is, mode, neuronFormat);
    if (!is)
        return;  // the caller reports the truncated stream
    const auto count = binary_io::read<std::uint64_t>(is);
    if (!is)
        return;
    if (count != samples_.size())
        throw std::runtime_error("cochlea deserialize: data has " + std::to_string(count) +
                                 " samples, the window is " + std::to_string(samples_.size()));
    std::vector<float> values(samples_.size());
    binary_io::read(is, std::span<float>(values));
    if (!is)
        return;
    if (mode == DeserializeMode::FullState)
        samples_ = std::move(values);
    else
        std::ranges::fill(samples_, 0.0f);
}

} // namespace exr
