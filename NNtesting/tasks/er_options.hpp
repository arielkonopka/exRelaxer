#pragma once
// E-R options shared by experiments: the threshold growth rule by name.
#include <stdexcept>
#include <string>
#include "neuron.hpp"
#include "layers/neuron_layer.hpp"

namespace er_options {

// "linear" (the default), "log" (the original), "fixed" or "multiplicative";
// `amount` is used by all but log (see exr::ThresholdGrowth).
inline exr::ThresholdGrowth thresholdGrowth(const std::string& rule, double amount)
{
    using Rule = exr::ThresholdGrowth::Rule;
    exr::ThresholdGrowth g;
    g.amount = static_cast<float>(amount);
    if (rule == "log")
        g.rule = Rule::Log;
    else if (rule == "linear")
        g.rule = Rule::Linear;
    else if (rule == "fixed")
        g.rule = Rule::Fixed;
    else if (rule == "multiplicative")
        g.rule = Rule::Multiplicative;
    else
        throw std::invalid_argument("growth must be log, linear, fixed or multiplicative");
    if (!g.valid())
        throw std::invalid_argument("growth_amount must be in [0, 1e6]");
    return g;
}

inline exr::Spontaneous spontaneous(double below, double amplitude, double rate)
{
    const exr::Spontaneous s{static_cast<float>(below), static_cast<float>(amplitude), static_cast<float>(rate)};
    if (!s.valid())
        throw std::invalid_argument("spontaneous_below and _amplitude in [0, max_output], spontaneous_rate in [0, 1]");
    return s;
}

// Mean 1 / |w| over a layer's neurons: how much its normalised sums shrink
// relative to raw ones (1 when the layer is not normalised).
inline float meanInverseNorm(exr::neuron_layer& layer)
{
    if (!layer.normalized() || layer.size() == 0)
        return 1.0f;
    double sum = 0.0;
    for (size_t i = 0; i < layer.size(); ++i)
        sum += layer.inverseNorm(i);
    return static_cast<float>(sum / static_cast<double>(layer.size()));
}

// The `resting_threshold` option: a number, or "auto" for the default resting
// threshold scaled by meanInverseNorm(layer), so a normalised layer's E-R
// thresholds keep their size relative to its (smaller) sums. Returns the
// factor applied to the default (1 for a number equal to the default).
inline float calibrateRestingThreshold(exr::neuron_layer& layer, const std::string& setting)
{
    const float factor = setting == "auto" ? meanInverseNorm(layer) : std::stof(setting) / exr::baseline_threshold;
    layer.setRestingThreshold(exr::baseline_threshold * factor);
    return factor;
}

} // namespace er_options
