#pragma once
// E-R options shared by experiments: the threshold growth rule by name.
#include <stdexcept>
#include <string>
#include "neuron.hpp"

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

} // namespace er_options
