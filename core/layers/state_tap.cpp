#include "state_tap.hpp"
#include <algorithm>
#include <stdexcept>
#include "neuron_layer.hpp"

namespace exr {

state_tap::state_tap(bool output, bool threshold, bool habituation)
    : output_field_(output), threshold_(threshold), habituation_(habituation)
{
    if (!output && !threshold && !habituation)
        throw std::invalid_argument("state tap: enable at least one of output, threshold and habituation");
}

void state_tap::join(layer& source)
{
    if (source_ != nullptr) {
        if (same(*source_, source))
            return;
        throw std::logic_error("state tap: reads one layer only");
    }
    if (!source.hasNeurons())
        throw std::logic_error("state tap: the source has no neurons");
    source_ = &dynamic_cast<const neuron_layer&>(source);
    readFrom(source);
    resizeOutput();
}

void state_tap::sourceGrew(const layer&, size_t, size_t)
{
    resizeOutput();
}

void state_tap::resizeOutput()
{
    const size_t old_size = output_.size();
    output_.resize(source_->neurons().size() * fields(), 0.0f);
    outputGrew(old_size);
}

void state_tap::forward()
{
    if (source_ == nullptr)
        return;
    const auto neurons = source_->neurons();
    const auto outputs = source_->output();
    const float onset = static_cast<float>(source_->habituationRule().onset());
    float* out = output_.data();
    for (size_t i = 0; i < neurons.size(); ++i) {
        const neuron& n = neurons[i];
        if (output_field_)
            *out++ = outputs[i];
        if (threshold_)
            *out++ = n.hasER() ? n.threshold() - n.restingThreshold() : 0.0f;
        if (habituation_)
            *out++ = n.hasHabituation() ? std::min(static_cast<float>(n.habituationStreak()) / onset, 1.0f) : 0.0f;
    }
}

void state_tap::serialize(std::ostream&) const {}

void state_tap::deserialize(std::istream&, DeserializeMode, std::uint32_t) {}

} // namespace exr
