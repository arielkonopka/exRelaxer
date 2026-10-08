#include "development.hpp"
#include <cmath>
#include <stdexcept>

namespace exr {

void ActivitySpec::validate() const
{
    if (!(inputEpsilon >= 0.0f) || !std::isfinite(inputEpsilon))
        throw std::invalid_argument("ActivitySpec: inputEpsilon must be finite and >= 0");
    if (!(fatigueRatio >= 0.0f) || !std::isfinite(fatigueRatio))
        throw std::invalid_argument("ActivitySpec: fatigueRatio must be finite and >= 0");
    if (window == 0)
        throw std::invalid_argument("ActivitySpec: window must be >= 1");
    if (!(saturatedShare > 0.0f && saturatedShare <= 1.0f) || !(blockedShare > 0.0f && blockedShare <= 1.0f))
        throw std::invalid_argument("ActivitySpec: saturatedShare and blockedShare must be in (0, 1]");
}

activity_monitor::activity_monitor(const ActivitySpec& spec)
    : spec_(spec)
{
    spec_.validate();
    window_.assign(spec_.window, 0);
}

void activity_monitor::reset()
{
    std::fill(inactive_.begin(), inactive_.end(), 0);
    std::fill(firings_.begin(), firings_.end(), 0);
    std::fill(state_.begin(), state_.end(), TickState::NoInput);
    std::fill(window_.begin(), window_.end(), 0);
    next_ = filled_ = saturated_count_ = ticks_ = 0;
    last_saturated_ = false;
}

namespace {

TickState classify(const neuron& n, const ActivitySpec& spec)
{
    const float sum = std::abs(n.lastSum());
    switch (n.lastFiring()) {
    case neuron::Firing::Fired:
        // A relaxed E-R threshold lets even a vanishing input through:
        // firing on no input is not activity.
        return sum > spec.inputEpsilon ? TickState::Fired : TickState::NoInput;
    case neuron::Firing::Spontaneous:
        return TickState::Spontaneous;
    case neuron::Firing::Silent:
        break;
    }
    if (!(sum > spec.inputEpsilon))
        return TickState::NoInput;
    if (!n.hasER())
        return TickState::Weak;  // below the gate (or rectified away): fixed, not fatigue
    const float thr = n.lastThreshold();
    if (sum > thr)
        return TickState::Habituated;  // E-R would have fired on the raw sum
    return thr > spec.fatigueRatio * n.restingThreshold() ? TickState::Fatigued : TickState::Weak;
}

} // namespace

void activity_monitor::observe(const neuron_layer& layer)
{
    const auto neurons = layer.neurons();
    if (neurons.size() != state_.size()) {
        state_.assign(neurons.size(), TickState::NoInput);
        inactive_.assign(neurons.size(), 0);
        firings_.assign(neurons.size(), 0);
        reset();
    }
    size_t fired = 0, fatigued = 0;
    for (size_t i = 0; i < neurons.size(); ++i) {
        const TickState s = classify(neurons[i], spec_);
        state_[i] = s;
        switch (s) {
        case TickState::Fired:
            ++fired;
            ++firings_[i];
            inactive_[i] = 0;
            break;
        case TickState::Fatigued:
            ++fatigued;
            break;  // temporary: neither counted nor reset
        case TickState::Habituated:
            break;
        case TickState::Spontaneous:
        case TickState::Weak:
        case TickState::NoInput:
            ++inactive_[i];
            break;
        }
    }
    last_saturated_ = !neurons.empty() && fired == 0 &&
                      static_cast<float>(fatigued) >= spec_.blockedShare * static_cast<float>(neurons.size());
    if (filled_ == window_.size())
        saturated_count_ -= window_[next_];
    else
        ++filled_;
    window_[next_] = last_saturated_ ? 1 : 0;
    saturated_count_ += window_[next_];
    next_ = (next_ + 1) % window_.size();
    ++ticks_;
}

float activity_monitor::saturation() const
{
    return filled_ == 0 ? 0.0f : static_cast<float>(saturated_count_) / static_cast<float>(filled_);
}

bool activity_monitor::saturated() const
{
    return filled_ == window_.size() &&
           static_cast<float>(saturated_count_) >= spec_.saturatedShare * static_cast<float>(window_.size());
}

} // namespace exr
