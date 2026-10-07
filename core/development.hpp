// Structural development (doc/development.md): watching a population's
// activity to decide when it should grow or lose neurons. The growth and
// pruning themselves are network calls (growLayer, pruneNeurons,
// pruneNewest, pruneCandidates); what to do when is up to the caller.
#pragma once
#include <cstddef>
#include <cstdint>
#include <vector>
#include "layers/neuron_layer.hpp"

namespace exr {

// How an activity_monitor reads a tick. Everything comes from the neurons'
// own E-R state (neuron::lastSum, lastThreshold, lastFiring); nothing here
// is a second threshold.
struct ActivitySpec
{
    float inputEpsilon = 1e-3f;  // a raw sum with |sum| at or below this is "no input"
    float fatigueRatio = 1.0f;   // a threshold above fatigueRatio * resting threshold is raised (fatigue)
    size_t window = 100;         // saturation is judged over the last `window` ticks; >= 1
    float saturatedShare = 0.9f; // ... and needs at least this share of them saturated; (0, 1]
    float blockedShare = 0.5f;   // a tick is saturated when no neuron fired on its input and at least
                                 // this share of the neurons was fatigued; (0, 1]
    void validate() const;       // throws std::invalid_argument
};

// What one neuron did on one tick.
enum class TickState : std::uint8_t
{
    Fired,        // fired on its input (|sum| > inputEpsilon)
    Spontaneous,  // fired on its own (E-R spontaneous firing): counts as silent
    Fatigued,     // silent: input present, but below a threshold raised by earlier firing (E-R)
    Habituated,   // silent: the raw input would have fired it, habituation suppressed it
    Weak,         // silent: input present, threshold not raised; the input is too weak
    NoInput       // |sum| <= inputEpsilon (silent, or fired on a vanishing input after E-R relaxed)
};

// Watches one layer of neurons tick by tick (call observe() after every
// step). Per neuron it counts how long the neuron has gone without firing on
// its input, not counting ticks it was fatigued or habituated (temporary
// silence); per population it keeps, over a sliding window, the ticks on
// which the population was saturated: nobody fired, input was present, and
// thresholds were raised above it. A single silent tick is never
// saturation, and fatigue that recovers within the window is not either.
//
// The monitor starts again (reset) when the layer's size changes, e.g. after
// growth or pruning, so new neurons are judged from their first tick.
// Not saved with the network.
class activity_monitor
{
public:
    explicit activity_monitor(const ActivitySpec& spec = {});

    const ActivitySpec& spec() const { return spec_; }
    void observe(const neuron_layer& layer);
    void reset();

    size_t size() const { return state_.size(); }
    size_t ticks() const { return ticks_; }  // observed since the last reset

    // --- Per neuron -----------------------------------------------------
    TickState state(size_t index) const { return state_.at(index); }
    // Ticks without firing on its input since the last such firing (or the
    // reset); fatigued and habituated ticks are not counted.
    size_t inactiveTicks(size_t index) const { return inactive_.at(index); }
    size_t firings(size_t index) const { return firings_.at(index); }  // since the reset

    // --- Population -----------------------------------------------------
    bool saturatedTick() const { return last_saturated_; }  // the last tick
    size_t saturatedTicks() const { return saturated_count_; }  // in the window
    // saturatedTicks() / ticks in the window so far (0 before any tick).
    float saturation() const;
    // The window is full and at least saturatedShare of it was saturated:
    // a candidate trigger for width growth.
    bool saturated() const;

private:
    ActivitySpec spec_;
    std::vector<TickState> state_;
    std::vector<size_t> inactive_, firings_;
    std::vector<std::uint8_t> window_;  // ring buffer: 1 = saturated tick
    size_t next_ = 0, filled_ = 0, saturated_count_ = 0, ticks_ = 0;
    bool last_saturated_ = false;
};

} // namespace exr
