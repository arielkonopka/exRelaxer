#pragma once
// Test inputs and timings expressed relative to the E-R constants in
// neuron.hpp, so that tuning those constants does not silently change what a
// test means (e.g. a "weak" stimulus becoming too weak to ever fire).
#include <cmath>
#include "../core/neuron.hpp"

// A weak stimulus: a few times the resting E-R threshold. Scaled
// multiplicatively, so its ratio to the threshold (what E-R responds to)
// stays the same when baseline_threshold changes. 0.5 at baseline 0.1.
inline const float WEAK_STIMULUS = 5.0f * baseline_threshold;

// Silent ticks until a threshold starting at `fromThreshold` has relaxed to
// min_threshold, i.e. until the neuron fires spontaneously.
inline int ticksUntilSpontaneousFiring(float fromThreshold)
{
    return static_cast<int>(std::ceil(std::log(min_threshold / fromThreshold) / std::log(recovery_factor)));
}

// Upper bound on an E-R threshold in practice: a firing can only raise the
// threshold while |output| > threshold, and outputs are clamped to max_output.
inline const float MAX_PRACTICAL_THRESHOLD = 2.0f * max_output;
