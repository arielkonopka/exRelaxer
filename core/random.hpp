// The random streams the library draws from. One stream per purpose, so
// adding draws for one purpose never shifts the values of another; all of
// them are reset together by exr::reseed().
#pragma once
#include <cstdint>
#include <random>
#include <span>

namespace exr {

// Resets every random stream (initial weights, weights added by growth and by
// sensors, the seeds of each new neuron's spontaneous-firing generator, and
// per-neuron jitter). Call before building a network to make it independent
// of whatever ran earlier in the process.
void reseed(std::uint32_t seed);

namespace rng {

enum class WeightStream : std::uint8_t {
    Initial,  // a new wiring group's weights (join, addNeurons)
    Growth,   // weights appended when a group's inputs grow
    Sensor    // weights for attached sensors
};

// Fills `out` with uniform draws from [-1, 1], in order.
void drawWeights(WeightStream stream, std::span<float> out);

// Seed for a new neuron's own spontaneous-firing generator.
std::uint32_t spontaneousSeed();

// Stream for per-neuron jitter (see Jitter in neuron.hpp).
std::mt19937& jitter();

} // namespace rng
} // namespace exr
