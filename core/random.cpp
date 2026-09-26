#include "random.hpp"
#include <iterator>

namespace exr {
namespace {

// Kept at file scope (not function-local statics) so reseed() can reset them.
std::mt19937 initial_generator(12345);
std::mt19937 growth_generator(67890);
std::mt19937 sensor_generator(12345);
std::mt19937 spontaneous_generator(54321);
std::mt19937 jitter_generator(24680);
std::mt19937 learning_generator(97531);

std::mt19937& generatorOf(rng::WeightStream stream)
{
    switch (stream) {
    case rng::WeightStream::Initial: return initial_generator;
    case rng::WeightStream::Growth: return growth_generator;
    case rng::WeightStream::Sensor: break;
    }
    return sensor_generator;
}

} // namespace

void reseed(std::uint32_t seed)
{
    std::seed_seq seq{seed};
    std::uint32_t seeds[4];
    seq.generate(std::begin(seeds), std::end(seeds));
    initial_generator.seed(seeds[0]);
    growth_generator.seed(seeds[1]);
    sensor_generator.seed(seeds[2]);
    spontaneous_generator.seed(seeds[3]);
    // Separate seed sequence: adding this stream must not change the four
    // above (seed_seq output depends on how many values are generated).
    std::seed_seq jitter_seq{seed, 0x6a09e667u};
    std::uint32_t jitter_seed;
    jitter_seq.generate(&jitter_seed, &jitter_seed + 1);
    jitter_generator.seed(jitter_seed);
    std::seed_seq learning_seq{seed, 0xbb67ae85u};
    std::uint32_t learning_seed;
    learning_seq.generate(&learning_seed, &learning_seed + 1);
    learning_generator.seed(learning_seed);
}

namespace rng {

void drawWeights(WeightStream stream, std::span<float> out)
{
    std::uniform_real_distribution<float> distribution(-1.0f, 1.0f);
    std::mt19937& generator = generatorOf(stream);
    for (float& w : out)
        w = distribution(generator);
}

std::uint32_t spontaneousSeed() { return spontaneous_generator(); }

std::mt19937& jitter() { return jitter_generator; }

std::mt19937& learning() { return learning_generator; }

} // namespace rng
} // namespace exr
