#include "conv2d.hpp"
#include <algorithm>
#include <stdexcept>
#include <string>
#include "../kernels.hpp"
#include "../parallel.hpp"
#include "../random.hpp"

namespace exr {

conv2d::conv2d(size_t channels, const Window2D& window, bool hasHabituation, bool hasER,
               const Jitter& recoveryJitter, const Jitter& learningJitter, const Jitter& alphaJitter)
    : spatial_neuron_layer(channels, window, hasHabituation, hasER, recoveryJitter, learningJitter, alphaJitter)
{
}

std::span<const float> conv2d::kernelRow(size_t channel) const
{
    return std::span(kernels_).subspan(channel * windowSize(), windowSize());
}

std::vector<float> conv2d::kernel(size_t channel) const
{
    if (channel >= outputChannels())
        throw std::out_of_range("conv2d: no channel " + std::to_string(channel));
    if (!wired())
        return {};
    const std::span<const float> row = kernelRow(channel);
    return {row.begin(), row.end()};
}

void conv2d::setKernel(size_t channel, const std::vector<float>& weights)
{
    if (channel >= outputChannels())
        throw std::out_of_range("conv2d: no channel " + std::to_string(channel));
    if (weights.size() != windowSize())
        throw std::invalid_argument("conv2d::setKernel: the kernel has " + std::to_string(windowSize()) +
                                    " weights, got " + std::to_string(weights.size()));
    std::ranges::copy(weights, kernels_.begin() + static_cast<std::ptrdiff_t>(channel * windowSize()));
    weightsChanged();
}

void conv2d::createWeights()
{
    kernels_.resize(outputChannels() * windowSize());
    rng::drawWeights(rng::WeightStream::Initial, kernels_);
    weightsChanged();
}

float conv2d::squaredWeightNorm(size_t index) const
{
    if (!wired())
        return 0.0f;
    float s = 0.0f;
    for (float x : kernelRow(index / positions()))
        s += x * x;
    return s;
}

void conv2d::appendInputs(size_t count)
{
    // Each kernel gets `count` new weights at its end, drawn kernel after kernel.
    const size_t old_size = windowSize() - count;
    std::vector<float> added(outputChannels() * count);
    rng::drawWeights(rng::WeightStream::Growth, added);
    std::vector<float> grown;
    grown.reserve(outputChannels() * windowSize());
    for (size_t c = 0; c < outputChannels(); ++c) {
        const auto old_row = std::span(kernels_).subspan(c * old_size, old_size);
        grown.insert(grown.end(), old_row.begin(), old_row.end());
        const auto new_row = std::span(added).subspan(c * count, count);
        grown.insert(grown.end(), new_row.begin(), new_row.end());
    }
    kernels_ = std::move(grown);
    weightsChanged();
}

void conv2d::forward()
{
    if (!wired())
        return;
    beginForward();
    gatherInputs();
    traceSnapshot();
    const bool plain = plainForward();
    const size_t P = positions(), K = windowSize(), C = outputChannels();
    const size_t tiles = (P + tile_positions - 1) / tile_positions;
    parallelChunks(tiles, kernels::threadsFor(P * K * C), [&](size_t t0, size_t t1) {
        kernels::weight_matrix patches;  // one row per position of the tile
        std::vector<float> window(K), sums;
        for (size_t t = t0; t < t1; ++t) {
            const size_t first = t * tile_positions, rows = std::min(tile_positions, P - first);
            patches.reshape(rows, K);
            for (size_t r = 0; r < rows; ++r) {
                windowAt(first + r, window);
                patches.setRow(r, window);
            }
            sums.resize(patches.paddedRows());
            for (size_t c = 0; c < C; ++c) {
                patches.multiply(kernelRow(c), sums, 0, patches.blocks());
                for (size_t r = 0; r < rows; ++r) {
                    const size_t i = neuronAt(c, first + r);
                    output_[i] = fire(i, sums[r], plain);
                }
            }
        }
    });
}

void conv2d::updateWeights()
{
    if (!wired())
        return;
    const size_t P = positions(), K = windowSize(), C = outputChannels();
    const bool scaled = scaledUpdates();

    // Each neuron's step is step_delta_[neuronAt(c, p)] = step_delta_[c * P + p]
    // (0 when it does not learn); count the neurons that learn per channel
    // and (scaled rules) sum their shrink factors.
    const std::span<const float> delta = step_delta_;
    std::vector<size_t> eligible(C, 0);
    std::vector<float> keep_sum(C, 0.0f);
    parallelChunks(C, kernels::threadsFor(P * C * 16), [&](size_t c0, size_t c1) {
        for (size_t c = c0; c < c1; ++c) {
            size_t count = 0;  // local: neighbouring counters would share a cache line between threads
            float keep = 0.0f;
            for (size_t p = 0; p < P; ++p) {
                const size_t i = neuronAt(c, p);
                count += step_active_[i];
                if (scaled && step_active_[i])
                    keep += step_keep_[i];
            }
            eligible[c] = count;
            keep_sum[c] = keep;
        }
    });
    if (std::ranges::all_of(eligible, [](size_t e) { return e == 0; }))
        return;

    if (learnsFromSigns())
        gatherInputs();
    // Sum over positions of delta * pre, per kernel weight (pre: the signs
    // of the inputs, or their trace). Each tile
    // of positions is laid out transposed (a row per window input, a column
    // per position), so one multiply by a channel's deltas sums every input
    // over the tile's positions in order. Tiles add into a partial sum per
    // fixed chunk of positions, and chunks are reduced in order.
    const size_t chunks = (P + learning_chunk - 1) / learning_chunk;
    std::vector<float> partial(chunks * C * K, 0.0f);
    parallelChunks(chunks, kernels::threadsFor(P * K * C), [&](size_t c0, size_t c1) {
        kernels::weight_matrix signs;  // window inputs x positions of the tile
        std::vector<float> window(K), sums;
        for (size_t chunk = c0; chunk < c1; ++chunk) {
            const size_t chunk_end = std::min(P, (chunk + 1) * learning_chunk);
            for (size_t first = chunk * learning_chunk; first < chunk_end; first += tile_positions) {
                const size_t count = std::min(tile_positions, chunk_end - first);
                signs.reshape(K, count);
                for (size_t r = 0; r < count; ++r) {
                    learningWindowAt(first + r, window);
                    signs.setColumn(r, window);
                }
                sums.resize(signs.paddedRows());
                for (size_t c = 0; c < C; ++c) {
                    const std::span<const float> d = delta.subspan(c * P + first, count);
                    if (eligible[c] == 0 || std::ranges::all_of(d, [](float v) { return v == 0.0f; }))
                        continue;
                    signs.multiply(d, sums, 0, signs.blocks());
                    const std::span<float> acc = std::span(partial).subspan((chunk * C + c) * K, K);
                    for (size_t j = 0; j < K; ++j)
                        acc[j] += sums[j];
                }
            }
        }
    });

    // Each weight sums its chunks in chunk order, whichever thread does it.
    parallelChunks(C, kernels::threadsFor(chunks * C * K), [&](size_t c0, size_t c1) {
        for (size_t c = c0; c < c1; ++c) {
            if (eligible[c] == 0)
                continue;
            const float scale = 1.0f / static_cast<float>(eligible[c]);
            // Scaled rules shrink the shared kernel by its neurons' mean shrink factor.
            const float keep = scaled ? keep_sum[c] * scale : 1.0f;
            for (size_t j = 0; j < K; ++j) {
                float total = 0.0f;
                for (size_t chunk = 0; chunk < chunks; ++chunk)
                    total += partial[(chunk * C + c) * K + j];
                float& w = kernels_[c * K + j];
                const float kept = scaled ? w * keep : w;
                w = std::min(std::max(kept + total * scale, -max_weight), max_weight);
            }
        }
    });
}

} // namespace exr
