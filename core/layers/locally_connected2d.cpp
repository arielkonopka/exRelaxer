#include "locally_connected2d.hpp"
#include <stdexcept>
#include <string>
#include "../parallel.hpp"
#include "../random.hpp"

namespace exr {

locally_connected2d::locally_connected2d(size_t channels, const Window2D& window, bool hasHabituation, bool hasER,
                                         const Jitter& recoveryJitter, const Jitter& learningJitter,
                                         const Jitter& alphaJitter)
    : spatial_neuron_layer(channels, window, hasHabituation, hasER, recoveryJitter, learningJitter, alphaJitter)
{
}

void locally_connected2d::createWeights()
{
    // Drawn position after position, each position's rows in channel order.
    std::vector<float> w(outputChannels() * windowSize());
    weights_.clear();
    weights_.reserve(positions());
    for (size_t p = 0; p < positions(); ++p) {
        rng::drawWeights(rng::WeightStream::Initial, w);
        weights_.emplace_back(outputChannels(), windowSize(), w);
    }
    weightsChanged();
}

void locally_connected2d::appendInputs(size_t count)
{
    std::vector<float> w(outputChannels() * count);
    for (kernels::weight_matrix& m : weights_) {
        rng::drawWeights(rng::WeightStream::Growth, w);
        m.appendColumns(count, w);
    }
    weightsChanged();
}

void locally_connected2d::forward()
{
    if (!wired())
        return;
    beginForward();
    gatherInputs();
    traceSnapshot();
    const bool plain = plainForward();
    const size_t P = positions(), K = windowSize(), C = outputChannels();
    parallelChunks(P, kernels::threadsFor(P * K * C), [&](size_t p0, size_t p1) {
        std::vector<float> window(K), sums(weights_.empty() ? 0 : weights_[0].paddedRows());
        for (size_t p = p0; p < p1; ++p) {
            windowAt(p, window);
            weights_[p].multiply(window, sums, 0, weights_[p].blocks());
            for (size_t c = 0; c < C; ++c) {
                const size_t i = neuronAt(c, p);
                output_[i] = fire(i, sums[c], plain);
            }
        }
    });
}

void locally_connected2d::updateWeights()
{
    if (!wired())
        return;
    if (learnsFromSigns())
        gatherInputs();
    const size_t P = positions(), K = windowSize(), C = outputChannels();
    const bool scaled = scaledUpdates();
    parallelChunks(P, kernels::threadsFor(P * K * C), [&](size_t p0, size_t p1) {
        const size_t padded = weights_[0].paddedRows();
        std::vector<float> pre(K), delta(padded), keep(padded, 1.0f);
        std::vector<std::uint8_t> active(padded);
        for (size_t p = p0; p < p1; ++p) {
            bool any = false;
            std::ranges::fill(active, std::uint8_t{0});
            for (size_t c = 0; c < C; ++c) {
                const size_t i = neuronAt(c, p);
                if (step_active_[i]) {
                    active[c] = 1;
                    delta[c] = step_delta_[i];
                    keep[c] = scaled ? step_keep_[i] : 1.0f;
                    any = true;
                }
            }
            if (!any)
                continue;
            learningWindowAt(p, pre);
            if (scaled)
                weights_[p].learnScaled(pre, delta, keep, active, max_weight, 0, weights_[p].blocks());
            else
                weights_[p].learn(pre, delta, active, max_weight, 0, weights_[p].blocks());
        }
    });
}

std::vector<float> locally_connected2d::weights(size_t index) const
{
    std::vector<float> out;
    copyWeights(index, out);
    return out;
}

void locally_connected2d::copyWeights(size_t index, std::vector<float>& out) const
{
    if (!wired()) {
        out.clear();
        return;
    }
    out.resize(windowSize());
    weights_.at(index % positions()).copyRow(index / positions(), out);
}

void locally_connected2d::setWeights(size_t index, const std::vector<float>& weights)
{
    if (index >= size())
        throw std::out_of_range("locally_connected2d: no neuron " + std::to_string(index));
    if (weights.size() != windowSize())
        throw std::invalid_argument("locally_connected2d::setWeights: neuron reads " + std::to_string(windowSize()) +
                                    " inputs, got " + std::to_string(weights.size()) + " weights");
    storeWeights(index, weights);
}

void locally_connected2d::storeWeights(size_t index, std::span<const float> weights)
{
    weights_.at(index % positions()).setRow(index / positions(), weights);
    weightsChanged();
}

} // namespace exr
