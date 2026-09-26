#include "dense.hpp"
#include "../parallel.hpp"
#include <algorithm>
#include <stdexcept>
#include <string>

namespace exr {
// --- Group ------------------------------------------------------------------

bool dense::Group::reads(const layer& source) const
{
    return std::ranges::any_of(sources, [&source](const layer& s) { return same(s, source); });
}

void dense::Group::append(const InputRange& range)
{
    if (range.count == 0)
        return;
    // Consecutive runs of the same buffer merge, so a gather copies them at once.
    if (!inputs.empty() && inputs.back().sameBuffer(range) &&
        inputs.back().offset + inputs.back().count == range.offset) {
        inputs.back().count += range.count;
        return;
    }
    inputs.push_back(range);
}

std::span<const float> dense::Group::gather()
{
    // Copying first makes the dot products read memory sequentially, and
    // makes a group reading its own layer see the values from before it ran.
    values.resize(weights.cols());
    auto out = values.begin();
    for (const InputRange& range : inputs)
        out = std::ranges::copy(range.values(), out).out;
    return values;
}

// --- dense ------------------------------------------------------------------

dense::dense(size_t count, bool hasHabituation, bool hasER, const Jitter& recoveryJitter,
             const Jitter& learningJitter, const Jitter& alphaJitter)
    : neuron_layer(count, hasHabituation, hasER, recoveryJitter, learningJitter, alphaJitter),
      group_of_(count, no_group)
{
}

void dense::addGroup(Group group)
{
    const NeuronRange range = group.neurons;
    if (range.first + range.count > neurons_.size())
        throw std::logic_error("dense: wiring group past the last neuron");
    // A neuron has one weight row, laid out over its group's pool; a second
    // group would give it a second, conflicting one. Checked before anything
    // changes, so a refused group leaves the layer as it was.
    for (size_t i = range.first; i < range.first + range.count; ++i)
        if (group_of_[i] != no_group)
            throw std::logic_error("dense: neuron " + std::to_string(i) + " is already in wiring group " +
                                   std::to_string(group_of_[i]) + "; a neuron belongs to one group");
    for (size_t i = range.first; i < range.first + range.count; ++i)
        group_of_[i] = groups_.size();
    groups_.push_back(std::move(group));
}

void dense::appendToGroups(const InputRange& range, Source source, rng::WeightStream stream)
{
    for (Group& group : groups_) {
        if (source && group.reads(*source))
            continue;
        // Drawn row after row: each neuron's new weights in turn.
        std::vector<float> w(group.neurons.count * range.count);
        rng::drawWeights(stream, w);
        group.weights.appendColumns(range.count, w);
        group.append(range);
        if (source)
            group.sources.push_back(*source);
    }
}

void dense::groupUnwired(const InputRange& range, Source source, rng::WeightStream stream)
{
    // Unwired neurons are always one contiguous run: the neurons a layer was
    // constructed with (feedback neurons are wired when they are created).
    const auto first = std::find(group_of_.begin(), group_of_.end(), no_group);
    if (first == group_of_.end())
        return;
    const auto last = std::find_if(first, group_of_.end(), [](size_t g) { return g != no_group; });
    if (std::find(last, group_of_.end(), no_group) != group_of_.end())
        throw std::logic_error("dense: unwired neurons are not contiguous");

    Group group;
    group.neurons = {static_cast<size_t>(first - group_of_.begin()), static_cast<size_t>(last - first)};
    std::vector<float> w(group.neurons.count * range.count);
    rng::drawWeights(stream, w);
    group.weights = kernels::weight_matrix(group.neurons.count, range.count, w);
    group.append(range);
    if (source)
        group.sources.push_back(*source);
    addGroup(std::move(group));
}

void dense::join(layer& source)
{
    // A neuron belongs to exactly one group, so wired neurons get `source`
    // appended to their group's pool instead of a second group: they
    // integrate all their sources in one weighted sum.
    const InputRange range(source.outputBuffer(), 0, source.size());
    appendToGroups(range, source, rng::WeightStream::Growth);
    groupUnwired(range, source, rng::WeightStream::Initial);
    readFrom(source);
}

void dense::attachInputs(const InputRange& sensors)
{
    // Same rule as join(). Sensors are not a layer, so they never grow.
    if (sensors.count == 0)
        throw std::invalid_argument("dense: attachInputs needs at least one sensor");
    appendToGroups(sensors, std::nullopt, rng::WeightStream::Sensor);
    groupUnwired(sensors, std::nullopt, rng::WeightStream::Sensor);
}

void dense::addNeurons(size_t count, layer& source)
{
    // The new neurons read source's output as it is now; if source is this
    // layer, the growth notification below extends them to their own outputs.
    const InputRange range(source.outputBuffer(), 0, source.size());
    const size_t first = neurons_.size();
    for (size_t i = 0; i < count; ++i) {
        newNeuron();
        group_of_.push_back(no_group);
    }
    if (count > 0) {
        Group group;
        group.neurons = {first, count};
        std::vector<float> w(count * range.count);
        rng::drawWeights(rng::WeightStream::Initial, w);
        group.weights = kernels::weight_matrix(count, range.count, w);
        group.append(range);
        group.sources.push_back(source);
        addGroup(std::move(group));
    }
    readFrom(source);
    outputGrew(first);
}

void dense::sourceGrew(const layer& source, size_t offset, size_t count)
{
    // Appending keeps each neuron's weights aligned with its pool even when
    // the group reads several sources: the new entries go at the end of both.
    const InputRange range(source.outputBuffer(), offset, count);
    for (Group& group : groups_) {
        if (!group.reads(source))
            continue;
        std::vector<float> w(group.neurons.count * count);
        rng::drawWeights(rng::WeightStream::Growth, w);
        group.weights.appendColumns(count, w);
        group.append(range);
    }
}

void dense::forward()
{
    // Groups run one after another; a later group of this layer sees the
    // outputs earlier groups just wrote.
    for (Group& group : groups_) {
        const std::span<const float> x = group.gather();
        group.scratch.resize(group.weights.paddedRows());
        const size_t first = group.neurons.first, count = group.neurons.count;
        parallelChunks(group.weights.blocks(), kernels::threadsFor(count * x.size()), [&](size_t b0, size_t b1) {
            group.weights.multiply(x, group.scratch, b0, b1);
            const size_t end = std::min(b1 * kernels::weight_matrix::lanes, count);
            for (size_t r = b0 * kernels::weight_matrix::lanes; r < end; ++r)
                output_[first + r] = neurons_[first + r].activate(group.scratch[r]);
        });
    }
}

void dense::applyReward(float reward, float learningRate)
{
    for (Group& group : groups_) {
        // The inputs as they are now; only their signs matter.
        group.gather();
        kernels::signs(group.values, group.values);
        const size_t first = group.neurons.first, count = group.neurons.count;
        group.scratch.assign(group.weights.paddedRows(), 0.0f);
        group.active.assign(group.weights.paddedRows(), 0);
        bool any = false;
        for (size_t r = 0; r < count; ++r) {
            const neuron& n = neurons_[first + r];
            if (n.eligible()) {
                group.active[r] = 1;
                group.scratch[r] = n.learningDelta(reward, learningRate);
                any = true;
            }
        }
        if (!any)
            continue;
        parallelChunks(group.weights.blocks(), kernels::threadsFor(count * group.values.size()), [&](size_t b0, size_t b1) {
            group.weights.learn(group.values, group.scratch, group.active, max_weight, b0, b1);
        });
    }
}

std::vector<float> dense::weights(size_t index) const
{
    std::vector<float> out;
    copyWeights(index, out);
    return out;
}

void dense::copyWeights(size_t index, std::vector<float>& out) const
{
    const size_t g = group_of_.at(index);
    if (g == no_group) {
        out.clear();
        return;
    }
    out.resize(groups_[g].weights.cols());
    groups_[g].weights.copyRow(index - groups_[g].neurons.first, out);
}

void dense::setWeights(size_t index, const std::vector<float>& weights)
{
    if (weights.size() != inputCount(index))
        throw std::invalid_argument("dense::setWeights: neuron " + std::to_string(index) + " reads " +
                                    std::to_string(inputCount(index)) + " inputs, got " +
                                    std::to_string(weights.size()) + " weights");
    if (!weights.empty())
        storeWeights(index, weights);
}

void dense::storeWeights(size_t index, std::span<const float> weights)
{
    Group& group = groups_[group_of_.at(index)];
    group.weights.setRow(index - group.neurons.first, weights);
}

size_t dense::inputCount(size_t index) const
{
    const size_t g = group_of_.at(index);
    return g == no_group ? 0 : groups_[g].weights.cols();
}

dense::NeuronRange dense::groupNeurons(size_t group) const
{
    return groups_.at(group).neurons;
}

} // namespace exr
