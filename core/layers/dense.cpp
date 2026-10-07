#include "dense.hpp"
#include "../binary_io.hpp"
#include "../parallel.hpp"
#include <algorithm>
#include <istream>
#include <ostream>
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
    weightsChanged();
}

void dense::initWeights(rng::WeightStream stream, WeightInit init, std::span<float> w)
{
    if (init == WeightInit::Zero)
        std::fill(w.begin(), w.end(), 0.0f);
    else
        rng::drawWeights(stream, w);
}

void dense::appendToGroups(const InputRange& range, Source source, rng::WeightStream stream)
{
    for (Group& group : groups_) {
        if (source && group.reads(*source))
            continue;
        // Drawn row after row: each neuron's new weights in turn.
        std::vector<float> w(group.neurons.count * range.count);
        initWeights(stream, incomingInit(), w);
        group.weights.appendColumns(range.count, w);
        group.append(range);
        if (!group.frozen.empty())
            group.frozen.resize(group.weights.cols(), 0);
        if (source)
            group.sources.push_back(*source);
    }
    history_.clear();
    weightsChanged();
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
    initWeights(stream, incomingInit(), w);
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
        initWeights(rng::WeightStream::Initial, incomingInit(), w);
        group.weights = kernels::weight_matrix(count, range.count, w);
        group.append(range);
        group.sources.push_back(source);
        addGroup(std::move(group));
    }
    readFrom(source);
    history_.clear();
    outputGrew(first);
}

void dense::growNeurons(size_t count, size_t group)
{
    if (group >= groups_.size())
        throw std::out_of_range("dense::growNeurons: no wiring group " + std::to_string(group) + " (the layer has " +
                                std::to_string(groups_.size()) + ")");
    if (count == 0)
        return;
    const size_t first = neurons_.size();
    Group grown;
    grown.inputs = groups_[group].inputs;
    grown.sources = groups_[group].sources;
    grown.neurons = {first, count};
    std::vector<float> w(count * groups_[group].weights.cols());
    initWeights(rng::WeightStream::Initial, incomingInit(), w);
    grown.weights = kernels::weight_matrix(count, groups_[group].weights.cols(), w);
    for (size_t i = 0; i < count; ++i) {
        newNeuron();
        group_of_.push_back(no_group);
    }
    addGroup(std::move(grown));
    history_.clear();
    // Readers (this layer too, when the group reads it) get the new outputs.
    outputGrew(first);
}

void dense::removeNeurons(std::span<const size_t> indices)
{
    for (size_t k = 0; k < indices.size(); ++k)
        if (indices[k] >= neurons_.size() || (k > 0 && indices[k] <= indices[k - 1]))
            throw std::invalid_argument("dense::removeNeurons: indices must be sorted, unique and below " +
                                        std::to_string(neurons_.size()));
    if (indices.empty())
        return;
    if (!readersFollowShrinking())
        throw std::logic_error("dense::removeNeurons: a layer reading this one cannot lose inputs (only Dense layers can)");

    // Each group loses its own rows; groups left empty are dropped.
    std::vector<Group> kept;
    size_t removed_before = 0;
    for (Group& group : groups_) {
        const size_t first = group.neurons.first, end = first + group.neurons.count;
        std::vector<size_t> rows;
        for (size_t i : indices)
            if (i >= first && i < end)
                rows.push_back(i - first);
        group.weights.removeRows(rows);
        for (kernels::weight_matrix* m : {&group.elig, &group.adapt, &group.grad})
            if (m->rows() == group.neurons.count)
                m->removeRows(rows);
        group.neurons = {first - removed_before, group.neurons.count - rows.size()};
        removed_before += rows.size();
        if (group.neurons.count > 0)
            kept.push_back(std::move(group));
    }
    // Unwired neurons keep no_group; the rest point at their (renumbered) group.
    std::vector<size_t> group_of;
    for (size_t i = 0, next = 0; i < group_of_.size(); ++i) {
        if (next < indices.size() && indices[next] == i) {
            ++next;
            continue;
        }
        group_of.push_back(no_group);
    }
    groups_ = std::move(kept);
    for (size_t g = 0; g < groups_.size(); ++g)
        for (size_t i = groups_[g].neurons.first; i < groups_[g].neurons.first + groups_[g].neurons.count; ++i)
            group_of[i] = g;
    group_of_ = std::move(group_of);
    eraseNeuronState(indices);
    history_.clear();
    weightsChanged();
    outputShrank(indices);
}

void dense::sourceShrank(const layer& source, std::span<const size_t> removed)
{
    const InputRange whole(source.outputBuffer());
    for (Group& group : groups_) {
        if (!group.reads(source))
            continue;
        // Walk the pool: entries of `source` that are gone lose their
        // column, the others move down by the entries removed before them.
        std::vector<InputRange> inputs;
        std::vector<size_t> columns;
        Group rebuilt;  // only for append()'s merging
        size_t col = 0;
        for (const InputRange& range : group.inputs) {
            if (!range.sameBuffer(whole)) {
                rebuilt.append(range);
                col += range.count;
                continue;
            }
            for (size_t k = 0; k < range.count; ++k, ++col) {
                const size_t index = range.offset + k;
                const auto at = std::lower_bound(removed.begin(), removed.end(), index);
                if (at != removed.end() && *at == index) {
                    columns.push_back(col);
                    continue;
                }
                const size_t moved = index - static_cast<size_t>(at - removed.begin());
                rebuilt.append(InputRange(range.buffer.get(), moved, 1));
            }
        }
        group.inputs = std::move(rebuilt.inputs);
        group.weights.removeColumns(columns);
        for (kernels::weight_matrix* m : {&group.elig, &group.adapt, &group.grad})
            if (m->cols() == group.weights.cols() + columns.size())
                m->removeColumns(columns);
        const auto erase = [&columns](auto& v) {
            if (v.empty())
                return;
            size_t next = 0, out = 0;
            for (size_t c = 0; c < v.size(); ++c) {
                if (next < columns.size() && columns[next] == c) {
                    ++next;
                    continue;
                }
                v[out++] = v[c];
            }
            v.resize(out);
        };
        erase(group.values);
        erase(group.trace);
        erase(group.frozen);
    }
    history_.clear();
    weightsChanged();
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
        initWeights(rng::WeightStream::Growth, source.outgoingInit(), w);
        group.weights.appendColumns(count, w);
        group.append(range);
        if (!group.frozen.empty())
            group.frozen.resize(group.weights.cols(), 0);
    }
    history_.clear();
    weightsChanged();
}

void dense::forward()
{
    // Groups run one after another; a later group of this layer sees the
    // outputs earlier groups just wrote.
    beginForward();
    const bool plain = plainForward();
    for (Group& group : groups_) {
        const std::span<const float> x = group.gather();
        if (tracesInputs())
            traceInputs(group.trace, x);
        group.scratch.resize(group.weights.paddedRows());
        const size_t first = group.neurons.first, count = group.neurons.count;
        parallelChunks(group.weights.blocks(), kernels::threadsFor(count * x.size()), [&](size_t b0, size_t b1) {
            group.weights.multiply(x, group.scratch, b0, b1);
            const size_t end = std::min(b1 * kernels::weight_matrix::lanes, count);
            for (size_t r = b0 * kernels::weight_matrix::lanes; r < end; ++r)
                output_[first + r] = fire(first + r, group.scratch[r], plain);
        });
        if (learningRule().type == LearningRuleType::Eligibility || learningRule().type == LearningRuleType::EProp)
            traceSynapses(group);
    }
    if (learningRule().type == LearningRuleType::Surrogate)
        record();
}

void dense::syncSynapseState(Group& group) const
{
    const LearningRuleType t = learningRule().type;
    const size_t rows = group.neurons.count, cols = group.weights.cols();
    const auto fit = [rows, cols](kernels::weight_matrix& m, bool wanted) {
        if (!wanted) {
            if (m.rows() != 0 || m.cols() != 0)
                m = kernels::weight_matrix();
            return;
        }
        if (m.rows() == rows && m.cols() == cols)
            return;
        if (m.rows() == rows && m.cols() < cols) {
            const size_t added = cols - m.cols();
            m.appendColumns(added, std::vector<float>(rows * added, 0.0f));  // new inputs: no history yet
            return;
        }
        m.reshape(rows, cols);
    };
    fit(group.elig, t == LearningRuleType::Eligibility || t == LearningRuleType::EProp);
    fit(group.adapt, t == LearningRuleType::EProp);
    fit(group.grad, t == LearningRuleType::Surrogate);
}

void dense::traceSynapses(Group& group)
{
    syncSynapseState(group);
    const LearningRule& rule = learningRule();
    const size_t first = group.neurons.first, count = group.neurons.count, padded = group.weights.paddedRows();
    const std::span<const float> x = group.values;
    for (std::vector<float>* v : {&group.rowA, &group.rowB, &group.rowC, &group.rowD})
        v->assign(padded, 0.0f);
    if (rule.type == LearningRuleType::Eligibility) {
        // Hebbian: how strongly the neuron fired, times each input.
        for (size_t r = 0; r < count; ++r)
            group.rowA[r] = std::abs(output_[first + r]);
        parallelChunks(group.weights.blocks(), kernels::threadsFor(count * x.size()), [&](size_t b0, size_t b1) {
            group.elig.trace(x, group.rowA, rule.trace, b0, b1);
        });
        return;
    }
    // EProp: derivatives with respect to the weights see the normalisation.
    for (size_t r = 0; r < count; ++r) {
        const float scale = normScale(first + r);
        group.rowA[r] = derivativeDyds()[first + r] * scale;
        group.rowB[r] = derivativeDydthr()[first + r];
        group.rowC[r] = derivativeDthrdthr()[first + r];
        group.rowD[r] = derivativeDthrds()[first + r] * scale;
    }
    parallelChunks(group.weights.blocks(), kernels::threadsFor(count * x.size()), [&](size_t b0, size_t b1) {
        group.elig.eprop(group.adapt, x, group.rowA, group.rowB, group.rowC, group.rowD, rule.trace, b0, b1);
    });
}

void dense::record()
{
    const size_t window = learningRule().window;
    Frame frame;
    if (history_.size() >= window) {
        frame = std::move(history_.back());  // reuse the oldest frame's storage
        history_.pop_back();
    }
    while (history_.size() >= window)
        history_.pop_back();
    frame.inputs.resize(groups_.size());
    for (size_t g = 0; g < groups_.size(); ++g)
        frame.inputs[g].assign(groups_[g].values.begin(), groups_[g].values.end());
    frame.dyds.assign(derivativeDyds().begin(), derivativeDyds().end());
    frame.dydthr.assign(derivativeDydthr().begin(), derivativeDydthr().end());
    frame.dthrdthr.assign(derivativeDthrdthr().begin(), derivativeDthrdthr().end());
    frame.dthrds.assign(derivativeDthrds().begin(), derivativeDthrds().end());
    history_.push_front(std::move(frame));
}

void dense::backpropagate(size_t ago, std::span<const float> gy, std::span<float> gthr, const SourceGradient& toSource)
{
    if (ago >= history_.size())
        return;
    const Frame& frame = history_[ago];
    const size_t n = neurons_.size();
    if (gy.size() != n || gthr.size() != n || frame.dyds.size() != n || frame.inputs.size() != groups_.size())
        throw std::logic_error("dense::backpropagate: the gradients or the history do not match the layer");
    // dL / ds of every neuron at that tick, and dL / dthr before it.
    std::vector<float> gs(n);
    for (size_t i = 0; i < n; ++i) {
        gs[i] = gy[i] * frame.dyds[i] + gthr[i] * frame.dthrds[i];
        gthr[i] = gy[i] * frame.dydthr[i] + gthr[i] * frame.dthrdthr[i];
    }
    if (!bias_grad_.empty())
        for (size_t i = 0; i < n; ++i)
            bias_grad_[i] += gs[i];
    std::vector<float> gx;
    for (size_t g = 0; g < groups_.size(); ++g) {
        Group& group = groups_[g];
        const std::vector<float>& x = frame.inputs[g];
        if (x.size() != group.weights.cols())
            continue;  // rewired since then
        syncSynapseState(group);
        const size_t first = group.neurons.first, count = group.neurons.count;
        group.rowA.assign(group.weights.paddedRows(), 0.0f);
        for (size_t r = 0; r < count; ++r)
            group.rowA[r] = gs[first + r] * normScale(first + r);
        parallelChunks(group.weights.blocks(), kernels::threadsFor(count * x.size()), [&](size_t b0, size_t b1) {
            group.grad.trace(x, group.rowA, 1.0f, b0, b1);
        });
        if (!toSource)
            continue;
        gx.assign(x.size(), 0.0f);
        group.weights.multiplyTransposed(group.rowA, gx);
        size_t col = 0;
        for (const InputRange& range : group.inputs) {
            for (const layer& source : group.sources)
                if (range.sameBuffer(InputRange(source.outputBuffer()))) {
                    toSource(source, range.offset, std::span<const float>(gx).subspan(col, range.count));
                    break;
                }
            col += range.count;
        }
    }
}

void dense::applyGradients(float learningRate)
{
    const LearningRule& rule = learningRule();
    const float shrink = std::max(0.0f, 1.0f - learningRate * rule.decay);
    for (Group& group : groups_) {
        if (group.grad.rows() != group.neurons.count || group.grad.cols() != group.weights.cols())
            continue;
        const size_t first = group.neurons.first, count = group.neurons.count, padded = group.weights.paddedRows();
        group.scratch.assign(padded, 0.0f);
        group.keep.assign(padded, 1.0f);
        group.active.assign(padded, 0);
        for (size_t r = 0; r < count; ++r) {
            if (neuronFrozen(first + r))
                continue;
            group.active[r] = 1;
            group.scratch[r] = -learningRate * neurons_[first + r].learningGain();
            group.keep[r] = shrink;
        }
        const std::vector<float> saved = saveFrozen(group);
        parallelChunks(group.weights.blocks(), kernels::threadsFor(count * group.weights.cols()), [&](size_t b0, size_t b1) {
            group.weights.learnFrom(group.grad, group.scratch, group.keep, group.active, max_weight, b0, b1);
        });
        restoreFrozen(group, saved);
        group.grad.zero();
    }
    applyBiasGradient(learningRate);
    weightsChanged();
}

std::vector<float> dense::saveFrozen(const Group& group) const
{
    std::vector<float> saved;
    if (group.frozen.empty())
        return saved;
    std::vector<float> column(group.weights.rows());
    for (size_t c = 0; c < group.frozen.size() && c < group.weights.cols(); ++c)
        if (group.frozen[c]) {
            group.weights.copyColumn(c, column);
            saved.insert(saved.end(), column.begin(), column.end());
        }
    return saved;
}

void dense::restoreFrozen(Group& group, const std::vector<float>& saved) const
{
    if (saved.empty())
        return;
    const size_t rows = group.weights.rows();
    size_t at = 0;
    for (size_t c = 0; c < group.frozen.size() && c < group.weights.cols(); ++c)
        if (group.frozen[c]) {
            group.weights.setColumn(c, std::span<const float>(saved).subspan(at, rows));
            at += rows;
        }
}

size_t dense::setInputsFrozen(const InputRange& inputs, bool frozen)
{
    size_t changed = 0;
    for (Group& group : groups_) {
        if (!group.frozen.empty())
            group.frozen.resize(group.weights.cols(), 0);
        size_t col = 0;
        for (const InputRange& range : group.inputs) {
            if (range.sameBuffer(inputs)) {
                for (size_t k = 0; k < range.count; ++k) {
                    const size_t index = range.offset + k;
                    if (index < inputs.offset || index >= inputs.offset + inputs.count)
                        continue;
                    if (group.frozen.empty()) {
                        if (!frozen)
                            continue;
                        group.frozen.assign(group.weights.cols(), 0);
                    }
                    if ((group.frozen[col + k] != 0) != frozen) {
                        group.frozen[col + k] = frozen ? 1 : 0;
                        ++changed;
                    }
                }
            }
            col += range.count;
        }
        if (!group.frozen.empty() && std::ranges::none_of(group.frozen, [](std::uint8_t f) { return f != 0; }))
            group.frozen.clear();
    }
    return changed;
}

size_t dense::frozenInputCount() const
{
    size_t count = 0;
    for (const Group& group : groups_)
        count += static_cast<size_t>(std::ranges::count(group.frozen, std::uint8_t{1}));
    return count;
}

namespace {

std::vector<float> rowOf(const kernels::weight_matrix& m, size_t rows, size_t cols, size_t row)
{
    if (m.rows() != rows || m.cols() != cols)
        return {};
    std::vector<float> out(cols);
    m.copyRow(row, out);
    return out;
}

} // namespace

std::vector<float> dense::synapseTrace(size_t index) const
{
    const size_t g = group_of_.at(index);
    if (g == no_group)
        return {};
    const Group& group = groups_[g];
    return rowOf(group.elig, group.neurons.count, group.weights.cols(), index - group.neurons.first);
}

std::vector<float> dense::thresholdTrace(size_t index) const
{
    const size_t g = group_of_.at(index);
    if (g == no_group)
        return {};
    const Group& group = groups_[g];
    return rowOf(group.adapt, group.neurons.count, group.weights.cols(), index - group.neurons.first);
}

void dense::resetSynapseState()
{
    for (Group& group : groups_) {
        group.elig.zero();
        group.adapt.zero();
        group.grad.zero();
    }
    history_.clear();
}

namespace {

void writeMatrix(std::ostream& os, const kernels::weight_matrix& m)
{
    binary_io::write(os, static_cast<std::uint64_t>(m.rows()));
    binary_io::write(os, static_cast<std::uint64_t>(m.cols()));
    std::vector<float> row(m.cols());
    for (size_t r = 0; r < m.rows(); ++r) {
        m.copyRow(r, row);
        binary_io::write(os, std::span<const float>(row));
    }
}

// Reads a matrix written by writeMatrix; nullopt-like empty matrix on a bad size.
bool readMatrix(std::istream& is, kernels::weight_matrix& out)
{
    const auto rows = binary_io::read<std::uint64_t>(is);
    const auto cols = binary_io::read<std::uint64_t>(is);
    if (!is || rows * cols > max_serialized_weights || (cols != 0 && rows > max_serialized_weights / cols))
        throw std::runtime_error("layer deserialize: implausible per-synapse state size");
    std::vector<float> values(static_cast<size_t>(rows * cols));
    binary_io::read(is, std::span<float>(values));
    if (!is)
        return false;
    out = kernels::weight_matrix(static_cast<size_t>(rows), static_cast<size_t>(cols), values);
    return true;
}

} // namespace

void dense::writeLayerExtras(std::ostream& os) const
{
    binary_io::write(os, static_cast<std::uint64_t>(groups_.size()));
    for (const Group& group : groups_) {
        binary_io::write(os, static_cast<std::uint64_t>(group.frozen.size()));
        binary_io::write(os, std::span<const std::uint8_t>(group.frozen));
        writeMatrix(os, group.elig);
        writeMatrix(os, group.adapt);
    }
}

void dense::readLayerExtras(std::istream& is, DeserializeMode mode)
{
    const auto count = binary_io::read<std::uint64_t>(is);
    if (!is)
        return;
    if (count > max_serialized_weights)
        throw std::runtime_error("layer deserialize: implausible group count");
    for (size_t g = 0; g < count; ++g) {
        const auto frozen_count = binary_io::read<std::uint64_t>(is);
        if (!is || frozen_count > max_serialized_weights)
            throw std::runtime_error("layer deserialize: implausible frozen input count");
        std::vector<std::uint8_t> frozen(static_cast<size_t>(frozen_count));
        binary_io::read(is, std::span<std::uint8_t>(frozen));
        kernels::weight_matrix elig, adapt;
        if (!is || !readMatrix(is, elig) || !readMatrix(is, adapt))
            return;  // the caller reports the truncated stream
        if (count != groups_.size())
            continue;  // saved with another wiring: start from nothing
        Group& group = groups_[g];
        if (frozen.size() == group.weights.cols())
            group.frozen = std::move(frozen);
        const auto fits = [&group](const kernels::weight_matrix& m) {
            return m.rows() == group.neurons.count && m.cols() == group.weights.cols();
        };
        if (mode == DeserializeMode::FullState) {
            if (fits(elig))
                group.elig = std::move(elig);
            if (fits(adapt))
                group.adapt = std::move(adapt);
        }
    }
}

void dense::updateWeights()
{
    const bool scaled = scaledUpdates();
    for (Group& group : groups_) {
        const size_t first = group.neurons.first, count = group.neurons.count, padded = group.weights.paddedRows();
        group.scratch.assign(padded, 0.0f);
        group.active.assign(padded, 0);
        bool any = false;
        for (size_t r = 0; r < count; ++r)
            if (step_active_[first + r]) {
                group.active[r] = 1;
                group.scratch[r] = step_delta_[first + r];
                any = true;
            }
        if (!any)
            continue;
        const std::vector<float> saved = saveFrozen(group);
        if (learningRule().type == LearningRuleType::Eligibility || learningRule().type == LearningRuleType::EProp) {
            syncSynapseState(group);
            group.keep.assign(padded, 1.0f);
            std::copy_n(step_keep_.begin() + static_cast<std::ptrdiff_t>(first), count, group.keep.begin());
            parallelChunks(group.weights.blocks(), kernels::threadsFor(count * group.weights.cols()), [&](size_t b0, size_t b1) {
                group.weights.learnFrom(group.elig, group.scratch, group.keep, group.active, max_weight, b0, b1);
            });
            restoreFrozen(group, saved);
            continue;
        }
        std::span<const float> pre;
        if (learnsFromSigns()) {
            // The signs of the inputs this group's last forward() summed (time
            // t), not of the values its sources hold now: a source that runs
            // later in the update order, or this layer itself, has moved on
            // to t since then. Inputs added since that forward() count as 0.
            group.values.resize(group.weights.cols(), 0.0f);
            group.signs.resize(group.values.size());
            kernels::signs(group.values, group.signs);
            pre = group.signs;
        } else {
            if (group.trace.size() != group.weights.cols())
                group.trace.resize(group.weights.cols(), 0.0f);  // grew since the last forward()
            pre = group.trace;
        }
        if (scaled) {
            group.keep.assign(padded, 1.0f);
            std::copy_n(step_keep_.begin() + static_cast<std::ptrdiff_t>(first), count, group.keep.begin());
        }
        parallelChunks(group.weights.blocks(), kernels::threadsFor(count * pre.size()), [&](size_t b0, size_t b1) {
            if (scaled)
                group.weights.learnScaled(pre, group.scratch, group.keep, group.active, max_weight, b0, b1);
            else
                group.weights.learn(pre, group.scratch, group.active, max_weight, b0, b1);
        });
        restoreFrozen(group, saved);
    }
}

void dense::copyInputTraces(std::vector<float>& out) const
{
    out.clear();
    if (!tracesInputs())
        return;
    for (const Group& group : groups_) {
        out.insert(out.end(), group.trace.begin(), group.trace.end());
        out.resize(out.size() + (group.weights.cols() - std::min(group.trace.size(), group.weights.cols())), 0.0f);
    }
}

void dense::storeInputTraces(std::span<const float> traces)
{
    size_t total = 0;
    for (const Group& group : groups_)
        total += group.weights.cols();
    if (traces.size() != total)
        return;  // not saved with this wiring: start from zero
    size_t offset = 0;
    for (Group& group : groups_) {
        const auto part = traces.subspan(offset, group.weights.cols());
        group.trace.assign(part.begin(), part.end());
        offset += part.size();
    }
}

void dense::clearInputTraces()
{
    for (Group& group : groups_)
        group.trace.clear();
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
    weightsChanged();
}

size_t dense::inputCount(size_t index) const
{
    const size_t g = group_of_.at(index);
    return g == no_group ? 0 : groups_[g].weights.cols();
}

std::vector<float> dense::lastInputs(size_t index) const
{
    const size_t g = group_of_.at(index);
    return g == no_group ? std::vector<float>{} : groups_[g].values;
}

std::vector<float> dense::inputTrace(size_t index) const
{
    const size_t g = group_of_.at(index);
    return g == no_group || !tracesInputs() ? std::vector<float>{} : groups_[g].trace;
}

dense::NeuronRange dense::groupNeurons(size_t group) const
{
    return groups_.at(group).neurons;
}

} // namespace exr
