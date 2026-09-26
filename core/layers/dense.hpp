// A fully connected layer. Neurons are organised in wiring groups: the
// neurons of a group read one shared input pool, the concatenated outputs of
// the group's sources (layers) and any attached sensors, with one weight row
// each. Every neuron belongs to at most one group, so a neuron reading
// several sources integrates them in one weighted sum.
//
//   hidden.join(input)            every neuron hidden has now reads input
//   hidden.addFeedback(out, 3)    3 NEW neurons in out, reading hidden
//
// Growth propagates: when a layer grows (neurons added to it), every group
// reading it gets the new outputs appended to its pool, with new weights.
#pragma once
#include <cstddef>
#include <functional>
#include <optional>
#include <span>
#include <vector>
#include "../kernels.hpp"
#include "../random.hpp"
#include "neuron_layer.hpp"

namespace exr {

class dense final : public neuron_layer
{
public:
    // recoveryJitter / learningJitter / alphaJitter: every neuron this layer
    // creates (now and through later growth) draws its recovery, learning
    // gain and alpha from them. Disabled = defaults.
    explicit dense(size_t count = 0, bool hasHabituation = true, bool hasER = true,
                   const Jitter& recoveryJitter = {}, const Jitter& learningJitter = {},
                   const Jitter& alphaJitter = {});

    LayerType type() const override { return LayerType::Dense; }

    // Every neuron this layer has now reads `source` (or the sensors): the
    // source is appended to the pool of every existing group (that does not
    // read it yet), and neurons in no group form a new group. Neurons added
    // later by feedback are in their own group and do not read it.
    void join(layer& source) override;
    void attachInputs(const InputRange& sensors) override;
    // `count` new neurons in a new group reading `source`'s current output.
    // Readers of this layer are extended with the new outputs.
    void addNeurons(size_t count, layer& source) override;

    // Each group: gather its pool into contiguous floats, compute every
    // neuron's weighted sum (SIMD, in parallel for large groups), activate.
    // A group reading this layer itself sees the outputs from before it ran.
    void forward() override;
    // Every eligible neuron of every group learns (see neuron::learningDelta)
    // against the group's current inputs.
    void applyReward(float reward, float learningRate) override;
    bool learns() const override { return true; }

    // --- Weights --------------------------------------------------------
    // Neuron `index`'s weights, in pool order (empty when not wired).
    std::vector<float> weights(size_t index) const;
    // Size must equal inputCount(index); throws std::invalid_argument otherwise.
    void setWeights(size_t index, const std::vector<float>& weights);
    size_t inputCount(size_t index) const;

    // --- Wiring inspection ----------------------------------------------
    struct NeuronRange
    {
        size_t first = 0;
        size_t count = 0;
        bool operator==(const NeuronRange&) const = default;
    };
    static constexpr size_t no_group = static_cast<size_t>(-1);
    size_t groupCount() const { return groups_.size(); }
    NeuronRange groupNeurons(size_t group) const;
    size_t groupOf(size_t index) const { return group_of_.at(index); }  // no_group when not wired

protected:
    void sourceGrew(const layer& source, size_t offset, size_t count) override;
    void copyWeights(size_t index, std::vector<float>& out) const override;
    size_t expectedWeights(size_t index) const override { return inputCount(index); }
    void storeWeights(size_t index, std::span<const float> weights) override;
    bool wired() const override { return !groups_.empty(); }
    void neuronsReplaced() override { group_of_.assign(neurons_.size(), no_group); }

private:
    struct Group
    {
        std::vector<InputRange> inputs;     // the pool, in weight-column order
        std::vector<std::reference_wrapper<const layer>> sources;  // layers whose outputs are in the pool
        NeuronRange neurons;                // a group's neurons are always contiguous
        kernels::weight_matrix weights;     // one row per neuron, one column per pool entry
        std::vector<float> values;          // the pool gathered into contiguous floats
        std::vector<float> scratch;         // per-row sums / deltas (padded to whole blocks)
        std::vector<std::uint8_t> active;   // per-row eligibility during learning

        bool reads(const layer& source) const;
        void append(const InputRange& range);
        std::span<const float> gather();
    };

    // The layer a range comes from; none for sensors.
    using Source = std::optional<std::reference_wrapper<const layer>>;
    // Appends `range` to the pool of every group (that does not read
    // `source` yet, when given), with weights from `stream`.
    void appendToGroups(const InputRange& range, Source source, rng::WeightStream stream);
    // Neurons in no group form a new one reading `range`, weights from `stream`.
    void groupUnwired(const InputRange& range, Source source, rng::WeightStream stream);
    // The only way a group is created. Throws std::logic_error if a neuron
    // is already in a group: a neuron belongs to exactly one group.
    void addGroup(Group group);

    std::vector<Group> groups_;
    std::vector<size_t> group_of_;  // per neuron: index into groups_, or no_group
};

} // namespace exr
