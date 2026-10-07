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
#include <deque>
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
    // Learning (applyReward and friends, see neuron_layer): every neuron
    // that learns updates its row against its group's inputs: their current
    // signs (Sign rule), their trace (Trace, FeedbackAlignment, ...) or its
    // per-synapse eligibility (Eligibility, EProp).
    bool learns() const override { return true; }
    bool supportsPerSynapse() const override { return true; }
    bool followsShrinkingSources() const override { return true; }

    // --- Growth and pruning -----------------------------------------------
    // `count` new neurons reading the same inputs as wiring group `group`
    // (a copy of its pool), with new weights drawn by incomingInit(); layers
    // reading this one get the new outputs, with weights drawn by
    // outgoingInit() (Zero: the new neurons start without influence). If the
    // group reads this layer itself, the new neurons read their own outputs
    // too. Throws std::out_of_range for a missing group.
    void growNeurons(size_t count, size_t group = 0);
    // Removes neurons (sorted, unique indices): their weights, state and
    // outputs; layers reading this one drop the matching inputs (only Dense
    // readers can). Throws std::invalid_argument for bad indices and
    // std::logic_error for a reader that cannot follow, before changing
    // anything.
    void removeNeurons(std::span<const size_t> indices);

    // --- Freezing inputs ----------------------------------------------------
    // The weights reading `inputs` (entries of a layer's output buffer or of
    // the sensors) stop learning, or learn again, in every group reading
    // them; inputs added later learn. Returns how many weight columns
    // changed state. Saved with the layer.
    size_t setInputsFrozen(const InputRange& inputs, bool frozen = true);
    // Frozen weight columns over all groups.
    size_t frozenInputCount() const;
    // The largest |weight| with which any neuron reads `inputs` (entries of
    // a layer's output buffer or of the sensors); 0 when none reads them.
    // E.g. whether a neuron of another layer still influences this one.
    float maxAbsWeightFrom(const InputRange& inputs) const;

    // --- Per-synapse rules (see learning.hpp) -------------------------------
    // Neuron `index`'s eligibility trace E, in pool order (Eligibility,
    // EProp; empty for other rules). Read-only probe.
    std::vector<float> synapseTrace(size_t index) const;
    // EProp: neuron `index`'s threshold eligibility, d thr / d w in pool
    // order (empty for other rules). Read-only probe.
    std::vector<float> thresholdTrace(size_t index) const;
    // Surrogate gradients. Every forward() records its inputs and the
    // neurons' local derivatives, keeping the last `window` ticks (cleared
    // when the wiring changes; not saved). network::applyError then calls
    // backpropagate() tick by tick and applyGradients() once.
    size_t historyFrames() const { return history_.size(); }
    // dL / dx of one input range that comes from a layer: `offset` is the
    // range's first entry in that layer's output.
    using SourceGradient = std::function<void(const layer& source, size_t offset, std::span<const float> gradient)>;
    // One tick of backpropagation, `ago` forwards back (0 = the last). `gy`:
    // dL / dy of every neuron at that tick; `gthr`: dL / dthr of every
    // neuron's threshold after that tick, replaced by dL / dthr before it.
    // Accumulates the weight (and bias) gradients and hands dL / dx of every
    // input range read from a layer to `toSource` (sensors are skipped).
    void backpropagate(size_t ago, std::span<const float> gy, std::span<float> gthr, const SourceGradient& toSource);
    // w -= learningRate * gain * accumulated dL / dw for unfrozen neurons
    // and inputs (and the same for the bias), then clears the gradients.
    void applyGradients(float learningRate);

    // --- Weights --------------------------------------------------------
    // Neuron `index`'s weights, in pool order (empty when not wired).
    std::vector<float> weights(size_t index) const;
    // Size must equal inputCount(index); throws std::invalid_argument otherwise.
    void setWeights(size_t index, const std::vector<float>& weights);
    size_t inputCount(size_t index) const;
    // Read-only probes of what learning will use for neuron `index`, in pool
    // order: the inputs its group summed in the last forward() (time t), and
    // the input trace X after that forward() (empty for the Sign rule).
    std::vector<float> lastInputs(size_t index) const;
    std::vector<float> inputTrace(size_t index) const;

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
    void updateWeights() override;
    void copyInputTraces(std::vector<float>& out) const override;
    void storeInputTraces(std::span<const float> traces) override;
    void clearInputTraces() override;
    void resetSynapseState() override;
    void writeLayerExtras(std::ostream& os) const override;
    void readLayerExtras(std::istream& is, DeserializeMode mode) override;
    void sourceShrank(const layer& source, std::span<const size_t> removed) override;

private:
    struct Group
    {
        std::vector<InputRange> inputs;     // the pool, in weight-column order
        std::vector<std::reference_wrapper<const layer>> sources;  // layers whose outputs are in the pool
        NeuronRange neurons;                // a group's neurons are always contiguous
        kernels::weight_matrix weights;     // one row per neuron, one column per pool entry
        std::vector<float> values;          // the pool gathered into contiguous floats by the last forward()
        std::vector<float> signs;           // Sign rule: signs of `values` during learning
        std::vector<float> scratch;         // per-row sums / deltas (padded to whole blocks)
        std::vector<std::uint8_t> active;   // per-row eligibility during learning
        std::vector<float> keep;            // per-row shrink factor during learning
        std::vector<float> trace;           // input trace X (rules other than Sign), pool order
        std::vector<std::uint8_t> frozen;   // per pool entry: 1 = its weights do not learn (empty: none)
        kernels::weight_matrix elig;        // Eligibility, EProp: per-synapse eligibility trace E
        kernels::weight_matrix adapt;       // EProp: per-synapse threshold eligibility
        kernels::weight_matrix grad;        // Surrogate: accumulated dL / dw
        std::vector<float> rowA, rowB, rowC, rowD;  // per-row scratch for the per-synapse kernels

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
    // Fills `w` from `stream`, or with zeros for WeightInit::Zero.
    static void initWeights(rng::WeightStream stream, WeightInit init, std::span<float> w);
    // Sizes the group's per-synapse matrices to its weights (new inputs and
    // rows start at 0).
    void syncSynapseState(Group& group) const;
    // Per-synapse updates after a group fired (Eligibility, EProp).
    void traceSynapses(Group& group);
    // Copies of the frozen columns' weights, put back after an update.
    std::vector<float> saveFrozen(const Group& group) const;
    void restoreFrozen(Group& group, const std::vector<float>& saved) const;

    // One recorded forward() for surrogate gradients.
    struct Frame
    {
        std::vector<std::vector<float>> inputs;  // per group: the pool values
        std::vector<float> dyds, dydthr, dthrdthr, dthrds;
    };
    void record();

    std::vector<Group> groups_;
    std::vector<size_t> group_of_;  // per neuron: index into groups_, or no_group
    std::deque<Frame> history_;     // Surrogate: newest first
};

} // namespace exr
