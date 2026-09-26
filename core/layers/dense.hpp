#pragma once
#include <vector>
#include <deque>
#include <memory>
#include <functional>
#include <cstddef>
#include <istream>
#include <ostream>
#include "../neuron.hpp"
#include "layer.hpp"

// A layer of neurons that can be wired to one or more distinct input
// sources over its lifetime. Each source gets its own "wiring group": the
// neurons reading from that source, plus the source's output pool. Growth
// is explicit and propagates automatically to anyone downstream:
//
//   input.joinDense(hidden)        // hidden's neurons read from input's output
//   hidden.addFeedback(hidden2, 3) // adds 3 NEW neurons to hidden2, wired
//                                  // ONLY to hidden's output - hidden2's
//                                  // existing neurons are untouched
//
// If some third layer already called `.joinDense(hidden2)` before that
// addFeedback, its wiring group sourced from hidden2 is automatically
// extended by 3 entries (with matching new weights) - it doesn't need to
// be told explicitly.
class dense : public layer
{
public:
    // recoveryJitter / learningJitter / alphaJitter: every neuron this layer
    // creates (now and through later growth) gets its recovery, learning gain
    // and alpha drawn from them (neuron::randomizeDynamics). Disabled = defaults.
    explicit dense(size_t nNumber, bool hasHabituation = true, bool hasER = true,
                   const Jitter& recoveryJitter = {}, const Jitter& learningJitter = {},
                   const Jitter& alphaJitter = {});

    // Redraws that parameter for every existing neuron, in neuron order, and
    // keeps the jitter for later growth.
    void setRecoveryJitter(const Jitter& jitter) final;
    void setLearningJitter(const Jitter& jitter) final;
    void setAlphaJitter(const Jitter& jitter) final;
    const Jitter& getRecoveryJitter() const { return recoveryJitter_; }
    const Jitter& getLearningJitter() const { return learningJitter_; }
    const Jitter& getAlphaJitter() const { return alphaJitter_; }

    // Wires every neuron CURRENTLY in this layer to read from source's
    // output, as a new wiring group. Call this once, early - before any
    // addFeedback calls add neurons this layer's own groups won't cover.
    virtual bool join(layer& source) final;

    // Adds `width` new neurons to targetLayer, wired only to this layer's
    // current output (their own isolated wiring group). targetLayer's
    // existing neurons/groups are untouched. Any layer already listening to
    // targetLayer's output gets its matching wiring group auto-extended.
    // virtual bool addFeedback(layer& targetLayer, size_t width);

    // Steps every neuron against its own wiring group's input pool.
    void forward();

    std::deque<neuron>& getNeurons() { return neurons; }
    const std::deque<neuron>& getNeurons() const { return neurons; }

    // Each entry aliases (via shared_ptr) one neuron's live output slot -
    // downstream layers/groups read through these, always seeing the
    // latest value without needing to be re-notified.
    std::vector<std::shared_ptr<float>>& getOutput() { return output; }
    const std::vector<std::shared_ptr<float>>& getOutput() const { return output; }

    size_t size() const { return neurons.size(); }

    void serialize(std::ostream& os) const;
    void deserialize(std::istream& is, DeserializeMode mode = DeserializeMode::FullState,
                     std::uint32_t neuronFormat = NEURON_FORMAT_VERSION);

    virtual LayerType getType() final { return LayerType::Dense ; };
    // Adds `width` new neurons in a new group sourced from `sourceOutput`.
    // Registers *this as a listener on `source`, so it can notify us later
    // if it grows further.
    virtual void addNeuronsWithGroup(size_t width, const std::vector<std::shared_ptr<float>>& sourceOutput, layer& source);


    virtual bool addFeedback(layer& targetLayer, size_t width) final;


    // Called by a source layer we're listening to when ITS output grows.
    // Extends every one of our groups whose source matches, and grows the
    // weights of every neuron in those groups to match.
    virtual void notifySourceGrew(layer& sourceLayer, const std::vector<std::shared_ptr<float>>& newOutputEntries) final;

    // Learning & Plasticity
    virtual void applyReward(float reward,float learningRate) final;

    virtual  bool attachInputs(const std::vector<std::shared_ptr<float>>& input_pointers);

private:
    struct WiringGroup
    {
        std::vector<std::shared_ptr<float>> inputs; // the source pool this group reads from
        std::vector<size_t> neuronIndices;           // indices into `neurons` using this group
        std::reference_wrapper<layer> source;        // which layer's output this tracks, for propagation
        std::vector<float> values;                   // `inputs` copied into contiguous floats, refreshed before each use
    };

    // Work (neurons x inputs multiply-adds) per thread. A group gets
    // work / parallel_work_per_thread threads, capped at the OpenMP maximum,
    // and runs serially below two. Measured on 20 threads: giving every
    // mid-sized group all threads was fastest but woke and parked 20 threads
    // each tick; on a reservoir workload (200 x 300 group) that cost 2.6x the
    // CPU of serial for a 2.6x speed-up, versus 1.2x the CPU for 2.0x with
    // this value. Large groups (>= 16384 x 20) still use every thread.
    static constexpr size_t parallel_work_per_thread = 16384;

    // Refreshes group.values from group.inputs and returns it.
    static std::span<const float> gather(WiringGroup& group);

    std::deque<neuron> neurons;   // deque, not vector: growth (emplace_back) must never invalidate
                                   // references to already-existing neurons held elsewhere (e.g. in groups)
    std::vector<std::shared_ptr<float>> output;
    std::vector<WiringGroup> groups;

    bool hasHabituation_, hasER_; // propagated to every neuron this layer constructs, including future growth
    Jitter recoveryJitter_, learningJitter_, alphaJitter_; // likewise
};
