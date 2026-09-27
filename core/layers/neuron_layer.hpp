// What every layer made of neurons shares: the neurons and their output
// slots, per-neuron dynamics (jitter), the learning rule and its per-neuron
// state (see learning.hpp), and serialization of neuron records. Derived
// layers decide how neurons are wired and own the weights.
#pragma once
#include <cstdint>
#include <span>
#include <vector>
#include "../kernels.hpp"
#include "../learning.hpp"
#include "layer.hpp"

namespace exr {

class neuron_layer : public layer
{
public:
    bool hasNeurons() const override { return true; }
    bool hasHabituation() const { return has_habituation_; }
    bool hasER() const { return has_er_; }

    // Dynamics state of each neuron (threshold, recovery, gain, ...).
    std::span<const neuron> neurons() const { return neurons_; }
    std::span<neuron> neurons() { return neurons_; }

    // Drives an output by hand (e.g. a layer used as a fixed source). The
    // next forward() overwrites it for wired neurons.
    void setOutput(size_t index, float value);

    // Per-neuron dynamics: redraws that parameter for every existing neuron,
    // in neuron order (a disabled jitter resets it to the default), and uses
    // the jitter for neurons added later. See Jitter in neuron.hpp.
    void setRecoveryJitter(const Jitter& jitter);
    void setLearningJitter(const Jitter& jitter);
    void setAlphaJitter(const Jitter& jitter);
    const Jitter& recoveryJitter() const { return recovery_jitter_; }
    const Jitter& learningJitter() const { return learning_jitter_; }
    const Jitter& alphaJitter() const { return alpha_jitter_; }

    // Neurons without E-R: a fixed firing threshold for every neuron, now and
    // later (see neuron::gate). 0 = linear. Throws std::invalid_argument for
    // a negative or non-finite value, or a non-zero gate on a layer with E-R.
    void setGate(float gate);
    float gate() const { return gate_; }
    // Neurons without E-R: rectification (ReLU; see neuron::rectified) for
    // every neuron, now and later. Throws std::invalid_argument on a layer
    // with E-R.
    void setRectified(bool rectified);
    bool rectified() const { return rectified_; }
    // How habituation suppresses repeated inputs (see Habituation) for
    // every neuron, now and later; used only by neurons with habituation.
    // Throws std::invalid_argument for an invalid rule.
    void setHabituationRule(const Habituation& rule);
    const Habituation& habituationRule() const { return habituation_rule_; }
    // How E-R thresholds grow on firing (see ThresholdGrowth) for every
    // neuron, now and later. Throws std::invalid_argument for an invalid rule.
    void setThresholdGrowth(const ThresholdGrowth& growth);
    const ThresholdGrowth& thresholdGrowth() const { return growth_; }

    // --- Learning -------------------------------------------------------
    // The rule this layer learns with (see learning.hpp). Setting it resets
    // the rule's per-neuron state (traces, baselines, bias, feedback
    // matrix); weights are kept. Throws std::invalid_argument for an invalid
    // rule, or a rule other than Sign on a layer that does not learn.
    const LearningRule& learningRule() const { return rule_; }
    void setLearningRule(const LearningRule& rule);

    // Learned bias of a neuron (0 when the rule has no bias).
    float bias(size_t index) const { return bias_.empty() ? 0.0f : bias_.at(index); }
    // Throws std::logic_error when the rule has no bias.
    void setBias(size_t index, float value);

    // Every neuron gets the same modulator: the reward.
    void applyReward(float reward, float learningRate) override;
    // A modulator per neuron (size() entries), e.g. each readout's own error.
    void applyModulators(std::span<const float> modulators, float learningRate);
    // Feedback alignment for a hidden layer: each neuron's modulator is its
    // row of a fixed random matrix times `errors` (scaled by
    // 1 / sqrt(errors.size())). The matrix is drawn from the learning random
    // stream the first time, and redrawn when the error size changes; rows
    // for new neurons are appended.
    void applyFeedback(std::span<const float> errors, float learningRate);
    // Fixed feedback matrix row of a neuron (empty before applyFeedback).
    std::vector<float> feedbackRow(size_t index) const;

    // Flags, neuron count, then one record per neuron (see neuron::serialize)
    // carrying that neuron's weights; then, for layer types with weights
    // shared by many neurons (e.g. Conv2D kernels), their count and values.
    void serialize(std::ostream& os) const override;
    // With the same neuron count, restores every neuron in place; the weight
    // counts must match the wiring. With another count, only an unwired layer
    // (no inputs, no readers) is rebuilt from the data. Throws
    // std::runtime_error otherwise; the layer is then left unchanged.
    void deserialize(std::istream& is, DeserializeMode mode = DeserializeMode::FullState,
                     std::uint32_t neuronFormat = NEURON_FORMAT_VERSION) override;

protected:
    // `count` neurons, not wired yet.
    neuron_layer(size_t count, bool hasHabituation, bool hasER, const Jitter& recoveryJitter,
                 const Jitter& learningJitter, const Jitter& alphaJitter);

    // Appends a neuron (with this layer's flags and jitter) and its output slot.
    neuron& newNeuron();

    // --- Hooks for the forward pass and learning --------------------------
    // Call once at the start of forward(): draws this tick's exploration noise.
    void beginForward()
    {
        if (!noise_.empty())
            drawNoise();
    }
    // Activates neuron i with its weighted sum (plus bias and noise) and
    // updates its traces; returns the output. Safe to call for different
    // neurons in parallel.
    float fire(size_t i, float sum) { return fire(i, sum, plain_); }
    // The same with plainForward() read once by the caller: hoisted out of a
    // layer's loop, the plain case costs nothing over activate().
    float fire(size_t i, float sum, bool plain)
    {
        if (plain)
            return neurons_[i].activate(sum);
        return fireWithRule(i, sum);
    }
    bool plainForward() const { return plain_; }
    // Whether the rule keeps input traces (then forward() calls
    // traceInputs() on its input snapshot), and the trace decay.
    bool tracesInputs() const { return rule_.usesTraces(); }
    // X = trace * X + x; `trace` is resized (zero-filled) to x's size.
    void traceInputs(std::vector<float>& trace, std::span<const float> x) const;
    // Whether weights learn from the sign of the current inputs (Sign) or
    // from the input trace (every other rule).
    bool learnsFromSigns() const { return rule_.type == LearningRuleType::Sign; }
    // Whether updates need learnScaled() (anything but the plain sign rule).
    bool scaledUpdates() const { return rule_.type != LearningRuleType::Sign || rule_.decay > 0.0f; }

    // Filled for every neuron before updateWeights() runs: whether it
    // learns, its step and (only when scaledUpdates()) its shrink factor.
    std::vector<float> step_delta_, step_keep_;
    std::vector<std::uint8_t> step_active_;

    // Applies step_* to the weights. Layers without weights keep the default.
    virtual void updateWeights() {}
    // Neurons that compete for learning (LearningRule::winners) are those
    // with the same index modulo this count: one group in a dense layer,
    // one per position in spatial layers.
    virtual size_t competitionPositions() const { return 1; }
    // Input traces for serialization (the layer's own layout).
    virtual void copyInputTraces(std::vector<float>& out) const { out.clear(); }
    virtual void storeInputTraces(std::span<const float>) {}
    virtual void clearInputTraces() {}

    // Weight hooks for serialization. The weights neuron `index` reads
    // (empty when it is not wired):
    virtual void copyWeights(size_t index, std::vector<float>& out) const = 0;
    // How many weights neuron `index` must have in loaded data:
    virtual size_t expectedWeights(size_t index) const = 0;
    virtual void storeWeights(size_t index, std::span<const float> weights) = 0;
    // True when the layer reads anything (so its neurons cannot be replaced).
    virtual bool wired() const = 0;
    // Weights shared by many neurons, saved after the neuron records. Only
    // layer types that return true from hasSharedWeights() write the block.
    virtual bool hasSharedWeights() const { return false; }
    virtual void copySharedWeights(std::vector<float>& out) const { out.clear(); }
    virtual void storeSharedWeights(std::span<const float>) {}
    // Called after deserialize() rebuilt an unwired layer with a new count.
    virtual void neuronsReplaced() {}

    std::vector<neuron> neurons_;

private:
    // One learning call: m(i) is neuron i's modulator.
    template <typename Modulator>
    void learn(Modulator m, float learningRate);
    void selectWinners();
    // Neuron i's step under a rule other than plain Sign: sets delta and
    // keep, returns whether it learns.
    bool ruleStep(size_t i, float m, float learningRate, float& delta, float& keep) const;
    float fireWithRule(size_t i, float sum);
    void drawNoise();
    // Sizes the rule's per-neuron state to the neuron count; new entries
    // get their initial values.
    void resizeLearningState();
    // Sets the rule and clears its state (the noise generator is kept).
    void resetLearningState(const LearningRule& rule);
    void writeLearningState(std::ostream& os) const;
    void readLearningState(std::istream& is, DeserializeMode mode);

    bool has_habituation_, has_er_;  // for every neuron this layer creates, including later growth
    Jitter recovery_jitter_, learning_jitter_, alpha_jitter_;  // likewise
    float gate_ = 0.0f;                                          // likewise
    bool rectified_ = false;                                     // likewise
    Habituation habituation_rule_;                               // likewise
    ThresholdGrowth growth_;                                     // likewise

    LearningRule rule_;
    bool plain_ = true;               // no bias, traces or noise: fire() just activates
    std::vector<float> bias_;         // rule.bias
    std::vector<float> post_;         // output trace P (every rule but Sign)
    std::vector<float> noise_;        // Perturbation: this tick's noise
    std::vector<float> noise_trace_;  // Perturbation: noise trace Z
    std::vector<float> baseline_;     // Trace, Perturbation: reward baseline b
    std::vector<float> theta_;        // BCM: sliding threshold
    kernels::weight_matrix feedback_; // feedback alignment: neurons x errors
    std::vector<float> scratch_;      // applyFeedback: modulators
    std::uint64_t noise_state_ = 0x9E3779B97F4A7C15ull;  // Perturbation noise (xorshift64*); seeded from rng::learning()
};

} // namespace exr
