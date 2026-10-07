#include "neuron_layer.hpp"
#include "../binary_io.hpp"
#include "../kernels.hpp"
#include "../parallel.hpp"
#include "../random.hpp"
#include <algorithm>
#include <atomic>
#include <cmath>
#include <istream>
#include <numeric>
#include <ostream>
#include <sstream>
#include <stdexcept>
#include <string>

namespace exr {

neuron_layer::neuron_layer(size_t count, bool hasHabituation, bool hasER, const Jitter& recoveryJitter,
                           const Jitter& learningJitter, const Jitter& alphaJitter)
    : has_habituation_(hasHabituation), has_er_(hasER),
      recovery_jitter_(recoveryJitter), learning_jitter_(learningJitter), alpha_jitter_(alphaJitter)
{
    neurons_.reserve(count);
    output_.reserve(count);
    for (size_t i = 0; i < count; ++i)
        newNeuron();
}

neuron& neuron_layer::newNeuron()
{
    neuron& n = neurons_.emplace_back(has_habituation_, has_er_);
    n.randomizeDynamics(recovery_jitter_, learning_jitter_, alpha_jitter_);
    n.setGate(gate_);
    n.setRectified(rectified_);
    n.setHabituation(habituation_rule_);
    n.setThresholdGrowth(growth_);
    n.setSpontaneous(spontaneous_);
    n.setRestingThreshold(resting_);
    n.setBinary(binary_);
    output_.push_back(n.output());
    frozen_.push_back(0);
    resizeLearningState();
    return n;
}

void neuron_layer::setOutput(size_t index, float value)
{
    neurons_.at(index).setOutput(value);
    output_[index] = value;
}

void neuron_layer::setRecoveryJitter(const Jitter& jitter)
{
    recovery_jitter_ = jitter;
    for (neuron& n : neurons_)
        n.randomizeRecovery(jitter);
}

void neuron_layer::setLearningJitter(const Jitter& jitter)
{
    learning_jitter_ = jitter;
    for (neuron& n : neurons_)
        n.randomizeLearningGain(jitter);
}

void neuron_layer::setGate(float gate)
{
    if (!std::isfinite(gate) || gate < 0.0f)
        throw std::invalid_argument("setGate: the gate must be a finite value >= 0");
    if (gate > 0.0f && has_er_)
        throw std::invalid_argument("setGate: a fixed threshold is for neurons without E-R");
    gate_ = gate;
    for (neuron& n : neurons_)
        n.setGate(gate);
}

void neuron_layer::setHabituationRule(const Habituation& rule)
{
    if (!rule.valid())
        throw std::invalid_argument("setHabituationRule: steps >= 1, tolerance in [0, 1), decay in [0, 1]");
    habituation_rule_ = rule;
    for (neuron& n : neurons_)
        n.setHabituation(rule);
}

void neuron_layer::setThresholdGrowth(const ThresholdGrowth& growth)
{
    if (!growth.valid() || !std::isfinite(growth.amount))
        throw std::invalid_argument("setThresholdGrowth: rule Log, Linear, Fixed or Multiplicative, amount in [0, 1e6]");
    growth_ = growth;
    for (neuron& n : neurons_)
        n.setThresholdGrowth(growth);
}

void neuron_layer::setRestingThreshold(float resting)
{
    if (!(resting > 0.0f) || !(resting <= max_output))
        throw std::invalid_argument("setRestingThreshold: the resting threshold must be in (0, max_output]");
    resting_ = resting;
    for (neuron& n : neurons_)
        n.setRestingThreshold(resting);
}

void neuron_layer::setSpontaneous(const Spontaneous& spontaneous)
{
    if (!spontaneous.valid() || !std::isfinite(spontaneous.below) || !std::isfinite(spontaneous.amplitude))
        throw std::invalid_argument("setSpontaneous: below and amplitude in [0, max_output], rate in [0, 1]");
    spontaneous_ = spontaneous;
    for (neuron& n : neurons_)
        n.setSpontaneous(spontaneous);
}

void neuron_layer::setRectified(bool rectified)
{
    if (rectified && has_er_)
        throw std::invalid_argument("setRectified: rectification is for neurons without E-R");
    rectified_ = rectified;
    for (neuron& n : neurons_)
        n.setRectified(rectified);
}

void neuron_layer::setBinary(bool binary)
{
    if (binary && has_er_)
        throw std::invalid_argument("setBinary: binary output is for neurons without E-R");
    binary_ = binary;
    for (neuron& n : neurons_)
        n.setBinary(binary);
}

void neuron_layer::setNeuronsFrozen(size_t first, size_t count, bool frozen)
{
    if (first > neurons_.size() || count > neurons_.size() - first)
        throw std::out_of_range("setNeuronsFrozen: neurons [" + std::to_string(first) + ", " +
                                std::to_string(first + count) + ") past the layer's " +
                                std::to_string(neurons_.size()));
    std::fill_n(frozen_.begin() + static_cast<std::ptrdiff_t>(first), count, frozen ? 1 : 0);
}

size_t neuron_layer::frozenNeuronCount() const
{
    return static_cast<size_t>(std::count(frozen_.begin(), frozen_.end(), std::uint8_t{1}));
}

void neuron_layer::setAlphaJitter(const Jitter& jitter)
{
    alpha_jitter_ = jitter;
    for (neuron& n : neurons_)
        n.randomizeAlpha(jitter);
}

void neuron_layer::serialize(std::ostream& os) const
{
    binary_io::write(os, has_habituation_);
    binary_io::write(os, has_er_);
    binary_io::write(os, neurons_.size());

    std::vector<float> weights;
    for (size_t i = 0; i < neurons_.size(); ++i) {
        copyWeights(i, weights);
        neurons_[i].serialize(os, weights);
    }
    if (hasSharedWeights()) {
        copySharedWeights(weights);
        binary_io::write(os, static_cast<std::uint64_t>(weights.size()));
        binary_io::write(os, std::span<const float>(weights));
    }
    writeLearningState(os);
    writeLayerExtras(os);
}

void neuron_layer::deserialize(std::istream& is, DeserializeMode mode, std::uint32_t neuronFormat)
{
    const auto has_habituation = binary_io::read<bool>(is);
    const auto has_er = binary_io::read<bool>(is);
    const auto count = binary_io::read<size_t>(is);
    if (!is)
        return;  // the caller reports the truncated stream
    if (count > max_serialized_weights)
        throw std::runtime_error("layer deserialize: implausible neuron count " + std::to_string(count));

    // Replacing the neurons of a wired layer would leave its wiring (and the
    // readers' view of its output) describing neurons that no longer exist.
    const bool in_place = count == neurons_.size();
    if (!in_place && (wired() || hasReaders()))
        throw std::runtime_error("layer deserialize: data has " + std::to_string(count) +
                                 " neurons but this wired layer has " + std::to_string(neurons_.size()));

    // Everything is read and checked before the layer changes.
    std::vector<neuron> loaded;
    loaded.reserve(count);
    std::vector<std::vector<float>> weights(count);
    for (size_t i = 0; i < count; ++i) {
        neuron n = in_place ? neurons_[i] : neuron(has_habituation, has_er);
        weights[i] = n.deserialize(is, mode, neuronFormat);
        if (!is)
            return;  // the caller reports the truncated stream
        // Every weighted sum reads one weight per input.
        if (in_place && weights[i].size() != expectedWeights(i))
            throw std::runtime_error("layer deserialize: neuron " + std::to_string(i) + " has " +
                                     std::to_string(weights[i].size()) + " weights but reads " +
                                     std::to_string(expectedWeights(i)) + " inputs");
        loaded.push_back(n);
    }

    std::vector<float> shared;
    if (hasSharedWeights()) {
        const auto shared_count = binary_io::read<std::uint64_t>(is);
        if (shared_count > max_serialized_weights)
            throw std::runtime_error("layer deserialize: implausible shared weight count " +
                                     std::to_string(shared_count));
        shared.resize(is ? static_cast<size_t>(shared_count) : 0);
        binary_io::read(is, std::span<float>(shared));
        if (!is)
            return;  // the caller reports the truncated stream
        std::vector<float> expected;
        copySharedWeights(expected);
        if (in_place && shared.size() != expected.size())
            throw std::runtime_error("layer deserialize: " + std::to_string(shared.size()) +
                                     " shared weights, the wiring needs " + std::to_string(expected.size()));
    }

    has_habituation_ = has_habituation;
    has_er_ = has_er;
    neurons_ = std::move(loaded);
    for (neuron& n : neurons_) {
        n.setGate(has_er_ ? 0.0f : gate_);
        n.setRectified(!has_er_ && rectified_);
        n.setHabituation(habituation_rule_);
        n.setThresholdGrowth(growth_);
        n.setSpontaneous(spontaneous_);
        n.setRestingThreshold(resting_, mode != DeserializeMode::FullState);  // a full state keeps its thresholds
        n.setBinary(!has_er_ && binary_);
    }
    if (!in_place)
        frozen_.assign(count, 0);
    output_.resize(count);
    for (size_t i = 0; i < count; ++i)
        output_[i] = neurons_[i].output();
    if (in_place) {
        for (size_t i = 0; i < count; ++i)
            if (!weights[i].empty())
                storeWeights(i, weights[i]);
        if (hasSharedWeights())
            storeSharedWeights(shared);
    } else {
        neuronsReplaced();  // unwired: the weights have nothing to belong to
    }
    if (neuronFormat >= 3)
        readLearningState(is, mode, neuronFormat);
    else
        resizeLearningState();
    if (neuronFormat >= 4 && is)
        readLayerExtras(is, mode);
}

// --- Learning rules -----------------------------------------------------------

void neuron_layer::setLearningRule(const LearningRule& rule)
{
    rule.validate();
    if (rule.type != LearningRuleType::Sign && !learns())
        throw std::invalid_argument("setLearningRule: this layer type does not learn");
    if (rule.perSynapse() && !supportsPerSynapse())
        throw std::invalid_argument(std::string("setLearningRule: the ") + learningRuleName(rule.type) +
                                    " rule keeps a value per synapse and is for Dense layers only");
    resetLearningState(rule);
    if (rule_.type == LearningRuleType::Perturbation)
        noise_state_ = ((std::uint64_t{rng::learning()()} << 32) | rng::learning()()) | 1u;  // never 0
}

void neuron_layer::clearTraces()
{
    std::fill(post_.begin(), post_.end(), 0.0f);
    std::fill(noise_trace_.begin(), noise_trace_.end(), 0.0f);
    std::fill(bias_elig_.begin(), bias_elig_.end(), 0.0f);
    std::fill(bias_adapt_.begin(), bias_adapt_.end(), 0.0f);
    std::fill(bias_grad_.begin(), bias_grad_.end(), 0.0f);
    clearInputTraces();
    resetSynapseState();
}

void neuron_layer::resetLearningState(const LearningRule& rule)
{
    rule_ = rule;
    bias_.clear();
    post_.clear();
    noise_.clear();
    noise_trace_.clear();
    baseline_.clear();
    theta_.clear();
    dyds_.clear();
    dydthr_.clear();
    dthrdthr_.clear();
    dthrds_.clear();
    bias_elig_.clear();
    bias_adapt_.clear();
    bias_grad_.clear();
    feedback_ = kernels::weight_matrix();
    clearInputTraces();
    resetSynapseState();
    resizeLearningState();
}

void neuron_layer::resizeLearningState()
{
    const size_t n = neurons_.size();
    const LearningRuleType t = rule_.type;
    bias_.resize(rule_.bias ? n : 0, 0.0f);
    post_.resize(rule_.usesTraces() ? n : 0, 0.0f);
    const bool perturb = t == LearningRuleType::Perturbation;
    noise_.resize(perturb ? n : 0, 0.0f);
    noise_trace_.resize(perturb ? n : 0, 0.0f);
    baseline_.resize(rule_.usesBaseline() ? n : 0, 0.0f);
    theta_.resize(t == LearningRuleType::BCM ? n : 0, 1.0f);
    const bool derivatives = rule_.usesDerivatives();
    for (std::vector<float>* v : {&dyds_, &dydthr_, &dthrdthr_, &dthrds_})
        v->resize(derivatives ? n : 0, 0.0f);
    bias_elig_.resize(rule_.bias && (t == LearningRuleType::Eligibility || t == LearningRuleType::EProp) ? n : 0, 0.0f);
    bias_adapt_.resize(rule_.bias && t == LearningRuleType::EProp ? n : 0, 0.0f);
    bias_grad_.resize(rule_.bias && t == LearningRuleType::Surrogate ? n : 0, 0.0f);
    plain_ = bias_.empty() && post_.empty() && noise_.empty() && !normalized_ && !rule_.perSynapse();
}

void neuron_layer::setNormalized(bool normalized)
{
    if (normalized && !learns())
        throw std::invalid_argument("setNormalized: this layer type has fixed filters, not learned weights");
    normalized_ = normalized;
    norms_stale_ = true;
    inverse_norm_.clear();
    resizeLearningState();
}

float neuron_layer::squaredWeightNorm(size_t index) const
{
    std::vector<float> w;
    copyWeights(index, w);
    float s = 0.0f;
    for (float x : w)
        s += x * x;
    return s;
}

void neuron_layer::refreshNorms()
{
    // A neuron with (almost) no weights keeps its raw sum: dividing by ~0
    // would blow up a sum that is itself ~0.
    const size_t n = neurons_.size();
    inverse_norm_.resize(n);
    parallelChunks(n, kernels::threadsFor(n * 64), [&](size_t i0, size_t i1) {
        for (size_t i = i0; i < i1; ++i) {
            const float norm = std::sqrt(squaredWeightNorm(i));
            inverse_norm_[i] = norm > normalization_epsilon ? 1.0f / norm : 1.0f;
        }
    });
    norms_stale_ = false;
}

float neuron_layer::inverseNorm(size_t index)
{
    if (!normalized_)
        return 1.0f;
    if (norms_stale_ || inverse_norm_.size() != neurons_.size())
        refreshNorms();
    return inverse_norm_.at(index);
}

void neuron_layer::setBias(size_t index, float value)
{
    if (bias_.empty())
        throw std::logic_error("setBias: the layer's learning rule has no bias");
    bias_.at(index) = value;
}

std::vector<float> neuron_layer::feedbackRow(size_t index) const
{
    if (index >= neurons_.size())
        throw std::out_of_range("feedbackRow: no neuron " + std::to_string(index));
    if (index >= feedback_.rows())
        return {};
    std::vector<float> row(feedback_.cols());
    feedback_.copyRow(index, row);
    return row;
}

float neuron_layer::fireWithRule(size_t i, float sum)
{
    if (normalized_)
        sum *= inverse_norm_[i];
    if (!bias_.empty())
        sum += bias_[i];
    if (!noise_.empty())
        sum += noise_[i];
    if (!dyds_.empty()) {
        Derivatives d;
        const float y = neurons_[i].activate(sum, rule_.width, d);
        dyds_[i] = d.dyds;
        dydthr_[i] = d.dydthr;
        dthrdthr_[i] = d.dthrdthr;
        dthrds_[i] = d.dthrds;
        if (!bias_adapt_.empty()) {
            // The bias is a weight whose input is always 1.
            const float now = d.dyds + d.dydthr * bias_adapt_[i];
            bias_adapt_[i] = d.dthrdthr * bias_adapt_[i] + d.dthrds;
            bias_elig_[i] = rule_.trace * bias_elig_[i] + now;
        }
        return y;
    }
    const float y = neurons_[i].activate(sum);
    if (!bias_elig_.empty())
        bias_elig_[i] = rule_.trace * bias_elig_[i] + std::abs(y);
    if (!post_.empty()) {
        const float p = rule_.trace * post_[i] + y;
        post_[i] = p;
        if (!theta_.empty())
            theta_[i] += rule_.bcmRate * (p * p - theta_[i]);
    }
    return y;
}

void neuron_layer::drawNoise()
{
    // Serial, in neuron order, from the layer's own generator: the same
    // noise whatever the thread count. Each value is the sum of four 16-bit
    // uniforms from one xorshift64* draw, scaled to mean 0 and standard
    // deviation `noise`: close to normal (within +-3.46 sd) and much cheaper
    // than std::normal_distribution.
    const float scale = rule_.noise * std::sqrt(3.0f) / 65535.0f;
    const float lambda = rule_.trace;
    std::uint64_t x = noise_state_;
    for (size_t i = 0; i < noise_.size(); ++i) {
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        const std::uint64_t r = x * 0x2545F4914F6CDD1Dull;
        const auto sum = static_cast<float>((r & 0xFFFF) + ((r >> 16) & 0xFFFF) + ((r >> 32) & 0xFFFF) + (r >> 48));
        noise_[i] = (sum - 2.0f * 65535.0f) * scale;
        noise_trace_[i] = lambda * noise_trace_[i] + noise_[i];
    }
    noise_state_ = x;
}

void neuron_layer::traceInputs(std::vector<float>& trace, std::span<const float> x) const
{
    if (trace.size() != x.size())
        trace.resize(x.size(), 0.0f);
    const float lambda = rule_.trace;
    if (lambda == 0.0f) {
        std::copy(x.begin(), x.end(), trace.begin());
        return;
    }
    for (size_t j = 0; j < x.size(); ++j)
        trace[j] = lambda * trace[j] + x[j];
}

void neuron_layer::applyReward(float reward, float learningRate)
{
    learn([reward](size_t) { return reward; }, learningRate);
}

void neuron_layer::applyModulators(std::span<const float> modulators, float learningRate)
{
    if (modulators.size() != neurons_.size())
        throw std::invalid_argument("applyModulators: " + std::to_string(neurons_.size()) + " neurons, got " +
                                    std::to_string(modulators.size()) + " modulators");
    learn([modulators](size_t i) { return modulators[i]; }, learningRate);
}

void neuron_layer::applyFeedback(std::span<const float> errors, float learningRate)
{
    if (!learns() || errors.empty())
        return;
    const size_t n = neurons_.size(), cols = errors.size();
    if (feedback_.cols() != cols)
        feedback_ = kernels::weight_matrix();  // the outputs changed: a new matrix
    if (feedback_.rows() < n) {
        // Rows for neurons that have none yet, drawn row after row.
        std::uniform_real_distribution<float> distribution(-1.0f, 1.0f);
        std::vector<float> rows(n * cols);
        for (size_t r = 0; r < feedback_.rows(); ++r)
            feedback_.copyRow(r, std::span(rows).subspan(r * cols, cols));
        for (size_t k = feedback_.rows() * cols; k < rows.size(); ++k)
            rows[k] = distribution(rng::learning());
        feedback_ = kernels::weight_matrix(n, cols, rows);
    }
    scratch_.resize(feedback_.paddedRows());
    parallelChunks(feedback_.blocks(), kernels::threadsFor(n * cols), [&](size_t b0, size_t b1) {
        feedback_.multiply(errors, scratch_, b0, b1);
    });
    const float scale = 1.0f / std::sqrt(static_cast<float>(cols));
    learn([this, scale](size_t i) { return scale * scratch_[i]; }, learningRate);
}

template <typename Modulator>
void neuron_layer::learn(Modulator m, float learningRate)
{
    if (!learns())
        return;
    const size_t n = neurons_.size();
    const LearningRule& r = rule_;
    const bool scaled = scaledUpdates();
    // Every entry is written below: resize, no fill.
    step_delta_.resize(n);
    step_active_.resize(n);
    step_keep_.resize(scaled ? n : 0);
    const float shrink = std::max(0.0f, 1.0f - learningRate * r.decay);
    std::atomic<bool> any_active{false};

    parallelChunks(n, kernels::threadsFor(n * 16), [&](size_t i0, size_t i1) {
        bool any = false;
        if (!scaled) {
            // The original sign rule, free of the general case's work.
            for (size_t i = i0; i < i1; ++i) {
                const neuron& nr = neurons_[i];
                const bool active = nr.eligible();
                step_active_[i] = active;
                step_delta_[i] = active ? nr.learningDelta(m(i), learningRate) : 0.0f;
                any = any || active;
            }
        } else {
            for (size_t i = i0; i < i1; ++i) {
                float delta = 0.0f, keep = 1.0f;
                const bool active = ruleStep(i, m(i), learningRate, delta, keep);
                step_delta_[i] = delta;
                step_keep_[i] = active && r.decay > 0.0f ? keep * shrink : keep;
                step_active_[i] = active;
                any = any || active;
            }
        }
        if (any)
            any_active.store(true, std::memory_order_relaxed);
    });
    if (r.unsupervised() && r.winners > 0)
        selectWinners();
    for (size_t i = 0; i < n; ++i)
        if (frozen_[i])
            step_active_[i] = 0;
    any_active.store(std::ranges::any_of(step_active_, [](std::uint8_t a) { return a != 0; }),
                     std::memory_order_relaxed);
    // The baseline follows the modulator whether or not anything learned.
    if (!baseline_.empty() && r.baseline > 0.0f)
        for (size_t i = 0; i < n; ++i)
            baseline_[i] += r.baseline * (m(i) - baseline_[i]);

    if (!any_active.load(std::memory_order_relaxed))
        return;
    updateWeights();
    norms_stale_ = true;
    if (!bias_.empty())
        for (size_t i = 0; i < n; ++i)
            if (step_active_[i]) {
                // The bias's input is 1; per-synapse rules use its eligibility.
                const float pre = bias_elig_.empty() ? 1.0f : bias_elig_[i];
                bias_[i] = std::min(
                    std::max(bias_[i] * (scaled ? step_keep_[i] : 1.0f) + step_delta_[i] * pre, -max_weight), max_weight);
            }
}

void neuron_layer::applyBiasGradient(float learningRate)
{
    if (bias_.empty() || bias_grad_.empty())
        return;
    for (size_t i = 0; i < neurons_.size(); ++i) {
        if (!frozen_[i])
            bias_[i] = std::min(std::max(bias_[i] - learningRate * neurons_[i].learningGain() * bias_grad_[i],
                                         -max_weight), max_weight);
        bias_grad_[i] = 0.0f;
    }
}

void neuron_layer::eraseNeuronState(std::span<const size_t> indices)
{
    if (indices.empty())
        return;
    const auto erase = [indices](auto& v) {
        if (v.empty())
            return;
        size_t next = 0, out = 0;
        for (size_t i = 0; i < v.size(); ++i) {
            if (next < indices.size() && indices[next] == i) {
                ++next;
                continue;
            }
            v[out++] = v[i];
        }
        v.resize(out);
    };
    erase(neurons_);
    erase(output_);
    erase(frozen_);
    for (std::vector<float>* v : {&bias_, &post_, &noise_, &noise_trace_, &baseline_, &theta_, &dyds_, &dydthr_,
                                  &dthrdthr_, &dthrds_, &bias_elig_, &bias_adapt_, &bias_grad_, &inverse_norm_})
        erase(*v);
    std::vector<size_t> rows;
    for (size_t i : indices)
        if (i < feedback_.rows())
            rows.push_back(i);
    feedback_.removeRows(rows);
    norms_stale_ = true;
}

bool neuron_layer::ruleStep(size_t i, float m, float learningRate, float& delta, float& keep) const
{
    const neuron& nr = neurons_[i];
    const LearningRule& r = rule_;
    const float rate = learningRate * nr.learningGain();
    switch (r.type) {
    case LearningRuleType::Sign:
        if (!nr.eligible())
            return false;
        delta = nr.learningDelta(m, learningRate);
        return true;
    case LearningRuleType::Trace:
        // |P|: how active the neuron was, whatever the sign. The modulator
        // is the direction to move the output (like Sign).
        delta = rate * (m - (r.baseline > 0.0f ? baseline_[i] : 0.0f)) * std::abs(post_[i]);
        return delta != 0.0f;
    case LearningRuleType::FeedbackAlignment: {
        // Surrogate derivative of the neuron: 1 while it takes part (with
        // E-R: eligible, i.e. fired recently; with a gate or rectification:
        // firing; otherwise always, its output is its clamped sum), 0 when
        // silent or held at the output clamp in the direction the modulator
        // pushes.
        const float y = nr.output();
        if ((nr.hasER() && !nr.eligible()) || ((nr.gate() > 0.0f || nr.rectified()) && y == 0.0f) || (y >= max_output && m > 0.0f) || (y <= -max_output && m < 0.0f))
            return false;
        delta = rate * m;
        return true;
    }
    case LearningRuleType::Perturbation:
        delta = rate * (m - (r.baseline > 0.0f ? baseline_[i] : 0.0f)) * noise_trace_[i] / r.noise;
        return delta != 0.0f;
    case LearningRuleType::Oja:
        delta = rate * post_[i];
        keep = std::max(0.0f, 1.0f - rate * post_[i] * post_[i]);
        return true;
    case LearningRuleType::BCM:
        delta = rate * post_[i] * (post_[i] - theta_[i]);
        return true;
    case LearningRuleType::Eligibility:
    case LearningRuleType::EProp:
        // The per-synapse traces (the derived layer's) carry the credit.
        delta = rate * (m - (r.baseline > 0.0f ? baseline_[i] : 0.0f));
        return delta != 0.0f;
    case LearningRuleType::Surrogate:
        return false;  // gradients come from network::applyError (backpropagation through time)
    }
    return false;
}

void neuron_layer::selectWinners()
{
    // Per competition group (same index modulo `positions`), only the
    // `winners` neurons with the largest |P| learn; ties go to the lower index.
    const size_t positions = std::max<size_t>(1, competitionPositions());
    const size_t n = neurons_.size();
    const size_t members = n / positions;
    if (members <= rule_.winners)
        return;
    parallelChunks(positions, kernels::threadsFor(n * 8), [&](size_t p0, size_t p1) {
        std::vector<size_t> group(members);
        for (size_t p = p0; p < p1; ++p) {
            for (size_t k = 0; k < members; ++k)
                group[k] = k * positions + p;
            const auto stronger = [this](size_t a, size_t b) {
                const float fa = std::abs(post_[a]), fb = std::abs(post_[b]);
                return fa != fb ? fa > fb : a < b;
            };
            std::nth_element(group.begin(), group.begin() + rule_.winners, group.end(), stronger);
            for (size_t k = rule_.winners; k < members; ++k)
                step_active_[group[k]] = 0;
        }
    });
}

// --- Serialization of the learning state -----------------------------------------

namespace {

void writeFloats(std::ostream& os, const std::vector<float>& v)
{
    binary_io::write(os, static_cast<std::uint64_t>(v.size()));
    binary_io::write(os, std::span<const float>(v));
}

std::vector<float> readFloats(std::istream& is)
{
    const auto count = binary_io::read<std::uint64_t>(is);
    if (!is)
        return {};
    if (count > max_serialized_weights)
        throw std::runtime_error("layer deserialize: implausible learning state size " + std::to_string(count));
    std::vector<float> v(static_cast<size_t>(count));
    binary_io::read(is, std::span<float>(v));
    return v;
}

} // namespace

void neuron_layer::writeLearningState(std::ostream& os) const
{
    binary_io::write(os, static_cast<std::uint8_t>(rule_.type));
    binary_io::write(os, static_cast<std::uint8_t>(rule_.bias));
    binary_io::write(os, rule_.decay);
    binary_io::write(os, rule_.trace);
    binary_io::write(os, rule_.baseline);
    binary_io::write(os, rule_.noise);
    binary_io::write(os, rule_.bcmRate);
    binary_io::write(os, rule_.winners);
    std::vector<float> feedback(feedback_.rows() * feedback_.cols());
    for (size_t r = 0; r < feedback_.rows(); ++r)
        feedback_.copyRow(r, std::span(feedback).subspan(r * feedback_.cols(), feedback_.cols()));
    const std::vector<float>* const state[] = {&bias_, &post_, &noise_, &noise_trace_, &baseline_, &theta_, &feedback};
    for (const std::vector<float>* v : state)
        writeFloats(os, *v);
    binary_io::write(os, static_cast<std::uint64_t>(feedback_.cols()));
    std::vector<float> traces;
    copyInputTraces(traces);
    writeFloats(os, traces);
    binary_io::write(os, noise_state_);
    // Neuron format 4: the per-synapse rules' parameters and the bias's
    // eligibility, and which neurons are frozen.
    binary_io::write(os, rule_.width);
    binary_io::write(os, rule_.window);
    writeFloats(os, bias_elig_);
    writeFloats(os, bias_adapt_);
    std::vector<float> frozen(frozen_.begin(), frozen_.end());
    writeFloats(os, frozen);
}

void neuron_layer::readLearningState(std::istream& is, DeserializeMode mode, std::uint32_t neuronFormat)
{
    LearningRule rule;
    const auto type = binary_io::read<std::uint8_t>(is);
    rule.type = static_cast<LearningRuleType>(type);
    rule.bias = binary_io::read<std::uint8_t>(is) != 0;
    rule.decay = binary_io::read<float>(is);
    rule.trace = binary_io::read<float>(is);
    rule.baseline = binary_io::read<float>(is);
    rule.noise = binary_io::read<float>(is);
    rule.bcmRate = binary_io::read<float>(is);
    rule.winners = binary_io::read<std::uint32_t>(is);
    if (!is)
        return;
    try {
        rule.validate();
    } catch (const std::invalid_argument& e) {
        throw std::runtime_error(std::string("layer deserialize: ") + e.what());
    }
    std::vector<float> state[7];
    for (std::vector<float>& v : state)
        v = readFloats(is);
    const auto feedback_cols = binary_io::read<std::uint64_t>(is);
    std::vector<float> traces = readFloats(is);
    const auto rng_state = binary_io::read<std::uint64_t>(is);
    std::vector<float> bias_elig, bias_adapt, frozen;
    if (neuronFormat >= 4) {
        rule.width = binary_io::read<float>(is);
        rule.window = binary_io::read<std::uint32_t>(is);
        bias_elig = readFloats(is);
        bias_adapt = readFloats(is);
        frozen = readFloats(is);
        if (is) {
            try {
                rule.validate();
            } catch (const std::invalid_argument& e) {
                throw std::runtime_error(std::string("layer deserialize: ") + e.what());
            }
        }
    }
    if (!is)
        return;
    if (rng_state == 0)
        throw std::runtime_error("layer deserialize: invalid noise generator state 0");
    const size_t n = neurons_.size();
    for (size_t k = 0; k < 6; ++k)
        if (!state[k].empty() && state[k].size() != n)
            throw std::runtime_error("layer deserialize: learning state does not match the neuron count");
    if (feedback_cols == 0 ? !state[6].empty() : state[6].size() % feedback_cols != 0)
        throw std::runtime_error("layer deserialize: feedback matrix does not match its width");

    resetLearningState(rule);  // sizes and initialises every per-neuron vector
    const auto restore = [n](std::vector<float>& to, std::vector<float>& from) {
        if (!to.empty() && from.size() == n)
            to = std::move(from);
    };
    restore(bias_, state[0]);
    if (frozen.size() == n)
        for (size_t i = 0; i < n; ++i)
            frozen_[i] = frozen[i] != 0.0f ? 1 : 0;
    if (feedback_cols > 0)
        feedback_ = kernels::weight_matrix(state[6].size() / feedback_cols, static_cast<size_t>(feedback_cols), state[6]);
    if (mode == DeserializeMode::FullState) {
        restore(post_, state[1]);
        restore(noise_, state[2]);
        restore(noise_trace_, state[3]);
        restore(baseline_, state[4]);
        restore(theta_, state[5]);
        storeInputTraces(traces);
        noise_state_ = rng_state;
        restore(bias_elig_, bias_elig);
        restore(bias_adapt_, bias_adapt);
    }
}

} // namespace exr
