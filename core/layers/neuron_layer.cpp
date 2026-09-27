#include "neuron_layer.hpp"
#include "../binary_io.hpp"
#include "../kernels.hpp"
#include "../parallel.hpp"
#include "../random.hpp"
#include <algorithm>
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
    output_.push_back(n.output());
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
        readLearningState(is, mode);
    else
        resizeLearningState();
}

// --- Learning rules -----------------------------------------------------------

void neuron_layer::setLearningRule(const LearningRule& rule)
{
    rule.validate();
    if (rule.type != LearningRuleType::Sign && !learns())
        throw std::invalid_argument("setLearningRule: this layer type does not learn");
    resetLearningState(rule);
    if (rule_.type == LearningRuleType::Perturbation)
        noise_rng_.seed(rng::learning()() % (std::minstd_rand::modulus - 1) + 1);
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
    feedback_.clear();
    feedback_cols_ = 0;
    clearInputTraces();
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
    baseline_.resize(t == LearningRuleType::Trace || perturb ? n : 0, 0.0f);
    theta_.resize(t == LearningRuleType::BCM ? n : 0, 1.0f);
    plain_ = bias_.empty() && post_.empty() && noise_.empty();
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
    if (feedback_cols_ == 0 || (index + 1) * feedback_cols_ > feedback_.size())
        return {};
    const auto first = feedback_.begin() + static_cast<std::ptrdiff_t>(index * feedback_cols_);
    return {first, first + static_cast<std::ptrdiff_t>(feedback_cols_)};
}

float neuron_layer::fireWithRule(size_t i, float sum)
{
    if (!bias_.empty())
        sum += bias_[i];
    if (!noise_.empty())
        sum += noise_[i];
    const float y = neurons_[i].activate(sum);
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
    // noise whatever the thread count.
    std::normal_distribution<float> distribution(0.0f, rule_.noise);
    for (size_t i = 0; i < noise_.size(); ++i) {
        noise_[i] = distribution(noise_rng_);
        noise_trace_[i] = rule_.trace * noise_trace_[i] + noise_[i];
    }
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
    std::uniform_real_distribution<float> distribution(-1.0f, 1.0f);
    if (feedback_cols_ != cols) {
        feedback_.clear();
        feedback_cols_ = cols;
    }
    // Rows for neurons that have none yet, drawn row after row.
    const size_t have = feedback_.size() / cols;
    feedback_.resize(std::max(have, n) * cols);
    for (size_t k = have * cols; k < feedback_.size(); ++k)
        feedback_[k] = distribution(rng::learning());

    const float scale = 1.0f / std::sqrt(static_cast<float>(cols));
    scratch_.resize(n);
    parallelChunks(n, kernels::threadsFor(n * cols), [&](size_t i0, size_t i1) {
        for (size_t i = i0; i < i1; ++i)
            scratch_[i] = scale * kernels::dot(errors, std::span<const float>(feedback_).subspan(i * cols, cols));
    });
    learn([this](size_t i) { return scratch_[i]; }, learningRate);
}

template <typename Modulator>
void neuron_layer::learn(Modulator m, float learningRate)
{
    if (!learns())
        return;
    const size_t n = neurons_.size();
    step_delta_.assign(n, 0.0f);
    step_keep_.assign(n, 1.0f);
    step_active_.assign(n, 0);
    const LearningRule& r = rule_;
    const float shrink = std::max(0.0f, 1.0f - learningRate * r.decay);

    parallelChunks(n, kernels::threadsFor(n * 16), [&](size_t i0, size_t i1) {
        for (size_t i = i0; i < i1; ++i) {
            const neuron& nr = neurons_[i];
            const float rate = learningRate * nr.learningGain();
            float delta = 0.0f, keep = 1.0f;
            bool active = false;
            switch (r.type) {
            case LearningRuleType::Sign:
                active = nr.eligible();
                if (active)
                    delta = nr.learningDelta(m(i), learningRate);
                break;
            case LearningRuleType::Trace:
                delta = rate * (m(i) - (r.baseline > 0.0f ? baseline_[i] : 0.0f)) * post_[i];
                active = delta != 0.0f;
                break;
            case LearningRuleType::FeedbackAlignment: {
                // Surrogate derivative of the neuron: 1 while it takes part
                // (eligible: fired recently with E-R, a non-zero output
                // without), 0 when silent or held at the output clamp in the
                // direction the modulator pushes.
                const float mi = m(i), y = nr.output();
                active = nr.eligible() && !(y >= max_output && mi > 0.0f) && !(y <= -max_output && mi < 0.0f);
                delta = active ? rate * mi : 0.0f;
                break;
            }
            case LearningRuleType::Perturbation:
                delta = rate * (m(i) - (r.baseline > 0.0f ? baseline_[i] : 0.0f)) * noise_trace_[i] / r.noise;
                active = delta != 0.0f;
                break;
            case LearningRuleType::Oja:
                delta = rate * post_[i];
                keep = std::max(0.0f, 1.0f - rate * post_[i] * post_[i]);
                active = true;
                break;
            case LearningRuleType::BCM:
                delta = rate * post_[i] * (post_[i] - theta_[i]);
                active = true;
                break;
            }
            if (active && r.decay > 0.0f)
                keep *= shrink;
            step_delta_[i] = delta;
            step_keep_[i] = keep;
            step_active_[i] = active;
        }
    });
    if (r.unsupervised() && r.winners > 0)
        selectWinners();
    // The baseline follows the modulator whether or not anything learned.
    if (!baseline_.empty() && r.baseline > 0.0f)
        for (size_t i = 0; i < n; ++i)
            baseline_[i] += r.baseline * (m(i) - baseline_[i]);

    if (std::ranges::none_of(step_active_, [](std::uint8_t a) { return a != 0; }))
        return;
    updateWeights();
    if (!bias_.empty())
        for (size_t i = 0; i < n; ++i)
            if (step_active_[i])
                bias_[i] = std::min(std::max(bias_[i] * step_keep_[i] + step_delta_[i], -max_weight), max_weight);
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
    for (const std::vector<float>* v : {&bias_, &post_, &noise_, &noise_trace_, &baseline_, &theta_, &feedback_})
        writeFloats(os, *v);
    binary_io::write(os, static_cast<std::uint64_t>(feedback_cols_));
    std::vector<float> traces;
    copyInputTraces(traces);
    writeFloats(os, traces);
    std::ostringstream rng_text;
    rng_text << noise_rng_;
    binary_io::write(os, static_cast<std::uint32_t>(std::stoul(rng_text.str())));
}

void neuron_layer::readLearningState(std::istream& is, DeserializeMode mode)
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
    const auto rng_state = binary_io::read<std::uint32_t>(is);
    if (!is)
        return;
    if (rng_state == 0 || rng_state >= std::minstd_rand::modulus)
        throw std::runtime_error("layer deserialize: invalid noise generator state " + std::to_string(rng_state));
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
    feedback_ = std::move(state[6]);
    feedback_cols_ = static_cast<size_t>(feedback_cols);
    if (mode == DeserializeMode::FullState) {
        restore(post_, state[1]);
        restore(noise_, state[2]);
        restore(noise_trace_, state[3]);
        restore(baseline_, state[4]);
        restore(theta_, state[5]);
        storeInputTraces(traces);
        std::istringstream(std::to_string(rng_state)) >> noise_rng_;
    }
}

} // namespace exr
