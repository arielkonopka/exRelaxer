#include "neuron.hpp"
#include "binary_io.hpp"
#include "kernels.hpp"
#include "random.hpp"
#include <algorithm>
#include <cmath>
#include <istream>
#include <ostream>
#include <sstream>
#include <stdexcept>
#include <string>

namespace exr {
namespace {

// Draws from `jitter` around `fallbackMean`, redrawing values outside the
// intersection of the jitter's limits and the parameter's valid range.
// `scaleOf(centre)` gives the parameter's scale for relative spreads.
template <typename Scale>
float drawJitter(const Jitter& jitter, float fallbackMean, float validMin, float validMax, Scale scaleOf)
{
    const float centre = jitter.mean.value_or(fallbackMean);
    const float spread = jitter.relative ? jitter.spread * scaleOf(centre) : jitter.spread;
    const float lo = std::max(jitter.min, validMin);
    const float hi = std::min(jitter.max, validMax);
    float value = centre;
    for (int attempt = 0; attempt < 1000; ++attempt) {
        if (jitter.distribution == Jitter::Distribution::Uniform) {
            std::uniform_real_distribution<float> d(centre - spread, centre + spread);
            value = d(rng::jitter());
        } else {
            std::normal_distribution<float> d(centre, spread);
            value = d(rng::jitter());
        }
        if (value >= lo && value <= hi)
            return value;
    }
    return std::clamp(value, lo, hi);  // limits (almost) unreachable: fall back to the nearest edge
}

using binary_io::read;
using binary_io::write;

} // namespace

neuron::neuron(bool hasHabituation, bool hasER, float alpha)
    : threshold_(baseline_threshold), has_habituation_(hasHabituation), has_er_(hasER), alpha_(alpha),
      rng_(rng::spontaneousSeed())
{
}

void neuron::randomizeRecovery(const Jitter& jitter)
{
    // Relative spreads scale with the distance from 1 (the relaxation speed),
    // so they stay symmetric and valid.
    recovery_ = jitter.enabled()
        ? drawJitter(jitter, recovery_factor, 0.01f, 0.999f, [](float c) { return 1.0f - c; })
        : std::clamp(jitter.mean.value_or(recovery_factor), 0.01f, 0.999f);
}

void neuron::randomizeLearningGain(const Jitter& jitter)
{
    learning_gain_ = jitter.enabled()
        ? drawJitter(jitter, default_learning_gain, 0.0f, std::numeric_limits<float>::infinity(),
                     [](float c) { return std::abs(c); })
        : std::max(jitter.mean.value_or(default_learning_gain), 0.0f);
}

void neuron::randomizeAlpha(const Jitter& jitter)
{
    alpha_ = jitter.enabled()
        ? drawJitter(jitter, default_alpha, 0.0f, std::numeric_limits<float>::infinity(),
                     [](float c) { return std::abs(c); })
        : std::max(jitter.mean.value_or(default_alpha), 0.0f);
}

void neuron::randomizeDynamics(const Jitter& recovery, const Jitter& learning, const Jitter& alpha)
{
    randomizeRecovery(recovery);
    randomizeLearningGain(learning);
    randomizeAlpha(alpha);
}

float neuron::activate(float weightedSum)
{
    // Bounded before habituation and E-R see it, so recurrent loops can't
    // run away and the threshold (grown from this value) stays finite.
    float sum = std::clamp(weightedSum, -max_output, max_output);
    last_sum_ = sum;
    last_threshold_ = threshold_;

    if (has_habituation_) {
        // Branchless habituation update:
        //   similar    == is this step's raw sum ~equal to last step's?
        //   counter    == similar ? (counter + 1) : 0          [streak length]
        //   habituated == counter has reached the rule's steps
        //   sum        == habituated ? 0 : sum                 [suppress the input]
        // With a decay, a habituated input fades by decay per tick instead
        // of being cut, starting at the fadeAfter-th repeat (Habituation::onset). previous_input_ exists purely for habituation's own
        // repeat detection and is not read anywhere else.
        const float tolerance = std::max(habituation_epsilon,
                                         habituation_.tolerance * std::max(std::abs(sum), std::abs(previous_input_)));
        const bool similar = std::abs(sum - previous_input_) <= tolerance;
        habituation_counter_ = static_cast<int>(similar) * (habituation_counter_ + 1);
        previous_input_ = sum;
        const int habituatedTicks = habituation_counter_ - static_cast<int>(habituation_.onset()) + 1;
        if (habituatedTicks > 0)
            sum *= habituation_.decay > 0.0f ? std::pow(habituation_.decay, static_cast<float>(habituatedTicks)) : 0.0f;
    }
    output_ = sum;
    if (has_er_) {
        // Firing decision and threshold growth are both magnitude-based:
        // inputs and outputs can be negative, the threshold is always positive.
        if (std::abs(sum) > threshold_) {
            excite(sum);
            last_firing_ = Firing::Fired;
        } else {
            output_ = 0.0f;
            last_firing_ = Firing::Silent;
            threshold_ *= recovery_;
            // Draws only when a rate is set, so the default keeps the
            // generator's sequence.
            const bool chance = spontaneous_.rate > 0.0f &&
                                std::uniform_real_distribution<float>(0.0f, 1.0f)(rng_) < spontaneous_.rate;
            if (threshold_ <= spontaneous_.below || chance) {
                // Effective input has been ~0 long enough to decay the
                // threshold (nothing coming in, or habituation suppressing a
                // repeating signal), or the random chance came up. The neuron
                // fires spontaneously, which re-excites the threshold through
                // excite().
                excite(spontaneousOutput());
                last_firing_ = Firing::Spontaneous;
            }
        }
        return output_;
    } else if (rectified_ ? sum <= gate_ : gate_ > 0.0f && std::abs(sum) <= gate_) {
        output_ = 0.0f;  // fixed threshold (one-sided when rectified): all-or-nothing, without adaptation
    } else if (binary_) {
        output_ = kernels::sign(sum);  // a perceptron: whether it passed, not by how much
    }
    last_firing_ = output_ != 0.0f ? Firing::Fired : Firing::Silent;
    return output_;
}

float neuron::activate(float weightedSum, float width, Derivatives& d)
{
    const float threshold = threshold_;
    const float y = activate(weightedSum);
    d = {};

    // How the effective sum (after the clamp and habituation) moves with
    // the given one.
    float chain = std::abs(weightedSum) <= max_output ? 1.0f : 0.0f;
    float s = std::clamp(weightedSum, -max_output, max_output);
    if (has_habituation_) {
        const int habituatedTicks = habituation_counter_ - static_cast<int>(habituation_.onset()) + 1;
        if (habituatedTicks > 0) {
            const float factor =
                habituation_.decay > 0.0f ? std::pow(habituation_.decay, static_cast<float>(habituatedTicks)) : 0.0f;
            chain *= factor;
            s *= factor;
        }
    }
    const float a = std::abs(s);
    const float sg = s >= 0.0f ? 1.0f : -1.0f;
    // Triangular pseudo-derivative of a step at |s| = edge, integrating to
    // 1; its half-width is `width` times `scale` (the edge, or the resting
    // threshold for a step at 0).
    const auto bump = [width, a](float edge, float scale) {
        const float half = width * std::max(scale, min_threshold);
        return std::max(0.0f, 1.0f - std::abs(a - edge) / half) / half;
    };

    if (has_er_) {
        const bool fired = a > threshold;
        const float b = bump(threshold, threshold);
        d.dyds = (fired ? 1.0f : 0.0f) + a * b;  // y = s * H(|s| - thr)
        d.dydthr = -s * b;
        if (fired) {
            // Which term of max(thr, 2 * resting, grown) set the new threshold.
            const float floor = std::max(threshold, resting_ * 2);
            float grown = threshold, dthr = 0.0f, ds = 0.0f;
            switch (growth_.rule) {
            case ThresholdGrowth::Rule::Log:
                grown = threshold + alpha_ * std::log(a / threshold);
                dthr = 1.0f - alpha_ / threshold;
                ds = alpha_ / a * sg;
                break;
            case ThresholdGrowth::Rule::Linear:
                grown = threshold + growth_.amount * (a - threshold);
                dthr = 1.0f - growth_.amount;
                ds = growth_.amount * sg;
                break;
            case ThresholdGrowth::Rule::Fixed:
                grown = threshold + growth_.amount;
                dthr = 1.0f;
                break;
            case ThresholdGrowth::Rule::Multiplicative:
                grown = threshold * (1.0f + growth_.amount);
                dthr = 1.0f + growth_.amount;
                break;
            }
            if (grown >= floor) {
                d.dthrdthr = dthr;
                d.dthrds = ds;
            } else if (threshold >= resting_ * 2) {
                d.dthrdthr = 1.0f;
            }
        } else {
            d.dthrdthr = recovery_;
        }
    } else {
        const bool gated = rectified_ || gate_ > 0.0f;
        const bool passed = rectified_ ? s > gate_ : a > gate_;
        // The step at the gate (at 0 without one): one-sided when rectified
        // (only at +gate).
        const float b = !rectified_ || s >= 0.0f ? bump(gate_, gate_ > 0.0f ? gate_ : resting_) : 0.0f;
        if (binary_)
            d.dyds = b;  // y = sign(s) * H(|s| - gate): only the step moves it
        else
            d.dyds = gated ? (passed ? 1.0f : 0.0f) + a * b : 1.0f;
    }
    d.dyds *= chain;
    d.dthrds *= chain;
    return y;
}

void neuron::excite(float effectiveSum)
{
    // Sets the output to the (signed) effective sum and grows the
    // (always-positive) threshold with the firing's magnitude.
    output_ = effectiveSum;
    const float s = std::abs(effectiveSum);
    float grown = threshold_;
    switch (growth_.rule) {
    case ThresholdGrowth::Rule::Log:
        grown = threshold_ + alpha_ * std::log(s / threshold_);
        break;
    case ThresholdGrowth::Rule::Linear:
        grown = threshold_ + growth_.amount * (s - threshold_);
        break;
    case ThresholdGrowth::Rule::Fixed:
        grown = threshold_ + growth_.amount;
        break;
    case ThresholdGrowth::Rule::Multiplicative:
        grown = threshold_ * (1.0f + growth_.amount);
        break;
    }
    // A real firing always exceeds the threshold, so every rule raises it; a
    // spontaneous one may be weaker, and then leaves it where it is.
    threshold_ = std::max({threshold_, resting_ * 2, grown});
}

float neuron::spontaneousOutput()
{
    std::uniform_real_distribution<float> distribution(-spontaneous_.amplitude, spontaneous_.amplitude);
    return distribution(rng_);
}

bool neuron::eligible() const
{
    // With E-R, a threshold still above baseline means the neuron fired
    // recently: a zero output this tick can be E-R's own refractory
    // relaxation after a real firing, and the neuron still took part in the
    // decision. Without E-R the threshold never moves, so the question is
    // simply whether it output something this tick.
    return has_er_ ? threshold_ > resting_ : std::abs(output_) > firing_epsilon;
}

float neuron::learningDelta(float reward, float learningRate) const
{
    // Eligibility with E-R: how far above baseline the threshold still sits,
    // as a ratio. The threshold decays exponentially every quiet tick, so
    // recent firings dominate without any extra state.
    const float eligibility = has_er_ ? threshold_ / resting_ - 1.0f : 1.0f;
    return learningRate * learning_gain_ * reward * eligibility;
}

float neuron::step(std::span<const float> inputs, std::span<const float> weights)
{
    return activate(kernels::dot(inputs.first(std::min(inputs.size(), weights.size())), weights));
}

void neuron::learn(std::span<float> weights, std::span<const float> inputs, float reward, float learningRate) const
{
    if (eligible())
        kernels::signRule(weights.first(std::min(inputs.size(), weights.size())), inputs,
                          learningDelta(reward, learningRate), max_weight);
}

void neuron::serialize(std::ostream& os, std::span<const float> weights) const
{
    // 1. Flags and hyperparameters
    write(os, has_habituation_);
    write(os, has_er_);
    write(os, alpha_);

    // 2. Weights
    const size_t weight_count = weights.size();
    write(os, weight_count);
    write(os, weights);

    // 3. Internal state
    write(os, threshold_);
    write(os, previous_input_);
    write(os, habituation_counter_);
    write(os, output_);

    // 4. Spontaneous-firing generator. minstd_rand exposes its state only as
    // text; that state is a single value below 2^31, stored as 4 bytes.
    std::ostringstream rng_text;
    rng_text << rng_;
    write(os, static_cast<std::uint32_t>(std::stoul(rng_text.str())));

    // 5. Per-neuron dynamics
    write(os, recovery_);
    write(os, learning_gain_);
}

std::vector<float> neuron::deserialize(std::istream& is, DeserializeMode mode, std::uint32_t format)
{
    // Everything is read and checked before the neuron changes.
    const auto has_habituation = read<bool>(is);
    const auto has_er = read<bool>(is);
    const auto alpha = read<float>(is);

    const auto weight_count = read<size_t>(is);
    if (weight_count > max_serialized_weights)
        throw std::runtime_error("neuron::deserialize: implausible weight count " + std::to_string(weight_count));
    std::vector<float> weights(is ? weight_count : 0);
    read(is, std::span<float>(weights));

    const auto threshold = read<float>(is);
    const auto previous_input = read<float>(is);
    const auto habituation_counter = read<int>(is);
    const auto output = read<float>(is);
    const auto rng_state = read<std::uint32_t>(is);
    // minstd_rand's state is in [1, 2^31 - 2]; anything else is corrupt data.
    if (is && (rng_state == 0 || rng_state >= std::minstd_rand::modulus))
        throw std::runtime_error("neuron::deserialize: invalid generator state " + std::to_string(rng_state));

    float recovery = recovery_, learning_gain = learning_gain_;
    if (format >= 2) {
        recovery = read<float>(is);
        learning_gain = read<float>(is);
    }

    // Parameters are restored in both modes.
    has_habituation_ = has_habituation;
    has_er_ = has_er;
    alpha_ = alpha;
    recovery_ = recovery;
    learning_gain_ = learning_gain;
    if (mode == DeserializeMode::FullState) {
        threshold_ = threshold;
        previous_input_ = previous_input;
        habituation_counter_ = habituation_counter;
        output_ = output;
        std::istringstream(std::to_string(rng_state)) >> rng_;
    } else {
        threshold_ = resting_;
        previous_input_ = 0.0f;
        habituation_counter_ = 0;
        output_ = 0.0f;
    }
    return weights;
}

} // namespace exr
