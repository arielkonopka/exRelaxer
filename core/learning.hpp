// Learning rules a layer can use, chosen per layer (LayerSpec::learningRule,
// network::setLearningRule). Every rule updates a neuron's weights as
//
//   w[j] = clamp(w[j] * keep[i] + delta[i] * pre[j], +-max_weight)
//
// for each neuron i that learns this call: a per-neuron step delta, a
// per-input factor pre and a per-neuron shrink factor keep (1 unless the rule
// or weight decay shrinks weights). The rules differ in how they compute
// them from the neuron's modulator m (the reward, or its share of an error):
//
//   Sign                the original rule: delta = rate * gain * m * eligibility,
//                       pre = sign(input now); only E-R-eligible neurons learn.
//   Trace               graded three-factor rule: delta = rate * gain * (m - b) * |P|,
//                       pre = X, where P and X are traces of the neuron's output
//                       and of its inputs and b a running reward baseline. Like
//                       Sign, the modulator is the direction to move the output
//                       in; |P| grades how much the neuron took part.
//   FeedbackAlignment   per-neuron credit from an error vector (network::applyError):
//                       output layers get their own error, hidden layers
//                       m = B * error through a fixed random matrix B;
//                       delta = rate * gain * m, pre = X.
//   Perturbation        node perturbation: forward adds noise xi to each
//                       neuron's sum; delta = rate * gain * (m - b) * Z / noise,
//                       pre = X, Z a trace of xi. Per-neuron credit from a
//                       scalar reward.
//   Oja                 unsupervised: delta = rate * gain * P, keep = 1 - rate * gain * P^2.
//                       The reward is ignored; the weights stay bounded by themselves.
//   BCM                 unsupervised: delta = rate * gain * P * (P - theta), theta
//                       a sliding average of P^2.
//
// Three rules keep a value per synapse (an eligibility trace or a gradient)
// and are for Dense layers only:
//
//   Eligibility         reward-modulated eligibility traces (three-factor):
//                       e[i][j] = trace * e[i][j] + |y_i| * x_j every tick, and
//                       w[i][j] += rate * gain * (m - b) * e[i][j]. Unlike Trace,
//                       which multiplies a trace of the output by a trace of
//                       the input, it remembers which input was active when
//                       the neuron fired.
//   EProp               e-prop (Bellec et al. 2020) for E-R neurons: an online
//                       gradient that follows how each weight moved the
//                       neuron's adaptive threshold. Per synapse, with the
//                       neuron's local derivatives (see Derivatives in
//                       neuron.hpp) and a threshold eligibility a:
//                         e = dy/ds * x + dy/dthr * a,  a = dthr'/dthr * a + dthr'/ds * x,
//                         E = trace * E + e,   w += rate * gain * (m - b) * E.
//                       m is the neuron's learning signal: its own error on
//                       an output layer, a fixed random projection of the
//                       errors on a hidden one (network::applyError), or the
//                       reward (applyReward; e.g. a TD error, see Critic).
//   Surrogate           surrogate gradients: truncated backpropagation through
//                       time over the last `window` ticks, through the same
//                       local derivatives, through the E-R thresholds and
//                       through the weights between Surrogate layers (also
//                       recurrent ones). Only network::applyError drives it.
//
// Traces: P = trace * P + output and X = trace * X + input every tick
// (trace = 0: this tick's values). Options for every rule: a learned bias
// per neuron (its input is a constant 1), weight decay (keep *= 1 - rate *
// decay for neurons that learn), and for Oja / BCM, competition (only the
// `winners` most active neurons learn, per position in spatial layers).
#pragma once
#include <cstddef>
#include <cstdint>
#include <string>

namespace exr {

// Saved in network files: never renumber.
enum class LearningRuleType : std::uint8_t {
    Sign = 0,
    Trace = 1,
    FeedbackAlignment = 2,
    Perturbation = 3,
    Oja = 4,
    BCM = 5,
    Eligibility = 6,
    EProp = 7,
    Surrogate = 8
};

struct LearningRule
{
    LearningRuleType type = LearningRuleType::Sign;
    bool bias = false;      // a learned bias per neuron
    float decay = 0.0f;     // weight decay per update, times the learning rate (>= 0)
    float trace = 0.0f;     // trace decay per tick, [0, 1); not used by Sign
    float baseline = 0.0f;  // reward-baseline rate, [0, 1]; 0 = no baseline (Trace, Perturbation)
    float noise = 0.1f;     // Perturbation: standard deviation of the exploration noise (> 0)
    float bcmRate = 0.01f;  // BCM: rate of the sliding threshold, (0, 1]
    std::uint32_t winners = 0;  // Oja / BCM: only this many most active neurons learn; 0 = all
    float width = 0.5f;     // EProp, Surrogate: pseudo-derivative half-width, relative to the threshold (> 0)
    std::uint32_t window = 8;  // Surrogate: ticks backpropagated through, [1, 1024]

    static LearningRule sign() { return {}; }
    static LearningRule traced(float trace = 0.0f, float baseline = 0.05f)
    {
        LearningRule r;
        r.type = LearningRuleType::Trace;
        r.trace = trace;
        r.baseline = baseline;
        return r;
    }
    static LearningRule feedbackAlignment(float trace = 0.0f)
    {
        LearningRule r;
        r.type = LearningRuleType::FeedbackAlignment;
        r.trace = trace;
        return r;
    }
    static LearningRule perturbation(float noise = 0.1f, float trace = 0.0f, float baseline = 0.05f)
    {
        LearningRule r;
        r.type = LearningRuleType::Perturbation;
        r.noise = noise;
        r.trace = trace;
        r.baseline = baseline;
        return r;
    }
    static LearningRule oja(std::uint32_t winners = 0)
    {
        LearningRule r;
        r.type = LearningRuleType::Oja;
        r.winners = winners;
        return r;
    }
    static LearningRule bcm(float bcmRate = 0.01f, std::uint32_t winners = 0, float decay = 0.0f)
    {
        LearningRule r;
        r.type = LearningRuleType::BCM;
        r.bcmRate = bcmRate;
        r.winners = winners;
        r.decay = decay;
        return r;
    }
    static LearningRule eligibility(float trace = 0.9f, float baseline = 0.05f)
    {
        LearningRule r;
        r.type = LearningRuleType::Eligibility;
        r.trace = trace;
        r.baseline = baseline;
        return r;
    }
    static LearningRule eprop(float trace = 0.0f, float width = 0.5f, float baseline = 0.0f)
    {
        LearningRule r;
        r.type = LearningRuleType::EProp;
        r.trace = trace;
        r.width = width;
        r.baseline = baseline;
        return r;
    }
    static LearningRule surrogate(std::uint32_t window = 8, float width = 0.5f)
    {
        LearningRule r;
        r.type = LearningRuleType::Surrogate;
        r.window = window;
        r.width = width;
        return r;
    }
    LearningRule withBias(bool on = true) const { LearningRule r = *this; r.bias = on; return r; }
    LearningRule withDecay(float value) const { LearningRule r = *this; r.decay = value; return r; }

    // Whether the rule ignores the reward (Oja, BCM).
    bool unsupervised() const { return type == LearningRuleType::Oja || type == LearningRuleType::BCM; }
    // Whether the rule keeps traces of outputs and inputs (Trace,
    // FeedbackAlignment, Perturbation, Oja, BCM).
    bool usesTraces() const
    {
        return type != LearningRuleType::Sign && !perSynapse();
    }
    // Whether the rule keeps a value per synapse (Eligibility, EProp,
    // Surrogate): Dense layers only.
    bool perSynapse() const
    {
        return type == LearningRuleType::Eligibility || type == LearningRuleType::EProp ||
               type == LearningRuleType::Surrogate;
    }
    // Whether the rule needs the neurons' local derivatives (EProp, Surrogate).
    bool usesDerivatives() const { return type == LearningRuleType::EProp || type == LearningRuleType::Surrogate; }
    // Whether the rule follows a running reward baseline (Trace,
    // Perturbation, Eligibility, EProp).
    bool usesBaseline() const
    {
        return type == LearningRuleType::Trace || type == LearningRuleType::Perturbation ||
               type == LearningRuleType::Eligibility || type == LearningRuleType::EProp;
    }
    // Throws std::invalid_argument when a parameter is out of range.
    void validate() const;

    bool operator==(const LearningRule&) const = default;
};

// "sign", "trace", "feedback_alignment", "perturbation", "oja", "bcm",
// "eligibility", "eprop", "surrogate".
const char* learningRuleName(LearningRuleType type);
// The rule and its non-default options in a few words, e.g. "trace(0.9) +bias".
std::string describeLearningRule(const LearningRule& rule);

} // namespace exr
