// A single neuron's dynamics: two bio-inspired adaptation mechanisms on top
// of an ordinary weighted sum.
//
//   1. Excitation-Relaxation (E-R): a per-neuron threshold that rises when
//      the neuron fires (fatigue/spike-frequency adaptation) and relaxes
//      back down when it doesn't. If the threshold decays all the way to
//      zero (long stretch of effective silence), the neuron starts firing
//      spontaneously at a small fixed amplitude, which re-excites the
//      threshold through the same mechanism a real firing would.
//
//   2. Habituation: tracks how many consecutive steps the weighted sum has
//      stayed ~unchanged; once that streak is long enough, the input is
//      treated as zero (the neuron "tunes out" a constant, unchanging
//      stimulus) until the signal actually changes again.
//
// The weights are not part of the neuron: layers own them, laid out for fast
// weighted sums (see kernels.hpp), and pass each neuron its sum
// (activate()). step() and learn() run one neuron against caller-owned
// weights, for experiments and tests.
#pragma once
#include <cstdint>
#include <iosfwd>
#include <limits>
#include <optional>
#include <random>
#include <span>
#include <vector>
#include "random.hpp"

namespace exr {

enum class DeserializeMode {
    WeightsOnly,
    FullState
};

// --- Tunable constants shared by every neuron ---------------------------
inline constexpr float habituation_epsilon = 0.0000000001f; // max |sum - previous_input| still counted as "the same signal"
inline constexpr int habituation_steps = 100;               // consecutive "same signal" steps before it's zeroed out
inline constexpr float recovery_factor = 0.5f;              // default per-step multiplicative threshold decay while not firing (0 < b < 1)
inline constexpr float legacy_recovery_factor = 0.9f;       // the default before 2026-10-08 (files older than format 20 keep it)
inline constexpr float min_threshold = 0.0000000001f;       // threshold floor; at/below this, spontaneous firing kicks in
inline constexpr float spontaneous_min_amplitude = 0.1f;    // default +/- amplitude of spontaneous (dormant) firing
inline constexpr float legacy_spontaneous_amplitude = 0.01f; // the amplitude before 2026-09-28 (files older than format 15)
inline constexpr float normalization_epsilon = 1e-6f;       // normalised layers: a weight vector shorter than this keeps its raw sum
inline constexpr float firing_epsilon = 1e-6f;              // outputs at or below this magnitude count as "didn't fire" for learning (hasER == false only)
inline constexpr float baseline_threshold = 0.2f;           // the neuron's resting E-R threshold and learning-eligibility boundary
inline constexpr float max_weight = 10.0f;                  // learning clamps every weight to [-max_weight, max_weight]
inline constexpr float max_output = 10.0f;                  // the weighted sum, and so the output, is clamped to [-max_output, max_output]
inline constexpr float default_alpha = 2.0f;                // default E-R threshold growth rate of the Log rule
inline constexpr float default_learning_gain = 2.0f;        // default per-neuron learning gain: multiplies every weight update

// Layout of a serialized neuron. 1: without recovery / learning gain
// (network formats 1-2); 2: with them; 3: the same neuron record, and layers
// of neurons append their learning rule and its state (network format 9);
// 4: the same, and layers append the per-synapse rules' state and frozen
// neurons and inputs (network format 18). The neuron model's state is not in
// the record: networks save it after everything else (format 21).
inline constexpr std::uint32_t NEURON_FORMAT_VERSION = 4;
// Upper bound for a serialized weight count, so corrupt data fails with an
// error instead of an enormous allocation.
inline constexpr std::uint64_t max_serialized_weights = std::uint64_t{1} << 26;

// Random per-neuron variation of one neuron parameter (E-R recovery, E-R
// alpha or learning gain), drawn when a layer creates a neuron. Disabled by
// default.
//
//   Jitter::uniform(0.05f)                     default +- 0.05, uniform
//   Jitter::normal(0.02f).around(0.95f)        normal, mean 0.95, sd 0.02
//   Jitter::uniform(0.3f).within(0.5f, 2.0f)   draws outside [0.5, 2.0] redrawn
//   Jitter::uniformRelative()                  +-50% of the parameter's scale
//   Jitter::normalRelative(0.2f)               sd = 20% of the scale
//   Jitter::none().around(0.9f)                no variation: every neuron gets 0.9
//
// Relative spreads are a fraction of the parameter's scale: for the learning
// gain and alpha their centre value (gain 2 +-50% = 1..3, alpha 2 +-50% =
// 1..3); for recovery its distance from 1, i.e. the relaxation speed
// (0.9 +-50% = 0.85..0.95).
//
// Draws outside the limits (and outside the parameter's own valid range) are
// redrawn rather than clamped, so no probability piles up at the edges.
struct Jitter
{
    enum class Distribution : std::uint8_t { None = 0, Uniform = 1, Normal = 2 };

    Distribution distribution = Distribution::None;
    float spread = 0.0f;           // Uniform: half-width; Normal: standard deviation
    bool relative = false;         // spread is a fraction of the parameter's scale (see above)
    std::optional<float> mean;     // centre; unset = the parameter's default value
    float min = -std::numeric_limits<float>::infinity();
    float max = std::numeric_limits<float>::infinity();

    bool enabled() const { return distribution != Distribution::None && spread > 0.0f; }

    static Jitter none() { return {}; }
    static Jitter uniform(float halfWidth) { Jitter j; j.distribution = Distribution::Uniform; j.spread = halfWidth; return j; }
    static Jitter normal(float stddev) { Jitter j; j.distribution = Distribution::Normal; j.spread = stddev; return j; }
    static Jitter uniformRelative(float fraction = 0.5f) { Jitter j = uniform(fraction); j.relative = true; return j; }
    static Jitter normalRelative(float fraction = 0.5f) { Jitter j = normal(fraction); j.relative = true; return j; }
    Jitter around(float centre) const { Jitter j = *this; j.mean = centre; return j; }
    Jitter within(float lo, float hi) const { Jitter j = *this; j.min = lo; j.max = hi; return j; }

    bool operator==(const Jitter&) const = default;
};

// How habituation works (for neurons that have it). Two modes:
//   cut (decay == 0, the default and the original behaviour): after `steps`
//     consecutive ticks whose raw sums are "the same", the input is cut to 0
//     until it changes;
//   fade (decay > 0): from the `fadeAfter`-th such tick on, the input is
//     scaled by decay^(ticks habituated), so it fades instead of vanishing.
struct Habituation
{
    std::uint32_t steps = habituation_steps;  // cut mode: streak length (ticks) before the input is cut; >= 1
    float tolerance = 0.0f;  // "the same signal": |sum - previous| <= max(habituation_epsilon,
                             // tolerance * max(|sum|, |previous|)); 0 = exact (up to the epsilon)
    float decay = 0.0f;      // 0 = cut mode; in (0, 1]: fade mode, the factor per habituated tick
    std::uint32_t fadeAfter = 2;  // fade mode: streak length (ticks) before fading starts; >= 1

    bool valid() const
    {
        return steps >= 1 && fadeAfter >= 1 && tolerance >= 0.0f && tolerance < 1.0f && decay >= 0.0f && decay <= 1.0f;
    }
    // The streak length at which suppression starts in this rule's mode.
    std::uint32_t onset() const { return decay > 0.0f ? fadeAfter : steps; }
    bool operator==(const Habituation&) const = default;
};

// Spontaneous E-R firing: a silent neuron fires on its own with a random
// output in [-amplitude, amplitude] when its threshold has relaxed to `below`
// or lower, and, with `rate` > 0, also with that probability on any silent
// tick. The defaults are the original behaviour (only after the threshold
// has decayed to min_threshold, amplitude spontaneous_min_amplitude). A
// spontaneous firing never lowers the threshold.
struct Spontaneous
{
    float below = min_threshold;                 // threshold at or below which a silent neuron fires; >= 0
    float amplitude = spontaneous_min_amplitude; // output drawn uniformly in [-amplitude, amplitude]; [0, max_output]
    float rate = 0.0f;                           // extra per-silent-tick firing probability; [0, 1]

    bool valid() const
    {
        return below >= 0.0f && below <= max_output && amplitude >= 0.0f && amplitude <= max_output &&
               rate >= 0.0f && rate <= 1.0f;
    }
    bool operator==(const Spontaneous&) const = default;
};

// How an E-R threshold grows when the neuron fires with magnitude s > thr.
// The default is Linear (since network format 14; files saved before load
// with Log, the original rule). Whatever the rule, the threshold is at
// least 2 * baseline_threshold after a firing.
//   Linear          thr + amount * (s - thr)    (moves part of the way towards s)
//   Log             thr + alpha * ln(s / thr)   (alpha: the neuron's own, see alpha())
//   Fixed           thr + amount                (the same jump whatever s is)
//   Multiplicative  thr * (1 + amount)          (in proportion to the threshold)
// Log's jump grows without bound as thr falls (after a long silence), the
// others stay bounded by s, by amount, or by the threshold itself.
struct ThresholdGrowth
{
    enum class Rule : std::uint8_t { Log = 0, Linear = 1, Fixed = 2, Multiplicative = 3 };

    Rule rule = Rule::Linear;
    float amount = 0.5f;  // Linear, Fixed, Multiplicative; ignored by Log; finite, >= 0

    bool valid() const
    {
        return static_cast<std::uint8_t>(rule) <= 3 && amount >= 0.0f && amount <= 1e6f;
    }
    bool operator==(const ThresholdGrowth&) const = default;
};

// What turns a neuron's weighted sum (after the clamp and habituation) into
// its output. The default, ER, is the original neuron: E-R with hasER, else
// linear, gated, rectified (ReLU) or binary (see neuron::gate). The others
// are for neurons without E-R; a layer chooses one (LayerSpec::neuronModel).
//
//   Rate units, y = f(gain * s); with smoothing m > 0 a leaky integrator
//   (continuous-time RNN unit): y = m * y + (1 - m) * f(gain * s).
//     Sigmoid     f(x) = 1 / (1 + e^-x)              in (0, 1)
//     Tanh        f(x) = tanh(x)                     in (-1, 1)
//     Softplus    f(x) = ln(1 + e^x), divided by gain (a smooth ReLU)
//   Spiking units, output 1 on a spike, else 0:
//     LIF         leaky integrate-and-fire: v = leak * v + s; a spike when
//                 v > threshold + a, then v = reset and `refractory` ticks
//                 ignore the input. With adaptation > 0 every spike raises
//                 a by `adaptation`, and a decays by adaptationDecay per
//                 tick: an adaptive LIF (ALIF, the e-prop neuron).
//     Izhikevich  Izhikevich (2003), one tick = 1 ms: v' = 0.04v^2 + 5v + 140
//                 - u + I, u' = a(bv - u), I = gain * s; a spike at v >= 30
//                 mV, then v = c, u += d. Presets: regularSpiking (default),
//                 fastSpiking, chattering, intrinsicallyBursting.
//     Resonator   resonate-and-fire (Izhikevich 2001): a damped oscillator
//                 z = leak * e^(2 pi i frequency) * z + i s; a spike when
//                 Im z > threshold, then z = 0 and `refractory` ticks ignore
//                 the input. Input near its own frequency builds up; slow
//                 or fast input cancels out.
// Saved in network files (format 21): never renumber.
struct NeuronModel
{
    enum class Type : std::uint8_t { ER = 0, Sigmoid = 1, Tanh = 2, Softplus = 3, LIF = 4, Izhikevich = 5, Resonator = 6 };

    Type type = Type::ER;
    float gain = 1.0f;             // rate units: slope; Izhikevich: input current per unit of sum; > 0
    float smoothing = 0.0f;        // rate units: leaky integration of the output, [0, 1)
    float leak = 0.9f;             // LIF: membrane decay per tick; Resonator: damping per tick; [0, 1)
    float threshold = 1.0f;        // LIF, Resonator: firing threshold, > 0
    float reset = 0.0f;            // LIF: membrane after a spike, < threshold
    std::uint32_t refractory = 0;  // LIF, Resonator: ticks after a spike that ignore the input
    float adaptation = 0.0f;       // LIF: threshold rise per spike (ALIF), >= 0; 0 = plain LIF
    float adaptationDecay = 0.95f; // LIF: decay of that rise per tick, [0, 1)
    float frequency = 0.1f;        // Resonator: cycles per tick, (0, 0.5]
    float a = 0.02f, b = 0.2f, c = -65.0f, d = 8.0f;  // Izhikevich (regular spiking)

    bool rate() const { return type == Type::Sigmoid || type == Type::Tanh || type == Type::Softplus; }
    bool spiking() const { return type == Type::LIF || type == Type::Izhikevich || type == Type::Resonator; }
    bool valid() const;
    bool operator==(const NeuronModel&) const = default;

    static NeuronModel er() { return {}; }
    static NeuronModel sigmoid(float gain = 1.0f, float smoothing = 0.0f) { return rateUnit(Type::Sigmoid, gain, smoothing); }
    static NeuronModel tanh(float gain = 1.0f, float smoothing = 0.0f) { return rateUnit(Type::Tanh, gain, smoothing); }
    static NeuronModel softplus(float gain = 1.0f, float smoothing = 0.0f) { return rateUnit(Type::Softplus, gain, smoothing); }
    static NeuronModel lif(float leak = 0.9f, float threshold = 1.0f, std::uint32_t refractory = 0, float reset = 0.0f)
    {
        NeuronModel m;
        m.type = Type::LIF;
        m.leak = leak;
        m.threshold = threshold;
        m.refractory = refractory;
        m.reset = reset;
        return m;
    }
    static NeuronModel alif(float leak = 0.9f, float threshold = 1.0f, float adaptation = 0.2f,
                            float adaptationDecay = 0.95f, std::uint32_t refractory = 0)
    {
        NeuronModel m = lif(leak, threshold, refractory);
        m.adaptation = adaptation;
        m.adaptationDecay = adaptationDecay;
        return m;
    }
    static NeuronModel izhikevich(float a = 0.02f, float b = 0.2f, float c = -65.0f, float d = 8.0f, float gain = 10.0f)
    {
        NeuronModel m;
        m.type = Type::Izhikevich;
        m.a = a;
        m.b = b;
        m.c = c;
        m.d = d;
        m.gain = gain;
        return m;
    }
    static NeuronModel regularSpiking(float gain = 10.0f) { return izhikevich(0.02f, 0.2f, -65.0f, 8.0f, gain); }
    static NeuronModel fastSpiking(float gain = 10.0f) { return izhikevich(0.1f, 0.2f, -65.0f, 2.0f, gain); }
    static NeuronModel chattering(float gain = 10.0f) { return izhikevich(0.02f, 0.2f, -50.0f, 2.0f, gain); }
    static NeuronModel intrinsicallyBursting(float gain = 10.0f) { return izhikevich(0.02f, 0.2f, -55.0f, 4.0f, gain); }
    static NeuronModel resonator(float frequency = 0.1f, float damping = 0.9f, float threshold = 1.0f,
                                 std::uint32_t refractory = 0)
    {
        NeuronModel m;
        m.type = Type::Resonator;
        m.frequency = frequency;
        m.leak = damping;
        m.threshold = threshold;
        m.refractory = refractory;
        return m;
    }

private:
    static NeuronModel rateUnit(Type type, float gain, float smoothing)
    {
        NeuronModel m;
        m.type = type;
        m.gain = gain;
        m.smoothing = smoothing;
        return m;
    }
};

// "er", "sigmoid", "tanh", "softplus", "lif", "izhikevich", "resonator".
const char* neuronModelName(NeuronModel::Type type);

// Local derivatives of one activate() call, for gradient-based learning
// rules (e-prop, surrogate gradients; see learning.hpp). s is the weighted
// sum the neuron was given, y its output, thr its E-R threshold before the
// tick and thr' after it. For the other neuron models (NeuronModel) thr is
// the model's state variable instead: a LIF or Izhikevich neuron's membrane
// v, a resonator's Im z, a smoothed rate unit's output (Izhikevich's
// recovery u and the resonator's Re z are ignored, and spikes' resets are
// not differentiated, as in e-prop). The firing step (y jumps when |s| crosses the
// threshold or the gate) has no useful derivative, so it is replaced by a
// triangular pseudo-derivative of half-width `width` times the threshold.
// Habituation and the output clamp scale the sum's derivatives; spontaneous
// firing counts as silence.
struct Derivatives
{
    float dyds = 0.0f;      // dy / ds
    float dydthr = 0.0f;    // dy / dthr (E-R)
    float dthrdthr = 0.0f;  // dthr' / dthr (E-R)
    float dthrds = 0.0f;    // dthr' / ds (E-R)
};

class neuron
{
public:
    // hasHabituation / hasER: independently toggle each mechanism.
    // alpha: threshold-growth rate of the Log growth rule (higher = threshold
    // climbs faster after a strong signal); the default Linear rule ignores it
    // (see ThresholdGrowth).
    explicit neuron(bool hasHabituation = true, bool hasER = true, float alpha = default_alpha);

    // --- Dynamics -------------------------------------------------------
    // One tick: takes this tick's weighted sum, clamps it, applies
    // habituation then E-R, updates the internal state and returns the output.
    float activate(float weightedSum);
    // The same, and its local derivatives (see Derivatives); `width` > 0 is
    // the pseudo-derivative's half-width relative to the threshold.
    float activate(float weightedSum, float width, Derivatives& derivatives);
    float output() const { return output_; }
    void setOutput(float value) { output_ = value; }

    // --- Learning -------------------------------------------------------
    // Reward-modulated Hebbian-style rule: an eligible neuron moves each
    // weight by learningDelta() in the direction of its input's sign (see
    // kernels::signRule). A positive reward strengthens the weights that
    // drove the firing; a negative reward pushes them the other way.
    //
    // Eligible: with E-R, a threshold still above baseline (the neuron fired
    // recently, even if it is refractory right now); without E-R, a non-zero
    // output this tick.
    bool eligible() const;
    // Only meaningful when eligible(): learningRate * gain * reward *
    // eligibility, where eligibility grades how recently / strongly the
    // neuron fired (threshold / baseline - 1 with E-R, 1 without).
    float learningDelta(float reward, float learningRate) const;

    // --- One neuron with its own weights (experiments, tests) -----------
    // activate(dot(inputs, weights)); both spans have the same length.
    float step(std::span<const float> inputs, std::span<const float> weights);
    // The learning rule applied to `weights`, if eligible.
    void learn(std::span<float> weights, std::span<const float> inputs, float reward, float learningRate) const;

    // --- Per-neuron dynamics --------------------------------------------
    // recovery: per-tick E-R threshold decay while not firing, valid range
    //   [0.01, 0.999], default recovery_factor; larger = slower relaxation.
    // learning gain: multiplies weight updates, >= 0, default default_learning_gain.
    // alpha: E-R threshold growth rate of the Log rule, >= 0, default default_alpha.
    // randomize* draws the value from `jitter` using the jitter random
    // stream; a disabled jitter sets its centre (the default if unset) and
    // draws nothing.
    void randomizeRecovery(const Jitter& jitter);
    void randomizeLearningGain(const Jitter& jitter);
    void randomizeAlpha(const Jitter& jitter);
    // All three, in this order.
    void randomizeDynamics(const Jitter& recovery, const Jitter& learning, const Jitter& alpha = {});
    float alpha() const { return alpha_; }
    void setAlpha(float value) { alpha_ = value; }
    float recovery() const { return recovery_; }
    void setRecovery(float value) { recovery_ = value; }
    float learningGain() const { return learning_gain_; }
    void setLearningGain(float value) { learning_gain_ = value; }
    bool hasHabituation() const { return has_habituation_; }
    bool hasER() const { return has_er_; }
    float threshold() const { return threshold_; }
    // --- Read-only state probes (experiments; nothing reads these back) --
    // The Sign rule's eligibility after the last tick: threshold / resting - 1
    // while the threshold is above rest (E-R), 1 while the output is non-zero
    // (without E-R), else 0. learningDelta() is rate * gain * reward * this.
    float eligibility() const
    {
        if (!eligible())
            return 0.0f;
        return has_er_ ? threshold_ / resting_ - 1.0f : 1.0f;
    }
    // Habituation: length of the current "same signal" streak, and the raw
    // (clamped, pre-habituation) sum of the last tick it compares against.
    int habituationStreak() const { return habituation_counter_; }
    float previousInput() const { return previous_input_; }
    // The last tick, for activity monitors (development.hpp): the clamped
    // raw sum before habituation, the E-R threshold the tick started with,
    // and whether the neuron fired on its input, fired spontaneously, or
    // stayed silent. Not saved (a loaded neuron reports a silent tick with
    // sum 0 until it steps).
    enum class Firing : std::uint8_t { Silent, Fired, Spontaneous };
    float lastSum() const { return last_sum_; }
    float lastThreshold() const { return last_threshold_; }
    Firing lastFiring() const { return last_firing_; }
    // Without E-R: a fixed firing threshold. An effective sum with
    // |sum| <= gate gives output 0, a larger one passes unchanged. 0 (the
    // default) keeps the neuron linear. Ignored with E-R, whose threshold
    // adapts. A layer-level setting (LayerSpec::gate): not serialized with
    // the neuron.
    float gate() const { return gate_; }
    void setGate(float value) { gate_ = value; }
    // Without E-R: rectification (ReLU). Only sums above the gate (0 by
    // default) pass; the rest give output 0. Ignored with E-R. A layer-level
    // setting (LayerSpec::rectify), like the gate.
    bool rectified() const { return rectified_; }
    void setRectified(bool value) { rectified_ = value; }
    // Without E-R: binary output. A sum that passes the gate gives its sign
    // (+1 or -1; only +1 when rectified) instead of its value, so a
    // rectified binary neuron is a classic perceptron (0 or 1). Ignored with
    // E-R. A layer-level setting (LayerSpec::binary), like the gate.
    bool binary() const { return binary_; }
    void setBinary(bool value) { binary_ = value; }
    // How habituation suppresses a repeated input (see Habituation). A
    // layer-level setting (LayerSpec::habituationRule), like the gate.
    const Habituation& habituation() const { return habituation_; }
    void setHabituation(const Habituation& value) { habituation_ = value; }
    // How the E-R threshold grows on firing (see ThresholdGrowth). A
    // layer-level setting (LayerSpec::thresholdGrowth), like the gate.
    const ThresholdGrowth& thresholdGrowth() const { return growth_; }
    void setThresholdGrowth(const ThresholdGrowth& value) { growth_ = value; }
    // When and how strongly the neuron fires spontaneously (see Spontaneous).
    // A layer-level setting (LayerSpec::spontaneous), like the gate.
    const Spontaneous& spontaneous() const { return spontaneous_; }
    void setSpontaneous(const Spontaneous& value) { spontaneous_ = value; }
    // E-R resting threshold (default baseline_threshold): where the
    // threshold relaxes towards, the learning-eligibility boundary, and half
    // the floor after a firing. A layer-level setting
    // (LayerSpec::restingThreshold), like the gate; lets a layer match E-R
    // to the scale of its sums (e.g. normalised ones). With
    // `moveThreshold`, a neuron now at rest moves to the new resting value.
    float restingThreshold() const { return resting_; }
    void setRestingThreshold(float value, bool moveThreshold = true)
    {
        if (moveThreshold && threshold_ == resting_)
            threshold_ = value;
        resting_ = value;
    }

    // What turns the sum into the output (see NeuronModel). A layer-level
    // setting (LayerSpec::neuronModel), like the gate; a model other than
    // ER replaces E-R and the gate, rectification and binary output.
    // Setting it puts the model's state at rest; setModelState restores a
    // saved one (see membrane, recoveryVariable; refractory ticks left).
    // The state is not in the serialized record.
    const NeuronModel& model() const { return model_; }
    void setModel(const NeuronModel& value);
    void setModelState(float membrane, float recoveryVariable, std::uint32_t refractory)
    {
        model_v_ = membrane;
        model_u_ = recoveryVariable;
        refractory_ = refractory;
    }
    std::uint32_t refractoryTicks() const { return refractory_; }
    // --- Model state probes (NeuronModel; 0 for ER) ------------------------
    // The state variable: the membrane v (LIF, Izhikevich, in mV for the
    // latter), Im z (Resonator), the smoothed output (rate units).
    float membrane() const { return model_v_; }
    // The second one: LIF's threshold rise a, Izhikevich's recovery u, Re z.
    float recoveryVariable() const { return model_u_; }
    // How close the neuron is to firing: 1 at the threshold, 0 at rest (LIF:
    // v / (threshold + a), Izhikevich: (v - c) / (30 - c), Resonator:
    // Im z / threshold); rate units: 0. State taps read it.
    float charge() const;
    // Feedback alignment's surrogate derivative for the last tick: dy/ds of
    // a rate unit, a triangular pseudo-derivative (1 at the threshold, 0
    // half a threshold away) for spiking models; 0 for ER.
    float slope() const { return last_slope_; }

    // --- Serialization --------------------------------------------------
    // One record: flags, alpha, the weights (count + values, passed in since
    // layers own them), the internal state, the spontaneous-firing generator
    // and the per-neuron dynamics. Binary, native endianness.
    void serialize(std::ostream& os, std::span<const float> weights) const;
    // Reads a record written with NEURON_FORMAT_VERSION `format` and returns
    // its weights. FullState restores the internal state too; WeightsOnly
    // resets it (the spontaneous-firing generator keeps its current state).
    // Format 1 has no recovery / learning gain, which keep their values.
    // Throws std::runtime_error on an implausible weight count or generator state.
    std::vector<float> deserialize(std::istream& is, DeserializeMode mode = DeserializeMode::FullState,
                                   std::uint32_t format = NEURON_FORMAT_VERSION);

private:
    void excite(float effectiveSum); // shared by real and spontaneous excitation: sets output, lifts threshold
    float stepModel(float sum);      // one tick of a model other than ER (sum after the clamp and habituation)
    void restModel();                // the model's state at rest
    // The model's local derivatives for a tick that started at state (v, u, refractory).
    void modelDerivatives(float s, float v, float u, std::uint32_t refractory, float width, Derivatives& d) const;
    float spontaneousOutput();       // random output in [-amplitude, amplitude] (see Spontaneous)

    float threshold_;             // E-R: current firing threshold, starts at resting_
    float resting_ = baseline_threshold;  // E-R: resting threshold (see setRestingThreshold)
    float previous_input_ = 0.0f; // habituation only: last raw (pre-habituation) weighted sum
    int habituation_counter_ = 0; // habituation only: consecutive "same signal" streak length
    bool has_habituation_;
    bool has_er_;
    float output_ = 0.0f;
    float last_sum_ = 0.0f;        // probe: the last tick's clamped raw sum
    float last_threshold_ = 0.0f;  // probe: the threshold the last tick started with
    Firing last_firing_ = Firing::Silent;  // probe
    float alpha_;
    float recovery_ = recovery_factor;
    float learning_gain_ = default_learning_gain;
    float gate_ = 0.0f;
    bool rectified_ = false;
    bool binary_ = false;
    Habituation habituation_;
    ThresholdGrowth growth_;
    Spontaneous spontaneous_;
    NeuronModel model_;
    float model_v_ = 0.0f;           // see membrane()
    float model_u_ = 0.0f;           // see recoveryVariable()
    std::uint32_t refractory_ = 0;   // LIF, Resonator: refractory ticks left
    float last_slope_ = 0.0f;        // see slope()
    std::minstd_rand rng_;        // per neuron, so neurons can step in parallel; seeded from rng::spontaneousSeed()
};

} // namespace exr
