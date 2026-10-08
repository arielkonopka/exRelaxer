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
inline constexpr float recovery_factor = 0.9f;              // default per-step multiplicative threshold decay while not firing (0 < b < 1)
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
// neurons and inputs (network format 18).
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

// Local derivatives of one activate() call, for gradient-based learning
// rules (e-prop, surrogate gradients; see learning.hpp). s is the weighted
// sum the neuron was given, y its output, thr its E-R threshold before the
// tick and thr' after it. The firing step (y jumps when |s| crosses the
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
    // stream; a disabled jitter sets the default and draws nothing.
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
    std::minstd_rand rng_;        // per neuron, so neurons can step in parallel; seeded from rng::spontaneousSeed()
};

} // namespace exr
