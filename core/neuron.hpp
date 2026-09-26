// A single "neuron" combining two bio-inspired adaptation mechanisms on top
// of an ordinary weighted sum:
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
// See neuron.cpp for the full walkthrough of step()'s logic.
#pragma once
#include <memory>
#include <vector>
#include <random>
#include <span>
#include <cstdint>
#include <limits>
#include <optional>


enum class DeserializeMode {
    WeightsOnly,
    FullState
};


// --- Tunable constants shared by every neuron ---------------------------
const float habituation_epsilon=0.0000000001f; // max |sum - previous_input| still counted as "the same signal"
const int habituation_steps=100;               // consecutive "same signal" steps before it's zeroed out
const float recovery_factor=0.9f;              // per-step multiplicative threshold decay while not firing (0 < b < 1)
const float min_threshold=0.0000000001f;       // threshold floor; at/below this, spontaneous firing kicks in
const float default_threshold=0.1f;            // currently unused - kept in case a reset-to-default path is reintroduced
const float spontaneous_min_amplitude=0.01f; // fixed +/- amplitude for spontaneous (dormant) firing
const float firing_epsilon=1e-6f;              // outputs at or below this magnitude count as "didn't fire" for learning purposes (hasER == false only)
const float baseline_threshold=0.2f;           // the neuron's resting E-R threshold (must match the constructor's initial value)
const float max_weight=10.0f;                  // updateWeights() clamps every weight to [-max_weight, max_weight]
const float max_output=10.0f;                  // step() clamps the weighted sum, and so the output, to [-max_output, max_output]
const float default_alpha=1.2f;                   // default E-R threshold growth rate on firing (neuron constructor)
const float default_learning_gain=2.0f;          // default per-neuron learning gain: multiplies every weight update
// Layout of a serialized neuron. 1: without recovery / learning gain
// (network formats 1-2); 2: with them.
constexpr std::uint32_t NEURON_FORMAT_VERSION = 2;

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

class neuron
{
public:
    // hasHabituation / hasER: independently toggle each mechanism.
    // alpha: threshold-growth rate on firing (higher = threshold climbs faster after a strong signal);
    // default_alpha when not given.
    neuron(bool hasHabituation,bool hasER) ;
    neuron(bool hasHabituation,bool hasER,float alpha);

    // Randomly initializes the weight vector to match `inputs`' size. Call
    // once, when this neuron is first wired to an input source (see
    // dense::joinDense / dense::addNeuronsWithGroup).
    void initializeWeights(const std::vector<std::shared_ptr<float>>& inputs); // It has to be run on the layer connection or initialization - it creates initial weights

    // Appends `additionalCount` new randomly-initialized weights without
    // touching the existing ones. Used when this neuron's input pool grows
    // after it has already been learning (see dense::notifySourceGrew).
    void growWeights(size_t additionalCount); // appends new random weights without touching existing ones

    // Draws one sample of "dormant" spontaneous activity from this neuron's
    // own generator, used only when the
    // threshold has fully decayed (see step()). Fixed amplitude - no longer
    // scaled by signal history, since that history is habituation's concern
    // only, not E-R's.
    float spontaneousOutput(); // random output at a fixed baseline amplitude

    // Resets every random stream shared by all neurons (weight init/growth,
    // the seeds of each new neuron's spontaneous-firing generator, and the
    // jitter drawn by randomizeDynamics). Call before building a network to
    // make it independent of whatever ran earlier in the process.
    static void reseed(std::uint32_t seed);

    // Per-neuron dynamics, defaults recovery_factor and default_learning_gain:
    //   recovery: per-tick E-R threshold decay while not firing, valid range
    //             [0.01, 0.999]; larger = slower relaxation = longer memory.
    //   learning gain: multiplies this neuron's weight updates, >= 0.
    // randomizeRecovery / randomizeLearningGain draw the value from `jitter`
    // using a dedicated random stream. A disabled jitter sets the default and
    // draws nothing, so other random streams are unaffected.
    void randomizeRecovery(const Jitter& jitter);
    void randomizeLearningGain(const Jitter& jitter);
    // alpha (E-R threshold growth on firing): default default_alpha, >= 0.
    void randomizeAlpha(const Jitter& jitter);
    // All three, in this order (a disabled jitter draws nothing).
    void randomizeDynamics(const Jitter& recovery, const Jitter& learning, const Jitter& alpha = {});
    float getAlpha() const { return this->alpha; }
    void setAlpha(float value) { this->alpha = value; }
    float getRecovery() const { return this->recovery; }
    void setRecovery(float value) { this->recovery = value; }
    float getLearningGain() const { return this->learning_gain; }
    void setLearningGain(float value) { this->learning_gain = value; }

    // Runs one forward step: computes the weighted sum over `inputs`,
    // applies habituation then E-R, updates internal state, and returns
    // (and stores in the shared output slot) this step's output.
    float step(std::span<const float> inputs);
    // Convenience overload: copies the pointed-to values, then steps.
    float step(const std::vector<std::shared_ptr<float>>& inputs);

    // Reward-modulated Hebbian-style update: only adjusts weights if this
    // neuron actually fired this step (see the firing_epsilon check inside).
    // A positive reward strengthens the weights that drove the firing;
    // a negative reward (punishment) pushes them the other way.
    void updateWeights(std::span<const float> inputs, float reward, float learningRate);
    // Convenience overload: copies the pointed-to values, then updates.
    void updateWeights(const std::vector<std::shared_ptr<float>>& inputs, float reward, float learningRate);

    // Direct weight access - used e.g. to copy weights between neurons in
    // tests, or for future serialization/inspection tooling.
    void setWeights(const std::vector<float>& newWeights) {
        this->weights = newWeights;
    }

    const std::vector<float>& getWeights() const {
        return this->weights;
    }
    std::vector<float>& getWeights() {
        return this->weights;
    }

    // Returns the neuron's live output slot. Callers (e.g. dense) copy the
    // shared_ptr itself, not just its value, so every holder always sees
    // this neuron's latest output without needing to be re-queried.
    const std::shared_ptr<float>& getOutput() const {
        return this->output;
    }
    std::shared_ptr<float>& getOutput() {
        return this->output;
    }

    // serialize the neuron
    void serialize(std::ostream& os) const;
    // deserialize the neuron. Depending on the mode, perform full deserialization with the internal state,
    // or perform deserialization without the internal state. `format` is the
    // NEURON_FORMAT_VERSION the data was written with; format 1 has no
    // recovery / learning gain, which then keep their current values.
    void deserialize(std::istream& is, DeserializeMode mode = DeserializeMode::FullState,
                     std::uint32_t format = NEURON_FORMAT_VERSION);
    void expandWeights(size_t size);

private:
    void excite(float effectiveSum); // shared by real excitation and spontaneous excitation: sets output, lifts threshold

    float threshold;           // E-R: current firing threshold, starts at 1.0
    float previous_input;      // habituation only: last raw (pre-habituation) weighted sum, for repeat detection
    int habituation_counter;   // habituation only: consecutive "same signal" streak length
    bool hasHabituation;       // toggles the habituation mechanism independently of E-R
    bool hasER;                // toggles the E-R mechanism independently of habituation
    std::shared_ptr<float> output; // shared so downstream consumers (dense layers) always see the live value
    float alpha;                // E-R: threshold-growth rate on firing
    float recovery = recovery_factor; // E-R: per-tick threshold decay while not firing
    float learning_gain = default_learning_gain; // multiplies weight updates
    std::vector<float> weights; // one weight per input, in the same order as whatever input vector is passed to step()
    std::minstd_rand spontaneous_rng; // per-neuron, so step() can run on many neurons in parallel; seeded from the shared stream at construction
};
