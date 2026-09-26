#include "./neuron.hpp"
#include <cmath>
#include <algorithm>
#include <istream>
#include <ostream>
#include <iterator>
#include <sstream>
#include <string>

namespace {
// One stream per purpose, shared by every neuron in the process. Kept at file
// scope (not function-local statics) so neuron::reseed() can reset them.
std::mt19937 init_generator(12345);
std::mt19937 grow_generator(67890);
std::mt19937 expand_generator(12345);
std::mt19937 spontaneous_generator(54321);
std::mt19937 jitter_generator(24680);

// Copies the values behind `inputs` into a per-thread buffer, so the hot
// loops read contiguous floats instead of chasing one pointer per input.
std::span<const float> gatherInputs(const std::vector<std::shared_ptr<float>>& inputs)
{
    thread_local std::vector<float> buffer;
    buffer.resize(inputs.size());
    for (size_t i = 0; i < inputs.size(); ++i)
        buffer[i] = *inputs[i];
    return buffer;
}

}

void neuron::reseed(std::uint32_t seed)
{
    std::seed_seq seq{seed};
    std::uint32_t seeds[4];
    seq.generate(std::begin(seeds), std::end(seeds));
    init_generator.seed(seeds[0]);
    grow_generator.seed(seeds[1]);
    expand_generator.seed(seeds[2]);
    spontaneous_generator.seed(seeds[3]);
    // Separate seed sequence: adding this stream must not change the four
    // above (seed_seq output depends on how many values are generated).
    std::seed_seq jitter_seq{seed, 0x6a09e667u};
    std::uint32_t jitter_seed;
    jitter_seq.generate(&jitter_seed, &jitter_seed + 1);
    jitter_generator.seed(jitter_seed);
}

// Draws from `jitter` around `fallbackMean`, redrawing values outside the
// intersection of the jitter's limits and the parameter's valid range.
// `scaleOf(centre)` gives the parameter's scale for relative spreads.
template <typename Scale>
static float drawJitter(const Jitter& jitter, float fallbackMean, float validMin, float validMax, Scale scaleOf)
{
    const float centre = jitter.mean.value_or(fallbackMean);
    const float spread = jitter.relative ? jitter.spread * scaleOf(centre) : jitter.spread;
    const float lo = std::max(jitter.min, validMin);
    const float hi = std::min(jitter.max, validMax);
    float value = centre;
    for (int attempt = 0; attempt < 1000; ++attempt) {
        if (jitter.distribution == Jitter::Distribution::Uniform) {
            std::uniform_real_distribution<float> d(centre - spread, centre + spread);
            value = d(jitter_generator);
        } else {
            std::normal_distribution<float> d(centre, spread);
            value = d(jitter_generator);
        }
        if (value >= lo && value <= hi)
            return value;
    }
    return std::clamp(value, lo, hi);  // limits (almost) unreachable: fall back to the nearest edge
}

void neuron::randomizeRecovery(const Jitter& jitter)
{
    // Relative spreads scale with the distance from 1 (the relaxation speed),
    // so they stay symmetric and valid.
    this->recovery = jitter.enabled()
        ? drawJitter(jitter, recovery_factor, 0.01f, 0.999f, [](float c) { return 1.0f - c; })
        : recovery_factor;
}

void neuron::randomizeLearningGain(const Jitter& jitter)
{
    this->learning_gain = jitter.enabled()
        ? drawJitter(jitter, default_learning_gain, 0.0f, std::numeric_limits<float>::infinity(),
                     [](float c) { return std::abs(c); })
        : default_learning_gain;
}

void neuron::randomizeAlpha(const Jitter& jitter)
{
    this->alpha = jitter.enabled()
        ? drawJitter(jitter, default_alpha, 0.0f, std::numeric_limits<float>::infinity(),
                     [](float c) { return std::abs(c); })
        : default_alpha;
}

void neuron::randomizeDynamics(const Jitter& recovery, const Jitter& learning, const Jitter& alpha)
{
    randomizeRecovery(recovery);
    randomizeLearningGain(learning);
    randomizeAlpha(alpha);
}

neuron::neuron(bool hasHabituation,bool hasER,float alpha)
{
    this->threshold=baseline_threshold;
    this->previous_input=0.0;
    this->habituation_counter=0;
    this->output=std::make_shared<float>(0.0f);
    this->alpha=alpha;
    this->hasER=hasER;
    this->hasHabituation=hasHabituation;
    this->spontaneous_rng.seed(spontaneous_generator());
}
neuron::neuron(bool hasHabituation,bool hasER): neuron(hasHabituation,hasER,default_alpha) {}

void neuron::initializeWeights(const std::vector<std::shared_ptr<float>>& inputs)
{
    // Fixed seed (12345): weight initialization is reproducible run-to-run,
    // which matters for debugging/comparing behavior across code changes.
    std::uniform_real_distribution<float> distribution(-1.0f, 1.0f);
    this->weights.resize(inputs.size());
    for (auto& weight : this->weights)
        weight = distribution(init_generator);
}

void neuron::growWeights(size_t additionalCount)
{
    // Separate RNG stream from initializeWeights() - appending shouldn't
    // reproduce the same sequence a fresh initialize would have used.
    std::uniform_real_distribution<float> distribution(-1.0f, 1.0f);
    const size_t old_size = this->weights.size();
    this->weights.resize(old_size + additionalCount);
    for (size_t i = old_size; i < this->weights.size(); ++i)
        this->weights[i] = distribution(grow_generator);
}

float neuron::step(const std::vector<std::shared_ptr<float>>& inputs)
{
    return this->step(gatherInputs(inputs));
}

float neuron::step(std::span<const float> inputs)
{
    // Ordinary weighted sum over this neuron's current inputs. Summed in
    // index order: reassociating (e.g. several accumulators) would be faster
    // but would change results in the last bits.
    const float* x = inputs.data();
    const float* w = this->weights.data();
    const size_t n = inputs.size();
    float sum = 0.0f;
    for (size_t i = 0; i < n; ++i)
        sum += x[i] * w[i];
    // Bounded before habituation and E-R see it, so recurrent loops can't
    // run away and the threshold (grown from this value) stays finite.
    sum = std::clamp(sum, -max_output, max_output);

    if (this->hasHabituation)
    {
        // Branchless habituation update (kept this way deliberately - see
        // note below on why this matters more once neurons are processed
        // in a batch/vectorized layer rather than one at a time):
        //
        //   similar   == is this step's raw sum ~equal to last step's?
        //   counter   == similar ? (counter + 1) : 0          [streak length]
        //   habituated == counter has reached habituation_steps
        //   sum       == habituated ? 0 : sum                 [suppress the input]
        //
        // previous_input is intentionally scoped to ONLY this block - it
        // exists purely for habituation's own repeat-detection, and (per
        // design decision) is not read anywhere outside hasHabituation,
        // including spontaneousOutput() below.
        const bool similar=(std::abs(sum - this->previous_input) <= habituation_epsilon);
        habituation_counter=static_cast<int>(similar) * (habituation_counter + 1);
        this->previous_input=sum;
        const bool habituated =(habituation_counter >= habituation_steps);
        sum *= static_cast<float>(!habituated); // multiply by 0 or 1 instead of branching
    }
    *this->output = sum;
    if (this->hasER)
    {
        // Firing decision and threshold growth are both magnitude-based
        // (std::abs), not sign-based - inputs/outputs can be negative, and
        // the threshold itself is always positive, so comparing/dividing by
        // raw signed values would misclassify negative firings (and could
        // even feed a negative ratio into std::log, producing NaN).
        if (std::abs(sum) > this->threshold)
        {
            this->excite(sum);
        }
        else
        {
            *this->output = 0.0f;
            this->threshold *= this->recovery;
            if (this->threshold <= min_threshold)
            {
                // Effective input has been ~0 for long enough to fully decay
                // the threshold - whether because there's genuinely nothing
                // coming in, or because habituation is suppressing a repeating
                // signal. The neuron doesn't need to know which; it fires
                // spontaneously either way, at spontaneousOutput()'s fixed
                // baseline amplitude, and that spontaneous burst re-excites
                // the threshold through the exact same excite() mechanism a
                // real firing would.
                float spontaneous = this->spontaneousOutput();
                this->excite(spontaneous);
            }
        }
    }

    return *this->output;
}

void neuron::excite(float effectiveSum)
{
    // Shared by both real firing (sum > threshold) and spontaneous firing:
    // sets the output to the (signed) effective sum, and grows the
    // (always-positive) threshold proportionally to the firing's magnitude.
    *this->output = effectiveSum;
    float ratio = std::abs(effectiveSum) / this->threshold;
    this->threshold=std::max(baseline_threshold*2,this->threshold *1.0f + this->alpha * std::log(ratio));
}

float neuron::spontaneousOutput()
{
    // Fixed amplitude, independent of any input history - see the header
    // comment on this method for why it's intentionally decoupled from
    // previous_input/habituation state.
    std::uniform_real_distribution<float> distribution(-spontaneous_min_amplitude, spontaneous_min_amplitude);
    return distribution(this->spontaneous_rng);
}


void neuron::updateWeights(const std::vector<std::shared_ptr<float>>& inputs, float reward, float learningRate)
{
    this->updateWeights(gatherInputs(inputs), reward, learningRate);
}

void neuron::updateWeights(std::span<const float> inputs, float reward, float learningRate) {
    // "eligibility" is a multiplier on the whole update: 0 effectively means
    // "skip this neuron entirely", any positive value grades HOW MUCH credit
    // this neuron gets (recently/strongly fired neurons get more than a
    // neuron that barely crossed into eligibility).
    float eligibility;

    if (this->hasER) {
        // A neuron whose threshold has decayed back down to (or below) its
        // resting baseline hasn't fired recently enough to still be "in
        // play" - it's genuinely at rest, not just quiet-this-instant.
        // Anything still elevated ABOVE baseline is eligible even if this
        // exact step's output is 0.0f, because that zero can simply be E-R's
        // own refractory relaxation following a recent real firing - the
        // neuron was still part of the decision that led here.
        if (this->threshold <= baseline_threshold)
            return; // not currently eligible at all - leave weights untouched
        // Graded by how far above baseline the threshold still sits, as a
        // ratio (0 right at the eligibility boundary, growing with more
        // recent/stronger firings). Deliberately NOT log-scaled: threshold
        // itself already decays exponentially every non-firing step
        // (threshold *= recovery_factor), so this ratio inherits that same
        // exponential shape "for free" - a fast initial drop right after
        // firing that flattens out as it approaches the eligibility
        // boundary, using zero additional per-neuron state. A log() here
        // would flatten that curve into a constant-per-step linear decay
        // instead, which loses the "recent firings dominate" recency bias.
        eligibility = (this->threshold / baseline_threshold) - 1.0f;
    } else {
        // No E-R on this neuron, so there's no threshold/refractory state to
        // read anything from - threshold just sits frozen at baseline
        // forever. Fall back to the plain question "did this neuron actually
        // output something this step", with no additional scaling.
        if (std::abs(*this->output) <= firing_epsilon)
            return;
        eligibility = 1.0f;
    }

    // Reward-modulated Hebbian-style update: a positive reward strengthens
    // the weights that contributed to this step's input pattern; a negative
    // reward (punishment) pushes them the other way. Each weight is clamped
    // to [-max_weight, max_weight], so repeated reward can't grow it without
    // bound. There is no weight decay yet.
    //
    // Bounded by min(weights, inputs) rather than weights.size() alone, as a
    // defensive guard: if this neuron's weight vector and its caller-
    // supplied input vector ever fall out of sync in size (e.g. from a dense
    // layer growth event that hasn't fully propagated), this avoids reading
    // past the end of `inputs`.
    const size_t n = std::min(this->weights.size(), inputs.size());
    // Hoisted out of the loop. Bit-identical to multiplying per weight in
    // the original order: the only per-weight factor is a sign of -1, 0 or
    // +1, and multiplying by it is exact.
    // With the default gain of exactly 1 this is bit-identical to
    // learningRate * reward * eligibility.
    const float delta = learningRate * this->learning_gain * reward * eligibility;
    float* w = this->weights.data();
    for (size_t i = 0; i < n; ++i) {
        float input_val = inputs[i];
        // Sign only, not magnitude: by the time this update fires, the
        // neuron may have been eligible for several steps (threshold still
        // relaxing back toward baseline), so *inputs[i] right now isn't a
        // reliable record of what the input actually was at the moment of
        // the firing that made this neuron eligible in the first place -
        // its current magnitude could be stale or have drifted. Direction
        // is the more trustworthy piece of information to credit/blame.
        float input_sign = static_cast<float>((input_val > 0.0f) - (input_val < 0.0f));
        // min/max instead of std::clamp: same result, but lets the compiler
        // vectorize the loop.
        w[i] = std::min(std::max(w[i] + input_sign * delta, -max_weight), max_weight);
    }
}


void neuron::serialize(std::ostream& os) const {
    // 1. Flags and hyperparameters
    os.write(reinterpret_cast<const char*>(&hasHabituation), sizeof(hasHabituation));
    os.write(reinterpret_cast<const char*>(&hasER), sizeof(hasER));
    os.write(reinterpret_cast<const char*>(&alpha), sizeof(alpha));

    // 2. Weights
    size_t weight_count = weights.size();
    os.write(reinterpret_cast<const char*>(&weight_count), sizeof(weight_count));
    if (weight_count > 0) {
        os.write(reinterpret_cast<const char*>(weights.data()), weight_count * sizeof(float));
    }

    // 3. Internal state (needed for a hot snapshot)
    os.write(reinterpret_cast<const char*>(&threshold), sizeof(threshold));
    os.write(reinterpret_cast<const char*>(&previous_input), sizeof(previous_input));
    os.write(reinterpret_cast<const char*>(&habituation_counter), sizeof(habituation_counter));

    float current_output = output ? *output : 0.0f;
    os.write(reinterpret_cast<const char*>(&current_output), sizeof(current_output));

    // 4. Spontaneous-firing generator. minstd_rand exposes its state only as
    // text; that state is a single value below 2^31, stored as 4 bytes.
    std::ostringstream rng_text;
    rng_text << spontaneous_rng;
    const std::uint32_t rng_state = static_cast<std::uint32_t>(std::stoul(rng_text.str()));
    os.write(reinterpret_cast<const char*>(&rng_state), sizeof(rng_state));

    // 5. Per-neuron dynamics
    os.write(reinterpret_cast<const char*>(&recovery), sizeof(recovery));
    os.write(reinterpret_cast<const char*>(&learning_gain), sizeof(learning_gain));
}

void neuron::deserialize(std::istream& is, DeserializeMode mode, std::uint32_t format) {
    // 1. Read flags and hyperparameters
    is.read(reinterpret_cast<char*>(&hasHabituation), sizeof(hasHabituation));
    is.read(reinterpret_cast<char*>(&hasER), sizeof(hasER));
    is.read(reinterpret_cast<char*>(&alpha), sizeof(alpha));

    // 2. Read weights
    size_t weight_count = 0;
    is.read(reinterpret_cast<char*>(&weight_count), sizeof(weight_count));
    weights.resize(weight_count);
    if (weight_count > 0) {
        is.read(reinterpret_cast<char*>(weights.data()), weight_count * sizeof(float));
    }

    // 3. Read internal state
    float saved_threshold, saved_prev_input, saved_output;
    int saved_hab_counter;

    is.read(reinterpret_cast<char*>(&saved_threshold), sizeof(saved_threshold));
    is.read(reinterpret_cast<char*>(&saved_prev_input), sizeof(saved_prev_input));
    is.read(reinterpret_cast<char*>(&saved_hab_counter), sizeof(saved_hab_counter));
    is.read(reinterpret_cast<char*>(&saved_output), sizeof(saved_output));

    std::uint32_t saved_rng_state = 0;
    is.read(reinterpret_cast<char*>(&saved_rng_state), sizeof(saved_rng_state));

    // Per-neuron dynamics are parameters, like alpha: restored in both modes.
    if (format >= 2) {
        is.read(reinterpret_cast<char*>(&recovery), sizeof(recovery));
        is.read(reinterpret_cast<char*>(&learning_gain), sizeof(learning_gain));
    }

    if (mode == DeserializeMode::FullState) {
        // FullState: restore exactly the state from before saving
        this->threshold = saved_threshold;
        this->previous_input = saved_prev_input;
        this->habituation_counter = saved_hab_counter;
        if (!this->output) this->output = std::make_shared<float>(0.0f);
        *this->output = saved_output;
        std::istringstream(std::to_string(saved_rng_state)) >> this->spontaneous_rng;
    } else {
        // WeightsOnly: reset internal state to initial values (the
        // spontaneous-firing generator keeps its fresh construction seed)
        this->threshold = baseline_threshold;
        this->previous_input = 0.0f;
        this->habituation_counter = 0;
        if (!this->output) this->output = std::make_shared<float>(0.0f);
        *this->output = 0.0f;
    }
}


void neuron::expandWeights(size_t size)
{
    std::uniform_real_distribution<float> distribution(-1.0f, 1.0f);
    auto last=this->weights.size();
    this->weights.resize(last+size);
    for(size_t i=0;i<size;i++)
    {
        this->weights[i+last]= distribution(expand_generator);
    }
}
