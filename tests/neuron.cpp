#include <gtest/gtest.h>
#include <sstream>
#include <cmath>
#include <algorithm>
#include "../core/neuron.hpp"
#include "er_scales.hpp"

// =============================================================================
// Habituation tests
// =============================================================================
TEST(NeuronTest, HabituationSuppressesRepeatedSignal)
{
    // neuron(hasHabituation = true, hasER = false)
    neuron n(true, false);

    // Constant-valued input
    auto input_val = std::make_shared<float>(0.5f);
    std::vector<std::shared_ptr<float>> inputs = { input_val };
    n.initializeWeights(inputs);

    // For the first habituation_steps steps the signal should pass through (output != 0)
    float out;
    for (int i = 0; i < habituation_steps ; ++i)
    {
        out = n.step(inputs);
        EXPECT_NE(out,0.0f);
    }

    // On the next step the signal should be suppressed
    float habituated_out = n.step(inputs);
    EXPECT_EQ(habituated_out, 0.0f);
    EXPECT_EQ(*n.getOutput(), 0.0f);

    // Changing the signal by more than habituation_epsilon wakes the neuron up
    *input_val = 0.8f;
    float woken_out = n.step(inputs);
    EXPECT_NE(woken_out, 0.0f);
}

// =============================================================================
// E-R tests (excitation-relaxation and spontaneous firing)
// =============================================================================
TEST(NeuronTest, ExcitationRelaxationAndSpontaneousFiring)
{
    // neuron(hasHabituation = false, hasER = true)
    neuron n(false, true);

    auto input_val = std::make_shared<float>(0.0f); // no external signal
    std::vector<std::shared_ptr<float>> inputs = { input_val };
    n.initializeWeights(inputs);

    // Feed only zeros: the firing threshold should decay (x recovery_factor per
    // tick) until it drops below min_threshold and triggers a spontaneous
    // firing. It starts at baseline_threshold; allow 10 ticks of slack.
    bool fired_spontaneously = false;
    const int max_steps = ticksUntilSpontaneousFiring(baseline_threshold) + 10;
    for (int step = 0; step < max_steps; ++step)
    {
        float out = n.step(inputs);
        if (out != 0.0f)
        {
            fired_spontaneously = true;
            // Spontaneous firing amplitude lies within [-spontaneous_min_amplitude, spontaneous_min_amplitude]
            EXPECT_LE(std::abs(out), spontaneous_min_amplitude);
            break;
        }
    }

    EXPECT_TRUE(fired_spontaneously);
}

// =============================================================================
// Reward-modulated Hebbian learning tests
// =============================================================================
TEST(NeuronTest, RewardModulatedWeightUpdate)
{
    neuron n(false, false); // E-R and habituation off, to test the weight update in isolation

    auto input_val = std::make_shared<float>(1.0f);
    std::vector<std::shared_ptr<float>> inputs = { input_val };

    // Known initial weight (0.5)
    n.setWeights({ 0.5f });

    // One step, so the neuron has a non-zero output (*output = 0.5)
    n.step(inputs);
    EXPECT_GT(std::abs(*n.getOutput()), firing_epsilon);

    // Apply a positive reward
    float reward = 1.0f;
    float learningRate = 0.1f;
    n.updateWeights(inputs, reward, learningRate);

    // New weight: 0.5 + learningRate * gain * reward * sign(input) = 0.5 + 0.1 * g * 1.0
    const float g = default_learning_gain;
    EXPECT_FLOAT_EQ(n.getWeights()[0], 0.5f + 0.1f * g);

    // Apply a punishment (negative reward)
    reward = -2.0f;
    n.updateWeights(inputs, reward, learningRate);

    // New weight: previous + 0.1 * g * -2.0
    EXPECT_FLOAT_EQ(n.getWeights()[0], 0.5f + 0.1f * g - 0.2f * g);
}


// =============================================================================
//  Deserializacja FullState (Hot Snapshot)
// =============================================================================
TEST(NeuronTest, SerializationFullStatePreservesExactBehavior) {
    neuron n_original(true, true);
    auto input_val = std::make_shared<float>(0.5f);
    std::vector<std::shared_ptr<float>> inputs = { input_val };
    n_original.initializeWeights(inputs);

    // Run 50 steps to change the internal state (habituation and E-R)
    for (int i = 0; i < 50; ++i) {
        n_original.step(inputs);
    }

    // Save the state to an in-memory stream
    std::stringstream ss;
    n_original.serialize(ss);

    // Create a blank neuron and restore the full state (FullState)
    neuron n_restored(false, false);
    n_restored.deserialize(ss, DeserializeMode::FullState);

    // Step 51 on the same input MUST give identical results in both neurons
    float out_orig = n_original.step(inputs);
    float out_rest = n_restored.step(inputs);

    EXPECT_FLOAT_EQ(out_orig, out_rest);
    EXPECT_FLOAT_EQ(*n_original.getOutput(), *n_restored.getOutput());
}

// =============================================================================
// Deserializacja WeightsOnly (Cold Start)
// =============================================================================
TEST(NeuronTest, SerializationWeightsOnlyResetsInternalCounters) {
    neuron n_original(true, true);
    auto input_val = std::make_shared<float>(5.0f);
    std::vector<std::shared_ptr<float>> inputs = { input_val };
    n_original.initializeWeights(inputs);

    // Bring the neuron to the edge of habituation (99 steps)
    for (int i = 0; i < 99; ++i) {
        n_original.step(inputs);
    }

    std::stringstream ss;
    n_original.serialize(ss);

    // Deserialize in WeightsOnly mode
    neuron n_restored(false, false);
    n_restored.deserialize(ss, DeserializeMode::WeightsOnly);

    // Weights must be identical
    EXPECT_EQ(n_original.getWeights(), n_restored.getWeights());

    // Internal state was reset, so step 100 in n_restored must NOT trigger habituation
    float out_rest = n_restored.step(inputs);
    EXPECT_NE(out_rest, 0.0f); // The restored neuron starts habituation from zero
}




// =============================================================================
// Per-neuron dynamics (recovery, learning gain)
// =============================================================================
TEST(NeuronTest, DefaultsAndRandomizedDynamics)
{
    neuron plain(false, true);
    EXPECT_EQ(plain.getRecovery(), recovery_factor);
    EXPECT_EQ(plain.getLearningGain(), default_learning_gain);

    plain.randomizeDynamics(Jitter::none(), Jitter::none());  // disabled: defaults, nothing drawn
    EXPECT_EQ(plain.getRecovery(), recovery_factor);
    EXPECT_EQ(plain.getLearningGain(), default_learning_gain);

    neuron::reseed(1);
    std::vector<float> recoveries, gains;
    for (int i = 0; i < 200; ++i) {
        neuron n(false, true);
        n.randomizeDynamics(Jitter::uniform(0.08f), Jitter::uniform(0.2f));
        recoveries.push_back(n.getRecovery());
        gains.push_back(n.getLearningGain());
        EXPECT_GE(n.getRecovery(), recovery_factor - 0.08f);
        EXPECT_LE(n.getRecovery(), recovery_factor + 0.08f);
        EXPECT_GE(n.getLearningGain(), default_learning_gain - 0.2f);
        EXPECT_LE(n.getLearningGain(), default_learning_gain + 0.2f);
    }
    EXPECT_GT(*std::max_element(recoveries.begin(), recoveries.end()) -
              *std::min_element(recoveries.begin(), recoveries.end()), 0.1f);  // actually spread

    neuron::reseed(1);  // reproducible
    neuron again(false, true);
    again.randomizeDynamics(Jitter::uniform(0.08f), Jitter::uniform(0.2f));
    EXPECT_EQ(again.getRecovery(), recoveries[0]);
    EXPECT_EQ(again.getLearningGain(), gains[0]);
}

TEST(NeuronTest, RecoverySetsThresholdDecay)
{
    // Same neuron twice, different recovery: after one firing, the threshold
    // relaxes at the neuron's own rate, so the slow one stays refractory longer.
    auto silentTicksUntilRefire = [](float recovery) {
        neuron n(false, true);
        n.setRecovery(recovery);
        auto x = std::make_shared<float>(5.0f);
        std::vector<std::shared_ptr<float>> in = {x};
        n.setWeights({1.0f});
        n.step(in);            // fires, threshold jumps up
        *x = 0.3f;             // a weak input fires again once the threshold decays below it
        for (int t = 1; t < 1000; ++t)
            if (n.step(in) != 0.0f) return t;
        return -1;
    };
    const int fast = silentTicksUntilRefire(0.8f);
    const int slow = silentTicksUntilRefire(0.97f);
    EXPECT_GT(fast, 0);
    EXPECT_GT(slow, 3 * fast);
}

TEST(NeuronTest, LearningGainScalesUpdates)
{
    auto input = std::make_shared<float>(1.0f);
    std::vector<std::shared_ptr<float>> in = {input};
    neuron n(false, false);
    n.setWeights({0.5f});
    n.setLearningGain(2.0f);
    n.step(in);
    n.updateWeights(in, 1.0f, 0.1f);
    EXPECT_FLOAT_EQ(n.getWeights()[0], 0.7f);  // 0.5 + 0.1 * 2.0 * 1.0
}

TEST(NeuronTest, SerializationKeepsDynamics)
{
    neuron n(true, true);
    n.setRecovery(0.95f);
    n.setLearningGain(1.3f);
    std::stringstream ss;
    n.serialize(ss);
    for (DeserializeMode mode : {DeserializeMode::FullState, DeserializeMode::WeightsOnly}) {
        std::stringstream copy(ss.str());
        neuron r(false, false);
        r.deserialize(copy, mode);
        EXPECT_EQ(r.getRecovery(), 0.95f);
        EXPECT_EQ(r.getLearningGain(), 1.3f);
    }
}

TEST(NeuronTest, JitterDistributionsAreControlled)
{
    auto sample = [](const Jitter& recovery, int count) {
        std::vector<float> values;
        for (int i = 0; i < count; ++i) {
            neuron n(false, true);
            n.randomizeRecovery(recovery);
            values.push_back(n.getRecovery());
        }
        return values;
    };
    auto mean = [](const std::vector<float>& v) { double s = 0; for (float x : v) s += x; return s / v.size(); };
    auto sd = [&](const std::vector<float>& v) {
        const double m = mean(v); double s = 0;
        for (float x : v) s += (x - m) * (x - m);
        return std::sqrt(s / (v.size() - 1));
    };
    neuron::reseed(3);

    // Uniform around the default: mean recovery_factor, sd = halfWidth / sqrt(3)
    const auto u = sample(Jitter::uniform(0.05f), 4000);
    EXPECT_NEAR(mean(u), recovery_factor, 0.003);
    EXPECT_NEAR(sd(u), 0.05 / std::sqrt(3.0), 0.002);

    // Normal with its own centre
    const auto n = sample(Jitter::normal(0.02f).around(0.95f), 4000);
    EXPECT_NEAR(mean(n), 0.95, 0.002);
    EXPECT_NEAR(sd(n), 0.02, 0.002);

    // Limits are enforced by redrawing: nothing outside, nothing piled on the edges
    const float lo = recovery_factor - 0.02f, hi = recovery_factor + 0.02f;
    const auto lim = sample(Jitter::normal(0.05f).within(lo, hi), 4000);
    int at_edge = 0;
    for (float x : lim) {
        EXPECT_GE(x, lo);
        EXPECT_LE(x, hi);
        at_edge += x == lo || x == hi;
    }
    EXPECT_LT(at_edge, 5);

    // The parameter's own valid range always applies: recovery stays below 0.999
    for (float x : sample(Jitter::uniform(0.5f).around(0.99f), 500))
        EXPECT_LE(x, 0.999f);
}

TEST(NeuronTest, RelativeJitterScalesWithTheParameter)
{
    neuron::reseed(4);
    std::vector<float> gains, recoveries;
    for (int i = 0; i < 4000; ++i) {
        neuron n(false, true);
        n.randomizeLearningGain(Jitter::uniformRelative());  // +-50% of the gain
        n.randomizeRecovery(Jitter::uniformRelative());      // +-50% of 1 - recovery
        gains.push_back(n.getLearningGain());
        recoveries.push_back(n.getRecovery());
    }
    const auto [gmin, gmax] = std::minmax_element(gains.begin(), gains.end());
    const auto [rmin, rmax] = std::minmax_element(recoveries.begin(), recoveries.end());

    // Gain: default_learning_gain * [0.5, 1.5] (e.g. 2 -> 1..3), filling the range
    EXPECT_GE(*gmin, 0.5f * default_learning_gain);
    EXPECT_LE(*gmax, 1.5f * default_learning_gain);
    EXPECT_LT(*gmin, 0.52f * default_learning_gain);
    EXPECT_GT(*gmax, 1.48f * default_learning_gain);

    // Recovery: 1 - (1 - recovery_factor) * [0.5, 1.5] (e.g. 0.9 -> 0.85..0.95)
    const float speed = 1.0f - recovery_factor;
    EXPECT_GE(*rmin, 1.0f - 1.5f * speed - 1e-6f);
    EXPECT_LE(*rmax, 1.0f - 0.5f * speed + 1e-6f);
    EXPECT_LT(*rmin, 1.0f - 1.45f * speed);
    EXPECT_GT(*rmax, 1.0f - 0.55f * speed);

    // Relative normal: sd is the fraction of the scale, around a custom centre
    std::vector<float> v;
    for (int i = 0; i < 4000; ++i) {
        neuron n(false, true);
        n.randomizeLearningGain(Jitter::normalRelative(0.1f).around(4.0f));  // sd 0.4
        v.push_back(n.getLearningGain());
    }
    double m = 0, s = 0;
    for (float x : v) m += x;
    m /= v.size();
    for (float x : v) s += (x - m) * (x - m);
    EXPECT_NEAR(m, 4.0, 0.03);
    EXPECT_NEAR(std::sqrt(s / (v.size() - 1)), 0.4, 0.02);
}

TEST(NeuronTest, AlphaJitter)
{
    neuron plain(false, true);
    EXPECT_EQ(plain.getAlpha(), default_alpha);
    plain.randomizeAlpha(Jitter::none());  // disabled: default, nothing drawn
    EXPECT_EQ(plain.getAlpha(), default_alpha);

    neuron::reseed(5);
    std::vector<float> alphas;
    for (int i = 0; i < 4000; ++i) {
        neuron n(false, true);
        n.randomizeAlpha(Jitter::uniformRelative());  // +-50% of alpha (alpha 2 -> 1..3)
        alphas.push_back(n.getAlpha());
        EXPECT_EQ(n.getRecovery(), recovery_factor);  // other parameters untouched
        EXPECT_EQ(n.getLearningGain(), default_learning_gain);
    }
    const auto [amin, amax] = std::minmax_element(alphas.begin(), alphas.end());
    EXPECT_GE(*amin, 0.5f * default_alpha);
    EXPECT_LE(*amax, 1.5f * default_alpha);
    EXPECT_LT(*amin, 0.52f * default_alpha);
    EXPECT_GT(*amax, 1.48f * default_alpha);

    // Alpha stays >= 0 even when the distribution reaches below zero
    for (int i = 0; i < 500; ++i) {
        neuron n(false, true);
        n.randomizeAlpha(Jitter::uniform(3.0f).around(1.0f));
        EXPECT_GE(n.getAlpha(), 0.0f);
    }
}

TEST(NeuronTest, AlphaSetsThresholdGrowth)
{
    // Same firing, larger alpha -> higher threshold afterwards -> longer
    // refractory period before a weaker input fires again.
    auto ticksUntilRefire = [](float alpha) {
        neuron n(false, true);
        n.setAlpha(alpha);
        auto x = std::make_shared<float>(5.0f);
        std::vector<std::shared_ptr<float>> in = {x};
        n.setWeights({1.0f});
        n.step(in);
        *x = 0.3f;
        for (int t = 1; t < 1000; ++t)
            if (n.step(in) != 0.0f) return t;
        return -1;
    };
    EXPECT_GT(ticksUntilRefire(3.0f), ticksUntilRefire(0.5f));
}
