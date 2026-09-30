// Eligibility traces on their own (learning.hpp, doc/model.md "Eligibility").
// Every rule but Sign keeps, per neuron and per input, after each forward():
//
//   P(t) = lambda * P(t-1) + y(t)      output trace
//   X(t) = lambda * X(t-1) + x(t)      input trace (per input, pool order)
//
// with P = X = 0 initially and lambda = LearningRule::trace in [0, 1)
// (0: this tick's values). The Sign rule keeps no traces; its eligibility is
// the E-R threshold above rest, tested at the end.
#include <gtest/gtest.h>
#include <cmath>
#include <limits>
#include <sstream>
#include <vector>
#include "../core/layers/dense.hpp"
#include "../core/network.hpp"

using namespace exr;

namespace {

// One plain neuron (output = weighted sum), weight 1 on each of `inputs`.
struct One
{
    network net;
    network::LayerId id;
    One(float lambda, size_t inputs = 1, LearningRuleType type = LearningRuleType::Trace)
    {
        LayerSpec spec = LayerSpec::Dense(1, false, false);
        spec.normalize = false;
        spec.learningRule = LearningRule::traced(lambda, 0.0f);
        spec.learningRule.type = type;
        id = net.addLayer("n", spec);
        net.addInputs(id, inputs);
        layer().setWeights(0, std::vector<float>(inputs, 1.0f));
    }
    dense& layer() { return net.layerAs<dense>(id); }
    void tick(float x)
    {
        net.setInputs({x});
        net.step();
    }
    float X() { return layer().inputTrace(0).at(0); }
    float P() { return layer().outputTrace(0); }
};

} // namespace

TEST(TraceTest, ZeroActivityKeepsTracesAtZero)
{
    One n(0.9f);
    for (int t = 0; t < 50; ++t) {
        n.tick(0.0f);
        ASSERT_EQ(n.X(), 0.0f);
        ASSERT_EQ(n.P(), 0.0f);
    }
}

TEST(TraceTest, IsolatedSpikeDecaysAsLambdaToTheK)
{
    for (float lambda : {0.0f, 0.5f, 0.9f, 0.99f}) {
        One n(lambda);
        n.tick(1.0f);
        float expected = 1.0f;
        for (int k = 0; k < 40; ++k) {
            ASSERT_EQ(n.X(), expected) << "lambda " << lambda << " k " << k;  // exact: same float ops
            ASSERT_EQ(n.P(), expected) << "output = input with weight 1";
            n.tick(0.0f);
            expected = lambda * expected + 0.0f;
        }
    }
}

TEST(TraceTest, ConstantActivityConvergesToAGeometricSum)
{
    // X(n) = c * (1 - lambda^n) / (1 - lambda) after n ticks of input c.
    for (float lambda : {0.0f, 0.5f, 0.9f, 0.99f}) {
        One n(lambda);
        const float c = 0.3f;
        for (int t = 1; t <= 2000; ++t) {
            n.tick(c);
            if (t == 1 || t == 10 || t == 100 || t == 2000) {
                const double closed = c * (1.0 - std::pow(double(lambda), t)) / (1.0 - lambda);
                EXPECT_NEAR(n.X(), closed, 1e-4 * closed) << "lambda " << lambda << " t " << t;
            }
        }
        EXPECT_NEAR(n.X(), c / (1.0f - lambda), 1e-4f * c / (1.0f - lambda));
    }
}

TEST(TraceTest, MultipleSpikesSuperpose)
{
    // Spikes at t = 0, 3, 4: X(t) = sum over spikes s <= t of lambda^(t - s).
    const float lambda = 0.8f;
    One n(lambda);
    const std::vector<float> xs = {1, 0, 0, 1, 1, 0, 0, 0, 0, 0};
    for (size_t t = 0; t < xs.size(); ++t) {
        n.tick(xs[t]);
        double expected = 0.0;
        for (size_t s = 0; s <= t; ++s)
            if (xs[s] != 0.0f)
                expected += std::pow(double(lambda), double(t - s));
        EXPECT_NEAR(n.X(), expected, 1e-6) << "t " << t;
    }
}

TEST(TraceTest, SignedInputsCancel)
{
    // The input trace is signed; the Trace rule uses |P| for the neuron but
    // the signed X per input.
    One n(0.5f);
    n.tick(1.0f);
    n.tick(-2.0f);  // X = 0.5 - 2 = -1.5
    EXPECT_FLOAT_EQ(n.X(), -1.5f);
    EXPECT_FLOAT_EQ(n.P(), -1.5f);
}

TEST(TraceTest, LongSilenceDecaysIntoDenormalsAndCanStickThere)
{
    // Measured behaviour, not a design choice: in float arithmetic
    // lambda * X rounds to X itself once X is a few denormal steps
    // (1.4e-45 each) above 0 and lambda is close to 1. For lambda 0.99 the
    // trace stops at 50 steps (7.0e-44) forever; for 0.5 it reaches 0.
    // Harmless for learning (the update is ~1e-44), but a trace is not
    // guaranteed to reach exactly 0.
    for (float lambda : {0.5f, 0.9f, 0.99f}) {
        One n(lambda);
        n.tick(10.0f);
        float previous = n.X();
        for (int t = 0; t < 20000; ++t) {
            n.tick(0.0f);
            ASSERT_TRUE(std::isfinite(n.X()));
            ASSERT_GE(n.X(), 0.0f);
            ASSERT_LE(n.X(), previous);
            previous = n.X();
        }
        EXPECT_LT(n.X(), 1e-42f) << "lambda " << lambda;
        const float stuck = n.X();
        n.tick(0.0f);
        EXPECT_EQ(n.X(), stuck) << "a fixed point: lambda * X rounds back to X";
        if (lambda == 0.5f)
            EXPECT_EQ(n.X(), 0.0f);
        if (lambda == 0.99f)
            EXPECT_EQ(n.X(), 50 * std::numeric_limits<float>::denorm_min());
    }
}

TEST(TraceTest, SaturatedInputStaysBoundedByTheGeometricLimit)
{
    // The output is clamped to +-max_output, so P <= max_output / (1 - lambda).
    const float lambda = 0.99f;
    One n(lambda);
    for (int t = 0; t < 5000; ++t)
        n.tick(50.0f);  // sum clamped to 10
    EXPECT_NEAR(n.P(), max_output / (1.0f - lambda), 1.0f);
    EXPECT_NEAR(n.X(), 50.0f / (1.0f - lambda), 5.0f) << "inputs are not clamped";
    EXPECT_TRUE(std::isfinite(n.X()));
}

TEST(TraceTest, ClearTracesAndRuleChangesReset)
{
    One n(0.9f);
    n.tick(1.0f);
    n.layer().clearTraces();
    EXPECT_EQ(n.P(), 0.0f);
    EXPECT_TRUE(n.layer().inputTrace(0).empty()) << "restarts at 0 on the next forward";
    n.tick(0.0f);
    EXPECT_EQ(n.X(), 0.0f);

    n.tick(1.0f);
    n.net.setLearningRule(n.id, LearningRule::traced(0.5f, 0.0f));
    EXPECT_EQ(n.P(), 0.0f);
    n.tick(0.0f);
    EXPECT_EQ(n.X(), 0.0f);
}

TEST(TraceTest, FullStateLoadKeepsTracesWeightsOnlyDropsThem)
{
    One n(0.9f);
    n.tick(1.0f);
    n.tick(0.0f);
    std::stringstream buffer;
    n.net.save(buffer);
    for (auto mode : {DeserializeMode::FullState, DeserializeMode::WeightsOnly}) {
        std::stringstream copy(buffer.str());
        auto loaded = network::load(copy, mode);
        auto& l = loaded->layerAs<dense>(0);
        if (mode == DeserializeMode::FullState) {
            EXPECT_FLOAT_EQ(l.outputTrace(0), 0.9f);
            EXPECT_FLOAT_EQ(l.inputTrace(0).at(0), 0.9f);
        } else {
            EXPECT_EQ(l.outputTrace(0), 0.0f);
        }
    }
}

TEST(TraceTest, LearningDoesNotAdvanceTraces)
{
    // Traces move once per forward(); applyReward only reads them.
    One n(0.5f);
    n.tick(1.0f);
    n.net.applyReward(1.0f, 0.01f);
    n.net.applyReward(1.0f, 0.01f);
    EXPECT_EQ(n.X(), 1.0f);
    EXPECT_EQ(n.P(), 1.0f);
}

TEST(TraceTest, EveryTracingRuleSharesTheSameInputTrace)
{
    for (auto type : {LearningRuleType::Trace, LearningRuleType::FeedbackAlignment, LearningRuleType::Oja,
                      LearningRuleType::BCM}) {
        One n(0.5f, 1, type);
        n.tick(2.0f);
        n.tick(0.0f);
        EXPECT_FLOAT_EQ(n.X(), 1.0f) << learningRuleName(type);
    }
    One sign(0.0f, 1, LearningRuleType::Sign);
    sign.tick(1.0f);
    EXPECT_TRUE(sign.layer().inputTrace(0).empty()) << "Sign keeps no input trace";
    EXPECT_EQ(sign.P(), 0.0f);
}

TEST(TraceTest, EROutputTraceSeesOnlyFirings)
{
    // With E-R, y(t) is 0 on silent ticks, so P decays between firings even
    // under a constant input the threshold has caught up with.
    network net;
    LayerSpec spec = LayerSpec::Dense(1, false, true);
    spec.normalize = false;
    spec.learningRule = LearningRule::traced(0.5f, 0.0f);
    const auto id = net.addLayer("n", spec);
    net.addInputs(id, 1);
    auto& l = net.layerAs<dense>(id);
    l.setWeights(0, {1.0f});
    float p = 0.0f;
    for (int t = 0; t < 30; ++t) {
        net.setInputs({1.0f});
        net.step();
        p = 0.5f * p + l.output()[0];
        ASSERT_FLOAT_EQ(l.outputTrace(0), p) << "t " << t;
    }
}

// --- The Sign rule's eligibility: the E-R threshold ------------------------

TEST(TraceTest, SignEligibilityWindowFollowsRecovery)
{
    // After one firing of magnitude 1 (linear growth 0.5, rest 0.2) the
    // threshold is 0.6; it relaxes by `recovery` each silent tick and the
    // neuron stays eligible while threshold > rest: k < ln(3) / -ln(recovery)
    // silent ticks. Eligibility = threshold / rest - 1 = 3 * recovery^k - 1.
    for (float recovery : {0.5f, 0.9f, 0.97f, 0.99f}) {
        network net;
        LayerSpec spec = LayerSpec::Dense(1, false, true);
        spec.normalize = false;
        const auto id = net.addLayer("n", spec);
        net.addInputs(id, 1);
        auto& l = net.layerAs<dense>(id);
        l.setWeights(0, {1.0f});
        l.neurons()[0].setRecovery(recovery);
        net.setInputs({1.0f});
        net.step();
        net.setInputs({0.0f});
        int window = 0;
        for (int k = 1; k < 2000 && l.neurons()[0].eligible(); ++k) {
            net.step();
            if (l.neurons()[0].eligible()) {
                window = k;
                EXPECT_NEAR(l.neurons()[0].eligibility(), 3.0 * std::pow(double(recovery), k) - 1.0, 1e-4);
            }
        }
        const int expected = static_cast<int>(std::ceil(std::log(3.0) / -std::log(double(recovery)))) - 1;
        EXPECT_NEAR(window, expected, 1) << "recovery " << recovery;  // float rounding at the boundary
        EXPECT_EQ(l.neurons()[0].eligibility(), 0.0f);
    }
}
