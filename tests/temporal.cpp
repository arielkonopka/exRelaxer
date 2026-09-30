// Temporal semantics of one network tick (doc/model.md, "One tick"): which
// value a neuron reads at time t, which one it learns from, and when learning
// becomes visible. Each test names the values it checks as belonging to t or
// t-1.
//
// Plain neurons (no habituation, no E-R, raw sums) make every value exact:
// output(t) = sum(t) = w . x(t).
#include <gtest/gtest.h>
#include <cmath>
#include <vector>
#include "../core/layers/conv2d.hpp"
#include "../core/layers/dense.hpp"
#include "../core/network.hpp"
#include "../core/random.hpp"

using namespace exr;

namespace {

LayerSpec linear(size_t size, const LearningRule& rule = {})
{
    LayerSpec spec = LayerSpec::Dense(size, false, false);
    spec.normalize = false;
    spec.learningRule = rule;
    return spec;
}

dense& D(network& net, network::LayerId id) { return net.layerAs<dense>(id); }

// x -> A -> B -> C, all one plain neuron with weight 1 on each input; the
// pulse travels the forward path within one tick.
struct Chain
{
    network net;
    network::LayerId a, b, c;
    explicit Chain(const LearningRule& rule = {})
    {
        a = net.addLayer("a", linear(1, rule));
        b = net.addLayer("b", linear(1, rule));
        c = net.addLayer("c", linear(1, rule));
        net.addInputs(a, 1);
        net.connect(a, b);
        net.connect(b, c);
        net.addOutput(c);
    }
};

} // namespace

// --- Forward pass ------------------------------------------------------------

TEST(TemporalTest, FeedForwardEdgesDeliverTheSameTick)
{
    Chain ch;
    for (auto id : {ch.a, ch.b, ch.c})
        D(ch.net, id).setWeights(0, {1.0f});
    const std::vector<float> xs = {1.0f, 0.0f, 2.0f, -1.0f};
    for (float x : xs) {
        ch.net.setInputs({x});
        ch.net.step();
        // a(t) = x(t), b(t) = a(t), c(t) = b(t): no delay on connect edges.
        EXPECT_EQ(D(ch.net, ch.a).output()[0], x);
        EXPECT_EQ(D(ch.net, ch.b).output()[0], x);
        EXPECT_EQ(D(ch.net, ch.c).output()[0], x);
        // What each neuron summed at t is its source's value at t.
        EXPECT_EQ(D(ch.net, ch.b).lastInputs(0), std::vector<float>{x});
        EXPECT_EQ(D(ch.net, ch.c).lastInputs(0), std::vector<float>{x});
    }
}

TEST(TemporalTest, FeedbackFromALaterLayerDeliversTMinusOne)
{
    // a reads x; b reads a; f (a feedback neuron in a) reads b. Default
    // order a, b: f runs before b, so f(t) = b(t-1).
    network net;
    const auto a = net.addLayer("a", linear(1));
    const auto b = net.addLayer("b", linear(1));
    net.addInputs(a, 1);
    net.connect(a, b);
    net.addFeedback(b, a, 1);
    D(net, a).setWeights(0, {1.0f});
    D(net, a).setWeights(1, {1.0f});
    // b now reads [a0, f]: weight 1 on a0, 0 on f.
    D(net, b).setWeights(0, {1.0f, 0.0f});

    float b_prev = 0.0f;
    for (float x : {1.0f, 0.0f, 3.0f, -2.0f, 0.0f}) {
        net.setInputs({x});
        net.step();
        const float f = D(net, a).output()[1];
        EXPECT_EQ(f, b_prev) << "f(t) must be b(t-1)";
        EXPECT_EQ(D(net, a).lastInputs(1), std::vector<float>{b_prev});
        EXPECT_EQ(D(net, b).output()[0], x) << "b(t) = a(t) = x(t)";
        b_prev = D(net, b).output()[0];
    }
}

TEST(TemporalTest, SelfConnectionReadsItsOwnOutputFromTMinusOne)
{
    // y(t) = x(t) + 0.5 * y(t-1): a leaky integrator made of one neuron.
    network net;
    const auto a = net.addLayer("a", linear(1));
    net.addInputs(a, 1);
    net.connect(a, a);
    D(net, a).setWeights(0, {1.0f, 0.5f});

    float y_prev = 0.0f;
    for (float x : {1.0f, 0.0f, 0.0f, 2.0f, 0.0f}) {
        net.setInputs({x});
        net.step();
        const float y = D(net, a).output()[0];
        EXPECT_FLOAT_EQ(y, x + 0.5f * y_prev);
        const std::vector<float> seen = {x, y_prev};  // x(t), y(t-1)
        EXPECT_EQ(D(net, a).lastInputs(0), seen);
        y_prev = y;
    }
}

TEST(TemporalTest, MultipleFeedbackLayersEachAddOneTickPerLoop)
{
    // x -> a -> b -> c. Feedback c -> a (neuron fa) and c -> b (neuron fb),
    // b -> a (neuron gb). Default order a, b, c; every feedback neuron runs
    // before its source, so each delivers the source's t-1 value, whatever
    // layer it lives in.
    network net;
    const auto a = net.addLayer("a", linear(1));
    const auto b = net.addLayer("b", linear(1));
    const auto c = net.addLayer("c", linear(1));
    net.addInputs(a, 1);
    net.connect(a, b);
    net.connect(b, c);
    net.addFeedback(c, a, 1);  // a[1] reads c
    net.addFeedback(c, b, 1);  // b[1] reads c
    net.addFeedback(b, a, 1);  // a[2] reads b (both of b's neurons)
    // Only the forward path carries the pulse; readers of grown layers get 0
    // weights on the new neurons.
    D(net, a).setWeights(0, {1.0f});
    D(net, a).setWeights(1, {1.0f});
    D(net, a).setWeights(2, {1.0f, 0.0f});
    ASSERT_EQ(D(net, b).inputCount(0), 3u);  // a0, fa, gb
    D(net, b).setWeights(0, {1.0f, 0.0f, 0.0f});
    D(net, b).setWeights(1, {1.0f});
    ASSERT_EQ(D(net, c).inputCount(0), 2u);  // b0, fb
    D(net, c).setWeights(0, {1.0f, 0.0f});

    std::vector<float> c_hist = {0.0f}, b_hist = {0.0f};
    const std::vector<float> xs = {1.0f, 0.0f, 0.0f, 5.0f, 0.0f, 0.0f};
    for (size_t t = 0; t < xs.size(); ++t) {
        net.setInputs({xs[t]});
        net.step();
        EXPECT_EQ(D(net, a).output()[1], c_hist.back()) << "fa(t) = c(t-1), t=" << t;
        EXPECT_EQ(D(net, b).output()[1], c_hist.back()) << "fb(t) = c(t-1), t=" << t;
        EXPECT_EQ(D(net, a).output()[2], b_hist.back()) << "gb(t) = b(t-1), t=" << t;
        EXPECT_EQ(D(net, c).output()[0], xs[t]) << "c(t) = x(t): the forward path is same-tick";
        c_hist.push_back(D(net, c).output()[0]);
        b_hist.push_back(D(net, b).output()[0]);
    }
}

TEST(TemporalTest, TheUpdateOrderDecidesWhichEdgeCarriesTheDelay)
{
    // The same graph as FeedbackFromALaterLayerDeliversTMinusOne, run b
    // first: now b(t) = a(t-1) and f(t) = b(t) (same tick). Timing belongs
    // to the order, not to the edge kind.
    network net;
    const auto a = net.addLayer("a", linear(1));
    const auto b = net.addLayer("b", linear(1));
    net.addInputs(a, 1);
    net.connect(a, b);
    net.addFeedback(b, a, 1);
    net.setUpdateOrder({b, a});
    D(net, a).setWeights(0, {1.0f});
    D(net, a).setWeights(1, {1.0f});
    D(net, b).setWeights(0, {1.0f, 0.0f});

    float a_prev = 0.0f;
    for (float x : {1.0f, 0.0f, 3.0f, 0.0f}) {
        net.setInputs({x});
        net.step();
        EXPECT_EQ(D(net, b).output()[0], a_prev) << "b(t) = a(t-1)";
        EXPECT_EQ(D(net, a).output()[1], D(net, b).output()[0]) << "f(t) = b(t)";
        a_prev = D(net, a).output()[0];
    }
}

// --- Learning ---------------------------------------------------------------

TEST(TemporalTest, OutputAtTUsesTheWeightsBeforeTheUpdate)
{
    // step() at t uses w(t-1); applyReward after it produces w(t), which
    // first affects step t+1. Learning never changes outputs, thresholds or
    // traces.
    reseed(3);
    network net;
    LayerSpec spec = LayerSpec::Dense(1, false, true);  // E-R, so Sign learns
    spec.normalize = false;
    const auto a = net.addLayer("a", spec);
    net.addInputs(a, 1);
    net.addOutput(a);
    D(net, a).setWeights(0, {1.0f});
    net.setInputs({1.0f});
    net.step();
    const float y = net.outputs()[0];
    const float thr = D(net, a).neurons()[0].threshold();
    net.applyReward(1.0f, 0.1f);
    EXPECT_GT(D(net, a).weights(0)[0], 1.0f) << "w(t) != w(t-1)";
    EXPECT_EQ(net.outputs()[0], y) << "learning does not touch outputs";
    EXPECT_EQ(D(net, a).neurons()[0].threshold(), thr) << "nor thresholds";
    net.step();
    EXPECT_EQ(net.outputs()[0], D(net, a).weights(0)[0]) << "t+1 uses w(t)";
}

TEST(TemporalTest, SignRuleLearnsFromTheInputsTheForwardPassSummed)
{
    // The audit's probe (appendix 1, Q1): feedback neuron f reads b at t-1.
    // It fires +1 on b(t-1) = +1 while b(t) turns -1. A reward of +1 must
    // strengthen the weight that drove the firing, whatever b holds by the
    // time learning runs. Before 2026-09-30 the rule re-read b at learning
    // time (t) and the weight fell.
    network net;
    LayerSpec aspec = LayerSpec::Dense(1, false, true);
    aspec.normalize = false;
    const auto a = net.addLayer("a", aspec);
    const auto b = net.addLayer("b", linear(1));
    net.addInputs(a, 1);
    net.addInputs(b, 1);  // b follows its own sensor
    net.connect(a, b);
    net.addFeedback(b, a, 1);
    net.freeze(b);
    D(net, a).setWeights(0, {0.0f});
    D(net, a).setWeights(1, {1.0f});
    D(net, b).setWeights(0, {1.0f, 0.0f, 0.0f});  // reads its sensor, a0, f

    net.setInputs({0.0f, 1.0f});
    net.step();  // b = +1
    net.setInputs({0.0f, -1.0f});
    net.step();  // f(t) = b(t-1) = +1 fires; b(t) = -1
    ASSERT_EQ(D(net, a).output()[1], 1.0f);
    ASSERT_EQ(D(net, b).output()[0], -1.0f);
    ASSERT_EQ(D(net, a).lastInputs(1), std::vector<float>{1.0f});
    net.applyReward(1.0f, 0.1f);
    EXPECT_GT(D(net, a).weights(1)[0], 1.0f);
}

TEST(TemporalTest, SignRuleIgnoresSensorsChangedAfterTheStep)
{
    // The sensors are set for t+1 before the reward for t arrives: learning
    // still uses the signs of x(t), in dense and in spatial layers. The
    // conv2d reads a plain relay layer (4 channels of 1 x 1) that copies the
    // sensors; the relay is re-run by hand to change what it holds.
    for (bool changeSensors : {false, true}) {
        reseed(5);
        network net;
        LayerSpec spec = LayerSpec::Dense(2, false, true);
        spec.normalize = false;
        const auto a = net.addLayer("a", spec);
        LayerSpec relaySpec = linear(4);
        relaySpec.frozen = true;
        const auto relay = net.addLayer("relay", relaySpec);
        LayerSpec cspec = LayerSpec::Conv2D(1, Window2D{1, 1, 1, 1}, false, true);
        cspec.normalize = false;
        const auto c = net.addLayer("c", cspec);
        net.addInputs(a, 2);
        net.addInputs(relay, 4);
        net.connect(relay, c);
        D(net, a).setWeights(0, {1.0f, 1.0f});
        D(net, a).setWeights(1, {1.0f, -1.0f});
        for (size_t i = 0; i < 4; ++i) {
            std::vector<float> w(4, 0.0f);
            w[i] = 1.0f;
            D(net, relay).setWeights(i, w);
        }
        net.layerAs<conv2d>(c).setKernel(0, {1.0f, 1.0f, 1.0f, 1.0f});
        net.setInputs({1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f});
        net.step();
        ASSERT_NE(net.layerAs<conv2d>(c).output()[0], 0.0f);
        if (changeSensors) {
            net.setInputs({-1.0f, -1.0f, -1.0f, -1.0f, -1.0f, -1.0f});
            D(net, relay).forward();  // conv2d's source now holds -1 everywhere
        }
        net.applyReward(1.0f, 0.1f);
        EXPECT_GT(D(net, a).weights(0)[0], 1.0f) << "dense, sensors changed: " << changeSensors;
        EXPECT_GT(net.layerAs<conv2d>(c).kernel(0)[0], 1.0f) << "conv2d, source changed: " << changeSensors;
    }
}

TEST(TemporalTest, SignEligibilityIsTheThresholdAfterTheTick)
{
    // The Sign rule's eligibility is read after step t: the threshold t left
    // behind (grown by a firing at t, or relaxed if silent at t).
    network net;
    LayerSpec spec = LayerSpec::Dense(1, false, true);
    spec.normalize = false;
    const auto a = net.addLayer("a", spec);
    net.addInputs(a, 1);
    D(net, a).setWeights(0, {1.0f});
    net.setInputs({1.0f});
    net.step();  // fires 1.0: threshold 0.2 -> max(0.4, 0.2 + 0.5 * 0.8) = 0.6
    const neuron& n = D(net, a).neurons()[0];
    ASSERT_FLOAT_EQ(n.threshold(), 0.6f);
    EXPECT_FLOAT_EQ(n.eligibility(), 0.6f / 0.2f - 1.0f);
    net.applyReward(1.0f, 0.1f);  // delta = 0.1 * gain 2 * 1 * 2
    EXPECT_FLOAT_EQ(D(net, a).weights(0)[0], 1.0f + 0.1f * 2.0f * 2.0f);
}

TEST(TemporalTest, DelayedRewardWithTracesCreditsTheEventThroughLambdaPowers)
{
    // Trace rule, lambda 0.5, no baseline: x(t0) = 1, then d silent ticks,
    // reward at t0 + d. X(t0+d) = lambda^d, P(t0+d) = lambda^d * y(t0), so
    // dw = rate * gain * reward * |P| * X = rate * gain * lambda^(2d) * y(t0).
    for (int d : {0, 1, 2, 4}) {
        const float lambda = 0.5f;
        network net;
        const auto a = net.addLayer("a", linear(1, LearningRule::traced(lambda, 0.0f)));
        net.addInputs(a, 1);
        D(net, a).setWeights(0, {0.5f});
        net.setInputs({1.0f});
        net.step();  // y(t0) = 0.5
        net.setInputs({0.0f});
        for (int k = 0; k < d; ++k)
            net.step();
        const float lp = std::pow(lambda, static_cast<float>(d));
        EXPECT_FLOAT_EQ(D(net, a).inputTrace(0)[0], lp) << "X(t0+d), d=" << d;
        EXPECT_FLOAT_EQ(D(net, a).outputTrace(0), 0.5f * lp) << "P(t0+d), d=" << d;
        net.applyReward(1.0f, 0.1f);
        EXPECT_FLOAT_EQ(D(net, a).weights(0)[0], 0.5f + 0.1f * 2.0f * 0.5f * lp * lp) << "d=" << d;
    }
}

TEST(TemporalTest, DelayedRewardWithTheSignRuleCreditsOnlyThePresentInputs)
{
    // The Sign rule keeps no input history: a reward at t0 + d moves the
    // weights by the signs of x(t0 + d), whatever fired at t0. Its only
    // memory is the E-R eligibility (threshold above rest).
    network net;
    LayerSpec spec = LayerSpec::Dense(1, false, true);
    spec.normalize = false;
    const auto a = net.addLayer("a", spec);
    net.addInputs(a, 2);
    D(net, a).setWeights(0, {1.0f, 0.01f});
    net.setInputs({1.0f, 0.0f});
    net.step();  // event on input 0 at t0: fires
    net.setInputs({0.0f, 1.0f});
    net.step();  // distractor on input 1 at t0 + 1: too weak to fire
    ASSERT_EQ(D(net, a).output()[0], 0.0f);
    ASSERT_GT(D(net, a).neurons()[0].eligibility(), 0.0f);  // still eligible from t0
    net.applyReward(1.0f, 0.1f);
    EXPECT_EQ(D(net, a).weights(0)[0], 1.0f) << "the event's synapse gets no credit";
    EXPECT_GT(D(net, a).weights(0)[1], 0.01f) << "the distractor's synapse does";
}

TEST(TemporalTest, FeedbackAlignmentOutputLearnsFromTracesOfTimeT)
{
    // applyError after step t: the output layer's step uses X(t) and y(t).
    network net;
    const auto a = net.addLayer("a", linear(1, LearningRule::feedbackAlignment(0.0f)));
    net.addInputs(a, 1);
    net.addOutput(a);
    D(net, a).setWeights(0, {0.5f});
    net.setInputs({2.0f});
    net.step();  // y = 1
    net.setInputs({-7.0f});  // the next tick's input: must not matter
    net.applyError(std::vector<float>{1.0f}, 0.1f);
    EXPECT_FLOAT_EQ(D(net, a).weights(0)[0], 0.5f + 0.1f * 2.0f * 1.0f * 2.0f);
}

TEST(TemporalTest, RecurrentSignLearningUsesItsOwnPreviousOutput)
{
    // A self-connected E-R neuron learns from sign(y(t-1)), the value it
    // summed, not from sign(y(t)).
    network net;
    LayerSpec spec = LayerSpec::Dense(1, false, true);
    spec.normalize = false;
    const auto a = net.addLayer("a", spec);
    net.addInputs(a, 1);
    net.connect(a, a);
    D(net, a).setWeights(0, {1.0f, 1.0f});
    net.setInputs({-1.0f});
    net.step();  // y(t0) = -1
    net.setInputs({3.0f});
    net.step();  // sum = 3 + (-1) = 2 > threshold: y(t1) = +2
    ASSERT_EQ(D(net, a).output()[0], 2.0f);
    const std::vector<float> seen = {3.0f, -1.0f};
    ASSERT_EQ(D(net, a).lastInputs(0), seen);
    const float w_self = D(net, a).weights(0)[1];
    net.applyReward(1.0f, 0.1f);
    EXPECT_LT(D(net, a).weights(0)[1], w_self) << "sign(y(t-1)) = -1: the self weight falls";
}
