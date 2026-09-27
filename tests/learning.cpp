// Learning rules (learning.hpp): each rule's update on a single neuron, the
// options (bias, decay, competition), feedback alignment through
// network::applyError, serialization, and small learning tasks per rule.
#include <gtest/gtest.h>
#include <cmath>
#include <memory>
#include <random>
#include <sstream>
#include <stdexcept>
#include <vector>
#include "../core/kernels.hpp"
#include "../core/layers/conv2d.hpp"
#include "../core/layers/dense.hpp"
#include "../core/layers/locally_connected2d.hpp"
#include "../core/network.hpp"
#include "../core/random.hpp"

using namespace exr;

namespace {

// Inputs -> `size` plain neurons (no habituation, no E-R), one output layer.
struct Plain
{
    network net;
    network::LayerId out;

    Plain(size_t inputs, size_t size, const LearningRule& rule)
    {
        LayerSpec spec = LayerSpec::Dense(size, false, false);
        spec.learningRule = rule;
        out = net.addLayer("out", spec);
        net.addInputs(out, inputs);
        net.addOutput(out);
    }
    dense& layer() { return net.layerAs<dense>(out); }
    void setWeights(size_t neuron, std::vector<float> w) { layer().setWeights(neuron, w); }
};

std::vector<float> savedBytes(const network& net)
{
    std::ostringstream os;
    net.save(os);
    const std::string s = os.str();
    return {s.begin(), s.end()};
}

} // namespace

TEST(LearningRuleTest, DefaultIsSignAndValidates)
{
    EXPECT_EQ(LayerSpec{}.learningRule, LearningRule::sign());
    EXPECT_NO_THROW(LearningRule::traced(0.5f).validate());
    EXPECT_THROW(LearningRule::traced(1.0f).validate(), std::invalid_argument);
    EXPECT_THROW(LearningRule::perturbation(0.0f).validate(), std::invalid_argument);
    EXPECT_THROW(LearningRule::sign().withDecay(-1.0f).validate(), std::invalid_argument);
    EXPECT_THROW(LearningRule::bcm(0.0f).validate(), std::invalid_argument);
    EXPECT_EQ(describeLearningRule(LearningRule::traced(0.5f, 0.1f).withBias()), "trace tr0.5 b0.1 +bias");
}

TEST(LearningRuleTest, ExplicitSignRuleMatchesTheDefaultBitForBit)
{
    auto run = [](bool explicitRule) {
        reseed(7);
        network net;
        LayerSpec spec = LayerSpec::Dense(12);
        if (explicitRule)
            spec.learningRule = LearningRule::sign();
        const auto hidden = net.addLayer("hidden", spec);
        const auto out = net.addLayer("out", LayerSpec::Dense(3));
        net.addInputs(hidden, 5);
        net.connect(hidden, out);
        net.addOutput(out);
        std::mt19937 gen(3);
        std::uniform_real_distribution<float> d(-1.0f, 1.0f);
        for (int t = 0; t < 200; ++t) {
            for (size_t i = 0; i < 5; ++i)
                net.setInput(i, d(gen));
            net.step();
            net.applyReward(d(gen), 0.05f);
        }
        return savedBytes(net);
    };
    EXPECT_EQ(run(false), run(true));
}

TEST(LearningRuleTest, LearnScaledWithUnitKeepEqualsLearn)
{
    std::vector<float> w(11 * 5);
    std::mt19937 gen(1);
    std::uniform_real_distribution<float> d(-1.0f, 1.0f);
    for (float& v : w)
        v = d(gen);
    kernels::weight_matrix a(11, 5, w), b(11, 5, w);
    std::vector<float> signs{1, -1, 0, 1, -1}, delta(a.paddedRows()), keep(a.paddedRows(), 1.0f);
    std::vector<std::uint8_t> active(a.paddedRows());
    for (size_t r = 0; r < 11; ++r) {
        delta[r] = d(gen);
        active[r] = r % 3 != 0;
    }
    a.learn(signs, delta, active, max_weight, 0, a.blocks());
    b.learnScaled(signs, delta, keep, active, max_weight, 0, b.blocks());
    for (size_t r = 0; r < 11; ++r)
        for (size_t c = 0; c < 5; ++c)
            EXPECT_EQ(a.at(r, c), b.at(r, c));
}

TEST(LearningRuleTest, TraceRuleIsGradedHebbian)
{
    Plain p(2, 1, LearningRule::traced(0.0f, 0.0f));
    p.setWeights(0, {0.5f, 0.25f});
    p.net.setInputs({2.0f, -1.0f});
    p.net.step();  // y = 0.5 * 2 - 0.25 = 0.75
    ASSERT_FLOAT_EQ(p.net.outputs()[0], 0.75f);
    p.net.applyReward(1.0f, 0.1f);
    // w += rate * gain * reward * y * x, gain 2
    const auto w = p.layer().weights(0);
    EXPECT_FLOAT_EQ(w[0], 0.5f + 0.1f * 2.0f * 0.75f * 2.0f);
    EXPECT_FLOAT_EQ(w[1], 0.25f + 0.1f * 2.0f * 0.75f * -1.0f);
}

TEST(LearningRuleTest, ZeroInputStillTeachesTheBias)
{
    // The sign rule cannot learn from 0 inputs; a biased trace rule can.
    Plain p(1, 1, LearningRule::traced(0.0f, 0.0f).withBias());
    p.setWeights(0, {0.0f});
    p.layer().setBias(0, 0.1f);
    for (int t = 0; t < 50; ++t) {
        p.net.setInputs({0.0f});
        p.net.step();
        p.net.applyReward(1.0f, 0.05f);
    }
    EXPECT_GT(p.layer().bias(0), 1.0f);
    EXPECT_GT(p.net.outputs()[0], 1.0f);
    EXPECT_EQ(p.layer().weights(0)[0], 0.0f);
}

TEST(LearningRuleTest, BaselineTurnsConstantRewardIntoNoUpdate)
{
    Plain p(1, 1, LearningRule::traced(0.0f, 1.0f));  // baseline follows the reward at once
    p.setWeights(0, {0.5f});
    p.net.setInputs({1.0f});
    p.net.step();
    p.net.applyReward(1.0f, 0.1f);  // first call: baseline 0 -> learns
    const float after_first = p.layer().weights(0)[0];
    EXPECT_GT(after_first, 0.5f);
    p.net.step();
    p.net.applyReward(1.0f, 0.1f);  // baseline 1 now: no surprise, no update
    EXPECT_EQ(p.layer().weights(0)[0], after_first);
}

TEST(LearningRuleTest, DecayShrinksWeightsOfNeuronsThatLearn)
{
    Plain p(2, 1, LearningRule::traced(0.0f, 0.0f).withDecay(1.0f));
    p.setWeights(0, {0.5f, 0.0f});
    p.net.setInputs({1.0f, 0.0f});
    p.net.step();  // y = 0.5
    p.net.applyReward(1.0f, 0.1f);
    // w = w * (1 - rate * decay) + rate * gain * y * x
    EXPECT_FLOAT_EQ(p.layer().weights(0)[0], 0.5f * 0.9f + 0.1f * 2.0f * 0.5f);
}

TEST(LearningRuleTest, OjaConvergesToTheUnitPrincipalComponent)
{
    reseed(11);
    Plain p(2, 1, LearningRule::oja());
    p.setWeights(0, {0.3f, -0.1f});
    std::mt19937 gen(5);
    std::normal_distribution<float> big(0.0f, 1.0f), small(0.0f, 0.1f);
    for (int t = 0; t < 4000; ++t) {
        // Main axis (1, 1) / sqrt(2), little variance across it.
        const float a = big(gen), b = small(gen);
        p.net.setInputs({(a + b) / std::sqrt(2.0f), (a - b) / std::sqrt(2.0f)});
        p.net.step();
        p.net.applyReward(0.0f, 0.005f);  // reward ignored
    }
    const auto w = p.layer().weights(0);
    const float norm = std::sqrt(w[0] * w[0] + w[1] * w[1]);
    EXPECT_NEAR(norm, 1.0f, 0.1f);
    EXPECT_NEAR(std::abs(w[0] + w[1]) / (std::sqrt(2.0f) * norm), 1.0f, 0.05f);
}

TEST(LearningRuleTest, BcmThresholdSlidesAndStopsGrowth)
{
    Plain p(1, 1, LearningRule::bcm(0.1f));
    p.setWeights(0, {0.5f});
    for (int t = 0; t < 2000; ++t) {
        p.net.setInputs({1.0f});
        p.net.step();
        p.net.applyReward(0.0f, 0.01f);
    }
    // y (y - theta) with theta ~ y^2: the fixed point is y = 1.
    EXPECT_NEAR(p.net.outputs()[0], 1.0f, 0.05f);
}

TEST(LearningRuleTest, WinnersLimitWhoLearns)
{
    Plain p(1, 4, LearningRule::oja(1));
    for (size_t i = 0; i < 4; ++i)
        p.setWeights(i, {0.1f * static_cast<float>(i + 1)});
    p.net.setInputs({1.0f});
    p.net.step();
    p.net.applyReward(0.0f, 0.1f);
    for (size_t i = 0; i < 3; ++i)
        EXPECT_EQ(p.layer().weights(i)[0], 0.1f * static_cast<float>(i + 1)) << i;
    EXPECT_NE(p.layer().weights(3)[0], 0.4f);  // the most active neuron
}

TEST(LearningRuleTest, PerturbationLearnsATargetFromAScalarReward)
{
    reseed(3);
    Plain p(2, 1, LearningRule::perturbation(0.2f, 0.0f, 0.2f));
    p.setWeights(0, {0.0f, 0.0f});
    const float target = 1.5f;
    for (int t = 0; t < 3000; ++t) {
        p.net.setInputs({1.0f, 0.5f});
        p.net.step();
        const float err = p.net.outputs()[0] - target;
        p.net.applyReward(-err * err, 0.02f);
    }
    float mean = 0.0f;
    for (int t = 0; t < 100; ++t) {
        p.net.step();
        mean += p.net.outputs()[0] / 100.0f;
    }
    EXPECT_NEAR(mean, target, 0.2f);
}

TEST(LearningRuleTest, OutputLayerLearnsItsOwnErrors)
{
    // applyError on an output layer is the delta rule, one error per neuron.
    Plain p(2, 2, LearningRule::feedbackAlignment().withBias());
    const std::vector<std::vector<float>> xs{{1, 0}, {0, 1}, {1, 1}, {-1, 1}};
    auto target = [](const std::vector<float>& x) { return std::vector<float>{x[0] - x[1], 0.5f * x[1] + 0.3f}; };
    for (int epoch = 0; epoch < 300; ++epoch)
        for (const auto& x : xs) {
            p.net.setInputs(x);
            p.net.step();
            const auto y = p.net.outputs();
            const auto t = target(x);
            p.net.applyError(std::vector<float>{t[0] - y[0], t[1] - y[1]}, 0.05f);
        }
    for (const auto& x : xs) {
        p.net.setInputs(x);
        p.net.step();
        const auto y = p.net.outputs();
        const auto t = target(x);
        EXPECT_NEAR(y[0], t[0], 0.05f);
        EXPECT_NEAR(y[1], t[1], 0.05f);
    }
    EXPECT_THROW(p.net.applyError(std::vector<float>{1.0f}, 0.1f), std::invalid_argument);
}

TEST(LearningRuleTest, FeedbackAlignmentTrainsAHiddenLayer)
{
    // A 2-neuron bottleneck in front of 4 outputs. The target depends on two
    // directions of the 6 inputs: a random bottleneck misses them, and only
    // a hidden layer that learns them can reach a small error.
    auto train = [](bool hiddenLearns) {
        reseed(21);
        network net;
        LayerSpec h = LayerSpec::Dense(2, false, false);
        h.learningRule = LearningRule::feedbackAlignment();
        h.frozen = !hiddenLearns;
        LayerSpec o = LayerSpec::Dense(4, false, false);
        o.learningRule = LearningRule::feedbackAlignment();
        const auto hidden = net.addLayer("hidden", h);
        const auto out = net.addLayer("out", o);
        net.addInputs(hidden, 6);
        net.connect(hidden, out);
        net.addOutput(out);
        auto target = [](const std::vector<float>& x) {
            const float a = x[0] + x[1], b = x[2] - x[3];
            return std::vector<float>{a, b, a - b, 0.5f * a};
        };
        std::mt19937 gen(4);
        std::normal_distribution<float> d(0.0f, 0.5f);
        auto sample = [&] {
            std::vector<float> x(6);
            for (float& v : x)
                v = d(gen);
            return x;
        };
        auto errors = [&](const std::vector<float>& x) {
            net.setInputs(x);
            net.step();
            net.step();  // the input reaches the outputs on the second tick
            const auto y = net.outputs();
            const auto t = target(x);
            return std::vector<float>{t[0] - y[0], t[1] - y[1], t[2] - y[2], t[3] - y[3]};
        };
        for (int t = 0; t < 5000; ++t)
            net.applyError(errors(sample()), 0.01f);
        float loss = 0.0f;
        for (int k = 0; k < 200; ++k)
            for (float e : errors(sample()))
                loss += e * e / 800.0f;
        EXPECT_EQ(net.layerAs<dense>(hidden).feedbackRow(0).size(), hiddenLearns ? 4u : 0u);
        return loss;
    };
    const float frozen = train(false), learned = train(true);
    EXPECT_GT(frozen, 0.3f);
    EXPECT_LT(learned, 0.05f);
}

TEST(LearningRuleTest, HiddenSignLayersIgnoreErrors)
{
    network net;
    const auto hidden = net.addLayer("hidden", LayerSpec::Dense(3, false, false));
    const auto out = net.addLayer("out", LayerSpec::Dense(1, false, false));
    net.addInputs(hidden, 2);
    net.connect(hidden, out);
    net.addOutput(out);
    net.setInputs({1.0f, -1.0f});
    net.step();
    net.step();
    const auto before = net.layerAs<dense>(hidden).weights(0);
    net.applyError(std::vector<float>{1.0f}, 0.1f);
    EXPECT_EQ(net.layerAs<dense>(hidden).weights(0), before);
}

TEST(LearningRuleTest, NonLearningLayersRejectRules)
{
    network net;
    LayerSpec spec = LayerSpec::Pool2D(Window2D::square(2));
    spec.learningRule = LearningRule::oja();
    EXPECT_THROW(net.addLayer("pool", spec), std::invalid_argument);
}

namespace {

// Builds a network whose layers use `rule`, runs it, saves it, loads it with
// FullState and checks both copies continue identically.
void expectContinuesAfterLoad(const LearningRule& rule, bool spatial)
{
    reseed(99);
    network net;
    network::LayerId first, last;
    if (spatial) {
        LayerSpec c = LayerSpec::Conv2D(2, Window2D::square(3), false, true);
        c.learningRule = rule;
        LayerSpec l = LayerSpec::LocallyConnected2D(2, Window2D::square(2), false, true);
        l.learningRule = rule;
        first = net.addLayer("image", LayerSpec::Retina(RetinaSpec{Shape{1, 6, 6}}, false, false));
        const auto conv = net.addLayer("conv", c);
        last = net.addLayer("local", l);
        net.addInputs(first, Shape{1, 6, 6});
        net.connect(first, conv);
        net.connect(conv, last);
    } else {
        LayerSpec h = LayerSpec::Dense(6);
        h.learningRule = rule;
        first = net.addLayer("hidden", h);
        last = net.addLayer("out", h);
        net.addInputs(first, 4);
        net.connect(first, last);
        net.addFeedback(last, first, 2);
    }
    net.addOutput(last);
    std::mt19937 gen(8);
    std::uniform_real_distribution<float> d(-1.0f, 1.0f);
    auto tick = [&](network& n, std::mt19937& g) {
        for (size_t i = 0; i < n.inputCount(); ++i)
            n.setInput(i, d(g));
        n.step();
        std::vector<float> errors(n.outputs().size());
        for (float& e : errors)
            e = d(g);
        n.applyError(errors, 0.02f);
        n.applyReward(d(g), 0.02f);
    };
    for (int t = 0; t < 20; ++t)
        tick(net, gen);

    std::stringstream ss;
    net.save(ss);
    const auto restored = network::load(ss, DeserializeMode::FullState);
    EXPECT_EQ(restored->layerSpec(last).learningRule, rule);
    // Draws from the global learning stream (new feedback rows) must match too.
    std::stringstream again;
    net.save(again);
    std::mt19937 gen_a = gen, gen_b = gen;
    for (int t = 0; t < 20; ++t) {
        tick(net, gen_a);
        tick(*restored, gen_b);
        ASSERT_EQ(net.outputs(), restored->outputs()) << describeLearningRule(rule) << " tick " << t;
    }
}

} // namespace

TEST(LearningRuleTest, EveryRuleContinuesExactlyAfterLoad)
{
    for (bool spatial : {false, true})
        for (const LearningRule& rule :
             {LearningRule::sign().withDecay(0.1f).withBias(), LearningRule::traced(0.5f).withBias(),
              LearningRule::feedbackAlignment(0.3f), LearningRule::perturbation(0.1f, 0.2f),
              LearningRule::oja(1), LearningRule::bcm(0.05f, 1, 0.01f).withBias()}) {
            SCOPED_TRACE(describeLearningRule(rule) + (spatial ? " spatial" : " dense"));
            expectContinuesAfterLoad(rule, spatial);
        }
}

TEST(LearningRuleTest, SetLearningRuleIsSavedAndDescribed)
{
    network net;
    const auto a = net.addLayer("a", LayerSpec::Dense(3));
    net.addInputs(a, 2);
    net.setLearningRule(a, LearningRule::bcm(0.02f, 2));
    EXPECT_EQ(net.layerSpec(a).learningRule, LearningRule::bcm(0.02f, 2));
    std::ostringstream text;
    net.describe(text);
    EXPECT_NE(text.str().find("bcm(0.02) k2"), std::string::npos) << text.str();
    std::stringstream ss;
    net.save(ss);
    EXPECT_EQ(network::load(ss)->layerSpec(a).learningRule, LearningRule::bcm(0.02f, 2));
}

TEST(LearningRuleTest, SpatialLayersLearnWithEveryRule)
{
    for (const LearningRule& rule : {LearningRule::traced().withBias(), LearningRule::oja(), LearningRule::bcm()}) {
        reseed(5);
        network net;
        LayerSpec c = LayerSpec::Conv2D(2, Window2D::square(3), false, false);
        c.learningRule = rule;
        const auto img = net.addLayer("image", LayerSpec::Retina(RetinaSpec{Shape{1, 5, 5}}, false, false));
        const auto conv = net.addLayer("conv", c);
        net.addInputs(img, Shape{1, 5, 5});
        net.connect(img, conv);
        net.addOutput(conv);
        for (size_t i = 0; i < 25; ++i)
            net.setInput(i, i % 2 ? 1.0f : 0.5f);
        const auto before = net.layerAs<conv2d>(conv).kernel(0);
        net.step();
        net.step();
        net.applyReward(1.0f, 0.01f);
        EXPECT_NE(net.layerAs<conv2d>(conv).kernel(0), before) << describeLearningRule(rule);
    }
}
