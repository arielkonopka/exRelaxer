// The critic (actor-critic TD error) and the curiosity model (intrinsic
// reward), reinforcement.hpp, and how a network drives them.
#include <gtest/gtest.h>
#include <cmath>
#include <sstream>
#include <vector>
#include "../core/layers/dense.hpp"
#include "../core/network.hpp"

using namespace exr;

TEST(CriticTest, LearnsTheDiscountedValueOfAState)
{
    // One state, reward 1 every step: V = 1 / (1 - gamma) = 2.
    network net;
    const auto h = net.addLayer("h", LayerSpec::Dense(1, false, false));
    net.addInputs(h, 1, "s");
    net.setInputs("s", std::vector<float>{1.0f});
    net.setCritic({{}, {"s"}, 0.5f, 0.0f, 0.2f});
    EXPECT_EQ(net.getCritic().features(), 1u);
    EXPECT_EQ(net.temporalDifference(1.0f), 0.0f);  // no previous state yet
    for (int t = 0; t < 2000; ++t)
        net.temporalDifference(1.0f);
    EXPECT_NEAR(net.criticValue(), 2.0f, 0.02f);
    EXPECT_NEAR(net.getCritic().lastError(), 0.0f, 0.02f);
}

TEST(CriticTest, CreditsTheStateThatLeadsToTheReward)
{
    // Three states in a cycle 0 -> 1 -> 2 -> 0, reward on entering 2: with
    // TD(lambda), V(1) (just before the reward) ends highest, V(0) next.
    network net;
    const auto h = net.addLayer("h", LayerSpec::Dense(1, false, false));
    net.addInputs(h, 3, "s");
    net.setCritic({{}, {"s"}, 0.9f, 0.7f, 0.1f});
    const auto enter = [&](int state) {
        std::vector<float> s(3, 0.0f);
        s[static_cast<size_t>(state)] = 1.0f;
        net.setInputs("s", s);
    };
    for (int t = 0; t < 3000; ++t) {
        const int state = t % 3;
        enter(state);
        net.temporalDifference(state == 2 ? 1.0f : 0.0f);
    }
    const auto value = [&](int state) {
        enter(state);
        return net.criticValue();
    };
    EXPECT_GT(value(1), value(0));
    EXPECT_GT(value(0), value(2));
}

TEST(CriticTest, TDErrorDrivesTheLayersAndFollowsGrowth)
{
    reseed(5);
    network net;
    const auto h = net.addLayer("h", LayerSpec::Dense(4, false, true));
    const auto out = net.addLayer("out", LayerSpec::Dense(2, false, false));
    net.addInputs(h, 2, "x");
    net.connect(h, out);
    net.addOutput(out);
    net.setCritic({{h}, {"x"}, 0.9f, 0.8f, 0.05f});
    EXPECT_EQ(net.getCritic().features(), 6u);
    net.setInputs("x", std::vector<float>{1.0f, 0.5f});
    net.step();
    net.applyRewardTD(0.0f, 0.1f);  // first step: delta 0
    net.step();
    const std::vector<float> before = net.layerAs<dense>(out).weights(0);
    const float delta = net.applyRewardTD(1.0f, 0.1f);
    EXPECT_NE(delta, 0.0f);
    EXPECT_EQ(delta, net.getCritic().lastError());
    EXPECT_NE(net.layerAs<dense>(out).weights(0), before);

    // The critic's features follow the layer: growth adds zero weights at
    // the end of h's block, pruning removes the right ones.
    const std::vector<float> w(net.getCritic().weights().begin(), net.getCritic().weights().end());
    net.growLayer(h, 2);
    std::vector<float> grown(net.getCritic().weights().begin(), net.getCritic().weights().end());
    ASSERT_EQ(grown.size(), 8u);
    EXPECT_EQ(grown, (std::vector<float>{w[0], w[1], w[2], w[3], 0.0f, 0.0f, w[4], w[5]}));
    net.pruneNeurons(h, {0, 5});
    std::vector<float> pruned(net.getCritic().weights().begin(), net.getCritic().weights().end());
    EXPECT_EQ(pruned, (std::vector<float>{w[1], w[2], w[3], 0.0f, w[4], w[5]}));
    net.step();
    net.applyRewardTD(1.0f, 0.1f, true);  // terminal: resets

    // Saved with the network.
    std::stringstream data;
    net.save(data);
    auto loaded = network::load(data);
    ASSERT_TRUE(loaded->hasCritic());
    EXPECT_EQ(loaded->getCritic().spec(), net.getCritic().spec());
    EXPECT_FLOAT_EQ(loaded->criticValue(), net.criticValue());

    EXPECT_THROW(net.setCritic({{99}, {}, 0.9f, 0.8f, 0.05f}), std::invalid_argument);
    EXPECT_THROW(net.setCritic({{h}, {}, 1.5f, 0.8f, 0.05f}), std::invalid_argument);
    net.removeCritic();
    EXPECT_FALSE(net.hasCritic());
    EXPECT_THROW(net.criticValue(), std::logic_error);
}

TEST(CuriosityTest, PredictableInputsStopBeingInteresting)
{
    // The input alternates between two patterns: a linear forward model
    // learns to predict the next one, so the intrinsic reward falls; a new
    // pattern is surprising again.
    network net;
    const auto h = net.addLayer("h", LayerSpec::Dense(1, false, false));
    net.addInputs(h, 2, "x");
    CuriositySpec spec;
    spec.predictInputs = {"x"};
    spec.fromInputs = {"x"};
    spec.rate = 0.5f;
    net.setCuriosity(spec);
    const std::vector<float> a = {1.0f, 0.0f}, b = {0.0f, 1.0f};
    net.setInputs("x", a);
    EXPECT_EQ(net.curiosityReward(), 0.0f);  // nothing to predict from yet
    float first = 0.0f, last = 0.0f;
    for (int t = 1; t < 200; ++t) {
        net.setInputs("x", t % 2 ? b : a);
        const float r = net.curiosityReward();
        if (t == 1)
            first = r;
        last = r;
    }
    EXPECT_GT(first, 0.1f);
    EXPECT_LT(last, 1e-3f);
    net.setInputs("x", std::vector<float>{1.0f, 1.0f});  // never seen after b
    EXPECT_GT(net.curiosityReward(), 0.1f);

    std::stringstream data;
    net.save(data);
    auto loaded = network::load(data);
    ASSERT_TRUE(loaded->hasCuriosity());
    EXPECT_EQ(loaded->getCuriosity().predict(), net.getCuriosity().predict());
}
