// The State layer (state_tap): a neuron layer's thresholds and habituation
// streaks as outputs (doc/model.md, "State as output").
#include <gtest/gtest.h>
#include <algorithm>
#include <sstream>
#include <vector>
#include "../core/layers/dense.hpp"
#include "../core/layers/state_tap.hpp"
#include "../core/network.hpp"
#include "../core/random.hpp"

using namespace exr;

namespace {

LayerSpec linear(size_t size)
{
    LayerSpec spec = LayerSpec::Dense(size, false, false);
    spec.normalize = false;
    return spec;
}

std::vector<float> drive(network& net, size_t tick)
{
    std::vector<float> x(net.inputCount());
    for (size_t i = 0; i < x.size(); ++i)
        x[i] = ((tick * 7 + i * 3) % 5 == 0) ? 0.0f : ((tick + i) % 3 == 0 ? 1.5f : -0.4f);
    return x;
}

// x (4) -> h (6 neurons, E-R + habituation) -> tap
struct Tapped
{
    network net;
    network::LayerId h, tap;
    explicit Tapped(bool threshold = true, bool habituation = true, bool withTap = true)
    {
        reseed(5);
        LayerSpec spec = LayerSpec::Dense(6, true, true);
        spec.habituationRule.steps = 3;
        h = net.addLayer("h", spec);
        net.addInputs(h, 4);
        net.addOutput(h);
        if (withTap) {
            tap = net.addLayer("tap", LayerSpec::State(threshold, habituation));
            net.connect(h, tap);
        }
    }
};

} // namespace

TEST(StateTapTest, OutputsTheSourceStateOfThisTickPerNeuron)
{
    Tapped t;
    const auto& h = t.net.layerAs<dense>(t.h);
    const auto& tap = t.net.getLayer(t.tap);
    ASSERT_EQ(tap.size(), 12u);
    for (size_t tick = 0; tick < 40; ++tick) {
        // Every fourth tick repeats the previous input, so streaks build up.
        t.net.setInputs(drive(t.net, tick < 20 ? tick : 20));
        t.net.step();
        for (size_t i = 0; i < 6; ++i) {
            const neuron& n = h.neurons()[i];
            EXPECT_EQ(tap.output()[2 * i], n.threshold() - n.restingThreshold()) << tick << " " << i;
            EXPECT_EQ(tap.output()[2 * i + 1], std::min(n.habituationStreak() / 3.0f, 1.0f)) << tick << " " << i;
        }
    }
    // The held input (ticks 20..39) has run every streak past the onset.
    for (size_t i = 0; i < 6; ++i)
        EXPECT_EQ(tap.output()[2 * i + 1], 1.0f);
}

TEST(StateTapTest, SingleFieldsAndNeuronsWithoutTheMechanism)
{
    Tapped only_threshold(true, false), only_habituation(false, true);
    EXPECT_EQ(only_threshold.net.getLayer(only_threshold.tap).size(), 6u);
    EXPECT_EQ(only_habituation.net.getLayer(only_habituation.tap).size(), 6u);

    network net;
    const auto h = net.addLayer("h", linear(3));
    const auto tap = net.addLayer("tap", LayerSpec::State());
    net.addInputs(h, 2);
    net.connect(h, tap);
    net.setInputs({1.0f, -1.0f});
    net.step();
    for (float v : net.getLayer(tap).output())
        EXPECT_EQ(v, 0.0f);  // no E-R, no habituation: nothing to show
}

TEST(StateTapTest, AddingATapChangesNothingElse)
{
    Tapped with, without(true, true, false);
    for (size_t tick = 0; tick < 50; ++tick) {
        with.net.setInputs(drive(with.net, tick));
        without.net.setInputs(drive(without.net, tick));
        with.net.step();
        without.net.step();
        ASSERT_EQ(with.net.outputs(), without.net.outputs()) << tick;
        with.net.applyReward(tick % 2 ? 1.0f : -1.0f, 0.05f);
        without.net.applyReward(tick % 2 ? 1.0f : -1.0f, 0.05f);
    }
}

TEST(StateTapTest, ReadersSeeTheStateOfTickTAndFeedbackSeesTMinusOne)
{
    // x -> h -> tap -> r (forward: r(t) reads theta_h(t) - rest); f, a
    // feedback neuron in h, reads tap: f(t) = theta_h0(t-1) - rest.
    network net;
    LayerSpec er = LayerSpec::Dense(1, false, true);
    er.normalize = false;
    const auto h = net.addLayer("h", er);
    const auto tap = net.addLayer("tap", LayerSpec::State(true, false));
    const auto r = net.addLayer("r", linear(1));
    net.addInputs(h, 1);
    net.connect(h, tap);
    net.connect(tap, r);
    net.addFeedback(tap, h, 1);  // h grows to 2 neurons, the tap to 2 values
    auto& H = net.layerAs<dense>(h);
    H.setWeights(0, {1.0f});
    H.setWeights(1, {1.0f, 0.0f});
    net.layerAs<dense>(r).setWeights(0, {1.0f, 0.0f});
    ASSERT_EQ(net.getLayer(tap).size(), 2u);

    float previous = 0.0f;  // the tap has not run before the first tick
    for (float x : {2.0f, 0.0f, 0.0f, 3.0f, 0.0f}) {
        net.setInputs({x});
        net.step();
        const float now = H.neurons()[0].threshold() - H.neurons()[0].restingThreshold();
        EXPECT_EQ(net.layerAs<dense>(r).output()[0], now);
        EXPECT_EQ(H.lastInputs(1)[0], previous);
        previous = now;
    }
}

TEST(StateTapTest, Errors)
{
    EXPECT_THROW(state_tap(false, false), std::invalid_argument);
    network net;
    const auto a = net.addLayer("a", linear(2));
    const auto b = net.addLayer("b", linear(2));
    const auto hist = net.addLayer("hist", LayerSpec::History(2));
    const auto tap = net.addLayer("tap", LayerSpec::State());
    net.connect(a, hist);
    net.connect(a, tap);
    EXPECT_THROW(net.connect(b, tap), std::logic_error);
    const auto tap2 = net.addLayer("tap2", LayerSpec::State());
    EXPECT_THROW(net.connect(hist, tap2), std::logic_error);
}

TEST(StateTapTest, SavedAndLoaded)
{
    Tapped t;
    for (size_t tick = 0; tick < 10; ++tick) {
        t.net.setInputs(drive(t.net, tick));
        t.net.step();
    }
    std::stringstream data;
    t.net.save(data);
    auto copy = network::load(data);
    for (size_t tick = 10; tick < 30; ++tick) {
        t.net.setInputs(drive(t.net, tick));
        copy->setInputs(drive(*copy, tick));
        t.net.step();
        copy->step();
        const auto a = t.net.getLayer(t.tap).output(), b = copy->getLayer(t.tap).output();
        ASSERT_TRUE(std::equal(a.begin(), a.end(), b.begin(), b.end())) << tick;
    }
}
