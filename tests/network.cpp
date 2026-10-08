#include <gtest/gtest.h>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <memory>
#include <vector>
#include <random>
#include <sstream>
#include <stdexcept>
#include "../core/network.hpp"
#include "../core/layers/dense.hpp"
#include "../core/layers/conv2d.hpp"
#include <numeric>
#include "pattern_benchmark.hpp"
#include "er_scales.hpp"

namespace {

// Undoes std::fixed / std::setprecision on std::cout, so the format of later
// output (other tests' results) does not depend on which tests ran before.
void restoreCoutFormat()
{
    std::cout.unsetf(std::ios::floatfield);
    std::cout.precision(6);
}

} // namespace

// =============================================================================
// layer_factory
// =============================================================================
TEST(LayerFactoryTest, CreatesDenseFromSpec)
{
    const auto created = layer_factory::instance().create({LayerType::Dense, 7, false, true});
    ASSERT_NE(created, nullptr);
    EXPECT_EQ(created->type(), LayerType::Dense);
    EXPECT_EQ(created->size(), 7u);
    EXPECT_NE(dynamic_cast<dense*>(created.get()), nullptr);
}

TEST(LayerFactoryTest, UnregisteredTypeThrows)
{
    layer_factory factory;
    const auto unknown = static_cast<LayerType>(200);  // no type uses this value
    EXPECT_FALSE(factory.isRegistered(unknown));
    EXPECT_THROW(factory.create({unknown, 4}), std::invalid_argument);
}

TEST(LayerFactoryTest, RegisteredTypeIsUsedByNetwork)
{
    // Stand-in for a future layer type: the network only ever sees `layer`.
    layer_factory factory;
    int created = 0;
    factory.registerType(LayerType::Conv2D, [&](const LayerSpec& spec) -> std::unique_ptr<layer> {
        ++created;
        return std::make_unique<dense>(spec.size * 2, spec.hasHabituation, spec.hasER);
    });

    network net(factory);
    const auto id = net.addLayer("conv", {LayerType::Conv2D, 3});
    EXPECT_EQ(created, 1);
    EXPECT_EQ(net.getLayer(id).size(), 6u);
    EXPECT_EQ(net.layerSpec(id).type, LayerType::Conv2D);
}

// =============================================================================
// Building
// =============================================================================
TEST(NetworkTest, BuildsGraphAndRecordsEdges)
{
    network net;
    const auto in = net.addLayer("in", {LayerType::Dense, 4});
    const auto hid = net.addLayer("hid", {LayerType::Dense, 6});
    const auto out = net.addLayer("out", {LayerType::Dense, 2});
    EXPECT_EQ(net.addInputs(in, 3), 0u);
    net.connect(in, hid);
    net.connect(hid, out);
    net.addFeedback(out, hid, 5);
    net.addOutput(out);

    EXPECT_EQ(net.layerCount(), 3u);
    EXPECT_EQ(net.inputCount(), 3u);
    EXPECT_EQ(net.findLayer("hid"), hid);
    EXPECT_EQ(net.layerName(out), "out");
    EXPECT_EQ(net.getLayer(hid).size(), 11u);  // 6 + 5 feedback neurons

    ASSERT_EQ(net.edges().size(), 3u);
    EXPECT_EQ(net.edges()[2].kind, network::EdgeKind::Feedback);
    EXPECT_EQ(net.edges()[2].from, out);
    EXPECT_EQ(net.edges()[2].to, hid);
    EXPECT_EQ(net.edges()[2].width, 5u);

    // out reads hid, so hid's growth reached out's weights: 11 each.
    const auto& outLayer = dynamic_cast<const dense&>(net.getLayer(out));
    for (size_t n = 0; n < outLayer.size(); ++n)
        EXPECT_EQ(outLayer.inputCount(n), 11u);

    EXPECT_EQ(net.outputs().size(), 2u);
}

TEST(NetworkTest, RejectsInvalidBuildOperations)
{
    network net;
    const auto a = net.addLayer("a", {LayerType::Dense, 2});
    const auto b = net.addLayer("b", {LayerType::Dense, 2});

    EXPECT_THROW(net.addLayer("a", {LayerType::Dense, 2}), std::invalid_argument);  // duplicate name
    EXPECT_THROW(net.addLayer("", {LayerType::Dense, 2}), std::invalid_argument);
    EXPECT_THROW(net.connect(a, 42), std::out_of_range);
    EXPECT_THROW(net.findLayer("missing"), std::out_of_range);
    EXPECT_THROW(net.addInputs(a, 0), std::invalid_argument);

    net.connect(a, b);
    EXPECT_THROW(net.connect(a, b), std::logic_error);  // each pair once

    net.addInputs(a, 2);
    EXPECT_THROW(net.setInputs({1.0f}), std::invalid_argument);  // needs 2 values
    EXPECT_THROW(net.setInput(2, 1.0f), std::out_of_range);
}

// =============================================================================
// Update order
// =============================================================================
TEST(NetworkTest, DefaultOrderIsTopologicalWithCreationOrderTieBreak)
{
    network net;
    const auto out = net.addLayer("out", {LayerType::Dense, 2});  // created first, runs last
    const auto b = net.addLayer("b", {LayerType::Dense, 2});
    const auto a = net.addLayer("a", {LayerType::Dense, 2});
    const auto in = net.addLayer("in", {LayerType::Dense, 2});
    net.connect(in, a);
    net.connect(in, b);
    net.connect(a, out);
    net.connect(b, out);
    net.addFeedback(out, in, 1);  // feedback does not constrain the order

    // a and b are both ready after in; b was created before a.
    const std::vector<network::LayerId> expected = {in, b, a, out};
    EXPECT_EQ(net.updateOrder(), expected);
}

TEST(NetworkTest, ForwardCycleThrowsUnlessOrderIsSet)
{
    network net;
    const auto a = net.addLayer("a", {LayerType::Dense, 2});
    const auto b = net.addLayer("b", {LayerType::Dense, 2});
    net.connect(a, b);
    net.connect(b, a);
    EXPECT_THROW(net.step(), std::logic_error);

    net.setUpdateOrder({b, a});
    EXPECT_NO_THROW(net.step());

    EXPECT_THROW(net.setUpdateOrder({a}), std::invalid_argument);     // missing b
    EXPECT_THROW(net.setUpdateOrder({a, a}), std::invalid_argument);  // duplicate

    net.addLayer("c", {LayerType::Dense, 1});
    EXPECT_THROW(net.step(), std::logic_error);  // custom order no longer covers every layer
}

TEST(NetworkTest, SelfConnectionIsIgnoredByOrdering)
{
    network net;
    const auto a = net.addLayer("a", {LayerType::Dense, 3});
    net.addInputs(a, 1);
    net.connect(a, a);
    const std::vector<network::LayerId> expected = {a};
    EXPECT_EQ(net.updateOrder(), expected);
    EXPECT_NO_THROW(net.step());
}

TEST(NetworkTest, CustomOrderBuildsDelayLine)
{
    // Same one-step memory as the sequence tests in dense.cpp: running delay
    // before taps makes taps[1] hold the previous input.
    network net;
    const auto taps = net.addLayer("taps", {LayerType::Dense, 1, false, false});
    const auto delay = net.addLayer("delay", {LayerType::Dense, 1, false, false});
    net.addInputs(taps, 1);
    net.connect(taps, delay);
    net.addFeedback(delay, taps, 1);
    net.addOutput(taps);
    net.setUpdateOrder({delay, taps});

    auto& t = dynamic_cast<dense&>(net.getLayer(taps));
    t.setWeights(0, {1.0f});
    t.setWeights(1, {1.0f});
    dynamic_cast<dense&>(net.getLayer(delay)).setWeights(0, {1.0f, 0.0f});

    for (float x : {0.0f, 0.0f, 1.0f, 2.0f}) {
        net.setInputs({x});
        net.step();
    }
    const std::vector<float> expected = {2.0f, 1.0f};  // x(t), x(t-1)
    EXPECT_EQ(net.outputs(), expected);
}

// =============================================================================
// Equivalence with manual wiring
// =============================================================================
TEST(NetworkTest, MatchesManuallyWiredLayersExactly)
{
    constexpr int STEPS = 300;
    constexpr float LEARNING_RATE = 0.005f;

    auto stimulus = [](int t) { return (t % 7 < 3) ? WEAK_STIMULUS : -WEAK_STIMULUS; };
    auto reward = [](int t) { return (t % 5 == 0) ? 1.0f : -0.2f; };

    // Manual: the Pavlovian topology from dense.cpp.
    reseed(7);
    dense A(5), B(5), C(5), D(5);
    B.join(A);
    C.join(B);
    D.join(C);
    A.addFeedback(C, 20);
    std::vector<float> sensor(1, 0.0f);
    A.attachInputs(sensor);

    std::vector<std::vector<float>> manual;
    for (int t = 0; t < STEPS; ++t) {
        sensor[0] = stimulus(t);
        A.forward(); B.forward(); C.forward(); D.forward();
        std::vector<float> out;
        for (float v : D.output()) out.push_back(v);
        manual.push_back(out);
        for (dense& l : {std::ref(A), std::ref(B), std::ref(C), std::ref(D)}) l.applyReward(reward(t), LEARNING_RATE);
    }

    // Guard against a vacuous comparison (E-R can silence outputs entirely).
    int nonzero = 0;
    for (const auto& out : manual)
        for (float v : out) nonzero += v != 0.0f;
    // Measured: 155 of 1500 values with the original E-R constants; how many
    // depends on the E-R tuning, so only require a clearly non-empty output.
    ASSERT_GE(nonzero, 15);

    // Same construction sequence through the network.
    reseed(7);
    network net;
    const auto a = net.addLayer("A", {LayerType::Dense, 5});
    const auto b = net.addLayer("B", {LayerType::Dense, 5});
    const auto c = net.addLayer("C", {LayerType::Dense, 5});
    const auto d = net.addLayer("D", {LayerType::Dense, 5});
    net.connect(a, b);
    net.connect(b, c);
    net.connect(c, d);
    net.addFeedback(a, c, 20);
    net.addInputs(a, 1);
    net.addOutput(d);

    for (int t = 0; t < STEPS; ++t) {
        net.setInputs({stimulus(t)});
        net.step();
        ASSERT_EQ(net.outputs(), manual[t]) << "diverged at step " << t;
        net.applyReward(reward(t), LEARNING_RATE);
    }
}

// =============================================================================
// Frozen layers
// =============================================================================
namespace {

std::vector<std::vector<float>> weightsOf(const network& net, network::LayerId id)
{
    std::vector<std::vector<float>> all;
    const auto& layer = dynamic_cast<const dense&>(net.getLayer(id));
    for (size_t i = 0; i < layer.size(); ++i)
        all.push_back(layer.weights(i));
    return all;
}

// E-R off so every neuron that outputs something is eligible to learn.
void trainBriefly(network& net)
{
    for (int t = 0; t < 20; ++t) {
        net.setInputs({(t % 2) ? 0.7f : -0.4f});
        net.step();
        net.applyReward((t % 3) ? 1.0f : -1.0f, 0.05f);
    }
}

} // namespace

TEST(NetworkFreezeTest, FrozenLayerStillRunsButDoesNotLearn)
{
    reseed(2);
    network net;
    const auto in = net.addLayer("in", {LayerType::Dense, 4, false, false, /*frozen*/ true});
    const auto out = net.addLayer("out", {LayerType::Dense, 3, false, false});
    net.addInputs(in, 1);
    net.connect(in, out);
    net.addOutput(out);

    EXPECT_TRUE(net.isFrozen(in));
    EXPECT_FALSE(net.isFrozen(out));

    const auto in_before = weightsOf(net, in);
    const auto out_before = weightsOf(net, out);
    trainBriefly(net);

    EXPECT_EQ(weightsOf(net, in), in_before);   // frozen: unchanged
    EXPECT_NE(weightsOf(net, out), out_before); // plastic: learned
    bool in_active = false;                     // and the frozen layer still computed outputs
    for (float v : net.getLayer(in).output()) in_active = in_active || v != 0.0f;
    EXPECT_TRUE(in_active);
}

TEST(NetworkFreezeTest, FreezeAndUnfreezeAtRuntime)
{
    reseed(3);
    network net;
    const auto in = net.addLayer("in", {LayerType::Dense, 4, false, false});
    net.addInputs(in, 1);

    net.freeze(in);
    EXPECT_TRUE(net.isFrozen(in));
    EXPECT_TRUE(net.layerSpec(in).frozen);
    const auto frozen_weights = weightsOf(net, in);
    trainBriefly(net);
    EXPECT_EQ(weightsOf(net, in), frozen_weights);

    net.unfreeze(in);
    EXPECT_FALSE(net.isFrozen(in));
    trainBriefly(net);
    EXPECT_NE(weightsOf(net, in), frozen_weights);

    EXPECT_THROW(net.freeze(42), std::out_of_range);
}


namespace {

// The Pavlovian-style topology, with habituation and E-R on, plus a custom
// update order option to check that it is restored too.
std::unique_ptr<network> buildSampleNetwork(bool customOrder)
{
    auto net = std::make_unique<network>();
    const auto a = net->addLayer("A", {LayerType::Dense, 5});
    const auto b = net->addLayer("B", {LayerType::Dense, 5});
    const auto c = net->addLayer("C", {LayerType::Dense, 5});
    const auto d = net->addLayer("D", {LayerType::Dense, 5});
    net->connect(a, b);
    net->connect(b, c);
    net->connect(c, d);
    net->addFeedback(a, c, 20);
    net->addInputs(a, 1);
    net->addOutput(d);
    net->addOutput(a);  // A shows spontaneous firing directly
    if (customOrder)
        net->setUpdateOrder({d, c, b, a});
    return net;
}

void runSteps(network& net, int from, int to, bool learn)
{
    for (int t = from; t < to; ++t) {
        net.setInputs({(t % 7 < 3) ? WEAK_STIMULUS : -WEAK_STIMULUS});
        net.step();
        if (learn)
            net.applyReward((t % 5 == 0) ? 1.0f : -0.2f, 0.005f);
    }
}

} // namespace

TEST(NetworkFreezeTest, JitterIsControlledFromTheNetwork)
{
    reseed(12);
    network net;
    const auto a = net.addLayer("a", {LayerType::Dense, 20, false, true});
    const auto b = net.addLayer("b", {LayerType::Dense, 3});
    net.connect(b, a);
    // A fresh view each time: growth reallocates the neuron storage.
    const auto& layerA = dynamic_cast<const dense&>(net.getLayer(a));
    auto neurons = [&] { return layerA.neurons(); };

    // No jitter by default
    for (const neuron& n : neurons()) {
        EXPECT_EQ(n.recovery(), recovery_factor);
        EXPECT_EQ(n.learningGain(), default_learning_gain);
    }

    // Recovery only: learning gain untouched
    net.setRecoveryJitter(a, Jitter::uniform(0.05f));
    EXPECT_EQ(net.layerSpec(a).recoveryJitter, Jitter::uniform(0.05f));
    std::vector<float> recoveries;
    for (const neuron& n : neurons()) {
        recoveries.push_back(n.recovery());
        EXPECT_EQ(n.learningGain(), default_learning_gain);
    }
    const auto [lowest, highest] = std::ranges::minmax(recoveries);
    EXPECT_GT(highest - lowest, 0.03f);

    // Learning gain only: recovery values kept
    net.setLearningJitter(a, Jitter::normal(0.1f));
    for (size_t i = 0; i < neurons().size(); ++i) {
        EXPECT_EQ(neurons()[i].recovery(), recoveries[i]);
        EXPECT_NE(neurons()[i].learningGain(), default_learning_gain);
    }

    // Growth uses the current settings
    net.addFeedback(b, a, 5);
    ASSERT_EQ(neurons().size(), 25u);
    for (size_t i = 20; i < 25; ++i) {
        EXPECT_NE(neurons()[i].recovery(), recovery_factor);
        EXPECT_NE(neurons()[i].learningGain(), default_learning_gain);
    }

    // Alpha: off so far; enabling it changes only alpha
    for (const neuron& n : neurons())
        EXPECT_EQ(n.alpha(), default_alpha);
    std::vector<float> gains;
    for (const neuron& n : neurons()) gains.push_back(n.learningGain());
    net.setAlphaJitter(a, Jitter::uniformRelative());
    for (size_t i = 0; i < neurons().size(); ++i) {
        EXPECT_NE(neurons()[i].alpha(), default_alpha);
        EXPECT_EQ(neurons()[i].learningGain(), gains[i]);
    }
    EXPECT_EQ(net.layerSpec(a).alphaJitter, Jitter::uniformRelative());

    // Disabling resets to the defaults
    net.setRecoveryJitter(a, Jitter::none());
    net.setAlphaJitter(a, Jitter::none());
    for (const neuron& n : neurons()) {
        EXPECT_EQ(n.recovery(), recovery_factor);
        EXPECT_EQ(n.alpha(), default_alpha);
    }
    EXPECT_THROW(net.setLearningJitter(42, Jitter::none()), std::out_of_range);
}

TEST(NetworkSerializationTest, RestoresTopology)
{
    reseed(3);
    auto original = buildSampleNetwork(true);
    std::stringstream ss;
    original->save(ss);
    const auto restored = network::load(ss);

    ASSERT_EQ(restored->layerCount(), original->layerCount());
    for (network::LayerId id = 0; id < original->layerCount(); ++id) {
        EXPECT_EQ(restored->layerName(id), original->layerName(id));
        EXPECT_EQ(restored->layerSpec(id).size, original->layerSpec(id).size);
        EXPECT_EQ(restored->layerSpec(id).hasER, original->layerSpec(id).hasER);
        EXPECT_EQ(restored->getLayer(id).size(), original->getLayer(id).size());
    }
    ASSERT_EQ(restored->edges().size(), original->edges().size());
    for (size_t i = 0; i < original->edges().size(); ++i) {
        EXPECT_EQ(restored->edges()[i].from, original->edges()[i].from);
        EXPECT_EQ(restored->edges()[i].to, original->edges()[i].to);
        EXPECT_EQ(restored->edges()[i].kind, original->edges()[i].kind);
        EXPECT_EQ(restored->edges()[i].width, original->edges()[i].width);
    }
    EXPECT_EQ(restored->outputLayers(), original->outputLayers());
    EXPECT_EQ(restored->inputCount(), original->inputCount());
    EXPECT_EQ(restored->updateOrder(), original->updateOrder());
}

TEST(NetworkSerializationTest, FullStateContinuesExactly)
{
    reseed(11);
    auto original = buildSampleNetwork(false);
    runSteps(*original, 0, 200, true);

    std::stringstream ss;
    original->save(ss);
    reseed(999);  // loading must not depend on the global RNG state
    const auto restored = network::load(ss, DeserializeMode::FullState);

    // Keep learning, then go silent long enough for E-R thresholds to decay
    // and neurons to fire spontaneously: that exercises each neuron's saved
    // random generator. The silence is computed from recovery_factor, from
    // the highest threshold a neuron can practically reach, plus slack.
    const int silence = ticksUntilSpontaneousFiring(MAX_PRACTICAL_THRESHOLD) + 50;
    int spontaneous = 0;
    for (int t = 200; t < 300 + silence; ++t) {
        const bool silent = t >= 300;
        const float x = silent ? 0.0f : ((t % 7 < 3) ? WEAK_STIMULUS : -WEAK_STIMULUS);
        original->setInputs({x});
        restored->setInputs({x});
        original->step();
        restored->step();
        const auto out = original->outputs();
        ASSERT_EQ(restored->outputs(), out) << "diverged at step " << t;
        if (silent)
            for (float v : out) spontaneous += v != 0.0f;
        if (!silent) {
            const float r = (t % 5 == 0) ? 1.0f : -0.2f;
            original->applyReward(r, 0.005f);
            restored->applyReward(r, 0.005f);
        }
    }
    EXPECT_GT(spontaneous, 0) << "no spontaneous firing happened; the generator state was not exercised";
}

TEST(NetworkSerializationTest, WeightsOnlyKeepsWeightsAndResetsState)
{
    reseed(5);
    auto original = buildSampleNetwork(false);
    runSteps(*original, 0, 150, true);

    std::stringstream ss;
    original->save(ss);
    const auto restored = network::load(ss, DeserializeMode::WeightsOnly);

    for (network::LayerId id = 0; id < original->layerCount(); ++id) {
        const auto& a = dynamic_cast<const dense&>(original->getLayer(id));
        const auto& b = dynamic_cast<const dense&>(restored->getLayer(id));
        ASSERT_EQ(a.size(), b.size());
        for (size_t i = 0; i < a.size(); ++i)
            EXPECT_EQ(a.weights(i), b.weights(i));
    }
    for (float v : restored->outputs())
        EXPECT_EQ(v, 0.0f);
    for (float input : restored->inputs())
        EXPECT_EQ(input, 0.0f);
}

TEST(NetworkSerializationTest, RestoresFrozenState)
{
    reseed(4);
    network net;
    const auto a = net.addLayer("a", {LayerType::Dense, 2, true, true, /*frozen*/ true});
    const auto b = net.addLayer("b", {LayerType::Dense, 2});
    const auto c = net.addLayer("c", {LayerType::Dense, 2});
    net.connect(a, b);
    net.connect(b, c);
    net.unfreeze(a);  // saved state is the current one, not the one at creation
    net.freeze(c);

    std::stringstream ss;
    net.save(ss);
    const auto restored = network::load(ss);
    EXPECT_FALSE(restored->isFrozen(a));
    EXPECT_FALSE(restored->isFrozen(b));
    EXPECT_TRUE(restored->isFrozen(c));
}

namespace {

// Hand-written stream in an older network format: layer "a" (2 neurons,
// habituation + E-R) fed by one sensor, and a layer "b" connected from it,
// frozen where the format supports it. Each neuron of "a" has one weight
// (weights_a), each neuron of "b" two. Every neuron carries non-default
// state (threshold, output) and, from version 3, recovery 0.95 / gain 1.5,
// and version 3-4 layers carry jitter settings, so the test can see that
// none of it is restored.
std::string legacyStream(std::uint32_t version, const std::vector<float>& weights_a, float input_value)
{
    std::ostringstream os;
    auto u8 = [&](std::uint8_t v) { os.write(reinterpret_cast<const char*>(&v), 1); };
    auto u32 = [&](std::uint32_t v) { os.write(reinterpret_cast<const char*>(&v), 4); };
    auto u64 = [&](std::uint64_t v) { os.write(reinterpret_cast<const char*>(&v), 8); };
    auto f32 = [&](float v) { os.write(reinterpret_cast<const char*>(&v), 4); };
    auto b = [&](bool v) { os.write(reinterpret_cast<const char*>(&v), sizeof(bool)); };
    auto sz = [&](size_t v) { os.write(reinterpret_cast<const char*>(&v), sizeof(size_t)); };
    auto addLayer = [&](const std::string& name, std::uint64_t size, bool frozen) {
        u8(0); u64(name.size()); os.write(name.data(), static_cast<std::streamsize>(name.size()));
        u8(0); u64(size); u8(1); u8(1);
        if (version >= 2) u8(frozen);
        if (version == 3) { f32(0.08f); f32(0.2f); }
        if (version == 4)                                  // two Jitter records, v4 layout
            for (int j = 0; j < 2; ++j) { u8(1); f32(0.05f); u8(0); f32(0.0f); f32(-1e30f); f32(1e30f); }
        if (version == 5)                                  // two Jitter records, with the relative flag
            for (int j = 0; j < 2; ++j) { u8(1); f32(0.5f); u8(1); u8(0); f32(0.0f); f32(-1e30f); f32(1e30f); }
    };
    auto neuronData = [&](const std::vector<float>& weights) {
        b(true); b(true); f32(1.2f);
        sz(weights.size());
        for (float w : weights) f32(w);
        f32(3.5f); f32(0.4f); os.write("\x07\0\0\0", 4); f32(2.0f);  // threshold, previous, counter, output
        u32(1234);                                                    // spontaneous generator
        if (version >= 3) { f32(0.95f); f32(1.5f); }                  // recovery, learning gain
    };

    os.write("EXRN", 4);
    u32(version);
    u64(4);                                   // operations
    addLayer("a", 2, false);
    addLayer("b", 2, true);
    u8(3); u64(0); u64(1);                    // Inputs: a, 1 sensor
    u8(1); u64(0); u64(1);                    // Connect: a -> b
    u64(1); u64(1);                           // outputs: b
    u8(0);                                    // default update order
    u64(1); f32(input_value);                 // input values
    for (int layer = 0; layer < 2; ++layer) { // layer data
        b(true); b(true); sz(2);
        for (int n = 0; n < 2; ++n)
            neuronData(layer == 0 ? std::vector<float>{weights_a[n]} : std::vector<float>{0.5f, -0.25f});
    }
    return os.str();
}

} // namespace

TEST(NetworkSerializationTest, OlderFormatsLoadAsWeightsOnly)
{
    const std::vector<float> weights_a = {0.75f, -0.5f};
    for (std::uint32_t version : {1u, 2u, 3u, 4u, 5u}) {
        SCOPED_TRACE("format version " + std::to_string(version));
        std::istringstream is(legacyStream(version, weights_a, 0.7f));
        // FullState is requested, but older files only give topology and weights.
        const auto net = network::load(is, DeserializeMode::FullState);

        ASSERT_EQ(net->layerCount(), 2u);
        EXPECT_EQ(net->layerName(1), "b");
        ASSERT_EQ(net->edges().size(), 1u);
        EXPECT_EQ(net->inputCount(), 1u);
        EXPECT_EQ(net->isFrozen(1), version >= 2);  // frozen flags exist from version 2

        const auto& a = dynamic_cast<const dense&>(net->getLayer(0));
        const auto& bl = dynamic_cast<const dense&>(net->getLayer(1));
        EXPECT_EQ(a.weights(0), std::vector<float>{0.75f});
        EXPECT_EQ(a.weights(1), std::vector<float>{-0.5f});
        EXPECT_EQ(bl.weights(1), (std::vector<float>{0.5f, -0.25f}));

        for (const dense& layer : {std::cref(a), std::cref(bl)})
            for (const neuron& n : layer.neurons()) {
                EXPECT_EQ(n.output(), 0.0f);                      // state not restored
                EXPECT_EQ(n.recovery(), legacy_recovery_factor); // dynamics not restored: the default of the time
                EXPECT_EQ(n.learningGain(), default_learning_gain);
            }
        EXPECT_EQ(net->inputs()[0], 0.0f);                        // input value not restored
        EXPECT_FALSE(net->layerSpec(0).recoveryJitter.enabled()); // jitter settings not restored
        EXPECT_EQ(net->layerSpec(0).recoveryJitter.mean, legacy_recovery_factor);
        EXPECT_FALSE(net->layerSpec(0).learningJitter.enabled());
        EXPECT_EQ(dynamic_cast<const dense&>(net->getLayer(0)).neurons()[0].alpha(), default_alpha);
    }
}

TEST(NetworkSerializationTest, RejectsUnknownFormatVersions)
{
    network net;
    net.addLayer("a", {LayerType::Dense, 1});
    std::stringstream ss;
    net.save(ss);
    for (char version : {0, 99}) {
        std::string bytes = ss.str();
        bytes[4] = version;
        std::stringstream s(bytes);
        EXPECT_THROW(network::load(s), std::runtime_error);
    }
}

TEST(NetworkSerializationTest, FilesBeforeFormat20KeepRecovery09)
{
    // Format 20 changed the default recovery from 0.9 to 0.5 without new data:
    // a file marked 19 grows (and resets) neurons at 0.9, its saved neurons
    // keep their saved values.
    network net;
    const auto a = net.addLayer("a", {LayerType::Dense, 2, false, true});
    const auto pinned = net.addLayer("pinned", {LayerType::Dense, 2, false, true, false,
                                                Jitter::none().around(0.7f)});
    net.addInputs(a, 1);
    net.connect(a, pinned);
    for (const neuron& n : dynamic_cast<const dense&>(net.getLayer(a)).neurons())
        EXPECT_EQ(n.recovery(), recovery_factor);
    for (const neuron& n : dynamic_cast<const dense&>(net.getLayer(pinned)).neurons())
        EXPECT_EQ(n.recovery(), 0.7f);  // a disabled jitter with a centre sets that value
    std::stringstream ss;
    net.save(ss);

    for (char version : {19, 20}) {
        SCOPED_TRACE("format version " + std::to_string(version));
        std::string bytes = ss.str();
        bytes[4] = version;
        std::stringstream s(bytes);
        const auto loaded = network::load(s);
        loaded->growLayer(a, 1);
        loaded->growLayer(pinned, 1);
        const auto& grown = dynamic_cast<const dense&>(loaded->getLayer(a)).neurons();
        ASSERT_EQ(grown.size(), 3u);
        EXPECT_EQ(grown[0].recovery(), recovery_factor);  // saved
        EXPECT_EQ(grown[2].recovery(), version < 20 ? legacy_recovery_factor : recovery_factor);
        EXPECT_EQ(dynamic_cast<const dense&>(loaded->getLayer(pinned)).neurons()[2].recovery(), 0.7f);
    }
}

TEST(NetworkSerializationTest, RestoresJitteredDynamics)
{
    reseed(8);
    network net;
    const Jitter recovery = Jitter::normal(0.02f).around(0.95f).within(0.9f, 0.99f);
    const Jitter learning = Jitter::uniformRelative(0.5f);
    const Jitter alpha = Jitter::normalRelative(0.2f);
    const auto a = net.addLayer("a", {LayerType::Dense, 6, true, true, false, recovery, learning, alpha});
    const auto b = net.addLayer("b", {LayerType::Dense, 4});
    net.connect(a, b);
    net.addFeedback(b, a, 3);  // growth: the 3 new neurons are jittered too

    std::stringstream ss;
    net.save(ss);
    reseed(99);        // loading must restore the saved values, not draw new ones
    const auto restored = network::load(ss);

    EXPECT_EQ(restored->layerSpec(a).recoveryJitter, recovery);
    EXPECT_EQ(restored->layerSpec(a).learningJitter, learning);
    EXPECT_TRUE(restored->layerSpec(a).learningJitter.relative);
    EXPECT_EQ(restored->layerSpec(a).alphaJitter, alpha);

    std::ostringstream text;
    restored->describe(text);
    EXPECT_NE(text.str().find("U+-50%"), std::string::npos) << text.str();
    const auto orig = dynamic_cast<const dense&>(net.getLayer(a)).neurons();
    const auto rest = dynamic_cast<const dense&>(restored->getLayer(a)).neurons();
    ASSERT_EQ(orig.size(), 9u);
    for (size_t i = 0; i < orig.size(); ++i) {
        EXPECT_EQ(rest[i].recovery(), orig[i].recovery());
        EXPECT_EQ(rest[i].learningGain(), orig[i].learningGain());
        EXPECT_EQ(rest[i].alpha(), orig[i].alpha());
        EXPECT_NE(orig[i].alpha(), default_alpha);  // jittered, including the grown neurons
    }
}

TEST(NetworkSerializationTest, RejectsMalformedData)
{
    std::stringstream garbage("definitely not a network");
    EXPECT_THROW(network::load(garbage), std::runtime_error);

    reseed(1);
    auto net = buildSampleNetwork(false);
    std::stringstream ss;
    net->save(ss);
    const std::string bytes = ss.str();

    std::stringstream truncated(bytes.substr(0, bytes.size() / 2));
    EXPECT_THROW(network::load(truncated), std::runtime_error);

    std::string wrong_version = bytes;
    wrong_version[4] = 99;
    std::stringstream versioned(wrong_version);
    EXPECT_THROW(network::load(versioned), std::runtime_error);
}


// =============================================================================
// Gapped pattern detection in a random stream (see pattern_benchmark.hpp):
//   ..., A, {0..n}, B, {0..n}, C, ...  ->  positive for the n ticks after C
// =============================================================================
namespace {

// Everything learns; nothing is hand-wired. Change the topology here.
//
//   sensor -> in (16) -> h1 (32) -> h2 (32) -> h3 (25) -> out (8)
//   feedback: h1 -> h1 (8), h2 -> h1 (16), h3 -> h2 (5)
//
// The feedback loops are the only memory the network has for holding
// "A seen", "A then B seen" across gaps.
std::unique_ptr<network> buildPatternNetwork(bool hasER)
{
    auto net = std::make_unique<network>();
    const auto in = net->addLayer("in", {LayerType::Dense, 16, true, hasER});
    const auto h1 = net->addLayer("h1", {LayerType::Dense, 32, true, hasER});
    const auto h2 = net->addLayer("h2", {LayerType::Dense, 32, true, hasER});
    const auto h3 = net->addLayer("h3", {LayerType::Dense, 25, true, hasER});
    const auto out = net->addLayer("out", {LayerType::Dense, 8, true, hasER});
    net->addInputs(in, 1);
    net->connect(in, h1);
    net->connect(h1, h2);
    net->connect(h2,h3);
    net->connect(h3, out);
    net->addFeedback(h3,h2,5);
    net->addFeedback(h2, h1, 16);
    net->addFeedback(h1, h1, 8);
    net->addOutput(out);
    return net;
}

} // namespace

// Only stability is asserted until a topology learns the pattern; the
// printed verdicts and shortcut rules show how far it got.
TEST(NetworkPatternTest, GappedPatternDetectionWithoutER)
{
    using namespace pattern_benchmark;
    printPatternHeader("E-R off", buildPatternNetwork, false);
    const PatternStats st = measurePattern(buildPatternNetwork, false, PATTERN_LEARNING_RATE);
    printPatternStats("E-R off", st);
    std::cout << "==========================================\n";
    EXPECT_EQ(st.diverged + st.diverged_control, 0);
}

TEST(NetworkPatternTest, GappedPatternDetectionWithER)
{
    using namespace pattern_benchmark;
    printPatternHeader("E-R on", buildPatternNetwork, true);
    const PatternStats st = measurePattern(buildPatternNetwork, true, PATTERN_LEARNING_RATE);
    printPatternStats("E-R on", st);
    std::cout << "==========================================\n";
    EXPECT_EQ(st.diverged + st.diverged_control, 0);
}

// =============================================================================
// Gapped pattern: frozen value detectors + delay window + learned readout
// =============================================================================
namespace {

constexpr int WINDOW_DEPTH = 5;  // lags 0..5: C at lag 1, B at 2..3, A at 3..5 for gaps 0..1

// Only the readout learns. Everything before it is frozen and hand-wired:
//
//   sensor x, bias 1 -> ramps (7) -> bands (4: A, B, C, bias; +-5)
//   bands -> tap1 -> tap2 -> ... -> tap5          (copies, one tick each)
//   window (24) = bands and tap1..tap5 side by side: the last 6 ticks
//   window -> out (8, learns)
//
// Update order: tap5 .. tap1 first (each copies the previous tap before it
// changes), then ramps, bands, window, out.
std::unique_ptr<network> buildWindowReadoutNetwork(bool hasER)
{
    using namespace pattern_benchmark;
    auto net = std::make_unique<network>();
    const ValueDetectors detectors = addValueDetectors(*net);
    std::vector<network::LayerId> order;
    const auto window = addDelayWindow(*net, detectors.bands, BAND_FEATURES, WINDOW_DEPTH, order);
    const auto out = net->addLayer("out", {LayerType::Dense, 8, false, hasER});
    net->connect(window, out);
    net->addOutput(out);
    order.insert(order.end(), {detectors.ramps, detectors.bands, window, out});
    net->setUpdateOrder(order);
    return net;
}

} // namespace

TEST(NetworkPatternTest, FrozenDetectorsAndWindowMatchTheStream)
{
    using namespace pattern_benchmark;
    network net;
    const ValueDetectors detectors = addValueDetectors(net);
    std::vector<network::LayerId> order;
    const auto window = addDelayWindow(net, detectors.bands, BAND_FEATURES, WINDOW_DEPTH, order);
    order.insert(order.end(), {detectors.ramps, detectors.bands, window});
    net.setUpdateOrder(order);
    net.addOutput(window);

    const PatternStream s = makePatternStream(7, 3000);
    const float symbols[3] = {PATTERN_A, PATTERN_B, PATTERN_C};
    for (size_t t = 0; t < s.values.size(); ++t) {
        patternStep(net, s.values[t]);
        const std::vector<float> out = net.outputs();
        for (int lag = 0; lag <= WINDOW_DEPTH && static_cast<size_t>(lag) <= t; ++lag) {
            const float x = s.values[t - lag];
            for (int i = 0; i < 3; ++i) {
                const float expected = std::abs(x - symbols[i]) < 1e-6f ? 5.0f : -5.0f;
                ASSERT_EQ(out[lag * BAND_FEATURES + i], expected) << "tick " << t << ", lag " << lag;
            }
            ASSERT_EQ(out[lag * BAND_FEATURES + 3], 5.0f);  // bias
        }
    }
}

TEST(NetworkPatternTest, WindowReadoutWithoutER)
{
    using namespace pattern_benchmark;
    printPatternHeader("frozen detectors + window, learned readout, E-R off", buildWindowReadoutNetwork, false);
    const PatternStats st = measurePattern(buildWindowReadoutNetwork, false, PATTERN_LEARNING_RATE,
                                           PATTERN_TRIALS, PATTERN_TRAIN_TICKS, RewardMode::Error);
    printPatternStats("E-R off", st);
    std::cout << "==========================================\n";

    // Measured: after C 0.964 (control 0.496), balanced 0.99.
    // The best shortcut rule ("C with B just before it") reaches 0.814.
    EXPECT_EQ(st.diverged + st.diverged_control, 0);
    EXPECT_GE(st.afterC.mean, 0.9f);
    EXPECT_LT(st.afterC.mean_control, 0.6f);
}

TEST(NetworkPatternTest, WindowReadoutWithER)
{
    using namespace pattern_benchmark;
    printPatternHeader("frozen detectors + window, learned readout, E-R on", buildWindowReadoutNetwork, true);
    const PatternStats st = measurePattern(buildWindowReadoutNetwork, true, PATTERN_LEARNING_RATE,
                                           PATTERN_TRIALS, PATTERN_TRAIN_TICKS, RewardMode::Error);
    printPatternStats("E-R on", st);
    std::cout << "==========================================\n";

    // How well an E-R readout learns depends on the E-R constants (the
    // features are fixed at +-5, so a low baseline_threshold means very large
    // learning steps). Measured after C, control ~0.50:
    //   recovery 0.9, baseline 0.2, gain 2.0: see test output
    //   recovery 0.9, baseline 0.1, gain 2.0: 0.610
    //   recovery 0.9, baseline 0.05, gain 2.0: 0.596
    //   recovery 0.8, baseline 0.1, gain 1.5: 0.794
    //   recovery 0.9, baseline 0.1, gain 1.0: 0.882
    // So assert that it learns (clearly beats the no-learning control), not a
    // tuned quality level.
    EXPECT_EQ(st.diverged + st.diverged_control, 0);
    EXPECT_GE(st.afterC.diff, 2.0f * st.afterC.stderr_diff);
    EXPECT_LT(st.afterC.mean_control, 0.6f);
}

// =============================================================================
// Gapped pattern: topology comparison
// =============================================================================
namespace {

using pattern_benchmark::NetworkBuilder;

using pattern_benchmark::FrontEnd;
using pattern_benchmark::addFrontEnd;
using pattern_benchmark::finishFrontEnd;

std::unique_ptr<network> buildWindowHiddenReadout(bool hasER, const Jitter& learningJitter)
{
    auto net = std::make_unique<network>();
    FrontEnd f = addFrontEnd(*net, WINDOW_DEPTH);
    const auto hid = net->addLayer("hid", {LayerType::Dense, 32, false, hasER, false, {}, learningJitter});
    const auto out = net->addLayer("out", {LayerType::Dense, 8, false, hasER, false, {}, learningJitter});
    net->connect(f.features, hid);
    net->connect(hid, out);
    finishFrontEnd(*net, f, {hid, out}, out);
    return net;
}

std::unique_ptr<network> buildWindowFrozenMixReadout(bool hasER)
{
    auto net = std::make_unique<network>();
    FrontEnd f = addFrontEnd(*net, WINDOW_DEPTH);
    const auto mix = net->addLayer("mix", {LayerType::Dense, 128, false, false, /*frozen*/ true});
    const auto out = net->addLayer("out", {LayerType::Dense, 8, false, hasER});
    net->connect(f.features, mix);
    net->connect(mix, out);
    finishFrontEnd(*net, f, {mix, out}, out);
    return net;
}

// Your feedback network, fed by the frozen detectors instead of the raw sensor.
std::unique_ptr<network> buildDetectorsFeedbackNetwork(bool hasER)
{
    auto net = std::make_unique<network>();
    FrontEnd f = addFrontEnd(*net, 0);
    const auto in = net->addLayer("in", {LayerType::Dense, 16, true, hasER});
    const auto h1 = net->addLayer("h1", {LayerType::Dense, 32, true, hasER});
    const auto h2 = net->addLayer("h2", {LayerType::Dense, 32, true, hasER});
    const auto h3 = net->addLayer("h3", {LayerType::Dense, 25, true, hasER});
    const auto out = net->addLayer("out", {LayerType::Dense, 8, true, hasER});
    net->connect(f.features, in);
    net->connect(in, h1);
    net->connect(h1, h2);
    net->connect(h2, h3);
    net->connect(h3, out);
    net->addFeedback(h3, h2, 5);
    net->addFeedback(h2, h1, 16);
    net->addFeedback(h1, h1, 8);
    finishFrontEnd(*net, f, {in, h1, h2, h3, out}, out);
    return net;
}

// Frozen detectors -> frozen random reservoir -> learned readout. Nothing
// between the detectors and the readout is designed for the task. The
// builder's E-R flag applies to the reservoir only; the readout always runs
// without E-R, which learns better (see TopologyComparison).
NetworkBuilder reservoirBuilder(size_t inputNeurons, size_t recurrentNeurons, float recurrentScale,
                                size_t readoutSize)
{
    return [=](bool hasER) {
        using namespace pattern_benchmark;
        auto net = std::make_unique<network>();
        FrontEnd f = addFrontEnd(*net, 0);
        const auto res = addReservoir(*net, f.features, BAND_FEATURES, inputNeurons, recurrentNeurons,
                                      3.0f, recurrentScale, hasER);
        const auto out = net->addLayer("out", {LayerType::Dense, readoutSize, false, false});
        net->connect(res, out);
        finishFrontEnd(*net, f, {res, out}, out);
        return net;
    };
}

} // namespace

// Runs every candidate under the same conditions (error-driven reward,
// COMPARISON_TRIALS seeds each, same streams) and prints them ranked by
// accuracy after C, the sequence measure. Add candidates to the list to
// compare new topologies.
TEST(NetworkPatternTest, TopologyComparison)
{
    using namespace pattern_benchmark;
    constexpr int COMPARISON_TRIALS = 20;

    struct Candidate { const char* name; NetworkBuilder build; };
    const Candidate candidates[] = {
        {"learned feedback net, raw sensor", buildPatternNetwork},
        {"detectors + learned feedback net", buildDetectorsFeedbackNetwork},
        {"detectors + window + readout", buildWindowReadoutNetwork},
        {"detectors + window + hidden + readout",
         [](bool hasER) { return buildWindowHiddenReadout(hasER, Jitter::none()); }},
        {"detectors + window + hidden + readout, learning jitter 0.1",
         [](bool hasER) { return buildWindowHiddenReadout(hasER, Jitter::uniform(0.1f)); }},
        {"detectors + window + frozen mix + readout", buildWindowFrozenMixReadout},
        {"detectors + random reservoir 40+60 + readout", reservoirBuilder(40, 60, 2.0f, 8)},
        {"detectors + E-R neurons, no recurrence + readout", reservoirBuilder(100, 0, 0.0f, 8)},
    };

    struct Row { std::string name, topology; bool hasER; PatternStats st; };
    std::vector<Row> rows;
    for (const Candidate& c : candidates)
        for (bool hasER : {false, true}) {
            reseed(0);
            const std::string topology = topologySummary(*c.build(hasER));
            rows.push_back({c.name, topology, hasER,
                            measurePattern(c.build, hasER, PATTERN_LEARNING_RATE, COMPARISON_TRIALS,
                                           PATTERN_TRAIN_TICKS, RewardMode::Error)});
        }
    std::sort(rows.begin(), rows.end(),
              [](const Row& a, const Row& b) { return a.st.afterC.mean > b.st.afterC.mean; });

    std::cout << "\n==========================================\n"
              << " [Gapped pattern: topology comparison - " << COMPARISON_TRIALS
              << " trials each, error-driven reward]\n"
              << "==========================================\n";
    printShortcutBaselines(COMPARISON_TRIALS);
    std::cout << " Ranked by accuracy after C (valid vs decoy; 0.5 = chance, shortcut best ~0.81).\n"
              << " E-R on/off applies to every layer that is not hand-built, except that reservoir\n"
              << " candidates keep their readout E-R off (E-R in the reservoir only).\n"
              << " Training: " << PATTERN_TRAIN_TICKS << " ticks; reservoirs do better with longer training\n"
              << " (see ReservoirLearnsTheSequence).\n"
              << " Topology: layers in update order, name(neurons + flags: * frozen, E E-R, h habituation,\n"
              << " r recovery jitter, l learning jitter);\n"
              << " fb = feedback edge (new neurons).\n\n";
    int rank = 1;
    for (const Row& r : rows) {
        const Measure& m = r.st.afterC;
        const float t = tStatistic(m.diff, m.stderr_diff);
        std::cout << std::fixed << std::setprecision(3)
                  << " " << rank++ << ". after C " << m.mean << " +- " << m.stderr_mean
                  << "  (control " << m.mean_control << ", t " << std::setprecision(1) << t << ")"
                  << std::setprecision(3) << "  balanced " << r.st.balanced.mean
                  << "  " << r.name << ", E-R " << (r.hasER ? "on" : "off") << "\n"
                  << "    " << r.topology << "\n";
        restoreCoutFormat();
    }
    std::cout << "==========================================\n";

    for (const Row& r : rows)
        EXPECT_EQ(r.st.diverged + r.st.diverged_control, 0) << r.name;
    EXPECT_GE(rows.front().st.afterC.mean, 0.9f);  // the best topology learns the sequence
}


// =============================================================================
// Gapped pattern: memory without hand-built delay lines
// =============================================================================
namespace {

constexpr int RESERVOIR_TRIALS = 20;
constexpr size_t RESERVOIR_TRAIN_TICKS = 40000;  // reservoirs need longer training than the window

void printReservoirResult(const char* label, const pattern_benchmark::PatternStats& st)
{
    pattern_benchmark::ThreeDigits digits;
    const auto& m = st.afterC;
    std::cout << "   after C " << m.mean << " +- " << m.stderr_mean << " (control " << m.mean_control
              << "), balanced " << st.balanced.mean << "  <- " << label << "\n";
}

} // namespace

// The hand-built delay window replaced by a frozen random recurrent
// reservoir: the readout still finds A..B..C in the reservoir's activity.
TEST(NetworkPatternTest, ReservoirLearnsTheSequence)
{
    using namespace pattern_benchmark;
    const NetworkBuilder build = reservoirBuilder(40, 60, 2.0f, 1);
    printPatternHeader("detectors + random reservoir 40+60 + 1-neuron readout, E-R off", build, false,
                       RESERVOIR_TRIALS);
    const PatternStats st = measurePattern(build, false, PATTERN_LEARNING_RATE, RESERVOIR_TRIALS,
                                           RESERVOIR_TRAIN_TICKS, RewardMode::Error);
    printPatternStats("E-R off", st, RESERVOIR_TRIALS);
    std::cout << "==========================================\n";

    // Measured: after C 0.798 (control 0.490); the hand-built window reaches
    // ~1.0 and the best shortcut rule 0.81.
    EXPECT_GE(st.afterC.mean, 0.75f);
    EXPECT_LT(st.afterC.mean_control, 0.6f);
    EXPECT_EQ(st.diverged + st.diverged_control, 0);
}

// E-R neurons with no connections between them: the only memory is each
// neuron's own adaptive threshold. With E-R off the same layer is memoryless.
TEST(NetworkPatternTest, ERNeuronsCarryMemoryWithoutRecurrence)
{
    using namespace pattern_benchmark;
    const NetworkBuilder build = reservoirBuilder(100, 0, 0.0f, 1);
    printPatternHeader("detectors + 100 unconnected neurons + 1-neuron readout", build, true, RESERVOIR_TRIALS);
    const PatternStats off = measurePattern(build, false, PATTERN_LEARNING_RATE, RESERVOIR_TRIALS,
                                            RESERVOIR_TRAIN_TICKS, RewardMode::Error);
    const PatternStats on = measurePattern(build, true, PATTERN_LEARNING_RATE, RESERVOIR_TRIALS,
                                           RESERVOIR_TRAIN_TICKS, RewardMode::Error);
    printReservoirResult("E-R off", off);
    printReservoirResult("E-R on", on);
    std::cout << "==========================================\n";

    // Measured: E-R off 0.500 exactly (no memory at all), E-R on 0.757 +- 0.030.
    EXPECT_LE(off.afterC.mean, 0.55f);
    EXPECT_GE(on.afterC.mean, 0.62f);
    EXPECT_GT(on.afterC.mean - off.afterC.mean, 2.0f * on.afterC.stderr_mean);
    EXPECT_EQ(off.diverged + off.diverged_control + on.diverged + on.diverged_control, 0);
}


// =============================================================================
// Per-neuron learning-speed jitter
// =============================================================================

// Does a small per-neuron learning-speed jitter (gain +- 0.1) help a learned
// E-R hidden layer? Measured at base gains 1.0 and default_learning_gain,
// paired by seed (same networks and streams, only the jitter differs).
//
// Reported, not asserted: the effect has appeared and vanished with every
// change of the E-R constants. Examples:
//   recovery 0.9, baseline 0.1: gain 1.0 +0.166 (t 5.0), gain 1.5 -0.019 (t -0.7)
//   recovery 0.8, baseline 0.1: gain 1.0 +0.239 (t 7.2), gain 1.5 -0.017 (t -0.6)
//   recovery 0.9, baseline 0.2: gain 1.0 +0.028 (t 1.4), gain 2.0 +0.203 (t 6.5)
// It is not a robust property of the model.
TEST(NetworkPatternTest, LearningJitterEffectDependsOnBaseGain)
{
    using namespace pattern_benchmark;
    constexpr int TRIALS = 50;

    struct Result { float base, jittered, diff, se; };
    auto measure = [&](float gain) {
        auto build = [gain](bool jitter) {
            return [gain, jitter](bool hasER) {
                const Jitter j = jitter ? Jitter::uniform(0.1f).around(gain)
                                        : Jitter::uniform(1e-9f).around(gain);  // no spread: exactly `gain`
                return buildWindowHiddenReadout(hasER, j);
            };
        };
        std::vector<float> base, jittered, diffs;
        for (int trial = 0; trial < TRIALS; ++trial) {
            const auto seed = static_cast<std::uint32_t>(trial);
            base.push_back(runPatternTrial(build(false), seed, true, PATTERN_LEARNING_RATE, PATTERN_TRAIN_TICKS,
                                           RewardMode::Error).afterC);
            jittered.push_back(runPatternTrial(build(true), seed, true, PATTERN_LEARNING_RATE, PATTERN_TRAIN_TICKS,
                                               RewardMode::Error).afterC);
            diffs.push_back(jittered.back() - base.back());
        }
        const float diff = meanOf(diffs);
        return Result{meanOf(base), meanOf(jittered), diff, standardError(diffs, diff)};
    };

    const Result at1 = measure(1.0f);
    const Result atDefault = measure(default_learning_gain);

    pattern_benchmark::ThreeDigits digits;
    std::cout << "\n==========================================\n"
              << " [Learning jitter +-0.1, E-R hidden layer + readout - " << TRIALS << " paired trials]\n"
              << "==========================================\n";
    for (const auto& [gain, r] : {std::pair{1.0f, at1}, std::pair{default_learning_gain, atDefault}})
        std::cout << "   base gain " << gain << ": after C " << r.base << " -> " << r.jittered
                  << " with jitter, paired difference " << r.diff << " +- " << r.se << " (t " << r.diff / r.se << ")\n";
    std::cout << "==========================================\n";

    EXPECT_TRUE(std::isfinite(at1.diff) && std::isfinite(atDefault.diff));
}


// =============================================================================
// Does alpha jitter help? Pavlovian sign inversion and 3-number sequences
// =============================================================================
namespace {

struct AlphaSetting
{
    const char* name;
    Jitter jitter;
};

// The alpha jitter settings compared, all relative to alpha's value so they
// keep their meaning if default_alpha changes. The first is the baseline.
const AlphaSetting ALPHA_SETTINGS[] = {
    {"none", Jitter::none()},
    {"uniform +-25%", Jitter::uniformRelative(0.25f)},
    {"uniform +-50%", Jitter::uniformRelative(0.5f)},
    {"normal sd 25%", Jitter::normalRelative(0.25f)},
};

// One row: setting, main measure, extra column, and (except for the
// baseline) a paired comparison with the baseline on the same seeds.
// `higherIsBetter`: accuracy (true) or epochs needed (false).
void printAlphaRow(const char* setting, const std::vector<float>& base, const std::vector<float>& with,
                   bool higherIsBetter, const std::string& extra)
{
    using namespace pattern_benchmark;
    std::cout << "   " << std::left << std::setw(15) << setting << std::right << std::fixed << std::setprecision(3)
              << std::setw(7) << meanOf(with) << "   " << std::left << std::setw(16) << extra << std::right;
    if (&base == &with) {
        std::cout << "baseline\n";
    } else {
        std::vector<float> diffs;
        for (size_t i = 0; i < with.size(); ++i) diffs.push_back(with[i] - base[i]);
        const float d = meanOf(diffs), se = standardError(diffs, d);
        const float t = tStatistic(higherIsBetter ? d : -d, se);
        std::cout << std::showpos << d << std::noshowpos << " +- " << se << "  (t " << std::setprecision(1)
                  << tStatistic(d, se) << ")  " << verdict(t, "HELPS", "HURTS") << "\n";
    }
    restoreCoutFormat();
}

// --- Pavlovian sign inversion ----------------------------------------------

constexpr int ALPHA_PAVLOV_TRIALS = 50;
constexpr int ALPHA_PAVLOV_MAX_EPOCHS = 60;

// One fixed topology: A -> B -> C -> D (5 neurons each, habituation + E-R),
// feedback A -> C (20 neurons), the stimulus on A. Alpha jitter on every layer.
std::unique_ptr<network> buildPavlovianNetwork(const Jitter& alphaJitter)
{
    auto net = std::make_unique<network>();
    auto layer = [&](const char* name) {
        return net->addLayer(name, {LayerType::Dense, 5, true, true, false, {}, {}, alphaJitter});
    };
    const auto a = layer("A"), b = layer("B"), c = layer("C"), d = layer("D");
    net->connect(a, b);
    net->connect(b, c);
    net->connect(c, d);
    net->addFeedback(a, c, 20);
    net->addInputs(a, 1);
    net->addOutput(d);
    return net;
}

struct PavlovianLearning
{
    int epochs;    // training epochs until both responses have the right sign (MAX + 1 if never)
    bool learned;  // correct after all epochs
};

// Trains -stimulus -> +, +stimulus -> - (reward = desired sign, both stimuli
// per epoch) and tests after every epoch. Responses are averaged over 4
// ticks from rest, as in the dense.cpp Pavlovian tests.
PavlovianLearning runAlphaPavlovianTrial(std::uint32_t seed, const Jitter& alphaJitter)
{
    reseed(seed);
    const auto net = buildPavlovianNetwork(alphaJitter);
    auto propagate = [&](float x, int ticks) {
        net->setInputs({x});
        for (int t = 0; t < ticks; ++t) net->step();
    };
    auto response = [&](float x) {
        propagate(0.0f, 6);
        float sum = 0.0f;
        for (int t = 0; t < 4; ++t) {
            propagate(x, 1);
            float mean = 0.0f;
            for (float v : net->outputs()) mean += v / 5.0f;
            sum += mean / 4.0f;
        }
        return sum;
    };
    auto train = [&](float x, float desired) {
        propagate(0.0f, 20);
        propagate(x, 4);
        net->applyReward(desired, 0.005f);
    };

    PavlovianLearning r{ALPHA_PAVLOV_MAX_EPOCHS + 1, false};
    for (int epoch = 1; epoch <= ALPHA_PAVLOV_MAX_EPOCHS; ++epoch) {
        train(-WEAK_STIMULUS, +1.0f);
        train(+WEAK_STIMULUS, -1.0f);
        const bool correct = response(-WEAK_STIMULUS) > 0.0f && response(+WEAK_STIMULUS) < 0.0f;
        if (correct && r.epochs > ALPHA_PAVLOV_MAX_EPOCHS)
            r.epochs = epoch;
        r.learned = correct;
    }
    return r;
}

// --- 3-number sequence detection, no decoys --------------------------------

// Three sequences of 3 numbers, 9 distinct values. The first is the trigger:
// the network should respond positively on the tick after its last number.
constexpr float SEQUENCES[3][3] = {{3.0f, 7.0f, 5.0f}, {8.0f, 2.0f, 6.0f}, {4.0f, 9.0f, 1.0f}};
constexpr float SEQ_BAND = 0.25f;       // detector half-width
constexpr float SEQ_EXCLUSION = 0.5f;   // distractors never within this of a sequence value
// Ticks between the trigger's last number and the tick that must respond;
// the E-R trace of the trigger has to survive this long. With a delay of 1
// every setting scored ~1.0 (a ceiling that cannot show an effect); 15 puts
// the baseline around 0.8.
constexpr int SEQ_DELAY = 15;
constexpr int ALPHA_SEQ_TRIALS = 50;
constexpr size_t ALPHA_SEQ_TRAIN = 20000;
constexpr size_t ALPHA_SEQ_TEST = 3000;

struct SequenceStream
{
    std::vector<float> values;
    std::vector<int> target;  // +1 on the tick after the trigger's last number
};

// Random distractors in [0, 10] (away from the sequence values), with whole
// sequences embedded: after 1-4 distractors, a sequence with probability 0.6,
// each of the three equally likely. No partial or reordered sequences.
SequenceStream makeSequenceStream(std::uint32_t seed, size_t length)
{
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> chance(0.0f, 1.0f), value(0.0f, 10.0f);
    std::uniform_int_distribution<int> which(0, 2), filler(1, 4);
    SequenceStream s;
    auto distractor = [&] {
        for (;;) {
            const float v = value(rng);
            bool near = false;
            for (const auto& seq : SEQUENCES)
                for (float x : seq) near = near || std::abs(v - x) < SEQ_EXCLUSION;
            if (!near) return v;
        }
    };
    while (s.values.size() < length) {
        for (int i = filler(rng); i > 0; --i) { s.values.push_back(distractor()); s.target.push_back(-1); }
        if (chance(rng) < 0.6f) {
            const int k = which(rng);
            for (float x : SEQUENCES[k]) { s.values.push_back(x); s.target.push_back(-1); }
            for (int i = 1; i < SEQ_DELAY; ++i) { s.values.push_back(distractor()); s.target.push_back(-1); }
            s.values.push_back(distractor());
            s.target.push_back(k == 0 ? +1 : -1);  // SEQ_DELAY ticks after the trigger's last number
        }
    }
    s.values.resize(length);
    s.target.resize(length);
    return s;
}

// One fixed topology: sensor + bias -> frozen detectors for the 9 values ->
// 100 unconnected E-R neurons (frozen; their only memory is the E-R
// threshold, which alpha shapes) with the alpha jitter -> learned 1-neuron
// readout (no E-R).
std::unique_ptr<network> buildSequenceNetwork(const Jitter& alphaJitter)
{
    using namespace pattern_benchmark;
    std::vector<float> symbols;
    for (const auto& seq : SEQUENCES) symbols.insert(symbols.end(), std::begin(seq), std::end(seq));
    auto net = std::make_unique<network>();
    const ValueDetectors d = addSymbolDetectors(*net, symbols, SEQ_BAND);
    // Recovery 0.9 (the default before network format 20): at 0.5 the
    // reservoir forgets the trigger and learning stays at chance (0.53).
    const auto memory = addReservoir(*net, d.bands, symbols.size() + 1, 100, 0, 3.0f, 0.0f, true,
                                     Jitter::none().around(legacy_recovery_factor), alphaJitter);
    const auto out = net->addLayer("out", {LayerType::Dense, 1, false, false});
    net->connect(memory, out);
    net->addOutput(out);
    net->setUpdateOrder({d.ramps, d.bands, memory, out});
    return net;
}

struct SequenceScore
{
    float balanced, hits;  // balanced accuracy; share of trigger ticks answered positively
    bool diverged;
};

// learningRate 0 gives the no-learning control.
SequenceScore runAlphaSequenceTrial(std::uint32_t seed, const Jitter& alphaJitter, float learningRate)
{
    using namespace pattern_benchmark;
    reseed(seed);
    const auto net = buildSequenceNetwork(alphaJitter);
    bool diverged = false;

    const SequenceStream train = makeSequenceStream(3000 + seed, ALPHA_SEQ_TRAIN);
    const auto positives = std::count(train.target.begin(), train.target.end(), +1);
    const float pos_weight = static_cast<float>(ALPHA_SEQ_TRAIN - positives) / static_cast<float>(positives);
    for (size_t t = 0; t < train.values.size(); ++t) {
        const float r = patternStep(*net, train.values[t]);
        diverged = diverged || isDivergent(r);
        const bool positive = train.target[t] > 0;
        if ((r > 0.0f) != positive)  // error-driven reward
            net->applyReward(positive ? pos_weight : -1.0f, learningRate);
    }

    const SequenceStream test = makeSequenceStream(4000 + seed, ALPHA_SEQ_TEST);
    int pos = 0, neg = 0, tp = 0, tn = 0;
    for (size_t t = 0; t < test.values.size(); ++t) {
        const float r = patternStep(*net, test.values[t]);
        diverged = diverged || isDivergent(r);
        if (test.target[t] > 0) { ++pos; tp += r > 0.0f; } else { ++neg; tn += r <= 0.0f; }
    }
    const float hits = static_cast<float>(tp) / pos;
    return {0.5f * (hits + static_cast<float>(tn) / neg), hits, diverged};
}

} // namespace

TEST(AlphaJitterTest, PavlovianSignInversion)
{
    std::vector<std::vector<float>> epochs;
    std::vector<int> learned;
    for (const AlphaSetting& s : ALPHA_SETTINGS) {
        std::vector<float> e;
        int ok = 0;
        for (int trial = 0; trial < ALPHA_PAVLOV_TRIALS; ++trial) {
            const PavlovianLearning r = runAlphaPavlovianTrial(static_cast<std::uint32_t>(trial), s.jitter);
            e.push_back(static_cast<float>(r.epochs));
            ok += r.learned;
        }
        epochs.push_back(e);
        learned.push_back(ok);
    }

    reseed(0);
    std::cout << "\n==========================================\n"
              << " [Alpha jitter: Pavlovian sign inversion - " << ALPHA_PAVLOV_TRIALS << " paired trials]\n"
              << "==========================================\n"
              << " Topology: " << pattern_benchmark::topologySummary(*buildPavlovianNetwork(ALPHA_SETTINGS[2].jitter))
              << "\n Stimulus +-" << WEAK_STIMULUS << "; measure: epochs until both responses have the right\n"
              << " sign (fewer is better; " << ALPHA_PAVLOV_MAX_EPOCHS + 1 << " = never).\n\n"
              << "   alpha jitter    epochs    learned by " << ALPHA_PAVLOV_MAX_EPOCHS << "   vs none (paired)\n";
    for (size_t i = 0; i < std::size(ALPHA_SETTINGS); ++i)
        printAlphaRow(ALPHA_SETTINGS[i].name, epochs[0], epochs[i], false,
                      std::to_string(learned[i]) + "/" + std::to_string(ALPHA_PAVLOV_TRIALS));
    std::cout << "==========================================\n";

    // Measured (recovery 0.9, baseline 0.2, gain 2.0, alpha 1.2): baseline 5.68
    // epochs, 50/50 learned by every setting; every jitter within noise
    // (t 0.0 to 1.4). The question is answered by the printout; assert only
    // that the baseline learns, so a broken setup does not look like "no effect".
    EXPECT_GE(learned[0], ALPHA_PAVLOV_TRIALS * 9 / 10);
}

TEST(AlphaJitterTest, ThreeNumberSequenceDetection)
{
    std::vector<std::vector<float>> balanced, hits;
    std::vector<float> control;
    bool diverged = false;
    for (const AlphaSetting& s : ALPHA_SETTINGS) {
        std::vector<float> b, h;
        for (int trial = 0; trial < ALPHA_SEQ_TRIALS; ++trial) {
            const SequenceScore r = runAlphaSequenceTrial(static_cast<std::uint32_t>(trial), s.jitter,
                                                          pattern_benchmark::PATTERN_LEARNING_RATE);
            b.push_back(r.balanced);
            h.push_back(r.hits);
            diverged = diverged || r.diverged;
        }
        balanced.push_back(b);
        hits.push_back(h);
    }
    for (int trial = 0; trial < ALPHA_SEQ_TRIALS; ++trial)
        control.push_back(runAlphaSequenceTrial(static_cast<std::uint32_t>(trial), Jitter::none(), 0.0f).balanced);

    reseed(0);
    std::cout << "\n==========================================\n"
              << " [Alpha jitter: 3-number sequence detection, no decoys - " << ALPHA_SEQ_TRIALS
              << " paired trials]\n"
              << "==========================================\n"
              << " Sequences: trigger (3, 7, 5); others (8, 2, 6), (4, 9, 1). Respond " << SEQ_DELAY
              << " ticks after the\n trigger's last number (random numbers in between).\n"
              << " Topology: " << pattern_benchmark::topologySummary(*buildSequenceNetwork(ALPHA_SETTINGS[2].jitter))
              << "\n Balanced accuracy, 0.5 = chance; no-learning control " << std::setprecision(3)
              << pattern_benchmark::meanOf(control) << ".\n\n"
              << "   alpha jitter  balanced   hits on trigger   vs none (paired)\n";
    restoreCoutFormat();
    for (size_t i = 0; i < std::size(ALPHA_SETTINGS); ++i) {
        std::ostringstream h;
        h << std::fixed << std::setprecision(2) << pattern_benchmark::meanOf(hits[i]);
        printAlphaRow(ALPHA_SETTINGS[i].name, balanced[0], balanced[i], true, h.str());
    }
    std::cout << "==========================================\n";

    // Measured (same constants): baseline 0.847, control 0.497; uniform +-25%
    // 0.765 (t -2.1), uniform +-50% 0.839 (t -0.2), normal sd 25% 0.806
    // (t -1.2). No setting helps; the single t = -2.1 among three
    // comparisons, with the larger +-50% jitter showing nothing, is most
    // likely noise.
    EXPECT_FALSE(diverged);
    EXPECT_GT(pattern_benchmark::meanOf(balanced[0]), pattern_benchmark::meanOf(control) + 0.1f);  // it learns
}

// =============================================================================
// Fixed firing threshold (gate) for neurons without E-R
// =============================================================================
TEST(GateTest, FixedThresholdPassesOnlyStrongSums)
{
    neuron n(false, false);
    n.setGate(0.2f);
    EXPECT_EQ(n.activate(0.1f), 0.0f);
    EXPECT_EQ(n.activate(-0.2f), 0.0f);
    EXPECT_EQ(n.activate(0.5f), 0.5f);
    EXPECT_EQ(n.activate(-0.7f), -0.7f);
    EXPECT_EQ(n.threshold(), baseline_threshold);  // nothing adapts
    EXPECT_TRUE(n.eligible());                     // it fired this tick
    n.activate(0.1f);
    EXPECT_FALSE(n.eligible());

    neuron er(false, true);
    er.setGate(5.0f);  // ignored with E-R
    EXPECT_EQ(er.activate(1.0f), 1.0f);
}

TEST(GateTest, LayerSpecGateIsAppliedSavedAndChecked)
{
    network net;
    LayerSpec spec = LayerSpec::Dense(3, false, false);
    spec.gate = 0.5f;
    const auto in = net.addLayer("in", LayerSpec::Dense(2, false, false));
    const auto gated = net.addLayer("gated", spec);
    net.addInputs(in, 2);
    net.connect(in, gated);
    net.addFeedback(in, gated, 2);  // neurons added later get the gate too
    for (const neuron& n : net.layerAs<neuron_layer>(gated).neurons())
        EXPECT_EQ(n.gate(), 0.5f);
    std::ostringstream text;
    net.describe(text);
    EXPECT_NE(text.str().find("=0.5"), std::string::npos) << text.str();

    std::stringstream data;
    net.save(data);
    auto loaded = network::load(data);
    EXPECT_EQ(loaded->layerSpec(gated).gate, 0.5f);
    for (const neuron& n : loaded->layerAs<neuron_layer>(gated).neurons())
        EXPECT_EQ(n.gate(), 0.5f);

    LayerSpec bad = LayerSpec::Dense(1, false, true);
    bad.gate = 0.2f;
    EXPECT_THROW(net.addLayer("er", bad), std::invalid_argument);
    bad.hasER = false;
    bad.gate = -1.0f;
    EXPECT_THROW(net.addLayer("negative", bad), std::invalid_argument);
}

TEST(GateTest, RectifiedNeuronIsAReLU)
{
    neuron n(false, false);
    n.setRectified(true);
    EXPECT_EQ(n.activate(-0.5f), 0.0f);
    EXPECT_FALSE(n.eligible());
    EXPECT_EQ(n.activate(0.0f), 0.0f);
    EXPECT_EQ(n.activate(0.3f), 0.3f);
    EXPECT_EQ(n.activate(20.0f), max_output);  // still clamped
    n.setGate(0.5f);                           // with a gate: only sums above it
    EXPECT_EQ(n.activate(0.4f), 0.0f);
    EXPECT_EQ(n.activate(-0.9f), 0.0f);
    EXPECT_EQ(n.activate(0.6f), 0.6f);

    network net;
    LayerSpec spec = LayerSpec::Dense(2, false, false);
    spec.rectify = true;
    const auto relu = net.addLayer("relu", spec);
    net.addInputs(relu, 1);
    net.addOutput(relu);
    net.layerAs<dense>(relu).setWeights(0, {1.0f});
    net.layerAs<dense>(relu).setWeights(1, {-1.0f});
    net.setInputs(std::vector<float>{0.7f});
    net.step();
    EXPECT_EQ(net.outputs()[0], 0.7f);
    EXPECT_EQ(net.outputs()[1], 0.0f);
    std::ostringstream text;
    net.describe(text);
    EXPECT_NE(text.str().find("relu"), std::string::npos) << text.str();

    std::stringstream data;
    net.save(data);
    auto loaded = network::load(data);
    EXPECT_TRUE(loaded->layerSpec(relu).rectify);
    for (const neuron& n2 : loaded->layerAs<neuron_layer>(relu).neurons())
        EXPECT_TRUE(n2.rectified());

    LayerSpec bad = LayerSpec::Dense(1, false, true);
    bad.rectify = true;
    EXPECT_THROW(net.addLayer("er", bad), std::invalid_argument);
}

TEST(HabituationTest, DefaultRuleCutsAfterExactRepeats)
{
    neuron n(true, false);
    for (int i = 0; i < habituation_steps - 1; ++i)
        EXPECT_EQ(n.activate(0.5f), 0.5f) << i;  // the first tick starts the streak only if the sum equals 0
    float last = 0.5f;
    for (int i = 0; i < 3; ++i)
        last = n.activate(0.5f);
    EXPECT_EQ(last, 0.0f);
    EXPECT_EQ(n.activate(0.6f), 0.6f);  // a change restores it
}

TEST(HabituationTest, FasterFadingAndTolerantRules)
{
    neuron n(true, false);
    n.setHabituation({100, 0.0f, 0.5f, 3});  // fade mode: fading starts at the 3rd repeat, `steps` unused
    // Streak: tick 1 differs from the initial 0; ticks 2..4 repeat it.
    EXPECT_EQ(n.activate(1.0f), 1.0f);
    EXPECT_EQ(n.activate(1.0f), 1.0f);
    EXPECT_EQ(n.activate(1.0f), 1.0f);
    EXPECT_FLOAT_EQ(n.activate(1.0f), 0.5f);   // habituated: fades
    EXPECT_FLOAT_EQ(n.activate(1.0f), 0.25f);
    EXPECT_EQ(n.activate(2.0f), 2.0f);         // a change restores it

    neuron quick(true, false);
    quick.setHabituation({100, 0.0f, 0.5f});   // fadeAfter defaults to 2
    EXPECT_EQ(quick.activate(1.0f), 1.0f);
    EXPECT_EQ(quick.activate(1.0f), 1.0f);
    EXPECT_FLOAT_EQ(quick.activate(1.0f), 0.5f);  // the 2nd repeat already fades

    neuron flicker(true, false);
    flicker.setHabituation({3, 0.01f, 0.0f});  // 1 % counts as the same signal
    float y = 0.0f;
    for (int i = 0; i < 6; ++i)
        y = flicker.activate(i % 2 ? 1.0f : 1.005f);
    EXPECT_EQ(y, 0.0f);
    neuron exact(true, false);
    exact.setHabituation({3, 0.0f, 0.0f});
    for (int i = 0; i < 6; ++i)
        y = exact.activate(i % 2 ? 1.0f : 1.005f);
    EXPECT_NE(y, 0.0f);  // without tolerance, flicker never habituates

    network net;
    LayerSpec spec = LayerSpec::Dense(2, true, false);
    spec.habituationRule = {5, 0.02f, 0.9f};
    const auto id = net.addLayer("h", spec);
    net.addInputs(id, 1);
    for (const neuron& m : net.layerAs<neuron_layer>(id).neurons())
        EXPECT_EQ(m.habituation(), spec.habituationRule);
    std::stringstream data;
    net.save(data);
    auto loaded = network::load(data);
    EXPECT_EQ(loaded->layerSpec(id).habituationRule, spec.habituationRule);
    for (const neuron& m : loaded->layerAs<neuron_layer>(id).neurons())
        EXPECT_EQ(m.habituation(), spec.habituationRule);

    LayerSpec bad = LayerSpec::Dense(1, true, false);
    bad.habituationRule.decay = 1.5f;
    EXPECT_THROW(net.addLayer("bad", bad), std::invalid_argument);
}

TEST(SpontaneousTest, TriggerLevelRateAndAmplitude)
{
    // Recovery 0.9: silent for about 200 ticks (0.2 * 0.9^t reaches 1e-10),
    // then a small spontaneous firing.
    auto firstFiring = [](Spontaneous s, float* amplitude = nullptr) {
        neuron n(false, true);
        n.setRecovery(legacy_recovery_factor);
        n.setSpontaneous(s);
        for (int t = 1; t < 10000; ++t) {
            const float y = n.activate(0.0f);
            if (y != 0.0f) {
                if (amplitude)
                    *amplitude = std::abs(y);
                return t;
            }
        }
        return -1;
    };
    float a = 0.0f;
    const int original = firstFiring({}, &a);
    EXPECT_GT(original, 150);
    EXPECT_LE(a, spontaneous_min_amplitude);
    const int early = firstFiring({1e-3f, 1.0f, 0.0f}, &a);  // fires once the threshold reaches 1e-3
    EXPECT_LT(early, original);
    EXPECT_GT(early, 40);
    EXPECT_LE(a, 1.0f);
    EXPECT_LT(firstFiring({min_threshold, 0.5f, 0.2f}), 60);  // 20 % per silent tick

    // A weak spontaneous firing never lowers a high threshold.
    neuron high(false, true);
    high.setSpontaneous({min_threshold, 0.01f, 1.0f});
    high.activate(8.0f);
    const float before = high.threshold();
    high.activate(0.0f);
    EXPECT_GE(high.threshold(), before * recovery_factor);

    network net;
    LayerSpec spec = LayerSpec::Dense(2, true, true);
    spec.spontaneous = {0.01f, 0.5f, 0.05f};
    spec.habituationRule = {100, 0.0f, 0.9f, 4};
    const auto id = net.addLayer("h", spec);
    net.addInputs(id, 1);
    std::stringstream data;
    net.save(data);
    auto loaded = network::load(data);
    EXPECT_EQ(loaded->layerSpec(id).spontaneous, spec.spontaneous);
    EXPECT_EQ(loaded->layerSpec(id).habituationRule, spec.habituationRule);
    for (const neuron& m : loaded->layerAs<neuron_layer>(id).neurons()) {
        EXPECT_EQ(m.spontaneous(), spec.spontaneous);
        EXPECT_EQ(m.habituation(), spec.habituationRule);
    }

    LayerSpec bad = LayerSpec::Dense(1, false, true);
    bad.spontaneous.rate = 2.0f;
    EXPECT_THROW(net.addLayer("bad", bad), std::invalid_argument);
}

TEST(ThresholdGrowthTest, RulesAndSaving)
{
    using Rule = ThresholdGrowth::Rule;
    // One firing of magnitude 5 from rest (threshold baseline_threshold).
    auto grownBy = [](ThresholdGrowth g) {
        neuron n(false, true);
        n.setThresholdGrowth(g);
        EXPECT_EQ(n.activate(5.0f), 5.0f);
        return n.threshold();
    };
    const float b = baseline_threshold;
    EXPECT_FLOAT_EQ(grownBy({}), b + 0.5f * (5.0f - b));  // the default: linear
    EXPECT_FLOAT_EQ(grownBy({Rule::Log, 0.5f}), b + default_alpha * std::log(5.0f / b));  // the original
    EXPECT_FLOAT_EQ(grownBy({Rule::Linear, 0.5f}), b + 0.5f * (5.0f - b));
    EXPECT_FLOAT_EQ(grownBy({Rule::Fixed, 1.0f}), b + 1.0f);
    EXPECT_FLOAT_EQ(grownBy({Rule::Multiplicative, 0.5f}), 2.0f * b);  // 1.5 b, raised to the floor 2 b
    EXPECT_FLOAT_EQ(grownBy({Rule::Fixed, 0.0f}), 2.0f * b);

    network net;
    LayerSpec spec = LayerSpec::Dense(3, false, true);
    spec.thresholdGrowth = {Rule::Log, 0.25f};
    const auto id = net.addLayer("h", spec);
    net.addInputs(id, 1);
    for (const neuron& m : net.layerAs<neuron_layer>(id).neurons())
        EXPECT_EQ(m.thresholdGrowth(), spec.thresholdGrowth);
    std::stringstream data;
    net.save(data);
    auto loaded = network::load(data);
    EXPECT_EQ(loaded->layerSpec(id).thresholdGrowth, spec.thresholdGrowth);
    for (const neuron& m : loaded->layerAs<neuron_layer>(id).neurons())
        EXPECT_EQ(m.thresholdGrowth(), spec.thresholdGrowth);

    // A format-13 file (no growth rule saved) loads with the log rule, the
    // only one it knew: drop this layer's rule bytes and the later fields
    // after them (fadeAfter, spontaneous below / amplitude / rate, resting
    // threshold, normalize, binary, bus, minimum size, grown), and mark it version 13. (The layer
    // data is then read as neuron format 3; the format-4 bytes after it are
    // never reached.)
    network plain;
    const auto p = plain.addLayer("h", LayerSpec::Dense(2, false, true));
    plain.addInputs(p, 1);
    std::stringstream current;
    plain.save(current);
    std::string bytes = current.str();
    const float amount = 0.5f;
    std::string ruleBytes(1, static_cast<char>(ThresholdGrowth::Rule::Linear));
    ruleBytes.append(reinterpret_cast<const char*>(&amount), sizeof amount);
    const size_t at = bytes.find(ruleBytes);
    ASSERT_NE(at, std::string::npos);
    ASSERT_EQ(bytes.find(ruleBytes, at + 1), std::string::npos);
    bytes.erase(at, ruleBytes.size() + sizeof(std::uint32_t) + 4 * sizeof(float) + 3 * sizeof(std::uint8_t) +
                        sizeof(std::uint64_t) + sizeof(std::uint8_t));
    const std::uint32_t v13 = 13;
    bytes.replace(4, sizeof v13, reinterpret_cast<const char*>(&v13), sizeof v13);
    std::stringstream old(bytes);
    auto oldNet = network::load(old);
    EXPECT_EQ(oldNet->layerSpec(p).thresholdGrowth.rule, Rule::Log);
    for (const neuron& m : oldNet->layerAs<neuron_layer>(p).neurons())
        EXPECT_EQ(m.thresholdGrowth().rule, Rule::Log);

    LayerSpec bad = LayerSpec::Dense(1, false, true);
    bad.thresholdGrowth.amount = -1.0f;
    EXPECT_THROW(net.addLayer("bad", bad), std::invalid_argument);
}

TEST(NormalizedSumTest, DividesByWeightLengthAndFollowsChanges)
{
    // Linear neurons (no E-R, no habituation) show the normalised sum as is.
    network net;
    LayerSpec spec = LayerSpec::Dense(1, false, false);
    spec.normalize = true;
    const auto id = net.addLayer("h", spec);
    net.addInputs(id, 2);
    auto& layer = net.layerAs<dense>(id);
    layer.setWeights(0, {3.0f, 4.0f});
    net.setInputs({1.0f, 1.0f});
    net.step();
    EXPECT_FLOAT_EQ(layer.neurons()[0].output(), 7.0f / 5.0f);
    layer.setWeights(0, {6.0f, 8.0f});  // same direction, twice as long: same output
    net.step();
    EXPECT_FLOAT_EQ(layer.neurons()[0].output(), 7.0f / 5.0f);
    layer.setWeights(0, {0.0f, 0.0f});  // no weights: the raw sum (0), not a division by 0
    net.step();
    EXPECT_EQ(layer.neurons()[0].output(), 0.0f);

    // After learning, the next step uses the new weights' length.
    layer.setWeights(0, {3.0f, 4.0f});
    net.addOutput(id);
    net.step();
    net.applyReward(1.0f, 0.5f);
    const std::vector<float> w = layer.weights(0);
    ASSERT_NE(w[0], 3.0f);
    net.step();
    EXPECT_FLOAT_EQ(layer.neurons()[0].output(), (w[0] + w[1]) / std::hypot(w[0], w[1]));

    // Saved and loaded with the network.
    std::stringstream data;
    net.save(data);
    auto loaded = network::load(data);
    EXPECT_TRUE(loaded->layerSpec(id).normalize);
    EXPECT_TRUE(loaded->layerAs<neuron_layer>(id).normalized());
    loaded->setInputs({1.0f, 1.0f});
    loaded->step();
    EXPECT_FLOAT_EQ(loaded->layerAs<neuron_layer>(id).neurons()[0].output(), (w[0] + w[1]) / std::hypot(w[0], w[1]));

    // On by default through the weighted-layer builders, off in a bare spec;
    // fixed-filter layers refuse it.
    network plain;
    EXPECT_TRUE(plain.layerAs<neuron_layer>(plain.addLayer("p", LayerSpec::Dense(1))).normalized());
    EXPECT_TRUE(LayerSpec::Conv2D(1, Window2D::square(3)).normalize);
    EXPECT_TRUE(LayerSpec::LocallyConnected2D(1, Window2D::square(3)).normalize);
    EXPECT_FALSE(LayerSpec{}.normalize);
    LayerSpec retina = LayerSpec::Retina(RetinaSpec{Shape{1, 4, 4}}, false, false);
    retina.normalize = true;
    EXPECT_THROW(plain.addLayer("r", retina), std::invalid_argument);
}

TEST(NormalizedSumTest, ConvolutionUsesItsKernelLength)
{
    network net;
    const auto image = net.addLayer("image", LayerSpec::Retina(RetinaSpec{Shape{1, 2, 2}}, false, false));
    LayerSpec c = LayerSpec::Conv2D(1, Window2D::square(2), false, false);
    c.normalize = true;
    const auto conv = net.addLayer("conv", c);
    net.addInputs(image, Shape{1, 2, 2});
    net.connect(image, conv);
    auto& layer = net.layerAs<conv2d>(conv);
    ASSERT_EQ(layer.size(), 1u);
    layer.setKernel(0, {1.0f, 1.0f, 1.0f, 1.0f});
    net.setInputs({1.0f, 1.0f, 1.0f, 1.0f});
    net.step();
    net.step();  // the retina's output reaches the convolution
    const float retinaSum = std::accumulate(net.getLayer(image).output().begin(), net.getLayer(image).output().end(), 0.0f);
    EXPECT_FLOAT_EQ(layer.neurons()[0].output(), retinaSum / 2.0f);
}

TEST(RestingThresholdTest, ScalesRestFloorAndEligibilityAndIsSaved)
{
    neuron n(false, true);
    n.setRestingThreshold(0.05f);
    EXPECT_FLOAT_EQ(n.threshold(), 0.05f);  // at rest: moves with it
    EXPECT_EQ(n.activate(0.04f), 0.0f);
    EXPECT_EQ(n.activate(0.06f), 0.06f);    // fires above the new rest (0.2 would block it)
    EXPECT_GE(n.threshold(), 0.1f);         // floor after firing: twice the rest
    EXPECT_LT(n.threshold(), 0.2f);
    EXPECT_TRUE(n.eligible());
    EXPECT_FLOAT_EQ(n.learningDelta(1.0f, 1.0f), n.learningGain() * (n.threshold() / 0.05f - 1.0f));

    network net;
    LayerSpec spec = LayerSpec::Dense(3, false, true);
    spec.restingThreshold = 0.05f;
    const auto id = net.addLayer("h", spec);
    net.addInputs(id, 2);
    for (const neuron& m : net.layerAs<neuron_layer>(id).neurons())
        EXPECT_FLOAT_EQ(m.threshold(), 0.05f);
    net.setInputs({1.0f, -1.0f});
    net.step();
    std::stringstream data;
    net.save(data);
    const std::string bytes = data.str();
    std::stringstream full(bytes), weights(bytes);
    auto restored = network::load(full, DeserializeMode::FullState);
    auto reset = network::load(weights, DeserializeMode::WeightsOnly);
    EXPECT_FLOAT_EQ(restored->layerSpec(id).restingThreshold, 0.05f);
    for (size_t i = 0; i < 3; ++i) {
        EXPECT_EQ(restored->layerAs<neuron_layer>(id).neurons()[i].threshold(),
                  net.layerAs<neuron_layer>(id).neurons()[i].threshold());  // full state keeps thresholds
        EXPECT_FLOAT_EQ(reset->layerAs<neuron_layer>(id).neurons()[i].threshold(), 0.05f);  // back at the new rest
        EXPECT_FLOAT_EQ(restored->layerAs<neuron_layer>(id).neurons()[i].restingThreshold(), 0.05f);
    }

    LayerSpec bad = LayerSpec::Dense(1);
    bad.restingThreshold = 0.0f;
    EXPECT_THROW(net.addLayer("bad", bad), std::invalid_argument);
}
