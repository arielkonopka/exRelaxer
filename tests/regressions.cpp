// Regression tests, one per bug found in the code analysis of 2026-09-26.
// Each test states the bug it guards against and how it used to fail.
//
// DISABLED_ tests describe bugs that are still open: they fail today and are
// enabled together with the fix. Run them anyway with
//   ./exrelaxer_tests --gtest_filter='RegressionTest.*' --gtest_also_run_disabled_tests
#include <gtest/gtest.h>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>
#include "../core/network.hpp"
#include "../core/layers/dense.hpp"

using namespace exr;

namespace {

dense& layerOf(network& net, network::LayerId id) { return dynamic_cast<dense&>(net.getLayer(id)); }

void setAllWeights(dense& layer, const std::vector<float>& weights)
{
    for (size_t i = 0; i < layer.size(); ++i)
        layer.setWeights(i, weights);
}

// Plain weighted sums, so outputs can be checked exactly.
LayerSpec linear(size_t size) { return {LayerType::Dense, size, /*hasHabituation*/ false, /*hasER*/ false}; }

// Every neuron is in at most one group's range, and groupOf() agrees with
// the ranges.
void expectOneGroupPerNeuron(const dense& layer, const std::string& name)
{
    std::vector<size_t> memberships(layer.size(), 0);
    for (size_t g = 0; g < layer.groupCount(); ++g) {
        const dense::NeuronRange range = layer.groupNeurons(g);
        ASSERT_LE(range.first + range.count, layer.size()) << name;
        for (size_t idx = range.first; idx < range.first + range.count; ++idx) {
            ++memberships[idx];
            EXPECT_EQ(layer.groupOf(idx), g) << name << " neuron " << idx;
        }
    }
    for (size_t idx = 0; idx < layer.size(); ++idx) {
        EXPECT_LE(memberships[idx], 1u) << name << " neuron " << idx << " is in several groups";
        if (memberships[idx] == 0) {
            EXPECT_EQ(layer.groupOf(idx), dense::no_group) << name << " neuron " << idx;
        }
    }
}

void expectOneGroupPerNeuron(network& net)
{
    for (network::LayerId id = 0; id < net.layerCount(); ++id)
        expectOneGroupPerNeuron(layerOf(net, id), net.layerName(id));
}

} // namespace

// =============================================================================
// Bug 1: a neuron in two wiring groups
// =============================================================================

// The invariant itself: whatever the wiring calls and their order, no neuron
// ends up in two groups, and every neuron that reads something has a group.
TEST(RegressionTest, EveryWiringKeepsOneGroupPerNeuron)
{
    network net;
    const auto in = net.addLayer("in", {LayerType::Dense, 4});
    const auto a = net.addLayer("a", {LayerType::Dense, 5});
    const auto b = net.addLayer("b", {LayerType::Dense, 3});
    const auto c = net.addLayer("c", {LayerType::Dense, 2});
    const auto grown = net.addLayer("grown", {LayerType::Dense, 0});
    net.addInputs(in, 2);
    net.connect(in, a);
    net.connect(in, b);
    net.connect(a, c);        // c reads a and b: one group
    net.connect(b, c);
    net.addInputs(b, 1);      // sensor mixed into b's group (i -> a -> b + j)
    net.addFeedback(c, a, 2); // a grows: c's group extended
    net.connect(b, a);        // after feedback: both of a's groups read b
    net.connect(c, c);        // self-connection
    net.addFeedback(c, c, 3); // self-feedback
    net.addFeedback(a, grown, 2);
    net.addFeedback(b, grown, 2);
    net.addInputs(grown, 1);
    net.addFeedback(grown, in, 1);  // growth of the first layer cascades

    expectOneGroupPerNeuron(net);
    for (network::LayerId id = 0; id < net.layerCount(); ++id) {
        const dense& layer = layerOf(net, id);
        for (size_t idx = 0; idx < layer.size(); ++idx)
            EXPECT_NE(layer.groupOf(idx), dense::no_group) << net.layerName(id) << " neuron " << idx;
    }
    net.setUpdateOrder({in, b, a, c, grown});
    net.setInputs({0.5f, -0.5f, 0.25f, 1.0f});
    EXPECT_NO_THROW(net.step());
}

// A layer's neurons wired only later stay ungrouped until then, and join
// only them into a new group.
TEST(RegressionTest, OnlyUngroupedNeuronsFormANewGroup)
{
    dense source(2), x(3);
    source.addFeedback(x, 2);  // x: 3 unwired + 2 feedback neurons
    for (size_t idx = 0; idx < 3; ++idx)
        EXPECT_EQ(x.groupOf(idx), dense::no_group);
    ASSERT_EQ(x.groupCount(), 1u);

    dense other(4);
    x.join(other);             // feedback group extended, the 3 get a new group
    ASSERT_EQ(x.groupCount(), 2u);
    EXPECT_EQ(x.groupNeurons(1), (dense::NeuronRange{0, 3}));
    expectOneGroupPerNeuron(x, "x");
}

// connect() from a second source used to create a second group over the same
// neurons and re-initialize their single weight vector to the second
// source's size: x's neurons had 4 weights while group a fed them 16 inputs,
// and step() read past the end of the weights (container-overflow under
// ASan). Now both sources share one group: 16 + 4 weights, one sum.
TEST(RegressionTest, LayerConnectedFromTwoSourcesSumsBothWithOneWeightPerInput)
{
    network net;
    const auto a = net.addLayer("a", linear(16));
    const auto b = net.addLayer("b", linear(4));
    const auto x = net.addLayer("x", linear(3));
    net.addInputs(a, 1);
    net.addInputs(b, 1);
    net.connect(a, x);
    net.connect(b, x);

    setAllWeights(layerOf(net, a), {1.0f});
    setAllWeights(layerOf(net, b), {1.0f});
    for (size_t i = 0; i < layerOf(net, x).size(); ++i)
        ASSERT_EQ(layerOf(net, x).inputCount(i), 20u);
    setAllWeights(layerOf(net, x), std::vector<float>(20, 1.0f));

    net.setInputs({0.25f, 0.5f});
    net.step();
    for (float out : layerOf(net, x).output())
        EXPECT_FLOAT_EQ(out, 16 * 0.25f + 4 * 0.5f);
}

// connect(x, x) after addInputs(x, ...) used to re-initialize x's weights to
// x's own size, overwriting the learned/assigned sensor weights. Now the
// self-connection is appended after the sensor weights.
TEST(RegressionTest, ConnectingAfterSensorsKeepsTheSensorWeights)
{
    network net;
    const auto x = net.addLayer("x", linear(3));
    net.addInputs(x, 2);
    setAllWeights(layerOf(net, x), {0.5f, -0.5f});

    net.connect(x, x);
    for (size_t i = 0; i < layerOf(net, x).size(); ++i) {
        const std::vector<float> w = layerOf(net, x).weights(i);
        ASSERT_EQ(w.size(), 2u + 3u);
        EXPECT_EQ(w[0], 0.5f);
        EXPECT_EQ(w[1], -0.5f);
    }
}

// With two groups, applyReward() updated a neuron once per group, so it
// learned twice per reward. One group per neuron: exactly one update.
TEST(RegressionTest, NeuronReadingTwoSourcesLearnsOncePerReward)
{
    network net;
    const auto a = net.addLayer("a", linear(1));
    const auto b = net.addLayer("b", linear(1));
    const auto x = net.addLayer("x", linear(1));
    net.addInputs(a, 1);
    net.addInputs(b, 1);
    net.connect(a, x);
    net.connect(b, x);
    layerOf(net, a).setWeights(0, {1.0f});
    layerOf(net, b).setWeights(0, {1.0f});
    layerOf(net, x).setWeights(0, {1.0f, 1.0f});

    net.setInputs({0.5f, 0.5f});
    net.step();  // x outputs 1, so it is eligible
    constexpr float LEARNING_RATE = 0.1f;
    net.getLayer(x).applyReward(1.0f, LEARNING_RATE);

    const float expected = 1.0f + LEARNING_RATE * default_learning_gain;
    for (float w : layerOf(net, x).weights(0))
        EXPECT_FLOAT_EQ(w, expected);
}

// =============================================================================
// Bug 2: sensor inputs extended when their own layer grows
// =============================================================================

// Sensor groups used to name the layer itself as their source, so once the
// layer listened to itself, its growth was appended to the sensor group too:
// 11 weights instead of 2 sensors + 3 + 4 grown = 9.
TEST(RegressionTest, SelfConnectedLayerGrowthIsAppendedOnce)
{
    network net;
    const auto x = net.addLayer("x", {LayerType::Dense, 3});
    const auto y = net.addLayer("y", {LayerType::Dense, 2});
    net.addInputs(x, 2);
    net.connect(x, x);
    net.addFeedback(y, x, 4);

    const dense& layer = layerOf(net, x);
    ASSERT_EQ(layer.size(), 7u);
    for (size_t k = 0; k < 3; ++k)
        EXPECT_EQ(layer.inputCount(k), 2u + 7u);  // sensors + all of x
    for (size_t k = 3; k < 7; ++k)
        EXPECT_EQ(layer.inputCount(k), 2u);       // y only
    EXPECT_NO_THROW(net.step());
}

// Same bug without connect(): self-feedback made x a listener of itself, and
// the sensor-only neurons then read x's new neurons as well (4 weights).
TEST(RegressionTest, SelfFeedbackDoesNotExtendSensorOnlyNeurons)
{
    network net;
    const auto x = net.addLayer("x", {LayerType::Dense, 3});
    net.addInputs(x, 2);
    net.addFeedback(x, x, 2);

    const dense& layer = layerOf(net, x);
    ASSERT_EQ(layer.size(), 5u);
    for (size_t k = 0; k < 3; ++k)
        EXPECT_EQ(layer.inputCount(k), 2u);       // sensors only
    for (size_t k = 3; k < 5; ++k)
        EXPECT_EQ(layer.inputCount(k), 3u + 2u);  // all of x, themselves included
    EXPECT_NO_THROW(net.step());
}

// =============================================================================
// Bug: attachInputs reached only the first group
// =============================================================================

// Sensors used to be appended to groups.front() only. When that group came
// from feedback, the layer's original neurons were never wired and stayed at
// 0. Now every neuron the layer has reads the sensors.
TEST(RegressionTest, SensorsReachNeuronsOutsideTheFirstGroup)
{
    network net;
    const auto x = net.addLayer("x", linear(3));
    const auto y = net.addLayer("y", linear(2));
    net.addFeedback(y, x, 2);  // x's first group: the 2 feedback neurons
    net.addInputs(x, 1);

    dense& layer = layerOf(net, x);
    ASSERT_EQ(layer.size(), 5u);
    for (size_t k = 0; k < 3; ++k)
        ASSERT_EQ(layer.inputCount(k), 1u);       // the sensor
    for (size_t k = 3; k < 5; ++k)
        EXPECT_EQ(layer.inputCount(k), 2u + 1u);  // y, then the sensor

    for (size_t k = 0; k < 3; ++k) layer.setWeights(k, {1.0f});
    net.setInputs({0.5f});
    net.step();
    for (size_t k = 0; k < 3; ++k)
        EXPECT_FLOAT_EQ(layer.output()[k], 0.5f);
}

// =============================================================================
// Bug 3: deserialization trusting the data
// =============================================================================

// Weights that do not match the layer's wiring used to load silently, and
// the next step() read past the end of the weights.
TEST(RegressionTest, DeserializeRejectsWeightCountThatDoesNotMatchWiring)
{
    dense source3(3), saved(2);
    saved.join(source3);
    std::stringstream data;
    saved.serialize(data);

    dense source2(2), target(2);
    target.join(source2);
    EXPECT_THROW(target.deserialize(data), std::runtime_error);
}

// A corrupt per-neuron weight count used to go straight into
// weights.resize(), throwing std::length_error / std::bad_alloc (or
// allocating gigabytes) instead of the std::runtime_error network::load
// promises for bad data.
TEST(RegressionTest, DeserializeRejectsImplausibleWeightCount)
{
    dense source(2), saved(1);
    saved.join(source);
    std::stringstream data;
    saved.serialize(data);

    // Layer: hasHabituation, hasER, neuron count; neuron: hasHabituation,
    // hasER, alpha, then the weight count.
    std::string bytes = data.str();
    const size_t offset = 2 * sizeof(bool) + sizeof(size_t) + 2 * sizeof(bool) + sizeof(float);
    const size_t corrupt = size_t{1} << 62;
    std::memcpy(bytes.data() + offset, &corrupt, sizeof(corrupt));

    dense source2(2), target(1);
    target.join(source2);
    std::stringstream corrupted(bytes);
    EXPECT_THROW(target.deserialize(corrupted), std::runtime_error);
}

// Data with another neuron count used to clear a wired layer's neurons and
// outputs but keep its groups (indices past the end) and the output pointers
// downstream layers read. Now refused, and the layer is left unchanged.
TEST(RegressionTest, DeserializeRejectsOtherNeuronCountOnWiredLayer)
{
    dense source(2), saved(2);
    saved.join(source);
    std::stringstream data;
    saved.serialize(data);

    dense source2(2), target(3, false, false);
    target.join(source2);
    EXPECT_THROW(target.deserialize(data), std::runtime_error);
    EXPECT_EQ(target.size(), 3u);
    expectOneGroupPerNeuron(target, "target");
    std::stringstream probe;
    target.serialize(probe);  // flags untouched: still no habituation, no E-R
    const std::string bytes = probe.str();
    EXPECT_EQ(bytes[0], 0);
    EXPECT_EQ(bytes[1], 0);

    // A layer read by others is wired too, even without groups of its own.
    dense upstream(2), reader(1);
    reader.join(upstream);
    std::stringstream three;
    dense(3).serialize(three);
    EXPECT_THROW(upstream.deserialize(three), std::runtime_error);
    EXPECT_EQ(upstream.size(), 2u);
}

// =============================================================================
// Smaller open issues
// =============================================================================

// Open: network::load documents std::runtime_error for data it cannot load,
// but a layer type missing from the factory throws std::invalid_argument.
TEST(RegressionTest, DISABLED_LoadReportsUnregisteredLayerTypeAsRuntimeError)
{
    const auto custom = static_cast<LayerType>(200);  // a type only the saving program registers
    layer_factory withCustom;
    withCustom.registerType(custom, [](const LayerSpec& spec) -> std::unique_ptr<layer> {
        return std::make_unique<dense>(spec.size);
    });
    network saved(withCustom);
    saved.addLayer("custom", {custom, 2});
    std::stringstream data;
    saved.save(data);

    layer_factory builtIn;
    EXPECT_THROW(network::load(data, DeserializeMode::FullState, builtIn), std::runtime_error);
}

// Open: loading replays the construction, which draws from the shared random
// streams, so a network built after a load gets different weights than the
// same network built without it, even after exr::reseed().
TEST(RegressionTest, DISABLED_LoadingDoesNotChangeNetworksBuiltAfterIt)
{
    auto build = [] {
        auto net = std::make_unique<network>();
        const auto in = net->addLayer("in", {LayerType::Dense, 4});
        const auto out = net->addLayer("out", {LayerType::Dense, 3});
        net->addInputs(in, 2);
        net->connect(in, out);
        return net;
    };
    auto weightsOf = [](network& net) {
        std::vector<float> all;
        for (network::LayerId id = 0; id < net.layerCount(); ++id)
            for (size_t i = 0; i < layerOf(net, id).size(); ++i) {
                const std::vector<float> w = layerOf(net, id).weights(i);
                all.insert(all.end(), w.begin(), w.end());
            }
        return all;
    };

    std::stringstream data;
    build()->save(data);

    reseed(1);
    const auto fresh = weightsOf(*build());

    reseed(1);
    network::load(data);
    const auto afterLoad = weightsOf(*build());

    EXPECT_EQ(fresh, afterLoad);
}
