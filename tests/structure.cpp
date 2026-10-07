// Changing a running network: growing layers in width and depth, pruning
// neurons, freezing parts of it, and buses (network.hpp).
#include <gtest/gtest.h>
#include <cmath>
#include <sstream>
#include <vector>
#include "../core/layers/dense.hpp"
#include "../core/network.hpp"

using namespace exr;

namespace {

// in (sensors) -> h (E-R, recurrent) -> out (readout)
struct Small
{
    network net;
    network::LayerId h, out;

    Small()
    {
        reseed(11);
        h = net.addLayer("h", LayerSpec::Dense(6, false, true));
        out = net.addLayer("out", LayerSpec::Dense(3, false, false));
        net.addInputs(h, 4, "x");
        net.connect(h, h);
        net.connect(h, out);
        net.addOutput(out);
    }
    std::vector<float> run(network& n, int ticks, int from = 0)
    {
        std::vector<float> all;
        for (int t = from; t < from + ticks; ++t) {
            n.setInputs("x", std::vector<float>{std::sin(0.4f * t), std::cos(0.9f * t), 0.5f, -0.3f});
            n.step();
            const std::vector<float> y = n.outputs();
            all.insert(all.end(), y.begin(), y.end());
        }
        return all;
    }
};

std::unique_ptr<network> roundTrip(const network& net)
{
    std::stringstream data;
    net.save(data);
    return network::load(data);
}

} // namespace

TEST(GrowthTest, NewNeuronsWithZeroOutgoingWeightsChangeNothing)
{
    Small a, b;
    a.run(a.net, 10);
    b.run(b.net, 10);
    b.net.growLayer(b.h, 4);  // outgoing weights 0 by default
    EXPECT_EQ(b.net.getLayer(b.h).size(), 10u);
    EXPECT_EQ(b.net.layerAs<dense>(b.out).inputCount(0), 10u);
    // The old neurons read the new ones with weight 0, and so does the readout.
    const auto& h = b.net.layerAs<dense>(b.h);
    for (size_t i = 0; i < 6; ++i) {
        const std::vector<float> w = h.weights(i);
        ASSERT_EQ(w.size(), 4u + 10u);
        for (size_t j = 10; j < 14; ++j)
            EXPECT_EQ(w[j], 0.0f);
    }
    // The new neurons read the same inputs (sensors and h, including themselves).
    EXPECT_EQ(h.inputCount(6), 14u);
    EXPECT_EQ(a.run(a.net, 20, 10), b.run(b.net, 20, 10));

    // Saved and loaded: the same structure and behaviour.
    auto loaded = roundTrip(b.net);
    EXPECT_EQ(loaded->getLayer(b.h).size(), 10u);
    EXPECT_EQ(b.run(b.net, 5, 30), b.run(*loaded, 5, 30));
}

TEST(GrowthTest, FreezeExistingLeavesOnlyNewNeuronsLearning)
{
    Small s;
    s.run(s.net, 5);
    s.net.growLayer(s.h, 2, WeightInit::Random, true);
    const auto& h = s.net.layerAs<dense>(s.h);
    EXPECT_EQ(h.frozenNeuronCount(), 6u);
    EXPECT_FALSE(h.neuronFrozen(6));
    std::vector<std::vector<float>> before;
    for (size_t i = 0; i < h.size(); ++i)
        before.push_back(h.weights(i));
    for (int k = 0; k < 20; ++k) {
        s.run(s.net, 1, k);
        s.net.applyReward(1.0f, 0.05f);
    }
    for (size_t i = 0; i < 6; ++i)
        EXPECT_EQ(h.weights(i), before[i]) << "old neuron " << i;
    // Random outgoing weights: the readout reads the new neurons at once.
    EXPECT_NE(s.net.layerAs<dense>(s.out).weights(0)[6], 0.0f);
}

TEST(GrowthTest, DepthWithAZeroConnection)
{
    // A new layer reads h and joins the readout with zero weights.
    Small a, b;
    a.run(a.net, 10);
    b.run(b.net, 10);
    const auto deep = b.net.addLayer("deep", LayerSpec::Dense(5, false, true));
    b.net.connect(b.h, deep);
    b.net.connect(deep, b.out, WeightInit::Zero);
    b.net.freezeInputs(b.out, b.h);  // the readout keeps what it learned from h
    EXPECT_EQ(b.net.layerAs<dense>(b.out).frozenInputCount(), 6u);  // one column per h neuron
    EXPECT_EQ(a.run(a.net, 10, 10), b.run(b.net, 10, 10));
    auto loaded = roundTrip(b.net);
    EXPECT_EQ(loaded->layerAs<dense>(b.out).frozenInputCount(), 6u);  // one column per h neuron
    EXPECT_EQ(b.run(b.net, 5, 20), b.run(*loaded, 5, 20));
}

TEST(PruningTest, RemovesNeuronsAndTheirInputsEverywhere)
{
    Small s;
    s.run(s.net, 5);
    auto& h = s.net.layerAs<dense>(s.h);
    auto& out = s.net.layerAs<dense>(s.out);
    const std::vector<float> h0 = h.weights(0), out0 = out.weights(0);
    s.net.pruneNeurons(s.h, {4, 1});  // any order
    EXPECT_EQ(h.size(), 4u);
    // Readers drop the columns of neurons 1 and 4; the rest keep their weights.
    const auto keep = [](const std::vector<float>& w, std::vector<size_t> drop) {
        std::vector<float> kept;
        for (size_t j = 0; j < w.size(); ++j)
            if (std::find(drop.begin(), drop.end(), j) == drop.end())
                kept.push_back(w[j]);
        return kept;
    };
    EXPECT_EQ(out.weights(0), keep(out0, {1, 4}));
    EXPECT_EQ(h.weights(0), keep(h0, {4 + 1, 4 + 4}));  // 4 sensors, then h itself
    s.run(s.net, 5, 5);

    auto loaded = roundTrip(s.net);
    EXPECT_EQ(loaded->getLayer(s.h).size(), 4u);
    EXPECT_EQ(s.run(s.net, 5, 10), s.run(*loaded, 5, 10));

    EXPECT_THROW(s.net.pruneNeurons(s.h, {9}), std::out_of_range);
    // A State layer cannot lose inputs: refused before anything changes.
    const auto state = s.net.addLayer("state", LayerSpec::State());
    s.net.connect(s.h, state);
    EXPECT_THROW(s.net.pruneNeurons(s.h, {0}), std::logic_error);
    EXPECT_EQ(h.size(), 4u);
}

TEST(BusTest, ItsOwnNeuronTypeFrozenByDefault)
{
    network net;
    const auto a = net.addLayer("a", LayerSpec::Dense(4, true, true));
    const auto bus = net.addBus("bus", LayerSpec::Perceptron(5, 0.1f));
    const auto r1 = net.addLayer("r1", LayerSpec::Dense(3, false, true));
    const auto r2 = net.addLayer("r2", LayerSpec::Dense(2, false, false));
    net.addInputs(a, 3, "x");
    net.connectInputs("x", bus);
    net.writeBus(a, bus);
    net.subscribe(r1, bus);
    net.subscribe(r2, bus);
    EXPECT_THROW(net.subscribe(r1, a), std::invalid_argument);  // a is not a bus
    EXPECT_TRUE(net.isBus(bus));
    EXPECT_FALSE(net.isBus(a));
    EXPECT_EQ(net.buses(), std::vector<network::LayerId>{bus});
    EXPECT_EQ(net.busWriters(bus), std::vector<network::LayerId>{a});
    EXPECT_EQ(net.busReaders(bus), (std::vector<network::LayerId>{r1, r2}));
    EXPECT_TRUE(net.isFrozen(bus));
    EXPECT_THROW(net.addBus("conv", LayerSpec::Conv2D(2, Window2D::square(3))), std::invalid_argument);

    auto& b = net.layerAs<dense>(bus);
    const std::vector<float> w = b.weights(0);
    for (int t = 0; t < 10; ++t) {
        net.setInputs("x", std::vector<float>{std::sin(1.0f * t), 0.5f, -0.5f});
        net.step();
        for (float y : b.output())
            EXPECT_TRUE(y == 0.0f || y == 1.0f);  // perceptrons
        net.applyReward(1.0f, 0.1f);
    }
    EXPECT_EQ(b.weights(0), w);  // frozen

    auto loaded = roundTrip(net);
    EXPECT_TRUE(loaded->isBus(bus));
    EXPECT_TRUE(loaded->layerSpec(bus).binary);
    EXPECT_EQ(loaded->busReaders(bus), (std::vector<network::LayerId>{r1, r2}));
    std::ostringstream text;
    net.describe(text);
    EXPECT_NE(text.str().find("bus bus: writers 'x' a; readers r1 r2"), std::string::npos) << text.str();
}
