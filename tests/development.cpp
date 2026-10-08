// Structural development (doc/development.md): the protected minimum size of
// a population and which of its neurons were grown.
#include <gtest/gtest.h>
#include <cmath>
#include <sstream>
#include <vector>
#include "../core/layers/dense.hpp"
#include "../core/network.hpp"

using namespace exr;

namespace {

// in (sensors) -> h (E-R) -> out (readout)
struct Population
{
    network net;
    network::LayerId h, out;

    explicit Population(size_t size = 4)
    {
        reseed(5);
        h = net.addLayer("h", LayerSpec::Dense(size, false, true));
        out = net.addLayer("out", LayerSpec::Dense(3, false, false));
        net.addInputs(h, 4, "x");
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

TEST(MinimumSizeTest, PruningStopsAtTheProtectedCore)
{
    Population p(4);
    p.net.setMinimumSize(p.h, 4);
    EXPECT_EQ(p.net.minimumSize(p.h), 4u);
    // At the minimum: nothing can go, and nothing changes.
    EXPECT_THROW(p.net.pruneNeurons(p.h, {3}), std::invalid_argument);
    EXPECT_EQ(p.net.getLayer(p.h).size(), 4u);

    p.net.growLayer(p.h, 4);
    p.net.pruneNeurons(p.h, {6, 7});  // above the minimum: allowed
    EXPECT_EQ(p.net.getLayer(p.h).size(), 6u);
    // 6 - 3 = 3 < 4: refused as a whole, not partly done.
    EXPECT_THROW(p.net.pruneNeurons(p.h, {0, 4, 5}), std::invalid_argument);
    EXPECT_EQ(p.net.getLayer(p.h).size(), 6u);
    p.net.pruneNeurons(p.h, {4, 5});
    EXPECT_EQ(p.net.getLayer(p.h).size(), 4u);

    EXPECT_THROW(p.net.setMinimumSize(p.h, 5), std::invalid_argument);  // more than it has
    EXPECT_THROW(p.net.setMinimumSize(p.out + 1, 1), std::out_of_range);
    LayerSpec bad = LayerSpec::Dense(2);
    bad.minimumSize = 3;
    EXPECT_THROW(p.net.addLayer("bad", bad), std::invalid_argument);
}

TEST(MinimumSizeTest, WithoutAFloorPruningIsUnchanged)
{
    Population a(4), b(4);
    EXPECT_EQ(a.net.minimumSize(a.h), 0u);
    a.net.pruneNeurons(a.h, {0, 1, 2});
    EXPECT_EQ(a.net.getLayer(a.h).size(), 1u);
    // Setting and clearing a floor changes nothing about how the network runs.
    b.net.setMinimumSize(b.h, 2);
    b.net.setMinimumSize(b.h, 0);
    Population c(4);
    EXPECT_EQ(b.run(b.net, 20), c.run(c.net, 20));
}

TEST(GrownNeuronsTest, BaseAndGrownAreToldApart)
{
    Population p(4);
    auto& h = p.net.layerAs<dense>(p.h);
    EXPECT_TRUE(p.net.grownNeurons(p.h).empty());
    EXPECT_EQ(h.grownNeuronCount(), 0u);

    p.net.growLayer(p.h, 2);  // A, A
    p.net.growLayer(p.h, 1);  // B
    EXPECT_EQ(p.net.grownNeurons(p.h), (std::vector<size_t>{4, 5, 6}));
    EXPECT_EQ(h.growthOrder(0), 0u);
    EXPECT_EQ(h.growthOrder(4), 1u);
    EXPECT_EQ(h.growthOrder(5), 2u);
    EXPECT_EQ(h.growthOrder(6), 3u);

    // Feedback neurons are built with the layer, not grown.
    p.net.addFeedback(p.out, p.h, 2);
    EXPECT_FALSE(h.neuronGrown(7));
    EXPECT_FALSE(h.neuronGrown(8));

    // Pruning keeps the others' order; later growth keeps counting.
    p.net.pruneNeurons(p.h, {5});
    EXPECT_EQ(p.net.grownNeurons(p.h), (std::vector<size_t>{4, 5}));
    EXPECT_EQ(h.growthOrder(5), 3u);
    p.net.growLayer(p.h, 1);
    EXPECT_EQ(h.growthOrder(h.size() - 1), 4u);

    // Not a Dense layer of neurons: nothing grown.
    const auto state = p.net.addLayer("state", LayerSpec::State());
    p.net.connect(p.out, state);
    EXPECT_TRUE(p.net.grownNeurons(state).empty());
}

TEST(GrownNeuronsTest, SavedNetworkKeepsMinimumGrowthAndTags)
{
    Population p(4);
    p.run(p.net, 5);
    p.net.growLayer(p.h, 3);
    p.net.pruneNeurons(p.h, {0});  // a base neuron, while there was no floor yet
    p.net.setMinimumSize(p.h, 6);  // now the whole layer is protected
    LayerSpec deep = LayerSpec::Dense(2, false, true);
    deep.grown = true;
    const auto d = p.net.addLayer("deep", deep);
    p.net.connect(p.h, d);
    p.net.connect(d, p.out, WeightInit::Zero);
    p.run(p.net, 5, 5);

    auto loaded = roundTrip(p.net);
    EXPECT_EQ(loaded->minimumSize(p.h), 6u);
    EXPECT_EQ(loaded->getLayer(p.h).size(), 6u);
    EXPECT_EQ(loaded->grownNeurons(p.h), p.net.grownNeurons(p.h));
    const auto& a = p.net.layerAs<dense>(p.h);
    const auto& b = loaded->layerAs<dense>(p.h);
    for (size_t i = 0; i < a.size(); ++i)
        EXPECT_EQ(a.growthOrder(i), b.growthOrder(i));
    EXPECT_TRUE(loaded->layerSpec(d).grown);
    EXPECT_FALSE(loaded->layerSpec(p.h).grown);
    EXPECT_EQ(p.run(p.net, 10, 10), p.run(*loaded, 10, 10));
    EXPECT_THROW(loaded->pruneNeurons(p.h, {0}), std::invalid_argument);

    std::ostringstream text;
    loaded->describe(text);
    EXPECT_NE(text.str().find("growth h: 6 neurons, 3 grown, minimum 6"), std::string::npos) << text.str();
    EXPECT_NE(text.str().find("growth deep: 2 neurons, 0 grown, minimum 0, a grown layer"), std::string::npos);
}

TEST(MinimumSizeTest, StartsFromOneNeuron)
{
    // Like an organism from one cell: a single neuron is the protected core.
    Population p(1);
    p.net.setMinimumSize(p.h, 1);
    p.run(p.net, 5);
    EXPECT_THROW(p.net.pruneNeurons(p.h, {0}), std::invalid_argument);
    p.net.growLayer(p.h, 1, WeightInit::Zero, true);
    p.net.growLayer(p.h, 2, WeightInit::Zero, true);
    EXPECT_EQ(p.net.getLayer(p.h).size(), 4u);
    EXPECT_EQ(p.net.grownNeurons(p.h), (std::vector<size_t>{1, 2, 3}));
    p.run(p.net, 5, 5);
    p.net.pruneNeurons(p.h, {1, 2, 3});
    EXPECT_EQ(p.net.getLayer(p.h).size(), 1u);
}
