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

namespace {

// x (one sensor) -> h (E-R, plain weighted sums, weights 1) -> out
struct Watched
{
    network net;
    network::LayerId h, out;

    explicit Watched(size_t size = 4, float recovery = recovery_factor)
    {
        reseed(9);
        LayerSpec spec = LayerSpec::Dense(size, false, true);
        spec.normalize = false;
        h = net.addLayer("h", spec);
        out = net.addLayer("out", LayerSpec::Dense(2, false, false));
        net.addInputs(h, 1, "x");
        net.connect(h, out);
        net.addOutput(out);
        auto& layer = net.layerAs<dense>(h);
        for (size_t i = 0; i < size; ++i) {
            layer.setWeights(i, {1.0f});
            layer.neurons()[i].setRecovery(recovery);
        }
    }
    dense& layer() { return net.layerAs<dense>(h); }
    void tick(activity_monitor& m, float x)
    {
        net.setInputs("x", std::vector<float>{x});
        net.step();
        m.observe(layer());
    }
};

bool has(const std::vector<network::PruneCandidate>& c, size_t index, std::uint32_t reason)
{
    for (const auto& k : c)
        if (k.index == index)
            return (k.reasons & reason) != 0;
    return false;
}

} // namespace

TEST(PruneCandidatesTest, InvalidZeroDisconnectedAndUnread)
{
    Watched w(4);
    w.layer().setWeights(1, {std::nanf("")});
    w.layer().setWeights(2, {0.0f});
    auto c = w.net.pruneCandidates(w.h);
    EXPECT_TRUE(has(c, 1, network::Invalid));
    EXPECT_TRUE(has(c, 2, network::ZeroIncoming));
    EXPECT_FALSE(has(c, 0, network::Invalid | network::ZeroIncoming | network::Unread));
    EXPECT_FALSE(has(c, 3, network::Invalid | network::ZeroIncoming | network::Unread));

    // Grown with zero outgoing weights: nothing reads them yet.
    w.net.growLayer(w.h, 2);
    c = w.net.pruneCandidates(w.h);
    EXPECT_TRUE(has(c, 4, network::Unread));
    EXPECT_TRUE(has(c, 5, network::Unread));
    EXPECT_FALSE(has(c, 0, network::Unread));
    // The readout learns to read neuron 4: no longer a candidate for that.
    auto& out = w.net.layerAs<dense>(w.out);
    std::vector<float> row = out.weights(0);
    row[4] = 0.5f;
    out.setWeights(0, row);
    EXPECT_FALSE(has(w.net.pruneCandidates(w.h), 4, network::Unread));

    // A layer that reads nothing; an output layer is always read.
    const auto lonely = w.net.addLayer("lonely", LayerSpec::Dense(2, false, true));
    w.net.addOutput(lonely);
    c = w.net.pruneCandidates(lonely);
    ASSERT_EQ(c.size(), 2u);
    EXPECT_EQ(c[0].reasons, network::Disconnected);
    // Nothing is removed by asking.
    EXPECT_EQ(w.layer().size(), 6u);
}

TEST(PruneCandidatesTest, PersistentInactivityButNotFatigue)
{
    // Neuron 0 gets no input (weight 0); neuron 1 is fatigued by a strong
    // pulse and recovers slowly; neurons 2, 3 also fatigued.
    Watched w(4, 0.999f);
    w.layer().setWeights(0, {0.0f});
    activity_monitor m;
    for (int t = 0; t < 3; ++t)
        w.tick(m, 5.0f);
    for (int t = 0; t < 300; ++t)
        w.tick(m, 0.5f);
    EXPECT_EQ(m.state(0), TickState::NoInput);
    EXPECT_EQ(m.state(1), TickState::Fatigued);
    EXPECT_GE(m.inactiveTicks(0), 300u);
    EXPECT_EQ(m.inactiveTicks(1), 0u);  // silent for 300 ticks, but only fatigued
    const auto c = w.net.pruneCandidates(w.h, &m, 200);
    EXPECT_TRUE(has(c, 0, network::Inactive));
    EXPECT_FALSE(has(c, 1, network::Inactive));
}

TEST(LifoPruningTest, NewestGrownNeuronGoesFirst)
{
    Watched w(4);
    w.net.setMinimumSize(w.h, 4);
    w.net.growLayer(w.h, 1);  // A: index 4
    w.net.growLayer(w.h, 1);  // B: index 5
    w.net.growLayer(w.h, 1);  // C: index 6
    auto& h = w.layer();
    const std::vector<float> a = h.weights(4), b = h.weights(5);
    w.layer().setWeights(4, {0.25f});
    w.layer().setWeights(5, {0.5f});

    EXPECT_EQ(w.net.pruneNewest(w.h), (std::vector<size_t>{6}));  // C
    EXPECT_EQ(h.size(), 6u);
    EXPECT_EQ(h.weights(4), std::vector<float>{0.25f});  // A remains
    EXPECT_EQ(h.weights(5), std::vector<float>{0.5f});   // B remains
    EXPECT_EQ(w.net.pruneNewest(w.h), (std::vector<size_t>{5}));  // B
    EXPECT_EQ(h.weights(4), std::vector<float>{0.25f});
    // Asking for more than there is: only grown neurons, never below the minimum.
    EXPECT_EQ(w.net.pruneNewest(w.h, 10), (std::vector<size_t>{4}));
    EXPECT_TRUE(w.net.pruneNewest(w.h, 10).empty());
    EXPECT_EQ(h.size(), 4u);

    // A floor below the base size.
    Watched v(4);
    v.net.setMinimumSize(v.net.findLayer("h"), 3);
    v.net.growLayer(v.h, 3);
    EXPECT_EQ(v.net.pruneNewest(v.h, 10), (std::vector<size_t>{6, 5, 4}));
    EXPECT_EQ(v.layer().size(), 4u);  // the floor (3) would allow one more, but LIFO never takes base neurons
}

TEST(SaturationTest, TemporaryFatigueIsNotSaturation)
{
    Watched w(4);  // default recovery 0.9: thresholds come back down within ~20 ticks
    activity_monitor m({.window = 100});
    for (int t = 0; t < 3; ++t)
        w.tick(m, 5.0f);
    w.tick(m, 0.5f);
    EXPECT_TRUE(m.saturatedTick());  // a silent, fatigued tick...
    EXPECT_FALSE(m.saturated());     // ...is not saturation
    for (int t = 0; t < 300; ++t)
        w.tick(m, 0.5f);
    EXPECT_FALSE(m.saturated());
    EXPECT_LT(m.saturation(), 0.9f);
}

TEST(SaturationTest, PersistentSilenceWithRaisedThresholdsAndInput)
{
    Watched w(4, 0.999f);  // slow recovery: thresholds stay above the input
    activity_monitor m({.window = 100});
    for (int t = 0; t < 3; ++t)
        w.tick(m, 5.0f);
    for (int t = 0; t < 50; ++t)
        w.tick(m, 0.5f);
    EXPECT_FALSE(m.saturated());  // window not full yet
    for (int t = 0; t < 100; ++t)
        w.tick(m, 0.5f);
    EXPECT_TRUE(m.saturated());
    EXPECT_FLOAT_EQ(m.saturation(), 1.0f);

    // Without input the population is silent too, but that is not saturation.
    activity_monitor quiet({.window = 100});
    Watched v(4, 0.999f);
    for (int t = 0; t < 3; ++t)
        v.tick(quiet, 5.0f);
    for (int t = 0; t < 200; ++t)
        v.tick(quiet, 0.0f);
    EXPECT_FALSE(quiet.saturated());

    // Growth resets the monitor: new neurons are judged from their first tick.
    w.net.growLayer(w.h, 2);
    w.layer().setWeights(4, {1.0f});
    w.tick(m, 0.5f);
    EXPECT_EQ(m.ticks(), 1u);
    EXPECT_FALSE(m.saturated());
    EXPECT_EQ(m.state(4), TickState::Fired);  // fresh threshold: the input fires it
}
