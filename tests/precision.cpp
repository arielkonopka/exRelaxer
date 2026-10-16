// 8-bit number formats (Precision in neuron.hpp): the rounding kernel, 8-bit
// weights and outputs of a Dense layer, learning on the 8-bit grid, and
// saving.
#include <gtest/gtest.h>
#include <cmath>
#include <sstream>
#include <stdexcept>
#include <vector>
#include "../core/kernels.hpp"
#include "../core/layers/conv2d.hpp"
#include "../core/layers/dense.hpp"
#include "../core/network.hpp"
#include "../core/random.hpp"

using namespace exr;

namespace {

// Every weight of `layer` is a code in [-127, 127] times its row's step.
void expectOnGrid(const dense& layer)
{
    for (size_t i = 0; i < layer.size(); ++i) {
        const std::vector<float> w = layer.weights(i);
        float m = 0.0f;
        for (float x : w)
            m = std::max(m, std::abs(x));
        if (m == 0.0f)
            continue;
        float step = 1.0f;
        while (127.0f * step < m)
            step *= 2.0f;
        while (127.0f * step * 0.5f >= m)
            step *= 0.5f;
        for (float x : w) {
            const float code = x / step;
            EXPECT_EQ(code, std::nearbyint(code)) << "neuron " << i << " weight " << x;
            EXPECT_LE(std::abs(code), 127.0f);
        }
    }
}

struct Net
{
    network net;
    network::LayerId hidden, out;

    explicit Net(const Precision& precision, const LearningRule& rule = {})
    {
        LayerSpec h = LayerSpec::Dense(16, false, true);
        h.precision = precision;
        h.learningRule = rule;
        LayerSpec o = LayerSpec::Dense(4, false, true);
        o.precision = precision;
        o.learningRule = rule;
        hidden = net.addLayer("h", h);
        out = net.addLayer("o", o);
        net.addInputs(hidden, 8);
        net.connect(hidden, out);
        net.addOutput(out);
    }
    void run(int ticks, float rate, int seed)
    {
        std::mt19937 gen(static_cast<unsigned>(seed));
        std::uniform_real_distribution<float> x(-1.0f, 1.0f);
        for (int t = 0; t < ticks; ++t) {
            std::vector<float> in(8);
            for (float& v : in)
                v = x(gen);
            net.setInputs(in);
            net.step();
            net.applyReward(x(gen), rate);
        }
    }
};

} // namespace

TEST(PrecisionTest, KernelRoundsRowsToPowerOfTwoSteps)
{
    // Row 0: largest |w| 1.0 -> step 2^-6 (127 / 64 = 1.98 >= 1, 127 / 128 < 1).
    // Row 1: all zero, left alone. Row 2: largest 10 -> step 2^-3.
    const std::vector<float> rows = {1.0f, 0.3f, -0.01f, 0.0f, 0.0f, 0.0f, -10.0f, 0.06f, 0.07f};
    kernels::weight_matrix m(3, 3, rows);
    EXPECT_EQ(m.step8(0), 1.0f / 64.0f);
    EXPECT_EQ(m.step8(1), 0.0f);
    EXPECT_EQ(m.step8(2), 0.125f);
    m.quantize8({}, 0, 0, m.blocks());
    EXPECT_EQ(m.at(0, 0), 1.0f);
    EXPECT_EQ(m.at(0, 1), 19.0f / 64.0f);  // 0.3 * 64 = 19.2
    EXPECT_EQ(m.at(0, 2), -1.0f / 64.0f);  // -0.64 -> -1
    EXPECT_EQ(m.at(1, 0), 0.0f);
    EXPECT_EQ(m.at(2, 0), -10.0f);
    EXPECT_EQ(m.at(2, 1), 0.0f);    // 0.48 -> 0
    EXPECT_EQ(m.at(2, 2), 0.125f);  // 0.56 -> 1
    // On the grid already: neither rounding moves it.
    const kernels::weight_matrix before = m;
    m.quantize8({}, 12345, 0, m.blocks());
    for (size_t r = 0; r < 3; ++r)
        for (size_t c = 0; c < 3; ++c)
            EXPECT_EQ(m.at(r, c), before.at(r, c));
}

TEST(PrecisionTest, StochasticRoundingIsUnbiased)
{
    // 0.3 codes above a grid point: rounds up about 30% of the time.
    const float step = 1.0f / 64.0f;
    std::vector<float> row(1000, 10.3f * step);
    row[0] = 1.0f;  // fixes the step
    double sum = 0.0;
    for (std::uint64_t seed = 1; seed <= 20; ++seed) {
        kernels::weight_matrix m(1, row.size(), row);
        m.quantize8({}, seed, 0, m.blocks());
        for (size_t c = 1; c < row.size(); ++c) {
            const float code = m.at(0, c) / step;
            EXPECT_TRUE(code == 10.0f || code == 11.0f);
            sum += code;
        }
    }
    EXPECT_NEAR(sum / (20.0 * 999.0), 10.3, 0.02);
}

TEST(PrecisionTest, DenseWeightsAndOutputsStayEightBit)
{
    reseed(3);
    Net n(Precision::Int8());
    expectOnGrid(n.net.layerAs<dense>(n.hidden));
    expectOnGrid(n.net.layerAs<dense>(n.out));
    n.run(200, 0.01f, 1);
    for (network::LayerId id : {n.hidden, n.out}) {
        const dense& layer = n.net.layerAs<dense>(id);
        expectOnGrid(layer);
        for (float y : layer.output()) {
            const float code = y * 16.0f;
            EXPECT_EQ(code, std::nearbyint(code));
            EXPECT_LE(std::abs(code), 127.0f);
        }
    }
    // Growth and new connections land on the grid too.
    n.net.growLayer(n.hidden, 3);
    expectOnGrid(n.net.layerAs<dense>(n.hidden));
    expectOnGrid(n.net.layerAs<dense>(n.out));
}

TEST(PrecisionTest, UpdatesSmallerThanAStepStillLearn)
{
    // A rate whose updates are far below one step: nearest rounding would
    // undo every one of them, stochastic rounding moves the weights.
    reseed(4);
    Net n(Precision::Int8());
    const dense& out = n.net.layerAs<dense>(n.out);
    std::vector<std::vector<float>> before;
    for (size_t i = 0; i < out.size(); ++i)
        before.push_back(out.weights(i));
    n.run(300, 1e-4f, 2);
    size_t changed = 0;
    for (size_t i = 0; i < out.size(); ++i)
        for (size_t c = 0; c < before[i].size(); ++c)
            changed += out.weights(i)[c] != before[i][c];
    EXPECT_GT(changed, 0u);
}

TEST(PrecisionTest, WeightsOnlyOrValuesOnly)
{
    reseed(5);
    Precision weights;
    weights.int8Weights = true;
    Net w(weights);
    w.run(50, 0.01f, 3);
    expectOnGrid(w.net.layerAs<dense>(w.hidden));
    bool fractional = false;  // float outputs: not all on the 1/16 grid
    for (float y : w.net.layerAs<dense>(w.hidden).output())
        fractional = fractional || y * 16.0f != std::nearbyint(y * 16.0f);
    EXPECT_TRUE(fractional);

    Precision values;
    values.int8Values = true;
    values.valueStep = 1.0f / 32.0f;
    Net v(values);
    v.run(50, 0.01f, 3);
    for (float y : v.net.layerAs<dense>(v.hidden).output())
        EXPECT_EQ(y * 32.0f, std::nearbyint(y * 32.0f));
}

TEST(PrecisionTest, SavingKeepsFormatAndRoundingSequence)
{
    reseed(6);
    Net a(Precision::Int8(), LearningRule::traced(0.9f));
    a.run(50, 0.01f, 4);
    std::stringstream ss;
    a.net.save(ss);
    auto b = network::load(ss);
    EXPECT_EQ(b->layerSpec(a.hidden).precision, Precision::Int8());
    a.run(50, 0.01f, 5);
    // The same inputs and rewards on the loaded copy.
    std::mt19937 gen(5u);
    std::uniform_real_distribution<float> x(-1.0f, 1.0f);
    for (int t = 0; t < 50; ++t) {
        std::vector<float> in(8);
        for (float& v : in)
            v = x(gen);
        b->setInputs(in);
        b->step();
        b->applyReward(x(gen), 0.01f);
    }
    for (network::LayerId id : {a.hidden, a.out})
        for (size_t i = 0; i < a.net.getLayer(id).size(); ++i)
            EXPECT_EQ(b->layerAs<dense>(id).weights(i), a.net.layerAs<dense>(id).weights(i));
}

TEST(PrecisionTest, RejectsInvalidSettings)
{
    dense d(2);
    Precision bad;
    bad.valueStep = 0.1f;  // not a power of two
    EXPECT_THROW(d.setPrecision(bad), std::invalid_argument);
    bad.valueStep = 1.0f;  // too coarse
    EXPECT_THROW(d.setPrecision(bad), std::invalid_argument);
    EXPECT_NO_THROW(d.setPrecision(Precision::Int8(1.0f / 64.0f)));

    network net;
    LayerSpec conv = LayerSpec::Conv2D(2, Window2D::square(3));
    conv.precision.int8Weights = true;
    EXPECT_THROW(net.addLayer("c", conv), std::invalid_argument);
    conv.precision = {};
    conv.precision.int8Values = true;
    EXPECT_NO_THROW(net.addLayer("c", conv));
    LayerSpec pool = LayerSpec::Pool2D(Window2D::square(2, 2));
    pool.precision.int8Values = true;
    EXPECT_THROW(net.addLayer("p", pool), std::invalid_argument);
}
