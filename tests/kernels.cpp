// The SIMD kernels must give exactly the results of plain scalar code: every
// weighted sum in input order, every weight update element by element. Sizes
// are chosen to leave partial blocks and to take the multi-threaded path.
#include <gtest/gtest.h>
#include <bit>
#include <cstdint>
#include <span>
#include <random>
#include <vector>
#include "../core/kernels.hpp"
#include "../core/layers/dense.hpp"
#include "../core/neuron.hpp"

using namespace exr;
using kernels::weight_matrix;

namespace {

std::vector<float> randomValues(size_t n, std::uint32_t seed, float scale = 1.0f)
{
    std::mt19937 g(seed);
    std::uniform_real_distribution<float> d(-scale, scale);
    std::vector<float> v(n);
    for (float& x : v) x = d(g);
    return v;
}

bool sameBits(float a, float b) { return std::bit_cast<std::uint32_t>(a) == std::bit_cast<std::uint32_t>(b); }

} // namespace

TEST(KernelsTest, MultiplyEqualsScalarDotProductBitForBit)
{
    for (auto [rows, cols] : {std::pair<size_t, size_t>{1, 1}, {7, 3}, {8, 8}, {13, 37}, {33, 1}, {70, 129}}) {
        SCOPED_TRACE(std::to_string(rows) + " x " + std::to_string(cols));
        const auto w = randomValues(rows * cols, 1);
        const auto x = randomValues(cols, 2, 3.0f);
        const weight_matrix m(rows, cols, w);
        std::vector<float> sums(m.paddedRows());
        m.multiply(x, sums, 0, m.blocks());
        for (size_t r = 0; r < rows; ++r)
            EXPECT_TRUE(sameBits(sums[r], kernels::dot(x, std::span(w).subspan(r * cols, cols)))) << "row " << r;
    }
}

TEST(KernelsTest, LearnEqualsScalarSignRuleAndSkipsInactiveRows)
{
    const size_t rows = 21, cols = 19;
    auto w = randomValues(rows * cols, 3, 12.0f);  // some weights outside the clamp range
    auto x = randomValues(cols, 4);
    x[5] = 0.0f;                                   // sign 0: weight unchanged
    weight_matrix m(rows, cols, w);

    std::vector<float> signs(cols), delta(m.paddedRows(), 0.0f);
    std::vector<std::uint8_t> active(m.paddedRows(), 0);
    kernels::signs(x, signs);
    for (size_t r = 0; r < rows; ++r) {
        active[r] = r % 3 != 0;
        delta[r] = 0.37f * static_cast<float>(r) - 3.0f;
    }
    m.learn(signs, delta, active, max_weight, 0, m.blocks());

    std::vector<float> row(cols);
    for (size_t r = 0; r < rows; ++r) {
        if (active[r])
            kernels::signRule(std::span(w).subspan(r * cols, cols), x, delta[r], max_weight);
        m.copyRow(r, row);
        for (size_t c = 0; c < cols; ++c)
            EXPECT_TRUE(sameBits(row[c], w[r * cols + c])) << "row " << r << " col " << c;
    }
}

TEST(KernelsTest, AppendColumnsKeepsExistingWeights)
{
    const size_t rows = 11;
    const auto w = randomValues(rows * 5, 5);
    const auto extra = randomValues(rows * 3, 6);
    weight_matrix m(rows, 5, w);
    m.appendColumns(3, extra);
    ASSERT_EQ(m.cols(), 8u);
    for (size_t r = 0; r < rows; ++r) {
        for (size_t c = 0; c < 5; ++c)
            EXPECT_EQ(m.at(r, c), w[r * 5 + c]);
        for (size_t c = 0; c < 3; ++c)
            EXPECT_EQ(m.at(r, 5 + c), extra[r * 3 + c]);
    }
}

// A dense layer, serial or multi-threaded, must match one neuron stepped at
// a time with neuron::step / neuron::learn on copies of its weights.
TEST(KernelsTest, DenseMatchesPerNeuronReferenceBitForBit)
{
    for (size_t size : {size_t{13}, size_t{300}}) {  // 300 x 300 takes the parallel path
        SCOPED_TRACE("layer size " + std::to_string(size));
        reseed(9);
        dense source(size, false, false), layer(size, true, true);
        std::vector<float> sensor(1, 0.0f);
        source.attachInputs(sensor);
        layer.join(source);
        layer.join(layer);  // self-connection: reads the values from before the step

        std::vector<neuron> reference(layer.neurons().begin(), layer.neurons().end());
        std::vector<std::vector<float>> weights;
        for (size_t i = 0; i < size; ++i)
            weights.push_back(layer.weights(i));

        std::vector<float> inputs(2 * size);
        for (int t = 0; t < 40; ++t) {
            sensor[0] = std::sin(0.3f * static_cast<float>(t)) * 2.0f;
            source.forward();
            // The pool: source's outputs, then this layer's previous outputs.
            std::copy(source.output().begin(), source.output().end(), inputs.begin());
            std::copy(layer.output().begin(), layer.output().end(), inputs.begin() + static_cast<std::ptrdiff_t>(size));
            layer.forward();
            for (size_t i = 0; i < size; ++i) {
                const float expected = reference[i].step(inputs, weights[i]);
                ASSERT_TRUE(sameBits(layer.output()[i], expected)) << "tick " << t << " neuron " << i;
            }
            // Learning reads the pool the step summed (time t): the self
            // inputs stay the pre-step outputs (see doc/model.md).
            const float reward = (t % 3 == 0) ? 1.0f : -0.5f;
            layer.applyReward(reward, 0.01f);
            for (size_t i = 0; i < size; ++i)
                reference[i].learn(weights[i], inputs, reward, 0.01f);
        }
        for (size_t i = 0; i < size; ++i) {
            const std::vector<float> actual = layer.weights(i);
            ASSERT_EQ(actual.size(), weights[i].size());
            for (size_t c = 0; c < actual.size(); ++c)
                ASSERT_TRUE(sameBits(actual[c], weights[i][c])) << "neuron " << i << " weight " << c;
        }
    }
}
