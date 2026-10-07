// The per-synapse learning rules (learning.hpp): the neurons' local
// derivatives, reward-modulated eligibility traces, e-prop and surrogate
// gradients (backpropagation through time), checked against finite
// differences of the network itself.
#include <gtest/gtest.h>
#include <cmath>
#include <functional>
#include <memory>
#include <sstream>
#include <vector>
#include "../core/layers/dense.hpp"
#include "../core/network.hpp"

using namespace exr;

namespace {

LayerSpec plainDense(size_t size, bool er)
{
    LayerSpec spec = LayerSpec::Dense(size, false, er);
    spec.normalize = false;
    return spec;
}

} // namespace

// --- Local derivatives --------------------------------------------------------

TEST(DerivativesTest, MatchFiniteDifferencesAwayFromTheStep)
{
    // A firing E-R neuron with the linear growth rule: y = s, thr' = thr +
    // a (|s| - thr), so dy/ds = 1, dthr'/ds = a * sign(s), and outside the
    // pseudo-derivative's bump dy/dthr = 0.
    for (float s : {3.0f, -3.0f}) {
        neuron n(false, true);
        Derivatives d;
        neuron copy = n;
        const float y = n.activate(s, 0.1f, d);
        EXPECT_FLOAT_EQ(y, s);
        EXPECT_FLOAT_EQ(d.dyds, 1.0f);
        EXPECT_FLOAT_EQ(d.dydthr, 0.0f);
        EXPECT_FLOAT_EQ(d.dthrdthr, 1.0f - 0.5f);
        EXPECT_FLOAT_EQ(d.dthrds, 0.5f * (s > 0 ? 1.0f : -1.0f));
        const float eps = 1e-3f;
        neuron moved = copy;
        moved.activate(s + eps);
        EXPECT_NEAR((moved.threshold() - n.threshold()) / eps, d.dthrds, 1e-3f);
    }
    // Silent: the threshold relaxes by the recovery factor.
    neuron quiet(false, true);
    Derivatives d;
    quiet.activate(0.05f, 0.1f, d);
    EXPECT_FLOAT_EQ(d.dthrdthr, recovery_factor);
    EXPECT_FLOAT_EQ(d.dthrds, 0.0f);
    EXPECT_FLOAT_EQ(d.dyds, 0.0f);
    // The log rule: thr' = thr + alpha ln(|s| / thr).
    neuron logged(false, true);
    logged.setThresholdGrowth({ThresholdGrowth::Rule::Log, 0.0f});
    neuron before = logged;
    logged.activate(4.0f, 0.1f, d);
    const float eps = 1e-3f;
    before.activate(4.0f + eps);
    EXPECT_NEAR((before.threshold() - logged.threshold()) / eps, d.dthrds, 2e-3f);
}

TEST(DerivativesTest, PseudoDerivativeNearTheThreshold)
{
    // Just below the threshold (0.2): silent, but the bump gives the
    // gradient a way to push the sum over it.
    neuron n(false, true);
    Derivatives d;
    n.activate(0.19f, 0.5f, d);
    EXPECT_EQ(n.output(), 0.0f);
    EXPECT_GT(d.dyds, 0.0f);
    EXPECT_LT(d.dydthr, 0.0f);  // a higher threshold would lower y
    // Linear neurons: dy/ds = 1; a clamped sum does not move: 0.
    neuron linear(false, false);
    linear.activate(0.3f, 0.5f, d);
    EXPECT_FLOAT_EQ(d.dyds, 1.0f);
    linear.activate(2 * max_output, 0.5f, d);
    EXPECT_FLOAT_EQ(d.dyds, 0.0f);
    // Perceptron: output 1 above the gate; only the bump around it moves y.
    neuron p(false, false);
    p.setRectified(true);
    p.setBinary(true);
    p.setGate(0.5f);
    EXPECT_FLOAT_EQ(p.activate(2.0f, 0.5f, d), 1.0f);
    EXPECT_FLOAT_EQ(d.dyds, 0.0f);
    EXPECT_FLOAT_EQ(p.activate(0.4f, 0.5f, d), 0.0f);
    EXPECT_GT(d.dyds, 0.0f);
    EXPECT_FLOAT_EQ(p.activate(-0.4f, 0.5f, d), 0.0f);
    EXPECT_FLOAT_EQ(d.dyds, 0.0f);  // one-sided
}

// --- Eligibility ----------------------------------------------------------------

TEST(EligibilityRuleTest, TraceRemembersWhichInputWasActiveAtFiring)
{
    network net;
    const auto id = net.addLayer("h", plainDense(1, false));
    net.addInputs(id, 2);
    net.setLearningRule(id, LearningRule::eligibility(0.5f, 0.0f));
    auto& layer = net.layerAs<dense>(id);
    layer.setWeights(0, {1.0f, 1.0f});
    // Tick 1: input 0 active, y = 2. Tick 2: input 1 active, y = 3.
    net.setInputs({2.0f, 0.0f});
    net.step();
    net.setInputs({0.0f, 3.0f});
    net.step();
    // e = 0.5 * (|2| * x1) + |3| * x2
    const std::vector<float> e = layer.synapseTrace(0);
    ASSERT_EQ(e.size(), 2u);
    EXPECT_FLOAT_EQ(e[0], 0.5f * 2.0f * 2.0f);
    EXPECT_FLOAT_EQ(e[1], 3.0f * 3.0f);
    // A product of traces (the Trace rule) would credit input 0 for the
    // second firing too; the eligibility trace does not.
    net.applyReward(1.0f, 0.01f);
    const std::vector<float> w = layer.weights(0);
    const float gain = default_learning_gain;
    EXPECT_FLOAT_EQ(w[0], 1.0f + 0.01f * gain * e[0]);
    EXPECT_FLOAT_EQ(w[1], 1.0f + 0.01f * gain * e[1]);
}

// --- e-prop ---------------------------------------------------------------------

namespace {

// One E-R neuron over two sensors, weights `w`, run through `inputs`.
std::unique_ptr<network> runEprop(std::vector<float> w, const std::vector<std::vector<float>>& inputs,
                                  LearningRule rule = LearningRule::eprop())
{
    auto net = std::make_unique<network>();
    const auto id = net->addLayer("h", plainDense(1, true));
    net->addInputs(id, 2);
    net->setLearningRule(id, rule);
    net->layerAs<dense>(id).setWeights(0, w);
    for (const std::vector<float>& x : inputs) {
        net->setInputs(x);
        net->step();
    }
    return net;
}

} // namespace

TEST(EPropTest, ThresholdEligibilityIsTheThresholdsGradient)
{
    // Fire, relax, fire again, relax: the threshold after the sequence
    // depends on the weights through both firings and the recoveries in
    // between. Away from the firing edges this is smooth, so the e-prop
    // threshold eligibility must equal a finite difference.
    const std::vector<std::vector<float>> inputs = {
        {1.0f, 0.5f}, {0.0f, 0.0f}, {0.0f, 0.0f}, {2.0f, 1.0f}, {0.0f, 0.0f}, {0.1f, 0.0f}};
    const std::vector<float> w = {1.2f, 0.7f};
    auto net = runEprop(w, inputs);
    const auto& layer = net->layerAs<dense>(0);
    const std::vector<float> a = layer.thresholdTrace(0);
    ASSERT_EQ(a.size(), 2u);
    const float eps = 1e-3f;
    for (size_t j = 0; j < 2; ++j) {
        std::vector<float> up = w, down = w;
        up[j] += eps;
        down[j] -= eps;
        const float numeric = (runEprop(up, inputs)->layerAs<dense>(0).neurons()[0].threshold() -
                               runEprop(down, inputs)->layerAs<dense>(0).neurons()[0].threshold()) /
                              (2 * eps);
        EXPECT_NEAR(a[j], numeric, 1e-3f) << "weight " << j;
        EXPECT_NE(a[j], 0.0f);
    }
}

TEST(EPropTest, LearnsFromTheErrorOfAnOutputLayer)
{
    // e-prop on an output layer is a gradient step: the error falls.
    reseed(3);
    network net;
    const auto id = net.addLayer("out", plainDense(1, false));
    net.addInputs(id, 3);
    net.addOutput(id);
    net.setLearningRule(id, LearningRule::eprop());
    const std::vector<float> target_w = {0.5f, -0.3f, 0.8f};
    std::vector<float> errors;
    for (int t = 0; t < 400; ++t) {
        const std::vector<float> x = {std::sin(0.3f * t), std::cos(0.7f * t), 0.5f};
        net.setInputs(x);
        net.step();
        const float target = target_w[0] * x[0] + target_w[1] * x[1] + target_w[2] * x[2];
        const float e = target - net.outputs()[0];
        errors.push_back(std::abs(e));
        net.applyError(std::vector<float>{e}, 0.05f);
    }
    float early = 0.0f, late = 0.0f;
    for (int t = 0; t < 50; ++t) {
        early += errors[t];
        late += errors[errors.size() - 1 - t];
    }
    EXPECT_LT(late, 0.1f * early);
}

// --- Surrogate gradients --------------------------------------------------------

TEST(SurrogateTest, MatchesEpropOnASingleLayer)
{
    // Without recurrence, backpropagation through time of a loss at the
    // last tick gives exactly e-prop's eligibility (with no trace filter)
    // times the error: the two rules must take the same step.
    const std::vector<std::vector<float>> inputs = {
        {1.0f, 0.5f}, {0.0f, 0.0f}, {0.3f, 0.1f}, {2.0f, 1.0f}, {0.0f, 0.4f}, {1.5f, -0.2f}};
    const std::vector<float> w = {1.2f, 0.7f};
    auto eprop = runEprop(w, inputs);
    auto surrogate = runEprop(w, inputs, LearningRule::surrogate(64));
    eprop->addOutput(0);
    surrogate->addOutput(0);
    EXPECT_EQ(surrogate->layerAs<dense>(0).historyFrames(), inputs.size());
    eprop->applyError(std::vector<float>{0.7f}, 0.01f);
    surrogate->applyError(std::vector<float>{0.7f}, 0.01f);
    const std::vector<float> a = eprop->layerAs<dense>(0).weights(0);
    const std::vector<float> b = surrogate->layerAs<dense>(0).weights(0);
    EXPECT_NE(a[0], w[0]);
    for (size_t j = 0; j < 2; ++j)
        EXPECT_NEAR(a[j], b[j], 1e-6f) << "weight " << j;
}

namespace {

// Linear recurrent hidden layer (3) -> linear output (1), fixed weights;
// returns the loss 0.5 * (target - y)^2 at the last tick, and the network.
struct Recurrent
{
    std::unique_ptr<network> net = std::make_unique<network>();
    network::LayerId h, out;

    explicit Recurrent(const std::vector<float>& hidden, size_t window = 16)
    {
        h = net->addLayer("h", plainDense(3, false));
        out = net->addLayer("out", plainDense(1, false));
        net->addInputs(h, 2);
        net->connect(h, h);
        net->connect(h, out);
        net->addOutput(out);
        net->setLearningRule(h, LearningRule::surrogate(static_cast<std::uint32_t>(window)));
        net->setLearningRule(out, LearningRule::surrogate(static_cast<std::uint32_t>(window)));
        auto& hl = net->layerAs<dense>(h);
        for (size_t i = 0; i < 3; ++i)
            hl.setWeights(i, std::vector<float>(hidden.begin() + i * 5, hidden.begin() + i * 5 + 5));
        net->layerAs<dense>(out).setWeights(0, {0.6f, -0.4f, 0.3f});
    }
    float run(float target)
    {
        for (int t = 0; t < 5; ++t) {
            net->setInputs({std::sin(1.0f + t), std::cos(0.5f * t)});
            net->step();
        }
        const float e = target - net->outputs()[0];
        return 0.5f * e * e;
    }
};

} // namespace

TEST(SurrogateTest, BackpropagatesThroughLayersAndRecurrence)
{
    // Linear neurons have exact derivatives, so the surrogate gradient is
    // the true gradient of the last tick's loss, through the readout and
    // through the hidden layer's own recurrent weights over five ticks.
    std::vector<float> hidden(15);
    for (size_t k = 0; k < hidden.size(); ++k)
        hidden[k] = 0.3f * std::sin(1.7f * static_cast<float>(k) + 0.2f);
    const float target = 0.9f, rate = 1e-3f;

    Recurrent trained(hidden);
    trained.run(target);
    const float e = target - trained.net->outputs()[0];
    trained.net->applyError(std::vector<float>{e}, rate);
    // w' = w - rate * gain * dL/dw.
    const float gain = default_learning_gain;
    const float eps = 1e-3f;
    for (size_t k = 0; k < hidden.size(); ++k) {
        std::vector<float> up = hidden, down = hidden;
        up[k] += eps;
        down[k] -= eps;
        const float numeric = (Recurrent(up).run(target) - Recurrent(down).run(target)) / (2 * eps);
        const float stepped = trained.net->layerAs<dense>(trained.h).weights(k / 5)[k % 5];
        EXPECT_NEAR((hidden[k] - stepped) / (rate * gain), numeric, 2e-3f) << "hidden weight " << k;
    }
    // A shorter window truncates the recurrent paths: a different step.
    Recurrent truncated(hidden, 1);
    truncated.run(target);
    truncated.net->applyError(std::vector<float>{e}, rate);
    EXPECT_NE(truncated.net->layerAs<dense>(truncated.h).weights(0)[3],
              trained.net->layerAs<dense>(trained.h).weights(0)[3]);
}

// --- Freezing and saving -----------------------------------------------------------

TEST(PerSynapseRuleTest, FrozenNeuronsAndInputsKeepTheirWeights)
{
    network net;
    const auto a = net.addLayer("a", plainDense(2, false));
    const auto h = net.addLayer("h", plainDense(2, false));
    net.addInputs(a, 1, "x");
    net.connectInputs("x", h);
    net.connect(a, h);
    net.setLearningRule(h, LearningRule::eligibility(0.0f, 0.0f));
    auto& layer = net.layerAs<dense>(h);
    layer.setWeights(0, {0.5f, 0.5f, 0.5f});
    layer.setWeights(1, {0.5f, 0.5f, 0.5f});
    net.freezeNeurons(h, 1, 1);
    net.freezeInputs(h, a);  // the two columns reading a
    EXPECT_EQ(layer.frozenInputCount(), 2u);
    net.setInputs({1.0f});
    net.step();
    net.step();
    net.applyReward(1.0f, 0.1f);
    const std::vector<float> w0 = layer.weights(0), w1 = layer.weights(1);
    EXPECT_NE(w0[0], 0.5f);   // reads x: learns
    EXPECT_EQ(w0[1], 0.5f);   // reads a: frozen input
    EXPECT_EQ(w0[2], 0.5f);
    EXPECT_EQ(w1, std::vector<float>(3, 0.5f));  // frozen neuron

    // Saved and loaded: the same frozen state and traces.
    std::stringstream data;
    net.save(data);
    auto loaded = network::load(data);
    const auto& copy = loaded->layerAs<dense>(h);
    EXPECT_EQ(copy.frozenInputCount(), 2u);
    EXPECT_TRUE(copy.neuronFrozen(1));
    EXPECT_FALSE(copy.neuronFrozen(0));
    EXPECT_EQ(copy.synapseTrace(0), layer.synapseTrace(0));
    EXPECT_EQ(loaded->layerSpec(h).learningRule, net.layerSpec(h).learningRule);
}

TEST(PerSynapseRuleTest, OnlyDenseLayersAndValidParameters)
{
    network net;
    const auto c = net.addLayer("c", LayerSpec::Conv2D(2, Window2D::square(3)));
    EXPECT_THROW(net.setLearningRule(c, LearningRule::eprop()), std::invalid_argument);
    EXPECT_THROW(LearningRule::surrogate(0).validate(), std::invalid_argument);
    EXPECT_THROW(LearningRule::eprop(0.0f, 0.0f).validate(), std::invalid_argument);
    EXPECT_EQ(describeLearningRule(LearningRule::eprop(0.9f)), "eprop tr0.9");
    EXPECT_EQ(describeLearningRule(LearningRule::surrogate(4)), "surrogate(4)");
}
