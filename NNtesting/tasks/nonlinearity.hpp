#pragma once
// Shared by the nonlinearity-substitution experiments (nl_static,
// nl_temporal): known target functions, a stack of equal-width hidden layers
// with one linear output, and meters for activity and cost on the test set.
//
//   x (d inputs) --> hidden 1 --> ... --> hidden depth --> output (1, linear)
//
// The compared models differ only in their hidden neurons:
//   relu    conventional: output = max(0, sum)   (LayerSpec::rectify)
//   er      production E-R, habituation off: adaptive threshold, state across ticks
//   er_memoryless
//           the same E-R neurons, but every hidden neuron's state (threshold,
//           output) is reset to rest before each presentation: E-R's transfer
//           within a presentation, no memory across them. A test-only
//           wrapper: it resets through the neuron's public serialization.
//   gate    static threshold: output = sum if |sum| > gate, else 0 (LayerSpec::gate)
//   clamp   the library's plain neuron: output = sum, clamped to +-max_output
//           (nearly linear for these inputs; "linear" is the same model)
// Everything else is the same: the task data, the weight initialization
// (uniform, variance 1 / fan-in, drawn from the trial seed), feedback
// alignment with a learned bias in every layer, the learning rate, the
// training budget and the stopping rule. No term anywhere counts activity.
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <memory>
#include <numbers>
#include <optional>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>
#include "er_options.hpp"
#include "experiment.hpp"
#include "layers/dense.hpp"
#include "network.hpp"

namespace nonlinearity {

using namespace exr;

// A static regression task: y = f(x), x uniform in [-1, 1]^d.
//   l0  y = x1 + x2
//   l1  y = x1 * x2
//   l2  y = sin(x1 * x2)
//   l3  y = sin(x1 * x2) + exp(-x3^2)
//   l4  y = K^-1/2 * sum_k sin(a_k . x + b_k), a_k uniform in [-2, 2]^d,
//       b_k uniform in [-pi, pi], drawn from task_seed (fixed per task, never
//       redrawn between models). The 1/sqrt(K) keeps the target's variance
//       near 0.5 for every K.
struct StaticTask
{
    std::string level;
    size_t k = 0;         // l4 components
    size_t inputs = 2;    // d
    std::vector<float> a; // l4: K x d
    std::vector<float> b; // l4: K

    StaticTask(const std::string& level_, size_t k_, size_t l4Inputs, std::uint32_t taskSeed) : level(level_), k(k_)
    {
        if (level == "l0" || level == "l1" || level == "l2")
            inputs = 2;
        else if (level == "l3")
            inputs = 3;
        else if (level == "l4") {
            if (k == 0 || l4Inputs == 0)
                throw std::invalid_argument("l4 needs k >= 1 and l4_inputs >= 1");
            inputs = l4Inputs;
            std::mt19937 g(taskSeed);
            std::uniform_real_distribution<float> freq(-2.0f, 2.0f), phase(-std::numbers::pi_v<float>, std::numbers::pi_v<float>);
            a.resize(k * inputs);
            b.resize(k);
            for (size_t c = 0; c < k; ++c) {
                for (size_t j = 0; j < inputs; ++j)
                    a[c * inputs + j] = freq(g);
                b[c] = phase(g);
            }
        } else
            throw std::invalid_argument("task must be l0, l1, l2, l3 or l4");
    }

    // Complexity on one axis for capacity curves: the level, or K for l4.
    double complexity() const { return level == "l4" ? static_cast<double>(k) : static_cast<double>(level[1] - '0'); }

    float operator()(const std::vector<float>& x) const
    {
        if (level == "l0")
            return x[0] + x[1];
        if (level == "l1")
            return x[0] * x[1];
        if (level == "l2")
            return std::sin(x[0] * x[1]);
        if (level == "l3")
            return std::sin(x[0] * x[1]) + std::exp(-x[2] * x[2]);
        float y = 0.0f;
        for (size_t c = 0; c < k; ++c) {
            float s = b[c];
            for (size_t j = 0; j < inputs; ++j)
                s += a[c * inputs + j] * x[j];
            y += std::sin(s);
        }
        return y / std::sqrt(static_cast<float>(k));
    }
};

// A temporal task on a stream x(0), x(1), ...: y(t) depends on x(t) and
// earlier values. Binary tasks draw x(t) in {0, 1} (fair coin) and present
// it as -1 / +1; the target is 0 or 1 and a decision is y > 0.5.
//   t1  y = x(t) XOR x(t-1)                       (delayed XOR)
//   t2  y = x(t) AND NOT x(t-3)                   (delayed conjunction)
//   t3  y = x(t) XOR x(t-1) XOR ... XOR x(t-n+1)   (parity of the last n)
//   t4  y = sin(x(t) * x(t-2)), x(t) uniform in [-1, 1]
// With a window W, the network also sees x(t-1) .. x(t-W) as extra inputs
// (a tapped delay line): enough window turns the task into a static one.
struct TemporalTask
{
    std::string level;
    size_t n = 2;       // t3: parity length
    size_t window = 0;  // extra past inputs shown to the network

    TemporalTask(const std::string& level_, size_t n_, size_t window_) : level(level_), n(n_), window(window_)
    {
        if (level != "t1" && level != "t2" && level != "t3" && level != "t4")
            throw std::invalid_argument("task must be t1, t2, t3 or t4");
        if (level == "t3" && n < 1)
            throw std::invalid_argument("t3 needs n >= 1");
    }

    bool binary() const { return level != "t4"; }
    size_t inputs() const { return window + 1; }
    // How far back the target reaches.
    size_t lag() const { return level == "t1" ? 1 : level == "t2" ? 3 : level == "t3" ? n - 1 : 2; }
    double complexity() const { return level == "t3" ? static_cast<double>(n) : static_cast<double>(level[1] - '0'); }

    // Stream values: {0, 1} for binary tasks, [-1, 1] for t4.
    float draw(std::mt19937& g) const
    {
        if (binary())
            return static_cast<float>(std::uniform_int_distribution<int>(0, 1)(g));
        return std::uniform_real_distribution<float>(-1.0f, 1.0f)(g);
    }

    // Target at the end of `history` (history.back() is x(t)); needs lag()+1 values.
    float target(const std::vector<float>& h) const
    {
        const size_t t = h.size() - 1;
        auto bit = [&](size_t back) { return h[t - back] > 0.5f; };
        if (level == "t1")
            return static_cast<float>(bit(0) != bit(1));
        if (level == "t2")
            return static_cast<float>(bit(0) && !bit(3));
        if (level == "t3") {
            bool parity = false;
            for (size_t b = 0; b < n; ++b)
                parity ^= bit(b);
            return static_cast<float>(parity);
        }
        return std::sin(h[t] * h[t - 2]);
    }

    // The network's input at the end of `history`: x(t), x(t-1) .. x(t-window).
    void present(const std::vector<float>& h, std::vector<float>& x) const
    {
        x.resize(inputs());
        for (size_t w = 0; w <= window; ++w) {
            const float v = w < h.size() ? h[h.size() - 1 - w] : 0.0f;
            x[w] = binary() ? 2.0f * v - 1.0f : v;
        }
    }
};

// n samples, x uniform in [-1, 1]^d, from their own seed.
struct Dataset
{
    std::vector<std::vector<float>> x;
    std::vector<float> y;
    double variance = 0.0;

    Dataset(const StaticTask& task, size_t n, std::uint32_t seed)
    {
        std::mt19937 g(seed);
        std::uniform_real_distribution<float> u(-1.0f, 1.0f);
        x.assign(n, std::vector<float>(task.inputs));
        y.resize(n);
        double mean = 0.0;
        for (size_t i = 0; i < n; ++i) {
            for (float& v : x[i])
                v = u(g);
            y[i] = task(x[i]);
            mean += y[i];
        }
        mean /= static_cast<double>(std::max<size_t>(n, 1));
        for (float v : y)
            variance += (v - mean) * (v - mean);
        variance /= static_cast<double>(std::max<size_t>(n, 1));
    }
};

// Activity and cost of hidden neurons over the ticks it observed.
struct Activity
{
    size_t hidden = 0, ticks = 0, samples = 0;
    double active = 0.0;           // sum over ticks of active hidden neurons
    double unique = 0.0;           // sum over samples of neurons active at least once in the sample
    double eventSynops = 0.0;      // sum over ticks of (active sources x their fan-out)
    std::vector<std::uint8_t> used, everUsed;
    std::vector<float> thresholds; // E-R: every hidden threshold at the end of each sample

    double activePerTick() const { return ticks ? active / static_cast<double>(ticks) : 0.0; }
    double uniquePerSample() const { return samples ? unique / static_cast<double>(samples) : 0.0; }
    size_t neverActive() const { return static_cast<size_t>(std::count(everUsed.begin(), everUsed.end(), 0)); }
};

inline double quantile(std::vector<float> v, double q)
{
    if (v.empty())
        return 0.0;
    const size_t i = std::min(v.size() - 1, static_cast<size_t>(q * static_cast<double>(v.size() - 1) + 0.5));
    std::nth_element(v.begin(), v.begin() + static_cast<std::ptrdiff_t>(i), v.end());
    return v[i];
}

class Mlp
{
public:
    Mlp(const std::string& model, size_t inputs, size_t depth, size_t width, float gate,
        const ThresholdGrowth& growth = {}, const Spontaneous& spontaneous = {},
        std::optional<Habituation> habituation = std::nullopt, bool normalize = false,
        bool normalizeReadout = false)
        : model_(model == "linear" ? "clamp" : model == "er_memoryless" ? "er" : model),
          memoryless_(model == "er_memoryless"), inputs_(inputs), depth_(depth), width_(width)
    {
        if (model_ != "relu" && model_ != "er" && model_ != "gate" && model_ != "clamp")
            throw std::invalid_argument("model must be relu, er, er_memoryless, gate or clamp");
        if (depth == 0 || width == 0)
            throw std::invalid_argument("depth and width must be at least 1");
        const LearningRule rule = LearningRule::feedbackAlignment().withBias();
        for (size_t l = 0; l < depth; ++l) {
            LayerSpec spec = LayerSpec::Dense(width, habituation.has_value(), model_ == "er");
            if (habituation)
                spec.habituationRule = *habituation;
            spec.rectify = model_ == "relu";
            spec.normalize = normalize;  // hidden layers; the readout has its own setting
            if (model_ == "gate")
                spec.gate = gate;
            if (model_ == "er") {
                spec.thresholdGrowth = growth;
                spec.spontaneous = spontaneous;
            }
            spec.learningRule = rule;
            hidden_.push_back(net_.addLayer("h" + std::to_string(l + 1), spec));
        }
        LayerSpec out = LayerSpec::Dense(1, false, false);
        out.normalize = normalizeReadout;  // the builder turns it on
        out.learningRule = rule;
        out_ = net_.addLayer("out", out);
        net_.addInputs(hidden_[0], inputs, "x");
        for (size_t l = 1; l < depth; ++l)
            net_.connect(hidden_[l - 1], hidden_[l]);
        net_.connect(hidden_.back(), out_);
        net_.addOutput(out_);
        // One initialization for every model: the library's uniform [-1, 1]
        // draw, scaled to variance 1 / fan-in.
        for (size_t l = 0; l <= depth; ++l) {
            auto& d = net_.layerAs<dense>(l < depth ? hidden_[l] : out_);
            const size_t fanIn = l == 0 ? inputs : width;
            const float scale = std::sqrt(3.0f / static_cast<float>(fanIn));
            for (size_t i = 0; i < d.size(); ++i) {
                std::vector<float> w = d.weights(i);
                for (float& v : w)
                    v *= scale;
                d.setWeights(i, w);
            }
        }
    }

    size_t depth() const { return depth_; }
    size_t width() const { return width_; }
    size_t hiddenNeurons() const { return depth_ * width_; }
    // Trainable: every weight and bias, output layer included.
    size_t parameters() const { return inputs_ * width_ + width_ + (depth_ - 1) * (width_ * width_ + width_) + width_ + 1; }
    // Weighted inputs computed per tick by a dense implementation.
    size_t denseSynops() const { return inputs_ * width_ + (depth_ - 1) * width_ * width_ + width_; }
    // Ticks for an input to reach the output, plus `settle`.
    size_t hold(size_t settle) const { return depth_ + 1 + settle; }
    network& net() { return net_; }
    // The `resting_threshold` option on every hidden layer (see
    // er_options::calibrateRestingThreshold). With "auto", a gate model's
    // gate is scaled by the same factor. Returns the mean factor.
    float calibrateThresholds(const std::string& setting)
    {
        double sum = 0.0;
        for (size_t l = 0; l < depth_; ++l) {
            auto& layer = net_.layerAs<neuron_layer>(hidden_[l]);
            const float factor = er_options::calibrateRestingThreshold(layer, setting);
            if (model_ == "gate" && setting == "auto")
                layer.setGate(layer.gate() * factor);
            sum += factor;
        }
        return static_cast<float>(sum / static_cast<double>(depth_));
    }
    const neuron_layer& hidden(size_t l) const { return net_.layerAs<neuron_layer>(hidden_[l]); }
    bool memoryless() const { return memoryless_; }
    const std::string& model() const { return model_; }

    // Takes over another network's weights and biases (same depth, width
    // and inputs): training one kind of neuron, then running another.
    void copyFrom(const Mlp& other)
    {
        if (other.depth_ != depth_ || other.width_ != width_ || other.inputs_ != inputs_)
            throw std::invalid_argument("copyFrom: different architecture");
        for (size_t l = 0; l <= depth_; ++l) {
            const auto id = l < depth_ ? hidden_[l] : out_;
            const auto& from = other.net_.layerAs<dense>(l < depth_ ? other.hidden_[l] : other.out_);
            auto& to = net_.layerAs<dense>(id);
            for (size_t i = 0; i < to.size(); ++i) {
                to.setWeights(i, from.weights(i));
                to.setBias(i, from.bias(i));
            }
        }
    }

    // Puts every hidden neuron back at rest (resting threshold, zero output,
    // no habituation streak) without touching weights or per-neuron
    // dynamics: a WeightsOnly round trip through the neuron's own format.
    void resetHiddenState()
    {
        for (size_t l = 0; l < depth_; ++l)
            for (neuron& n : net_.layerAs<neuron_layer>(hidden_[l]).neurons()) {
                std::stringstream buffer;
                n.serialize(buffer, {});
                n.deserialize(buffer, DeserializeMode::WeightsOnly);
            }
    }

    // Presents x for `ticks` ticks and returns the output at the last one.
    // With a meter, counts hidden activity on every tick; with `trace`, also
    // writes every hidden neuron's output and threshold per tick.
    float present(const std::vector<float>& x, size_t ticks, Activity* meter = nullptr, std::ostream* trace = nullptr,
                  const std::string& tracePrefix = "")
    {
        if (memoryless_)
            resetHiddenState();
        net_.setInputs("x", x);
        if (meter) {
            meter->used.assign(hiddenNeurons(), 0);
            if (meter->everUsed.size() != hiddenNeurons())
                meter->everUsed.assign(hiddenNeurons(), 0);
            meter->hidden = hiddenNeurons();
        }
        for (size_t t = 0; t < ticks; ++t) {
            net_.step();
            if (meter) {
                ++meter->ticks;
                meter->eventSynops += static_cast<double>(inputs_ * width_);  // the inputs are always on
                for (size_t l = 0; l < depth_; ++l) {
                    const auto neurons = hidden(l).neurons();
                    size_t active = 0;
                    for (size_t i = 0; i < neurons.size(); ++i) {
                        const float y = neurons[i].output();
                        if (std::abs(y) > firing_epsilon) {
                            ++active;
                            meter->used[l * width_ + i] = 1;
                            meter->everUsed[l * width_ + i] = 1;
                        }
                        if (trace)
                            *trace << tracePrefix << ',' << t << ',' << l + 1 << ',' << i << ',' << y << ','
                                   << neurons[i].threshold() << '\n';
                    }
                    meter->active += static_cast<double>(active);
                    meter->eventSynops += static_cast<double>(active * (l + 1 < depth_ ? width_ : 1));
                }
            }
        }
        if (meter) {
            ++meter->samples;
            meter->unique += static_cast<double>(std::count(meter->used.begin(), meter->used.end(), 1));
            if (model_ == "er")
                for (size_t l = 0; l < depth_; ++l)
                    for (const neuron& n : hidden(l).neurons())
                        meter->thresholds.push_back(n.threshold());
        }
        return net_.outputs()[0];
    }

    void learn(float target, float y, float lr)
    {
        const float error = target - y;
        net_.applyError(std::span<const float>(&error, 1), lr);
    }

    // Training presentation that learns on every tick, lr / ticks each
    // time: the update follows the activity over the whole presentation
    // rather than its last tick. Returns the output at the last tick.
    float presentLearningEveryTick(const std::vector<float>& x, size_t ticks, float target, float lr)
    {
        if (memoryless_)
            resetHiddenState();
        net_.setInputs("x", x);
        float y = 0.0f;
        for (size_t t = 0; t < ticks; ++t) {
            net_.step();
            y = net_.outputs()[0];
            learn(target, y, lr / static_cast<float>(ticks));
        }
        return y;
    }

private:
    std::string model_;
    bool memoryless_ = false;
    size_t inputs_, depth_, width_;
    network net_;
    std::vector<network::LayerId> hidden_;
    network::LayerId out_ = 0;
};

} // namespace nonlinearity
