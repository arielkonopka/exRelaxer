// Reinforcement signals a network computes from its own activity (see
// network::setCritic, network::setCuriosity):
//
//   critic      a value predictor V(s) = v . phi(s) + b over feature
//               activity phi, learned online by TD(lambda). Its temporal-
//               difference error delta = r + gamma * V(s') - V(s) is the
//               actor-critic learning signal: a reward with the expected
//               part taken out, which network::applyRewardTD feeds to the
//               layers instead of the raw reward.
//   curiosity   a linear forward model that predicts the next activity of
//               some layers or inputs from the current activity of others,
//               learned online. How badly it predicted is an intrinsic
//               reward: high for what the network cannot foresee yet,
//               falling as it learns to.
//
// Both learn with a normalised step (rate / (1 + |input|^2)), so the rate
// does not depend on how many features there are or how large they are.
#pragma once
#include <cstddef>
#include <cstdint>
#include <iosfwd>
#include <span>
#include <string>
#include <vector>

namespace exr {

struct CriticSpec
{
    std::vector<size_t> layers;       // feature layers: their outputs, concatenated in this order
    std::vector<std::string> inputs;  // named input sources, after the layers
    float gamma = 0.95f;              // discount per tick, [0, 1]
    float lambda = 0.8f;              // eligibility trace decay of TD(lambda), [0, 1]
    float rate = 0.05f;               // learning rate of the value weights (> 0)

    // Throws std::invalid_argument when a parameter is out of range.
    void validate() const;
    bool operator==(const CriticSpec&) const = default;
};

struct CuriositySpec
{
    std::vector<size_t> predictLayers;        // what is predicted: these layers' next outputs...
    std::vector<std::string> predictInputs;   // ...and these named inputs' next values
    std::vector<size_t> fromLayers;           // from these layers' current outputs...
    std::vector<std::string> fromInputs;      // ...and these named inputs
    float rate = 0.1f;                        // learning rate of the forward model (> 0)
    float scale = 1.0f;                       // intrinsic reward = scale * mean squared prediction error (>= 0)

    void validate() const;
    bool operator==(const CuriositySpec&) const = default;
};

class critic
{
public:
    critic(const CriticSpec& spec, size_t features);

    const CriticSpec& spec() const { return spec_; }
    size_t features() const { return weights_.size(); }
    // V(phi) with the current weights.
    float value(std::span<const float> phi) const;
    // One TD step: `phi` is the state reached, `reward` what the step from
    // the previous state earned. Returns delta = reward + gamma * V(phi) -
    // V(previous) (V(phi) taken as 0 when `terminal`), after moving the
    // weights along their traces; the first call after reset() has no
    // previous state and returns 0. A terminal step ends the episode
    // (reset() follows).
    float update(std::span<const float> phi, float reward, bool terminal);
    // Forgets the previous state and the traces (an episode boundary).
    void reset();
    float lastError() const { return last_error_; }
    std::span<const float> weights() const { return weights_; }
    float bias() const { return bias_; }

    // The features changed: `count` new ones (weight 0) before position
    // `at`, or the ones at sorted positions `removed` are gone.
    void insertFeatures(size_t at, size_t count);
    void removeFeatures(std::span<const size_t> removed);

    // The spec, then the weights and state. load: readSpec(), build a critic
    // with it, then deserialize() the rest. Both throw std::runtime_error on
    // malformed data.
    void serialize(std::ostream& os) const;
    static CriticSpec readSpec(std::istream& is);
    void deserialize(std::istream& is, bool fullState);

private:
    CriticSpec spec_;
    std::vector<float> weights_, traces_, previous_;
    float bias_ = 0.0f, bias_trace_ = 0.0f;
    bool has_previous_ = false;
    float last_error_ = 0.0f;
};

class curiosity
{
public:
    curiosity(const CuriositySpec& spec, size_t targets, size_t inputs);

    const CuriositySpec& spec() const { return spec_; }
    size_t targets() const { return bias_.size(); }
    size_t inputs() const { return input_count_; }
    // One step: predicts `target` from the inputs remembered by the last
    // call, learns from the miss, remembers `input`, and returns the
    // intrinsic reward (0 on the first call after reset()).
    float update(std::span<const float> target, std::span<const float> input);
    // The prediction of the next target from the remembered inputs.
    std::vector<float> predict() const;
    void reset();
    float last() const { return last_; }

    void insertTargets(size_t at, size_t count);
    void removeTargets(std::span<const size_t> removed);
    void insertInputs(size_t at, size_t count);
    void removeInputs(std::span<const size_t> removed);

    // Like critic: serialize(); readSpec() then deserialize().
    void serialize(std::ostream& os) const;
    static CuriositySpec readSpec(std::istream& is);
    void deserialize(std::istream& is, bool fullState);

private:
    CuriositySpec spec_;
    size_t input_count_ = 0;
    std::vector<float> weights_;  // targets x inputs, row-major
    std::vector<float> bias_;     // per target
    std::vector<float> previous_;
    bool has_previous_ = false;
    float last_ = 0.0f;
};

} // namespace exr
