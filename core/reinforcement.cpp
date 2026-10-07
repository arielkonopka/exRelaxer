#include "reinforcement.hpp"
#include "binary_io.hpp"
#include <algorithm>
#include <cmath>
#include <istream>
#include <ostream>
#include <stdexcept>

namespace exr {
namespace {

constexpr std::uint64_t max_count = std::uint64_t{1} << 26;

void writeFloats(std::ostream& os, std::span<const float> v)
{
    binary_io::write(os, static_cast<std::uint64_t>(v.size()));
    binary_io::write(os, v);
}

std::vector<float> readFloats(std::istream& is)
{
    const auto count = binary_io::read<std::uint64_t>(is);
    if (!is || count > max_count)
        throw std::runtime_error("network::load: implausible reinforcement state size");
    std::vector<float> v(static_cast<size_t>(count));
    binary_io::read(is, std::span<float>(v));
    if (!is)
        throw std::runtime_error("network::load: unexpected end of data");
    return v;
}

void writeIds(std::ostream& os, const std::vector<size_t>& ids)
{
    binary_io::write(os, static_cast<std::uint64_t>(ids.size()));
    for (size_t id : ids)
        binary_io::write(os, static_cast<std::uint64_t>(id));
}

std::vector<size_t> readIds(std::istream& is)
{
    const auto count = binary_io::read<std::uint64_t>(is);
    if (!is || count > max_count)
        throw std::runtime_error("network::load: implausible layer list");
    std::vector<size_t> ids(static_cast<size_t>(count));
    for (size_t& id : ids)
        id = static_cast<size_t>(binary_io::read<std::uint64_t>(is));
    return ids;
}

void writeNames(std::ostream& os, const std::vector<std::string>& names)
{
    binary_io::write(os, static_cast<std::uint64_t>(names.size()));
    for (const std::string& name : names) {
        binary_io::write(os, static_cast<std::uint64_t>(name.size()));
        binary_io::writeChars(os, name);
    }
}

std::vector<std::string> readNames(std::istream& is)
{
    const auto count = binary_io::read<std::uint64_t>(is);
    if (!is || count > max_count)
        throw std::runtime_error("network::load: implausible input list");
    std::vector<std::string> names(static_cast<size_t>(count));
    for (std::string& name : names) {
        const auto length = binary_io::read<std::uint64_t>(is);
        if (!is || length > 4096)
            throw std::runtime_error("network::load: implausible input name");
        name.assign(static_cast<size_t>(length), '\0');
        binary_io::readChars(is, name);
    }
    return names;
}

float squaredNorm(std::span<const float> v)
{
    float s = 0.0f;
    for (float x : v)
        s += x * x;
    return s;
}

template <typename T>
void eraseSorted(std::vector<T>& v, std::span<const size_t> removed)
{
    size_t next = 0, out = 0;
    for (size_t i = 0; i < v.size(); ++i) {
        if (next < removed.size() && removed[next] == i) {
            ++next;
            continue;
        }
        v[out++] = v[i];
    }
    v.resize(out);
}

} // namespace

void CriticSpec::validate() const
{
    if (!(gamma >= 0.0f && gamma <= 1.0f))
        throw std::invalid_argument("CriticSpec: gamma must be in [0, 1]");
    if (!(lambda >= 0.0f && lambda <= 1.0f))
        throw std::invalid_argument("CriticSpec: lambda must be in [0, 1]");
    if (!(rate > 0.0f) || !std::isfinite(rate))
        throw std::invalid_argument("CriticSpec: rate must be > 0");
    if (layers.empty() && inputs.empty())
        throw std::invalid_argument("CriticSpec: name at least one feature layer or input");
}

void CuriositySpec::validate() const
{
    if (!(rate > 0.0f) || !std::isfinite(rate))
        throw std::invalid_argument("CuriositySpec: rate must be > 0");
    if (!(scale >= 0.0f) || !std::isfinite(scale))
        throw std::invalid_argument("CuriositySpec: scale must be >= 0");
    if (predictLayers.empty() && predictInputs.empty())
        throw std::invalid_argument("CuriositySpec: name at least one layer or input to predict");
    if (fromLayers.empty() && fromInputs.empty())
        throw std::invalid_argument("CuriositySpec: name at least one layer or input to predict from");
}

// --- critic -------------------------------------------------------------------

critic::critic(const CriticSpec& spec, size_t features)
    : spec_(spec), weights_(features, 0.0f), traces_(features, 0.0f)
{
    spec_.validate();
}

float critic::value(std::span<const float> phi) const
{
    if (phi.size() != weights_.size())
        throw std::invalid_argument("critic: " + std::to_string(weights_.size()) + " features, got " +
                                    std::to_string(phi.size()));
    float v = bias_;
    for (size_t j = 0; j < phi.size(); ++j)
        v += weights_[j] * phi[j];
    return v;
}

float critic::update(std::span<const float> phi, float reward, bool terminal)
{
    const float now = terminal ? (static_cast<void>(value(phi)), 0.0f) : value(phi);  // value() checks the size
    if (!has_previous_) {
        last_error_ = 0.0f;
        if (terminal) {
            reset();
        } else {
            previous_.assign(phi.begin(), phi.end());
            has_previous_ = true;
        }
        return 0.0f;
    }
    const float delta = reward + spec_.gamma * now - value(previous_);
    const float decay = spec_.gamma * spec_.lambda;
    for (size_t j = 0; j < traces_.size(); ++j)
        traces_[j] = decay * traces_[j] + previous_[j];
    bias_trace_ = decay * bias_trace_ + 1.0f;
    const float step = spec_.rate / (1.0f + squaredNorm(previous_)) * delta;
    if (std::isfinite(step)) {
        for (size_t j = 0; j < weights_.size(); ++j)
            weights_[j] += step * traces_[j];
        bias_ += step * bias_trace_;
    }
    last_error_ = delta;
    if (terminal)
        reset();
    else
        previous_.assign(phi.begin(), phi.end());
    return delta;
}

void critic::reset()
{
    std::fill(traces_.begin(), traces_.end(), 0.0f);
    bias_trace_ = 0.0f;
    previous_.clear();
    has_previous_ = false;
}

void critic::insertFeatures(size_t at, size_t count)
{
    at = std::min(at, weights_.size());
    const auto pos = static_cast<std::ptrdiff_t>(at);
    weights_.insert(weights_.begin() + pos, count, 0.0f);
    traces_.insert(traces_.begin() + pos, count, 0.0f);
    if (has_previous_)
        previous_.insert(previous_.begin() + pos, count, 0.0f);
}

void critic::removeFeatures(std::span<const size_t> removed)
{
    eraseSorted(weights_, removed);
    eraseSorted(traces_, removed);
    if (has_previous_)
        eraseSorted(previous_, removed);
}

void critic::serialize(std::ostream& os) const
{
    writeIds(os, spec_.layers);
    writeNames(os, spec_.inputs);
    binary_io::write(os, spec_.gamma);
    binary_io::write(os, spec_.lambda);
    binary_io::write(os, spec_.rate);
    writeFloats(os, weights_);
    binary_io::write(os, bias_);
    writeFloats(os, traces_);
    binary_io::write(os, bias_trace_);
    binary_io::write<std::uint8_t>(os, has_previous_);
    writeFloats(os, previous_);
    binary_io::write(os, last_error_);
}

CriticSpec critic::readSpec(std::istream& is)
{
    CriticSpec spec;
    spec.layers = readIds(is);
    spec.inputs = readNames(is);
    spec.gamma = binary_io::read<float>(is);
    spec.lambda = binary_io::read<float>(is);
    spec.rate = binary_io::read<float>(is);
    if (!is)
        throw std::runtime_error("network::load: unexpected end of data in the critic");
    try {
        spec.validate();
    } catch (const std::invalid_argument& e) {
        throw std::runtime_error(std::string("network::load: ") + e.what());
    }
    return spec;
}

void critic::deserialize(std::istream& is, bool fullState)
{
    std::vector<float> weights = readFloats(is);
    const float bias = binary_io::read<float>(is);
    std::vector<float> traces = readFloats(is);
    const float bias_trace = binary_io::read<float>(is);
    const bool has_previous = binary_io::read<std::uint8_t>(is) != 0;
    std::vector<float> previous = readFloats(is);
    const float last_error = binary_io::read<float>(is);
    if (!is)
        throw std::runtime_error("network::load: unexpected end of data in the critic");
    if (weights.size() != weights_.size() || traces.size() != weights_.size() ||
        (has_previous && previous.size() != weights_.size()))
        throw std::runtime_error("network::load: the critic does not match its feature layers");
    weights_ = std::move(weights);
    bias_ = bias;
    if (fullState) {
        traces_ = std::move(traces);
        bias_trace_ = bias_trace;
        has_previous_ = has_previous;
        previous_ = std::move(previous);
        last_error_ = last_error;
    }
}

// --- curiosity ----------------------------------------------------------------

curiosity::curiosity(const CuriositySpec& spec, size_t targets, size_t inputs)
    : spec_(spec), input_count_(inputs), weights_(targets * inputs, 0.0f), bias_(targets, 0.0f)
{
    spec_.validate();
}

std::vector<float> curiosity::predict() const
{
    std::vector<float> out(bias_);
    if (!has_previous_)
        return out;
    for (size_t t = 0; t < out.size(); ++t) {
        const float* w = weights_.data() + t * input_count_;
        float s = 0.0f;
        for (size_t j = 0; j < input_count_; ++j)
            s += w[j] * previous_[j];
        out[t] += s;
    }
    return out;
}

float curiosity::update(std::span<const float> target, std::span<const float> input)
{
    if (target.size() != bias_.size() || input.size() != input_count_)
        throw std::invalid_argument("curiosity: expected " + std::to_string(bias_.size()) + " targets and " +
                                    std::to_string(input_count_) + " inputs, got " + std::to_string(target.size()) +
                                    " and " + std::to_string(input.size()));
    float reward = 0.0f;
    if (has_previous_) {
        const std::vector<float> prediction = predict();
        const float step = spec_.rate / (1.0f + squaredNorm(previous_));
        float squared = 0.0f;
        for (size_t t = 0; t < bias_.size(); ++t) {
            const float error = target[t] - prediction[t];
            squared += error * error;
            const float e = step * error;
            float* w = weights_.data() + t * input_count_;
            for (size_t j = 0; j < input_count_; ++j)
                w[j] += e * previous_[j];
            bias_[t] += e;
        }
        reward = bias_.empty() ? 0.0f : spec_.scale * squared / static_cast<float>(bias_.size());
    }
    previous_.assign(input.begin(), input.end());
    has_previous_ = true;
    last_ = reward;
    return reward;
}

void curiosity::reset()
{
    previous_.clear();
    has_previous_ = false;
}

void curiosity::insertTargets(size_t at, size_t count)
{
    at = std::min(at, bias_.size());
    weights_.insert(weights_.begin() + static_cast<std::ptrdiff_t>(at * input_count_), count * input_count_, 0.0f);
    bias_.insert(bias_.begin() + static_cast<std::ptrdiff_t>(at), count, 0.0f);
}

void curiosity::removeTargets(std::span<const size_t> removed)
{
    std::vector<size_t> entries;
    for (size_t t : removed)
        for (size_t j = 0; j < input_count_; ++j)
            entries.push_back(t * input_count_ + j);
    eraseSorted(weights_, entries);
    eraseSorted(bias_, removed);
}

void curiosity::insertInputs(size_t at, size_t count)
{
    at = std::min(at, input_count_);
    const size_t inputs = input_count_ + count;
    std::vector<float> weights(bias_.size() * inputs, 0.0f);
    for (size_t t = 0; t < bias_.size(); ++t)
        for (size_t j = 0; j < input_count_; ++j)
            weights[t * inputs + (j < at ? j : j + count)] = weights_[t * input_count_ + j];
    weights_ = std::move(weights);
    input_count_ = inputs;
    if (has_previous_)
        previous_.insert(previous_.begin() + static_cast<std::ptrdiff_t>(at), count, 0.0f);
}

void curiosity::removeInputs(std::span<const size_t> removed)
{
    std::vector<size_t> entries;
    for (size_t t = 0; t < bias_.size(); ++t)
        for (size_t j : removed)
            entries.push_back(t * input_count_ + j);
    eraseSorted(weights_, entries);
    input_count_ -= removed.size();
    if (has_previous_)
        eraseSorted(previous_, removed);
}

void curiosity::serialize(std::ostream& os) const
{
    writeIds(os, spec_.predictLayers);
    writeNames(os, spec_.predictInputs);
    writeIds(os, spec_.fromLayers);
    writeNames(os, spec_.fromInputs);
    binary_io::write(os, spec_.rate);
    binary_io::write(os, spec_.scale);
    binary_io::write(os, static_cast<std::uint64_t>(input_count_));
    writeFloats(os, weights_);
    writeFloats(os, bias_);
    binary_io::write<std::uint8_t>(os, has_previous_);
    writeFloats(os, previous_);
    binary_io::write(os, last_);
}

CuriositySpec curiosity::readSpec(std::istream& is)
{
    CuriositySpec spec;
    spec.predictLayers = readIds(is);
    spec.predictInputs = readNames(is);
    spec.fromLayers = readIds(is);
    spec.fromInputs = readNames(is);
    spec.rate = binary_io::read<float>(is);
    spec.scale = binary_io::read<float>(is);
    if (!is)
        throw std::runtime_error("network::load: unexpected end of data in the curiosity model");
    try {
        spec.validate();
    } catch (const std::invalid_argument& e) {
        throw std::runtime_error(std::string("network::load: ") + e.what());
    }
    return spec;
}

void curiosity::deserialize(std::istream& is, bool fullState)
{
    const auto inputs = binary_io::read<std::uint64_t>(is);
    std::vector<float> weights = readFloats(is);
    std::vector<float> bias = readFloats(is);
    const bool has_previous = binary_io::read<std::uint8_t>(is) != 0;
    std::vector<float> previous = readFloats(is);
    const float last = binary_io::read<float>(is);
    if (!is)
        throw std::runtime_error("network::load: unexpected end of data in the curiosity model");
    if (inputs != input_count_ || bias.size() != bias_.size() || weights.size() != weights_.size() ||
        (has_previous && previous.size() != input_count_))
        throw std::runtime_error("network::load: the curiosity model does not match its layers");
    weights_ = std::move(weights);
    bias_ = std::move(bias);
    if (fullState) {
        has_previous_ = has_previous;
        previous_ = std::move(previous);
        last_ = last;
    }
}

} // namespace exr
