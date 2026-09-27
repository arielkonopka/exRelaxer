#include "network.hpp"
#include "binary_io.hpp"
#include <algorithm>
#include <cmath>
#include <array>
#include <functional>
#include <iomanip>
#include <sstream>
#include <queue>
#include <stdexcept>
#include <string_view>

namespace exr {

network::network(const layer_factory& factory)
    : factory(factory)
{
}

void network::checkId(LayerId id) const
{
    if (id >= this->nodes.size())
        throw std::out_of_range("network: no layer with id " + std::to_string(id));
}

network::LayerId network::addLayer(const std::string& name, const LayerSpec& spec)
{
    if (name.empty())
        throw std::invalid_argument("network: layer name must not be empty");
    for (const Node& node : this->nodes)
        if (node.name == name)
            throw std::invalid_argument("network: duplicate layer name '" + name + "'");

    std::unique_ptr<layer> impl = this->factory.create(spec);
    if (spec.learningRule != LearningRule{}) {
        if (!impl->hasNeurons())
            throw std::invalid_argument("network: layer '" + name + "' has no neurons to take a learning rule");
        dynamic_cast<neuron_layer&>(*impl).setLearningRule(spec.learningRule);
    }
    if (spec.gate != 0.0f) {
        if (!impl->hasNeurons())
            throw std::invalid_argument("network: layer '" + name + "' has no neurons to take a gate");
        dynamic_cast<neuron_layer&>(*impl).setGate(spec.gate);
    }
    if (spec.habituationRule != Habituation{}) {
        if (!impl->hasNeurons())
            throw std::invalid_argument("network: layer '" + name + "' has no neurons to take a habituation rule");
        dynamic_cast<neuron_layer&>(*impl).setHabituationRule(spec.habituationRule);
    }
    if (spec.thresholdGrowth != ThresholdGrowth{}) {
        if (!impl->hasNeurons())
            throw std::invalid_argument("network: layer '" + name + "' has no neurons to take a threshold growth rule");
        dynamic_cast<neuron_layer&>(*impl).setThresholdGrowth(spec.thresholdGrowth);
    }
    if (spec.rectify) {
        if (!impl->hasNeurons())
            throw std::invalid_argument("network: layer '" + name + "' has no neurons to rectify");
        dynamic_cast<neuron_layer&>(*impl).setRectified(true);
    }
    this->nodes.push_back({name, spec, std::move(impl)});
    this->ops_.push_back({OpKind::AddLayer, this->nodes.size() - 1, 0, 0});
    this->orderValid = false;
    return this->nodes.size() - 1;
}

void network::connect(LayerId from, LayerId to)
{
    checkId(from);
    checkId(to);
    for (const Edge& e : this->edges_)
        if (e.kind == EdgeKind::Forward && e.from == from && e.to == to)
            throw std::logic_error("network: '" + this->nodes[from].name + "' -> '" +
                                   this->nodes[to].name + "' is already connected");

    this->nodes[to].impl->join(*this->nodes[from].impl);
    this->edges_.push_back({from, to, EdgeKind::Forward, 0});
    this->ops_.push_back({OpKind::Connect, from, to, 0});
    this->orderValid = false;
}

void network::addFeedback(LayerId from, LayerId to, size_t width)
{
    checkId(from);
    checkId(to);
    this->nodes[to].impl->addNeurons(width, *this->nodes[from].impl);
    this->edges_.push_back({from, to, EdgeKind::Feedback, width});
    this->ops_.push_back({OpKind::Feedback, from, to, width});
}

size_t network::addInputs(LayerId target, const Shape& image, const std::string& name)
{
    return addSource(target, image, name);
}

size_t network::addInputs(LayerId target, size_t count, const std::string& name)
{
    return addSource(target, Shape::flat(count), name);
}

size_t network::addSource(LayerId target, const Shape& shape, const std::string& name)
{
    checkId(target);
    const size_t count = shape.size();
    if (count == 0)
        throw std::invalid_argument("network: addInputs needs at least one input");
    if (!name.empty() && std::ranges::any_of(this->sources_, [&](const InputSource& s) { return s.name == name; }))
        throw std::invalid_argument("network: an input source named '" + name + "' already exists");

    // The layer keeps a range of inputs_ (the vector object, not its data),
    // so later sensors may reallocate it.
    const size_t first = this->inputs_.size();
    this->inputs_.resize(first + count, 0.0f);
    try {
        this->nodes[target].impl->attachInputs(InputRange(this->inputs_, first, count));
    } catch (...) {
        this->inputs_.resize(first);
        throw;
    }
    this->sources_.push_back({name, first, shape, {target}});
    this->ops_.push_back({OpKind::Inputs, target, this->sources_.size() - 1, count});
    return first;
}

size_t network::findSource(const std::string& name) const
{
    for (size_t i = 0; i < this->sources_.size(); ++i)
        if (!name.empty() && this->sources_[i].name == name)
            return i;
    throw std::out_of_range("network: no input source named '" + name + "'");
}

const network::InputSource& network::inputSource(const std::string& name) const
{
    return this->sources_[findSource(name)];
}

std::span<const float> network::inputs(const std::string& name) const
{
    const InputSource& source = inputSource(name);
    return std::span<const float>(this->inputs_).subspan(source.first, source.size());
}

void network::connectInputs(const std::string& name, LayerId target)
{
    checkId(target);
    const size_t index = findSource(name);
    InputSource& source = this->sources_[index];
    if (std::ranges::find(source.targets, target) != source.targets.end())
        throw std::invalid_argument("network: input source '" + name + "' already feeds layer '" +
                                    this->nodes[target].name + "'");
    this->nodes[target].impl->attachInputs(InputRange(this->inputs_, source.first, source.size()));
    source.targets.push_back(target);
    this->ops_.push_back({OpKind::ConnectInputs, target, index, 0});
}

void network::addOutput(LayerId id)
{
    checkId(id);
    this->outputs_.push_back(id);
}

void network::freeze(LayerId id)
{
    checkId(id);
    this->nodes[id].spec.frozen = true;
}

void network::unfreeze(LayerId id)
{
    checkId(id);
    this->nodes[id].spec.frozen = false;
}

bool network::isFrozen(LayerId id) const
{
    checkId(id);
    return this->nodes[id].spec.frozen;
}

neuron_layer& network::neuronLayer(LayerId id)
{
    checkId(id);
    layer& target = *this->nodes[id].impl;
    if (!target.hasNeurons())
        throw std::invalid_argument("network: layer '" + this->nodes[id].name + "' has no neurons");
    return dynamic_cast<neuron_layer&>(target);
}

void network::setRecoveryJitter(LayerId id, const Jitter& jitter)
{
    neuronLayer(id).setRecoveryJitter(jitter);
    this->nodes[id].spec.recoveryJitter = jitter;
}

void network::setLearningJitter(LayerId id, const Jitter& jitter)
{
    neuronLayer(id).setLearningJitter(jitter);
    this->nodes[id].spec.learningJitter = jitter;
}

void network::setAlphaJitter(LayerId id, const Jitter& jitter)
{
    neuronLayer(id).setAlphaJitter(jitter);
    this->nodes[id].spec.alphaJitter = jitter;
}

void network::setLearningRule(LayerId id, const LearningRule& rule)
{
    neuronLayer(id).setLearningRule(rule);
    this->nodes[id].spec.learningRule = rule;
}

void network::setUpdateOrder(std::vector<LayerId> order)
{
    std::vector<bool> seen(this->nodes.size(), false);
    for (LayerId id : order)
    {
        checkId(id);
        if (seen[id])
            throw std::invalid_argument("network: layer '" + this->nodes[id].name +
                                        "' appears twice in the update order");
        seen[id] = true;
    }
    if (order.size() != this->nodes.size())
        throw std::invalid_argument("network: the update order must list every layer exactly once");

    this->order_ = std::move(order);
    this->customOrder = true;
    this->orderValid = true;
}

void network::useDefaultUpdateOrder()
{
    this->customOrder = false;
    this->orderValid = false;
}

std::vector<network::LayerId> network::topologicalOrder() const
{
    // Kahn's algorithm over forward edges. Among layers that are ready, the
    // one created first goes first, so the order is stable and predictable.
    const size_t n = this->nodes.size();
    std::vector<size_t> indegree(n, 0);
    std::vector<std::vector<LayerId>> next(n);
    for (const Edge& e : this->edges_)
    {
        if (e.kind != EdgeKind::Forward || e.from == e.to)
            continue;
        next[e.from].push_back(e.to);
        ++indegree[e.to];
    }

    std::priority_queue<LayerId, std::vector<LayerId>, std::greater<>> ready;
    for (LayerId id = 0; id < n; ++id)
        if (indegree[id] == 0)
            ready.push(id);

    std::vector<LayerId> order;
    order.reserve(n);
    while (!ready.empty())
    {
        const LayerId id = ready.top();
        ready.pop();
        order.push_back(id);
        for (LayerId to : next[id])
            if (--indegree[to] == 0)
                ready.push(to);
    }

    if (order.size() != n)
        throw std::logic_error("network: forward connections form a cycle; use addFeedback for "
                               "recurrent links or set an explicit update order");
    return order;
}

const std::vector<network::LayerId>& network::updateOrder() const
{
    if (this->customOrder)
    {
        if (this->order_.size() != this->nodes.size())
            throw std::logic_error("network: layers were added after setUpdateOrder(); set the order again");
        return this->order_;
    }
    if (!this->orderValid)
    {
        this->order_ = topologicalOrder();
        this->orderValid = true;
    }
    return this->order_;
}

void network::setInput(size_t index, float value)
{
    if (index >= this->inputs_.size())
        throw std::out_of_range("network: no input with index " + std::to_string(index));
    this->inputs_[index] = value;
}

void network::setInputs(std::span<const float> values)
{
    if (values.size() != this->inputs_.size())
        throw std::invalid_argument("network: expected " + std::to_string(this->inputs_.size()) +
                                    " input values, got " + std::to_string(values.size()));
    std::copy(values.begin(), values.end(), this->inputs_.begin());
}

void network::setInputs(std::initializer_list<float> values)
{
    setInputs(std::span<const float>(values.begin(), values.size()));
}

void network::setInputs(const std::string& name, std::span<const float> values)
{
    const InputSource& source = inputSource(name);
    if (values.size() != source.size())
        throw std::invalid_argument("network: input source '" + name + "' has " + std::to_string(source.size()) +
                                    " sensors, got " + std::to_string(values.size()) + " values");
    std::ranges::copy(values, this->inputs_.begin() + static_cast<std::ptrdiff_t>(source.first));
}

void network::step()
{
    for (LayerId id : updateOrder())
        this->nodes[id].impl->forward();
}

void network::applyReward(float reward, float learningRate)
{
    for (Node& node : this->nodes)
        if (!node.spec.frozen)
            node.impl->applyReward(reward, learningRate);
}

void network::applyError(std::span<const float> errors, float learningRate)
{
    size_t total = 0;
    for (LayerId id : this->outputs_)
        total += this->nodes[id].impl->size();
    if (errors.size() != total)
        throw std::invalid_argument("network::applyError: " + std::to_string(total) + " outputs, got " +
                                    std::to_string(errors.size()) + " errors");
    float squared = 0.0f;
    for (float e : errors)
        squared += e * e;

    for (LayerId id = 0; id < this->nodes.size(); ++id) {
        Node& node = this->nodes[id];
        if (node.spec.frozen || !node.impl->learns() || !node.impl->hasNeurons())
            continue;
        auto& target = dynamic_cast<neuron_layer&>(*node.impl);
        const LearningRule& rule = target.learningRule();
        if (rule.unsupervised()) {
            target.applyReward(0.0f, learningRate);
            continue;
        }
        if (rule.type == LearningRuleType::Perturbation) {
            target.applyReward(-0.5f * squared, learningRate);
            continue;
        }
        // The layer's own errors when it is an output layer.
        size_t offset = 0;
        bool is_output = false;
        for (LayerId out : this->outputs_) {
            if (out == id) {
                is_output = true;
                break;
            }
            offset += this->nodes[out].impl->size();
        }
        if (is_output)
            target.applyModulators(errors.subspan(offset, target.size()), learningRate);
        else if (rule.type == LearningRuleType::FeedbackAlignment)
            target.applyFeedback(errors, learningRate);
    }
}

std::vector<float> network::outputs() const
{
    std::vector<float> values;
    for (LayerId id : this->outputs_) {
        const std::span<const float> out = this->nodes[id].impl->output();
        values.insert(values.end(), out.begin(), out.end());
    }
    return values;
}

layer& network::getLayer(LayerId id)
{
    checkId(id);
    return *this->nodes[id].impl;
}

const layer& network::getLayer(LayerId id) const
{
    checkId(id);
    return *this->nodes[id].impl;
}

network::LayerId network::findLayer(const std::string& name) const
{
    for (LayerId id = 0; id < this->nodes.size(); ++id)
        if (this->nodes[id].name == name)
            return id;
    throw std::out_of_range("network: no layer named '" + name + "'");
}

const std::string& network::layerName(LayerId id) const
{
    checkId(id);
    return this->nodes[id].name;
}

const LayerSpec& network::layerSpec(LayerId id) const
{
    checkId(id);
    return this->nodes[id].spec;
}

namespace {

// "0.9" without jitter; "0.9 U+-0.05", "0.95 N(sd 0.02) in [0.8, 0.99]",
// "2 U+-50%", "0.9 U+-50% of 1-r" with. `scaleName` names the scale of a
// relative spread ("" = the value itself).
std::string describeJitter(float defaultValue, const Jitter& jitter, std::string_view scaleName = "")
{
    std::ostringstream text;
    text << jitter.mean.value_or(defaultValue);
    if (!jitter.enabled())
        return text.str();
    std::ostringstream spread;
    if (jitter.relative) {
        spread << jitter.spread * 100.0f << "%";
        if (!scaleName.empty()) spread << " of " << scaleName;
    } else {
        spread << jitter.spread;
    }
    if (jitter.distribution == Jitter::Distribution::Uniform)
        text << " U+-" << spread.str();
    else
        text << " N(sd " << spread.str() << ")";
    if (std::isfinite(jitter.min) || std::isfinite(jitter.max))
        text << " in [" << jitter.min << ", " << jitter.max << "]";
    return text.str();
}

} // namespace

namespace {

// A float as the shortest text that reads back as the same value ("0.2").
std::string describeNumber(float value)
{
    std::ostringstream text;
    text << value;
    return text.str();
}

// "channels x height x width" for spatial layers, "-" for flat ones.
std::string describeShape(const Shape& shape)
{
    if (shape.height == 1 && shape.width == 1)
        return "-";
    return std::to_string(shape.channels) + "x" + std::to_string(shape.height) + "x" + std::to_string(shape.width);
}

} // namespace

void network::describe(std::ostream& os) const
{
    size_t name_width = 5;
    for (const Node& node : this->nodes)
        name_width = std::max(name_width, node.name.size());
    const auto name_col = static_cast<int>(name_width) + 2;
    size_t recovery_width = 8;
    for (const Node& node : this->nodes)
        recovery_width = std::max(recovery_width, describeJitter(recovery_factor, node.spec.recoveryJitter, "1-r").size());
    const auto recovery_col = static_cast<int>(recovery_width) + 2;
    size_t learning_width = 13;
    for (const Node& node : this->nodes)
        learning_width = std::max(learning_width, describeJitter(default_learning_gain, node.spec.learningJitter).size());
    const auto learning_col = static_cast<int>(learning_width) + 2;
    size_t alpha_width = 5;
    for (const Node& node : this->nodes)
        alpha_width = std::max(alpha_width, describeJitter(default_alpha, node.spec.alphaJitter).size());
    const auto alpha_col = static_cast<int>(alpha_width) + 2;

    os << "  layers (" << this->nodes.size() << "):\n"
       << "    " << std::left << std::setw(4) << "id" << std::setw(name_col) << "name"
       << std::setw(9) << "outputs" << std::setw(14) << "shape" << std::setw(6) << "hab" << std::setw(6) << "E-R" << std::setw(8) << "learns"
       << std::setw(recovery_col) << "recovery" << std::setw(learning_col) << "learning gain" << std::setw(alpha_col) << "alpha" << "rule\n";
    for (LayerId id = 0; id < this->nodes.size(); ++id)
    {
        const Node& node = this->nodes[id];
        os << "    " << std::setw(4) << id << std::setw(name_col) << node.name
           << std::setw(9) << node.impl->size() << std::setw(14) << describeShape(node.impl->shape())
           << std::setw(6) << (node.impl->hasNeurons() && node.spec.hasHabituation ? "on" : "-")
           << std::setw(6) << (!node.impl->hasNeurons() ? std::string("-")
                               : node.spec.hasER   ? std::string("on")
                               : node.spec.rectify ? (node.spec.gate > 0.0f ? ">" + describeNumber(node.spec.gate)
                                                                            : std::string("relu"))
                               : node.spec.gate > 0.0f ? "=" + describeNumber(node.spec.gate)
                                                       : std::string("-"))
           << std::setw(8) << (!node.impl->learns() ? "-" : node.spec.frozen ? "frozen" : "yes");
        if (!node.impl->hasNeurons()) {
            os << "\n";  // no neurons: no per-neuron dynamics
            continue;
        }
        os << std::setw(recovery_col) << describeJitter(recovery_factor, node.spec.recoveryJitter, "1-r")
           << std::setw(learning_col) << describeJitter(default_learning_gain, node.spec.learningJitter)
           << std::setw(alpha_col) << describeJitter(default_alpha, node.spec.alphaJitter)
           << (node.impl->learns() ? describeLearningRule(node.spec.learningRule) : "-") << "\n";
    }
    os << std::right;

    for (const InputSource& source : this->sources_) {
        os << "  inputs";
        if (!source.name.empty())
            os << " '" << source.name << "'";
        os << ": " << source.size();
        if (source.shape.height > 1 || source.shape.width > 1)
            os << " (" << describeShape(source.shape) << ")";
        os << " ->";
        for (size_t i = 0; i < source.targets.size(); ++i)
            os << (i ? ", " : " ") << this->nodes[source.targets[i]].name;
        os << "\n";
    }

    if (!this->edges_.empty())
    {
        os << "  edges:\n";
        for (const Edge& e : this->edges_)
        {
            os << "    " << this->nodes[e.from].name << " -> " << this->nodes[e.to].name;
            if (e.kind == EdgeKind::Feedback)
                os << "  (feedback, " << e.width << " new neurons)";
            os << "\n";
        }
    }

    os << "  outputs:";
    for (LayerId id : this->outputs_)
        os << " " << this->nodes[id].name;
    os << "\n  update order" << (this->customOrder ? " (custom):" : ":");
    try
    {
        for (LayerId id : updateOrder())
            os << " " << this->nodes[id].name;
    }
    catch (const std::logic_error&)
    {
        os << " invalid (forward edges form a cycle)";
    }
    os << "\n";
}

// --- Serialization ----------------------------------------------------------
namespace {

constexpr char NETWORK_MAGIC[4] = {'E', 'X', 'R', 'N'};
// Version history:
//   1  construction history, outputs, order, inputs, layer data
//   2  + each layer's frozen flag
//   3  + per-layer jitter (two floats) and each neuron's recovery and
//        learning gain (neuron format 2)
//   4  jitter as a full distribution (Jitter)
//   5  + relative-spread flag per jitter
//   6  + alpha jitter per layer
//   7  + spatial parameters per layer (window, pooling mode, retina)
//   8  + audio parameters per layer (cochlea)
//   9  + learning rule per layer; layers of neurons append the rule's state
//        (neuron format 3)
//  10  + cochlea channels, resize and disparity parameters per layer; named
//        input sources (name and shape per addInputs) and connectInputs
//  11  + fixed firing threshold (gate) per layer
//  12  + rectification (ReLU) per layer
//  13  + habituation rule (steps, tolerance, decay) per layer
//  14  + E-R threshold growth rule (rule, amount) per layer
// Older versions load as weights only (see network::load).
constexpr std::uint32_t NETWORK_FORMAT_VERSION = 14;
// Files from this version on carry the full state; older ones load as
// weights only. (Versions 7, 8, 10 and 11 only added parameters whose defaults
// are right for older files.)
constexpr std::uint32_t FIRST_FULL_STATE_VERSION = 6;

// Upper bounds for counts read from a stream, so corrupt data fails with an
// error instead of an enormous allocation.
constexpr std::uint64_t MAX_NAME_LENGTH = 4096;
constexpr std::uint64_t MAX_COUNT = std::uint64_t{1} << 26;

template <typename T>
void writeValue(std::ostream& os, T value)
{
    binary_io::write(os, value);
}

template <typename T>
T readValue(std::istream& is)
{
    const T value = binary_io::read<T>(is);
    if (!is)
        throw std::runtime_error("network::load: unexpected end of data");
    return value;
}

void writeCount(std::ostream& os, size_t value) { writeValue<std::uint64_t>(os, value); }

void writeJitter(std::ostream& os, const Jitter& jitter)
{
    writeValue(os, static_cast<std::uint8_t>(jitter.distribution));
    writeValue(os, jitter.spread);
    writeValue<std::uint8_t>(os, jitter.relative);
    writeValue<std::uint8_t>(os, jitter.mean.has_value());
    writeValue(os, jitter.mean.value_or(0.0f));
    writeValue(os, jitter.min);
    writeValue(os, jitter.max);
}

Jitter readJitter(std::istream& is, std::uint32_t version);

void writeSpatial(std::ostream& os, const LayerSpec& spec)
{
    for (size_t v : {spec.window.kernelHeight, spec.window.kernelWidth, spec.window.strideY, spec.window.strideX,
                     spec.window.padY, spec.window.padX})
        writeCount(os, v);
    writeValue(os, static_cast<std::uint8_t>(spec.pool));
    writeCount(os, spec.retina.input.channels);
    writeCount(os, spec.retina.input.height);
    writeCount(os, spec.retina.input.width);
    writeValue(os, static_cast<std::uint8_t>(spec.retina.sampling));
    writeValue(os, spec.retina.spacing);
    writeValue(os, spec.retina.radius);
}

void readSpatial(std::istream& is, LayerSpec& spec);

void writeAudio(std::ostream& os, const LayerSpec& spec)
{
    const CochleaSpec& c = spec.cochlea;
    writeValue(os, c.sampleRate);
    writeCount(os, c.hop);
    writeCount(os, c.window);
    writeCount(os, c.bands);
    writeValue(os, c.minFrequency);
    writeValue(os, c.maxFrequency);
    writeValue(os, static_cast<std::uint8_t>(c.scale));
    writeValue(os, static_cast<std::uint8_t>(c.compression));
    writeValue(os, c.gain);
}

void readAudio(std::istream& is, LayerSpec& spec);

size_t readCount(std::istream& is, std::string_view what);

// Version 10: cochlea channels, resize and disparity.
void writeMultimodal(std::ostream& os, const LayerSpec& spec)
{
    writeCount(os, spec.cochlea.channels);
    writeCount(os, spec.resize.height);
    writeCount(os, spec.resize.width);
    writeValue(os, static_cast<std::uint8_t>(spec.resize.interpolation));
    writeValue<std::int32_t>(os, spec.disparity.minDisparity);
    writeValue<std::int32_t>(os, spec.disparity.maxDisparity);
    writeCount(os, spec.disparity.window);
    writeValue(os, static_cast<std::uint8_t>(spec.disparity.measure));
}

void readMultimodal(std::istream& is, LayerSpec& spec)
{
    spec.cochlea.channels = readCount(is, "cochlea channels");
    spec.resize.height = readCount(is, "resize height");
    spec.resize.width = readCount(is, "resize width");
    const auto interpolation = readValue<std::uint8_t>(is);
    if (interpolation > static_cast<std::uint8_t>(Interpolation::Area))
        throw std::runtime_error("network::load: unknown interpolation " + std::to_string(interpolation));
    spec.resize.interpolation = static_cast<Interpolation>(interpolation);
    spec.disparity.minDisparity = readValue<std::int32_t>(is);
    spec.disparity.maxDisparity = readValue<std::int32_t>(is);
    spec.disparity.window = readCount(is, "disparity window");
    const auto measure = readValue<std::uint8_t>(is);
    if (measure > static_cast<std::uint8_t>(DisparityMeasure::Normalized))
        throw std::runtime_error("network::load: unknown disparity measure " + std::to_string(measure));
    spec.disparity.measure = static_cast<DisparityMeasure>(measure);
}

void writeLearningRule(std::ostream& os, const LearningRule& rule)
{
    writeValue(os, static_cast<std::uint8_t>(rule.type));
    writeValue<std::uint8_t>(os, rule.bias);
    writeValue(os, rule.decay);
    writeValue(os, rule.trace);
    writeValue(os, rule.baseline);
    writeValue(os, rule.noise);
    writeValue(os, rule.bcmRate);
    writeValue(os, rule.winners);
}

LearningRule readLearningRule(std::istream& is)
{
    LearningRule rule;
    rule.type = static_cast<LearningRuleType>(readValue<std::uint8_t>(is));
    rule.bias = readValue<std::uint8_t>(is) != 0;
    rule.decay = readValue<float>(is);
    rule.trace = readValue<float>(is);
    rule.baseline = readValue<float>(is);
    rule.noise = readValue<float>(is);
    rule.bcmRate = readValue<float>(is);
    rule.winners = readValue<std::uint32_t>(is);
    try {
        rule.validate();
    } catch (const std::invalid_argument& e) {
        throw std::runtime_error(std::string("network::load: ") + e.what());
    }
    return rule;
}

size_t readCount(std::istream& is, std::string_view what)
{
    const auto value = readValue<std::uint64_t>(is);
    if (value > MAX_COUNT)
        throw std::runtime_error("network::load: implausible " + std::string(what) + " " + std::to_string(value));
    return static_cast<size_t>(value);
}

void readSpatial(std::istream& is, LayerSpec& spec)
{
    Window2D& w = spec.window;
    for (size_t& v : {std::ref(w.kernelHeight), std::ref(w.kernelWidth), std::ref(w.strideY), std::ref(w.strideX),
                      std::ref(w.padY), std::ref(w.padX)})
        v = readCount(is, "window size");
    const auto pool = readValue<std::uint8_t>(is);
    if (pool > static_cast<std::uint8_t>(PoolMode::Average))
        throw std::runtime_error("network::load: unknown pooling mode " + std::to_string(pool));
    spec.pool = static_cast<PoolMode>(pool);
    spec.retina.input.channels = readCount(is, "image channels");
    spec.retina.input.height = readCount(is, "image height");
    spec.retina.input.width = readCount(is, "image width");
    const auto sampling = readValue<std::uint8_t>(is);
    if (sampling > static_cast<std::uint8_t>(Sampling::Spiral))
        throw std::runtime_error("network::load: unknown retina sampling " + std::to_string(sampling));
    spec.retina.sampling = static_cast<Sampling>(sampling);
    spec.retina.spacing = readValue<float>(is);
    spec.retina.radius = readValue<float>(is);
}

void readAudio(std::istream& is, LayerSpec& spec)
{
    CochleaSpec& c = spec.cochlea;
    c.sampleRate = readValue<float>(is);
    c.hop = readCount(is, "cochlea hop");
    c.window = readCount(is, "cochlea window");
    c.bands = readCount(is, "cochlea bands");
    c.minFrequency = readValue<float>(is);
    c.maxFrequency = readValue<float>(is);
    const auto scale = readValue<std::uint8_t>(is);
    if (scale > static_cast<std::uint8_t>(FrequencyScale::Linear))
        throw std::runtime_error("network::load: unknown frequency scale " + std::to_string(scale));
    c.scale = static_cast<FrequencyScale>(scale);
    const auto compression = readValue<std::uint8_t>(is);
    if (compression > static_cast<std::uint8_t>(Compression::Linear))
        throw std::runtime_error("network::load: unknown compression " + std::to_string(compression));
    c.compression = static_cast<Compression>(compression);
    c.gain = readValue<float>(is);
}

Jitter readJitter(std::istream& is, std::uint32_t version)
{
    Jitter jitter;
    const auto distribution = readValue<std::uint8_t>(is);
    if (distribution > static_cast<std::uint8_t>(Jitter::Distribution::Normal))
        throw std::runtime_error("network::load: unknown jitter distribution " + std::to_string(distribution));
    jitter.distribution = static_cast<Jitter::Distribution>(distribution);
    jitter.spread = readValue<float>(is);
    if (version >= 5)
        jitter.relative = readValue<std::uint8_t>(is) != 0;
    const bool has_mean = readValue<std::uint8_t>(is) != 0;
    const float mean = readValue<float>(is);
    if (has_mean)
        jitter.mean = mean;
    jitter.min = readValue<float>(is);
    jitter.max = readValue<float>(is);
    return jitter;
}

} // namespace

void network::save(std::ostream& os) const
{
    binary_io::write(os, std::span<const char>(NETWORK_MAGIC));
    writeValue(os, NETWORK_FORMAT_VERSION);

    // 1. Construction history
    writeCount(os, this->ops_.size());
    for (const BuildOp& op : this->ops_)
    {
        writeValue(os, static_cast<std::uint8_t>(op.kind));
        switch (op.kind)
        {
        case OpKind::AddLayer: {
            const Node& node = this->nodes[op.a];
            writeCount(os, node.name.size());
            binary_io::writeChars(os, node.name);
            writeValue(os, static_cast<std::uint8_t>(node.spec.type));
            writeCount(os, node.spec.size);
            writeValue<std::uint8_t>(os, node.spec.hasHabituation);
            writeValue<std::uint8_t>(os, node.spec.hasER);
            writeValue<std::uint8_t>(os, node.spec.frozen);  // current state, not the one at creation
            writeJitter(os, node.spec.recoveryJitter);  // current settings, like frozen
            writeJitter(os, node.spec.learningJitter);
            writeJitter(os, node.spec.alphaJitter);
            writeSpatial(os, node.spec);
            writeAudio(os, node.spec);
            writeLearningRule(os, node.spec.learningRule);
            writeMultimodal(os, node.spec);
            writeValue(os, node.spec.gate);
            writeValue<std::uint8_t>(os, node.spec.rectify);
            writeValue<std::uint32_t>(os, node.spec.habituationRule.steps);
            writeValue(os, node.spec.habituationRule.tolerance);
            writeValue(os, node.spec.habituationRule.decay);
            writeValue(os, static_cast<std::uint8_t>(node.spec.thresholdGrowth.rule));
            writeValue(os, node.spec.thresholdGrowth.amount);
            break;
        }
        case OpKind::Connect:
            writeCount(os, op.a);
            writeCount(os, op.b);
            break;
        case OpKind::Feedback:
            writeCount(os, op.a);
            writeCount(os, op.b);
            writeCount(os, op.count);
            break;
        case OpKind::Inputs: {
            const InputSource& source = this->sources_[op.b];
            writeCount(os, op.a);
            writeCount(os, op.count);
            writeCount(os, source.name.size());
            binary_io::writeChars(os, source.name);
            writeCount(os, source.shape.channels);
            writeCount(os, source.shape.height);
            writeCount(os, source.shape.width);
            break;
        }
        case OpKind::ConnectInputs:
            writeCount(os, op.b);
            writeCount(os, op.a);
            break;
        }
    }

    // 2. Output layers and custom update order
    writeCount(os, this->outputs_.size());
    for (LayerId id : this->outputs_)
        writeCount(os, id);
    writeValue<std::uint8_t>(os, this->customOrder);
    if (this->customOrder)
    {
        writeCount(os, this->order_.size());
        for (LayerId id : this->order_)
            writeCount(os, id);
    }

    // 3. Input values
    writeCount(os, this->inputs_.size());
    for (float input : this->inputs_)
        writeValue(os, input);

    // 4. Layer states, in id order
    for (const Node& node : this->nodes)
        node.impl->serialize(os);

    if (!os)
        throw std::runtime_error("network::save: write failed");
}

std::unique_ptr<network> network::load(std::istream& is, DeserializeMode mode, const layer_factory& factory)
{
    std::array<char, sizeof(NETWORK_MAGIC)> magic{};
    binary_io::read(is, std::span<char>(magic));
    if (!is || !std::ranges::equal(magic, NETWORK_MAGIC))
        throw std::runtime_error("network::load: not a network stream");
    const auto version = readValue<std::uint32_t>(is);
    if (version < 1 || version > NETWORK_FORMAT_VERSION)
        throw std::runtime_error("network::load: unsupported format version " + std::to_string(version));

    // Older files: topology, frozen flags and weights only. Their state and
    // dynamics parameters are not restored.
    const bool legacy = version < FIRST_FULL_STATE_VERSION;
    if (legacy)
        mode = DeserializeMode::WeightsOnly;
    const std::uint32_t neuron_format = version <= 2 ? 1 : version <= 8 ? 2 : 3;

    auto net = std::make_unique<network>(factory);

    // 1. Replay the construction. The build methods validate ids and wiring
    // exactly as they did originally.
    const size_t op_count = readCount(is, "operation count");
    for (size_t i = 0; i < op_count; ++i)
    {
        const auto kind = static_cast<OpKind>(readValue<std::uint8_t>(is));
        switch (kind)
        {
        case OpKind::AddLayer: {
            const auto name_length = readValue<std::uint64_t>(is);
            if (name_length > MAX_NAME_LENGTH)
                throw std::runtime_error("network::load: implausible layer name length");
            std::string name(static_cast<size_t>(name_length), '\0');
            binary_io::readChars(is, name);
            LayerSpec spec;
            spec.type = static_cast<LayerType>(readValue<std::uint8_t>(is));
            spec.size = readCount(is, "layer size");
            spec.hasHabituation = readValue<std::uint8_t>(is) != 0;
            spec.hasER = readValue<std::uint8_t>(is) != 0;
            if (version >= 2)
                spec.frozen = readValue<std::uint8_t>(is) != 0;
            if (version == 3) {
                readValue<float>(is);  // old uniform jitter widths: not restored
                readValue<float>(is);
            } else if (version >= 4) {
                // Parsed to keep the stream aligned; for older versions the
                // settings are dropped (legacy files load without parameters).
                spec.recoveryJitter = readJitter(is, version);
                spec.learningJitter = readJitter(is, version);
                if (version >= 6)
                    spec.alphaJitter = readJitter(is, version);
                if (legacy) {
                    spec.recoveryJitter = Jitter::none();
                    spec.learningJitter = Jitter::none();
                    spec.alphaJitter = Jitter::none();
                }
            }
            if (version >= 7)
                readSpatial(is, spec);
            if (version >= 8)
                readAudio(is, spec);
            if (version >= 9)
                spec.learningRule = readLearningRule(is);
            if (version >= 10)
                readMultimodal(is, spec);
            if (version >= 11) {
                spec.gate = readValue<float>(is);
                if (!std::isfinite(spec.gate) || spec.gate < 0.0f)
                    throw std::runtime_error("network::load: invalid gate");
            }
            if (version >= 12)
                spec.rectify = readValue<std::uint8_t>(is) != 0;
            if (version >= 13) {
                spec.habituationRule.steps = readValue<std::uint32_t>(is);
                spec.habituationRule.tolerance = readValue<float>(is);
                spec.habituationRule.decay = readValue<float>(is);
                if (!spec.habituationRule.valid() || !std::isfinite(spec.habituationRule.tolerance) ||
                    !std::isfinite(spec.habituationRule.decay))
                    throw std::runtime_error("network::load: invalid habituation rule");
            }
            if (version < 14)
                spec.thresholdGrowth.rule = ThresholdGrowth::Rule::Log;  // the only rule before format 14
            else {
                spec.thresholdGrowth.rule = static_cast<ThresholdGrowth::Rule>(readValue<std::uint8_t>(is));
                spec.thresholdGrowth.amount = readValue<float>(is);
                if (!spec.thresholdGrowth.valid() || !std::isfinite(spec.thresholdGrowth.amount))
                    throw std::runtime_error("network::load: invalid threshold growth rule");
            }
            net->addLayer(name, spec);
            break;
        }
        case OpKind::Connect: {
            const size_t from = readCount(is, "layer id");
            const size_t to = readCount(is, "layer id");
            net->connect(from, to);
            break;
        }
        case OpKind::Feedback: {
            const size_t from = readCount(is, "layer id");
            const size_t to = readCount(is, "layer id");
            const size_t width = readCount(is, "feedback width");
            net->addFeedback(from, to, width);
            break;
        }
        case OpKind::Inputs: {
            const size_t target = readCount(is, "layer id");
            const size_t count = readCount(is, "input count");
            if (version < 10) {
                net->addInputs(target, count);
                break;
            }
            const size_t name_length = readCount(is, "input source name length");
            if (name_length > MAX_NAME_LENGTH)
                throw std::runtime_error("network::load: implausible input source name length");
            std::string name(name_length, '\0');
            binary_io::readChars(is, name);
            Shape shape;
            shape.channels = readCount(is, "input channels");
            shape.height = readCount(is, "input height");
            shape.width = readCount(is, "input width");
            if (shape.size() != count)
                throw std::runtime_error("network::load: input source shape does not match its sensor count");
            net->addInputs(target, shape, name);
            break;
        }
        case OpKind::ConnectInputs: {
            if (version < 10)
                throw std::runtime_error("network::load: unknown operation 4");
            const size_t index = readCount(is, "input source");
            const size_t target = readCount(is, "layer id");
            if (index >= net->sources_.size() || net->sources_[index].name.empty())
                throw std::runtime_error("network::load: connectInputs names an unknown input source");
            net->connectInputs(net->sources_[index].name, target);
            break;
        }
        default:
            throw std::runtime_error("network::load: unknown operation " +
                                     std::to_string(static_cast<int>(kind)));
        }
    }

    // 2. Output layers and custom update order
    const size_t output_count = readCount(is, "output count");
    for (size_t i = 0; i < output_count; ++i)
        net->addOutput(readCount(is, "layer id"));
    if (readValue<std::uint8_t>(is) != 0)
    {
        std::vector<LayerId> order(readCount(is, "update order length"));
        for (LayerId& id : order)
            id = readCount(is, "layer id");
        net->setUpdateOrder(std::move(order));
    }

    // 3. Input values (only restored with FullState; `mode` is WeightsOnly for older files)
    const size_t input_count = readCount(is, "input count");
    if (input_count != net->inputCount())
        throw std::runtime_error("network::load: input count does not match the construction history");
    for (size_t i = 0; i < input_count; ++i)
    {
        const float value = readValue<float>(is);
        if (mode == DeserializeMode::FullState)
            net->inputs_[i] = value;
    }

    // 4. Layer states. The replayed layers have the same neurons and wiring
    // as the saved ones, so each deserializes in place.
    for (Node& node : net->nodes)
    {
        const size_t expected_size = node.impl->size();
        node.impl->deserialize(is, mode, neuron_format);
        if (!is)
            throw std::runtime_error("network::load: unexpected end of data in layer '" + node.name + "'");
        if (node.impl->size() != expected_size)
            throw std::runtime_error("network::load: layer '" + node.name + "' does not match its saved state");
        if (legacy && node.impl->hasNeurons()) {
            // Version 3 layer data carries per-neuron recovery / gain: reset
            // them to the defaults, like the (unrestored) jitter settings.
            auto& neurons = dynamic_cast<neuron_layer&>(*node.impl);
            neurons.setRecoveryJitter(Jitter::none());
            neurons.setLearningJitter(Jitter::none());
            neurons.setAlphaJitter(Jitter::none());
        }
    }
    return net;
}

} // namespace exr
