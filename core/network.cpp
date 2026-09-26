#include "network.hpp"
#include <cmath>
#include <cstring>
#include <functional>
#include <iomanip>
#include <sstream>
#include <queue>
#include <stdexcept>

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

    this->nodes.push_back({name, spec, this->factory.create(spec)});
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

    if (!this->nodes[to].impl->join(*this->nodes[from].impl))
        throw std::runtime_error("network: joining '" + this->nodes[to].name + "' to '" +
                                 this->nodes[from].name + "' failed");
    this->edges_.push_back({from, to, EdgeKind::Forward, 0});
    this->ops_.push_back({OpKind::Connect, from, to, 0});
    this->orderValid = false;
}

void network::addFeedback(LayerId from, LayerId to, size_t width)
{
    checkId(from);
    checkId(to);
    if (!this->nodes[from].impl->addFeedback(*this->nodes[to].impl, width))
        throw std::runtime_error("network: feedback '" + this->nodes[from].name + "' -> '" +
                                 this->nodes[to].name + "' failed");
    this->edges_.push_back({from, to, EdgeKind::Feedback, width});
    this->ops_.push_back({OpKind::Feedback, from, to, width});
}

size_t network::addInputs(LayerId target, size_t count)
{
    checkId(target);
    if (count == 0)
        throw std::invalid_argument("network: addInputs needs at least one input");

    std::vector<std::shared_ptr<float>> sensors;
    for (size_t i = 0; i < count; ++i)
        sensors.push_back(std::make_shared<float>(0.0f));
    if (!this->nodes[target].impl->attachInputs(sensors))
        throw std::runtime_error("network: attaching inputs to '" + this->nodes[target].name + "' failed");

    const size_t first = this->inputs_.size();
    this->inputs_.insert(this->inputs_.end(), sensors.begin(), sensors.end());
    this->ops_.push_back({OpKind::Inputs, target, 0, count});
    return first;
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

void network::setRecoveryJitter(LayerId id, const Jitter& jitter)
{
    checkId(id);
    this->nodes[id].spec.recoveryJitter = jitter;
    this->nodes[id].impl->setRecoveryJitter(jitter);
}

void network::setLearningJitter(LayerId id, const Jitter& jitter)
{
    checkId(id);
    this->nodes[id].spec.learningJitter = jitter;
    this->nodes[id].impl->setLearningJitter(jitter);
}

void network::setAlphaJitter(LayerId id, const Jitter& jitter)
{
    checkId(id);
    this->nodes[id].spec.alphaJitter = jitter;
    this->nodes[id].impl->setAlphaJitter(jitter);
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
    *this->inputs_[index] = value;
}

void network::setInputs(std::span<const float> values)
{
    if (values.size() != this->inputs_.size())
        throw std::invalid_argument("network: expected " + std::to_string(this->inputs_.size()) +
                                    " input values, got " + std::to_string(values.size()));
    for (size_t i = 0; i < values.size(); ++i)
        *this->inputs_[i] = values[i];
}

void network::setInputs(std::initializer_list<float> values)
{
    setInputs(std::span<const float>(values.begin(), values.size()));
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

std::vector<float> network::outputs() const
{
    std::vector<float> values;
    for (LayerId id : this->outputs_)
        for (const auto& v : this->nodes[id].impl->getOutput())
            values.push_back(*v);
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
std::string describeJitter(float defaultValue, const Jitter& jitter, const char* scaleName = "")
{
    std::ostringstream text;
    text << jitter.mean.value_or(defaultValue);
    if (!jitter.enabled())
        return text.str();
    std::ostringstream spread;
    if (jitter.relative) {
        spread << jitter.spread * 100.0f << "%";
        if (*scaleName) spread << " of " << scaleName;
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

    os << "  layers (" << this->nodes.size() << "):\n"
       << "    " << std::left << std::setw(4) << "id" << std::setw(name_col) << "name"
       << std::setw(9) << "neurons" << std::setw(6) << "hab" << std::setw(6) << "E-R" << std::setw(8) << "learns"
       << std::setw(recovery_col) << "recovery" << std::setw(learning_col) << "learning gain" << "alpha\n";
    for (LayerId id = 0; id < this->nodes.size(); ++id)
    {
        const Node& node = this->nodes[id];
        os << "    " << std::setw(4) << id << std::setw(name_col) << node.name
           << std::setw(9) << node.impl->size()
           << std::setw(6) << (node.spec.hasHabituation ? "on" : "-")
           << std::setw(6) << (node.spec.hasER ? "on" : "-")
           << std::setw(8) << (node.spec.frozen ? "frozen" : "yes");
        os << std::setw(recovery_col) << describeJitter(recovery_factor, node.spec.recoveryJitter, "1-r")
           << std::setw(learning_col) << describeJitter(default_learning_gain, node.spec.learningJitter)
           << describeJitter(default_alpha, node.spec.alphaJitter) << "\n";
    }
    os << std::right;

    for (const BuildOp& op : this->ops_)
        if (op.kind == OpKind::Inputs)
            os << "  inputs: " << op.count << " -> " << this->nodes[op.a].name << "\n";

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
// Older versions load as weights only (see network::load).
constexpr std::uint32_t NETWORK_FORMAT_VERSION = 6;

// Upper bounds for counts read from a stream, so corrupt data fails with an
// error instead of an enormous allocation.
constexpr std::uint64_t MAX_NAME_LENGTH = 4096;
constexpr std::uint64_t MAX_COUNT = std::uint64_t{1} << 26;

template <typename T>
void writeValue(std::ostream& os, T value)
{
    os.write(reinterpret_cast<const char*>(&value), sizeof(value));
}

template <typename T>
T readValue(std::istream& is)
{
    T value{};
    is.read(reinterpret_cast<char*>(&value), sizeof(value));
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

size_t readCount(std::istream& is, const char* what)
{
    const auto value = readValue<std::uint64_t>(is);
    if (value > MAX_COUNT)
        throw std::runtime_error(std::string("network::load: implausible ") + what + " " + std::to_string(value));
    return static_cast<size_t>(value);
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
    os.write(NETWORK_MAGIC, sizeof(NETWORK_MAGIC));
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
            os.write(node.name.data(), static_cast<std::streamsize>(node.name.size()));
            writeValue(os, static_cast<std::uint8_t>(node.spec.type));
            writeCount(os, node.spec.size);
            writeValue<std::uint8_t>(os, node.spec.hasHabituation);
            writeValue<std::uint8_t>(os, node.spec.hasER);
            writeValue<std::uint8_t>(os, node.spec.frozen);  // current state, not the one at creation
            writeJitter(os, node.spec.recoveryJitter);  // current settings, like frozen
            writeJitter(os, node.spec.learningJitter);
            writeJitter(os, node.spec.alphaJitter);
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
        case OpKind::Inputs:
            writeCount(os, op.a);
            writeCount(os, op.count);
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
    for (const auto& input : this->inputs_)
        writeValue(os, *input);

    // 4. Layer states, in id order
    for (const Node& node : this->nodes)
        node.impl->serialize(os);

    if (!os)
        throw std::runtime_error("network::save: write failed");
}

std::unique_ptr<network> network::load(std::istream& is, DeserializeMode mode, const layer_factory& factory)
{
    char magic[sizeof(NETWORK_MAGIC)] = {};
    is.read(magic, sizeof(magic));
    if (!is || std::memcmp(magic, NETWORK_MAGIC, sizeof(magic)) != 0)
        throw std::runtime_error("network::load: not a network stream");
    const auto version = readValue<std::uint32_t>(is);
    if (version < 1 || version > NETWORK_FORMAT_VERSION)
        throw std::runtime_error("network::load: unsupported format version " + std::to_string(version));

    // Older files: topology, frozen flags and weights only. Their state and
    // dynamics parameters are not restored.
    const bool legacy = version < NETWORK_FORMAT_VERSION;
    if (legacy)
        mode = DeserializeMode::WeightsOnly;
    const std::uint32_t neuron_format = version <= 2 ? 1 : 2;

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
            is.read(name.data(), static_cast<std::streamsize>(name.size()));
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
            net->addInputs(target, count);
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
            *net->inputs_[i] = value;
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
        if (legacy) {
            // Version 3 layer data carries per-neuron recovery / gain: reset
            // them to the defaults, like the (unrestored) jitter settings.
            node.impl->setRecoveryJitter(Jitter::none());
            node.impl->setLearningJitter(Jitter::none());
            node.impl->setAlphaJitter(Jitter::none());
        }
    }
    return net;
}
