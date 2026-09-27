#pragma once
#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <istream>
#include <ostream>
#include <memory>
#include <span>
#include <string>
#include <vector>
#include "layers/layer.hpp"
#include "layers/layer_factory.hpp"
#include "layers/neuron_layer.hpp"

namespace exr {

// A graph of layers plus the sensors that feed it and the layers read as its
// output. Layers are created through a layer_factory and handled only
// through the `layer` interface, so any registered layer type can be used.
//
//   network net;
//   auto in  = net.addLayer("in",  {LayerType::Dense, 16});
//   auto hid = net.addLayer("hid", {LayerType::Dense, 32});
//   auto out = net.addLayer("out", {LayerType::Dense, 4});
//   net.addInputs(in, 2);          // two sensors feeding "in"
//   net.connect(in, hid);          // hid reads in's output
//   net.connect(hid, out);
//   net.addFeedback(out, hid, 8);  // 8 new neurons in hid reading out
//   net.addOutput(out);
//
//   net.setInputs({0.5f, -0.2f});
//   net.step();
//   std::vector<float> y = net.outputs();
//
// Timing: step() runs every layer's forward() once, in update order. By
// default that is the topological order of the forward (connect) edges, so
// a value travels through the whole forward path in one step, and a
// feedback edge from a later layer to an earlier one delivers the previous
// step's value. setUpdateOrder() overrides this (e.g. to build a delay line).
class network
{
public:
    using LayerId = size_t;

    enum class EdgeKind : uint8_t
    {
        Forward,   // connect(from, to): every neuron `to` has at that moment reads `from`
        Feedback   // addFeedback(from, to, width): `width` new neurons in `to` read `from`
    };

    struct Edge
    {
        LayerId from;
        LayerId to;
        EdgeKind kind;
        size_t width;  // neurons added to `to` (Feedback only; 0 for Forward)
    };

    explicit network(const layer_factory& factory = layer_factory::instance());

    // Layers hold references to each other (readers, output buffers), and the network hands
    // out references to its layers, so a network is neither copied nor moved.
    network(const network&) = delete;
    network& operator=(const network&) = delete;

    // --- Building -------------------------------------------------------
    // Names must be unique and non-empty.
    LayerId addLayer(const std::string& name, const LayerSpec& spec);

    // `to` reads `from`'s output. As with layer::join, only the neurons `to`
    // has right now are wired, and each of them sums `from` together with
    // everything it already reads; call this before growing `to` with feedback.
    // Each pair can be connected once. connect(x, x) makes every neuron of x
    // read x's own output (recurrence) and is ignored by the update order.
    void connect(LayerId from, LayerId to);

    // Adds `width` new neurons to `to`, wired to read `from`'s output. Layers
    // already reading `to` are extended automatically.
    void addFeedback(LayerId from, LayerId to, size_t width);

    // Creates `count` sensors owned by the network (initially 0) and
    // attaches them to `target`. Returns the index of the first new sensor.
    // With a `name`, the sensors are a named input source (a camera, a
    // microphone...): set them with setInputs(name, values) and attach them
    // to more layers with connectInputs. Names must be unique; unnamed
    // sources are only reachable by index.
    size_t addInputs(LayerId target, size_t count, const std::string& name = "");
    // Sensors for an image, channel after channel, row after row (e.g. for a
    // retina); like addInputs(target, image.size(), name), and the source
    // remembers the shape.
    size_t addInputs(LayerId target, const Shape& image, const std::string& name = "");
    // Attaches the named source's sensors to one more layer: several layers
    // read the same sensors (e.g. one camera feeding a retina and a dense
    // layer). Throws std::out_of_range for an unknown name.
    void connectInputs(const std::string& name, LayerId target);

    // Marks a layer as output; outputs() concatenates output layers in the
    // order they were marked.
    void addOutput(LayerId id);

    // A frozen layer still runs forward() but does not learn:
    // applyReward() skips it. Layers can also start frozen via
    // LayerSpec::frozen. Calling applyReward() on the layer object directly
    // bypasses this.
    void freeze(LayerId id);
    void unfreeze(LayerId id);
    bool isFrozen(LayerId id) const;

    // Per-neuron dynamics of a layer made of neurons (see Jitter in
    // neuron.hpp; other layer types throw std::invalid_argument): E-R recovery,
    // learning gain and E-R alpha, each independently optional. Normally
    // chosen at layer creation through LayerSpec; these setters change it
    // later: they redraw that parameter for every existing neuron of the
    // layer (a disabled jitter resets it to the default) and use the jitter
    // for neurons added later. The current setting
    // is in layerSpec(id) and is saved with the network.
    void setRecoveryJitter(LayerId id, const Jitter& jitter);
    void setLearningJitter(LayerId id, const Jitter& jitter);
    void setAlphaJitter(LayerId id, const Jitter& jitter);

    // The layer's learning rule (see learning.hpp), normally chosen through
    // LayerSpec::learningRule. Resets the rule's state; weights are kept.
    // Throws std::invalid_argument for a layer without neurons, or a rule
    // other than Sign on a layer that does not learn.
    void setLearningRule(LayerId id, const LearningRule& rule);

    // Replaces the default update order. Must list every layer exactly once;
    // adding a layer afterwards requires setting it again.
    void setUpdateOrder(std::vector<LayerId> order);
    void useDefaultUpdateOrder();

    // --- Running --------------------------------------------------------
    void setInput(size_t index, float value);
    void setInputs(std::span<const float> values);  // values.size() must equal inputCount()
    void setInputs(std::initializer_list<float> values);
    // Only the named source's sensors (values.size() must equal its size).
    void setInputs(const std::string& name, std::span<const float> values);

    void step();                                     // forward() on every layer, in update order
    void applyReward(float reward, float learningRate);  // every layer that is not frozen
    // Learning from an error per output (errors.size() == outputs().size();
    // error = desired output - output, the direction to move it). Every
    // unfrozen layer that learns gets, by its rule:
    //   Oja, BCM              an unsupervised update (the error is ignored)
    //   Perturbation          the scalar reward -0.5 * sum of squared errors
    //   an output layer       its own errors, one per neuron (the first time
    //                         it is marked as output)
    //   FeedbackAlignment     (hidden) its fixed random projection of the errors
    //   Sign, Trace (hidden)  nothing: they need a scalar reward
    // Throws std::invalid_argument on a size mismatch.
    void applyError(std::span<const float> errors, float learningRate);
    std::vector<float> outputs() const;

    // --- Inspection -----------------------------------------------------
    size_t layerCount() const { return nodes.size(); }
    layer& getLayer(LayerId id);
    const layer& getLayer(LayerId id) const;
    // The layer as its concrete type, e.g. layerAs<conv2d>(id); throws
    // std::bad_cast if it is another type.
    template <typename T>
    T& layerAs(LayerId id) { return dynamic_cast<T&>(getLayer(id)); }
    template <typename T>
    const T& layerAs(LayerId id) const { return dynamic_cast<const T&>(getLayer(id)); }
    LayerId findLayer(const std::string& name) const;  // throws std::out_of_range if absent
    const std::string& layerName(LayerId id) const;
    const LayerSpec& layerSpec(LayerId id) const;

    const std::vector<Edge>& edges() const { return edges_; }
    const std::vector<LayerId>& outputLayers() const { return outputs_; }
    size_t inputCount() const { return inputs_.size(); }
    std::span<const float> inputs() const { return inputs_; }

    // A block of sensors created by one addInputs call.
    struct InputSource
    {
        std::string name;             // "" if unnamed
        size_t first;                 // index of its first sensor
        Shape shape;                  // flat(count) unless created with a shape
        std::vector<LayerId> targets; // the layers reading it, in attach order
        size_t size() const { return shape.size(); }
    };
    const std::vector<InputSource>& inputSources() const { return sources_; }
    // Throws std::out_of_range for an unknown name.
    const InputSource& inputSource(const std::string& name) const;
    std::span<const float> inputs(const std::string& name) const;

    // Throws std::logic_error if the forward edges form a cycle (other than
    // self-connections) and no custom order is set.
    const std::vector<LayerId>& updateOrder() const;

    // Human-readable summary for logs and test output: every layer (current
    // neuron count, habituation / E-R, frozen), where the inputs attach, the
    // edges in creation order, the output layers and the update order.
    void describe(std::ostream& os) const;

    // --- Serialization --------------------------------------------------
    // Writes the construction history (every addLayer, connect, addFeedback,
    // addInputs and connectInputs call, in order, with each layer's current frozen flag),
    // the output layers, a custom update order if set, the input values and
    // each layer's own state. Replaying the
    // history is what restores the wiring, which layers do not save.
    // Binary, native endianness: not portable across platforms.
    void save(std::ostream& os) const;

    // Rebuilds a network written by save(): replays the construction, then
    // restores every layer's state (and, with FullState, the input values).
    // Files from older format versions load as weights only, whatever `mode`
    // says: topology, frozen flags and weights; jitter settings and
    // per-neuron dynamics come back as defaults.
    // Layer types must be registered in `factory`. Throws std::runtime_error
    // on malformed or truncated data.
    static std::unique_ptr<network> load(std::istream& is,
                                         DeserializeMode mode = DeserializeMode::FullState,
                                         const layer_factory& factory = layer_factory::instance());

private:
    struct Node
    {
        std::string name;
        LayerSpec spec;
        std::unique_ptr<layer> impl;
    };

    // One entry per successful build call, replayed by load().
    enum class OpKind : uint8_t { AddLayer, Connect, Feedback, Inputs, ConnectInputs };
    struct BuildOp
    {
        OpKind kind;
        size_t a;      // AddLayer: layer id; Connect/Feedback: from; Inputs, ConnectInputs: target
        size_t b;      // Connect/Feedback: to; Inputs, ConnectInputs: input source index
        size_t count;  // Feedback: width; Inputs: sensor count
    };

    void checkId(LayerId id) const;
    neuron_layer& neuronLayer(LayerId id);
    std::vector<LayerId> topologicalOrder() const;
    size_t addSource(LayerId target, const Shape& shape, const std::string& name);
    size_t findSource(const std::string& name) const;

    const layer_factory& factory;
    std::vector<Node> nodes;
    std::vector<Edge> edges_;
    std::vector<float> inputs_;  // every sensor, in addInputs order; layers read ranges of it
    std::vector<InputSource> sources_;
    std::vector<LayerId> outputs_;
    std::vector<BuildOp> ops_;

    bool customOrder = false;
    mutable std::vector<LayerId> order_;
    mutable bool orderValid = false;  // default order is recomputed after the graph changes
};

} // namespace exr
