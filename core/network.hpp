#pragma once
#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <istream>
#include <ostream>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <vector>
#include "layers/dense.hpp"
#include "layers/layer.hpp"
#include "layers/layer_factory.hpp"
#include "layers/neuron_layer.hpp"
#include "development.hpp"
#include "reinforcement.hpp"

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
    // `init`: how the new weights start; Zero adds a connection that changes
    // nothing until it learns (e.g. a new layer joining a readout).
    void connect(LayerId from, LayerId to, WeightInit init = WeightInit::Random);

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

    // --- Buses ------------------------------------------------------------
    // A bus is a named group of neurons that some layers write into and any
    // number of later layers read from: growing a network then means
    // subscribing a new layer to buses instead of rewiring. A bus is a Dense
    // layer with its own neuron type, whatever the layers around it use
    // (e.g. non-habituating E-R, or perceptrons: LayerSpec::Perceptron).
    // Buses start frozen (their weights from the writers do not learn);
    // unfreeze() lets them learn. Sensors can write into a bus too
    // (connectInputs).
    //
    //   auto vision = net.addBus("vision", LayerSpec::Dense(28, false, true));
    //   net.connectInputs("eye", vision);
    //   net.subscribe(column, vision);          // column reads the bus
    //   net.writeBus(column, motor_bus);         // column writes into another
    //
    // Throws std::invalid_argument when the spec is not Dense.
    LayerId addBus(const std::string& name, LayerSpec spec, bool frozen = true);
    // `writer` writes into the bus: connect(writer, bus, init).
    void writeBus(LayerId writer, LayerId bus, WeightInit init = WeightInit::Random);
    // `reader` reads the bus: connect(bus, reader, init).
    void subscribe(LayerId reader, LayerId bus, WeightInit init = WeightInit::Random);
    bool isBus(LayerId id) const;
    std::vector<LayerId> buses() const;
    // The layers writing into / reading a bus (forward edges), in order.
    std::vector<LayerId> busWriters(LayerId bus) const;
    std::vector<LayerId> busReaders(LayerId bus) const;

    // --- Changing a running network ----------------------------------------
    // Everything here works between steps, keeps the rest of the network as
    // it is and is saved: a loaded network has the same structure.
    //
    // Width: `count` new neurons in a Dense layer, reading the same inputs as
    // its wiring group `group` (0: the neurons it was built with). Layers
    // reading it get the new outputs with `outgoing` weights (default Zero:
    // the network behaves exactly as before until the new neurons' weights
    // learn). `freezeExisting` freezes the layer's current neurons, so only
    // the new ones learn.
    void growLayer(LayerId id, size_t count, WeightInit outgoing = WeightInit::Zero, bool freezeExisting = false,
                   size_t group = 0);
    // Removes neurons (by index) from a Dense layer; every layer reading it
    // must be Dense, and drops the matching inputs. The critic and the
    // curiosity model drop their features too. Throws std::invalid_argument,
    // changing nothing, when it would leave fewer neurons than the layer's
    // minimum size.
    void pruneNeurons(LayerId id, std::vector<size_t> indices);
    // The protected core of a Dense layer: pruning never leaves fewer than
    // `minimum` neurons (0, the default: no floor). Which neurons survive is
    // up to the caller (e.g. pruning grown neurons first); the floor is a
    // count. Throws std::invalid_argument when the layer has fewer neurons
    // than `minimum` now. Saved (LayerSpec::minimumSize).
    void setMinimumSize(LayerId id, size_t minimum);
    size_t minimumSize(LayerId id) const;
    // Neurons growLayer added (see neuron_layer::growthOrder), in index order;
    // the others are the layer's base neurons.
    std::vector<size_t> grownNeurons(LayerId id) const;
    // LIFO pruning: removes up to `count` grown neurons, the most recently
    // grown first, never going below the minimum size; base neurons are
    // never removed. Returns the indices removed (as they were before).
    std::vector<size_t> pruneNewest(LayerId id, size_t count = 1);

    // Why a neuron may be unnecessary (flags; see pruneCandidates).
    enum PruneReason : std::uint32_t
    {
        Invalid = 1,       // a weight, its bias, threshold or output is NaN or infinite
        Disconnected = 2,  // reads nothing
        ZeroIncoming = 4,  // every weight it reads with is 0
        Unread = 8,        // nothing downstream reads it with a non-zero weight (not an output layer,
                           // nor read by the critic or curiosity model); fresh neurons grown with
                           // zero outgoing weights are Unread until their readers learn
        Inactive = 16      // the activity monitor saw it go `inactiveAfter` ticks without firing on its
                           // input (fatigued and habituated ticks not counted)
    };
    struct PruneCandidate
    {
        size_t index;
        std::uint32_t reasons;  // PruneReason flags
        bool operator==(const PruneCandidate&) const = default;
    };
    // Neurons of a Dense layer that may be unnecessary, with why, in index
    // order. Nothing is removed: what to prune (and whether the minimum
    // allows it) is the caller's choice. `activity`, when given, must watch
    // this layer; Inactive needs it.
    std::vector<PruneCandidate> pruneCandidates(LayerId id, const activity_monitor* activity = nullptr,
                                                size_t inactiveAfter = 1000) const;
    // Freezes (or unfreezes) single neurons of a layer: they run but do not
    // learn (see neuron_layer::setNeuronsFrozen).
    void freezeNeurons(LayerId id, size_t first, size_t count, bool frozen = true);
    // Freezes (or unfreezes) the weights with which Dense layer `id` reads
    // `source` (a layer, or a named input source): e.g. a readout keeps what
    // it learned from old layers while it learns to read a new one.
    void freezeInputs(LayerId id, LayerId source, bool frozen = true);
    void freezeInputs(LayerId id, const std::string& inputSource, bool frozen = true);

    // --- Reinforcement signals (see reinforcement.hpp) ----------------------
    // Actor-critic: a TD(lambda) value predictor over the outputs of
    // spec.layers (and named inputs). Replaces any critic set before.
    // Throws std::invalid_argument for unknown layers or inputs.
    void setCritic(const CriticSpec& spec);
    void removeCritic();
    bool hasCritic() const { return critic_.has_value(); }
    const critic& getCritic() const;  // throws std::logic_error without a critic
    // V of the current state (after the last step).
    float criticValue() const;
    // One TD step after step(): `reward` is what the step earned. Returns the
    // TD error; the critic learns from it, the layers do not.
    float temporalDifference(float reward, bool terminal = false);
    // temporalDifference(), then applyReward(TD error, learningRate): the
    // layers learn from the critic's error instead of the raw reward.
    float applyRewardTD(float reward, float learningRate, bool terminal = false);
    // An episode boundary: the critic forgets the previous state and its traces.
    void resetCritic();

    // Curiosity: a forward model predicting spec.predict* from spec.from*
    // one tick earlier. Replaces any model set before.
    void setCuriosity(const CuriositySpec& spec);
    void removeCuriosity();
    bool hasCuriosity() const { return curiosity_.has_value(); }
    const curiosity& getCuriosity() const;  // throws std::logic_error without one
    // After step(): the model predicts this tick from the last one, learns
    // from the miss and returns the intrinsic reward (0 right after a reset).
    // Add it to the reward, e.g. applyRewardTD(reward + curiosityReward(), lr).
    float curiosityReward();
    void resetCuriosity();

    // A frozen layer still runs forward() but does not learn:
    // applyReward() skips it. Layers can also start frozen via
    // LayerSpec::frozen. Calling applyReward() on the layer object directly
    // bypasses this.
    void freeze(LayerId id);
    void unfreeze(LayerId id);
    bool isFrozen(LayerId id) const;

    // Per-neuron dynamics of a layer made of neurons (see Jitter in
    // neuron.hpp; other layer types throw std::invalid_argument): E-R recovery,
    // learning gain and E-R alpha (log growth rule only), each independently optional. Normally
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
    //   EProp (hidden)        its fixed random projection of the errors
    //   Surrogate             backpropagation through time (see learning.hpp):
    //                         from the output layers that are Surrogate,
    //                         through every Surrogate layer, over the last
    //                         `window` ticks
    //   Sign, Trace, Eligibility (hidden)  nothing: they need a scalar reward
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

    // Human-readable summary for logs and test output: every layer (outputs,
    // shape, habituation, E-R or gate / relu, learning, jitter, learning
    // rule; not the habituation or growth rule), where the inputs attach, the
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
    enum class OpKind : uint8_t { AddLayer, Connect, Feedback, Inputs, ConnectInputs, Grow, Prune };
    struct BuildOp
    {
        OpKind kind;
        size_t a;      // AddLayer: layer id; Connect/Feedback: from; Inputs, ConnectInputs: target; Grow, Prune: layer
        size_t b;      // Connect/Feedback: to; Inputs, ConnectInputs: input source index; Grow: wiring group
        size_t count;  // Feedback: width; Inputs: sensor count; Grow: new neurons
        WeightInit init = WeightInit::Random;  // Connect: the new weights; Grow: the outgoing weights
        std::vector<size_t> indices = {};      // Prune: the neurons removed
    };

    void checkId(LayerId id) const;
    neuron_layer& neuronLayer(LayerId id);
    std::vector<LayerId> topologicalOrder() const;
    size_t addSource(LayerId target, const Shape& shape, const std::string& name);
    size_t findSource(const std::string& name) const;
    dense& denseLayer(LayerId id, const char* what);
    // The outputs of `layers` and the values of the named `inputs`, concatenated.
    std::vector<float> gather(const std::vector<LayerId>& layers, const std::vector<std::string>& inputs) const;
    // Size of gather(layers, {}) and where layer `id`'s outputs start in it
    // (every occurrence).
    std::vector<size_t> offsetsOf(const std::vector<LayerId>& layers, LayerId id) const;
    void checkFeatures(const std::vector<LayerId>& layers, const std::vector<std::string>& inputs) const;
    // Keep the critic and curiosity model in step with a layer that grew by
    // `count` (its old size `oldSize`) or lost `removed`.
    void layerGrew(LayerId id, size_t oldSize, size_t count);
    void layerShrank(LayerId id, std::span<const size_t> removed);
    // Surrogate gradients for applyError.
    void backpropagateSurrogate(std::span<const float> errors, float learningRate);

    const layer_factory& factory;
    std::vector<Node> nodes;
    std::vector<Edge> edges_;
    std::vector<float> inputs_;  // every sensor, in addInputs order; layers read ranges of it
    std::vector<InputSource> sources_;
    std::vector<LayerId> outputs_;
    std::vector<BuildOp> ops_;
    std::optional<critic> critic_;
    std::optional<curiosity> curiosity_;

    bool customOrder = false;
    mutable std::vector<LayerId> order_;
    mutable bool orderValid = false;  // default order is recomputed after the graph changes
};

} // namespace exr
