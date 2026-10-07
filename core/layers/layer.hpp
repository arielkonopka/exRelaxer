// Base class of every layer type: a contiguous output buffer, a shape, and
// the growth notifications that keep readers wired when a layer grows.
#pragma once
#include <cstddef>
#include <cstdint>
#include <functional>
#include <iosfwd>
#include <memory>
#include <span>
#include <vector>
#include "../neuron.hpp"

namespace exr {

// Saved in network files: never renumber.
enum class LayerType : std::uint8_t {
    Dense = 0,
    Conv2D = 1,
    Pool2D = 2,
    LocallyConnected2D = 3,
    Retina = 4,
    Cochlea = 5,
    History = 6,
    Resize2D = 7,
    Disparity = 8,
    State = 9
};

// Layout of a layer's output: channels x height x width, channel-major
// (all of channel 0, then channel 1, ...), so a layer that grows by whole
// channels only ever appends to its output.
struct Shape
{
    size_t channels = 0;
    size_t height = 1;
    size_t width = 1;

    constexpr size_t size() const { return channels * height * width; }
    static constexpr Shape flat(size_t count) { return {count, 1, 1}; }
    bool operator==(const Shape&) const = default;
};

// A run of values a layer reads: `count` entries of `buffer` from `offset`.
// It refers to the buffer object, not its data, so it stays valid when the
// buffer reallocates as its owner grows; the buffer must outlive the reader.
// Implicitly built from a whole vector (e.g. sensors owned by the caller);
// a temporary vector is rejected at compile time.
struct InputRange
{
    std::reference_wrapper<const std::vector<float>> buffer;
    size_t offset = 0;
    size_t count = 0;

    InputRange(const std::vector<float>& values) : buffer(values), count(values.size()) {}
    InputRange(const std::vector<float>& values, size_t offset, size_t count)
        : buffer(values), offset(offset), count(count) {}
    InputRange(const std::vector<float>&&) = delete;
    InputRange(const std::vector<float>&&, size_t, size_t) = delete;

    // The values as they are now (bounds-checked in debug builds).
    std::span<const float> values() const { return std::span(buffer.get()).subspan(offset, count); }
    bool sameBuffer(const InputRange& other) const
    {
        return std::addressof(buffer.get()) == std::addressof(other.buffer.get());
    }
};

// How weights are drawn when wiring changes (see layer::setIncomingInit).
// Saved in network files: never renumber.
enum class WeightInit : std::uint8_t {
    Random = 0,  // the library's weight streams (the default)
    Zero = 1     // zeros: the new connection changes nothing until it learns
};

class layer
{
public:
    virtual ~layer();
    layer(const layer&) = delete;
    layer& operator=(const layer&) = delete;

    virtual LayerType type() const = 0;
    // True for layers made of neurons (neuron_layer).
    virtual bool hasNeurons() const { return false; }
    // True when applyReward() can change the layer (it has learnable weights).
    virtual bool learns() const { return false; }
    virtual Shape shape() const { return Shape::flat(size()); }
    size_t size() const { return output_.size(); }

    // This tick's output. The span is invalidated when the layer grows; the
    // buffer object (outputBuffer(), what readers keep) is not.
    std::span<const float> output() const { return output_; }
    const std::vector<float>& outputBuffer() const { return output_; }

    // --- Running --------------------------------------------------------
    virtual void forward() = 0;
    // Default: the layer does not learn.
    virtual void applyReward(float reward, float learningRate);

    // --- Wiring ---------------------------------------------------------
    // Layer types that do not support an operation throw std::logic_error.
    //
    // join: every neuron this layer has now reads `source`'s whole output.
    virtual void join(layer& source);
    // attachInputs: every neuron this layer has now reads `sensors`.
    virtual void attachInputs(const InputRange& sensors);
    // addNeurons: `count` new neurons reading `source`'s output.
    virtual void addNeurons(size_t count, layer& source);
    // Feedback from this layer: `count` new neurons in `target` read it.
    void addFeedback(layer& target, size_t count) { target.addNeurons(count, *this); }

    // How new weights are drawn. Incoming: the weights this layer creates
    // when it is wired (join, attachInputs, addNeurons, growth). Outgoing:
    // the weights its readers create for its new outputs when it grows
    // (Zero: new neurons start without influence). Both default to Random;
    // the network sets them around a call (connect, growLayer).
    void setIncomingInit(WeightInit init) { incoming_init_ = init; }
    WeightInit incomingInit() const { return incoming_init_; }
    void setOutgoingInit(WeightInit init) { outgoing_init_ = init; }
    WeightInit outgoingInit() const { return outgoing_init_; }

    // --- Shrinking -------------------------------------------------------
    // Whether this layer can follow a source that loses outputs (pruning;
    // see sourceShrank): Dense only.
    virtual bool followsShrinkingSources() const { return false; }
    // Whether every reader of this layer can (so its outputs can be removed).
    bool readersFollowShrinking() const;

    // --- Serialization --------------------------------------------------
    // The layer's own state; the wiring is not saved (network::save replays
    // the construction instead). `neuronFormat`: the NEURON_FORMAT_VERSION
    // the data was written with.
    virtual void serialize(std::ostream& os) const = 0;
    virtual void deserialize(std::istream& is, DeserializeMode mode = DeserializeMode::FullState,
                             std::uint32_t neuronFormat = NEURON_FORMAT_VERSION) = 0;

    // True when other layers read this one.
    bool hasReaders() const { return !readers_.empty(); }

protected:
    layer() = default;

    // Registers *this as a reader of `source`, once: `source` then tells it
    // when its output grows (sourceGrew).
    void readFrom(layer& source);
    // Call after appending to output_: tells every reader, in the order they
    // registered, that entries [oldSize, size()) are new.
    void outputGrew(size_t oldSize);
    // A source this layer reads grew by `count` entries at `offset`.
    virtual void sourceGrew(const layer& source, size_t offset, size_t count);
    // Call after removing entries of output_: tells every reader which
    // entries (sorted indices into the old output) are gone.
    void outputShrank(std::span<const size_t> removed);
    // A source this layer reads lost the entries `removed` (sorted indices
    // into its old output). Default: throws std::logic_error.
    virtual void sourceShrank(const layer& source, std::span<const size_t> removed);

    std::vector<float> output_;

private:
    WeightInit incoming_init_{};
    WeightInit outgoing_init_{};
    std::vector<std::reference_wrapper<layer>> readers_;  // layers reading this one
    std::vector<std::reference_wrapper<layer>> sources_;  // layers this one reads (to unregister on destruction)
};

// Identity (the same object), not equality.
inline bool same(const layer& a, const layer& b) { return std::addressof(a) == std::addressof(b); }

} // namespace exr
