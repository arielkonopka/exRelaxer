// A neuron layer's output together with its internal state, as an ordinary
// output that other layers (and readouts) can read: per neuron of its
// source, three values: its output, its E-R threshold and how far its
// habituation streak has run (doc/state_output.md). Experimental: it exists
// to test whether downstream neurons can use the state the source's outputs
// do not show. The source neurons are not changed; without a State layer
// nothing differs.
//
// Output, flat, per source neuron i (fields in this order, only those
// enabled; all three by default):
//   output       the neuron's output this tick, as the source's own output;
//   threshold    the neuron's threshold after this tick minus its resting
//                threshold (0 without E-R): 0 at rest, positive after a
//                firing (up to max_output - rest), negative while it relaxes
//                below rest during a long silence (down to -rest); the same
//                units as outputs and sums;
//   habituation  min(streak / onset, 1), where streak is the number of
//                consecutive ticks the neuron's sum has been "the same" and
//                onset the streak length at which suppression starts
//                (Habituation::onset); 1 = the input is being suppressed
//                (0 without habituation).
// Per neuron, so a growing source only appends.
//
// A tap reads its source when it runs: put it after the source in the
// update order (connect(source, tap) does) so it sees the state of this
// tick, as a reader of the source's output would. No neurons, no weights,
// no learning, no state of its own.
#pragma once
#include "layer.hpp"

namespace exr {

class neuron_layer;

class state_tap final : public layer
{
public:
    // Throws std::invalid_argument when no field is enabled.
    state_tap(bool output, bool threshold, bool habituation);

    LayerType type() const override { return LayerType::State; }
    bool readsOutput() const { return output_field_; }
    bool readsThreshold() const { return threshold_; }
    bool readsHabituation() const { return habituation_; }
    size_t fields() const
    {
        return static_cast<size_t>(output_field_) + static_cast<size_t>(threshold_) + static_cast<size_t>(habituation_);
    }

    // One source, a layer made of neurons. Throws std::logic_error for a
    // second source or a layer without neurons.
    void join(layer& source) override;
    void forward() override;

    // Nothing to save: the output is recomputed from the source every tick.
    void serialize(std::ostream& os) const override;
    void deserialize(std::istream& is, DeserializeMode mode = DeserializeMode::FullState,
                     std::uint32_t neuronFormat = NEURON_FORMAT_VERSION) override;

protected:
    void sourceGrew(const layer& source, size_t offset, size_t count) override;

private:
    void resizeOutput();

    bool output_field_, threshold_, habituation_;
    const neuron_layer* source_ = nullptr;
};

} // namespace exr
