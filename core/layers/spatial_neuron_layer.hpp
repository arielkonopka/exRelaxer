// Base of the spatial layers made of neurons (Conv2D, LocallyConnected2D):
// `channels` output channels over the positions of a sliding window. Each
// neuron reads the window at its position over every input channel, as one
// weighted sum; derived types decide how weights are stored and shared.
#pragma once
#include <span>
#include <vector>
#include "neuron_layer.hpp"
#include "spatial.hpp"

namespace exr {

class spatial_neuron_layer : public neuron_layer
{
public:
    // channels x output height x output width (channel-major); 0 x 1 x 1 until wired.
    Shape shape() const override;
    const Window2D& window() const { return window_; }
    size_t outputChannels() const { return channels_; }
    size_t inputChannels() const { return inputs_.channels(); }
    // Inputs of one neuron: input channels x window area, channel by channel,
    // row by row within the window.
    size_t windowSize() const { return inputs_.channels() * window_.area(); }

    // The first source fixes the input height x width and so the output
    // size; the neurons are created then. Later sources (same height x
    // width) add input channels, read by every neuron.
    void join(layer& source) override;

protected:
    spatial_neuron_layer(size_t channels, const Window2D& window, bool hasHabituation, bool hasER,
                         const Jitter& recoveryJitter, const Jitter& learningJitter, const Jitter& alphaJitter);

    size_t positions() const { return out_height_ * out_width_; }
    size_t neuronAt(size_t channel, size_t position) const { return channel * positions() + position; }

    // Copies the inputs into a contiguous snapshot: call at the start of
    // forward() and applyReward(). Every neuron then sees the inputs from
    // before the layer ran, also when it reads itself.
    void gatherInputs() { inputs_.gather(tensor_); }
    // Sign rule: learning reads the snapshot the last forward() took (the
    // inputs at time t), not the sources' current outputs. Sized to the
    // current input shape; entries added since that forward() count as 0.
    void keepInputSnapshot() { tensor_.resize(inputs_.channels() * inputs_.height() * inputs_.width(), 0.0f); }
    // The window of `position` over the snapshot; `out` has windowSize() entries.
    void windowAt(size_t position, std::span<float> out) const;
    // Input traces (rules other than Sign): forward() calls traceSnapshot()
    // after gatherInputs(); traceWindowAt() is windowAt() over the trace.
    void traceSnapshot()
    {
        if (tracesInputs())
            traceInputs(trace_, tensor_);
    }
    void traceWindowAt(size_t position, std::span<float> out) const;
    // The weights' pre-synaptic factor for `position`: the signs of the
    // current window (Sign rule) or the window of the input trace.
    void learningWindowAt(size_t position, std::span<float> out) const;
    size_t competitionPositions() const override { return positions(); }
    void copyInputTraces(std::vector<float>& out) const override { out = trace_; }
    void storeInputTraces(std::span<const float> traces) override
    {
        if (traces.size() == tensor_.size() || tensor_.empty())
            trace_.assign(traces.begin(), traces.end());
    }
    void clearInputTraces() override { trace_.clear(); }

    // First join: draw the initial weights for windowSize() inputs.
    virtual void createWeights() = 0;
    // `count` inputs were appended to every neuron's window (new input channels).
    virtual void appendInputs(size_t count) = 0;

    void sourceGrew(const layer& source, size_t offset, size_t count) override;
    bool wired() const override { return !inputs_.empty(); }

private:
    spatial_inputs inputs_;
    Window2D window_;
    size_t channels_;
    size_t out_height_ = 0;
    size_t out_width_ = 0;
    std::vector<float> tensor_;  // input snapshot: input channels x height x width
    std::vector<float> trace_;   // input trace, same layout (rules other than Sign)
};

} // namespace exr
