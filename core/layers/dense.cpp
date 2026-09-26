
#include "dense.hpp"
#include <algorithm>
#include <numeric>
#ifdef _OPENMP
#include <omp.h>
#endif

dense::dense(size_t nNumber, bool hasHabituation, bool hasER, const Jitter& recoveryJitter,
             const Jitter& learningJitter, const Jitter& alphaJitter)
    : hasHabituation_(hasHabituation), hasER_(hasER),
      recoveryJitter_(recoveryJitter), learningJitter_(learningJitter), alphaJitter_(alphaJitter)
{
    // Construct `nNumber` neurons with no wiring yet - they have no weights
    // until joinDense() or addNeuronsWithGroup() gives them an input source.
    for (size_t i = 0; i < nNumber; ++i)
    {
        this->neurons.emplace_back(hasHabituation, hasER);
        this->neurons.back().randomizeDynamics(recoveryJitter, learningJitter, alphaJitter);
        this->output.push_back(this->neurons.back().getOutput());
    }
}

bool dense::join(layer& source)
{
    // Creates ONE new wiring group covering every neuron currently in this
    // layer, all reading from `source`'s output. Intended to be called once,
    // early - any neurons added later via addNeuronsWithGroup() get their
    // own separate group instead (see addFeedback).
    // run only once per relation!
    WiringGroup group{ source.getOutput(), {}, source, {} };
    group.neuronIndices.reserve(this->neurons.size());
    for (size_t i = 0; i < this->neurons.size(); ++i)
    {
        group.neuronIndices.push_back(i);
        this->neurons[i].initializeWeights(group.inputs);
    }
    this->groups.push_back(std::move(group));

    // Register as a listener on `source`, so if source's own output grows
    // later, notifySourceGrew() can extend this group to match. Guarded
    // against duplicates: wiring the same pair of layers together twice
    // should not cause a later growth event to double-apply.
    bool alreadyListening = false;
    for (layer& existing : source.listeners)
        if (&existing == this) { alreadyListening = true; break; }
    if (!alreadyListening)
        source.listeners.push_back(*this);
    return true;
}

void dense::addNeuronsWithGroup(size_t width, const std::vector<std::shared_ptr<float>>& sourceOutput, layer& source)
{
    // Adds `width` brand-new neurons in their own isolated wiring group,
    // reading only from `sourceOutput` - any neurons already in this layer
    // (in other groups) are completely untouched by this call.
    WiringGroup group{ sourceOutput, {}, source, {} };

    for (size_t i = 0; i < width; ++i)
    {
        this->neurons.emplace_back(this->hasHabituation_, this->hasER_);
        neuron& n = this->neurons.back();
        n.randomizeDynamics(this->recoveryJitter_, this->learningJitter_, this->alphaJitter_);
        n.initializeWeights(group.inputs);
        this->output.push_back(n.getOutput());
        group.neuronIndices.push_back(this->neurons.size() - 1);
    }

    this->groups.push_back(std::move(group));

    // Same duplicate-listener guard as joinDense() - see comment there.
    bool alreadyListening = false;
    for (layer& existing : source.listeners)
        if (&existing == this) { alreadyListening = true; break; }
    if (!alreadyListening)
        source.listeners.push_back(*this);

    // This layer's own output vector just grew by `width` new entries.
    // Anyone who was ALREADY listening to us (i.e. had joined/fed-back into
    // this layer before this call) needs their matching wiring group
    // extended to include the new neurons' outputs - that's the "layer
    // below adapts to changes" propagation. Layers that join us AFTER this
    // call don't need this, since joinDense()/addNeuronsWithGroup() always
    // reads our *current* (already-grown) output at the time they're called.
    if (width > 0)
    {
        std::vector<std::shared_ptr<float>> newOutputEntries(
            this->output.end() - static_cast<std::ptrdiff_t>(width), this->output.end());
        for (layer& listener : this->listeners)
            listener.notifySourceGrew(*this, newOutputEntries);
    }
}

void dense::notifySourceGrew(layer& sourceLayer, const std::vector<std::shared_ptr<float>>& newOutputEntries)
{
    // A layer we're listening to just grew. Find every wiring group of ours
    // that's actually sourced from it (there could be more than one, or
    // none, depending on how this layer has been wired) and extend each:
    // append the new output entries to the group's input pool, and grow
    // every neuron in that group by the same number of new weights, so
    // weights.size() stays in sync with the group's inputs.size().
    for (WiringGroup& group : this->groups)
    {
        // Identity comparison needs an address either way; this is a
        // transient local check, never stored - `source` itself stays a
        // reference_wrapper, not a pointer field (see dense.hpp).
        if (&group.source.get() != &sourceLayer)
            continue;
        for (const auto& ptr : newOutputEntries)
            group.inputs.push_back(ptr);
        for (size_t idx : group.neuronIndices)
            this->neurons[idx].growWeights(newOutputEntries.size());
    }
}

bool dense::addFeedback(layer& targetLayer, size_t width)
{
    // "this" is the feedback SOURCE; targetLayer is who receives new
    // neurons wired to read this layer's output. All the actual growth
    // and propagation logic lives in addNeuronsWithGroup() - this is just
    // the public-facing name/direction for that operation.
    targetLayer.addNeuronsWithGroup(width, this->getOutput(), *this);
    return true;
}

std::span<const float> dense::gather(WiringGroup& group)
{
    group.values.resize(group.inputs.size());
    for (size_t i = 0; i < group.inputs.size(); ++i)
        group.values[i] = *group.inputs[i];
    return group.values;
}

void dense::forward()
{
    // Groups run one after another; the neurons inside a group run in
    // parallel when the group is big enough. Each group first copies its
    // inputs into contiguous floats: the dot products then read memory
    // sequentially instead of chasing one pointer per input.
    //
    // The copy is also what makes a group sourced from this same layer
    // (self-feedback) safe to parallelize: its neurons read the copy, not
    // outputs other neurons of the group are writing. Every neuron in such a
    // group sees the values from before the group ran, independent of neuron
    // order and thread count.
    for (WiringGroup& group : this->groups)
    {
        const std::span<const float> inputs = gather(group);
        const std::vector<size_t>& indices = group.neuronIndices;
        int threads = 1;
#ifdef _OPENMP
        const size_t by_work = indices.size() * inputs.size() / parallel_work_per_thread;
        threads = static_cast<int>(std::min<size_t>(by_work, static_cast<size_t>(omp_get_max_threads())));
#endif
        if (threads >= 2)
        {
            // Plain branch rather than OpenMP's if() clause: that clause still
            // sets up a parallel region on every call, which made small
            // layers several times slower.
            const long count = static_cast<long>(indices.size());
#pragma omp parallel for schedule(static) num_threads(threads)
            for (long k = 0; k < count; ++k)
                this->neurons[indices[k]].step(inputs);
        }
        else
        {
            for (size_t idx : indices)
                this->neurons[idx].step(inputs);
        }
    }
}

void dense::serialize(std::ostream& os) const
{
    // Save layer configuration
    os.write(reinterpret_cast<const char*>(&hasHabituation_), sizeof(hasHabituation_));
    os.write(reinterpret_cast<const char*>(&hasER_), sizeof(hasER_));

    // save number of neurons
    size_t count = this->neurons.size();
    os.write(reinterpret_cast<const char*>(&count), sizeof(count));

    // serialize the neurons
    for (const auto& n : this->neurons)
    {
        n.serialize(os);
    }
}

void dense::setRecoveryJitter(const Jitter& jitter)
{
    this->recoveryJitter_ = jitter;
    for (neuron& n : this->neurons)
        n.randomizeRecovery(jitter);
}

void dense::setLearningJitter(const Jitter& jitter)
{
    this->learningJitter_ = jitter;
    for (neuron& n : this->neurons)
        n.randomizeLearningGain(jitter);
}

void dense::setAlphaJitter(const Jitter& jitter)
{
    this->alphaJitter_ = jitter;
    for (neuron& n : this->neurons)
        n.randomizeAlpha(jitter);
}

void dense::deserialize(std::istream& is, DeserializeMode mode, std::uint32_t neuronFormat)
{
    // read layer configuration
    is.read(reinterpret_cast<char*>(&hasHabituation_), sizeof(hasHabituation_));
    is.read(reinterpret_cast<char*>(&hasER_), sizeof(hasER_));

    // read the neuron number
    size_t count = 0;
    is.read(reinterpret_cast<char*>(&count), sizeof(count));

    if (this->neurons.size() == count)
    {
        // if the neurons are there, we deserialize them
        for (size_t i = 0; i < count; ++i)
        {
            this->neurons[i].deserialize(is, mode, neuronFormat);
        }
    }
    else
    {
        // we are reading empty layer
        this->neurons.clear();
        this->output.clear();
        for (size_t i = 0; i < count; ++i)
        {
            this->neurons.emplace_back(this->hasHabituation_, this->hasER_);
            this->neurons.back().deserialize(is, mode, neuronFormat);
            this->output.push_back(this->neurons.back().getOutput());
        }
    }
}

// Learning & Plasticity
void dense::applyReward(float reward, float learningRate) {

    for (WiringGroup& group : this->groups)
    {
        const std::span<const float> inputs = gather(group);
        for (size_t idx : group.neuronIndices)
            this->neurons[idx].updateWeights(inputs, reward, learningRate);
    }

};

bool dense::attachInputs(const std::vector<std::shared_ptr<float>>& input_pointers)
{
    if (input_pointers.empty()) return false;

    // 1. The layer already has at least one initialized input group:
    if (!groups.empty())
    {
        auto& existing_group = groups.front();
        size_t added_count = input_pointers.size();

        // Append the smart pointers to the existing input vector
        existing_group.inputs.insert(
            existing_group.inputs.end(),
            input_pointers.begin(),
            input_pointers.end()
        );

        // Grow each neuron's weights to cover the added inputs
        for (size_t idx : existing_group.neuronIndices)
        {
            neurons[idx].expandWeights(added_count);
        }

        return true;
    }

    // 2. The layer has no group yet (first sensory connection):
    WiringGroup sensory_group{input_pointers, {}, *this, {}};

    // Connect every neuron in this layer
    sensory_group.neuronIndices.resize(neurons.size());
    std::iota(sensory_group.neuronIndices.begin(), sensory_group.neuronIndices.end(), 0);

    // Initial weights for each neuron in the new sensory group
    size_t initial_count = input_pointers.size();
    for (auto& n : neurons)
    {
        n.expandWeights(initial_count);
    }

    groups.push_back(std::move(sensory_group));
    return true;
}
