// Python bindings for exrelaxer (module exrelaxer._core), with nanobind.
//
// Names follow Python conventions (snake_case); the objects are the C++
// ones, so docs in core/ apply. Arrays in and out are numpy float32; inputs
// in other dtypes are converted.
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/filesystem.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/unique_ptr.h>
#include <nanobind/stl/vector.h>

#include <filesystem>
#include <fstream>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <typeinfo>
#include <vector>
#ifdef _OPENMP
#include <omp.h>
#endif

#include "filters.hpp"
#include "network.hpp"
#include "neuron.hpp"
#include "random.hpp"
#include "layers/conv2d.hpp"
#include "layers/dense.hpp"
#include "layers/locally_connected2d.hpp"
#include "layers/retina.hpp"
#include "layers/cochlea.hpp"
#include "layers/history.hpp"

namespace nb = nanobind;
using namespace nb::literals;
using namespace exr;

namespace {

using FloatArray = nb::ndarray<nb::numpy, float>;
using FloatIn = nb::ndarray<const float, nb::c_contig, nb::device::cpu>;
using LayerId = network::LayerId;

// A numpy array that owns `values`.
FloatArray toNumpy(std::vector<float> values, std::vector<size_t> shape)
{
    auto* owned = new std::vector<float>(std::move(values));
    nb::capsule deleter(owned, [](void* p) noexcept { delete static_cast<std::vector<float>*>(p); });
    return FloatArray(owned->data(), shape.size(), shape.data(), deleter);
}

FloatArray toNumpy(std::vector<float> values)
{
    const size_t n = values.size();
    return toNumpy(std::move(values), {n});
}

std::span<const float> flat(const FloatIn& a) { return {a.data(), a.size()}; }

// The layer as T, with a readable error instead of std::bad_cast.
template <typename T>
T& layerAs(network& net, LayerId id, const char* what)
{
    try {
        return net.layerAs<T>(id);
    } catch (const std::bad_cast&) {
        throw std::invalid_argument("layer '" + net.layerName(id) + "' is not a " + what + " layer");
    }
}

neuron_layer& neuronLayer(network& net, LayerId id)
{
    return layerAs<neuron_layer>(net, id, "neuron");
}

LayerSpec withOptions(LayerSpec spec, bool frozen, const Jitter& recovery, const Jitter& learning,
                      const Jitter& alpha)
{
    spec.frozen = frozen;
    spec.recoveryJitter = recovery;
    spec.learningJitter = learning;
    spec.alphaJitter = alpha;
    return spec;
}

std::string toString(const Shape& s)
{
    return "Shape(" + std::to_string(s.channels) + ", " + std::to_string(s.height) + ", " + std::to_string(s.width) +
           ")";
}

} // namespace

NB_MODULE(_core, m)
{
    m.doc() = "exrelaxer: biologically inspired neurons (excitation-relaxation, habituation, reward-modulated "
              "Hebbian learning) - C++ core";

    m.def("reseed", &reseed, "seed"_a,
          "Resets every random stream of the library; call before building a network for reproducible weights.");

    m.def(
        "threads", [] {
#ifdef _OPENMP
            return omp_get_max_threads();
#else
            return 1;
#endif
        },
        "OpenMP threads the layers may use (1 without OpenMP).");
    m.def(
        "set_threads", [](int count) {
            if (count < 1)
                throw std::invalid_argument("set_threads: count must be at least 1");
#ifdef _OPENMP
            omp_set_num_threads(count);
#endif
        },
        "count"_a, "Sets the OpenMP thread count (results do not depend on it).");
    m.def(
        "build_info", [] {
            nb::dict d;
#if defined(__clang__)
            d["compiler"] = "clang " __clang_version__;
#elif defined(__GNUC__)
            d["compiler"] = "gcc " __VERSION__;
#elif defined(_MSC_VER)
            d["compiler"] = "msvc " + std::to_string(_MSC_VER);
#else
            d["compiler"] = "unknown";
#endif
            d["build"] = EXRELAXER_BUILD_TYPE;
            d["native"] = static_cast<bool>(EXRELAXER_NATIVE_BUILD);
#ifdef _OPENMP
            d["openmp"] = true;
#else
            d["openmp"] = false;
#endif
            return d;
        },
        "How the extension was compiled: compiler, build type, native, openmp.");

    // --- Constants (core/neuron.hpp) ---------------------------------------
    nb::module_ constants = m.def_submodule("constants", "Tunable constants shared by every neuron (read only).");
    constants.attr("habituation_steps") = habituation_steps;
    constants.attr("habituation_epsilon") = habituation_epsilon;
    constants.attr("recovery_factor") = recovery_factor;
    constants.attr("min_threshold") = min_threshold;
    constants.attr("spontaneous_min_amplitude") = spontaneous_min_amplitude;
    constants.attr("firing_epsilon") = firing_epsilon;
    constants.attr("baseline_threshold") = baseline_threshold;
    constants.attr("max_weight") = max_weight;
    constants.attr("max_output") = max_output;
    constants.attr("default_alpha") = default_alpha;
    constants.attr("default_learning_gain") = default_learning_gain;

    // --- Enums ---------------------------------------------------------------
    nb::enum_<LayerType>(m, "LayerType")
        .value("Dense", LayerType::Dense)
        .value("Conv2D", LayerType::Conv2D)
        .value("Pool2D", LayerType::Pool2D)
        .value("LocallyConnected2D", LayerType::LocallyConnected2D)
        .value("Retina", LayerType::Retina)
        .value("Cochlea", LayerType::Cochlea)
        .value("History", LayerType::History);
    nb::enum_<FrequencyScale>(m, "FrequencyScale")
        .value("Mel", FrequencyScale::Mel)
        .value("Linear", FrequencyScale::Linear);
    nb::enum_<Compression>(m, "Compression").value("Log", Compression::Log).value("Linear", Compression::Linear);
    nb::enum_<PoolMode>(m, "PoolMode").value("Max", PoolMode::Max).value("Average", PoolMode::Average);
    nb::enum_<Sampling>(m, "Sampling").value("Grid", Sampling::Grid).value("Spiral", Sampling::Spiral);
    nb::enum_<DeserializeMode>(m, "DeserializeMode")
        .value("WeightsOnly", DeserializeMode::WeightsOnly)
        .value("FullState", DeserializeMode::FullState);
    nb::enum_<network::EdgeKind>(m, "EdgeKind")
        .value("Forward", network::EdgeKind::Forward)
        .value("Feedback", network::EdgeKind::Feedback);

    // --- Geometry ------------------------------------------------------------
    nb::class_<Shape>(m, "Shape", "Layout of a layer's output: channels x height x width, channel-major.")
        .def(nb::init<>())
        .def("__init__", [](Shape* s, size_t c, size_t h, size_t w) { new (s) Shape{c, h, w}; }, "channels"_a,
             "height"_a = 1, "width"_a = 1)
        .def_rw("channels", &Shape::channels)
        .def_rw("height", &Shape::height)
        .def_rw("width", &Shape::width)
        .def_prop_ro("size", &Shape::size)
        .def_static("flat", &Shape::flat, "count"_a)
        .def("__eq__", [](const Shape& a, const Shape& b) { return a == b; })
        .def("__iter__", [](const Shape& s) {
            return nb::iter(nb::make_tuple(s.channels, s.height, s.width));
        })
        .def("__repr__", &toString);

    nb::class_<Window2D>(m, "Window2D", "A sliding window with zero padding.")
        .def("__init__",
             [](Window2D* w, size_t kh, size_t kw, size_t sy, size_t sx, size_t py, size_t px) {
                 new (w) Window2D{kh, kw, sy, sx, py, px};
             },
             "kernel_height"_a = 1, "kernel_width"_a = 1, "stride_y"_a = 1, "stride_x"_a = 1, "pad_y"_a = 0,
             "pad_x"_a = 0)
        .def_static("square", &Window2D::square, "kernel"_a, "stride"_a = 1, "padding"_a = 0)
        .def_rw("kernel_height", &Window2D::kernelHeight)
        .def_rw("kernel_width", &Window2D::kernelWidth)
        .def_rw("stride_y", &Window2D::strideY)
        .def_rw("stride_x", &Window2D::strideX)
        .def_rw("pad_y", &Window2D::padY)
        .def_rw("pad_x", &Window2D::padX)
        .def_prop_ro("area", &Window2D::area)
        .def("output_height", &Window2D::outputHeight, "height"_a)
        .def("output_width", &Window2D::outputWidth, "width"_a)
        .def("__eq__", [](const Window2D& a, const Window2D& b) { return a == b; });

    nb::class_<RetinaSpec>(m, "RetinaSpec", "An image input: grid (one sample per pixel) or spiral sampling.")
        .def("__init__",
             [](RetinaSpec* r, const Shape& input, Sampling sampling, float spacing, float radius) {
                 new (r) RetinaSpec{input, sampling, spacing, radius};
             },
             "input"_a, "sampling"_a = Sampling::Grid, "spacing"_a = 1.0f, "radius"_a = 0.0f)
        .def_rw("input", &RetinaSpec::input)
        .def_rw("sampling", &RetinaSpec::sampling)
        .def_rw("spacing", &RetinaSpec::spacing)
        .def_rw("radius", &RetinaSpec::radius)
        .def("__eq__", [](const RetinaSpec& a, const RetinaSpec& b) { return a == b; });

    nb::class_<CochleaSpec>(m, "CochleaSpec",
                            "A sound input: `hop` new samples per tick, the last `window` analysed (Hann window, "
                            "FFT) into `bands` triangular frequency bands.")
        .def("__init__",
             [](CochleaSpec* c, float sampleRate, size_t hop, size_t window, size_t bands, float minFrequency,
                float maxFrequency, FrequencyScale scale, Compression compression, float gain) {
                 new (c) CochleaSpec{sampleRate, hop, window, bands, minFrequency, maxFrequency, scale, compression,
                                     gain};
             },
             "sample_rate"_a = 16000.0f, "hop"_a = 160, "window"_a = 512, "bands"_a = 40, "min_frequency"_a = 50.0f,
             "max_frequency"_a = 0.0f, "scale"_a = FrequencyScale::Mel, "compression"_a = Compression::Log,
             "gain"_a = 100.0f)
        .def_rw("sample_rate", &CochleaSpec::sampleRate)
        .def_rw("hop", &CochleaSpec::hop)
        .def_rw("window", &CochleaSpec::window)
        .def_rw("bands", &CochleaSpec::bands)
        .def_rw("min_frequency", &CochleaSpec::minFrequency)
        .def_rw("max_frequency", &CochleaSpec::maxFrequency)
        .def_rw("scale", &CochleaSpec::scale)
        .def_rw("compression", &CochleaSpec::compression)
        .def_rw("gain", &CochleaSpec::gain)
        .def("__eq__", [](const CochleaSpec& a, const CochleaSpec& b) { return a == b; });

    // --- Jitter --------------------------------------------------------------
    nb::class_<Jitter>(m, "Jitter", "Random per-neuron variation of E-R recovery, learning gain or alpha.")
        .def(nb::init<>())
        .def_static("none", &Jitter::none)
        .def_static("uniform", &Jitter::uniform, "half_width"_a)
        .def_static("normal", &Jitter::normal, "stddev"_a)
        .def_static("uniform_relative", &Jitter::uniformRelative, "fraction"_a = 0.5f)
        .def_static("normal_relative", &Jitter::normalRelative, "fraction"_a = 0.5f)
        .def("around", &Jitter::around, "centre"_a)
        .def("within", &Jitter::within, "lo"_a, "hi"_a)
        .def_prop_ro("enabled", &Jitter::enabled)
        .def_prop_ro("distribution", [](const Jitter& j) {
            switch (j.distribution) {
            case Jitter::Distribution::Uniform: return "uniform";
            case Jitter::Distribution::Normal: return "normal";
            default: return "none";
            }
        })
        .def_ro("spread", &Jitter::spread)
        .def_ro("relative", &Jitter::relative)
        .def_ro("mean", &Jitter::mean)
        .def_ro("min", &Jitter::min)
        .def_ro("max", &Jitter::max)
        .def("__eq__", [](const Jitter& a, const Jitter& b) { return a == b; });

    // --- LayerSpec -------------------------------------------------------------
    const Jitter noJitter{};
    nb::class_<LayerSpec>(m, "LayerSpec",
                          "Everything needed to construct a layer; use the builders dense(), conv2d(), ...")
        .def(nb::init<>())
        .def_rw("type", &LayerSpec::type)
        .def_rw("size", &LayerSpec::size)
        .def_rw("has_habituation", &LayerSpec::hasHabituation)
        .def_rw("has_er", &LayerSpec::hasER)
        .def_rw("frozen", &LayerSpec::frozen)
        .def_rw("recovery_jitter", &LayerSpec::recoveryJitter)
        .def_rw("learning_jitter", &LayerSpec::learningJitter)
        .def_rw("alpha_jitter", &LayerSpec::alphaJitter)
        .def_rw("window", &LayerSpec::window)
        .def_rw("pool", &LayerSpec::pool)
        .def_rw("retina_spec", &LayerSpec::retina)
        .def_rw("cochlea_spec", &LayerSpec::cochlea)
        .def_static(
            "dense",
            [](size_t size, bool habituation, bool er, bool frozen, const Jitter& rj, const Jitter& lj,
               const Jitter& aj) { return withOptions(LayerSpec::Dense(size, habituation, er), frozen, rj, lj, aj); },
            "size"_a, "habituation"_a = true, "er"_a = true, nb::kw_only(), "frozen"_a = false,
            "recovery_jitter"_a = noJitter, "learning_jitter"_a = noJitter, "alpha_jitter"_a = noJitter)
        .def_static(
            "conv2d",
            [](size_t channels, const Window2D& window, bool habituation, bool er, bool frozen, const Jitter& rj,
               const Jitter& lj, const Jitter& aj) {
                return withOptions(LayerSpec::Conv2D(channels, window, habituation, er), frozen, rj, lj, aj);
            },
            "channels"_a, "window"_a, "habituation"_a = true, "er"_a = true, nb::kw_only(), "frozen"_a = false,
            "recovery_jitter"_a = noJitter, "learning_jitter"_a = noJitter, "alpha_jitter"_a = noJitter)
        .def_static(
            "locally_connected2d",
            [](size_t channels, const Window2D& window, bool habituation, bool er, bool frozen, const Jitter& rj,
               const Jitter& lj, const Jitter& aj) {
                return withOptions(LayerSpec::LocallyConnected2D(channels, window, habituation, er), frozen, rj, lj,
                                   aj);
            },
            "channels"_a, "window"_a, "habituation"_a = true, "er"_a = true, nb::kw_only(), "frozen"_a = false,
            "recovery_jitter"_a = noJitter, "learning_jitter"_a = noJitter, "alpha_jitter"_a = noJitter)
        .def_static("pool2d", &LayerSpec::Pool2D, "window"_a, "mode"_a = PoolMode::Max)
        .def_static(
            "retina",
            [](const RetinaSpec& retina, bool habituation, bool er, bool frozen, const Jitter& rj, const Jitter& lj,
               const Jitter& aj) {
                return withOptions(LayerSpec::Retina(retina, habituation, er), frozen, rj, lj, aj);
            },
            "retina"_a, "habituation"_a = true, "er"_a = true, nb::kw_only(), "frozen"_a = false,
            "recovery_jitter"_a = noJitter, "learning_jitter"_a = noJitter, "alpha_jitter"_a = noJitter)
        .def_static(
            "cochlea",
            [](const CochleaSpec& cochlea, bool habituation, bool er, bool frozen, const Jitter& rj, const Jitter& lj,
               const Jitter& aj) {
                return withOptions(LayerSpec::Cochlea(cochlea, habituation, er), frozen, rj, lj, aj);
            },
            "cochlea"_a, "habituation"_a = true, "er"_a = true, nb::kw_only(), "frozen"_a = false,
            "recovery_jitter"_a = noJitter, "learning_jitter"_a = noJitter, "alpha_jitter"_a = noJitter)
        .def_static("history", &LayerSpec::History, "length"_a,
                    "The last `length` ticks of its sources side by side along the width, newest last.");

    // --- Filters (core/filters.hpp) ------------------------------------------
    nb::module_ fm = m.def_submodule("filters", "Fixed filter banks for Conv2D layers.");
    nb::enum_<filters::Polarity>(fm, "Polarity")
        .value("OnCentre", filters::Polarity::OnCentre)
        .value("OffCentre", filters::Polarity::OffCentre);
    nb::class_<filters::Filter>(fm, "Filter", "One single-channel filter.")
        .def_ro("height", &filters::Filter::height)
        .def_ro("width", &filters::Filter::width)
        .def_prop_ro("weights", [](const filters::Filter& f) { return toNumpy(f.weights, {f.height, f.width}); },
                     nb::rv_policy::move, "height x width weights (a copy)");
    fm.def("gaussian", &filters::gaussian, "size"_a, "sigma"_a, "gain"_a = 1.0f);
    fm.def("difference_of_gaussians", &filters::differenceOfGaussians, "size"_a, "centre_sigma"_a,
           "surround_sigma"_a, "polarity"_a = filters::Polarity::OnCentre, "gain"_a = 1.0f);
    fm.def("gabor", &filters::gabor, "size"_a, "orientation"_a, "wavelength"_a, "sigma"_a, "phase"_a = 0.0f,
           "aspect"_a = 1.0f, "gain"_a = 1.0f);
    fm.def(
        "gabor_bank",
        [](size_t size, size_t orientations, float wavelength, float sigma, std::optional<std::vector<float>> phases,
           float aspect, float gain) {
            if (!phases)
                return filters::gaborBank(size, orientations, wavelength, sigma, {0.0f, std::numbers::pi_v<float> / 2},
                                          aspect, gain);
            // gaborBank takes an initializer_list: build the bank phase by phase in the same order.
            std::vector<filters::Filter> bank;
            for (size_t o = 0; o < orientations; ++o)
                for (float phase : *phases)
                    bank.push_back(filters::gabor(size, std::numbers::pi_v<float> * static_cast<float>(o) /
                                                            static_cast<float>(orientations),
                                                  wavelength, sigma, phase, aspect, gain));
            return bank;
        },
        "size"_a, "orientations"_a, "wavelength"_a, "sigma"_a, "phases"_a = nb::none(), "aspect"_a = 1.0f,
        "gain"_a = 1.0f,
        "orientations evenly spaced in [0, pi), each with every phase (default 0 and pi/2), orientation-major.");
    fm.def("centre_surround_bank", &filters::centreSurroundBank, "size"_a, "centre_sigma"_a, "surround_sigma"_a,
           "gain"_a = 1.0f);

    // --- Network -------------------------------------------------------------
    nb::class_<network::Edge>(m, "Edge")
        .def_ro("source", &network::Edge::from)
        .def_ro("target", &network::Edge::to)
        .def_ro("kind", &network::Edge::kind)
        .def_ro("width", &network::Edge::width);

    nb::class_<network>(m, "Network", "A graph of layers, the sensors feeding it and the layers read as output.")
        .def(nb::init<>())

        // Building
        .def("add_layer", &network::addLayer, "name"_a, "spec"_a, "Adds a layer; returns its id.")
        .def("connect", &network::connect, "source"_a, "target"_a,
             "target reads source's output (only the neurons target has now).")
        .def("add_feedback", &network::addFeedback, "source"_a, "target"_a, "width"_a,
             "Adds width new neurons to target, reading source.")
        .def("add_inputs", nb::overload_cast<LayerId, size_t>(&network::addInputs), "target"_a, "count"_a,
             "Creates count sensors feeding target; returns the index of the first.")
        .def("add_inputs", nb::overload_cast<LayerId, const Shape&>(&network::addInputs), "target"_a, "image"_a,
             "Sensors for an image (channel after channel, row after row).")
        .def("add_output", &network::addOutput, "layer"_a)
        .def("freeze", &network::freeze, "layer"_a)
        .def("unfreeze", &network::unfreeze, "layer"_a)
        .def("is_frozen", &network::isFrozen, "layer"_a)
        .def("set_recovery_jitter", &network::setRecoveryJitter, "layer"_a, "jitter"_a)
        .def("set_learning_jitter", &network::setLearningJitter, "layer"_a, "jitter"_a)
        .def("set_alpha_jitter", &network::setAlphaJitter, "layer"_a, "jitter"_a)
        .def("set_update_order", &network::setUpdateOrder, "order"_a)
        .def("use_default_update_order", &network::useDefaultUpdateOrder)

        // Running
        .def("set_input", &network::setInput, "index"_a, "value"_a)
        .def(
            "set_inputs", [](network& net, const FloatIn& values) { net.setInputs(flat(values)); }, "values"_a,
            "Sets every sensor; values (any shape) must have input_count entries.")
        .def(
            "set_inputs", [](network& net, const std::vector<float>& values) { net.setInputs(values); }, "values"_a)
        .def("step", &network::step, nb::call_guard<nb::gil_scoped_release>(),
             "forward() on every layer, in update order.")
        .def("apply_reward", &network::applyReward, "reward"_a, "learning_rate"_a,
             nb::call_guard<nb::gil_scoped_release>(), "Every layer that is not frozen learns.")
        .def(
            "apply_reward_to",
            [](network& net, LayerId id, float reward, float learningRate) {
                if (!net.isFrozen(id))
                    net.getLayer(id).applyReward(reward, learningRate);
            },
            "layer"_a, "reward"_a, "learning_rate"_a, nb::call_guard<nb::gil_scoped_release>(),
            "Only this layer learns (unless frozen): a reward per output layer, e.g. one-vs-rest readouts.")
        .def("outputs", [](const network& net) { return toNumpy(net.outputs()); },
             "The output layers' values, concatenated.")
        .def(
            "run",
            [](network& net, const FloatIn& inputs, std::optional<FloatIn> rewards, float learningRate) {
                const size_t ticks = inputs.ndim() == 0 ? 0 : inputs.shape(0);
                const size_t width = net.inputCount();
                if (inputs.ndim() != 2 || inputs.shape(1) != width)
                    throw std::invalid_argument("run: inputs must be ticks x " + std::to_string(width));
                if (rewards && rewards->size() != ticks)
                    throw std::invalid_argument("run: rewards must have one value per tick");
                std::vector<float> all;
                size_t outputs = 0;
                {
                    nb::gil_scoped_release release;
                    for (size_t t = 0; t < ticks; ++t) {
                        net.setInputs(std::span<const float>(inputs.data() + t * width, width));
                        net.step();
                        const std::vector<float> y = net.outputs();
                        outputs = y.size();
                        all.insert(all.end(), y.begin(), y.end());
                        if (rewards)
                            net.applyReward(rewards->data()[t], learningRate);
                    }
                }
                if (ticks == 0)
                    outputs = net.outputs().size();
                return toNumpy(std::move(all), {ticks, outputs});
            },
            "inputs"_a, "rewards"_a = nb::none(), "learning_rate"_a = 0.0f,
            "Runs one tick per row of inputs (ticks x input_count) and returns the outputs (ticks x outputs).\n"
            "With rewards (one per tick), apply_reward(rewards[t], learning_rate) follows each tick's step.")

        // Inspection
        .def_prop_ro("layer_count", &network::layerCount)
        .def_prop_ro("input_count", &network::inputCount)
        .def_prop_ro(
            "inputs",
            [](const network& net) {
                const auto in = net.inputs();
                return toNumpy(std::vector<float>(in.begin(), in.end()));
            },
            nb::rv_policy::move, "The sensor values (a copy).")
        .def("find_layer", &network::findLayer, "name"_a)
        .def("layer_name", &network::layerName, "layer"_a)
        .def("layer_spec", &network::layerSpec, "layer"_a)
        .def("layer_type", [](const network& net, LayerId id) { return net.getLayer(id).type(); }, "layer"_a)
        .def("layer_shape", [](const network& net, LayerId id) { return net.getLayer(id).shape(); }, "layer"_a)
        .def("layer_size", [](const network& net, LayerId id) { return net.getLayer(id).size(); }, "layer"_a)
        .def(
            "layer_output",
            [](const network& net, LayerId id) {
                const layer& l = net.getLayer(id);
                const auto out = l.output();
                const Shape s = l.shape();
                return toNumpy(std::vector<float>(out.begin(), out.end()), {s.channels, s.height, s.width});
            },
            "layer"_a, "The layer's current output as channels x height x width (a copy).")
        .def(
            "neuron_state",
            [](network& net, LayerId id) {
                const auto neurons = neuronLayer(net, id).neurons();
                std::vector<float> threshold, recovery, gain, alpha;
                for (const neuron& n : neurons) {
                    threshold.push_back(n.threshold());
                    recovery.push_back(n.recovery());
                    gain.push_back(n.learningGain());
                    alpha.push_back(n.alpha());
                }
                nb::dict d;
                d["threshold"] = toNumpy(std::move(threshold));
                d["recovery"] = toNumpy(std::move(recovery));
                d["learning_gain"] = toNumpy(std::move(gain));
                d["alpha"] = toNumpy(std::move(alpha));
                return d;
            },
            "layer"_a, "Per-neuron E-R threshold, recovery, learning gain and alpha (copies).")
        .def(
            "set_output", [](network& net, LayerId id, size_t index, float value) {
                neuronLayer(net, id).setOutput(index, value);
            },
            "layer"_a, "index"_a, "value"_a, "Drives one neuron's output by hand until its next forward().")
        .def(
            "weights", [](network& net, LayerId id, size_t index) {
                return toNumpy(layerAs<dense>(net, id, "Dense").weights(index));
            },
            "layer"_a, "neuron"_a, "A Dense neuron's weights, in input order (a copy).")
        .def(
            "set_weights", [](network& net, LayerId id, size_t index, const FloatIn& w) {
                const auto values = flat(w);
                layerAs<dense>(net, id, "Dense").setWeights(index, std::vector<float>(values.begin(), values.end()));
            },
            "layer"_a, "neuron"_a, "weights"_a)
        .def(
            "set_weights", [](network& net, LayerId id, size_t index, const std::vector<float>& w) {
                layerAs<dense>(net, id, "Dense").setWeights(index, w);
            },
            "layer"_a, "neuron"_a, "weights"_a)
        .def(
            "kernel", [](network& net, LayerId id, size_t channel) {
                return toNumpy(layerAs<conv2d>(net, id, "Conv2D").kernel(channel));
            },
            "layer"_a, "channel"_a, "A Conv2D output channel's kernel, in window order (a copy).")
        .def(
            "set_kernel", [](network& net, LayerId id, size_t channel, const FloatIn& w) {
                const auto values = flat(w);
                layerAs<conv2d>(net, id, "Conv2D").setKernel(channel, std::vector<float>(values.begin(), values.end()));
            },
            "layer"_a, "channel"_a, "weights"_a)
        .def(
            "set_kernel", [](network& net, LayerId id, size_t channel, const std::vector<float>& w) {
                layerAs<conv2d>(net, id, "Conv2D").setKernel(channel, w);
            },
            "layer"_a, "channel"_a, "weights"_a)
        .def(
            "load_filters", [](network& net, LayerId id, const std::vector<filters::Filter>& bank) {
                filters::load(layerAs<conv2d>(net, id, "Conv2D"), bank);
            },
            "layer"_a, "bank"_a, "Output channel k of a wired Conv2D gets bank[k] (see filters).")
        .def(
            "retina_points", [](network& net, LayerId id) {
                const retina& r = layerAs<retina>(net, id, "Retina");
                std::vector<float> xy;
                for (size_t k = 0; k < r.samples(); ++k) {
                    const retina::Point p = r.samplePoint(k);
                    xy.push_back(static_cast<float>(p.x));
                    xy.push_back(static_cast<float>(p.y));
                }
                return toNumpy(std::move(xy), {r.samples(), 2});
            },
            "layer"_a, "Where each retina sample is in the image: samples x (x, y), in pixels.")
        .def(
            "cochlea_bands", [](network& net, LayerId id) {
                const cochlea& c = layerAs<cochlea>(net, id, "Cochlea");
                std::vector<float> hz;
                for (const cochlea::Band& b : c.bands()) {
                    hz.push_back(b.lowFrequency);
                    hz.push_back(b.centreFrequency);
                    hz.push_back(b.highFrequency);
                }
                return toNumpy(std::move(hz), {c.bands().size(), 3});
            },
            "layer"_a, "Each band's (low, centre, high) frequency in Hz, lowest band first: bands x 3.")
        .def(
            "cochlea_power", [](network& net, LayerId id) {
                const cochlea& c = layerAs<cochlea>(net, id, "Cochlea");
                return toNumpy(std::vector<float>(c.power()), {c.power().size()});
            },
            "layer"_a, "The power spectrum of the last step: window / 2 + 1 bins (bin k is k * sample_rate / window Hz).")
        .def_prop_ro("edges", [](const network& net) { return net.edges(); })
        .def_prop_ro("output_layers", [](const network& net) { return net.outputLayers(); })
        .def_prop_ro("update_order", [](const network& net) { return net.updateOrder(); })
        .def("describe", [](const network& net) {
            std::ostringstream os;
            net.describe(os);
            return os.str();
        })
        .def("__repr__", [](const network& net) {
            return "<exrelaxer.Network: " + std::to_string(net.layerCount()) + " layers, " +
                   std::to_string(net.inputCount()) + " inputs>";
        })

        // Serialization
        .def(
            "to_bytes", [](const network& net) {
                std::ostringstream os(std::ios::binary);
                net.save(os);
                const std::string data = os.str();
                return nb::bytes(data.data(), data.size());
            },
            "The whole network (wiring and state), in the library's binary format.")
        .def_static(
            "from_bytes", [](nb::bytes data, DeserializeMode mode) {
                std::istringstream is(std::string(data.c_str(), data.size()), std::ios::binary);
                return network::load(is, mode);
            },
            "data"_a, "mode"_a = DeserializeMode::FullState)
        .def(
            "save", [](const network& net, const std::filesystem::path& path) {
                std::ofstream os(path, std::ios::binary);
                if (!os)
                    throw std::runtime_error("cannot write " + path.string());
                net.save(os);
            },
            "path"_a)
        .def_static(
            "load", [](const std::filesystem::path& path, DeserializeMode mode) {
                std::ifstream is(path, std::ios::binary);
                if (!is)
                    throw std::runtime_error("cannot read " + path.string());
                return network::load(is, mode);
            },
            "path"_a, "mode"_a = DeserializeMode::FullState);
}
