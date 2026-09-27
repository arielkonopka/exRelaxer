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
#include "layers/disparity.hpp"
#include "layers/resize2d.hpp"

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
                      const Jitter& alpha, const LearningRule& rule = {})
{
    spec.learningRule = rule;
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
        .value("History", LayerType::History)
        .value("Resize2D", LayerType::Resize2D)
        .value("Disparity", LayerType::Disparity);
    nb::enum_<FrequencyScale>(m, "FrequencyScale")
        .value("Mel", FrequencyScale::Mel)
        .value("Linear", FrequencyScale::Linear);
    nb::enum_<Compression>(m, "Compression").value("Log", Compression::Log).value("Linear", Compression::Linear);
    nb::enum_<PoolMode>(m, "PoolMode").value("Max", PoolMode::Max).value("Average", PoolMode::Average);
    nb::enum_<Sampling>(m, "Sampling").value("Grid", Sampling::Grid).value("Spiral", Sampling::Spiral);
    nb::enum_<Interpolation>(m, "Interpolation")
        .value("Nearest", Interpolation::Nearest)
        .value("Bilinear", Interpolation::Bilinear)
        .value("Area", Interpolation::Area);
    nb::enum_<DisparityMeasure>(m, "DisparityMeasure")
        .value("Correlation", DisparityMeasure::Correlation)
        .value("Difference", DisparityMeasure::Difference)
        .value("Normalized", DisparityMeasure::Normalized);
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
                float maxFrequency, FrequencyScale scale, Compression compression, float gain, size_t channels) {
                 new (c) CochleaSpec{sampleRate, hop, window, bands, minFrequency, maxFrequency, scale, compression,
                                     gain, channels};
             },
             "sample_rate"_a = 16000.0f, "hop"_a = 160, "window"_a = 512, "bands"_a = 40, "min_frequency"_a = 50.0f,
             "max_frequency"_a = 0.0f, "scale"_a = FrequencyScale::Mel, "compression"_a = Compression::Log,
             "gain"_a = 100.0f, "channels"_a = 1)
        .def_rw("sample_rate", &CochleaSpec::sampleRate)
        .def_rw("hop", &CochleaSpec::hop)
        .def_rw("window", &CochleaSpec::window)
        .def_rw("bands", &CochleaSpec::bands)
        .def_rw("min_frequency", &CochleaSpec::minFrequency)
        .def_rw("max_frequency", &CochleaSpec::maxFrequency)
        .def_rw("scale", &CochleaSpec::scale)
        .def_rw("compression", &CochleaSpec::compression)
        .def_rw("gain", &CochleaSpec::gain)
        .def_rw("channels", &CochleaSpec::channels)
        .def("__eq__", [](const CochleaSpec& a, const CochleaSpec& b) { return a == b; });

    nb::class_<ResizeSpec>(m, "ResizeSpec", "Resize2D: every input channel resampled to height x width.")
        .def("__init__",
             [](ResizeSpec* r, size_t height, size_t width, Interpolation interpolation) {
                 new (r) ResizeSpec{height, width, interpolation};
             },
             "height"_a, "width"_a, "interpolation"_a = Interpolation::Bilinear)
        .def_rw("height", &ResizeSpec::height)
        .def_rw("width", &ResizeSpec::width)
        .def_rw("interpolation", &ResizeSpec::interpolation)
        .def("__eq__", [](const ResizeSpec& a, const ResizeSpec& b) { return a == b; });

    nb::class_<DisparitySpec>(m, "DisparitySpec",
                              "Disparity: channel d compares left pixel x with right pixel x - (min_disparity + d) "
                              "over a window x window box.")
        .def("__init__",
             [](DisparitySpec* d, int minDisparity, int maxDisparity, size_t window, DisparityMeasure measure) {
                 new (d) DisparitySpec{minDisparity, maxDisparity, window, measure};
             },
             "min_disparity"_a = 0, "max_disparity"_a = 4, "window"_a = 3,
             "measure"_a = DisparityMeasure::Correlation)
        .def_rw("min_disparity", &DisparitySpec::minDisparity)
        .def_rw("max_disparity", &DisparitySpec::maxDisparity)
        .def_rw("window", &DisparitySpec::window)
        .def_rw("measure", &DisparitySpec::measure)
        .def_prop_ro("count", &DisparitySpec::count)
        .def("__eq__", [](const DisparitySpec& a, const DisparitySpec& b) { return a == b; });

    // --- Jitter --------------------------------------------------------------
    nb::class_<Habituation>(m, "Habituation",
                            "How habituation suppresses a repeated input (the same sum within `tolerance`, "
                            "relative): decay 0 cuts it after `steps` ticks; decay > 0 fades it by decay^ticks "
                            "from the `fade_after`-th tick on.")
        .def(nb::init<>())
        .def("__init__",
             [](Habituation* h, std::uint32_t steps, float tolerance, float decay, std::uint32_t fadeAfter) {
                 new (h) Habituation{steps, tolerance, decay, fadeAfter};
             },
             "steps"_a = habituation_steps, "tolerance"_a = 0.0f, "decay"_a = 0.0f, "fade_after"_a = 2u)
        .def_rw("fade_after", &Habituation::fadeAfter)
        .def_rw("steps", &Habituation::steps)
        .def_rw("tolerance", &Habituation::tolerance)
        .def_rw("decay", &Habituation::decay)
        .def("__eq__", [](const Habituation& a, const Habituation& b) { return a == b; });

    nb::class_<Spontaneous>(m, "Spontaneous",
                            "Spontaneous E-R firing: a silent neuron fires with a value in [-amplitude, amplitude] "
                            "once its threshold is <= `below`, and with probability `rate` on any silent tick.")
        .def(nb::init<>())
        .def("__init__",
             [](Spontaneous* s, float below, float amplitude, float rate) { new (s) Spontaneous{below, amplitude, rate}; },
             "below"_a = min_threshold, "amplitude"_a = spontaneous_min_amplitude, "rate"_a = 0.0f)
        .def_rw("below", &Spontaneous::below)
        .def_rw("amplitude", &Spontaneous::amplitude)
        .def_rw("rate", &Spontaneous::rate)
        .def("__eq__", [](const Spontaneous& a, const Spontaneous& b) { return a == b; });

    nb::class_<ThresholdGrowth> growth(m, "ThresholdGrowth",
                                       "How an E-R threshold grows on firing with magnitude s: LINEAR thr + amount*(s - thr) "
                                       "(default), LOG thr + alpha*ln(s/thr) (the original), FIXED thr + amount, "
                                       "MULTIPLICATIVE thr*(1 + amount).");
    nb::enum_<ThresholdGrowth::Rule>(growth, "Rule")
        .value("LOG", ThresholdGrowth::Rule::Log)
        .value("LINEAR", ThresholdGrowth::Rule::Linear)
        .value("FIXED", ThresholdGrowth::Rule::Fixed)
        .value("MULTIPLICATIVE", ThresholdGrowth::Rule::Multiplicative);
    growth.def(nb::init<>())
        .def("__init__",
             [](ThresholdGrowth* g, ThresholdGrowth::Rule rule, float amount) { new (g) ThresholdGrowth{rule, amount}; },
             "rule"_a = ThresholdGrowth::Rule::Linear, "amount"_a = 0.5f)
        .def_rw("rule", &ThresholdGrowth::rule)
        .def_rw("amount", &ThresholdGrowth::amount)
        .def("__eq__", [](const ThresholdGrowth& a, const ThresholdGrowth& b) { return a == b; });

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

    // --- LearningRule (core/learning.hpp) ---------------------------------------
    nb::enum_<LearningRuleType>(m, "LearningRuleType")
        .value("Sign", LearningRuleType::Sign)
        .value("Trace", LearningRuleType::Trace)
        .value("FeedbackAlignment", LearningRuleType::FeedbackAlignment)
        .value("Perturbation", LearningRuleType::Perturbation)
        .value("Oja", LearningRuleType::Oja)
        .value("BCM", LearningRuleType::BCM);
    nb::class_<LearningRule>(m, "LearningRule",
                             "How a layer learns: sign (default), trace, feedback_alignment, perturbation, oja or bcm.")
        .def(nb::init<>())
        .def_rw("type", &LearningRule::type)
        .def_rw("bias", &LearningRule::bias)
        .def_rw("decay", &LearningRule::decay)
        .def_rw("trace", &LearningRule::trace)
        .def_rw("baseline", &LearningRule::baseline)
        .def_rw("noise", &LearningRule::noise)
        .def_rw("bcm_rate", &LearningRule::bcmRate)
        .def_rw("winners", &LearningRule::winners)
        .def_static("sign", &LearningRule::sign, "The original rule: rate * gain * reward * eligibility * sign(input).")
        .def_static("traced", &LearningRule::traced, "trace"_a = 0.0f, "baseline"_a = 0.05f,
                    "Graded three-factor rule: (reward - baseline) * output trace * input trace.")
        .def_static("feedback_alignment", &LearningRule::feedbackAlignment, "trace"_a = 0.0f,
                    "Per-neuron credit from Network.apply_error through a fixed random feedback matrix.")
        .def_static("perturbation", &LearningRule::perturbation, "noise"_a = 0.1f, "trace"_a = 0.0f,
                    "baseline"_a = 0.05f, "Node perturbation: exploration noise correlated with the reward.")
        .def_static("oja", &LearningRule::oja, "winners"_a = 0, "Unsupervised Oja rule (normalised Hebbian).")
        .def_static("bcm", &LearningRule::bcm, "bcm_rate"_a = 0.01f, "winners"_a = 0, "decay"_a = 0.0f,
                    "Unsupervised BCM rule with a sliding threshold.")
        .def("with_bias", &LearningRule::withBias, "on"_a = true)
        .def("with_decay", &LearningRule::withDecay, "decay"_a)
        .def_prop_ro("unsupervised", &LearningRule::unsupervised)
        .def("validate", &LearningRule::validate)
        .def("__eq__", [](const LearningRule& a, const LearningRule& b) { return a == b; })
        .def("__repr__", [](const LearningRule& r) { return "<LearningRule " + describeLearningRule(r) + ">"; });

    // --- LayerSpec -------------------------------------------------------------
    const Jitter noJitter{};
    const LearningRule signRule{};
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
        .def_rw("learning_rule", &LayerSpec::learningRule)
        .def_rw("gate", &LayerSpec::gate,
                "Neurons without E-R: fixed firing threshold (|sum| <= gate gives 0); 0 = linear.")
        .def_rw("habituation_rule", &LayerSpec::habituationRule,
                "How habituation suppresses repeated inputs (Habituation; default: cut after 100 exact repeats).")
        .def_rw("spontaneous", &LayerSpec::spontaneous,
                "When and how strongly E-R neurons fire on their own (Spontaneous; default: the original).")
        .def_rw("threshold_growth", &LayerSpec::thresholdGrowth,
                "How E-R thresholds grow on firing (ThresholdGrowth; default: linear, amount 0.5).")
        .def_rw("rectify", &LayerSpec::rectify,
                "Neurons without E-R: ReLU, only sums above the gate pass (the rest give 0).")
        .def_rw("normalize", &LayerSpec::normalize,
                "Dense, Conv2D, LocallyConnected2D: divide each weighted sum by the length of the neuron's weights.")
        .def_rw("window", &LayerSpec::window)
        .def_rw("pool", &LayerSpec::pool)
        .def_rw("retina_spec", &LayerSpec::retina)
        .def_rw("cochlea_spec", &LayerSpec::cochlea)
        .def_rw("resize_spec", &LayerSpec::resize)
        .def_rw("disparity_spec", &LayerSpec::disparity)
        .def_static(
            "dense",
            [](size_t size, bool habituation, bool er, bool frozen, const Jitter& rj, const Jitter& lj,
               const Jitter& aj, const LearningRule& rule) {
                return withOptions(LayerSpec::Dense(size, habituation, er), frozen, rj, lj, aj, rule);
            },
            "size"_a, "habituation"_a = true, "er"_a = true, nb::kw_only(), "frozen"_a = false,
            "recovery_jitter"_a = noJitter, "learning_jitter"_a = noJitter, "alpha_jitter"_a = noJitter,
            "learning_rule"_a = signRule)
        .def_static(
            "conv2d",
            [](size_t channels, const Window2D& window, bool habituation, bool er, bool frozen, const Jitter& rj,
               const Jitter& lj, const Jitter& aj, const LearningRule& rule) {
                return withOptions(LayerSpec::Conv2D(channels, window, habituation, er), frozen, rj, lj, aj, rule);
            },
            "channels"_a, "window"_a, "habituation"_a = true, "er"_a = true, nb::kw_only(), "frozen"_a = false,
            "recovery_jitter"_a = noJitter, "learning_jitter"_a = noJitter, "alpha_jitter"_a = noJitter,
            "learning_rule"_a = signRule)
        .def_static(
            "locally_connected2d",
            [](size_t channels, const Window2D& window, bool habituation, bool er, bool frozen, const Jitter& rj,
               const Jitter& lj, const Jitter& aj, const LearningRule& rule) {
                return withOptions(LayerSpec::LocallyConnected2D(channels, window, habituation, er), frozen, rj, lj,
                                   aj, rule);
            },
            "channels"_a, "window"_a, "habituation"_a = true, "er"_a = true, nb::kw_only(), "frozen"_a = false,
            "recovery_jitter"_a = noJitter, "learning_jitter"_a = noJitter, "alpha_jitter"_a = noJitter,
            "learning_rule"_a = signRule)
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
                    "The last `length` ticks of its sources side by side along the width, newest last.")
        .def_static("resize2d", &LayerSpec::Resize2D, "height"_a, "width"_a,
                    "interpolation"_a = Interpolation::Bilinear,
                    "Every channel of its sources resampled to height x width.")
        .def_static("disparity", &LayerSpec::Disparity, "disparity"_a,
                    "Stereo matching: the first source joined is the left view, the second the right.");

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
        .def("add_inputs", nb::overload_cast<LayerId, size_t, const std::string&>(&network::addInputs), "target"_a,
             "count"_a, "name"_a = "",
             "Creates count sensors feeding target; returns the index of the first. A name makes them a named "
             "input source (set_inputs(name, values), connect_inputs).")
        .def("add_inputs", nb::overload_cast<LayerId, const Shape&, const std::string&>(&network::addInputs),
             "target"_a, "image"_a, "name"_a = "", "Sensors for an image (channel after channel, row after row).")
        .def("connect_inputs", &network::connectInputs, "name"_a, "target"_a,
             "The named input source also feeds target.")
        .def("add_output", &network::addOutput, "layer"_a)
        .def("freeze", &network::freeze, "layer"_a)
        .def("unfreeze", &network::unfreeze, "layer"_a)
        .def("is_frozen", &network::isFrozen, "layer"_a)
        .def("set_recovery_jitter", &network::setRecoveryJitter, "layer"_a, "jitter"_a)
        .def("set_learning_jitter", &network::setLearningJitter, "layer"_a, "jitter"_a)
        .def("set_alpha_jitter", &network::setAlphaJitter, "layer"_a, "jitter"_a)
        .def("set_learning_rule", &network::setLearningRule, "layer"_a, "rule"_a,
             "Changes a layer's learning rule; resets the rule's state, keeps the weights.")
        .def("set_update_order", &network::setUpdateOrder, "order"_a)
        .def("use_default_update_order", &network::useDefaultUpdateOrder)

        // Running
        .def("set_input", &network::setInput, "index"_a, "value"_a)
        .def(
            "set_inputs", [](network& net, const FloatIn& values) { net.setInputs(flat(values)); }, "values"_a,
            "Sets every sensor; values (any shape) must have input_count entries.")
        .def(
            "set_inputs", [](network& net, const std::vector<float>& values) { net.setInputs(values); }, "values"_a)
        .def(
            "set_inputs",
            [](network& net, const std::string& name, const FloatIn& values) { net.setInputs(name, flat(values)); },
            "name"_a, "values"_a, "Sets the named input source; values (any shape) must have its size.")
        .def(
            "set_inputs",
            [](network& net, const std::string& name, const std::vector<float>& values) { net.setInputs(name, values); },
            "name"_a, "values"_a)
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
        .def(
            "apply_error", [](network& net, const FloatIn& errors, float learningRate) {
                const auto e = flat(errors);
                nb::gil_scoped_release release;
                net.applyError(e, learningRate);
            },
            "errors"_a, "learning_rate"_a,
            "Learning from an error per output (desired - actual): output layers learn their own errors,\n"
            "hidden feedback-alignment layers a random projection of them (see network.hpp).")
        .def(
            "apply_error", [](network& net, const std::vector<float>& errors, float learningRate) {
                nb::gil_scoped_release release;
                net.applyError(errors, learningRate);
            },
            "errors"_a, "learning_rate"_a)
        .def(
            "apply_modulators_to",
            [](network& net, LayerId id, const FloatIn& modulators, float learningRate) {
                const auto m = flat(modulators);
                neuron_layer& target = neuronLayer(net, id);
                if (net.isFrozen(id))
                    return;
                nb::gil_scoped_release release;
                target.applyModulators(m, learningRate);
            },
            "layer"_a, "modulators"_a, "learning_rate"_a,
            "Only this layer learns (unless frozen), with its own reward per neuron.")
        .def(
            "bias", [](network& net, LayerId id) {
                const neuron_layer& l = neuronLayer(net, id);
                std::vector<float> b(l.size());
                for (size_t i = 0; i < b.size(); ++i)
                    b[i] = l.bias(i);
                return toNumpy(std::move(b));
            },
            "layer"_a, "Each neuron's learned bias (zeros when the rule has none).")
        .def(
            "set_bias", [](network& net, LayerId id, size_t index, float value) {
                neuronLayer(net, id).setBias(index, value);
            },
            "layer"_a, "neuron"_a, "value"_a)
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
        .def(
            "input_values",
            [](const network& net, const std::string& name) {
                const auto in = net.inputs(name);
                return toNumpy(std::vector<float>(in.begin(), in.end()));
            },
            "name"_a, nb::rv_policy::move, "The named input source's sensor values (a copy).")
        .def_prop_ro(
            "input_sources",
            [](const network& net) {
                nb::list sources;
                for (const network::InputSource& s : net.inputSources()) {
                    nb::dict source;
                    source["name"] = s.name;
                    source["first"] = s.first;
                    source["shape"] = s.shape;
                    source["targets"] = s.targets;
                    sources.append(source);
                }
                return sources;
            },
            "Every addInputs block: dicts with name ('' if unnamed), first sensor index, shape and target layers.")
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
                const size_t channels = c.spec().channels;
                if (channels == 1)
                    return toNumpy(std::vector<float>(c.power()), {c.power().size()});
                return toNumpy(std::vector<float>(c.power()), {channels, c.power().size() / channels});
            },
            "layer"_a,
            "The power spectrum of the last step: window / 2 + 1 bins (bin k is k * sample_rate / window Hz); "
            "channels x bins with several microphones.")
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
