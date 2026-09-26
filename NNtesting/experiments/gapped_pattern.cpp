// Gapped pattern detection in a random stream: respond after
// A, {0..1 ticks}, B, {0..1 ticks}, C, and not after decoys (see
// NNtesting/tasks/pattern_benchmark.hpp and doc/pattern_benchmark.md).
//
// Every topology starts with frozen value detectors (the symbols as +-5
// features); `topology` chooses what memory and learning come after:
//   window_readout  frozen delay window (lags 0..depth) -> learned readout
//   window_mix      frozen delay window -> frozen random mix -> learned readout
//   reservoir       frozen random reservoir (recurrent) -> learned readout
#include <stdexcept>
#include "experiment.hpp"
#include "pattern_benchmark.hpp"

namespace {

using namespace pattern_benchmark;

NetworkBuilder builder(const nnt::Params& p)
{
    const std::string topology = p.getString("topology");
    const int depth = static_cast<int>(p.getInt("window_depth"));
    const size_t readout = static_cast<size_t>(p.getInt("readout"));
    if (topology == "window_readout")
        return [=](bool hasER) {
            auto net = std::make_unique<network>();
            FrontEnd f = addFrontEnd(*net, depth);
            const auto out = net->addLayer("out", LayerSpec::Dense(readout, false, hasER));
            net->connect(f.features, out);
            finishFrontEnd(*net, f, {out}, out);
            return net;
        };
    if (topology == "window_mix") {
        const size_t mix = static_cast<size_t>(p.getInt("mix"));
        return [=](bool hasER) {
            auto net = std::make_unique<network>();
            FrontEnd f = addFrontEnd(*net, depth);
            const auto mixer = net->addLayer("mix", {LayerType::Dense, mix, false, false, /*frozen*/ true});
            const auto out = net->addLayer("out", LayerSpec::Dense(readout, false, hasER));
            net->connect(f.features, mixer);
            net->connect(mixer, out);
            finishFrontEnd(*net, f, {mixer, out}, out);
            return net;
        };
    }
    if (topology == "reservoir") {
        const size_t in = static_cast<size_t>(p.getInt("reservoir_in"));
        const size_t rec = static_cast<size_t>(p.getInt("reservoir_rec"));
        const float scale = static_cast<float>(p.getDouble("recurrent_scale"));
        // E-R applies to the reservoir; the readout runs without it.
        return [=](bool hasER) {
            auto net = std::make_unique<network>();
            FrontEnd f = addFrontEnd(*net, 0);
            const auto res = addReservoir(*net, f.features, BAND_FEATURES, in, rec, 3.0f, scale, hasER);
            const auto out = net->addLayer("out", LayerSpec::Dense(readout, false, false));
            net->connect(res, out);
            finishFrontEnd(*net, f, {res, out}, out);
            return net;
        };
    }
    throw std::invalid_argument("topology must be window_readout, window_mix or reservoir");
}

nnt::Register experiment({
    .name = "gapped_pattern",
    .description = "respond after A..B..C with gaps, not after decoys; frozen detectors + memory + learned readout",
    .tags = {"temporal", "learning"},
    .params = {
        {"topology", "window_readout", "window_readout, window_mix or reservoir"},
        {"er", "false", "E-R in the learned readout (in the reservoir for topology=reservoir)"},
        {"window_depth", "5", "window lags (window topologies)"},
        {"mix", "128", "frozen mixing neurons (window_mix)"},
        {"reservoir_in", "40", "reservoir neurons reading the detectors"},
        {"reservoir_rec", "60", "recurrent reservoir neurons"},
        {"recurrent_scale", "2.0", "recurrent weight scale (reservoir)"},
        {"readout", "8", "learned readout neurons"},
        {"train_ticks", "10000", "training stream length"},
        {"lr", "0.005", "learning rate"},
        {"reward", "error", "error: reward only wrong answers; target: reward every tick"},
    },
    .trials = 20,
    .expect = {{.metric = "after_c", .min = 0.9}},
    .run = [](nnt::Trial& t) {
        const nnt::Params& p = t.params();
        const NetworkBuilder build = builder(p);
        const bool er = p.getBool("er");
        const size_t ticks = static_cast<size_t>(p.getInt("train_ticks"));
        const float lr = static_cast<float>(p.getDouble("lr"));
        const std::string reward = p.getString("reward");
        if (reward != "error" && reward != "target")
            throw std::invalid_argument("reward must be error or target");
        const RewardMode mode = reward == "error" ? RewardMode::Error : RewardMode::Target;

        // runPatternTrial reseeds with the trial seed itself.
        const PatternScore trained = runPatternTrial(build, t.seed(), er, lr, ticks, mode);
        const PatternScore control = runPatternTrial(build, t.seed(), er, 0.0f, ticks, mode);
        t.record("balanced", trained.balanced);
        t.record("after_c", trained.afterC);
        t.record("balanced_control", control.balanced);
        t.record("after_c_control", control.afterC);
        t.record("after_c_gain", trained.afterC - control.afterC);
        t.record("diverged", trained.diverged ? 1.0 : 0.0);
    },
});

} // namespace
