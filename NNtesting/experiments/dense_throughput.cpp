// Speed of a dense layer: n neurons reading n neurons, step and step +
// learning. Machine-dependent, so no checks; compare runs with
// tools/compare.py.
#include "experiment.hpp"
#include "network.hpp"

namespace {

using namespace exr;

nnt::Register experiment({
    .name = "dense_throughput",
    .description = "time per step and per step + reward of an n x n dense layer",
    .tags = {"performance", "quick"},
    .params = {
        {"n", "1000", "neurons in each of the two layers"},
        {"er", "true", "E-R in the layers"},
        {"iterations", "200", "steps per timed repeat"},
    },
    .trials = 3,
    .run = [](nnt::Trial& t) {
        const nnt::Params& p = t.params();
        const size_t n = static_cast<size_t>(p.getInt("n"));
        const bool er = p.getBool("er");
        const int iterations = static_cast<int>(p.getInt("iterations"));
        network net;
        const auto a = net.addLayer("a", LayerSpec::Dense(n, false, er));
        const auto b = net.addLayer("b", LayerSpec::Dense(n, false, er));
        net.addInputs(a, 1);
        net.connect(a, b);
        net.setInputs({0.5f});
        const double step = nnt::Trial::bestMs(5, iterations, [&] { net.step(); });
        const double both = nnt::Trial::bestMs(5, iterations, [&] { net.step(); net.applyReward(1.0f, 0.001f); });
        t.record("step_ms", step);
        t.record("step_reward_ms", both);
        t.record("gmac_per_s", static_cast<double>(n) * n / (step * 1e6));
    },
});

} // namespace
