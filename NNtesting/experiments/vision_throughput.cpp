// Speed of a camera pipeline, layer by layer:
//
//   retina (grid) -> Conv2D c1 -> Pool2D 2x2 -> Conv2D c2 -> Pool2D 4x4 -> LocallyConnected2D -> dense
//
// Machine-dependent, so no checks; compare runs with tools/compare.py.
#include <cmath>
#include "experiment.hpp"
#include "network.hpp"

namespace {

using namespace exr;

nnt::Register experiment({
    .name = "vision_throughput",
    .description = "time per step and per layer of a camera pipeline (default 320 x 200)",
    .tags = {"performance", "vision"},
    .params = {
        {"width", "320", "image width"},
        {"height", "200", "image height"},
        {"c1", "16", "first Conv2D channels"},
        {"c2", "32", "second Conv2D channels"},
        {"kernel", "5", "Conv2D kernel size (same padding)"},
        {"er", "true", "E-R in the retina and the Conv2D layers"},
    },
    .trials = 3,
    .run = [](nnt::Trial& t) {
        const nnt::Params& p = t.params();
        const Shape image{1, static_cast<size_t>(p.getInt("height")), static_cast<size_t>(p.getInt("width"))};
        const size_t k = static_cast<size_t>(p.getInt("kernel"));
        const bool er = p.getBool("er");
        network net;
        const auto eye = net.addLayer("eye", LayerSpec::Retina({image}, er, er));
        const auto v1 = net.addLayer("v1", LayerSpec::Conv2D(static_cast<size_t>(p.getInt("c1")),
                                                             Window2D::square(k, 1, k / 2), false, er));
        const auto p1 = net.addLayer("p1", LayerSpec::Pool2D(Window2D::square(2, 2)));
        const auto v2 = net.addLayer("v2", LayerSpec::Conv2D(static_cast<size_t>(p.getInt("c2")),
                                                             Window2D::square(k, 1, k / 2), false, er));
        const auto p2 = net.addLayer("p2", LayerSpec::Pool2D(Window2D::square(4, 4)));
        const auto lc = net.addLayer("lc", LayerSpec::LocallyConnected2D(16, Window2D::square(3, 1, 1), false, er));
        const auto out = net.addLayer("out", LayerSpec::Dense(10, false, false));
        net.addInputs(eye, image);
        net.connect(eye, v1);
        net.connect(v1, p1);
        net.connect(p1, v2);
        net.connect(v2, p2);
        net.connect(p2, lc);
        net.connect(lc, out);
        net.addOutput(out);

        std::vector<float> frame(image.size());
        for (size_t i = 0; i < frame.size(); ++i)
            frame[i] = 0.5f + 0.5f * std::sin(0.01f * static_cast<float>(i));
        net.setInputs(frame);
        for (int i = 0; i < 3; ++i)
            net.step();

        for (auto [id, name] : {std::pair{eye, "retina"}, std::pair{v1, "conv1"}, std::pair{p1, "pool1"},
                                std::pair{v2, "conv2"}, std::pair{p2, "pool2"}, std::pair{lc, "lc"}})
            t.record(std::string(name) + "_forward_ms", nnt::Trial::bestMs(5, 3, [&] { net.getLayer(id).forward(); }));
        t.record("step_ms", nnt::Trial::bestMs(5, 3, [&] { net.step(); }));
        t.record("step_reward_ms", nnt::Trial::bestMs(5, 3, [&] { net.step(); net.applyReward(1.0f, 0.001f); }));
    },
});

} // namespace
