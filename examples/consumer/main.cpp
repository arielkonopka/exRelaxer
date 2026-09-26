// A program using the installed library: a network learns the sign of its
// input, is saved and loaded, and hears a tone through a cochlea.
#include <cmath>
#include <cstdio>
#include <numbers>
#include <sstream>
#include <vector>
#include "network.hpp"
#include "layers/cochlea.hpp"

int main()
{
    using namespace exr;
    reseed(1);

    // Learn: output the sign of the input (error-driven reward).
    network net;
    const auto in = net.addLayer("in", LayerSpec::Dense(8, false, false));
    const auto out = net.addLayer("out", LayerSpec::Dense(1, false, false));
    net.addInputs(in, 1);
    net.connect(in, out);
    net.addOutput(out);
    auto answer = [&](float x) {
        net.setInputs({x});
        net.step();
        return net.outputs()[0];
    };
    for (int t = 0; t < 400; ++t) {
        const float x = t % 2 ? 1.0f : -1.0f;
        if (answer(x) * x <= 0.0f)
            net.applyReward(x, 0.01f);
    }
    std::stringstream saved;
    net.save(saved);
    auto loaded = network::load(saved);
    loaded->setInputs({1.0f});
    loaded->step();
    const bool learned = answer(1.0f) > 0.0f && answer(-1.0f) < 0.0f && loaded->outputs()[0] > 0.0f;

    // Hear: a 1 kHz tone lands in the band around 1 kHz.
    network ear_net;
    CochleaSpec spec;
    spec.sampleRate = 8000.0f;
    spec.hop = 80;
    spec.window = 256;
    spec.bands = 16;
    const auto ear = ear_net.addLayer("ear", LayerSpec::Cochlea(spec, false, false));
    ear_net.addInputs(ear, spec.hop);
    ear_net.addOutput(ear);
    std::vector<float> sound(spec.hop);
    for (size_t t = 0; t < 10; ++t) {
        for (size_t n = 0; n < spec.hop; ++n)
            sound[n] = 0.5f * static_cast<float>(std::sin(2.0 * std::numbers::pi * 1000.0 *
                                                          static_cast<double>(t * spec.hop + n) / spec.sampleRate));
        ear_net.setInputs(sound);
        ear_net.step();
    }
    const std::vector<float> bands = ear_net.outputs();
    size_t loudest = 0;
    for (size_t b = 1; b < bands.size(); ++b)
        if (bands[b] > bands[loudest])
            loudest = b;
    const auto& band = ear_net.layerAs<cochlea>(ear).bands()[loudest];
    const bool heard = band.lowFrequency <= 1000.0f && 1000.0f <= band.highFrequency;

    std::printf("learned the sign: %s; 1 kHz heard in band %zu (%.0f-%.0f Hz): %s\n", learned ? "yes" : "no",
                loudest, band.lowFrequency, band.highFrequency, heard ? "yes" : "no");
    return learned && heard ? 0 : 1;
}
