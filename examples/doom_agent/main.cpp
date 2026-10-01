// Loads the evolved Doom agent and runs one game step on a blank screen and silence.
#include <cstdio>
#include <fstream>
#include <iostream>
#include <vector>
#include "network.hpp"

int main(int argc, char** argv)
{
    const char* path = argc > 1 ? argv[1] : "results/dynamic/doom_agent/untildeath_d1/best.exr";
    std::ifstream is(path, std::ios::binary);
    if (!is) {
        std::fprintf(stderr, "cannot read %s\n", path);
        return 1;
    }
    auto net = exr::network::load(is);
    net->describe(std::cout);

    std::vector<float> eye(300, 0.5f);  // 160 x 120 gray pooled 8 x 8 to 20 x 15, scaled to 0..1
    std::vector<float> mic(420, 0.0f);  // 210 left + 210 right samples per tick (1260 per ear per step / 6 ticks)
    net->setInputs("eye", eye);
    for (int tick = 0; tick < 6; ++tick) {  // 6 network ticks per game step
        net->setInputs("mic", mic);
        net->step();
    }
    const std::vector<float> y = net->outputs();  // forward, backward, turn left/right, strafe left/right, attack, use
    size_t best = 0;
    for (size_t i = 0; i < y.size(); ++i) {
        std::printf("%zu: %+.4f\n", i, y[i]);
        if (y[i] > y[best])
            best = i;
    }
    std::printf("action: %zu\n", best);
}
