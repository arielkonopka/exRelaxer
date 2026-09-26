// Runs a Spiral retina (no habituation, no E-R: output = the sampled sums) on a raw
// float32 image and writes its outputs and sample positions; check_spiral.py
// compares them with spiral.py.
// Usage: retina_dump C H W spacing radius image.f32 values.f32 points.f64
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <vector>
#include "layers/retina.hpp"
using namespace exr;
int main(int, char** argv) {
    const Shape shape{std::stoul(argv[1]), std::stoul(argv[2]), std::stoul(argv[3])};
    retina eye(RetinaSpec{shape, Sampling::Spiral, std::stof(argv[4]), std::stof(argv[5])}, false, false);
    std::vector<float> image(shape.size());
    std::ifstream(argv[6], std::ios::binary).read(reinterpret_cast<char*>(image.data()), image.size() * 4);
    eye.attachInputs(image);
    eye.forward();
    std::ofstream(argv[7], std::ios::binary).write(reinterpret_cast<const char*>(eye.output().data()), eye.size() * 4);
    std::ofstream points(argv[8], std::ios::binary);
    for (size_t k = 0; k < eye.samples(); ++k) {
        const retina::Point p = eye.samplePoint(k);
        points.write(reinterpret_cast<const char*>(&p.x), 8);
        points.write(reinterpret_cast<const char*>(&p.y), 8);
    }
}
