// Rising vs falling chirps, heard through a cochlea, with frozen Gabor
// features on the spectrogram and a learned readout:
//
//   cochlea (hop samples per tick) -> history (bands x ticks spectrogram)
//     -> Conv2D Gabor bank (frozen) -> Pool2D max -> dense mix (frozen, random) -> dense out (learns)
//
// Each sound is a gap of silence (so the cochlea's window forgets the last
// sound), then a chirp gliding up or down that fills the history. On the
// spectrogram a chirp is a diagonal line, so its direction is an
// orientation, the same problem as bar_orientation. The answer is the
// readout's output on the chirp's last tick.
#include <cmath>
#include <numbers>
#include <random>
#include <vector>
#include "experiment.hpp"
#include "filters.hpp"
#include "network.hpp"
#include "layers/conv2d.hpp"

namespace {

using namespace exr;

struct Chirps
{
    CochleaSpec ear;
    size_t ticks = 24;       // chirp length in ticks (the history's length)
    size_t gap = 4;          // ticks of silence before each chirp
    double lowMin = 200, lowMax = 500, highMin = 1500, highMax = 3000;  // Hz
    double noise = 0.02;

    // Fills `sound` (gap + ticks hops) with silence then a chirp; returns +1
    // for rising, -1 for falling.
    float draw(std::mt19937& g, std::vector<float>& sound) const
    {
        std::uniform_real_distribution<double> unit(0.0, 1.0);
        const bool rising = unit(g) < 0.5;
        const double low = lowMin + (lowMax - lowMin) * unit(g), high = highMin + (highMax - highMin) * unit(g);
        const double from = rising ? low : high, to = rising ? high : low;
        const double amplitude = 0.2 + 0.4 * unit(g), phase = 2.0 * std::numbers::pi * unit(g);
        const double rate = ear.sampleRate, seconds = static_cast<double>(ticks * ear.hop) / rate;
        const double k = std::log(to / from) / seconds;  // exponential glide: straight on a mel-like axis
        sound.assign((gap + ticks) * ear.hop, 0.0f);
        for (size_t n = 0; n < ticks * ear.hop; ++n) {
            const double t = static_cast<double>(n) / rate;
            const double angle = 2.0 * std::numbers::pi * from * (std::exp(k * t) - 1.0) / k + phase;
            sound[gap * ear.hop + n] =
                static_cast<float>(amplitude * std::sin(angle) + noise * (2.0 * unit(g) - 1.0));
        }
        return rising ? 1.0f : -1.0f;
    }
};

struct Net
{
    std::unique_ptr<network> net = std::make_unique<network>();
    size_t hop;

    Net(const nnt::Params& p, const Chirps& task) : hop(task.ear.hop)
    {
        const size_t kernel = static_cast<size_t>(p.getInt("kernel"));
        const size_t orientations = static_cast<size_t>(p.getInt("orientations"));
        const size_t pool = static_cast<size_t>(p.getInt("pool"));
        const auto ear = net->addLayer("ear", LayerSpec::Cochlea(task.ear, false, false));
        const auto past = net->addLayer("spectrogram", LayerSpec::History(task.ticks));
        const auto v1 = net->addLayer("v1", LayerSpec::Conv2D(2 * orientations, Window2D::square(kernel, 1, kernel / 2),
                                                              false, false));
        const auto pooled = net->addLayer("pool", LayerSpec::Pool2D(Window2D::square(pool, pool)));
        const auto mix = net->addLayer("mix", LayerSpec::Dense(static_cast<size_t>(p.getInt("mix")), false, false));
        const auto out = net->addLayer("out", LayerSpec::Dense(1, false, false));
        net->addInputs(ear, task.ear.hop);
        net->connect(ear, past);
        net->connect(past, v1);
        net->connect(v1, pooled);
        net->connect(pooled, mix);
        net->connect(mix, out);
        net->addOutput(out);
        filters::load(net->layerAs<conv2d>(v1),
                      filters::gaborBank(kernel, orientations, static_cast<float>(p.getDouble("wavelength")),
                                         static_cast<float>(p.getDouble("sigma"))));
        net->freeze(v1);
        net->freeze(mix);
    }

    // Plays the sound tick by tick; the answer is the last tick's output.
    float classify(const std::vector<float>& sound)
    {
        for (size_t at = 0; at < sound.size(); at += hop) {
            net->setInputs(std::span<const float>(sound).subspan(at, hop));
            net->step();
        }
        return net->outputs()[0];
    }
};

double accuracy(Net& n, const Chirps& task, std::uint32_t seed, int count)
{
    std::mt19937 g(seed);
    std::vector<float> sound;
    int correct = 0;
    for (int i = 0; i < count; ++i) {
        const float label = task.draw(g, sound);
        correct += n.classify(sound) * label > 0.0f;
    }
    return static_cast<double>(correct) / count;
}

void train(Net& n, const Chirps& task, std::uint32_t seed, int count, float learningRate)
{
    std::mt19937 g(seed);
    std::vector<float> sound;
    for (int i = 0; i < count; ++i) {
        const float label = task.draw(g, sound);
        if (n.classify(sound) * label <= 0.0f)  // error-driven: reward only wrong answers
            n.net->applyReward(label, learningRate);
    }
}

nnt::Register experiment({
    .name = "chirp_direction",
    .description = "cochlea + history spectrogram, frozen Gabor bank + frozen mix + learned readout: rising vs falling chirps",
    .tags = {"audio", "learning", "quick"},
    .params = {
        {"sample_rate", "8000", "samples per second"},
        {"hop", "80", "samples per tick"},
        {"window", "256", "cochlea analysis window (power of two)"},
        {"bands", "24", "cochlea mel bands"},
        {"ticks", "24", "chirp length in ticks (history length)"},
        {"noise", "0.02", "uniform noise amplitude added to the chirp"},
        {"kernel", "5", "Gabor kernel size"},
        {"orientations", "4", "Gabor orientations (2 phases each)"},
        {"wavelength", "4", "Gabor wavelength in spectrogram cells"},
        {"sigma", "1.5", "Gabor envelope sigma in cells"},
        {"pool", "6", "max-pooling window and stride"},
        {"mix", "64", "frozen random mixing neurons"},
        {"train", "400", "training chirps"},
        {"test", "100", "test chirps"},
        {"lr", "0.01", "learning rate"},
    },
    .trials = 5,
    .expect = {{.metric = "accuracy", .min = 0.9}, {.metric = "accuracy_control", .max = 0.7}},
    .run = [](nnt::Trial& t) {
        const nnt::Params& p = t.params();
        Chirps task;
        task.ear.sampleRate = static_cast<float>(p.getDouble("sample_rate"));
        task.ear.hop = static_cast<size_t>(p.getInt("hop"));
        task.ear.window = static_cast<size_t>(p.getInt("window"));
        task.ear.bands = static_cast<size_t>(p.getInt("bands"));
        task.ear.minFrequency = 100.0f;
        task.ticks = static_cast<size_t>(p.getInt("ticks"));
        task.gap = task.ear.window / task.ear.hop + 1;
        task.noise = p.getDouble("noise");
        const int train_count = static_cast<int>(p.getInt("train")), test_count = static_cast<int>(p.getInt("test"));
        const float lr = static_cast<float>(p.getDouble("lr"));
        const std::uint32_t test_seed = 1000 + t.seed(), train_seed = 2000 + t.seed();

        Net learner(p, task);
        t.record("accuracy_before", accuracy(learner, task, test_seed, test_count));
        train(learner, task, train_seed, train_count, lr);
        const double after = accuracy(learner, task, test_seed, test_count);
        t.record("accuracy", after);

        exr::reseed(t.seed());  // the same network, trained without learning
        Net control(p, task);
        train(control, task, train_seed, train_count, 0.0f);
        const double control_accuracy = accuracy(control, task, test_seed, test_count);
        t.record("accuracy_control", control_accuracy);
        t.record("gain", after - control_accuracy);
    },
});

} // namespace
