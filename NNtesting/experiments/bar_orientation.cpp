// Vertical vs horizontal bars with frozen Gabor features and a learned
// readout:
//
//   retina -> Conv2D Gabor bank (frozen) -> Pool2D max -> dense mix (frozen, random) -> dense out (learns)
//
// Pooled orientation energies are all >= 0 and the learning rule only uses
// the sign of each input, so the frozen random mixing layer turns them into
// features whose signs differ between the classes.
#include <random>
#include <vector>
#include "bars.hpp"
#include "experiment.hpp"
#include "filters.hpp"
#include "network.hpp"
#include "layers/conv2d.hpp"

namespace {

using namespace exr;

struct Net
{
    std::unique_ptr<network> net = std::make_unique<network>();

    Net(const nnt::Params& p, size_t side)
    {
        const Shape image{1, side, side};
        const size_t kernel = static_cast<size_t>(p.getInt("kernel"));
        const size_t orientations = static_cast<size_t>(p.getInt("orientations"));
        const auto eye = net->addLayer("eye", LayerSpec::Retina({image}, false, false));
        const auto v1 = net->addLayer("v1", LayerSpec::Conv2D(2 * orientations, Window2D::square(kernel, 1, kernel / 2),
                                                              false, false));
        const size_t pool = static_cast<size_t>(p.getInt("pool"));
        const auto pooled = net->addLayer("pool", LayerSpec::Pool2D(Window2D::square(pool, pool)));
        const auto mix = net->addLayer("mix", LayerSpec::Dense(static_cast<size_t>(p.getInt("mix")), false, false));
        const auto out = net->addLayer("out", LayerSpec::Dense(1, false, p.getBool("readout_er")));
        net->addInputs(eye, image);
        net->connect(eye, v1);
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

    float classify(const std::vector<float>& image)
    {
        net->setInputs(image);
        net->step();
        return net->outputs()[0];
    }
};

double accuracy(Net& n, const bars::BarImages& task, std::uint32_t seed, int count)
{
    std::mt19937 g(seed);
    std::vector<float> image;
    int correct = 0;
    for (int i = 0; i < count; ++i) {
        const float label = task.draw(g, image);
        correct += n.classify(image) * label > 0.0f;
    }
    return static_cast<double>(correct) / count;
}

void train(Net& n, const bars::BarImages& task, std::uint32_t seed, int count, float learningRate, bool errorDriven)
{
    std::mt19937 g(seed);
    std::vector<float> image;
    for (int i = 0; i < count; ++i) {
        const float label = task.draw(g, image);
        const bool wrong = n.classify(image) * label <= 0.0f;
        if (wrong || !errorDriven)
            n.net->applyReward(label, learningRate);
    }
}

nnt::Register experiment({
    .name = "bar_orientation",
    .description = "frozen Gabor bank + frozen random mix + learned readout: vertical vs horizontal bars",
    .tags = {"vision", "learning", "quick"},
    .params = {
        {"side", "24", "image side in pixels"},
        {"noise", "0.1", "uniform pixel noise amplitude"},
        {"kernel", "7", "Gabor kernel size"},
        {"orientations", "4", "Gabor orientations (2 phases each)"},
        {"wavelength", "5", "Gabor wavelength in pixels"},
        {"sigma", "2", "Gabor envelope sigma in pixels"},
        {"pool", "6", "max-pooling window and stride"},
        {"mix", "64", "frozen random mixing neurons"},
        {"readout_er", "false", "E-R in the learned readout"},
        {"train", "1500", "training images"},
        {"test", "200", "test images"},
        {"lr", "0.01", "learning rate"},
        {"reward", "error", "error: reward only wrong answers; target: reward every image"},
    },
    .trials = 10,
    .expect = {{.metric = "accuracy", .min = 0.95}, {.metric = "accuracy_control", .max = 0.6}},
    .run = [](nnt::Trial& t) {
        const nnt::Params& p = t.params();
        bars::BarImages task;
        task.side = static_cast<size_t>(p.getInt("side"));
        task.noise = static_cast<float>(p.getDouble("noise"));
        const int train_count = static_cast<int>(p.getInt("train")), test_count = static_cast<int>(p.getInt("test"));
        const float lr = static_cast<float>(p.getDouble("lr"));
        const bool error_driven = p.getString("reward") == "error";
        if (!error_driven && p.getString("reward") != "target")
            throw std::invalid_argument("reward must be error or target");
        const std::uint32_t test_seed = 1000 + t.seed(), train_seed = 2000 + t.seed();

        Net learner(p, task.side);
        t.record("accuracy_before", accuracy(learner, task, test_seed, test_count));
        train(learner, task, train_seed, train_count, lr, error_driven);
        const double after = accuracy(learner, task, test_seed, test_count);
        t.record("accuracy", after);

        exr::reseed(t.seed());  // the same network, trained without learning
        Net control(p, task.side);
        train(control, task, train_seed, train_count, 0.0f, error_driven);
        const double control_accuracy = accuracy(control, task, test_seed, test_count);
        t.record("accuracy_control", control_accuracy);
        t.record("gain", after - control_accuracy);
    },
});

} // namespace
