// Sound and sight together: objects that look and sound different.
//
// Each of `objects` objects has a look, a bar at its own orientation at a
// random place, and a sound, a tone at its own pitch. Each sample shows one
// object for `ticks` ticks: its image, held still, and its sound, heard
// through a cochlea. Both are noisy, so neither sense alone is always right.
//
//   camera (named source) -> retina -> Gabor bank (frozen) -> Pool2D max  = sight
//   microphone (named source) -> cochlea (mel bands)                        = sound
//
// Two questions:
//
// 1. Fusion: one readout per object (one vs rest, error-driven), reading
//    sight only, sound only, or both. Does hearing and seeing beat either?
//    Metrics accuracy_sight, accuracy_sound, accuracy_both.
//
// 2. Learning what things sound like by watching them: readouts on sight
//    learn the objects from labels, with silence. Then, with no labels, the
//    network sees and hears objects together, and readouts on sound learn
//    to agree with what sight says (sight's choice is their target,
//    error-driven). The test is by sound alone, in the dark:
//    accuracy_transfer is how often the network names an object from its
//    sound, although it was only ever told what objects look like.
//    accuracy_transfer_control skips the watching (chance is 1 / objects);
//    accuracy_teacher is how often sight was right while it taught.
#include <cmath>
#include <numbers>
#include <optional>
#include <random>
#include <string>
#include <vector>
#include "experiment.hpp"
#include "filters.hpp"
#include "network.hpp"
#include "layers/conv2d.hpp"

namespace {

using namespace exr;

struct Objects
{
    size_t count = 4;
    size_t side = 16;          // image side
    size_t length = 10;        // bar length in pixels
    double visionNoise = 0.5;  // uniform +- noise on every pixel
    CochleaSpec ear;
    size_t ticks = 4;          // ticks per sample: window / hop, so the cochlea hears only this sound
    double lowest = 300.0;     // Hz, object 0's pitch
    double step = 2.0;         // pitch ratio between neighbouring objects
    double detune = 0.15;      // pitch jitter, a fraction of the ratio
    double soundNoise = 0.5;   // uniform +- noise on every sample

    // Draws object `k`: `image` (side x side) and `sound` (ticks x hop samples).
    void draw(std::mt19937& g, size_t k, std::vector<float>& image, std::vector<float>& sound) const
    {
        std::uniform_real_distribution<double> unit(0.0, 1.0);
        image.assign(side * side, 0.0f);
        for (float& v : image)
            v = static_cast<float>(visionNoise * (2.0 * unit(g) - 1.0));
        const double angle = std::numbers::pi * static_cast<double>(k) / static_cast<double>(count);
        const double margin = static_cast<double>(length) / 2.0;
        const double cx = margin + unit(g) * (static_cast<double>(side) - 2.0 * margin);
        const double cy = margin + unit(g) * (static_cast<double>(side) - 2.0 * margin);
        for (double s = -margin; s <= margin; s += 0.5) {
            const auto x = static_cast<long>(std::lround(cx + s * std::cos(angle)));
            const auto y = static_cast<long>(std::lround(cy + s * std::sin(angle)));
            if (x >= 0 && y >= 0 && x < static_cast<long>(side) && y < static_cast<long>(side))
                image[static_cast<size_t>(y) * side + static_cast<size_t>(x)] = 1.0f;
        }
        const double pitch = lowest * std::pow(step, static_cast<double>(k) + detune * (2.0 * unit(g) - 1.0));
        const double amplitude = 0.2 + 0.3 * unit(g), phase = 2.0 * std::numbers::pi * unit(g);
        sound.resize(ticks * ear.hop);
        for (size_t n = 0; n < sound.size(); ++n)
            sound[n] = static_cast<float>(
                amplitude * std::sin(2.0 * std::numbers::pi * pitch * static_cast<double>(n) / ear.sampleRate + phase) +
                soundNoise * (2.0 * unit(g) - 1.0));
    }
};

struct Senses
{
    std::unique_ptr<network> net = std::make_unique<network>();
    network::LayerId sight = 0, sound = 0;
    std::vector<network::LayerId> readouts;
    const Objects& task;

    // uses: "sight", "sound", "both" (one readout per object on those
    // features), or "cross" (readouts on sight, then readouts on sound).
    Senses(const nnt::Params& p, const Objects& objects, const std::string& uses) : task(objects)
    {
        const Shape image{1, task.side, task.side};
        const auto eye = net->addLayer("eye", LayerSpec::Retina({image}, false, false));
        net->addInputs(eye, image, "camera");
        const size_t kernel = 5, orientations = static_cast<size_t>(p.getInt("orientations"));
        const auto v1 = net->addLayer("v1", LayerSpec::Conv2D(2 * orientations, Window2D::square(kernel, 1, 2),
                                                              false, false));
        const size_t pool = static_cast<size_t>(p.getInt("pool"));
        sight = net->addLayer("sight", LayerSpec::Pool2D(Window2D::square(pool, pool)));
        sound = net->addLayer("sound", LayerSpec::Cochlea(task.ear, false, false));
        net->addInputs(sound, task.ear.hop, "microphone");
        net->connect(eye, v1);
        net->connect(v1, sight);
        filters::load(net->layerAs<conv2d>(v1), filters::gaborBank(kernel, orientations, 4.0f, 1.5f));
        net->freeze(v1);

        LayerSpec readout = LayerSpec::Dense(1, false, false);
        // The sign rule only sees the sign of each input, and the cochlea's
        // bands are all positive: the graded rule is the default.
        if (p.getString("rule") == "trace")
            readout.learningRule = LearningRule::traced();
        else if (p.getString("rule") != "sign")
            throw std::invalid_argument("rule must be sign or trace");
        std::vector<std::vector<network::LayerId>> groups;  // the features each group of readouts reads
        if (uses == "sight")
            groups = {{sight}};
        else if (uses == "sound")
            groups = {{sound}};
        else if (uses == "both")
            groups = {{sight, sound}};
        else if (uses == "cross")
            groups = {{sight}, {sound}};  // readouts on sight, then readouts on sound
        else
            throw std::invalid_argument("unknown senses '" + uses + "'");
        for (size_t g = 0; g < groups.size(); ++g)
            for (size_t k = 0; k < task.count; ++k) {
                const auto r = net->addLayer((g ? "hears" : "object") + std::to_string(k), readout);
                for (auto f : groups[g])
                    net->connect(f, r);
                net->addOutput(r);
                readouts.push_back(r);
            }
    }

    // Shows one sample (either sense may be switched off); returns the outputs.
    std::vector<float> perceive(const std::vector<float>& image, const std::vector<float>& sound, bool see, bool hear)
    {
        const std::vector<float> dark(image.size(), 0.0f), silence(task.ear.hop, 0.0f);
        net->setInputs("camera", see ? image : dark);
        for (size_t t = 0; t < task.ticks; ++t) {
            net->setInputs("microphone",
                           hear ? std::span<const float>(sound).subspan(t * task.ear.hop, task.ear.hop) : silence);
            net->step();
        }
        return net->outputs();
    }
};

size_t argmax(const std::vector<float>& y)
{
    return static_cast<size_t>(std::ranges::max_element(y) - y.begin());
}

// Plays `count` samples; readouts [from, from + objects) answer, and learn
// (one vs rest, error-driven) when lr > 0: from the label, or with
// `teacher`, from the choice of readouts [teacher, teacher + objects).
// Returns the accuracy of the answers, or with a teacher, of the teacher's.
double session(Senses& s, std::uint32_t seed, int count, bool see, bool hear, float lr, size_t from = 0,
               std::optional<size_t> teacher = std::nullopt)
{
    std::mt19937 g(seed);
    const size_t K = s.task.count;
    std::uniform_int_distribution<size_t> pick(0, K - 1);
    std::vector<float> image, sound;
    int correct = 0;
    for (int i = 0; i < count; ++i) {
        const size_t k = pick(g);
        s.task.draw(g, k, image, sound);
        const std::vector<float> y = s.perceive(image, sound, see, hear);
        const std::vector<float> answer(y.begin() + static_cast<std::ptrdiff_t>(from),
                                        y.begin() + static_cast<std::ptrdiff_t>(from + K));
        size_t target = k;
        if (teacher)
            target = argmax(std::vector<float>(y.begin() + static_cast<std::ptrdiff_t>(*teacher),
                                               y.begin() + static_cast<std::ptrdiff_t>(*teacher + K)));
        correct += (teacher ? target : argmax(answer)) == k;
        if (lr > 0.0f)
            for (size_t r = 0; r < K; ++r) {
                const float want = r == target ? 1.0f : -1.0f;
                if (answer[r] * want <= 0.0f)
                    s.net->getLayer(s.readouts[from + r]).applyReward(want, lr);
            }
    }
    return static_cast<double>(correct) / count;
}

nnt::Register experiment({
    .name = "audiovisual",
    .description = "objects that look (bar orientation) and sound (pitch) different: fusion of sight and sound, and "
                   "learning what objects sound like by watching them (no labels for sound)",
    .tags = {"audio", "vision", "multimodal", "learning"},
    .params = {
        {"objects", "4", "number of objects"},
        {"vision_noise", "0.5", "uniform +- noise on every pixel"},
        {"sound_noise", "0.5", "uniform +- noise on every sample"},
        {"detune", "0.15", "pitch jitter, a fraction of the pitch step"},
        {"orientations", "4", "Gabor orientations (2 phases each)"},
        {"pool", "4", "max pooling window and stride"},
        {"bands", "16", "cochlea mel bands"},
        {"train", "1000", "labelled training samples"},
        {"pairs", "1000", "unlabelled samples seen and heard, for learning sounds from sight"},
        {"test", "400", "test samples"},
        {"rule", "trace", "readout learning rule: trace (graded) or sign"},
        {"lr", "0.1", "readout learning rate (0.01 before the normalised default)"},
    },
    .trials = 5,
    .expect = {{.metric = "accuracy_both", .min = 0.9},
               {.metric = "accuracy_transfer", .min = 0.5},
               {.metric = "accuracy_transfer_control", .max = 0.4}},
    .run = [](nnt::Trial& t) {
        const nnt::Params& p = t.params();
        Objects task;
        task.count = static_cast<size_t>(p.getInt("objects"));
        task.visionNoise = p.getDouble("vision_noise");
        task.soundNoise = p.getDouble("sound_noise");
        task.detune = p.getDouble("detune");
        task.ear.sampleRate = 8000.0f;
        task.ear.hop = 64;
        task.ear.window = 256;
        task.ear.bands = static_cast<size_t>(p.getInt("bands"));
        task.ear.minFrequency = 100.0f;
        task.ticks = task.ear.window / task.ear.hop;
        task.lowest = 250.0;
        task.step = std::pow(3500.0 / 250.0, 1.0 / static_cast<double>(std::max<size_t>(task.count, 2) - 1));
        const int train = static_cast<int>(p.getInt("train")), pairs = static_cast<int>(p.getInt("pairs")),
                  test = static_cast<int>(p.getInt("test"));
        const float lr = static_cast<float>(p.getDouble("lr"));
        const std::uint32_t train_seed = 2000 + t.seed(), pair_seed = 3000 + t.seed(), test_seed = 1000 + t.seed();

        for (const std::string uses : {"sight", "sound", "both"}) {
            exr::reseed(t.seed());
            Senses s(p, task, uses);
            const bool see = uses != "sound", hear = uses != "sight";
            session(s, train_seed, train, see, hear, lr);
            t.record("accuracy_" + uses, session(s, test_seed, test, see, hear, 0.0f));
        }

        for (const bool watched : {true, false}) {
            exr::reseed(t.seed());
            Senses s(p, task, "cross");
            const size_t K = task.count;
            session(s, train_seed, train, true, false, lr);  // told what objects look like, in silence
            if (watched)
                t.record("accuracy_teacher", session(s, pair_seed, pairs, true, true, lr, K, 0));
            t.record(watched ? "accuracy_transfer" : "accuracy_transfer_control",
                     session(s, test_seed, test, false, true, 0.0f, K));
        }
    },
});

} // namespace
