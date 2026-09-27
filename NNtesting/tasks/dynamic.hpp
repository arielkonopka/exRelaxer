#pragma once
// Time-varying input tasks for the dynamic ladder (dyn_ladder): a small
// world that produces one frame per step. What the network must output
// depends on how the frames change, never on one frame alone, so a model
// without state (and without a frame window) is stuck at the ceiling of a
// single frame.
//
//   dir     a dot moves on a 1-D retina of `size` pixels (wrapping) at
//           velocity -2, -1, +1 or +2; the velocity is redrawn with
//           probability `switch_p` per step. Target: 1 if it moves right.
//   change  the frame is one of 4 random binary patterns of `size`
//           pixels; with probability `switch_p` a different one is shown.
//           Target: 1 on the step the pattern changed.
//   vel     a dot at continuous position p in [0, 1] (bouncing off both
//           ends) with velocity v in [-0.08, 0.08], redrawn with
//           probability `switch_p`; the frame is a Gaussian population
//           code over `size` pixels. Target: v / 0.08 (regression).
//   catch   closed loop: a ball falls one row per step down a `size` x
//           `size` field, moving dx in {-2, -1, +1, +2} columns per step
//           and bouncing off the side walls. A one-cell paddle on the
//           bottom row moves left, stays or moves right (one cell). The
//           frame is the field (ball = 1) plus one row for the paddle.
//           Target: the action an oracle that knows dx takes (move
//           towards where the ball will land), as -1, 0 or +1. The agent's
//           own action is executed (DAgger-style imitation), so it learns
//           in the states it visits. A policy that only chases the ball's
//           current column misses whenever the ball bounces or outruns it.
#include <algorithm>
#include <cmath>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace dynamic {

class World
{
public:
    World(const std::string& task, size_t size, double switchP, std::uint32_t patternSeed)
        : task_(task), size_(size), switchP_(switchP)
    {
        if (task != "dir" && task != "change" && task != "vel" && task != "catch")
            throw std::invalid_argument("task must be dir, change, vel or catch");
        if (size < 4)
            throw std::invalid_argument("size must be at least 4");
        if (task == "change") {
            std::mt19937 g(patternSeed);
            patterns_.assign(4, std::vector<float>(size));
            for (auto& p : patterns_)
                for (float& v : p)
                    v = static_cast<float>(std::uniform_int_distribution<int>(0, 1)(g));
        }
    }

    bool binary() const { return task_ == "dir" || task_ == "change"; }
    bool control() const { return task_ == "catch"; }
    size_t frameSize() const { return task_ == "catch" ? size_ * size_ + size_ : size_; }

    // Starts a new stream or game.
    void reset(std::mt19937& g)
    {
        std::uniform_int_distribution<int> cell(0, static_cast<int>(size_) - 1);
        if (task_ == "dir") {
            pos_ = cell(g);
            drawVelocity(g);
        } else if (task_ == "change") {
            pattern_ = std::uniform_int_distribution<int>(0, 3)(g);
            changed_ = false;
        } else if (task_ == "vel") {
            p_ = std::uniform_real_distribution<double>(0.0, 1.0)(g);
            vel_ = std::uniform_real_distribution<double>(-0.08, 0.08)(g);
        } else {
            ballX_ = cell(g);
            ballY_ = 0;
            static const int dxs[] = {-2, -1, 1, 2};
            dx_ = dxs[std::uniform_int_distribution<int>(0, 3)(g)];
            paddle_ = static_cast<int>(size_) / 2;
            over_ = false;
            caught_ = false;
        }
    }

    void frame(std::vector<float>& f) const
    {
        f.assign(frameSize(), 0.0f);
        if (task_ == "dir")
            f[static_cast<size_t>(pos_)] = 1.0f;
        else if (task_ == "change")
            f = patterns_[static_cast<size_t>(pattern_)];
        else if (task_ == "vel") {
            const double sigma = 1.0 / static_cast<double>(size_);
            for (size_t i = 0; i < size_; ++i) {
                const double c = (static_cast<double>(i) + 0.5) / static_cast<double>(size_);
                f[i] = static_cast<float>(std::exp(-0.5 * (c - p_) * (c - p_) / (sigma * sigma)));
            }
        } else {
            f[static_cast<size_t>(ballY_) * size_ + static_cast<size_t>(ballX_)] = 1.0f;
            f[size_ * size_ + static_cast<size_t>(paddle_)] = 1.0f;
        }
    }

    // The target for the current frame.
    float target() const
    {
        if (task_ == "dir")
            return v_ > 0 ? 1.0f : 0.0f;
        if (task_ == "change")
            return changed_ ? 1.0f : 0.0f;
        if (task_ == "vel")
            return static_cast<float>(vel_ / 0.08);
        const int land = landing();
        return static_cast<float>((land > paddle_) - (land < paddle_));
    }

    // The column a policy that only sees the current frame would chase.
    float chase() const { return static_cast<float>((ballX_ > paddle_) - (ballX_ < paddle_)); }

    // Advances one step; `action` (-1, 0, +1) is used by catch only.
    void advance(std::mt19937& g, int action = 0)
    {
        std::uniform_real_distribution<double> u(0.0, 1.0);
        if (task_ == "dir") {
            if (u(g) < switchP_)
                drawVelocity(g);
            pos_ = wrap(pos_ + v_);
        } else if (task_ == "change") {
            changed_ = u(g) < switchP_;
            if (changed_)
                pattern_ = (pattern_ + std::uniform_int_distribution<int>(1, 3)(g)) % 4;
        } else if (task_ == "vel") {
            if (u(g) < switchP_)
                vel_ = std::uniform_real_distribution<double>(-0.08, 0.08)(g);
            p_ += vel_;
            if (p_ < 0.0) {
                p_ = -p_;
                vel_ = -vel_;
            } else if (p_ > 1.0) {
                p_ = 2.0 - p_;
                vel_ = -vel_;
            }
        } else {
            paddle_ = std::clamp(paddle_ + action, 0, static_cast<int>(size_) - 1);
            moveBall(ballX_, dx_);
            ++ballY_;
            if (ballY_ == static_cast<int>(size_) - 1) {
                over_ = true;
                caught_ = ballX_ == paddle_;
            }
        }
    }

    bool over() const { return over_; }
    bool caught() const { return caught_; }
    size_t size() const { return size_; }

private:
    int wrap(int x) const
    {
        const int n = static_cast<int>(size_);
        return ((x % n) + n) % n;
    }
    void drawVelocity(std::mt19937& g)
    {
        static const int vs[] = {-2, -1, 1, 2};
        v_ = vs[std::uniform_int_distribution<int>(0, 3)(g)];
    }
    // One horizontal ball step with reflection off the side walls.
    void moveBall(int& x, int& dx) const
    {
        const int last = static_cast<int>(size_) - 1;
        x += dx;
        if (x < 0) {
            x = -x;
            dx = -dx;
        } else if (x > last) {
            x = 2 * last - x;
            dx = -dx;
        }
    }
    // Where the ball will be when it reaches the paddle row.
    int landing() const
    {
        int x = ballX_, dx = dx_;
        for (int y = ballY_; y < static_cast<int>(size_) - 1; ++y)
            moveBall(x, dx);
        return x;
    }

    std::string task_;
    size_t size_;
    double switchP_;
    std::vector<std::vector<float>> patterns_;
    int pos_ = 0, v_ = 1, pattern_ = 0;
    bool changed_ = false;
    double p_ = 0.5, vel_ = 0.0;  // vel: position and velocity
    int ballX_ = 0, ballY_ = 0, dx_ = 1, paddle_ = 0;
    bool over_ = false, caught_ = false;
};

} // namespace dynamic
