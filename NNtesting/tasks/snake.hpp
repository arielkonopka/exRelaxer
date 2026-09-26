#pragma once
// Snake without a screen: a snake on a walled grid turns left, goes straight
// or turns right each step, grows by eating apples, and dies when it hits a
// wall or itself. A small control task with delayed consequences.
//
// The game and its state vector are mirrored exactly by
// NNtesting/experiments/snake/game.py (same random numbers, same apples,
// same states), so the C++ and Python experiments play the same games.
//
// The state the network sees (State::size values, all in [-1, 1]), in the
// snake's own frame (forward = the way the head points, so the three actions
// mean the same thing whichever way it moves):
//   [0, 8)    the close neighbourhood: the 8 cells around the head, row by
//             row from the front-left (front-left, front, front-right, left,
//             right, back-left, back, back-right): +1 wall or body, -1 free
//   [8, 10)   the direction of the apple: a unit vector (forward, right)
//   [10, 18)  4 points seen by the snake: looking forward, left, right and
//             back, the first wall or body cell in that direction, as its map
//             position (x, y), scaled to [-1, 1] over the walled map
//   [18, 22)  the distances to those 4 points as closeness: 2 / d - 1 (1 when
//             adjacent, towards -1 far away)
//   [22]      1 (a bias input: the neurons have no bias of their own)
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <vector>

namespace snake {

// SplitMix64: tiny and easy to reproduce exactly in Python.
struct Rng
{
    std::uint64_t state;
    explicit Rng(std::uint64_t seed) : state(seed) {}
    std::uint64_t next()
    {
        std::uint64_t z = (state += 0x9E3779B97F4A7C15ull);
        z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
        z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
        return z ^ (z >> 31);
    }
    size_t below(size_t n) { return static_cast<size_t>(next() % n); }
    double uniform() { return static_cast<double>(next() >> 11) * 0x1.0p-53; }
};

enum class Action : int { Left = 0, Straight = 1, Right = 2 };
enum class Outcome : int { Moved = 0, Ate = 1, Died = 2, Starved = 3 };

struct Cell
{
    int x, y;
    bool operator==(const Cell&) const = default;
};

// Directions clockwise from up: up, right, down, left (y grows downwards).
inline constexpr std::array<Cell, 4> directions{{{0, -1}, {1, 0}, {0, 1}, {-1, 0}}};

struct State
{
    static constexpr size_t size = 23;
    std::array<float, size> values{};
};

class Game
{
public:
    // A width x height field of free cells inside the walls. The snake starts
    // with `length` cells in the middle, pointing a random way. The game ends
    // when it dies or goes `starveAfter` steps without eating (0: width *
    // height * 2).
    Game(int width, int height, std::uint64_t seed, size_t length = 3, size_t starveAfter = 0)
        : width_(width), height_(height), rng_(seed),
          starve_after_(starveAfter ? starveAfter : static_cast<size_t>(width * height * 2))
    {
        heading_ = static_cast<int>(rng_.below(4));
        const Cell back = directions[static_cast<size_t>((heading_ + 2) % 4)];
        Cell c{width / 2, height / 2};
        for (size_t i = 0; i < length; ++i, c = {c.x + back.x, c.y + back.y})
            body_.push_back(c);
        placeApple();
    }

    int width() const { return width_; }
    int height() const { return height_; }
    const std::deque<Cell>& body() const { return body_; }  // head first
    Cell head() const { return body_.front(); }
    Cell apple() const { return apple_; }
    int heading() const { return heading_; }
    bool over() const { return over_; }
    size_t score() const { return score_; }   // apples eaten
    size_t steps() const { return steps_; }

    // Wall (outside the field) or body.
    bool blocked(Cell c) const
    {
        if (c.x < 0 || c.y < 0 || c.x >= width_ || c.y >= height_)
            return true;
        for (const Cell& b : body_)
            if (b == c)
                return true;
        return false;
    }

    // The direction `turn` quarter turns clockwise from the heading, as a step.
    Cell relative(int turn) const { return directions[static_cast<size_t>(((heading_ + turn) % 4 + 4) % 4)]; }

    Outcome step(Action action)
    {
        heading_ = (heading_ + static_cast<int>(action) - 1 + 4) % 4;
        const Cell d = directions[static_cast<size_t>(heading_)];
        const Cell next{head().x + d.x, head().y + d.y};
        ++steps_;
        ++hungry_;
        const bool eats = next == apple_;
        if (!eats)
            body_.pop_back();  // the tail moves away first, so the head may take its cell
        if (blocked(next)) {
            over_ = true;
            return Outcome::Died;
        }
        body_.push_front(next);
        if (eats) {
            ++score_;
            hungry_ = 0;
            over_ = !placeApple();  // the snake fills the field: nothing left to eat
            return Outcome::Ate;
        }
        if (hungry_ >= starve_after_) {
            over_ = true;
            return Outcome::Starved;
        }
        return Outcome::Moved;
    }

    State state() const
    {
        State s;
        const Cell h = head();
        const Cell f = relative(0), r = relative(1);
        size_t k = 0;
        // Neighbourhood, snake frame: rows front (+1), level (0), back (-1);
        // columns left (-1), centre (0), right (+1).
        for (int along : {1, 0, -1})
            for (int side : {-1, 0, 1}) {
                if (along == 0 && side == 0)
                    continue;
                const Cell c{h.x + along * f.x + side * r.x, h.y + along * f.y + side * r.y};
                s.values[k++] = blocked(c) ? 1.0f : -1.0f;
            }
        // Apple direction as a unit vector (forward, right).
        const double dx = apple_.x - h.x, dy = apple_.y - h.y;
        const double ahead = dx * f.x + dy * f.y, right = dx * r.x + dy * r.y;
        const double norm = std::sqrt(ahead * ahead + right * right);
        s.values[k++] = norm > 0 ? static_cast<float>(ahead / norm) : 0.0f;
        s.values[k++] = norm > 0 ? static_cast<float>(right / norm) : 0.0f;
        // What it sees forward, left, right and back: the first blocked cell.
        std::array<int, 4> distance{};
        const std::array<int, 4> turns{0, 3, 1, 2};
        for (size_t i = 0; i < 4; ++i) {
            const Cell d = relative(turns[i]);
            Cell c = h;
            int n = 0;
            do {
                c = {c.x + d.x, c.y + d.y};
                ++n;
            } while (!blocked(c));
            distance[i] = n;
            // Walls are at -1 and width / height: scaled to [-1, 1].
            s.values[k++] = static_cast<float>(2.0 * (c.x + 1) / (width_ + 1) - 1.0);
            s.values[k++] = static_cast<float>(2.0 * (c.y + 1) / (height_ + 1) - 1.0);
        }
        for (int d : distance)
            s.values[k++] = static_cast<float>(2.0 / d - 1.0);
        s.values[k++] = 1.0f;
        return s;
    }

    // Manhattan distance from the head to the apple.
    int appleDistance() const { return std::abs(apple_.x - head().x) + std::abs(apple_.y - head().y); }

private:
    bool placeApple()
    {
        std::vector<Cell> free;
        for (int y = 0; y < height_; ++y)
            for (int x = 0; x < width_; ++x)
                if (!blocked({x, y}))
                    free.push_back({x, y});
        if (free.empty())
            return false;
        apple_ = free[rng_.below(free.size())];
        return true;
    }

    int width_, height_;
    Rng rng_;
    size_t starve_after_;
    std::deque<Cell> body_;
    Cell apple_{};
    int heading_ = 0;
    bool over_ = false;
    size_t score_ = 0, steps_ = 0, hungry_ = 0;
};

} // namespace snake
