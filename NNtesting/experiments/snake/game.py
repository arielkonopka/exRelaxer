"""Snake without a screen: the Python copy of NNtesting/tasks/snake.hpp.

Same rules, same random numbers (SplitMix64), same state vector, so a game
with a given seed and actions is the same game in C++ and in Python
(EXrelaxer.py/tests/test_snake.py checks reference values from the C++
tests). See snake.hpp for the rules and the state layout:

    [0, 8)    the 8 cells around the head in the snake's frame: +1 wall or body, -1 free
    [8, 10)   the apple's direction: unit vector (forward, right)
    [10, 18)  4 points seen forward, left, right, back: map (x, y) of the first
              wall or body cell, scaled to [-1, 1]
    [18, 22)  their distances as closeness 2 / d - 1
    [22]      1 (bias)
"""
import math

import numpy as np

MASK = (1 << 64) - 1
LEFT, STRAIGHT, RIGHT = 0, 1, 2
MOVED, ATE, DIED, STARVED = 0, 1, 2, 3
DIRECTIONS = ((0, -1), (1, 0), (0, 1), (-1, 0))  # up, right, down, left (y grows downwards)
STATE_SIZE = 23


class Rng:
    """SplitMix64, bit for bit as in snake.hpp."""

    def __init__(self, seed):
        self.state = seed & MASK

    def next(self):
        self.state = (self.state + 0x9E3779B97F4A7C15) & MASK
        z = self.state
        z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & MASK
        z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & MASK
        return z ^ (z >> 31)

    def below(self, n):
        return self.next() % n

    def uniform(self):
        return (self.next() >> 11) * 2.0 ** -53


class Game:
    def __init__(self, width, height, seed, length=3, starve_after=0):
        self.width, self.height = width, height
        self.rng = Rng(seed)
        self.starve_after = starve_after or width * height * 2
        self.heading = self.rng.below(4)
        bx, by = DIRECTIONS[(self.heading + 2) % 4]
        x, y = width // 2, height // 2
        self.body = [(x + i * bx, y + i * by) for i in range(length)]  # head first
        self.over = False
        self.score = self.steps = self.hungry = 0
        self.apple = None
        self._place_apple()

    @property
    def head(self):
        return self.body[0]

    def blocked(self, cell):
        x, y = cell
        return x < 0 or y < 0 or x >= self.width or y >= self.height or cell in self.body

    def relative(self, turn):
        """The direction `turn` quarter turns clockwise from the heading."""
        return DIRECTIONS[(self.heading + turn) % 4]

    def apple_distance(self):
        return abs(self.apple[0] - self.head[0]) + abs(self.apple[1] - self.head[1])

    def step(self, action):
        self.heading = (self.heading + action - 1) % 4
        dx, dy = DIRECTIONS[self.heading]
        nxt = (self.head[0] + dx, self.head[1] + dy)
        self.steps += 1
        self.hungry += 1
        eats = nxt == self.apple
        if not eats:
            self.body.pop()  # the tail moves away first
        if self.blocked(nxt):
            self.over = True
            return DIED
        self.body.insert(0, nxt)
        if eats:
            self.score += 1
            self.hungry = 0
            self.over = not self._place_apple()
            return ATE
        if self.hungry >= self.starve_after:
            self.over = True
            return STARVED
        return MOVED

    def state(self):
        s = np.empty(STATE_SIZE, dtype=np.float32)
        hx, hy = self.head
        fx, fy = self.relative(0)
        rx, ry = self.relative(1)
        k = 0
        for along in (1, 0, -1):
            for side in (-1, 0, 1):
                if along == 0 and side == 0:
                    continue
                s[k] = 1.0 if self.blocked((hx + along * fx + side * rx, hy + along * fy + side * ry)) else -1.0
                k += 1
        dx, dy = self.apple[0] - hx, self.apple[1] - hy
        ahead, right = dx * fx + dy * fy, dx * rx + dy * ry
        norm = math.sqrt(ahead * ahead + right * right)
        s[k], s[k + 1] = (ahead / norm, right / norm) if norm > 0 else (0.0, 0.0)
        k += 2
        distances = []
        for turn in (0, 3, 1, 2):  # forward, left, right, back
            ddx, ddy = self.relative(turn)
            x, y, n = hx, hy, 0
            while True:
                x, y, n = x + ddx, y + ddy, n + 1
                if self.blocked((x, y)):
                    break
            distances.append(n)
            s[k] = 2.0 * (x + 1) / (self.width + 1) - 1.0
            s[k + 1] = 2.0 * (y + 1) / (self.height + 1) - 1.0
            k += 2
        for d in distances:
            s[k] = 2.0 / d - 1.0
            k += 1
        s[k] = 1.0
        return s

    def _place_apple(self):
        occupied = set(self.body)
        free = [(x, y) for y in range(self.height) for x in range(self.width) if (x, y) not in occupied]
        if not free:
            return False
        self.apple = free[self.rng.below(len(free))]
        return True
