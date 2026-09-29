"""Occluded-motion clips: a small object moves left or right, the screen goes
blank for `gap` frames, then the object reappears and stands still.

    visible motion (V frames) -> blank (gap frames, all zeros) -> reappearance (2 frames, static)

The reappearance frames never depend on the direction: the object comes
back with the same size, brightness and row it had, at a column drawn
independently of the direction. Size, brightness, row, speed, visible
duration and background noise are drawn the same way for both classes.
Only the frames before the blank carry the answer.

Two tasks differ in what those frames give away:

side   (the primary task) the object moves towards the centre of the screen
       and vanishes 1-4 pixels short of it, so a right-moving object is
       last seen left of centre and a left-moving one right of it. It
       reappears at the centre. Remembering where the object was is enough.
order  motion wraps around horizontally and starts anywhere, so every
       single frame has the same distribution for both classes; the
       reappearance column is uniform. Only the order of two visible frames
       tells the direction: the network must keep ordered information.
"""
import numpy as np

WIDTH, HEIGHT = 20, 15
PIXELS = WIDTH * HEIGHT
READOUT_FRAMES = 2
LEFT, RIGHT = 0, 1
TASKS = ("side", "order")

# Ranges of the per-clip random draws (all independent of the label).
SIZES = (2, 3)               # square side, pixels
BRIGHTNESS = (0.5, 1.0)
SPEEDS = (1, 2)              # pixels per frame
VISIBLE = (3, 4, 5, 6)       # frames of visible motion
NOISE = (0.0, 0.05)          # per-clip standard deviation of the background noise
SHORT = (1, 2, 3, 4)         # side: pixels between the last visible column and the centre


def draw(frame, x, y, size, brightness, wrap):
    """Draws a size x size square with its left column at x, row y; without
    wrap, the part outside the screen is not drawn."""
    cols = np.arange(size) + int(x)
    cols = cols % WIDTH if wrap else cols[(cols >= 0) & (cols < WIDTH)]
    frame[y:y + size, cols] = np.maximum(frame[y:y + size, cols], brightness)


def noisy(frame, sigma, rng):
    if sigma > 0:
        frame = frame + rng.normal(0.0, sigma, frame.shape)
    return np.clip(frame, 0.0, 1.0)


def make_clip(label, gap, rng, task="side"):
    """Returns (frames [n, HEIGHT, WIDTH] float32, info dict). Nothing is
    drawn for the blank frames, so the same generator state gives the same
    clip at every gap."""
    if task not in TASKS:
        raise ValueError("task must be side or order")
    size = int(rng.choice(SIZES))
    brightness = float(rng.uniform(*BRIGHTNESS))
    y = int(rng.integers(0, HEIGHT - size + 1))
    speed = int(rng.choice(SPEEDS))
    visible = int(rng.choice(VISIBLE))
    sigma = float(rng.uniform(*NOISE))
    step = speed if label == RIGHT else -speed
    if task == "side":
        wrap = False
        xr = (WIDTH - size) // 2
        x_last = xr - step // speed * int(rng.choice(SHORT))
    else:
        wrap = True
        x_last = int(rng.integers(0, WIDTH))
        xr = int(rng.integers(0, WIDTH))
    frames = []
    for k in range(visible):
        f = np.zeros((HEIGHT, WIDTH))
        draw(f, x_last - (visible - 1 - k) * step, y, size, brightness, wrap)
        frames.append(noisy(f, sigma, rng))
    frames += [np.zeros((HEIGHT, WIDTH))] * gap   # blank: input exactly zero
    for _ in range(READOUT_FRAMES):
        f = np.zeros((HEIGHT, WIDTH))
        draw(f, xr, y, size, brightness, wrap)
        frames.append(noisy(f, sigma, rng))
    info = dict(label=label, size=size, brightness=brightness, y=y, speed=speed, visible=visible,
                sigma=sigma, x_last=x_last, x_reappear=xr)
    return np.asarray(frames, dtype=np.float32), info


def make_set(n, gap, seed, task="side"):
    """n clips, exactly half of each class in random order. The same (n,
    seed, task) gives the same clips at every gap."""
    rng = np.random.default_rng(seed)
    labels = np.array([LEFT, RIGHT] * (n // 2) + [LEFT] * (n % 2))
    rng.shuffle(labels)
    clips = [make_clip(int(l), gap, rng, task) for l in labels]
    return [c[0] for c in clips], [c[1] for c in clips], labels


def split_seed(seed, split):
    """Seed of one split (0 train, 1 validation, 2 test) of a trial seed; the
    same at every gap and for every model, so comparisons are paired."""
    return [int(seed), int(split), 7919]
