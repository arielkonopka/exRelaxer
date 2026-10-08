"""A Gardens of Eris network that grows in columns, and the skill rooms it is taught in.

    eye: (2r + 1)^2 cells x 12 channels -+-> retina (E-R + fading habituation, no weights) -+
                                         +-> raw (the same image, passed through) ----------+
                                                                                            |
    column k:   c<k>  Conv2D, `filters` kernels 3 x 3 over retina and raw (shared weights) <-+
                p<k>  max pool 3 x 3, stride 2
                h<k>  `width` E-R neurons, fading habituation, reading p<k>, the body, h<k-1>
                      and (recurrent) itself
    readouts:   14 actions, reading h1 .. h<k>

The eye has the 8 channels of experiment.py (wall, free, enemy, collectible, danger,
door, apple, moving), plus `avatar` (a spare avatar), `in_sight`, `novelty` (the game's
1 / sqrt(1 + times seen), remembered per board cell for the episode) and `visited`
(the player has stood on the cell). The body is energy, ammo, spare avatars and the
number of items in each of the 5 inventory sections.

The retina adapts per cell: a still view fades, a change stands out. The raw copy
keeps the static view, so a still player does not go blind. Ties between readouts
(all silent at the start) are broken at random, seeded by the world, so an
untrained network walks about instead of pressing the first action.

Growing (grow.py): a new column gets its own convolution kernels (new features of the
eye), its own E-R layer reading them, the body and the previous column, and readout
lines that start at zero, so the grown network plays exactly as before. Older
columns are frozen; the readouts keep learning.

Rooms: fixed chunks (Gardens-of-Eris chunk patterns) that teach one skill each,
laid out anew from every world seed: explore, collect, doors, avatar, mines; `maze`
is the game's own random maze.
"""
import math

import numpy as np

import exrelaxer as exr

import experiment as base

try:
    import goe
    from goe.pattern import ChunkPattern
except ImportError:
    goe = ChunkPattern = None

ACTIONS = base.ACTIONS
CHANNELS = base.CHANNELS + ["avatar", "in_sight", "novelty", "visited"]
CELL_FEATURES = ["type", "steppable", "killable", "collectible", "in_sight", "moving", "novelty", "visits"]
PLAYER_FEATURES = ["energy", "max_energy", "ammo", "avatars"]
SECTIONS = ["weapons", "usables", "keys", "mods", "tokens"]
BODY = 3 + len(SECTIONS)
EVENTS = ["score", "collect", "apple", "use", "open", "teleport", "kill", "mine", "hurt", "death", "explore", "avatar"]
SKILLS = ["explore", "collect", "doors", "avatar", "mines", "maze"]

PARAMS = {
    # the eye and the game
    "radius": (6, "vision radius (13 x 13 cells); 6 is what a 2-minute game's sight reaches"),
    "maze_moves": (750, "moves in a maze world (750 = 2 minutes, as G2)"),
    "room_moves": (150, "moves in a skill room"),
    "ticks": (3, "network ticks per game step"),
    "readout": ("last", "action values: last (the step's last tick) or sum"),
    "ties": ("random", "readout ties: random (seeded by the world) or first (the lowest action, as goe_rl)"),
    # the reward: the game's events weighed (goe reward_weights)
    "w_score": (0.1, "per point of the game's score (mostly new cells stood on)"),
    "w_collect": (5.0, "per item collected"),
    "w_apple": (20.0, "per golden apple"),
    "w_use": (2.0, "per use of the usable in hand"),
    "w_open": (10.0, "per door opened"),
    "w_teleport": (5.0, "per teleport"),
    "w_kill": (10.0, "per monster killed"),
    "w_mine": (5.0, "per mine set off by the player's shots"),
    "w_hurt": (-0.2, "per energy point lost"),
    "w_death": (-50.0, "per avatar lost (stepping on a landmine is a death)"),
    "w_explore": (0.1, "per cell seen for the first time"),
    "w_avatar": (10.0, "per spare avatar woken"),
    # the network
    "retina": (True, "the habituating retina in front (false: only the raw image)"),
    "retina_er": (True, "retina neurons are E-R"),
    "filters": (8, "convolution kernels per column"),
    "conv_er": (True, "convolution neurons are E-R (false: relu)"),
    "width": (64, "E-R neurons per column"),
    "recurrent": (True, "each column's E-R layer also reads itself"),
    "recurrent_scale": (0.5, "scale of the recurrent weights (x sqrt(3 / width))"),
    "habituation_decay": (0.9, "fading habituation: factor per habituated tick"),
    "habituation_fade_after": (2, "fading habituation: repeats before fading starts"),
    "habituation_tolerance": (0.05, "fading habituation: relative change still counted as a repeat"),
}


def defaults():
    return {k: v[0] for k, v in PARAMS.items()}


# --- the game -----------------------------------------------------------------------------------
def make_game(p):
    if "avatar" not in getattr(goe, "EVENTS", ()):
        raise RuntimeError("columns.py needs a goe package with the avatar and explore events and the novelty "
                           "cell features (Gardens-of-Eris PR #303 and #305); update it")
    weights = {k: p["w_" + k] for k in EVENTS}
    return goe.Game(vision_radius=p["radius"], cell_features=CELL_FEATURES, player_features=PLAYER_FEATURES,
                    inventory_sections=SECTIONS, inventory_slots=1, reward_weights=weights)


def encode(state):
    """The eye: CHANNELS x rows x columns in [0, 1]."""
    v = state.vision
    t, step, kill, coll, sight, moving, novelty, visits = (v[i] for i in range(len(CELL_FEATURES)))
    seen = sight > 0
    r = t.shape[0] // 2
    centre = np.zeros_like(seen)
    centre[r, r] = True
    ch = np.stack([np.isin(t, base.WALLS), step > 0, (kill > 0) & (t != base.PLAYER) & ~np.isin(t, base.DANGER),
                   coll > 0, np.isin(t, base.DANGER), np.isin(t, base.DOORS), t == base.APPLE, moving > 0,
                   (t == base.PLAYER) & ~centre]) & seen
    extra = np.stack([seen, np.clip(novelty, 0.0, 1.0), visits > 0])
    return np.concatenate([ch.astype(np.float32), extra.astype(np.float32)])


def body(state):
    energy, max_energy, ammo, avatars = state.player
    counts = np.minimum(np.asarray(state.inventory_counts, dtype=np.float32), 3.0) / 3.0
    return np.concatenate([[energy / max(max_energy, 1.0), min(ammo, 10.0) / 10.0, min(avatars, 3.0) / 3.0],
                           counts]).astype(np.float32)


# --- the network --------------------------------------------------------------------------------
class Net:
    """The network with `columns` columns. Weights come from `frozen` (layer name -> list of
    weight vectors, one per neuron or kernel) and `theta` (the last column and the readouts,
    flat, in trainable() order); without them, the library's random weights scaled by fan-in."""

    def __init__(self, p, columns, frozen=None, theta=None, net_seed=0):
        self.p, self.columns = p, columns
        exr.reseed(net_seed)
        side = 2 * p["radius"] + 1
        self.image = exr.Shape(len(CHANNELS), side, side)
        self.net = net = exr.Network()
        hab = exr.Habituation(tolerance=p["habituation_tolerance"], decay=p["habituation_decay"],
                              fade_after=p["habituation_fade_after"])
        self.eyes = []
        if p["retina"]:
            spec = exr.LayerSpec.retina(exr.RetinaSpec(self.image), True, p["retina_er"], frozen=True)
            spec.habituation_rule = hab
            self.eyes.append(net.add_layer("retina", spec))
            net.add_inputs(self.eyes[0], self.image, "eye")
        raw = net.add_layer("raw", exr.LayerSpec.retina(exr.RetinaSpec(self.image), False, False, frozen=True))
        if self.eyes:
            net.connect_inputs("eye", raw)
        else:
            net.add_inputs(raw, self.image, "eye")
        self.eyes.append(raw)
        self.convs, self.pools, self.hidden = [], [], []
        self.sources = {}  # hidden layer -> [(what, size)] in weight order
        for k in range(1, columns + 1):
            cspec = exr.LayerSpec.conv2d(p["filters"], exr.Window2D.square(3, 1, 1), False, p["conv_er"], frozen=True)
            cspec.rectify = not p["conv_er"]
            c = net.add_layer(f"c{k}", cspec)
            for e in self.eyes:
                net.connect(e, c)
            pool = net.add_layer(f"p{k}", exr.LayerSpec.pool2d(exr.Window2D.square(3, 2, 1)))
            net.connect(c, pool)
            hspec = exr.LayerSpec.dense(p["width"], True, True, frozen=True)
            hspec.habituation_rule = hab
            h = net.add_layer(f"h{k}", hspec)
            net.connect(pool, h)
            src = [("pool", net.layer_size(pool))]
            if k == 1:
                net.add_inputs(h, BODY, "body")
            else:
                net.connect_inputs("body", h)
            src.append(("body", BODY))
            if self.hidden:
                net.connect(self.hidden[-1], h)
                src.append(("previous", p["width"]))
            if p["recurrent"]:
                net.connect(h, h)
                src.append(("self", p["width"]))
            self.sources[h] = src
            self.convs.append(c)
            self.pools.append(pool)
            self.hidden.append(h)
        self.readouts = []
        for name in ACTIONS:
            out = net.add_layer(name.lower(), exr.LayerSpec.dense(1, False, False, frozen=True))
            for h in self.hidden:
                net.connect(h, out)
            net.add_output(out)
            self.readouts.append(out)
        self._init_weights()
        if frozen:
            self.load(frozen)
        if theta is not None:
            self.set_trainable(theta)
        self.metered = {"retina": self.eyes[:-1], "conv": self.convs, "hidden": self.hidden}
        self.spikes = {k: 0 for k in self.metered}
        self.ticks_seen = 0
        self.rng = np.random.default_rng(0)

    # weights: kernels for convolutions, weight vectors for dense layers
    def _vectors(self, layer):
        if layer in self.convs:
            return [np.asarray(self.net.kernel(layer, c), dtype=np.float32) for c in range(self.p["filters"])]
        return [np.asarray(self.net.weights(layer, i), dtype=np.float32) for i in range(self.net.layer_size(layer))]

    def _set_vectors(self, layer, vectors):
        for i, w in enumerate(vectors):
            w = np.asarray(w, dtype=np.float32)
            if layer in self.convs:
                self.net.set_kernel(layer, i, w)
            else:
                self.net.set_weights(layer, i, w)

    def _init_weights(self):
        """Variance 1 / fan-in per source; recurrent_scale / sqrt(width) on a layer's own weights;
        readout lines from a column after the first start at zero (it plays as before)."""
        for c in self.convs:
            self._set_vectors(c, [w * math.sqrt(3.0 / len(w)) for w in self._vectors(c)])
        for h in self.hidden:
            out = []
            for w in self._vectors(h):
                at, fan = 0, sum(n for what, n in self.sources[h] if what != "self")
                for what, n in self.sources[h]:
                    scale = self.p["recurrent_scale"] * math.sqrt(3.0 / n) if what == "self" else math.sqrt(3.0 / fan)
                    w[at:at + n] *= scale
                    at += n
                out.append(w)
            self._set_vectors(h, out)
        width = self.p["width"]
        for r in self.readouts:
            w = self._vectors(r)[0] * math.sqrt(3.0 / width)
            w[width:] = 0.0
            self._set_vectors(r, [w])

    def trainable(self):
        """What evolves now: the readouts, the newest column's kernels and its E-R layer."""
        return [("readouts", self.readouts), (f"c{self.columns}", [self.convs[-1]]),
                (f"h{self.columns}", [self.hidden[-1]])]

    def get_trainable(self):
        """(flat weights, RMS of each weight's group over its non-zero weights)."""
        theta, rms = [], []
        for _, layers in self.trainable():
            w = np.concatenate([v for layer in layers for v in self._vectors(layer)]).astype(np.float64)
            nz = w[w != 0]
            theta.append(w)
            rms.append(np.full(len(w), math.sqrt(np.mean(nz ** 2)) if len(nz) else 1.0))
        return np.concatenate(theta), np.concatenate(rms)

    def set_trainable(self, theta):
        at = 0
        for _, layers in self.trainable():
            for layer in layers:
                vectors = []
                for v in self._vectors(layer):
                    vectors.append(theta[at:at + len(v)])
                    at += len(v)
                self._set_vectors(layer, vectors)
        if at != len(theta):
            raise ValueError(f"theta has {len(theta)} weights, the network trains {at}")

    def frozen(self):
        """Every column's weights by layer name, for freezing them when the network grows."""
        return {f"c{k + 1}": self._vectors(c) for k, c in enumerate(self.convs)} | \
               {f"h{k + 1}": self._vectors(h) for k, h in enumerate(self.hidden)}

    def load(self, frozen):
        for k in range(self.columns):
            for name, layer in ((f"c{k + 1}", self.convs[k]), (f"h{k + 1}", self.hidden[k])):
                if name in frozen:
                    self._set_vectors(layer, frozen[name])

    def neurons(self):
        return {k: sum(self.net.layer_size(l) for l in ls) for k, ls in self.metered.items()}

    # playing
    def reset(self, seed):
        """A fresh state for a world: a new network is built per candidate, so only the tie rng."""
        self.rng = np.random.default_rng(seed)

    def act(self, state, meter=False):
        self.net.set_inputs("eye", encode(state).ravel())
        self.net.set_inputs("body", body(state))
        total = np.zeros(len(ACTIONS))
        for _ in range(self.p["ticks"]):
            self.net.step()
            total += np.asarray(self.net.outputs())
            if meter:
                for k, layers in self.metered.items():
                    for layer in layers:
                        self.spikes[k] += np.count_nonzero(np.abs(self.net.layer_output(layer)) > 1e-6)
                self.ticks_seen += 1
        y = total if self.p["readout"] == "sum" else np.asarray(self.net.outputs())
        y = np.where(np.isnan(y), -np.inf, y)  # a network whose weights diverged (online learning)
        if self.p["ties"] == "random":
            best = np.flatnonzero(y >= y.max() - 1e-9)
            return int(best[0] if len(best) == 1 else self.rng.choice(best))
        return int(np.argmax(y))


class RandomPlayer:
    def __init__(self):
        self.rng = np.random.default_rng(0)

    def reset(self, seed):
        self.rng = np.random.default_rng(seed + 1)

    def act(self, state, meter=False):
        return int(self.rng.integers(len(ACTIONS)))


# --- rooms --------------------------------------------------------------------------------------
LEGEND = {"#": "wall", ".": None, "@": "player", "P": "player", "g": "gun", "b": "bazooka", "A": "golden_apple",
          "m": "landmine", "1": ("key", 1), "2": ("key", 2), "3": ("key", 3), "D": ("door", 1), "E": ("door", 2),
          "F": ("door", 3)}
CORNER = 24  # the room's top left cell in the start chunk


def _room(n, rng, walls=0.0):
    """An n x n room of floor inside a chunk of wall, with the player on a random cell and
    `walls` of the other cells walls; returns (grid, player cell)."""
    g = np.full((goe.pattern.CHUNK_SIZE, goe.pattern.CHUNK_SIZE), "#", dtype="<U1")
    g[CORNER:CORNER + n, CORNER:CORNER + n] = "."
    inner = g[CORNER:CORNER + n, CORNER:CORNER + n]
    if walls > 0:
        inner[rng.random((n, n)) < walls] = "#"
    y, x = rng.integers(0, n, 2)
    inner[max(y - 1, 0):y + 2, max(x - 1, 0):x + 2] = "."  # room to move from the start
    inner[y, x] = "@"
    return g, inner, (y, x)


def _place(inner, rng, what, count, avoid=None):
    free = np.argwhere(inner == ".")
    if avoid is not None:
        free = free[[avoid(y, x) for y, x in free]] if len(free) else free
    for y, x in free[rng.permutation(len(free))[:count]]:
        inner[y, x] = what


def room(kind, seed):
    """The start chunk's pattern for a skill room, laid out from the seed."""
    rng = np.random.default_rng(seed)
    if kind == "explore":   # a large room with scattered walls: new ground in every direction
        g, inner, _ = _room(25, rng, walls=0.25)
    elif kind == "collect":  # items on the floor
        g, inner, _ = _room(13, rng, walls=0.08)
        _place(inner, rng, "g", 2)
        _place(inner, rng, "1", 2)
    elif kind == "doors":   # a wall across the room with a locked door; its key on the player's side
        g, inner, (py, px) = _room(13, rng)
        x0 = int(rng.integers(3, 10))
        if abs(px - x0) < 2:
            inner[py, px] = "."
            px = 0 if x0 > 6 else 12
            inner[py, px] = "@"
        inner[:, x0] = "#"
        k = int(rng.integers(1, 4))
        inner[int(rng.integers(0, 13)), x0] = "DEF"[k - 1]
        mine_side = (lambda y, x: x < x0) if px < x0 else (lambda y, x: x > x0)
        other_side = (lambda y, x: x > x0) if px < x0 else (lambda y, x: x < x0)
        _place(inner, rng, str(k), 1, mine_side)
        _place(inner, rng, "g", 2, other_side)
    elif kind == "avatar":  # spare avatars to wake
        g, inner, _ = _room(13, rng, walls=0.08)
        _place(inner, rng, "P", 2)
    elif kind == "mines":   # items beyond a field of landmines with one or two safe gaps
        g, inner, (py, px) = _room(13, rng)
        y0 = int(rng.integers(4, 9))
        if abs(py - y0) < 2:
            inner[py, px] = "."
            py = 0 if y0 > 6 else 12
            inner[py, px] = "@"
        inner[y0, :] = "m"
        inner[y0, rng.choice(13, int(rng.integers(1, 3)), replace=False)] = "."
        other_side = (lambda y, x: y > y0) if py < y0 else (lambda y, x: y < y0)
        _place(inner, rng, "g", 2, other_side)
        _place(inner, rng, "m", 4, other_side)
    else:
        raise ValueError(f"unknown room {kind!r}; rooms: {SKILLS[:-1]}")
    return ChunkPattern.from_rows(["".join(r) for r in g], LEGEND)


WALL = None  # the pattern of every chunk around a room: solid wall


def prepare(game, kind, seed):
    """Sets the game's patterns for a world (a room, or the random maze) and starts it."""
    global WALL
    if kind == "maze":
        game.clear_chunk_patterns()
    else:
        if WALL is None:
            WALL = ChunkPattern.from_rows(["#"], {"#": "wall"})
        game.set_default_pattern(WALL)
        game.set_chunk_pattern((0, 0), room(kind, seed))
    game.new_episode(seed=int(seed))


def play(game, player, p, worlds, meter=False):
    """One episode per (kind, seed); returns per-world rewards and summed events."""
    index = [goe.ACTIONS.index(a) for a in ACTIONS]
    rewards, events = [], {k: 0.0 for k in EVENTS}
    for kind, seed in worlds:
        prepare(game, kind, seed)
        player.reset(seed)
        limit = p["maze_moves"] if kind == "maze" else p["room_moves"]
        reward, steps = 0.0, 0
        while not game.is_episode_finished() and steps < limit:
            reward += game.make_action(index[player.act(game.get_state(), meter)])
            steps += 1
        for k, v in game.episode_events.items():
            events[k] = events.get(k, 0.0) + v
        rewards.append(reward)
    return np.array(rewards), events
