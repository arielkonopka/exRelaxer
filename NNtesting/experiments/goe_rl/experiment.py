"""Gardens of Eris played by an E-R network. See README.md here.

    vision grid (2r + 1)^2 cells x 8 channels ("eye") ---\\
    player energy, ammo, avatars ("body") ----------------+-> hidden (er | relu | gate), habituation
                                                                     |
                                    14 action readouts: move, shoot, interact (x4 directions), next gun, use

The game is the `goe` package of https://github.com/arielkonopka/Gardens-of-Eris
(agent/python): headless, one game per process, the same seed builds the same
world. One step is one move (8 game ticks); the network runs `ticks` ticks
on each step's view and the largest readout picks the action.

The reward is the game's own score (+1 per new cell visited, +1 per item
collected, + the energy of what the player kills) with a penalty `w_death`
per avatar lost. No term counts neural activity.

The nntest experiment here plays the untrained network against a random
player on the same worlds; es.py evolves the network on the reward.
"""
import math

import numpy as np

import exrelaxer as exr
from exrelaxer import harness as nnt

try:
    import goe
except ImportError:  # the experiment reports it in run()
    goe = None

ACTIONS = ["MOVE_UP", "MOVE_DOWN", "MOVE_LEFT", "MOVE_RIGHT",
           "SHOOT_UP", "SHOOT_DOWN", "SHOOT_LEFT", "SHOOT_RIGHT",
           "INTERACT_UP", "INTERACT_DOWN", "INTERACT_LEFT", "INTERACT_RIGHT",
           "NEXT_GUN", "USE"]
CELL_FEATURES = ["type", "steppable", "killable", "collectible", "in_sight", "moving"]
PLAYER_FEATURES = ["energy", "max_energy", "ammo", "avatars"]
# element types (the game's bElemTypes)
WALLS = (4, 8)                    # wall, brick cluster
DANGER = (201, 205, 602, 603)     # missiles, bomb, landmine
DOORS = (51, 52)                  # key, door
APPLE = 900
PLAYER = 100
CHANNELS = ["wall", "free", "enemy", "collectible", "danger", "door", "apple", "moving"]


def make_game(p):
    return goe.Game(vision_radius=p["radius"], cell_features=CELL_FEATURES, player_features=PLAYER_FEATURES,
                    inventory_sections=["weapons"], inventory_slots=1, episode_ticks=p["episode_ticks"])


def encode(state):
    """The vision grid as CHANNELS x cells in [0, 1]: what is in each cell, as the
    E-R layer should tell apart (cells out of sight are all zero)."""
    v = state.vision
    t, step, kill, coll, sight, moving = (v[i] for i in range(len(CELL_FEATURES)))
    seen = sight > 0
    ch = np.stack([np.isin(t, WALLS), step > 0, (kill > 0) & (t != PLAYER) & ~np.isin(t, DANGER), coll > 0,
                   np.isin(t, DANGER), np.isin(t, DOORS), t == APPLE, moving > 0]) & seen
    return ch.astype(np.float32).ravel()


def body(state):
    energy, max_energy, ammo, avatars = state.player
    return np.array([energy / max(max_energy, 1.0), min(ammo, 10.0) / 10.0, min(avatars, 3.0) / 3.0],
                    dtype=np.float32)


class Player:
    def __init__(self, p):
        model = p["model"]
        if model not in ("relu", "er", "gate"):
            raise ValueError("model must be relu, er or gate")
        side = 2 * p["radius"] + 1
        self.ticks = p["ticks"]
        self.readout = p["readout"]
        self.net = net = exr.Network()

        def neurons(size):
            spec = exr.LayerSpec.dense(size, p["habituation"], model == "er", frozen=True)
            if p["habituation"]:
                spec.habituation_rule = exr.Habituation(tolerance=p["habituation_tolerance"],
                                                        decay=p["habituation_decay"],
                                                        fade_after=p["habituation_fade_after"])
            spec.rectify = model == "relu"
            if model == "gate":
                spec.gate = p["gate"]
            return spec

        self.layers = []
        for l in range(p["depth"]):
            h = net.add_layer(f"h{l + 1}", neurons(p["width"]))
            if l == 0:
                net.add_inputs(h, len(CHANNELS) * side * side, "eye")
                net.add_inputs(h, 3, "body")
            else:
                net.connect(self.layers[-1], h)
            self.layers.append(h)
        recurrent = p["recurrent"]
        if recurrent:
            for h in self.layers:
                net.connect(h, h)
        self.readouts = []
        for name in ACTIONS:
            out = net.add_layer(name.lower(), exr.LayerSpec.dense(1, False, False))
            net.connect(self.layers[-1], out)
            net.add_output(out)
            self.readouts.append(out)
        # Uniform weights with variance 1 / fan-in; a layer reading itself
        # gets recurrent_scale / sqrt(n) on those n weights (below 1 the echo fades).
        for layer in self.layers + self.readouts:
            size = net.layer_size(layer)
            own = size if recurrent and layer in self.layers else 0
            for i in range(size):
                w = np.asarray(net.weights(layer, i), dtype=np.float32)
                if own:
                    w[:-own] *= math.sqrt(3.0 / (len(w) - own))
                    w[-own:] *= p["recurrent_scale"] * math.sqrt(3.0 / own)
                else:
                    w = w * math.sqrt(3.0 / len(w))
                net.set_weights(layer, i, w)
        self.neurons = sum(net.layer_size(l) for l in self.layers)
        self.spikes = self.ticks_seen = 0

    def act(self, state, meter=False):
        self.net.set_inputs("eye", encode(state))
        self.net.set_inputs("body", body(state))
        total = np.zeros(len(ACTIONS))
        for _ in range(self.ticks):
            self.net.step()
            total += np.asarray(self.net.outputs())
            if meter:
                for layer in self.layers:
                    self.spikes += np.count_nonzero(np.abs(self.net.layer_output(layer)) > 1e-6)
                self.ticks_seen += 1
        y = total if self.readout == "sum" else np.asarray(self.net.outputs())
        return int(np.argmax(y))


class RandomPlayer:
    """The control: a uniformly random action every step."""
    def __init__(self, seed):
        self.rng = np.random.default_rng(seed)
        self.spikes = self.ticks_seen = 0

    def act(self, state, meter=False):
        return int(self.rng.integers(len(ACTIONS)))


def play(game, player, p, seeds, meter=False):
    """Plays one episode per world seed; returns the means of reward, score, avatars lost, death, steps."""
    index = [goe.ACTIONS.index(a) for a in ACTIONS]
    stats = {k: [] for k in ("reward", "score", "avatars_lost", "dead", "steps")}
    for seed in seeds:
        game.new_episode(seed=int(seed))
        score, steps = 0.0, 0
        while not game.is_episode_finished():
            score += game.make_action(index[player.act(game.get_state(), meter)])
            steps += 1
        lost = game.avatars_lost + int(game.is_player_dead())
        stats["reward"].append(score - p["w_death"] * lost)
        stats["score"].append(score)
        stats["avatars_lost"].append(game.avatars_lost)
        stats["dead"].append(float(game.is_player_dead()))
        stats["steps"].append(steps)
    return {k: float(np.mean(v)) for k, v in stats.items()}


PARAMS = {
    "radius": (6, "vision radius: the grid has 2r + 1 cells on each side"),
    "episode_ticks": (6000, "episode length in game ticks (50 a second; 6000 = 2 minutes = 750 moves)"),
    "w_death": (50.0, "penalty per avatar lost (the last one included)"),
    "model": ("er", "hidden neurons: er, relu or gate"),
    "width": (128, "hidden neurons per layer"),
    "depth": (1, "hidden layers"),
    "recurrent": (False, "each hidden layer also reads its own previous output"),
    "recurrent_scale": (0.5, "scale of recurrent weights (x sqrt(3 / layer size))"),
    "gate": (0.2, "gate model: the fixed threshold"),
    "habituation": (True, "hidden neurons habituate to repeated input (fade mode)"),
    "habituation_decay": (0.9, "habituation: fade factor per habituated tick"),
    "habituation_fade_after": (2, "habituation: repeats before fading starts"),
    "habituation_tolerance": (0.05, "habituation: relative change still counted as a repeat"),
    "ticks": (3, "network ticks per game step (the view is held)"),
    "readout": ("last", "action values: last (the step's last tick) or sum (summed over the step's ticks)"),
    "games": (10, "test games (worlds) per trial"),
}


@nnt.experiment(
    name="goe_rl",
    description="Gardens of Eris: vision grid (8 channels) + body -> E-R / relu / gate layer -> 14 action readouts; "
                "untrained network against a random player on the same worlds (es.py trains it)",
    tags=["er", "dynamic", "control"],
    params=PARAMS,
    trials=3,
)
def run(t):
    if goe is None:
        raise RuntimeError("goe is not installed: pip install <Gardens-of-Eris checkout>/agent/python")
    exr.set_threads(1)
    p = t.params
    seeds = [10000 * (t.seed + 1) + g for g in range(p["games"])]
    game = make_game(p)
    exr.reseed(t.seed)
    player = Player(p)
    result = play(game, player, p, seeds, meter=True)
    for k, v in result.items():
        t.record(k, v)
    t.record("spikes_per_step", player.spikes / max(player.ticks_seen, 1) * p["ticks"])
    t.record("active_fraction", player.spikes / max(player.ticks_seen, 1) / player.neurons)
    control = play(game, RandomPlayer(t.seed), p, seeds)
    for k in ("reward", "score", "avatars_lost", "dead"):
        t.record(k + "_random", control[k])
    game.close()
