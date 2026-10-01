"""Gardens of Eris played by an E-R network. See README.md here.

    vision grid (2r + 1)^2 cells x 8 channels ("eye") ---\\
    player energy, ammo, avatars ("body") ----------------+-> hidden (er | relu | gate), habituation
                                                                     |
                                    14 action readouts: move, shoot, interact (x4 directions), next gun, use

The game is the `goe` package of https://github.com/arielkonopka/Gardens-of-Eris
(agent/python): headless, one game per process, the same seed builds the same
world. One step is one move (8 game ticks); the network runs `ticks` ticks
on each step's view and the largest readout picks the action.

The reward (`reward` events, the default) weighs the game's events with the
w_* parameters (goe's reward_weights, Gardens-of-Eris PR #289): items,
golden apples, items used, doors opened, teleports, monsters killed and
mines set off are rewarded; energy lost and avatars lost are penalised; a
little of the game's score (new cells visited) keeps the player exploring.
`reward` score is the reward of G1 (doc/goe.md): the game's own score (+1
per new cell visited, +1 per item collected, + the energy of what the player
kills) and w_death per avatar lost. No term counts neural activity.

The nntest experiment here plays the untrained network against a random
player on the same worlds; es.py evolves the network on the reward.

Beyond the plain stack (h1 .. h<depth>), the network can have:
- a reservoir: `reservoir` E-R neurons reading h1, `reservoir_recurrent`
  more reading the reservoir (an echo state); frozen unless
  reservoir_evolve; the readouts read it;
- skip: the top hidden layer reads every layer below it, not only the one
  under it;
- a feedback ladder: every hidden layer above h1 (or only the top one)
  comes back to h1 as input, one tick late, through a rung layer r<k> of
  `feedback_first` neurons of their own model (relu by default, with
  habituation), which every h1 neuron reads next to the eye and the body;
- readout_from all: the readouts read every hidden layer.
With any of these the network is built by Player._build, one layer at a
time, recording what every weight reads (_shared/wiring.py), so es.py can
grow it (--grow-to) and carry the weights over. models/ holds the designed
configurations (doc/goe.md).
"""
import math
import os
import sys

import numpy as np

import exrelaxer as exr
from exrelaxer import harness as nnt

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "_shared"))
from wiring import Wiring  # noqa: E402

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


MOVE_TICKS = 8  # game ticks per move


def vision_radius(p):
    """The vision grid's radius. The player's sight grows with the steps it has
    made (2 + ln(steps) / 2 cells: 2 at the start, 5.3 after 750 moves, 6.7
    after 11250), so the grid must cover the furthest sight of an episode from
    the start; cells beyond the current sight read zero until it reaches them.
    "auto" sizes it for the episode's moves (episode_ticks / 8): 6 for the
    default 2 minutes, 7 for 30."""
    r = p["radius"]
    if r == "auto":
        return math.ceil(2 + math.log(max(p["episode_ticks"] / MOVE_TICKS, 1)) / 2)
    return int(r)


def eye_channels(p):
    return CHANNELS + (["seen"] if p.get("seen_channel", False) else [])


EVENTS = ["score", "collect", "apple", "use", "open", "teleport", "kill", "mine", "hurt", "death"]


def events_reward(p):
    """Whether the reward weighs the game's events (else: score and w_death, as in G1)."""
    reward = p.get("reward", "events")
    if reward not in ("events", "score"):
        raise ValueError("reward must be events or score")
    return reward == "events"


def make_game(p):
    weights = None
    if events_reward(p):
        if not hasattr(goe, "EVENTS"):
            raise RuntimeError("reward events needs a goe package with reward_weights (Gardens-of-Eris PR #289); "
                               "update it, or set reward=score")
        weights = {k: p["w_" + k] for k in EVENTS}
        weights["death"] = -p["w_death"]
    return goe.Game(vision_radius=vision_radius(p), cell_features=CELL_FEATURES, player_features=PLAYER_FEATURES,
                    inventory_sections=["weapons"], inventory_slots=1, episode_ticks=p["episode_ticks"],
                    **({"reward_weights": weights} if weights else {}))


def encode(state, seen_channel=False):
    """The vision grid as CHANNELS x cells in [0, 1]: what is in each cell, as the
    E-R layer should tell apart (cells out of sight are all zero). With
    seen_channel a last channel marks the cells in sight, so the network can
    tell an empty cell from an unseen one, and how far it sees."""
    v = state.vision
    t, step, kill, coll, sight, moving = (v[i] for i in range(len(CELL_FEATURES)))
    seen = sight > 0
    ch = np.stack([np.isin(t, WALLS), step > 0, (kill > 0) & (t != PLAYER) & ~np.isin(t, DANGER), coll > 0,
                   np.isin(t, DANGER), np.isin(t, DOORS), t == APPLE, moving > 0]) & seen
    if seen_channel:
        ch = np.concatenate([ch, seen[None]])
    return ch.astype(np.float32).ravel()


def body(state):
    energy, max_energy, ammo, avatars = state.player
    return np.array([energy / max(max_energy, 1.0), min(ammo, 10.0) / 10.0, min(avatars, 3.0) / 3.0],
                    dtype=np.float32)


def designed(p):
    """True when the network uses the reservoir, skip, ladder or readout_from all,
    and is built by Player._build; the plain stack keeps its original construction."""
    return (p.get("reservoir", 0) > 0 or p.get("skip", False) or p.get("feedback_first", 0) > 0
            or p.get("readout_from", "top") == "all")


class Player:
    def __init__(self, p):
        model = p["model"]
        if model not in ("relu", "er", "gate"):
            raise ValueError("model must be relu, er or gate")
        side = 2 * vision_radius(p) + 1
        self.seen_channel = p.get("seen_channel", False)
        self.ticks = p["ticks"]
        self.readout = p["readout"]
        self.net = net = exr.Network()
        self.wiring, self.rungs, self.reservoir = None, [], None

        def neurons(size, model=model, habituation=p["habituation"]):
            if model not in ("relu", "er", "gate"):
                raise ValueError("neuron model must be relu, er or gate")
            spec = exr.LayerSpec.dense(size, habituation, model == "er", frozen=True)
            if habituation:
                spec.habituation_rule = exr.Habituation(tolerance=p["habituation_tolerance"],
                                                        decay=p["habituation_decay"],
                                                        fade_after=p["habituation_fade_after"])
            spec.rectify = model == "relu"
            if model == "gate":
                spec.gate = p["gate"]
            return spec

        if designed(p):
            self._build(p, neurons, len(eye_channels(p)) * side * side)
            return
        self.layers = []
        for l in range(p["depth"]):
            h = net.add_layer(f"h{l + 1}", neurons(p["width"]))
            if l == 0:
                net.add_inputs(h, len(eye_channels(p)) * side * side, "eye")
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
        self.metered = self.layers
        self.neurons = sum(net.layer_size(l) for l in self.layers)
        self.spikes = self.ticks_seen = 0

    def _build(self, p, neurons, eye_size):
        """Built one layer at a time, as es.py --grow-to grows it: h1 (eye, body)
        and its own loop, the reservoir, the readouts; then each further layer
        with its connections, its rung and its readout lines."""
        net = self.net
        w = self.wiring = Wiring(net)
        recurrent, skip = p["recurrent"], p.get("skip", False)
        readout_all = p.get("readout_from", "top") == "all"
        rung, rung_from = p.get("feedback_first", 0), p.get("feedback_first_from", "all")
        if rung_from not in ("all", "top"):
            raise ValueError("feedback_first_from must be all or top")
        h1 = w.dense("h1", neurons(p["width"]), p["width"])
        w.inputs(h1, eye_size, "eye")
        w.inputs(h1, 3, "body")
        if recurrent:
            w.connect(h1, h1)
        self.layers = [h1]
        if p.get("reservoir", 0) > 0:
            self.reservoir = w.dense("reservoir", neurons(p["reservoir"]), p["reservoir"])
            w.connect(h1, self.reservoir)
            if p["reservoir_recurrent"] > 0:
                w.feedback(self.reservoir, self.reservoir, p["reservoir_recurrent"])
        self.readouts = []
        for name in ACTIONS:
            out = w.dense(name.lower(), exr.LayerSpec.dense(1, False, False), 1)
            if readout_all:
                w.connect(h1, out)
            if self.reservoir is not None:
                w.connect(self.reservoir, out)
            net.add_output(out)
            self.readouts.append(out)
        for l in range(2, p["depth"] + 1):
            h = w.dense(f"h{l}", neurons(p["width"]), p["width"])
            for below in (self.layers if skip else self.layers[-1:]):
                w.connect(below, h)
            if recurrent:
                w.connect(h, h)
            if rung > 0 and (rung_from == "all" or l == p["depth"]):
                r = w.dense(f"r{l}", neurons(rung, p.get("feedback_first_model", "relu"),
                                             p.get("feedback_first_habituation", True)), rung)
                w.connect(h, r)
                w.connect(r, h1)  # h1 reads the rung's previous tick (the update order below)
                self.rungs.append(r)
            self.layers.append(h)
            if readout_all:
                for out in self.readouts:
                    w.connect(h, out)
        if not readout_all:
            for out in self.readouts:
                w.connect(self.layers[-1], out)
        if self.rungs:
            # The rungs make forward cycles: h1 .. h<depth>, then the rungs, so h1 reads them one tick late.
            res = [self.reservoir] if self.reservoir is not None else []
            net.set_update_order(self.layers + self.rungs + res + self.readouts)
        weighted = self.layers + self.rungs + ([self.reservoir] if self.reservoir is not None else []) + self.readouts
        w.check(weighted)
        w.init_weights(weighted, p["recurrent_scale"])
        self.metered = self.layers + self.rungs + ([self.reservoir] if self.reservoir is not None else [])
        self.neurons = sum(net.layer_size(l) for l in self.metered)
        self.spikes = self.ticks_seen = 0

    def evolving(self, p):
        """The layers whose weights es.py evolves (besides the readouts) with evolve all:
        the hidden layers, the rungs, and the reservoir only if reservoir_evolve."""
        res = [self.reservoir] if self.reservoir is not None and p.get("reservoir_evolve", False) else []
        return self.layers + self.rungs + res

    def act(self, state, meter=False):
        self.net.set_inputs("eye", encode(state, self.seen_channel))
        self.net.set_inputs("body", body(state))
        total = np.zeros(len(ACTIONS))
        for _ in range(self.ticks):
            self.net.step()
            total += np.asarray(self.net.outputs())
            if meter:
                for layer in self.metered:
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
    """Plays one episode per world seed; returns the means of reward, score, avatars lost, death, steps and,
    with the events reward, each event's count (as ev_<event>; ev_score is the score's own count)."""
    index = [goe.ACTIONS.index(a) for a in ACTIONS]
    events = events_reward(p)
    stats = {k: [] for k in ("reward", "score", "avatars_lost", "dead", "steps")}
    for seed in seeds:
        game.new_episode(seed=int(seed))
        reward, steps = 0.0, 0
        while not game.is_episode_finished():
            reward += game.make_action(index[player.act(game.get_state(), meter)])
            steps += 1
        if events:
            counts = game.episode_events
            for k, v in counts.items():
                stats.setdefault("ev_" + k, []).append(v)
            stats["reward"].append(reward)
            stats["score"].append(counts["score"])
        else:
            lost = game.avatars_lost + int(game.is_player_dead())
            stats["reward"].append(reward - p["w_death"] * lost)
            stats["score"].append(reward)
        stats["avatars_lost"].append(game.avatars_lost)
        stats["dead"].append(float(game.is_player_dead()))
        stats["steps"].append(steps)
    return {k: float(np.mean(v)) for k, v in stats.items()}


PARAMS = {
    "radius": ("auto", "vision radius: the grid has 2r + 1 cells on each side; auto: the furthest the player's "
                       "growing sight reaches in an episode (6 for 2 minutes)"),
    "seen_channel": (False, "one more eye channel marking the cells in sight (the sight grows during the game)"),
    "episode_ticks": (6000, "episode length in game ticks (50 a second; 6000 = 2 minutes = 750 moves)"),
    "reward": ("events", "events: the game's events weighed by w_* (Gardens-of-Eris PR #289); score: the game's "
                         "score, minus w_death per avatar lost (G1)"),
    "w_score": (0.1, "events: reward per point of the game's score (mostly new cells visited)"),
    "w_collect": (5.0, "events: reward per item collected (each once an episode)"),
    "w_apple": (20.0, "events: reward per golden apple collected"),
    "w_use": (2.0, "events: reward per use of the usable in hand (a broken apple eaten)"),
    "w_open": (10.0, "events: reward per door opened (each once an episode)"),
    "w_teleport": (5.0, "events: reward per trip through a teleporter"),
    "w_kill": (10.0, "events: reward per monster, drone or puppet master killed by the player's shots or blasts"),
    "w_mine": (5.0, "events: reward per mine or bomb set off by the player's shots"),
    "w_hurt": (-0.2, "events: reward per energy point lost (negative: a penalty)"),
    "w_death": (50.0, "penalty per avatar lost, the last one included (events: weight -w_death)"),
    "model": ("er", "hidden neurons: er, relu or gate"),
    "width": (128, "hidden neurons per layer"),
    "depth": (1, "hidden layers"),
    "recurrent": (False, "each hidden layer also reads its own previous output"),
    "recurrent_scale": (0.5, "scale of recurrent weights (x sqrt(3 / layer size))"),
    "skip": (False, "the top hidden layer reads every hidden layer below it, not only the one under it"),
    "readout_from": ("top", "the readouts read the top hidden layer (top) or every hidden layer (all)"),
    "reservoir": (0, "echo-state reservoir: neurons reading h1 (0: none); the readouts read it"),
    "reservoir_recurrent": (128, "reservoir neurons that read the whole reservoir"),
    "reservoir_evolve": (False, "es.py evolves the reservoir too (default: it stays the frozen random one)"),
    "feedback_first": (0, "feedback ladder: neurons per rung layer r<k>, which reads h<k> and which h1 reads (0: none)"),
    "feedback_first_from": ("all", "feedback ladder: all (every layer above h1 has a rung) or top (only the top one)"),
    "feedback_first_model": ("relu", "feedback ladder: the rungs' neurons, relu, er or gate"),
    "feedback_first_habituation": (True, "feedback ladder: the rungs habituate (the habituation_* settings)"),
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
    for k, v in control.items():
        if k != "steps":
            t.record(k + "_random", v)
    game.close()
