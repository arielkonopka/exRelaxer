"""The reward for playing Doom, from the game's own state, the same for every
map. Each step's reward is the sum of these events since the last step;
the weights are experiment parameters (see experiment.py).

    hurt       -w per health point lost
    death      -w (large), once
    kill       +w per monster killed
    ammo       +w per round picked up; firing costs -w per round (very small)
    armor      +w per armor point gained
    item       +w per counted item picked up (ITEMCOUNT)
    key        +w per key card or skull key picked up
    door       +w the first time each closed door (or lift) opens
    exit       +w for leaving the level (the episode ends alive, before the timeout)
    idle       -w per step while the player has moved less than `idle_distance`
               map units over the last `idle_steps` steps
"""
import math
from collections import deque

import vizdoom as vzd

KEYS = {"BlueCard", "YellowCard", "RedCard", "BlueSkull", "YellowSkull", "RedSkull"}

VARIABLES = [vzd.GameVariable.HEALTH, vzd.GameVariable.ARMOR, vzd.GameVariable.KILLCOUNT,
             vzd.GameVariable.ITEMCOUNT, vzd.GameVariable.DEAD, vzd.GameVariable.POSITION_X,
             vzd.GameVariable.POSITION_Y] + [getattr(vzd.GameVariable, f"AMMO{i}") for i in range(10)]


class Shaper:
    def __init__(self, weights, idle_steps, idle_distance):
        self.w = weights
        self.idle_steps, self.idle_distance = idle_steps, idle_distance
        self.totals = {}

    def reset(self, state):
        self.prev = self._read(state)
        self.keys = self._keys(state)
        self.closed = {i for i, s in enumerate(state.sectors) if s.ceiling_height <= s.floor_height}
        self.opened = set()
        self.trail = deque([(self.prev["x"], self.prev["y"])], maxlen=self.idle_steps)
        self.totals = {k: 0.0 for k in ("hurt", "death", "kill", "ammo", "fire", "armor", "item", "key", "door",
                                        "exit", "idle")}
        self.counts = {k: 0 for k in ("damage", "kills", "items", "keys", "doors", "ammo_picked", "exits", "deaths")}
        self.distance = 0.0

    @staticmethod
    def _read(state):
        v = state.game_variables
        return {"health": v[0], "armor": v[1], "kills": v[2], "items": v[3], "dead": v[4], "x": v[5], "y": v[6],
                "ammo": sum(v[7:17])}

    @staticmethod
    def _keys(state):
        return sum(o.name in KEYS for o in state.objects)

    def _add(self, name, value):
        self.totals[name] += value
        return value

    def step(self, state):
        """The reward for the step that led to `state` (None when the episode ended)."""
        if state is None:
            return 0.0
        now, w, r = self._read(state), self.w, 0.0
        lost = self.prev["health"] - now["health"]
        if lost > 0:
            r += self._add("hurt", -w["hurt"] * lost)
            self.counts["damage"] += lost
        if now["kills"] > self.prev["kills"]:
            r += self._add("kill", w["kill"] * (now["kills"] - self.prev["kills"]))
            self.counts["kills"] += int(now["kills"] - self.prev["kills"])
        ammo = now["ammo"] - self.prev["ammo"]
        if ammo > 0:
            r += self._add("ammo", w["ammo"] * ammo)
            self.counts["ammo_picked"] += int(ammo)
        elif ammo < 0:
            r += self._add("fire", w["fire"] * ammo)
        if now["armor"] > self.prev["armor"]:
            r += self._add("armor", w["armor"] * (now["armor"] - self.prev["armor"]))
        if now["items"] > self.prev["items"]:
            r += self._add("item", w["item"] * (now["items"] - self.prev["items"]))
            self.counts["items"] += int(now["items"] - self.prev["items"])
        keys = self._keys(state)
        if keys < self.keys:
            r += self._add("key", w["key"] * (self.keys - keys))
            self.counts["keys"] += self.keys - keys
        self.keys = keys
        for i in self.closed - self.opened:
            s = state.sectors[i]
            if s.ceiling_height > s.floor_height + 16:
                self.opened.add(i)
                r += self._add("door", w["door"])
                self.counts["doors"] += 1
        self.distance += math.hypot(now["x"] - self.prev["x"], now["y"] - self.prev["y"])
        self.trail.append((now["x"], now["y"]))
        if len(self.trail) == self.idle_steps:
            x0, y0 = self.trail[0]
            if math.hypot(now["x"] - x0, now["y"] - y0) < self.idle_distance:
                r += self._add("idle", -w["idle"])
        self.prev = now
        return r

    def end(self, game):
        """The reward for how the episode ended: death, level exit or timeout."""
        if game.is_player_dead():
            self.counts["deaths"] += 1
            # Health lost to the killing blow is not seen by step(): charge it here.
            lost = max(self.prev["health"], 0.0)
            self._add("hurt", -self.w["hurt"] * lost)
            self.counts["damage"] += lost
            return self._add("death", -self.w["death"]) - self.w["hurt"] * lost
        if game.get_episode_time() < game.get_episode_timeout() - 1:
            self.counts["exits"] += 1
            return self._add("exit", self.w["exit"])
        return 0.0
