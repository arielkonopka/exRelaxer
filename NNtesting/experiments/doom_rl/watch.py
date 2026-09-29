#!/usr/bin/env python3
"""Watch an evolved Doom agent play.

Plays games with an agent saved by es.py (a folder with config.json and the
evolved weights, by default the current best agent) and either

- writes a replay: one self-contained HTML page (default) showing, step by
  step, exactly the 160 x 120 gray screen the network saw, the action it
  took, its 8 readouts, which hidden E-R neurons fired, and health, ammo,
  kills and reward; open it in any browser, play, pause, scrub, step; or
- with --live, shows the game in ViZDoom's own window in real time (needs a
  display; the window is the agent's 160 x 120 gray view).

The agent is rebuilt exactly as es.py builds it (the run's net seed, fresh
E-R state at the start of every game); seed 4242 starts the same games as
the fresh tests of doc/doom.md. A replay takes about 6 KB per step (8.75
steps a second): a 3-minute game is about 13 MB.

    python3 NNtesting/experiments/doom_rl/watch.py                      # replay.html, 1 game
    python3 NNtesting/experiments/doom_rl/watch.py --games 3 --minutes 5 --out games.html
    python3 NNtesting/experiments/doom_rl/watch.py --live --speed 0.5   # needs a display
"""
import argparse
import base64
import json
import os
import sys
import time
import zlib

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
DEFAULT_AGENT = os.path.join(REPO, "results", "dynamic", "doom_agent", "untildeath_d1")


def load_agent(folder, theta_name):
    import es
    cfg = json.load(open(os.path.join(folder, "config.json")))
    p = cfg["params"]
    names = [theta_name] if theta_name else ["best_theta.npy", "final_theta.npy"]
    path = next((os.path.join(folder, n) for n in names if os.path.exists(os.path.join(folder, n))), None)
    if path is None:
        raise SystemExit(f"no weights ({' or '.join(names)}) in {folder}")
    net_seed = cfg.get("args", {}).get("net_seed", 0)
    return p, net_seed, np.load(path), path


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--agent", default=DEFAULT_AGENT, help="folder with config.json and the weights")
    ap.add_argument("--theta", default="", help="weights file in that folder (default best_theta.npy)")
    ap.add_argument("--seed", type=int, default=4242, help="game seed (4242: the fresh test games)")
    ap.add_argument("--games", type=int, default=1)
    ap.add_argument("--minutes", type=float, default=3.0,
                    help="stop a game after this many game minutes (0: the run's own limit)")
    ap.add_argument("--out", default="replay.html", help="replay page to write")
    ap.add_argument("--live", action="store_true", help="show ViZDoom's window instead of writing a replay")
    ap.add_argument("--speed", type=float, default=1.0, help="--live: 1 is real time")
    args = ap.parse_args()

    import vizdoom as vzd
    import es
    import experiment as e

    p, net_seed, theta, theta_path = load_agent(args.agent, args.theta)
    if args.minutes > 0:
        p["episode_tics"] = min(p["episode_tics"], int(args.minutes * 60 * 35))
    import exrelaxer as exr
    exr.set_threads(1)
    game = e.make_game(p, args.seed, visible=args.live)
    shaper = e.make_shaper(p)
    buttons = np.eye(len(e.ACTIONS), dtype=int).tolist()
    rng = np.random.default_rng(0)
    ammo = [getattr(vzd.GameVariable, f"AMMO{i}") for i in range(10)]
    step_seconds = e.FRAME_SKIP / 35.0

    frames, steps, games = [], [], []
    for g in range(args.games):
        player = es.build(p, net_seed, theta)  # fresh E-R state every game, as in the tests
        hidden = player.layers[0]
        game.new_episode()
        state = game.get_state()
        shaper.reset(state)
        total, first = 0.0, len(steps)
        while not game.is_episode_finished():
            t0 = time.time()
            screen = state.screen_buffer
            spikes_before = player.spikes
            a = player.act(state, rng, 0.0, meter=True)
            fired = np.abs(np.asarray(player.net.layer_output(hidden))) > 1e-6
            vars_ = dict(health=game.get_game_variable(vzd.GameVariable.HEALTH),
                         kills=game.get_game_variable(vzd.GameVariable.KILLCOUNT),
                         ammo=max(game.get_game_variable(v) for v in ammo))
            game.make_action(buttons[a], e.FRAME_SKIP)
            state = None if game.is_episode_finished() else game.get_state()
            r = shaper.step(state) if state is not None else shaper.end(game)
            total += r
            if args.live:
                time.sleep(max(0.0, step_seconds / args.speed - (time.time() - t0)))
            else:
                frames.append(np.ascontiguousarray(screen, dtype=np.uint8).tobytes())
                steps.append([a, [round(float(v), 3) for v in player.y],
                              base64.b64encode(np.packbits(fired)).decode(),
                              int(player.spikes - spikes_before), int(vars_["health"]), int(vars_["ammo"]),
                              int(vars_["kills"]), round(total, 2)])
        died = bool(shaper.counts["deaths"])
        n = len(steps) - first if not args.live else None
        games.append(dict(first=first, steps=n, died=died, kills=shaper.counts["kills"], reward=round(total, 2)))
        print(f"game {g + 1}: {'died' if died else 'alive at the limit'} after "
              f"{(game.get_episode_time() / 35.0):.0f} s, {shaper.counts['kills']} kills, reward {total:+.2f}")
    game.close()
    if args.live:
        return

    meta = dict(agent=os.path.relpath(args.agent, REPO) if args.agent.startswith(REPO) else args.agent,
                weights=os.path.basename(theta_path), scenario=p["scenario"], seed=args.seed,
                width=160, height=120, step_seconds=step_seconds, actions=e.ACTIONS,
                hidden=int(player.net.layer_size(hidden)), ticks=p["ticks"], games=games, steps=steps)
    blob = base64.b64encode(zlib.compress(b"".join(frames), 9)).decode()
    page = open(os.path.join(HERE, "watch.html")).read()
    page = page.replace("/*META*/null", json.dumps(meta, separators=(",", ":"))).replace("/*FRAMES*/", blob)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as f:
        f.write(page)
    print(f"wrote {args.out} ({len(page) / 1e6:.1f} MB, {len(steps)} steps)")


if __name__ == "__main__":
    main()
