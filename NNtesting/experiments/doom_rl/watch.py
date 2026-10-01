#!/usr/bin/env python3
"""Watch an evolved Doom agent play.

Plays games with an agent saved by es.py (a folder with config.json and the
evolved weights, by default the current best agent) and either

- writes a replay (default): a video of exactly the 160 x 120 gray screen
  the network saw (replay.mp4), subtitles with the action,
  health, ammo, kills and reward of every step (replay.srt: VLC and mpv
  show them), and a small page (replay.html) that plays the video with
  the network beside it: its 8 readouts, which hidden E-R neurons fired,
  and step-by-step seeking. The page, the video and the subtitles must stay
  in the same folder. Needs ffmpeg (apt-get install ffmpeg); or
- with --live, shows the game in ViZDoom's own window in real time (needs a
  display; the window is the agent's 160 x 120 gray view).

--video chooses the video format: mp4 (H.264, the default: plays in
browsers and every player), mkv (H.264 with the subtitles inside) or mpg
(MPEG-2: plays in VLC and mpv, not in browsers, so not in the page). The
frames go straight to ffmpeg, so memory does not grow with the game; the
page holds only the per-step numbers (about 100 bytes a step).

The agent is rebuilt exactly as es.py builds it (the run's net seed, fresh
E-R state at the start of every game); seed 4242 starts the same games as
the fresh tests of doc/doom.md.

    python3 NNtesting/experiments/doom_rl/watch.py                      # replay.mp4 + .srt + .html, 1 game
    python3 NNtesting/experiments/doom_rl/watch.py --games 3 --minutes 5 --out games.html
    python3 NNtesting/experiments/doom_rl/watch.py --video mpg          # replay.mpg (MPEG-2) + .srt + .html
    python3 NNtesting/experiments/doom_rl/watch.py --live --speed 0.5   # needs a display
"""
import argparse
import base64
import json
import os
import shutil
import subprocess
import sys
import time

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
    ap.add_argument("--out", default="replay.html",
                    help="replay page to write; the video and subtitles get its name with their own extension")
    ap.add_argument("--video", choices=["mp4", "mkv", "mpg"], default="mp4",
                    help="video format: mp4 or mkv (H.264), mpg (MPEG-2, not playable in browsers)")
    ap.add_argument("--live", action="store_true", help="show ViZDoom's window instead of writing a replay")
    ap.add_argument("--speed", type=float, default=1.0, help="--live: 1 is real time")
    args = ap.parse_args()

    import vizdoom as vzd
    import es
    import experiment as e
    import rewards

    p, net_seed, theta, theta_path = load_agent(args.agent, args.theta)
    if args.minutes > 0:
        p["episode_tics"] = min(p["episode_tics"], int(args.minutes * 60 * 35))
    import exrelaxer as exr
    exr.set_threads(1)
    game = e.make_game(p, args.seed, visible=args.live)
    shaper = rewards.Shaper({k: p["w_" + k] for k in ("hurt", "death", "kill", "ammo", "fire", "armor", "item",
                                                      "key", "door", "exit", "idle")},
                            p["idle_steps"], p["idle_distance"])
    buttons = np.eye(len(e.ACTIONS), dtype=int).tolist()
    rng = np.random.default_rng(0)
    ammo = [getattr(vzd.GameVariable, f"AMMO{i}") for i in range(10)]
    step_seconds = e.FRAME_SKIP / 35.0

    stem = os.path.splitext(os.path.abspath(args.out))[0]
    video_path, srt_path = f"{stem}.{args.video}", stem + ".srt"
    encoder = None if args.live else start_video(video_path, args.video)
    steps, games = [], []
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
                encoder.stdin.write(np.ascontiguousarray(screen, dtype=np.uint8).tobytes())
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

    encoder.stdin.close()
    if encoder.wait() != 0:
        raise SystemExit(f"ffmpeg failed writing {video_path}")
    write_srt(srt_path, steps, games, step_seconds, e.ACTIONS)
    if args.video == "mkv":
        embed_subtitles(video_path, srt_path)

    meta = dict(agent=os.path.relpath(args.agent, REPO) if args.agent.startswith(REPO) else args.agent,
                weights=os.path.basename(theta_path), scenario=p["scenario"], seed=args.seed,
                width=160, height=120, step_seconds=step_seconds, actions=e.ACTIONS,
                hidden=int(player.net.layer_size(hidden)), ticks=p["ticks"], games=games, steps=steps,
                video=os.path.basename(video_path), browser_video=args.video != "mpg")
    page = open(os.path.join(HERE, "watch.html")).read()
    page = page.replace("/*META*/null", json.dumps(meta, separators=(",", ":")))
    with open(args.out, "w") as f:
        f.write(page)
    for path in (video_path, srt_path, args.out):
        print(f"wrote {path} ({os.path.getsize(path) / 1e6:.1f} MB)")
    print(f"{len(steps)} steps; open {args.out} in a browser, or {os.path.basename(video_path)} in any player")


def start_video(path, kind):
    """ffmpeg reading raw 160 x 120 gray frames, one per game step, from stdin."""
    if shutil.which("ffmpeg") is None:
        raise SystemExit("watch.py: ffmpeg not found (sudo apt-get install ffmpeg); --live needs no ffmpeg")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    # The native 160 x 120: players and the page scale it up. Doom at 8.75 steps a second changes much from
    # frame to frame, so the video takes about 3-4 KB a step (a 20-minute game: about 30 MB).
    codec = (["-c:v", "mpeg2video", "-q:v", "3", "-r", "25"] if kind == "mpg"  # MPEG-2 allows no 8.75 fps
             else ["-c:v", "libx264", "-crf", "18", "-preset", "slow", "-g", "35"])
    return subprocess.Popen(
        ["ffmpeg", "-loglevel", "error", "-y", "-f", "rawvideo", "-pix_fmt", "gray", "-s", "160x120",
         "-framerate", "35/4", "-i", "-", "-pix_fmt", "yuv420p", *codec,
         *(["-movflags", "+faststart"] if kind == "mp4" else []), path],
        stdin=subprocess.PIPE)


def write_srt(path, steps, games, step_seconds, actions):
    """One subtitle per step: game, action, health, ammo, kills, reward."""
    def stamp(t):
        ms = int(round(t * 1000))
        return f"{ms // 3600000:02d}:{ms // 60000 % 60:02d}:{ms // 1000 % 60:02d},{ms % 1000:03d}"
    game_of = [g for g, info in enumerate(games) for _ in range(info["steps"])]
    with open(path, "w") as f:
        for i, (a, _y, _bits, spikes, health, ammo, kills, reward) in enumerate(steps):
            g = game_of[i]
            f.write(f"{i + 1}\n{stamp(i * step_seconds)} --> {stamp((i + 1) * step_seconds)}\n"
                    f"game {g + 1} step {i - games[g]['first'] + 1}: {actions[a].lower().replace('_', ' ')}\n"
                    f"health {health}  ammo {ammo}  kills {kills}  reward {reward:+.2f}  spikes {spikes}\n\n")


def embed_subtitles(video_path, srt_path):
    tmp = video_path + ".tmp.mkv"
    subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-i", video_path, "-i", srt_path, "-map", "0", "-map", "1",
                    "-c", "copy", "-metadata:s:s:0", "title=agent", tmp], check=True)
    os.replace(tmp, video_path)


if __name__ == "__main__":
    main()
