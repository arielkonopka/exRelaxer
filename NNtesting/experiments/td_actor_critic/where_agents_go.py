"""Where do trained MAP01 agents go? Replays the last checkpoint of doom_td runs in fresh test games
(no learning) and draws their paths on the map's walking-distance field to the exit.

    python3 where_agents_go.py --runs RUN_DIR [RUN_DIR ...] --games 5 --out paths.png

Each RUN_DIR is a doom_td --checkpoint run on map01 (config.json + checkpoint.pkl). Writes the
picture and a JSON with each game's closest walking distance to the exit and where it was.
"""
import argparse
import json
import os
import pickle
import shutil
import subprocess
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "doom_rl"))
import doom_td as dt  # noqa: E402
import experiment as drl_ex  # noqa: E402

COLOURS = [(255, 80, 80), (80, 200, 255), (255, 160, 0), (220, 120, 255), (140, 255, 140), (255, 255, 255)]


class Unpickler(pickle.Unpickler):
    """Checkpoints pickled by doom_td run as a script name its classes under __main__."""

    def find_class(self, module, name):
        if module == "__main__":
            module = "doom_td"
        return super().find_class(module, name)


def start_video(path):
    """ffmpeg reading raw 160 x 120 gray frames (one per game step, 35/4 a second) from stdin."""
    return subprocess.Popen(
        ["ffmpeg", "-loglevel", "error", "-y", "-f", "rawvideo", "-pix_fmt", "gray", "-s", "160x120",
         "-r", "8.75", "-i", "-", "-vf", "scale=480:360:flags=neighbor", "-c:v", "libx264", "-crf", "28", "-pix_fmt", "yuv420p",
         "-movflags", "+faststart", path], stdin=subprocess.PIPE)


def replay(run, games, seed, record=None):
    cfg = json.load(open(os.path.join(run, "config.json")))
    p = dict(dt.DEFAULTS)
    p.update({k: v for k, v in cfg.items() if k in p})
    scale = cfg["reward_scale"]
    labels = p["features"] == "oracle" or bool(p["attack_reflex"])
    game = dt.make_game("map01", 30_000 + seed, p["skill"], labels, p["until_death"],
                        color=bool(p.get("detector_colour", 0)))
    learner = dt.Learner(p, game.get_available_buttons_size(), cfg["arm"], cfg["seed"])
    saved = Unpickler(open(os.path.join(run, "checkpoint.pkl"), "rb")).load()
    learner.agent, learner.rng = saved["agent"], saved["rng"]
    learner.vplace = saved.get("vplace", learner.vplace)
    if saved.get("landmarks") is not None:
        learner.landmarks, learner.lives, learner.origin = saved["landmarks"]
    learner.detector = saved.get("detector", learner.detector)
    learner.games = saved.get("games", learner.games)
    shaper = dt.drl.make_shaper(dt.doom_rl_params("map01", p["skill"], p["until_death"], p["approach"],
                                                  kill=p["kill_weight"],
                                                  heights=bool(p.get("approach_heights", 0))))
    paths = []
    read = dt.PlaceCells.read
    for n in range(games):
        path = []
        video = start_video(os.path.join(record, f"{os.path.basename(os.path.normpath(run))}_g{n}.mp4")) \
            if record else None

        last = [None]

        def recording(g, path=path, last=last):
            pos = read(g)
            path.append(pos[:2])
            tic = g.get_episode_time()
            if video is not None and not g.is_episode_finished() and tic != last[0]:  # one frame per step
                last[0] = tic
                st = g.get_state()
                if st is not None:
                    video.stdin.write(np.ascontiguousarray(st.screen_buffer, dtype=np.uint8).tobytes())
            return pos
        dt.PlaceCells.read = staticmethod(recording)
        try:
            s = dt.episode(game, learner, scale, False, shaper=shaper)
        finally:
            dt.PlaceCells.read = staticmethod(read)
            if video is not None:
                video.stdin.close()
                video.wait()
        paths.append((path, s))
    return game, paths


def draw(state, field, runs, out):
    from PIL import Image, ImageDraw
    d = field.dist
    w, h = d.shape
    k = 4
    fin = np.isfinite(d)
    v = np.where(fin, d, 0) / np.nanmax(np.where(fin, d, np.nan))
    img = np.zeros((h, w, 3), np.uint8)
    img[..., 0] = np.where(fin.T, v.T * 40 + 25, 12)
    img[..., 1] = np.where(fin.T, (1 - v.T) * 90 + 30, 12)
    img[..., 2] = np.where(fin.T, v.T * 110 + 30, 16)
    im = Image.fromarray(img[::-1]).resize((w * k, h * k), Image.NEAREST)
    dr = ImageDraw.Draw(im)

    def at(x, y):
        i, j = field._cell(x, y)
        return i * k + k // 2, (h - 1 - j) * k + k // 2
    for line in [ln for sec in state.sectors for ln in sec.lines]:
        dr.line([at(line.x1, line.y1), at(line.x2, line.y2)], fill=(200, 200, 200) if line.is_blocking else (90, 90, 90))
    for n, (name, paths) in enumerate(runs):
        c = COLOURS[n % len(COLOURS)]
        for path, _ in paths:
            if len(path) > 1:
                dr.line([at(x, y) for x, y in path], fill=c, width=2)
        dr.text((8, 22 + 14 * n), name, fill=c)
    for (x, y), c, t in (((216, -632), (255, 0, 0), "EXIT"), ((-192, -192), (255, 255, 255), "SPAWN")):
        a, b = at(x, y)
        dr.ellipse([a - 7, b - 7, a + 7, b + 7], fill=c)
        dr.text((a + 9, b - 6), t, fill=c)
    dr.text((8, 6), "MAP01 test games of trained agents (background: walking distance to exit, green near)",
            fill=(255, 255, 255))
    im.save(out)


def keep_best(runs, folder):
    games = [(name, n, s) for name, paths in runs for n, (_, s) in enumerate(paths)]
    picks = {}
    taken = set()
    # Three different games where possible: a game already kept is passed over (several games often
    # tie on the 5-minute cap, and one game may lead in everything).
    for label, key in (("kills", lambda g: (g[2]["kills"], g[2]["seconds"])),
                       ("longest", lambda g: (g[2]["seconds"], g[2]["visited"])),
                       ("doors", lambda g: (g[2]["doors"], g[2]["visited"]))):
        free = [g for g in games if (g[0], g[1]) not in taken] or games
        name, n, s = max(free, key=key)
        taken.add((name, n))
        shutil.copy(os.path.join(folder, f"{name}_g{n}.mp4"), os.path.join(folder, f"best_{label}.mp4"))
        picks[label] = dict(run=name, game=n, kills=s["kills"], seconds=s["seconds"], doors=s["doors"],
                            died=s["died"], game_reward=s["game_reward"], visited=s["visited"])
    exits = [g for g in games if g[2].get("exits", 0) > 0]
    if exits:  # also the quickest game that left the level
        name, n, s = min(exits, key=lambda g: g[2]["seconds"])
        shutil.copy(os.path.join(folder, f"{name}_g{n}.mp4"), os.path.join(folder, "best_exit.mp4"))
        picks["exit"] = dict(run=name, game=n, kills=s["kills"], seconds=s["seconds"], doors=s["doors"],
                             died=s["died"], game_reward=s["game_reward"], visited=s["visited"])
    for name, n, _ in games:
        os.remove(os.path.join(folder, f"{name}_g{n}.mp4"))
    with open(os.path.join(folder, "best.json"), "w") as f:
        json.dump(picks, f, indent=1)
    print("kept", json.dumps(picks), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--games", type=int, default=5)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--out", default="paths.png")
    ap.add_argument("--heights", action="store_true",
                    help="measure and draw the ledge-aware walking distance (exitmap heights=True)")
    ap.add_argument("--record", default=None,
                    help="folder: record every game as video, then keep only the games with the most kills, "
                         "the longest play and the most doors opened, three different games where possible "
                         "(best_kills.mp4, best_longest.mp4, best_doors.mp4)")
    args = ap.parse_args()
    if args.record:
        os.makedirs(args.record, exist_ok=True)
    runs, report = [], {}
    game = None
    for run in args.runs:
        game, paths = replay(run, args.games, args.seed, args.record)
        name = os.path.basename(os.path.normpath(run))
        runs.append((name, paths))
    state = None
    game.new_episode()
    state = game.get_state()
    field = drl_ex.exit_field("map01", state, args.heights)
    for name, paths in runs:
        rows = []
        for path, s in paths:
            dists = [field(x, y) for x, y in path]
            i = int(np.argmin(dists))
            rows.append(dict(closest=float(dists[i]), at=[float(path[i][0]), float(path[i][1])], exits=s["exits"],
                             seconds=s["seconds"], died=s["died"], kills=s["kills"], game_reward=s["game_reward"]))
        report[name] = rows
        print(name, [round(r["closest"]) for r in rows], flush=True)
    draw(state, field, runs, args.out)
    if args.record:
        keep_best(runs, args.record)
    with open(os.path.splitext(args.out)[0] + ".json", "w") as f:
        json.dump(report, f, indent=1)


if __name__ == "__main__":
    main()
