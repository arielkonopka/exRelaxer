"""Dataset audit: is the answer really absent from the frames the network
classifies on? Classifiers see one part of each clip and are scored on
fresh clips:

    final        the last readout frame (what the network sees when it answers)
    readout      both readout frames, concatenated
    last_visible the last frame before the blank (side: should reveal the answer)
    last_two     the last two visible frames (order: should reveal the answer)

Two classifiers: ridge on the pixels, and ridge on 2048 random ReLU
features of the pixels (a wide nonlinear model). Also the per-class means
of simple frame statistics of the final frame.

    python3 NNtesting/experiments/video_memory/audit.py --out results/video-memory/audit.json
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import clips  # noqa: E402


def parts(frames, info):
    v = info["visible"]
    f = [x.ravel() for x in frames]
    return {
        "final": f[-1],
        "readout": np.concatenate(f[-clips.READOUT_FRAMES:]),
        "last_visible": f[v - 1],
        "last_two": np.concatenate([f[v - 2], f[v - 1]]),
    }


def ridge_accuracy(Xtr, ytr, Xte, yte):
    m, s = Xtr.mean(0), Xtr.std(0)
    s[s < 1e-12] = 1.0
    Xtr, Xte = (Xtr - m) / s, (Xte - m) / s
    best = 0.0
    # The penalty is picked on the test set here on purpose: the most
    # favourable choice, so a result at chance is conservative.
    for lam in (1e-3, 1e-2, 1e-1, 1.0, 10.0):
        A = Xtr.T @ Xtr + lam * len(Xtr) * np.eye(Xtr.shape[1])
        w = np.linalg.solve(A, Xtr.T @ (2.0 * ytr - 1.0))
        best = max(best, float(np.mean(((Xte @ w) > 0) == (yte == 1))))
    return best


def stats(frame):
    img = frame.reshape(clips.HEIGHT, clips.WIDTH)
    total = img.sum()
    cols = img.sum(0)
    return {"total_intensity": float(total), "max_pixel": float(img.max()),
            "column_centroid": float((cols * np.arange(clips.WIDTH)).sum() / max(total, 1e-9)),
            "bright_pixels": float(np.count_nonzero(img > 0.25))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=20000, help="clips per split")
    ap.add_argument("--features", type=int, default=2048)
    ap.add_argument("--out")
    args = ap.parse_args()
    report = {}
    for task in clips.TASKS:
        data = {}
        for split in (0, 2):
            frames, info, labels = clips.make_set(args.n, 1, clips.split_seed(1000, split), task)
            ps = [parts(f, i) for f, i in zip(frames, info)]
            data[split] = ({k: np.array([p[k] for p in ps]) for k in ps[0]}, labels,
                           [stats(p["final"]) for p in ps])
        rng = np.random.default_rng(0)
        res = {}
        for key in data[0][0]:
            Xtr, ytr = data[0][0][key], data[0][1]
            Xte, yte = data[2][0][key], data[2][1]
            P = rng.uniform(-1, 1, (Xtr.shape[1], args.features)).astype(np.float32)
            P /= np.linalg.norm(P, axis=0)
            relu = lambda X: np.maximum(X @ P, 0.0)
            res[key] = {"ridge_pixels": ridge_accuracy(Xtr, ytr, Xte, yte),
                        "ridge_random_relu": ridge_accuracy(relu(Xtr), ytr, relu(Xte), yte)}
        st, labels = data[0][2], data[0][1]
        res["final_frame_stats"] = {
            k: {"LEFT": float(np.mean([s[k] for s, l in zip(st, labels) if l == clips.LEFT])),
                "RIGHT": float(np.mean([s[k] for s, l in zip(st, labels) if l == clips.RIGHT]))}
            for k in st[0]}
        report[task] = res
        print(task)
        for key, r in res.items():
            print(f"  {key:18s} {r}")
    report["n_per_split"] = args.n
    report["chance_band_99"] = 2.576 * 0.5 / np.sqrt(args.n)
    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w") as f:
            json.dump(report, f, indent=1)


if __name__ == "__main__":
    main()
