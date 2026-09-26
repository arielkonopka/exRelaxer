#!/usr/bin/env python3
"""Spiral sampling of images, exactly as exRelaxer's Spiral retina does it.

Sample 0 is the image centre ((width - 1) / 2, (height - 1) / 2); the next
samples follow the Archimedean spiral r = spacing * theta / (2 pi) outwards:
turns `spacing` pixels apart, consecutive samples `spacing` pixels apart
along the curve, up to the largest circle inside the image (or --radius).
Each sample interpolates its 4 nearest pixels (bilinear), summed in float32
in the same order as the C++ retina, so the values are bit-identical to its
sums.

  spiral.py info --width W --height H                    samples, radius, turns
  spiral.py points --width W --height H OUT.csv          sample positions
  spiral.py image IN.png [--out OUT.npy|.csv] [--polar OUT.png]
                         [--reconstruct OUT.png] [--overlay OUT.png]
  spiral.py dataset IN_images.npy OUT.npy                N x C x H x W -> N x C x 1 x S

Common options: --spacing S (default 1), --radius R (default 0 = the largest
circle inside the image). Image options: --grey (one channel), --raw (keep
0..255; default scales to 0..1, the range to feed a retina).

Needs numpy, and Pillow for image files.
"""
import argparse
import csv
import math
import sys
from pathlib import Path

import numpy as np


class Spiral:
    """Sample positions and bilinear taps for a width x height image."""

    def __init__(self, width, height, spacing=1.0, radius=0.0):
        if width < 1 or height < 1:
            raise ValueError("the image must not be empty")
        # The C++ retina takes spacing and radius as float: round them the same way.
        spacing = float(np.float32(spacing))
        radius = float(np.float32(radius))
        if not spacing > 0.0:
            raise ValueError("spacing must be > 0")
        self.width, self.height, self.spacing = width, height, spacing
        cx, cy = (width - 1.0) / 2.0, (height - 1.0) / 2.0
        max_radius = min(cx, cy)
        if radius > 0.0:
            max_radius = min(max_radius, radius)
        self.centre, self.max_radius = (cx, cy), max_radius

        # Same recurrence as retina::samplePoints (double precision).
        b = spacing / (2.0 * math.pi)
        xs, ys, thetas = [cx], [cy], [0.0]
        theta = 0.0
        while True:
            r = b * theta
            theta += spacing / math.sqrt(r * r + b * b)
            nxt = b * theta
            if nxt > max_radius:
                break
            xs.append(cx + nxt * math.cos(theta))
            ys.append(cy + nxt * math.sin(theta))
            thetas.append(theta)
        self.x, self.y, self.theta = np.array(xs), np.array(ys), np.array(thetas)

        # Bilinear taps: 4 per sample (weight 0 where the C++ retina has no tap).
        W, H = width, height
        x0 = np.minimum(np.floor(self.x).astype(np.int64), W - 2 if W > 1 else 0)
        y0 = np.minimum(np.floor(self.y).astype(np.int64), H - 2 if H > 1 else 0)
        fx = self.x - x0 if W > 1 else np.zeros_like(self.x)
        fy = self.y - y0 if H > 1 else np.zeros_like(self.y)
        weights = np.stack([(1 - fx) * (1 - fy), fx * (1 - fy), (1 - fx) * fy, fx * fy], axis=1)
        pixels = np.stack([y0 * W + x0, y0 * W + x0 + 1, (y0 + 1) * W + x0, (y0 + 1) * W + x0 + 1], axis=1)
        present = weights > 0.0
        self.tap_pixels = np.where(present, pixels, 0)       # in range wherever the weight is > 0
        self.tap_weights = np.where(present, weights, 0.0).astype(np.float32)

    def __len__(self):
        return len(self.x)

    @property
    def turns(self):
        return int(self.theta[-1] // (2 * math.pi)) + 1 if len(self) > 1 else 0

    def sample(self, images):
        """... x H x W -> ... x samples (float32), bit-identical to the retina's sums."""
        images = np.asarray(images, dtype=np.float32)
        if images.shape[-2:] != (self.height, self.width):
            raise ValueError(f"images are {images.shape[-1]} x {images.shape[-2]}, "
                             f"the spiral is for {self.width} x {self.height}")
        flat = images.reshape(images.shape[:-2] + (self.height * self.width,))
        total = np.zeros(images.shape[:-2] + (len(self),), dtype=np.float32)
        for t in range(4):  # tap order of the C++ retina; a weight-0 tap adds exactly 0
            total = total + self.tap_weights[:, t] * flat[..., self.tap_pixels[:, t]]
        return total

    def reconstruct(self, values):
        """C x samples -> C x H x W: each sample spread back over its 4 pixels."""
        values = np.asarray(values, dtype=np.float64)
        out = np.zeros((values.shape[0], self.height * self.width))
        weight = np.zeros(self.height * self.width)
        for t in range(4):
            np.add.at(weight, self.tap_pixels[:, t], self.tap_weights[:, t])
            for c in range(values.shape[0]):
                np.add.at(out[c], self.tap_pixels[:, t], self.tap_weights[:, t] * values[c])
        covered = weight > 1e-9
        out[:, covered] /= weight[covered]
        return out.reshape(values.shape[0], self.height, self.width)

    def polar(self, values):
        """C x samples -> C x turns x bins: row = turn (distance), column = angle.

        Inner turns have fewer samples than columns; each row is interpolated
        around the circle between its turn's samples.
        """
        values = np.asarray(values, dtype=np.float64)
        bins = max(1, int(round(2 * math.pi * self.max_radius / self.spacing)))
        centres = (np.arange(bins) + 0.5) / bins * 2 * math.pi
        turn = (self.theta // (2 * math.pi)).astype(int)
        angle = self.theta % (2 * math.pi)
        out = np.zeros((values.shape[0], self.turns, bins), dtype=np.float32)
        for t in range(self.turns):
            mine = turn == t
            for c in range(values.shape[0]):
                out[c, t] = np.interp(centres, angle[mine], values[c, mine], period=2 * math.pi)
        return out


# --- image files -------------------------------------------------------------------

def load_image(path, grey):
    from PIL import Image
    with Image.open(path) as img:
        img = img.convert("L" if grey or img.mode in ("L", "1", "I;16") else "RGB")
        pixels = np.asarray(img, dtype=np.float32)
    return pixels[None] if pixels.ndim == 2 else pixels.transpose(2, 0, 1)  # C x H x W


def save_png(path, image, value_max):
    """C x H x W (C = 1 or 3), values 0..value_max, as a PNG."""
    from PIL import Image
    scaled = np.clip(np.asarray(image, dtype=np.float64) / value_max * 255.0, 0, 255).astype(np.uint8)
    array = scaled[0] if scaled.shape[0] == 1 else scaled.transpose(1, 2, 0)
    Image.fromarray(array).save(path)


def save_overlay(path, image, spiral, value_max, scale=4):
    """The image enlarged `scale` times, with the samples marked in order (red centre -> blue edge)."""
    from PIL import Image, ImageDraw
    grey = np.asarray(image, dtype=np.float64).mean(axis=0)
    base = Image.fromarray(np.clip(grey / value_max * 255, 0, 255).astype(np.uint8)).convert("RGB")
    base = base.resize((spiral.width * scale, spiral.height * scale), Image.Resampling.NEAREST)
    draw = ImageDraw.Draw(base)
    n = len(spiral)
    for k in range(n):
        f = k / max(n - 1, 1)
        colour = (int(255 * (1 - f)), 40, int(255 * f))
        px, py = (spiral.x[k] + 0.5) * scale, (spiral.y[k] + 0.5) * scale
        draw.point((px, py), fill=colour)
        if k > 0:
            draw.line([((spiral.x[k - 1] + 0.5) * scale, (spiral.y[k - 1] + 0.5) * scale), (px, py)],
                      fill=colour, width=1)
    base.save(path)


# --- commands -----------------------------------------------------------------------

def cmd_info(args):
    s = Spiral(args.width, args.height, args.spacing, args.radius)
    print(f"{args.width} x {args.height}, spacing {s.spacing:g}: {len(s)} samples within radius "
          f"{s.max_radius:g} ({s.turns} turns); centre ({s.centre[0]:g}, {s.centre[1]:g}); "
          f"{len(s) / (args.width * args.height):.1%} of the pixel count")


def cmd_points(args):
    s = Spiral(args.width, args.height, args.spacing, args.radius)
    with open(args.out, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["k", "x", "y", "radius", "theta"])
        for k in range(len(s)):
            writer.writerow([k, f"{s.x[k]:.17g}", f"{s.y[k]:.17g}",
                             f"{math.hypot(s.x[k] - s.centre[0], s.y[k] - s.centre[1]):.17g}", f"{s.theta[k]:.17g}"])
    print(f"{len(s)} samples written to {args.out}")


def cmd_image(args):
    image = load_image(args.input, args.grey)
    if not args.raw:
        image = image / np.float32(255.0)
    value_max = 255.0 if args.raw else 1.0
    s = Spiral(image.shape[2], image.shape[1], args.spacing, args.radius)
    values = s.sample(image)  # C x samples
    print(f"{args.input}: {image.shape[0]} x {image.shape[1]} x {image.shape[2]} -> "
          f"{values.shape[0]} x {values.shape[1]} samples ({s.turns} turns)")
    if args.out:
        out = Path(args.out)
        if out.suffix == ".csv":
            with open(out, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["k", "x", "y"] + [f"channel{c}" for c in range(values.shape[0])])
                for k in range(len(s)):
                    writer.writerow([k, f"{s.x[k]:.6f}", f"{s.y[k]:.6f}"] + [f"{v:.9g}" for v in values[:, k]])
        else:
            np.save(out, values)
        print(f"  samples: {out}")
    if args.polar:
        save_png(args.polar, s.polar(values), value_max)
        print(f"  polar view (row = turn, column = angle): {args.polar}")
    if args.reconstruct:
        save_png(args.reconstruct, s.reconstruct(values), value_max)
        print(f"  reconstruction from the samples: {args.reconstruct}")
    if args.overlay:
        save_overlay(args.overlay, image, s, value_max)
        print(f"  sample order on the image: {args.overlay}")


def cmd_dataset(args):
    images = np.load(args.input, mmap_mode="r")
    if images.ndim != 4:
        sys.exit(f"{args.input}: expected N x C x H x W, got shape {images.shape}")
    scale = np.float32(1.0) if args.raw else np.float32(1.0 / 255.0)
    s = Spiral(images.shape[3], images.shape[2], args.spacing, args.radius)
    out = np.lib.format.open_memmap(args.output, mode="w+", dtype=np.float32,
                                    shape=(images.shape[0], images.shape[1], 1, len(s)))
    batch = max(1, (64 << 20) // (images[0].size * 4))
    for start in range(0, len(images), batch):
        chunk = np.asarray(images[start:start + batch], dtype=np.float32)
        if not args.raw:
            chunk = chunk * scale
        out[start:start + len(chunk), :, 0, :] = s.sample(chunk)
    out.flush()
    print(f"{args.input}: {images.shape} -> {args.output}: {out.shape} (N x C x 1 x samples, float32"
          f"{', 0..255' if args.raw else ', 0..1'})")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    def spiral_options(p, size=True):
        if size:
            p.add_argument("--width", type=int, required=True)
            p.add_argument("--height", type=int, required=True)
        p.add_argument("--spacing", type=float, default=1.0, help="pixels between samples and turns (default 1)")
        p.add_argument("--radius", type=float, default=0.0, help="outermost radius; 0 = largest inside the image")

    p = sub.add_parser("info", help="samples, radius and turns for an image size")
    spiral_options(p)
    p.set_defaults(func=cmd_info)

    p = sub.add_parser("points", help="write the sample positions as CSV")
    spiral_options(p)
    p.add_argument("out")
    p.set_defaults(func=cmd_points)

    p = sub.add_parser("image", help="sample one image file")
    p.add_argument("input")
    spiral_options(p, size=False)
    p.add_argument("--grey", action="store_true", help="convert to one grey channel")
    p.add_argument("--raw", action="store_true", help="keep 0..255 (default: 0..1)")
    p.add_argument("--out", help="samples as .npy (C x samples) or .csv")
    p.add_argument("--polar", help="PNG: rows = turns (distance), columns = angle")
    p.add_argument("--reconstruct", help="PNG: the image rebuilt from the samples")
    p.add_argument("--overlay", help="PNG: the sample path on the image")
    p.set_defaults(func=cmd_image)

    p = sub.add_parser("dataset", help="sample a prepared image array (N x C x H x W .npy)")
    p.add_argument("input")
    p.add_argument("output")
    spiral_options(p, size=False)
    p.add_argument("--raw", action="store_true", help="keep 0..255 (default: 0..1)")
    p.set_defaults(func=cmd_dataset)

    args = parser.parse_args()
    try:
        args.func(args)
    except (ValueError, OSError) as e:
        sys.exit(f"spiral.py: {e}")


if __name__ == "__main__":
    main()
