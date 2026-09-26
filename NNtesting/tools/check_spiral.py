#!/usr/bin/env python3
"""Checks that spiral.py samples exactly like the C++ Spiral retina.

  check_spiral.py PATH/TO/retina_dump

Runs the retina (via retina_dump) and spiral.py on random images of several
sizes, spacings and radii; sample positions and values must be bit-identical.
Run by CTest as spiral_matches_retina.
"""
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from spiral import Spiral  # noqa: E402

CASES = [  # channels, height, width, spacing, radius
    (1, 200, 320, 1.0, 0.0), (3, 31, 47, 0.7, 0.0), (1, 64, 64, 2.5, 0.0), (2, 100, 100, 1.0, 20.0),
    (1, 17, 1, 1.0, 0.0), (1, 1, 9, 1.0, 0.0), (1, 29, 33, 0.33, 0.0),
]


def main():
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    dump = sys.argv[1]
    rng = np.random.default_rng(1)
    failures = 0
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        for c, h, w, spacing, radius in CASES:
            image = rng.uniform(-9.5, 9.5, size=(c, h, w)).astype(np.float32)  # within the neurons' +-10
            image.tofile(tmp / "image.f32")
            subprocess.run([dump, str(c), str(h), str(w), repr(spacing), repr(radius), str(tmp / "image.f32"),
                            str(tmp / "values.f32"), str(tmp / "points.f64")], check=True)
            retina_values = np.fromfile(tmp / "values.f32", dtype=np.float32)
            retina_points = np.fromfile(tmp / "points.f64", dtype=np.float64).reshape(-1, 2)
            s = Spiral(w, h, spacing, radius)
            values = s.sample(image).reshape(-1)
            points_ok = len(retina_points) == len(s) and np.array_equal(retina_points[:, 0], s.x) \
                and np.array_equal(retina_points[:, 1], s.y)
            values_ok = retina_values.shape == values.shape and \
                np.array_equal(retina_values.view(np.uint32), values.view(np.uint32))
            failures += not (points_ok and values_ok)
            print(f"{c} x {h} x {w}, spacing {spacing}, radius {radius}: {len(s)} samples, "
                  f"positions {'identical' if points_ok else 'DIFFER'}, values {'identical' if values_ok else 'DIFFER'}")
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
