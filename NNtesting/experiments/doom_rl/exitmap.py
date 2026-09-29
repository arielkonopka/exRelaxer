"""Where a level's exit is, and how far every point of the map is from it.

The exit comes from the map itself: the linedefs with an exit special in
the WAD. The distance is the walking distance around walls: a breadth-first
search on a grid (`res` map units a cell) in which every blocking line
(ViZDoom's `is_blocking`: one-sided walls and impassable lines) is a wall.
Doors count as open, and heights (ledges, steps) are ignored, so it is an
approximation of the real path.
"""
import os
import struct
from collections import deque

import numpy as np

EXIT_SPECIALS = {11, 52, 197}  # normal exits: switch, walk-over, gun (secret exits are left out)


def wad_for(scenario):
    import vizdoom as vzd
    if scenario == "map01":
        return os.path.join(os.path.dirname(vzd.__file__), "freedoom2.wad"), "MAP01"
    return None, None


def exit_points(wad, map_name):
    """Midpoints of the map's exit lines, from a Doom-format WAD."""
    b = open(wad, "rb").read()
    _, n, off = struct.unpack("<4sii", b[:12])
    lumps = [struct.unpack("<ii8s", b[off + 16 * i:off + 16 * i + 16]) for i in range(n)]
    names = [x[2].rstrip(b"\0").decode("ascii", "replace") for x in lumps]
    at = names.index(map_name)

    def lump(name):
        pos, size, _ = lumps[names.index(name, at)]
        return b[pos:pos + size]

    verts = lump("VERTEXES")
    vx = [struct.unpack("<hh", verts[j:j + 4]) for j in range(0, len(verts), 4)]
    lines = lump("LINEDEFS")
    out = []
    for j in range(0, len(lines), 14):
        v1, v2, _, special, _, _, _ = struct.unpack("<HHHHHHH", lines[j:j + 14])
        if special in EXIT_SPECIALS:
            out.append(((vx[v1][0] + vx[v2][0]) / 2.0, (vx[v1][1] + vx[v2][1]) / 2.0))
    return out


class DistanceField:
    """Walking distance (map units) to the nearest exit from any point."""

    def __init__(self, sectors, exits, res=16.0):
        lines = [l for s in sectors for l in s.lines if l.is_blocking]
        xs = [v for l in lines for v in (l.x1, l.x2)] + [x for x, _ in exits]
        ys = [v for l in lines for v in (l.y1, l.y2)] + [y for _, y in exits]
        self.res, self.x0, self.y0 = res, min(xs) - 2 * res, min(ys) - 2 * res
        w = int((max(xs) - self.x0) / res) + 3
        h = int((max(ys) - self.y0) / res) + 3
        wall = np.zeros((w, h), bool)
        for l in lines:  # rasterise each wall
            n = int(max(abs(l.x2 - l.x1), abs(l.y2 - l.y1)) / (res / 4)) + 1
            for t in np.linspace(0.0, 1.0, n + 1):
                i, j = self._cell(l.x1 + t * (l.x2 - l.x1), l.y1 + t * (l.y2 - l.y1))
                wall[i, j] = True
        dist = np.full((w, h), np.inf)
        queue = deque()
        for x, y in exits:  # start from the open cells beside each exit line
            ci, cj = self._cell(x, y)
            for i in range(ci - 2, ci + 3):
                for j in range(cj - 2, cj + 3):
                    if 0 <= i < w and 0 <= j < h and not wall[i, j] and dist[i, j] == np.inf:
                        dist[i, j] = 0.0
                        queue.append((i, j))
        steps = [(1, 0, 1.0), (-1, 0, 1.0), (0, 1, 1.0), (0, -1, 1.0)]
        while queue:
            i, j = queue.popleft()
            for di, dj, c in steps:
                a, b = i + di, j + dj
                if 0 <= a < w and 0 <= b < h and not wall[a, b] and dist[a, b] == np.inf:
                    dist[a, b] = dist[i, j] + c
                    queue.append((a, b))
        self.dist = dist * res

    def _cell(self, x, y):
        return int((x - self.x0) / self.res), int((y - self.y0) / self.res)

    def __call__(self, x, y):
        """Distance from (x, y); a point in a wall cell takes its nearest reachable neighbour's."""
        i, j = self._cell(x, y)
        best = np.inf
        for a in range(i - 1, i + 2):
            for b in range(j - 1, j + 2):
                if 0 <= a < self.dist.shape[0] and 0 <= b < self.dist.shape[1]:
                    best = min(best, self.dist[a, b])
        return best
