"""
Terrain-aware A* for autonomous mobile robots.

Standard A* minimises distance and only avoids hard obstacles.
Terrain-aware A* adds:
  * accessibility constraints  - edges steeper than a max slope are not traversable
  * terrain-dependent costs    - slope, uphill effort and surface roughness raise the cost

Run:  python terrain_astar.py            -> prints metrics, saves terrain_astar_result.png
"""
import heapq
import math

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter

CELL = 1.0          # metres per grid cell
MAX_SLOPE = 25.0    # degrees; steeper edges are inaccessible
W_SLOPE = 6.0       # slope penalty weight
W_UP = 2.0          # extra uphill-effort weight
W_ROUGH = 3.0       # surface roughness weight

MOVES = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]


# ----------------------------------------------------------------- terrain
def make_terrain(n=100, seed=7):
    """Synthetic uneven terrain: hills, a ridge with one pass, water, boulders, rough patches."""
    rng = np.random.default_rng(seed)
    base = gaussian_filter(rng.standard_normal((n, n)), 12)
    detail = gaussian_filter(rng.standard_normal((n, n)), 2)
    z = base / np.abs(base).max() * 6 + detail / np.abs(detail).max() * 0.8

    yy, xx = np.mgrid[0:n, 0:n]
    ridge = 22 * np.exp(-((xx - n / 2) ** 2) / (2 * 5**2))
    gap = 1 - 0.95 * np.exp(-((yy - 0.78 * n) ** 2) / (2 * 6**2))   # the pass
    z = z + ridge * gap

    roughness = gaussian_filter(rng.standard_normal((n, n)), 4)
    roughness = (roughness - roughness.min()) / (roughness.max() - roughness.min())

    obstacles = z < np.percentile(z, 3)                    # water / pits
    obstacles |= rng.random((n, n)) < 0.01                 # boulders
    return z, roughness, obstacles


# ------------------------------------------------------------------- costs
def edge_slope(z, a, b):
    d = math.hypot(a[0] - b[0], a[1] - b[1]) * CELL
    dz = z[b] - z[a]
    return math.degrees(math.atan2(abs(dz), d)), dz, d


def standard_edge(z, rough, a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1]) * CELL


def terrain_edge(z, rough, a, b):
    slope, dz, d = edge_slope(z, a, b)
    if slope > MAX_SLOPE:
        return None                                        # inaccessible
    r = 0.5 * (rough[a] + rough[b])
    factor = 1 + W_SLOPE * (slope / MAX_SLOPE) ** 2 + W_UP * max(dz, 0) / d + W_ROUGH * r
    return d * factor


def octile(a, b):
    dr, dc = abs(a[0] - b[0]), abs(a[1] - b[1])
    return (max(dr, dc) + (math.sqrt(2) - 1) * min(dr, dc)) * CELL


# ------------------------------------------------------------------- A*
def astar(z, rough, obstacles, start, goal, edge_cost):
    """Generic 8-connected A*. Heuristic is admissible because every edge cost >= its distance."""
    n, m = z.shape
    open_heap = [(octile(start, goal), 0.0, start)]
    g = {start: 0.0}
    parent = {}
    closed = set()
    while open_heap:
        _, gc, cur = heapq.heappop(open_heap)
        if cur in closed:
            continue
        if cur == goal:
            path = [cur]
            while cur in parent:
                cur = parent[cur]
                path.append(cur)
            return path[::-1]
        closed.add(cur)
        for dr, dc in MOVES:
            nb = (cur[0] + dr, cur[1] + dc)
            if not (0 <= nb[0] < n and 0 <= nb[1] < m) or obstacles[nb] or nb in closed:
                continue
            c = edge_cost(z, rough, cur, nb)
            if c is None:
                continue
            ng = gc + c
            if ng < g.get(nb, math.inf):
                g[nb] = ng
                parent[nb] = cur
                heapq.heappush(open_heap, (ng + octile(nb, goal), ng, nb))
    return None


# --------------------------------------------------------------- metrics
def evaluate(z, rough, path):
    length = slopes = terrain_cost = 0.0
    viol = 0
    max_slope = 0.0
    for a, b in zip(path[:-1], path[1:]):
        s, dz, d = edge_slope(z, a, b)
        r = 0.5 * (rough[a] + rough[b])
        terrain_cost += d * (1 + W_SLOPE * (s / MAX_SLOPE) ** 2 + W_UP * max(dz, 0) / d + W_ROUGH * r)
        length += d
        slopes += s
        max_slope = max(max_slope, s)
        viol += s > MAX_SLOPE
    return {
        "length (m)": length,
        "terrain cost": terrain_cost,
        "mean slope (deg)": slopes / (len(path) - 1),
        "max slope (deg)": max_slope,
        "inaccessible steps": int(viol),
    }


# ------------------------------------------------------------------ main
def main():
    z, rough, obstacles = make_terrain()
    start, goal = (12, 8), (12, 91)
    obstacles[start] = obstacles[goal] = False

    p_std = astar(z, rough, obstacles, start, goal, standard_edge)
    p_ter = astar(z, rough, obstacles, start, goal, terrain_edge)
    if p_std is None or p_ter is None:
        raise SystemExit("No path found - adjust MAX_SLOPE or the terrain seed.")

    m_std, m_ter = evaluate(z, rough, p_std), evaluate(z, rough, p_ter)

    print(f"{'metric':<22}{'standard A*':>14}{'terrain-aware':>16}")
    for k in m_std:
        print(f"{k:<22}{m_std[k]:>14.1f}{m_ter[k]:>16.1f}")

    # ---- plots
    gy, gx = np.gradient(z, CELL)
    slope_map = np.degrees(np.arctan(np.hypot(gx, gy)))
    fig, ax = plt.subplots(1, 3, figsize=(18, 5.6))

    def draw(a, img, title, cmap, label):
        im = a.imshow(img, cmap=cmap, origin="upper")
        plt.colorbar(im, ax=a, fraction=0.046, label=label)
        for p, col, lab in ((p_std, "red", "Standard A*"), (p_ter, "cyan", "Terrain-aware A*")):
            r, c = zip(*p)
            a.plot(c, r, color=col, lw=2.2, label=lab)
        a.plot(start[1], start[0], "wo", mec="k", ms=9)
        a.plot(goal[1], goal[0], "w*", mec="k", ms=14)
        a.set_title(title)
        a.legend(loc="lower left")

    draw(ax[0], np.where(obstacles, np.nan, z), "Elevation", "terrain", "elevation (m)")
    draw(ax[1], slope_map, f"Slope (limit {MAX_SLOPE:.0f}°)", "magma", "slope (deg)")

    keys = ["length (m)", "terrain cost", "max slope (deg)"]
    xs = np.arange(len(keys))
    ax[2].bar(xs - 0.2, [m_std[k] for k in keys], 0.4, color="red", label="Standard A*")
    ax[2].bar(xs + 0.2, [m_ter[k] for k in keys], 0.4, color="teal", label="Terrain-aware A*")
    ax[2].set_xticks(xs, keys)
    ax[2].set_title("Route comparison")
    ax[2].legend()

    plt.tight_layout()
    plt.savefig("terrain_astar_result.png", dpi=130)
    print("saved terrain_astar_result.png")


if __name__ == "__main__":
    main()
