# Terrain-Aware A* for Autonomous Mobile Robots

Standard A* minimises distance and only avoids hard obstacles.
Terrain-aware A* adds:
  * accessibility constraints  - edges steeper than a max slope are not traversable
  * terrain-dependent costs    - slope, uphill effort and surface roughness raise the cost

Run:  `python terrain_astar.py`    -> prints metrics, saves `terrain_astar_result.png`