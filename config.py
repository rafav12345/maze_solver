"""Configuration constants, color palettes, and defaults."""

from __future__ import annotations


# --- Window ---
DEFAULT_WINDOW_WIDTH = 1000
DEFAULT_WINDOW_HEIGHT = 750
WINDOW_TITLE = "Maze Decision Engine"

# --- Grid defaults ---
DEFAULT_NUM_ROWS = 20
DEFAULT_NUM_COLS = 30
DEFAULT_CELL_WIDTH = 20
DEFAULT_CELL_HEIGHT = 20
DEFAULT_ORIGIN_X = 50
DEFAULT_ORIGIN_Y = 80

# --- Animation ---
DEFAULT_ANIMATION_DELAY_MS = 30
MIN_ANIMATION_DELAY_MS = 1
MAX_ANIMATION_DELAY_MS = 500

# --- Colors ---
BG_COLOR = "#d9d9d9"
WALL_COLOR = "black"
WALL_WIDTH = 2

# Solver path colors
SOLVER_COLORS: dict[str, dict[str, str]] = {
    "DFS": {"forward": "#e74c3c", "backtrack": "#bdc3c7"},
    "BFS": {"forward": "#3498db", "backtrack": "#85c1e9"},
    "A*": {"forward": "#2ecc71", "backtrack": "#82e0aa"},
    "Dijkstra": {"forward": "#9b59b6", "backtrack": "#d2b4de"},
    "Greedy": {"forward": "#e67e22", "backtrack": "#f0b27a"},
    "Wall Follower (Left)": {"forward": "#1abc9c", "backtrack": "#76d7c4"},
    "Wall Follower (Right)": {"forward": "#f39c12", "backtrack": "#f7dc6f"},
}

# Default solver colors (for unknown solvers)
DEFAULT_FORWARD_COLOR = "red"
DEFAULT_BACKTRACK_COLOR = "gray"

# Heatmap gradient (low cost -> high cost)
HEATMAP_LOW = "#2ecc71"   # green
HEATMAP_HIGH = "#e74c3c"  # red

# --- Stats panel ---
STATS_FONT = ("Courier", 10)
STATS_HEADER_FONT = ("Courier", 11, "bold")
