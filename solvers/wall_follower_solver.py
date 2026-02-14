"""Wall follower maze solver — left-hand / right-hand rule.

A simple, deterministic strategy that follows one wall. Works on simply-connected
mazes (no loops). Serves as a "dumb" baseline for comparison with smarter algorithms.
"""

from __future__ import annotations

from enum import Enum, auto
from typing import Iterator

from core.events import AlgorithmEvent
from core.grid import Grid, GridPosition
from solvers.base_solver import BaseSolver


class Direction(Enum):
    """Cardinal directions for the wall follower agent."""

    UP = auto()
    RIGHT = auto()
    DOWN = auto()
    LEFT = auto()


# Clockwise turn order
_CW = {
    Direction.UP: Direction.RIGHT,
    Direction.RIGHT: Direction.DOWN,
    Direction.DOWN: Direction.LEFT,
    Direction.LEFT: Direction.UP,
}

# Counter-clockwise turn order
_CCW = {
    Direction.UP: Direction.LEFT,
    Direction.LEFT: Direction.DOWN,
    Direction.DOWN: Direction.RIGHT,
    Direction.RIGHT: Direction.UP,
}

# Direction to (dcol, drow) delta
_DELTAS = {
    Direction.UP: (0, -1),
    Direction.DOWN: (0, 1),
    Direction.LEFT: (-1, 0),
    Direction.RIGHT: (1, 0),
}

# Direction to wall attribute on the current cell
_WALL_ATTRS = {
    Direction.UP: "has_top_wall",
    Direction.DOWN: "has_bottom_wall",
    Direction.LEFT: "has_left_wall",
    Direction.RIGHT: "has_right_wall",
}


class WallFollowerSolver(BaseSolver):
    """Solves a maze using wall following (left-hand or right-hand rule).

    The agent keeps one hand on the wall and follows it until
    reaching the goal. Works on all perfect (simply-connected) mazes.

    Args:
        grid: The maze grid.
        start: Starting position.
        goal: Goal position.
        hand: Which hand to keep on the wall — "left" or "right".
    """

    name = "Wall Follower"

    def __init__(
        self,
        grid: Grid,
        start: GridPosition,
        goal: GridPosition,
        hand: str = "left",
    ) -> None:
        super().__init__(grid, start, goal)
        if hand not in ("left", "right"):
            raise ValueError(f"hand must be 'left' or 'right', got {hand!r}")
        self._hand = hand

    def solve(self) -> Iterator[AlgorithmEvent]:
        """Solve using wall following."""
        grid = self._grid
        grid.reset_visited()

        current = self._start
        facing = Direction.DOWN  # Start facing down (entering from top)
        path: list[GridPosition] = [current]
        grid.get_cell(current).visited = True

        yield self._emit_visit(current)

        # For left-hand rule: try left first, then sweep clockwise
        # For right-hand rule: try right first, then sweep counter-clockwise
        if self._hand == "left":
            preferred_turn = _CCW
            sweep = _CW
        else:
            preferred_turn = _CW
            sweep = _CCW

        # Safety limit to avoid infinite loops on mazes with loops
        max_steps = grid.num_cols * grid.num_rows * 4

        for _ in range(max_steps):
            if current == self._goal:
                clean_path = self._clean_path(path)
                cost = sum(grid.get_cell(p).cost for p in clean_path)
                self._stats.path_length = len(clean_path)
                yield self._emit_path_found(clean_path, cost)
                return

            moved = False
            try_dir = preferred_turn[facing]

            for _ in range(4):
                dc, dr = _DELTAS[try_dir]
                next_pos = GridPosition(current.col + dc, current.row + dr)
                wall_attr = _WALL_ATTRS[try_dir]

                if (
                    grid.in_bounds(next_pos)
                    and not getattr(grid.get_cell(current), wall_attr)
                ):
                    prev = current
                    current = next_pos
                    facing = try_dir
                    path.append(current)

                    if not grid.get_cell(current).visited:
                        grid.get_cell(current).visited = True
                        yield self._emit_visit(current, prev)
                    else:
                        yield self._emit_backtrack(prev, current)

                    moved = True
                    break

                try_dir = sweep[try_dir]

            if not moved:
                yield self._emit_path_not_found()
                return

        yield self._emit_path_not_found()

    def _clean_path(self, path: list[GridPosition]) -> list[GridPosition]:
        """Remove loops from the wall follower path.

        When the agent revisits a cell, it means it went around a dead end.
        Remove the loop to get the actual solution path.
        """
        seen: dict[GridPosition, int] = {}
        clean: list[GridPosition] = []

        for pos in path:
            if pos in seen:
                idx = seen[pos]
                for removed in clean[idx + 1:]:
                    if removed in seen and seen[removed] > idx:
                        del seen[removed]
                clean = clean[: idx + 1]
            else:
                clean.append(pos)
            seen[pos] = len(clean) - 1

        return clean


class LeftWallFollowerSolver(WallFollowerSolver):
    """Wall follower using the left-hand rule."""

    name = "Wall Follower (Left)"

    def __init__(self, grid: Grid, start: GridPosition, goal: GridPosition) -> None:
        super().__init__(grid, start, goal, hand="left")


class RightWallFollowerSolver(WallFollowerSolver):
    """Wall follower using the right-hand rule."""

    name = "Wall Follower (Right)"

    def __init__(self, grid: Grid, start: GridPosition, goal: GridPosition) -> None:
        super().__init__(grid, start, goal, hand="right")
