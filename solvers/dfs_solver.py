"""DFS maze solver. Refactored from the original Maze.__solve_r.

Uses an explicit stack to avoid recursion limits.
"""

from __future__ import annotations

from typing import Iterator

from core.events import AlgorithmEvent
from core.grid import Grid, GridPosition
from solvers.base_solver import BaseSolver


class DFSSolver(BaseSolver):
    """Solves a maze using depth-first search.

    Explores as deep as possible before backtracking. Finds *a* path
    but not necessarily the shortest one.
    """

    name = "DFS"

    def __init__(self, grid: Grid, start: GridPosition, goal: GridPosition) -> None:
        super().__init__(grid, start, goal)

    def solve(self) -> Iterator[AlgorithmEvent]:
        """Solve using iterative DFS with explicit stack."""
        grid = self._grid
        grid.reset_visited()

        # Stack entries: (position, parent_position_or_None)
        stack: list[tuple[GridPosition, GridPosition | None]] = [(self._start, None)]
        parent: dict[GridPosition, GridPosition | None] = {}

        while stack:
            current, came_from = stack.pop()

            if current in parent:
                # Already visited — emit backtrack for the move that led here
                if came_from is not None:
                    yield self._emit_backtrack(came_from, current)
                continue

            parent[current] = came_from
            grid.get_cell(current).visited = True
            yield self._emit_visit(current, came_from)

            # Goal check
            if current == self._goal:
                path = self._reconstruct_path(parent, current)
                cost = sum(grid.get_cell(p).cost for p in path)
                yield self._emit_path_found(path, cost)
                return

            # Push passable unvisited neighbors (reverse order for consistent exploration)
            neighbors = grid.get_passable_unvisited_neighbors(current)
            for neighbor in reversed(neighbors):
                if neighbor not in parent:
                    stack.append((neighbor, current))

        yield self._emit_path_not_found()

    def _reconstruct_path(
        self, parent: dict[GridPosition, GridPosition | None], current: GridPosition
    ) -> list[GridPosition]:
        """Reconstruct path from start to current using parent dict."""
        path: list[GridPosition] = []
        node: GridPosition | None = current
        while node is not None:
            path.append(node)
            node = parent[node]
        path.reverse()
        return path
