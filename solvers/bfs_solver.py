"""BFS maze solver — breadth-first search using a queue.

Guarantees the shortest path (by number of cells) in an unweighted maze.
"""

from __future__ import annotations

from collections import deque
from typing import Iterator

from core.events import AlgorithmEvent
from core.grid import Grid, GridPosition
from solvers.base_solver import BaseSolver


class BFSSolver(BaseSolver):
    """Solves a maze using breadth-first search.

    Explores all cells at distance d before distance d+1.
    Guarantees shortest path in unweighted mazes.
    """

    name = "BFS"

    def __init__(self, grid: Grid, start: GridPosition, goal: GridPosition) -> None:
        super().__init__(grid, start, goal)

    def solve(self) -> Iterator[AlgorithmEvent]:
        """Solve using BFS with a deque."""
        grid = self._grid
        grid.reset_visited()

        queue: deque[GridPosition] = deque([self._start])
        parent: dict[GridPosition, GridPosition | None] = {self._start: None}
        grid.get_cell(self._start).visited = True

        yield self._emit_visit(self._start)

        while queue:
            current = queue.popleft()

            if current == self._goal:
                path = self._reconstruct_path(parent, current)
                cost = sum(grid.get_cell(p).cost for p in path)
                yield self._emit_path_found(path, cost)
                return

            for neighbor in grid.get_passable_neighbors(current):
                if neighbor not in parent:
                    parent[neighbor] = current
                    grid.get_cell(neighbor).visited = True
                    queue.append(neighbor)
                    yield self._emit_visit(neighbor, current)

        yield self._emit_path_not_found()

    def _reconstruct_path(
        self, parent: dict[GridPosition, GridPosition | None], current: GridPosition
    ) -> list[GridPosition]:
        path: list[GridPosition] = []
        node: GridPosition | None = current
        while node is not None:
            path.append(node)
            node = parent[node]
        path.reverse()
        return path
