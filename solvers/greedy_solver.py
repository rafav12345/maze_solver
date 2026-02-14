"""Greedy best-first search maze solver.

Uses only the heuristic (estimated distance to goal) to guide search.
Fast but does NOT guarantee optimal paths — can be tricked by dead ends.
"""

from __future__ import annotations

import heapq
from typing import Iterator

from core.events import AlgorithmEvent
from core.grid import Grid, GridPosition
from solvers.astar_solver import manhattan_distance
from solvers.base_solver import BaseSolver


class GreedySolver(BaseSolver):
    """Solves a maze using greedy best-first search.

    Priority = h(n) only (no path cost). Explores cells that appear
    closest to the goal first. Does not guarantee shortest path.
    """

    name = "Greedy"

    def __init__(self, grid: Grid, start: GridPosition, goal: GridPosition) -> None:
        super().__init__(grid, start, goal)

    def solve(self) -> Iterator[AlgorithmEvent]:
        """Solve using greedy best-first search."""
        grid = self._grid
        grid.reset_visited()

        counter = 0
        h0 = manhattan_distance(self._start, self._goal)
        pq: list[tuple[float, int, GridPosition]] = [(h0, counter, self._start)]
        parent: dict[GridPosition, GridPosition | None] = {self._start: None}
        visited: set[GridPosition] = set()

        while pq:
            _, _, current = heapq.heappop(pq)

            if current in visited:
                continue

            visited.add(current)
            grid.get_cell(current).visited = True
            yield self._emit_visit(current, parent.get(current))

            if current == self._goal:
                path = self._reconstruct_path(parent, current)
                cost = sum(grid.get_cell(p).cost for p in path)
                yield self._emit_path_found(path, cost)
                return

            for neighbor in grid.get_passable_neighbors(current):
                if neighbor not in visited:
                    if neighbor not in parent:
                        parent[neighbor] = current
                    h = manhattan_distance(neighbor, self._goal)
                    counter += 1
                    heapq.heappush(pq, (h, counter, neighbor))

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
