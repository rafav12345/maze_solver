"""Dijkstra's algorithm maze solver.

Finds the shortest path by total cost in weighted mazes. In unweighted mazes
(all costs = 1.0), behaves like BFS but with more overhead.
"""

from __future__ import annotations

import heapq
from typing import Iterator

from core.events import AlgorithmEvent
from core.grid import Grid, GridPosition
from solvers.base_solver import BaseSolver


class DijkstraSolver(BaseSolver):
    """Solves a maze using Dijkstra's algorithm.

    Uses a priority queue keyed on cumulative path cost.
    Guarantees optimal path in weighted mazes.
    """

    name = "Dijkstra"

    def __init__(self, grid: Grid, start: GridPosition, goal: GridPosition) -> None:
        super().__init__(grid, start, goal)

    def solve(self) -> Iterator[AlgorithmEvent]:
        """Solve using Dijkstra's algorithm."""
        grid = self._grid
        grid.reset_visited()

        # Priority queue: (cost, tie-breaker, position)
        counter = 0
        pq: list[tuple[float, int, GridPosition]] = [(0.0, counter, self._start)]
        cost_so_far: dict[GridPosition, float] = {self._start: 0.0}
        parent: dict[GridPosition, GridPosition | None] = {self._start: None}

        while pq:
            current_cost, _, current = heapq.heappop(pq)

            # Skip if we already found a better path
            if current_cost > cost_so_far.get(current, float("inf")):
                continue

            grid.get_cell(current).visited = True
            yield self._emit_visit(current, parent.get(current))

            if current == self._goal:
                path = self._reconstruct_path(parent, current)
                total_cost = sum(grid.get_cell(p).cost for p in path)
                yield self._emit_path_found(path, total_cost)
                return

            for neighbor in grid.get_passable_neighbors(current):
                new_cost = current_cost + grid.get_cell(neighbor).cost
                if new_cost < cost_so_far.get(neighbor, float("inf")):
                    cost_so_far[neighbor] = new_cost
                    parent[neighbor] = current
                    counter += 1
                    heapq.heappush(pq, (new_cost, counter, neighbor))

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
