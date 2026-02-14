"""A* maze solver with pluggable heuristics.

Combines path cost (like Dijkstra) with a heuristic estimate of remaining
distance. With an admissible heuristic, guarantees optimal paths.
"""

from __future__ import annotations

import heapq
import math
from enum import Enum, auto
from typing import Callable, Iterator

from core.events import AlgorithmEvent
from core.grid import Grid, GridPosition
from solvers.base_solver import BaseSolver


class Heuristic(Enum):
    """Available heuristic functions for A*."""

    MANHATTAN = auto()
    EUCLIDEAN = auto()
    CHEBYSHEV = auto()


def manhattan_distance(a: GridPosition, b: GridPosition) -> float:
    """Manhattan (L1) distance — sum of absolute differences."""
    return abs(a.col - b.col) + abs(a.row - b.row)


def euclidean_distance(a: GridPosition, b: GridPosition) -> float:
    """Euclidean (L2) distance — straight-line distance."""
    return math.sqrt((a.col - b.col) ** 2 + (a.row - b.row) ** 2)


def chebyshev_distance(a: GridPosition, b: GridPosition) -> float:
    """Chebyshev (L-inf) distance — max of absolute differences."""
    return max(abs(a.col - b.col), abs(a.row - b.row))


_HEURISTIC_FUNCS: dict[Heuristic, Callable[[GridPosition, GridPosition], float]] = {
    Heuristic.MANHATTAN: manhattan_distance,
    Heuristic.EUCLIDEAN: euclidean_distance,
    Heuristic.CHEBYSHEV: chebyshev_distance,
}


class AStarSolver(BaseSolver):
    """Solves a maze using A* search.

    Uses f(n) = g(n) + h(n) where g is path cost and h is heuristic.
    With admissible heuristic, finds the optimal path.

    Args:
        grid: The maze grid.
        start: Starting position.
        goal: Goal position.
        heuristic: Which heuristic to use (default: Manhattan).
    """

    name = "A*"

    def __init__(
        self,
        grid: Grid,
        start: GridPosition,
        goal: GridPosition,
        heuristic: Heuristic = Heuristic.MANHATTAN,
    ) -> None:
        super().__init__(grid, start, goal)
        self._heuristic_fn = _HEURISTIC_FUNCS[heuristic]

    def solve(self) -> Iterator[AlgorithmEvent]:
        """Solve using A*."""
        grid = self._grid
        grid.reset_visited()

        h = self._heuristic_fn
        counter = 0
        start_h = h(self._start, self._goal)
        pq: list[tuple[float, int, GridPosition]] = [(start_h, counter, self._start)]
        g_score: dict[GridPosition, float] = {self._start: 0.0}
        parent: dict[GridPosition, GridPosition | None] = {self._start: None}

        while pq:
            f_val, _, current = heapq.heappop(pq)
            current_g = g_score.get(current, float("inf"))

            # Skip stale entries
            if current_g + h(current, self._goal) < f_val - 1e-9:
                continue

            grid.get_cell(current).visited = True
            yield self._emit_visit(current, parent.get(current))

            if current == self._goal:
                path = self._reconstruct_path(parent, current)
                total_cost = sum(grid.get_cell(p).cost for p in path)
                yield self._emit_path_found(path, total_cost)
                return

            for neighbor in grid.get_passable_neighbors(current):
                new_g = current_g + grid.get_cell(neighbor).cost
                if new_g < g_score.get(neighbor, float("inf")):
                    g_score[neighbor] = new_g
                    f = new_g + h(neighbor, self._goal)
                    parent[neighbor] = current
                    counter += 1
                    heapq.heappush(pq, (f, counter, neighbor))

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
