"""Abstract base class for maze solvers."""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Iterator

from core.events import AlgorithmEvent, EventType
from core.grid import Grid, GridPosition


@dataclass
class SolverStats:
    """Statistics tracked during solving."""

    cells_explored: int = 0
    cells_backtracked: int = 0
    path_length: int = 0
    total_steps: int = 0
    path_cost: float = 0.0
    time_elapsed: float = 0.0


class BaseSolver(ABC):
    """ABC for maze solving algorithms.

    Subclasses implement solve() as a generator that yields AlgorithmEvent
    objects for each step. The visualization layer consumes these events.

    Args:
        grid: The maze grid to solve.
        start: Starting position.
        goal: Goal position.
    """

    name: str = "Base"

    def __init__(self, grid: Grid, start: GridPosition, goal: GridPosition) -> None:
        self._grid = grid
        self._start = start
        self._goal = goal
        self._stats = SolverStats()
        self._path: list[GridPosition] = []
        self._solved = False

    @abstractmethod
    def solve(self) -> Iterator[AlgorithmEvent]:
        """Solve the maze, yielding events for each step.

        Must yield PATH_FOUND with the path on success,
        or PATH_NOT_FOUND on failure.
        """

    def solve_complete(self) -> bool:
        """Run solver to completion. Returns True if path found."""
        start_time = time.perf_counter()
        for _ in self.solve():
            pass
        self._stats.time_elapsed = time.perf_counter() - start_time
        return self._solved

    def get_path(self) -> list[GridPosition]:
        """Get the solution path (empty if unsolved)."""
        return list(self._path)

    def get_stats(self) -> SolverStats:
        """Get current solver statistics."""
        return self._stats

    def _emit_visit(self, pos: GridPosition, from_pos: GridPosition | None = None) -> AlgorithmEvent:
        """Helper: emit a VISIT event and update stats."""
        self._stats.cells_explored += 1
        self._stats.total_steps += 1
        return AlgorithmEvent(
            event_type=EventType.VISIT,
            position=pos,
            secondary_position=from_pos,
        )

    def _emit_backtrack(self, pos: GridPosition, from_pos: GridPosition | None = None) -> AlgorithmEvent:
        """Helper: emit a BACKTRACK event and update stats."""
        self._stats.cells_backtracked += 1
        self._stats.total_steps += 1
        return AlgorithmEvent(
            event_type=EventType.BACKTRACK,
            position=pos,
            secondary_position=from_pos,
        )

    def _emit_path_found(self, path: list[GridPosition], cost: float) -> AlgorithmEvent:
        """Helper: emit PATH_FOUND and finalize stats."""
        self._path = list(path)
        self._solved = True
        self._stats.path_length = len(path)
        self._stats.path_cost = cost
        return AlgorithmEvent(
            event_type=EventType.PATH_FOUND,
            position=self._goal,
            path=list(path),
            metadata={"cost": cost},
        )

    def _emit_path_not_found(self) -> AlgorithmEvent:
        """Helper: emit PATH_NOT_FOUND."""
        self._solved = False
        return AlgorithmEvent(
            event_type=EventType.PATH_NOT_FOUND,
            position=self._start,
        )
