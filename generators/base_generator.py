"""Abstract base class for maze generators."""

from __future__ import annotations

import random
from abc import ABC, abstractmethod
from typing import Iterator

from core.events import AlgorithmEvent
from core.grid import Grid


class BaseGenerator(ABC):
    """ABC for maze generation algorithms.

    Subclasses implement generate() as a generator that yields AlgorithmEvent
    objects for each step. The visualization layer consumes these events.

    Args:
        grid: The grid to generate a maze on. All walls start intact.
        seed: Optional random seed for deterministic generation.
    """

    name: str = "Base"

    def __init__(self, grid: Grid, seed: int | None = None) -> None:
        self._grid = grid
        self._seed = seed
        self._rng = random.Random(seed)

    @abstractmethod
    def generate(self) -> Iterator[AlgorithmEvent]:
        """Generate the maze, yielding events for each step.

        Must break entrance (top of start cell) and exit (bottom of end cell).
        """

    def generate_complete(self) -> None:
        """Run generation to completion without yielding events."""
        for _ in self.generate():
            pass

    def _break_entrance_and_exit(self) -> Iterator[AlgorithmEvent]:
        """Remove entrance wall (top of [0][0]) and exit wall (bottom of last cell)."""
        from core.events import EventType

        start_cell = self._grid.get_cell(self._grid.start)
        start_cell.has_top_wall = False
        yield AlgorithmEvent(
            event_type=EventType.CELL_UPDATED,
            position=self._grid.start,
        )

        end_cell = self._grid.get_cell(self._grid.end)
        end_cell.has_bottom_wall = False
        yield AlgorithmEvent(
            event_type=EventType.CELL_UPDATED,
            position=self._grid.end,
        )
