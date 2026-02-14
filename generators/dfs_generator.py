"""DFS recursive backtracker maze generator.

Refactored from the original Maze.__break_walls_r. Uses an explicit stack
instead of recursion to avoid stack overflow on large grids.
"""

from __future__ import annotations

from typing import Iterator

from core.events import AlgorithmEvent, EventType
from core.grid import Grid, GridPosition
from generators.base_generator import BaseGenerator


class DFSGenerator(BaseGenerator):
    """Generates a perfect maze using randomized depth-first search (recursive backtracker).

    Produces mazes with long, winding corridors and relatively few dead ends.
    """

    name = "DFS (Recursive Backtracker)"

    def __init__(self, grid: Grid, seed: int | None = None) -> None:
        super().__init__(grid, seed)

    def generate(self) -> Iterator[AlgorithmEvent]:
        """Generate maze using iterative DFS with explicit stack."""
        grid = self._grid

        # Start at (0, 0)
        start = GridPosition(0, 0)
        grid.get_cell(start).visited = True
        stack: list[GridPosition] = [start]

        while stack:
            current = stack[-1]

            # Find unvisited neighbors
            unvisited = grid.get_unvisited_neighbors(current)

            if not unvisited:
                # Backtrack
                stack.pop()
                yield AlgorithmEvent(
                    event_type=EventType.CELL_UPDATED,
                    position=current,
                )
                continue

            # Pick random unvisited neighbor
            next_pos = self._rng.choice(unvisited)

            # Remove wall between current and next
            grid.remove_wall_between(current, next_pos)
            grid.get_cell(next_pos).visited = True
            stack.append(next_pos)

            yield AlgorithmEvent(
                event_type=EventType.WALL_REMOVED,
                position=current,
                secondary_position=next_pos,
            )

        # Reset visited flags — generation is done
        grid.reset_visited()

        # Break entrance and exit
        yield from self._break_entrance_and_exit()

        yield AlgorithmEvent(
            event_type=EventType.COMPLETE,
            position=grid.start,
        )
