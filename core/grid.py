"""Grid data structure — owns cells and provides adjacency queries.

Uses col-major indexing: cells[col][row]. This is unconventional but preserved
from the original codebase for consistency. Column index (i) maps to x-axis,
row index (j) maps to y-axis.

  cells[0][0] = top-left
  cells[num_cols-1][num_rows-1] = bottom-right
  i-1 = left, i+1 = right, j-1 = up, j+1 = down
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator

from core.cell import Cell


@dataclass(frozen=True)
class GridPosition:
    """Immutable (col, row) position in the grid."""

    col: int
    row: int

    def __iter__(self) -> Iterator[int]:
        yield self.col
        yield self.row


# Direction deltas: (dcol, drow, wall_from, wall_to)
_DIRECTIONS: list[tuple[int, int, str, str]] = [
    (-1, 0, "has_left_wall", "has_right_wall"),    # left
    (1, 0, "has_right_wall", "has_left_wall"),      # right
    (0, -1, "has_top_wall", "has_bottom_wall"),     # up
    (0, 1, "has_bottom_wall", "has_top_wall"),      # down
]


class Grid:
    """2D grid of cells with adjacency and wall queries.

    Args:
        num_cols: Number of columns (x-axis extent).
        num_rows: Number of rows (y-axis extent).
    """

    def __init__(self, num_cols: int, num_rows: int) -> None:
        if num_cols < 1 or num_rows < 1:
            raise ValueError(f"Grid dimensions must be >= 1, got {num_cols}x{num_rows}")

        self.num_cols = num_cols
        self.num_rows = num_rows

        # Col-major: _cells[col][row]
        self._cells: list[list[Cell]] = [
            [Cell() for _ in range(num_rows)] for _ in range(num_cols)
        ]

    def get_cell(self, pos: GridPosition) -> Cell:
        """Get cell at position. Raises IndexError if out of bounds."""
        if not self.in_bounds(pos):
            raise IndexError(f"Position {pos} out of bounds for {self.num_cols}x{self.num_rows} grid")
        return self._cells[pos.col][pos.row]

    def in_bounds(self, pos: GridPosition) -> bool:
        """Check if position is within grid bounds."""
        return 0 <= pos.col < self.num_cols and 0 <= pos.row < self.num_rows

    def get_neighbors(self, pos: GridPosition) -> list[tuple[GridPosition, str]]:
        """Get all in-bounds neighbors with their direction names.

        Returns list of (neighbor_pos, direction) where direction is
        'left', 'right', 'up', or 'down'.
        """
        direction_names = ["left", "right", "up", "down"]
        result: list[tuple[GridPosition, str]] = []
        for (dc, dr, _, _), name in zip(_DIRECTIONS, direction_names):
            npos = GridPosition(pos.col + dc, pos.row + dr)
            if self.in_bounds(npos):
                result.append((npos, name))
        return result

    def get_passable_neighbors(self, pos: GridPosition) -> list[GridPosition]:
        """Get neighbors reachable from pos (no wall between them)."""
        result: list[GridPosition] = []
        for dc, dr, wall_attr, _ in _DIRECTIONS:
            npos = GridPosition(pos.col + dc, pos.row + dr)
            if self.in_bounds(npos) and not getattr(self.get_cell(pos), wall_attr):
                result.append(npos)
        return result

    def get_unvisited_neighbors(self, pos: GridPosition) -> list[GridPosition]:
        """Get in-bounds neighbors that have not been visited (ignores walls)."""
        result: list[GridPosition] = []
        for dc, dr, _, _ in _DIRECTIONS:
            npos = GridPosition(pos.col + dc, pos.row + dr)
            if self.in_bounds(npos) and not self.get_cell(npos).visited:
                result.append(npos)
        return result

    def get_passable_unvisited_neighbors(self, pos: GridPosition) -> list[GridPosition]:
        """Get neighbors reachable (no wall) and not yet visited."""
        result: list[GridPosition] = []
        for dc, dr, wall_attr, _ in _DIRECTIONS:
            npos = GridPosition(pos.col + dc, pos.row + dr)
            if (
                self.in_bounds(npos)
                and not getattr(self.get_cell(pos), wall_attr)
                and not self.get_cell(npos).visited
            ):
                result.append(npos)
        return result

    def has_wall_between(self, a: GridPosition, b: GridPosition) -> bool:
        """Check if there is a wall between two adjacent cells."""
        dc = b.col - a.col
        dr = b.row - a.row
        for ddc, ddr, wall_attr, _ in _DIRECTIONS:
            if dc == ddc and dr == ddr:
                return getattr(self.get_cell(a), wall_attr)
        raise ValueError(f"Positions {a} and {b} are not adjacent")

    def remove_wall_between(self, a: GridPosition, b: GridPosition) -> None:
        """Remove the wall between two adjacent cells."""
        dc = b.col - a.col
        dr = b.row - a.row
        for ddc, ddr, wall_from, wall_to in _DIRECTIONS:
            if dc == ddc and dr == ddr:
                setattr(self.get_cell(a), wall_from, False)
                setattr(self.get_cell(b), wall_to, False)
                return
        raise ValueError(f"Positions {a} and {b} are not adjacent")

    def add_wall_between(self, a: GridPosition, b: GridPosition) -> None:
        """Add a wall between two adjacent cells."""
        dc = b.col - a.col
        dr = b.row - a.row
        for ddc, ddr, wall_from, wall_to in _DIRECTIONS:
            if dc == ddc and dr == ddr:
                setattr(self.get_cell(a), wall_from, True)
                setattr(self.get_cell(b), wall_to, True)
                return
        raise ValueError(f"Positions {a} and {b} are not adjacent")

    def reset_visited(self) -> None:
        """Reset all cells' visited flags to False."""
        for col in self._cells:
            for cell in col:
                cell.reset()

    def all_positions(self) -> Iterator[GridPosition]:
        """Iterate over all positions in the grid (col-major order)."""
        for col in range(self.num_cols):
            for row in range(self.num_rows):
                yield GridPosition(col, row)

    @property
    def start(self) -> GridPosition:
        """Default start position: top-left."""
        return GridPosition(0, 0)

    @property
    def end(self) -> GridPosition:
        """Default end position: bottom-right."""
        return GridPosition(self.num_cols - 1, self.num_rows - 1)
