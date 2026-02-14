"""Maze renderer — draws grid, walls, and paths on a Tk Canvas.

Decoupled from Cell: Cell is pure data, the renderer reads cell state and draws.
The renderer consumes AlgorithmEvents without knowing which algorithm is running.
"""

from __future__ import annotations

from tkinter import Canvas

from config import BG_COLOR, WALL_COLOR, WALL_WIDTH, DEFAULT_FORWARD_COLOR, DEFAULT_BACKTRACK_COLOR
from core.cell import Cell
from core.events import AlgorithmEvent, EventType
from core.grid import Grid, GridPosition


class MazeRenderer:
    """Renders a maze grid onto a Tk Canvas.

    Args:
        canvas: The Tkinter Canvas widget to draw on.
        grid: The Grid data structure.
        origin_x: Pixel x-offset for the maze origin.
        origin_y: Pixel y-offset for the maze origin.
        cell_width: Pixel width of each cell.
        cell_height: Pixel height of each cell.
    """

    def __init__(
        self,
        canvas: Canvas,
        grid: Grid,
        origin_x: int,
        origin_y: int,
        cell_width: int,
        cell_height: int,
    ) -> None:
        self._canvas = canvas
        self._grid = grid
        self._origin_x = origin_x
        self._origin_y = origin_y
        self._cell_width = cell_width
        self._cell_height = cell_height
        self._forward_color = DEFAULT_FORWARD_COLOR
        self._backtrack_color = DEFAULT_BACKTRACK_COLOR
        self._path_items: list[int] = []

    @property
    def cell_size_x(self) -> int:
        return self._cell_width

    @property
    def cell_size_y(self) -> int:
        return self._cell_height

    def set_colors(self, forward: str, backtrack: str) -> None:
        """Set the forward/backtrack colors for this renderer."""
        self._forward_color = forward
        self._backtrack_color = backtrack

    def get_cell_bounds(self, pos: GridPosition) -> tuple[int, int, int, int]:
        """Get pixel bounds (x1, y1, x2, y2) for a cell position."""
        x1 = self._origin_x + pos.col * self._cell_width
        y1 = self._origin_y + pos.row * self._cell_height
        x2 = x1 + self._cell_width
        y2 = y1 + self._cell_height
        return x1, y1, x2, y2

    def get_cell_center(self, pos: GridPosition) -> tuple[float, float]:
        """Get pixel center coordinates of a cell."""
        x1, y1, x2, y2 = self.get_cell_bounds(pos)
        return (x1 + x2) / 2, (y1 + y2) / 2

    def draw_cell(self, pos: GridPosition) -> None:
        """Draw/redraw a single cell's walls."""
        cell = self._grid.get_cell(pos)
        x1, y1, x2, y2 = self.get_cell_bounds(pos)

        # Left wall
        color = WALL_COLOR if cell.has_left_wall else BG_COLOR
        self._canvas.create_line(x1, y1, x1, y2, fill=color, width=WALL_WIDTH)

        # Top wall
        color = WALL_COLOR if cell.has_top_wall else BG_COLOR
        self._canvas.create_line(x1, y1, x2, y1, fill=color, width=WALL_WIDTH)

        # Right wall
        color = WALL_COLOR if cell.has_right_wall else BG_COLOR
        self._canvas.create_line(x2, y1, x2, y2, fill=color, width=WALL_WIDTH)

        # Bottom wall
        color = WALL_COLOR if cell.has_bottom_wall else BG_COLOR
        self._canvas.create_line(x1, y2, x2, y2, fill=color, width=WALL_WIDTH)

    def draw_entire_maze(self) -> None:
        """Draw all cells in the grid."""
        for pos in self._grid.all_positions():
            self.draw_cell(pos)

    def draw_move(self, from_pos: GridPosition, to_pos: GridPosition, color: str) -> None:
        """Draw a line between the centers of two cells."""
        cx1, cy1 = self.get_cell_center(from_pos)
        cx2, cy2 = self.get_cell_center(to_pos)
        item_id = self._canvas.create_line(cx1, cy1, cx2, cy2, fill=color, width=2)
        self._path_items.append(item_id)

    def clear_paths(self) -> None:
        """Remove all path lines from the canvas."""
        for item_id in self._path_items:
            self._canvas.delete(item_id)
        self._path_items.clear()

    def clear_all(self) -> None:
        """Clear everything from the canvas."""
        self._canvas.delete("all")
        self._path_items.clear()

    def handle_event(self, event: AlgorithmEvent) -> None:
        """Handle an algorithm event — the universal rendering callback.

        The renderer does not know which algorithm is running; it just
        renders state transitions.
        """
        if event.event_type == EventType.VISIT:
            if event.secondary_position is not None:
                self.draw_move(event.secondary_position, event.position, self._forward_color)

        elif event.event_type == EventType.BACKTRACK:
            if event.secondary_position is not None:
                self.draw_move(event.position, event.secondary_position, self._backtrack_color)

        elif event.event_type == EventType.WALL_REMOVED:
            # Redraw both cells to show the removed wall
            self.draw_cell(event.position)
            if event.secondary_position is not None:
                self.draw_cell(event.secondary_position)

        elif event.event_type == EventType.CELL_UPDATED:
            self.draw_cell(event.position)

        elif event.event_type == EventType.PATH_FOUND:
            # Draw final path in a bright color
            if event.path and len(event.path) >= 2:
                for i in range(len(event.path) - 1):
                    self.draw_move(event.path[i], event.path[i + 1], self._forward_color)

    def handle_generator_event(self, event: AlgorithmEvent) -> None:
        """Handle a generator event — draws walls being removed during generation."""
        if event.event_type == EventType.WALL_REMOVED:
            self.draw_cell(event.position)
            if event.secondary_position is not None:
                self.draw_cell(event.secondary_position)
        elif event.event_type == EventType.CELL_UPDATED:
            self.draw_cell(event.position)
