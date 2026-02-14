"""Pure data representation of a maze cell. No rendering logic."""

from __future__ import annotations


class Cell:
    """A single cell in the maze grid.

    Stores wall state, traversal metadata, and optional cost for weighted graphs.
    Drawing is handled entirely by the renderer — Cell is pure data.
    """

    __slots__ = (
        "has_left_wall",
        "has_right_wall",
        "has_top_wall",
        "has_bottom_wall",
        "visited",
        "cost",
    )

    def __init__(self) -> None:
        self.has_left_wall: bool = True
        self.has_right_wall: bool = True
        self.has_top_wall: bool = True
        self.has_bottom_wall: bool = True
        self.visited: bool = False
        self.cost: float = 1.0

    def reset(self) -> None:
        """Reset traversal state while preserving walls and cost."""
        self.visited = False

    def __repr__(self) -> str:
        walls = ""
        if self.has_left_wall:
            walls += "L"
        if self.has_right_wall:
            walls += "R"
        if self.has_top_wall:
            walls += "T"
        if self.has_bottom_wall:
            walls += "B"
        return f"Cell(walls={walls}, cost={self.cost})"
