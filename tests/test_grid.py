"""Tests for Grid class — tests through public interface only."""

import pytest
from core.cell import Cell
from core.grid import Grid, GridPosition


class TestGridCreation:
    def test_grid_dimensions(self) -> None:
        g = Grid(12, 10)
        assert g.num_cols == 12
        assert g.num_rows == 10

    def test_grid_small(self) -> None:
        g = Grid(2, 2)
        assert g.num_cols == 2
        assert g.num_rows == 2

    def test_grid_rectangular(self) -> None:
        g = Grid(20, 5)
        assert g.num_cols == 20
        assert g.num_rows == 5

    def test_grid_invalid_dimensions(self) -> None:
        with pytest.raises(ValueError):
            Grid(0, 5)
        with pytest.raises(ValueError):
            Grid(5, 0)

    def test_grid_1x1(self) -> None:
        g = Grid(1, 1)
        assert g.num_cols == 1
        assert g.num_rows == 1


class TestGridCellAccess:
    def test_get_cell(self) -> None:
        g = Grid(5, 5)
        cell = g.get_cell(GridPosition(0, 0))
        assert isinstance(cell, Cell)
        assert cell.has_left_wall is True

    def test_get_cell_out_of_bounds(self) -> None:
        g = Grid(5, 5)
        with pytest.raises(IndexError):
            g.get_cell(GridPosition(5, 0))
        with pytest.raises(IndexError):
            g.get_cell(GridPosition(0, 5))
        with pytest.raises(IndexError):
            g.get_cell(GridPosition(-1, 0))

    def test_in_bounds(self) -> None:
        g = Grid(5, 5)
        assert g.in_bounds(GridPosition(0, 0)) is True
        assert g.in_bounds(GridPosition(4, 4)) is True
        assert g.in_bounds(GridPosition(5, 0)) is False
        assert g.in_bounds(GridPosition(-1, 0)) is False

    def test_start_and_end(self) -> None:
        g = Grid(10, 8)
        assert g.start == GridPosition(0, 0)
        assert g.end == GridPosition(9, 7)


class TestGridNeighbors:
    def test_corner_neighbors(self) -> None:
        g = Grid(5, 5)
        neighbors = g.get_neighbors(GridPosition(0, 0))
        positions = [n for n, _ in neighbors]
        assert GridPosition(1, 0) in positions  # right
        assert GridPosition(0, 1) in positions  # down
        assert len(neighbors) == 2

    def test_center_neighbors(self) -> None:
        g = Grid(5, 5)
        neighbors = g.get_neighbors(GridPosition(2, 2))
        assert len(neighbors) == 4

    def test_passable_neighbors_all_walls(self) -> None:
        g = Grid(5, 5)
        # All walls intact — no passable neighbors
        passable = g.get_passable_neighbors(GridPosition(2, 2))
        assert len(passable) == 0

    def test_passable_neighbors_after_wall_removal(self) -> None:
        g = Grid(5, 5)
        a = GridPosition(2, 2)
        b = GridPosition(3, 2)
        g.remove_wall_between(a, b)
        passable = g.get_passable_neighbors(a)
        assert b in passable
        assert len(passable) == 1


class TestGridWalls:
    def test_remove_wall_between(self) -> None:
        g = Grid(5, 5)
        a = GridPosition(1, 1)
        b = GridPosition(2, 1)  # right neighbor
        assert g.has_wall_between(a, b) is True
        g.remove_wall_between(a, b)
        assert g.has_wall_between(a, b) is False
        # Symmetric
        assert g.has_wall_between(b, a) is False

    def test_remove_wall_vertical(self) -> None:
        g = Grid(5, 5)
        a = GridPosition(1, 1)
        b = GridPosition(1, 2)  # down neighbor
        g.remove_wall_between(a, b)
        assert g.get_cell(a).has_bottom_wall is False
        assert g.get_cell(b).has_top_wall is False

    def test_add_wall_between(self) -> None:
        g = Grid(5, 5)
        a = GridPosition(1, 1)
        b = GridPosition(2, 1)
        g.remove_wall_between(a, b)
        assert g.has_wall_between(a, b) is False
        g.add_wall_between(a, b)
        assert g.has_wall_between(a, b) is True

    def test_wall_between_non_adjacent(self) -> None:
        g = Grid(5, 5)
        with pytest.raises(ValueError):
            g.has_wall_between(GridPosition(0, 0), GridPosition(2, 0))


class TestGridTraversal:
    def test_reset_visited(self) -> None:
        g = Grid(3, 3)
        g.get_cell(GridPosition(0, 0)).visited = True
        g.get_cell(GridPosition(1, 1)).visited = True
        g.reset_visited()
        assert g.get_cell(GridPosition(0, 0)).visited is False
        assert g.get_cell(GridPosition(1, 1)).visited is False

    def test_all_positions(self) -> None:
        g = Grid(3, 4)
        positions = list(g.all_positions())
        assert len(positions) == 12
        assert positions[0] == GridPosition(0, 0)
        assert positions[-1] == GridPosition(2, 3)

    def test_unvisited_neighbors(self) -> None:
        g = Grid(5, 5)
        pos = GridPosition(2, 2)
        unvisited = g.get_unvisited_neighbors(pos)
        assert len(unvisited) == 4
        # Visit one neighbor
        g.get_cell(GridPosition(3, 2)).visited = True
        unvisited = g.get_unvisited_neighbors(pos)
        assert len(unvisited) == 3
        assert GridPosition(3, 2) not in unvisited
