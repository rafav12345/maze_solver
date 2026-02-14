"""Tests for maze generators."""

import pytest
from core.grid import Grid, GridPosition
from core.events import EventType
from generators.dfs_generator import DFSGenerator


class TestDFSGenerator:
    def test_generates_connected_maze(self) -> None:
        """Every cell should be reachable from start via passable neighbors."""
        grid = Grid(10, 10)
        gen = DFSGenerator(grid, seed=42)
        gen.generate_complete()

        # BFS from start to verify connectivity
        visited = set()
        queue = [grid.start]
        visited.add(grid.start)
        while queue:
            pos = queue.pop(0)
            for neighbor in grid.get_passable_neighbors(pos):
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append(neighbor)

        assert len(visited) == grid.num_cols * grid.num_rows

    def test_entrance_and_exit(self) -> None:
        """Entrance (top of [0,0]) and exit (bottom of last cell) should be open."""
        grid = Grid(12, 10)
        gen = DFSGenerator(grid, seed=0)
        gen.generate_complete()

        assert grid.get_cell(GridPosition(0, 0)).has_top_wall is False
        assert grid.get_cell(GridPosition(11, 9)).has_bottom_wall is False

    def test_seed_determinism(self) -> None:
        """Same seed produces identical maze."""
        grid1 = Grid(10, 10)
        DFSGenerator(grid1, seed=123).generate_complete()

        grid2 = Grid(10, 10)
        DFSGenerator(grid2, seed=123).generate_complete()

        for pos in grid1.all_positions():
            c1 = grid1.get_cell(pos)
            c2 = grid2.get_cell(pos)
            assert c1.has_left_wall == c2.has_left_wall
            assert c1.has_right_wall == c2.has_right_wall
            assert c1.has_top_wall == c2.has_top_wall
            assert c1.has_bottom_wall == c2.has_bottom_wall

    def test_different_seeds_different_mazes(self) -> None:
        """Different seeds produce different mazes (probabilistically)."""
        grid1 = Grid(10, 10)
        DFSGenerator(grid1, seed=1).generate_complete()

        grid2 = Grid(10, 10)
        DFSGenerator(grid2, seed=2).generate_complete()

        # At least one cell should differ
        any_different = False
        for pos in grid1.all_positions():
            c1 = grid1.get_cell(pos)
            c2 = grid2.get_cell(pos)
            if (c1.has_left_wall != c2.has_left_wall or
                c1.has_right_wall != c2.has_right_wall or
                c1.has_top_wall != c2.has_top_wall or
                c1.has_bottom_wall != c2.has_bottom_wall):
                any_different = True
                break
        assert any_different

    def test_small_maze(self) -> None:
        """2x2 maze works correctly."""
        grid = Grid(2, 2)
        gen = DFSGenerator(grid, seed=0)
        gen.generate_complete()

        # Should be connected
        visited = set()
        queue = [grid.start]
        visited.add(grid.start)
        while queue:
            pos = queue.pop(0)
            for neighbor in grid.get_passable_neighbors(pos):
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append(neighbor)
        assert len(visited) == 4

    def test_1x1_maze(self) -> None:
        """1x1 grid should have entrance and exit open."""
        grid = Grid(1, 1)
        gen = DFSGenerator(grid, seed=0)
        gen.generate_complete()

        cell = grid.get_cell(GridPosition(0, 0))
        assert cell.has_top_wall is False
        assert cell.has_bottom_wall is False

    def test_yields_events(self) -> None:
        """Generator should yield events during generation."""
        grid = Grid(5, 5)
        gen = DFSGenerator(grid, seed=42)
        events = list(gen.generate())
        assert len(events) > 0
        # Should have WALL_REMOVED and COMPLETE events
        types = {e.event_type for e in events}
        assert EventType.WALL_REMOVED in types
        assert EventType.COMPLETE in types

    def test_large_maze(self) -> None:
        """50x50 maze generates without error."""
        grid = Grid(50, 50)
        gen = DFSGenerator(grid, seed=99)
        gen.generate_complete()

        # Spot check: entrance and exit open
        assert grid.get_cell(GridPosition(0, 0)).has_top_wall is False
        assert grid.get_cell(GridPosition(49, 49)).has_bottom_wall is False

    def test_visited_reset_after_generation(self) -> None:
        """All cells should have visited=False after generation completes."""
        grid = Grid(10, 10)
        DFSGenerator(grid, seed=0).generate_complete()
        for pos in grid.all_positions():
            assert grid.get_cell(pos).visited is False
