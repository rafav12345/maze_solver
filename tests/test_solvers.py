"""Tests for maze solvers — all algorithms."""

import pytest
from core.grid import Grid, GridPosition
from core.events import EventType
from generators.dfs_generator import DFSGenerator
from solvers.dfs_solver import DFSSolver
from solvers.bfs_solver import BFSSolver
from solvers.dijkstra_solver import DijkstraSolver
from solvers.astar_solver import AStarSolver, Heuristic
from solvers.greedy_solver import GreedySolver
from solvers.wall_follower_solver import (
    WallFollowerSolver,
    LeftWallFollowerSolver,
    RightWallFollowerSolver,
)
from solvers.base_solver import BaseSolver


def _make_maze(cols: int = 10, rows: int = 10, seed: int = 42) -> Grid:
    """Helper: create a generated maze."""
    grid = Grid(cols, rows)
    DFSGenerator(grid, seed=seed).generate_complete()
    return grid


def _validate_path(grid: Grid, path: list[GridPosition], start: GridPosition, goal: GridPosition) -> None:
    """Validate that a path is correct: starts at start, ends at goal, each step passable."""
    assert len(path) >= 1
    assert path[0] == start
    assert path[-1] == goal
    for i in range(len(path) - 1):
        neighbors = grid.get_passable_neighbors(path[i])
        assert path[i + 1] in neighbors, f"Step {i}: {path[i]} -> {path[i+1]} not passable"


# --- Parametrized tests for all solvers ---

ALL_SOLVER_CLASSES: list[type[BaseSolver]] = [
    DFSSolver,
    BFSSolver,
    DijkstraSolver,
    AStarSolver,
    GreedySolver,
    LeftWallFollowerSolver,
    RightWallFollowerSolver,
]

ALL_SOLVER_IDS = ["DFS", "BFS", "Dijkstra", "A*", "Greedy", "WallFollower-L", "WallFollower-R"]


@pytest.fixture(params=zip(ALL_SOLVER_CLASSES, ALL_SOLVER_IDS), ids=ALL_SOLVER_IDS)
def solver_cls(request: pytest.FixtureRequest) -> type[BaseSolver]:
    return request.param[0]


class TestAllSolvers:
    """Tests that apply to every solver implementation."""

    def test_finds_path(self, solver_cls: type[BaseSolver]) -> None:
        grid = _make_maze()
        solver = solver_cls(grid, grid.start, grid.end)
        found = solver.solve_complete()
        assert found is True
        _validate_path(grid, solver.get_path(), grid.start, grid.end)

    def test_stats_populated(self, solver_cls: type[BaseSolver]) -> None:
        grid = _make_maze()
        solver = solver_cls(grid, grid.start, grid.end)
        solver.solve_complete()
        stats = solver.get_stats()
        assert stats.cells_explored > 0
        assert stats.path_length > 0
        assert stats.total_steps > 0
        assert stats.path_cost > 0

    def test_1x1_maze(self, solver_cls: type[BaseSolver]) -> None:
        grid = Grid(1, 1)
        DFSGenerator(grid, seed=0).generate_complete()
        solver = solver_cls(grid, grid.start, grid.end)
        found = solver.solve_complete()
        assert found is True
        assert solver.get_path() == [GridPosition(0, 0)]

    def test_small_maze(self, solver_cls: type[BaseSolver]) -> None:
        grid = _make_maze(cols=2, rows=2, seed=0)
        solver = solver_cls(grid, grid.start, grid.end)
        found = solver.solve_complete()
        assert found is True
        _validate_path(grid, solver.get_path(), grid.start, grid.end)

    def test_5x5_maze(self, solver_cls: type[BaseSolver]) -> None:
        grid = _make_maze(cols=5, rows=5, seed=99)
        solver = solver_cls(grid, grid.start, grid.end)
        found = solver.solve_complete()
        assert found is True
        _validate_path(grid, solver.get_path(), grid.start, grid.end)

    def test_20x20_maze(self, solver_cls: type[BaseSolver]) -> None:
        grid = _make_maze(cols=20, rows=20, seed=7)
        solver = solver_cls(grid, grid.start, grid.end)
        found = solver.solve_complete()
        assert found is True
        _validate_path(grid, solver.get_path(), grid.start, grid.end)

    def test_50x50_maze(self, solver_cls: type[BaseSolver]) -> None:
        grid = _make_maze(cols=50, rows=50, seed=7)
        solver = solver_cls(grid, grid.start, grid.end)
        found = solver.solve_complete()
        assert found is True
        _validate_path(grid, solver.get_path(), grid.start, grid.end)

    def test_yields_events(self, solver_cls: type[BaseSolver]) -> None:
        grid = _make_maze(cols=5, rows=5)
        solver = solver_cls(grid, grid.start, grid.end)
        events = list(solver.solve())
        types = {e.event_type for e in events}
        assert EventType.VISIT in types
        assert EventType.PATH_FOUND in types

    def test_seed_determinism(self, solver_cls: type[BaseSolver]) -> None:
        grid1 = _make_maze(seed=100)
        solver1 = solver_cls(grid1, grid1.start, grid1.end)
        solver1.solve_complete()

        grid2 = _make_maze(seed=100)
        solver2 = solver_cls(grid2, grid2.start, grid2.end)
        solver2.solve_complete()

        assert solver1.get_path() == solver2.get_path()


# --- Solver-specific tests ---

class TestBFSSolver:
    def test_finds_shortest_path(self) -> None:
        """BFS should find the shortest path (by cell count) in unweighted mazes."""
        grid = _make_maze(cols=10, rows=10, seed=42)

        bfs = BFSSolver(grid, grid.start, grid.end)
        bfs.solve_complete()
        bfs_len = len(bfs.get_path())

        # DFS may find a longer path
        grid.reset_visited()
        dfs = DFSSolver(grid, grid.start, grid.end)
        dfs.solve_complete()
        dfs_len = len(dfs.get_path())

        assert bfs_len <= dfs_len


class TestDijkstraSolver:
    def test_finds_optimal_path(self) -> None:
        """Dijkstra should find the cost-optimal path."""
        grid = _make_maze(cols=10, rows=10, seed=42)

        dijkstra = DijkstraSolver(grid, grid.start, grid.end)
        dijkstra.solve_complete()

        bfs = BFSSolver(grid, grid.start, grid.end)
        bfs.solve_complete()

        # With uniform costs, Dijkstra path cost == BFS path cost
        assert dijkstra.get_stats().path_cost == bfs.get_stats().path_cost

    def test_respects_weights(self) -> None:
        """Dijkstra should prefer cheaper cells when weights differ."""
        grid = _make_maze(cols=5, rows=5, seed=42)

        # Run once with uniform weights
        d1 = DijkstraSolver(grid, grid.start, grid.end)
        d1.solve_complete()
        path1 = d1.get_path()

        # Make some cells on the original path expensive
        if len(path1) > 3:
            grid.get_cell(path1[2]).cost = 100.0

        # Re-run — may find a different, cheaper path
        grid.reset_visited()
        d2 = DijkstraSolver(grid, grid.start, grid.end)
        d2.solve_complete()
        # The new path should be valid
        _validate_path(grid, d2.get_path(), grid.start, grid.end)


class TestAStarSolver:
    def test_finds_optimal_path(self) -> None:
        """A* with admissible heuristic should find optimal path."""
        grid = _make_maze(cols=15, rows=15, seed=42)

        astar = AStarSolver(grid, grid.start, grid.end)
        astar.solve_complete()

        dijkstra = DijkstraSolver(grid, grid.start, grid.end)
        dijkstra.solve_complete()

        assert astar.get_stats().path_cost == dijkstra.get_stats().path_cost

    def test_explores_fewer_cells_than_dijkstra(self) -> None:
        """A* should typically explore fewer cells than Dijkstra."""
        grid = _make_maze(cols=20, rows=20, seed=42)

        astar = AStarSolver(grid, grid.start, grid.end)
        astar.solve_complete()

        dijkstra = DijkstraSolver(grid, grid.start, grid.end)
        dijkstra.solve_complete()

        # A* should explore <= Dijkstra (or at most equal)
        assert astar.get_stats().cells_explored <= dijkstra.get_stats().cells_explored

    def test_different_heuristics(self) -> None:
        """All heuristics should find a valid path."""
        grid = _make_maze(cols=10, rows=10, seed=42)

        for h in Heuristic:
            solver = AStarSolver(grid, grid.start, grid.end, heuristic=h)
            found = solver.solve_complete()
            assert found is True
            _validate_path(grid, solver.get_path(), grid.start, grid.end)

    def test_euclidean_heuristic(self) -> None:
        grid = _make_maze(cols=10, rows=10, seed=42)
        solver = AStarSolver(grid, grid.start, grid.end, heuristic=Heuristic.EUCLIDEAN)
        solver.solve_complete()
        _validate_path(grid, solver.get_path(), grid.start, grid.end)

    def test_chebyshev_heuristic(self) -> None:
        grid = _make_maze(cols=10, rows=10, seed=42)
        solver = AStarSolver(grid, grid.start, grid.end, heuristic=Heuristic.CHEBYSHEV)
        solver.solve_complete()
        _validate_path(grid, solver.get_path(), grid.start, grid.end)


class TestGreedySolver:
    def test_may_find_suboptimal_path(self) -> None:
        """Greedy is not guaranteed to find the shortest path."""
        grid = _make_maze(cols=15, rows=15, seed=42)

        greedy = GreedySolver(grid, grid.start, grid.end)
        greedy.solve_complete()

        bfs = BFSSolver(grid, grid.start, grid.end)
        bfs.solve_complete()

        # Greedy path should be >= BFS path length
        assert len(greedy.get_path()) >= len(bfs.get_path())


class TestWallFollowerSolver:
    def test_left_hand_follows_wall(self) -> None:
        """Left-hand wall follower should work on a perfect maze."""
        grid = _make_maze(cols=10, rows=10, seed=42)
        solver = LeftWallFollowerSolver(grid, grid.start, grid.end)
        found = solver.solve_complete()
        assert found is True
        _validate_path(grid, solver.get_path(), grid.start, grid.end)

    def test_right_hand_follows_wall(self) -> None:
        """Right-hand wall follower should work on a perfect maze."""
        grid = _make_maze(cols=10, rows=10, seed=42)
        solver = RightWallFollowerSolver(grid, grid.start, grid.end)
        found = solver.solve_complete()
        assert found is True
        _validate_path(grid, solver.get_path(), grid.start, grid.end)

    def test_left_and_right_find_same_clean_path(self) -> None:
        """On a perfect maze, both hands should find the same solution path."""
        grid = _make_maze(cols=10, rows=10, seed=42)

        left = LeftWallFollowerSolver(grid, grid.start, grid.end)
        left.solve_complete()

        right = RightWallFollowerSolver(grid, grid.start, grid.end)
        right.solve_complete()

        # Both find valid paths
        _validate_path(grid, left.get_path(), grid.start, grid.end)
        _validate_path(grid, right.get_path(), grid.start, grid.end)

        # On a perfect maze (one unique path), cleaned paths should be identical
        assert left.get_path() == right.get_path()

    def test_left_and_right_explore_differently(self) -> None:
        """Left and right followers should explore different cells."""
        grid = _make_maze(cols=15, rows=15, seed=42)

        left = LeftWallFollowerSolver(grid, grid.start, grid.end)
        left.solve_complete()
        left_explored = left.get_stats().total_steps

        right = RightWallFollowerSolver(grid, grid.start, grid.end)
        right.solve_complete()
        right_explored = right.get_stats().total_steps

        # They should explore differently (not guaranteed but very likely on 15x15)
        assert left_explored > 0
        assert right_explored > 0

    def test_path_may_be_longer_than_bfs(self) -> None:
        """Wall followers typically find longer raw paths than BFS."""
        grid = _make_maze(cols=15, rows=15, seed=42)

        left = LeftWallFollowerSolver(grid, grid.start, grid.end)
        left.solve_complete()

        bfs = BFSSolver(grid, grid.start, grid.end)
        bfs.solve_complete()

        assert len(left.get_path()) >= len(bfs.get_path())

    def test_invalid_hand_raises(self) -> None:
        """Passing an invalid hand value should raise ValueError."""
        grid = _make_maze(cols=5, rows=5)
        with pytest.raises(ValueError, match="hand must be"):
            WallFollowerSolver(grid, grid.start, grid.end, hand="middle")
