"""Performance benchmarking suite for solvers and generators.

Uses pytest-benchmark to measure execution time across maze sizes,
algorithm variants, and maze topologies. Run with:

    pytest tests/test_benchmarks.py -v
    pytest tests/test_benchmarks.py -v --benchmark-columns=mean,stddev,rounds
    pytest tests/test_benchmarks.py -v --benchmark-save=baseline
    pytest tests/test_benchmarks.py -v --benchmark-compare=0001_baseline

Requires: pip install pytest-benchmark
"""

import pytest
from core.grid import Grid, GridPosition
from generators import (
    BaseGenerator,
    BinaryTreeGenerator,
    DFSGenerator,
    EllerGenerator,
    HuntAndKillGenerator,
    KruskalGenerator,
    PrimGenerator,
    SidewinderGenerator,
    WilsonGenerator,
)
from solvers.astar_solver import AStarSolver, Heuristic
from solvers.base_solver import BaseSolver
from solvers.bfs_solver import BFSSolver
from solvers.dfs_solver import DFSSolver
from solvers.dijkstra_solver import DijkstraSolver
from solvers.greedy_solver import GreedySolver
from solvers.wall_follower_solver import (
    LeftWallFollowerSolver,
    RightWallFollowerSolver,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_MAZE_CACHE: dict[tuple[int, int, int], Grid] = {}


def _cached_maze(cols: int, rows: int, seed: int = 42) -> Grid:
    """Return a fresh *copy* of a DFS-generated maze (cached by key).

    We cache the wall layout so benchmark iterations don't re-generate,
    then deep-copy walls into a new grid so visited state is clean.
    """
    key = (cols, rows, seed)
    if key not in _MAZE_CACHE:
        grid = Grid(cols, rows)
        DFSGenerator(grid, seed=seed).generate_complete()
        _MAZE_CACHE[key] = grid

    src = _MAZE_CACHE[key]
    dst = Grid(cols, rows)
    for pos in src.all_positions():
        sc = src.get_cell(pos)
        dc = dst.get_cell(pos)
        dc.has_left_wall = sc.has_left_wall
        dc.has_right_wall = sc.has_right_wall
        dc.has_top_wall = sc.has_top_wall
        dc.has_bottom_wall = sc.has_bottom_wall
        dc.cost = sc.cost
    return dst


# ---------------------------------------------------------------------------
# Solver benchmarks
# ---------------------------------------------------------------------------

ALL_SOLVERS: list[tuple[str, type[BaseSolver]]] = [
    ("DFS", DFSSolver),
    ("BFS", BFSSolver),
    ("Dijkstra", DijkstraSolver),
    ("A*", AStarSolver),
    ("Greedy", GreedySolver),
    ("WallFollower-L", LeftWallFollowerSolver),
    ("WallFollower-R", RightWallFollowerSolver),
]

SOLVER_SIZES = [10, 25, 50, 100]


@pytest.mark.parametrize("size", SOLVER_SIZES, ids=[f"{s}x{s}" for s in SOLVER_SIZES])
@pytest.mark.parametrize(
    "solver_name,solver_cls",
    ALL_SOLVERS,
    ids=[name for name, _ in ALL_SOLVERS],
)
def test_solver_benchmark(benchmark, solver_name, solver_cls, size):
    """Benchmark each solver on square mazes of increasing size."""
    grid = _cached_maze(size, size)

    def run():
        g = _cached_maze(size, size)
        solver = solver_cls(g, g.start, g.end)
        solver.solve_complete()
        return solver

    result = benchmark(run)
    assert result.get_path(), f"{solver_name} failed to solve {size}x{size}"


# ---------------------------------------------------------------------------
# Generator benchmarks
# ---------------------------------------------------------------------------

ALL_GENERATORS: list[tuple[str, type[BaseGenerator]]] = [
    ("DFS", DFSGenerator),
    ("Kruskal", KruskalGenerator),
    ("Prim", PrimGenerator),
    ("Wilson", WilsonGenerator),
    ("Eller", EllerGenerator),
    ("BinaryTree", BinaryTreeGenerator),
    ("Sidewinder", SidewinderGenerator),
    ("HuntAndKill", HuntAndKillGenerator),
]

GENERATOR_SIZES = [10, 25, 50, 100]


@pytest.mark.parametrize("size", GENERATOR_SIZES, ids=[f"{s}x{s}" for s in GENERATOR_SIZES])
@pytest.mark.parametrize(
    "gen_name,gen_cls",
    ALL_GENERATORS,
    ids=[name for name, _ in ALL_GENERATORS],
)
def test_generator_benchmark(benchmark, gen_name, gen_cls, size):
    """Benchmark each generator on square mazes of increasing size."""

    def run():
        grid = Grid(size, size)
        gen = gen_cls(grid, seed=42)
        gen.generate_complete()
        return grid

    result = benchmark(run)
    # Sanity: check entrance is open
    assert not result.get_cell(result.start).has_top_wall


# ---------------------------------------------------------------------------
# Scaling benchmarks — measure how solvers scale with maze area
# ---------------------------------------------------------------------------

SCALING_SIZES = [10, 20, 40, 80]


@pytest.mark.parametrize("size", SCALING_SIZES, ids=[f"{s}x{s}" for s in SCALING_SIZES])
def test_astar_scaling(benchmark, size):
    """Track A* scaling behaviour as maze size grows."""

    def run():
        g = _cached_maze(size, size)
        solver = AStarSolver(g, g.start, g.end)
        solver.solve_complete()
        return solver

    result = benchmark(run)
    assert result.get_path()


@pytest.mark.parametrize("size", SCALING_SIZES, ids=[f"{s}x{s}" for s in SCALING_SIZES])
def test_bfs_scaling(benchmark, size):
    """Track BFS scaling behaviour as maze size grows."""

    def run():
        g = _cached_maze(size, size)
        solver = BFSSolver(g, g.start, g.end)
        solver.solve_complete()
        return solver

    result = benchmark(run)
    assert result.get_path()


# ---------------------------------------------------------------------------
# A* heuristic comparison
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "heuristic",
    list(Heuristic),
    ids=[h.name for h in Heuristic],
)
def test_astar_heuristic_benchmark(benchmark, heuristic):
    """Compare A* performance across heuristic functions on a 50x50 maze."""

    def run():
        g = _cached_maze(50, 50)
        solver = AStarSolver(g, g.start, g.end, heuristic=heuristic)
        solver.solve_complete()
        return solver

    result = benchmark(run)
    assert result.get_path()


# ---------------------------------------------------------------------------
# Generator topology impact — same solver, different generator topologies
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "gen_name,gen_cls",
    ALL_GENERATORS,
    ids=[name for name, _ in ALL_GENERATORS],
)
def test_topology_impact_on_astar(benchmark, gen_name, gen_cls):
    """Benchmark A* on 50x50 mazes from each generator to show topology impact."""

    _topo_cache: dict[str, Grid] = {}

    def _make(name, cls):
        if name not in _topo_cache:
            grid = Grid(50, 50)
            cls(grid, seed=42).generate_complete()
            _topo_cache[name] = grid
        src = _topo_cache[name]
        dst = Grid(50, 50)
        for pos in src.all_positions():
            sc = src.get_cell(pos)
            dc = dst.get_cell(pos)
            dc.has_left_wall = sc.has_left_wall
            dc.has_right_wall = sc.has_right_wall
            dc.has_top_wall = sc.has_top_wall
            dc.has_bottom_wall = sc.has_bottom_wall
        return dst

    def run():
        g = _make(gen_name, gen_cls)
        solver = AStarSolver(g, g.start, g.end)
        solver.solve_complete()
        return solver

    result = benchmark(run)
    assert result.get_path()


# ---------------------------------------------------------------------------
# Event throughput — measure event generation overhead
# ---------------------------------------------------------------------------


def test_solver_event_throughput(benchmark):
    """Measure raw event throughput: iterate all solver events on a 50x50 maze."""

    def run():
        g = _cached_maze(50, 50)
        solver = AStarSolver(g, g.start, g.end)
        count = 0
        for _ in solver.solve():
            count += 1
        return count

    count = benchmark(run)
    assert count > 0


def test_generator_event_throughput(benchmark):
    """Measure raw event throughput for maze generation on a 50x50 grid."""

    def run():
        grid = Grid(50, 50)
        gen = DFSGenerator(grid, seed=42)
        count = 0
        for _ in gen.generate():
            count += 1
        return count

    count = benchmark(run)
    assert count > 0


# ---------------------------------------------------------------------------
# Memory-proxy: stats collection
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "solver_name,solver_cls",
    ALL_SOLVERS,
    ids=[name for name, _ in ALL_SOLVERS],
)
def test_solver_stats_on_large_maze(solver_name, solver_cls):
    """Collect and verify solver stats on a 100x100 maze (non-benchmark)."""
    grid = _cached_maze(100, 100)
    solver = solver_cls(grid, grid.start, grid.end)
    solver.solve_complete()
    stats = solver.get_stats()
    assert stats.cells_explored > 0
    assert stats.path_length > 0
    assert stats.path_cost > 0
    assert stats.time_elapsed >= 0
