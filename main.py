"""Maze Decision Engine — main entry point.

Creates the Tk window, control panel, maze canvas, and wires everything together.
"""

from __future__ import annotations

import time
from tkinter import Tk, Canvas, BOTH, Frame, ttk

import config
from core.events import AlgorithmEvent, EventType
from core.grid import Grid, GridPosition
from generators.base_generator import BaseGenerator
from generators.dfs_generator import DFSGenerator
from generators.kruskal_generator import KruskalGenerator
from generators.prim_generator import PrimGenerator
from generators.binary_tree_generator import BinaryTreeGenerator
from generators.wilson_generator import WilsonGenerator
from generators.eller_generator import EllerGenerator
from generators.sidewinder_generator import SidewinderGenerator
from generators.hunt_kill_generator import HuntAndKillGenerator
from solvers.base_solver import BaseSolver
from solvers.dfs_solver import DFSSolver
from solvers.bfs_solver import BFSSolver
from solvers.dijkstra_solver import DijkstraSolver
from solvers.astar_solver import AStarSolver
from solvers.greedy_solver import GreedySolver
from solvers.wall_follower_solver import LeftWallFollowerSolver, RightWallFollowerSolver
from visualization.animator import Animator
from visualization.controls import ControlPanel
from visualization.renderer import MazeRenderer
from visualization.stats_panel import StatsPanel, ComparisonTable


# Registry of available generators and solvers.
GENERATORS: dict[str, type[BaseGenerator]] = {
    "DFS (Recursive Backtracker)": DFSGenerator,
    "Kruskal's (Randomized MST)": KruskalGenerator,
    "Prim's (Growing Tree)": PrimGenerator,
    "Binary Tree": BinaryTreeGenerator,
    "Wilson's (Loop-Erased Walk)": WilsonGenerator,
    "Eller's (Row-by-Row)": EllerGenerator,
    "Sidewinder": SidewinderGenerator,
    "Hunt-and-Kill": HuntAndKillGenerator,
}

SOLVERS: dict[str, type[BaseSolver]] = {
    "DFS": DFSSolver,
    "BFS": BFSSolver,
    "Dijkstra": DijkstraSolver,
    "A*": AStarSolver,
    "Greedy": GreedySolver,
    "Wall Follower (Left)": LeftWallFollowerSolver,
    "Wall Follower (Right)": RightWallFollowerSolver,
}


class MazeApp:
    """Main application controller. Wires together all components."""

    def __init__(self) -> None:
        self._root = Tk()
        self._root.title(config.WINDOW_TITLE)
        self._root.geometry(f"{config.DEFAULT_WINDOW_WIDTH}x{config.DEFAULT_WINDOW_HEIGHT}")

        # --- Layout ---
        # Top: controls
        self._controls = ControlPanel(self._root)
        self._controls.set_generators(list(GENERATORS.keys()))
        self._controls.set_solvers(list(SOLVERS.keys()))

        # Middle: canvas (expandable)
        canvas_frame = ttk.Frame(self._root)
        canvas_frame.pack(fill=BOTH, expand=True, padx=5)

        self._canvas = Canvas(canvas_frame, bg=config.BG_COLOR)
        self._canvas.pack(fill=BOTH, expand=True)

        # Bottom: stats
        bottom_frame = ttk.Frame(self._root)
        bottom_frame.pack(fill="x", padx=5, pady=5)

        self._stats_panel = StatsPanel(bottom_frame)
        self._comparison_table = ComparisonTable(bottom_frame)

        # --- State ---
        self._grid: Grid | None = None
        self._renderer: MazeRenderer | None = None
        self._animator = Animator(self._root, config.DEFAULT_ANIMATION_DELAY_MS)
        self._current_solver: BaseSolver | None = None

        # --- Wire callbacks ---
        self._controls.on_generate(self._do_generate)
        self._controls.on_solve(self._do_solve)
        self._controls.on_compare(self._do_compare)
        self._controls.on_reset(self._do_reset)
        self._controls.on_speed_change(self._on_speed_change)

        # Auto-generate on startup
        self._root.after(100, self._do_generate)

    def run(self) -> None:
        """Start the Tk event loop."""
        self._root.mainloop()

    # --- Actions ---

    def _do_generate(self) -> None:
        """Generate a new maze."""
        self._animator.stop()
        self._canvas.delete("all")
        self._stats_panel.clear()
        self._comparison_table.clear()

        cols, rows = self._controls.get_grid_size()
        seed = self._controls.get_seed()
        gen_name = self._controls.generator_var.get()

        self._grid = Grid(cols, rows)

        # Calculate cell size based on canvas (use reasonable defaults)
        cell_w = config.DEFAULT_CELL_WIDTH
        cell_h = config.DEFAULT_CELL_HEIGHT

        self._renderer = MazeRenderer(
            self._canvas,
            self._grid,
            config.DEFAULT_ORIGIN_X,
            config.DEFAULT_ORIGIN_Y,
            cell_w,
            cell_h,
        )

        # Draw initial grid (all walls)
        self._renderer.draw_entire_maze()

        # Get generator class
        gen_cls = GENERATORS.get(gen_name, DFSGenerator)
        generator = gen_cls(self._grid, seed=seed)

        self._controls.set_buttons_state(generate=False, solve=False, compare=False)

        if self._controls.instant_var.get():
            self._animator.run_instant(
                generator.generate(),
                self._renderer.handle_generator_event,
                self._on_generation_complete,
            )
        else:
            self._animator.start(
                generator.generate(),
                self._renderer.handle_generator_event,
                self._on_generation_complete,
            )

    def _on_generation_complete(self) -> None:
        """Called when maze generation finishes."""
        self._controls.set_buttons_state(generate=True, solve=True, compare=True)

    def _do_solve(self) -> None:
        """Solve the current maze with the selected solver."""
        if self._grid is None or self._renderer is None:
            return

        self._animator.stop()
        self._renderer.clear_paths()
        self._stats_panel.clear()
        self._grid.reset_visited()

        solver_name = self._controls.solver_var.get()
        solver_cls = SOLVERS.get(solver_name, DFSSolver)

        # Set colors for this solver
        colors = config.SOLVER_COLORS.get(solver_name, {})
        self._renderer.set_colors(
            colors.get("forward", config.DEFAULT_FORWARD_COLOR),
            colors.get("backtrack", config.DEFAULT_BACKTRACK_COLOR),
        )

        solver = solver_cls(self._grid, self._grid.start, self._grid.end)
        self._current_solver = solver
        self._stats_panel.update_stats(solver.get_stats(), solver_name)

        self._controls.set_buttons_state(generate=False, solve=False, compare=False)

        start_time = time.perf_counter()

        def on_event(event: AlgorithmEvent) -> None:
            self._renderer.handle_event(event)
            stats = solver.get_stats()
            self._stats_panel.update_live(stats)

        def on_complete() -> None:
            stats = solver.get_stats()
            stats.time_elapsed = time.perf_counter() - start_time
            self._stats_panel.update_stats(stats, solver_name)
            self._controls.set_buttons_state(generate=True, solve=True, compare=True)

        if self._controls.instant_var.get():
            self._animator.run_instant(solver.solve(), on_event, on_complete)
        else:
            self._animator.start(solver.solve(), on_event, on_complete)

    def _do_compare(self) -> None:
        """Run all solvers on the current maze and compare stats."""
        if self._grid is None or self._renderer is None:
            return

        self._animator.stop()
        self._comparison_table.clear()
        self._comparison_table.show()

        for solver_name, solver_cls in SOLVERS.items():
            self._grid.reset_visited()
            self._renderer.clear_paths()

            solver = solver_cls(self._grid, self._grid.start, self._grid.end)

            start_time = time.perf_counter()
            solver.solve_complete()
            stats = solver.get_stats()
            stats.time_elapsed = time.perf_counter() - start_time

            self._comparison_table.add_result(solver_name, stats)

        # Show the last solver's path
        if self._grid is not None:
            self._grid.reset_visited()

    def _do_reset(self) -> None:
        """Reset to clean state."""
        self._animator.stop()
        self._canvas.delete("all")
        self._grid = None
        self._renderer = None
        self._current_solver = None
        self._stats_panel.clear()
        self._comparison_table.clear()
        self._comparison_table.hide()
        self._controls.set_buttons_state(generate=True, solve=False, compare=False)

    def _on_speed_change(self, delay_ms: int) -> None:
        """Update animation speed."""
        self._animator.set_speed(delay_ms)


def main() -> None:
    app = MazeApp()
    app.run()


if __name__ == "__main__":
    main()
