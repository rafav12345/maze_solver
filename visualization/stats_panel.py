"""Stats panel — displays real-time and final solver statistics."""

from __future__ import annotations

from tkinter import Frame, Label, StringVar, ttk

from config import STATS_FONT, STATS_HEADER_FONT
from solvers.base_solver import SolverStats


class StatsPanel:
    """Displays solver statistics in a Tk frame.

    Shows: cells explored, cells backtracked, path length, path cost,
    total steps, time elapsed.
    """

    def __init__(self, parent: Frame | ttk.Frame) -> None:
        self._frame = ttk.LabelFrame(parent, text="Statistics", padding=5)
        self._frame.pack(fill="x", padx=5, pady=2)

        self._vars: dict[str, StringVar] = {}
        self._labels: dict[str, Label] = {}

        fields = [
            ("solver", "Solver"),
            ("explored", "Explored"),
            ("backtracked", "Backtracked"),
            ("path_length", "Path Length"),
            ("path_cost", "Path Cost"),
            ("total_steps", "Total Steps"),
            ("time", "Time (s)"),
        ]

        for i, (key, label_text) in enumerate(fields):
            lbl = ttk.Label(self._frame, text=f"{label_text}:", font=STATS_HEADER_FONT)
            lbl.grid(row=i, column=0, sticky="w", padx=(0, 10))
            var = StringVar(value="—")
            val_lbl = ttk.Label(self._frame, textvariable=var, font=STATS_FONT)
            val_lbl.grid(row=i, column=1, sticky="w")
            self._vars[key] = var
            self._labels[key] = val_lbl

    def update_stats(self, stats: SolverStats, solver_name: str = "") -> None:
        """Update displayed statistics."""
        self._vars["solver"].set(solver_name or "—")
        self._vars["explored"].set(str(stats.cells_explored))
        self._vars["backtracked"].set(str(stats.cells_backtracked))
        self._vars["path_length"].set(str(stats.path_length))
        self._vars["path_cost"].set(f"{stats.path_cost:.1f}")
        self._vars["total_steps"].set(str(stats.total_steps))
        self._vars["time"].set(f"{stats.time_elapsed:.4f}")

    def update_live(self, stats: SolverStats) -> None:
        """Update only the fast-changing fields during solving."""
        self._vars["explored"].set(str(stats.cells_explored))
        self._vars["backtracked"].set(str(stats.cells_backtracked))
        self._vars["total_steps"].set(str(stats.total_steps))

    def clear(self) -> None:
        """Reset all fields to default."""
        for var in self._vars.values():
            var.set("—")

    @property
    def frame(self) -> ttk.LabelFrame:
        return self._frame


class ComparisonTable:
    """Displays side-by-side comparison of multiple solvers' stats."""

    def __init__(self, parent: Frame | ttk.Frame) -> None:
        self._frame = ttk.LabelFrame(parent, text="Comparison", padding=5)

        columns = ("solver", "explored", "backtracked", "path_len", "path_cost", "steps", "time")
        self._tree = ttk.Treeview(self._frame, columns=columns, show="headings", height=6)

        headings = {
            "solver": "Solver",
            "explored": "Explored",
            "backtracked": "Backtrack",
            "path_len": "Path Len",
            "path_cost": "Cost",
            "steps": "Steps",
            "time": "Time (s)",
        }
        widths = {
            "solver": 120,
            "explored": 70,
            "backtracked": 70,
            "path_len": 70,
            "path_cost": 60,
            "steps": 60,
            "time": 80,
        }

        for col_id in columns:
            self._tree.heading(col_id, text=headings[col_id])
            self._tree.column(col_id, width=widths[col_id], anchor="center")

        self._tree.pack(fill="x")

    def add_result(self, solver_name: str, stats: SolverStats) -> None:
        """Add a solver's results to the comparison table."""
        self._tree.insert(
            "",
            "end",
            values=(
                solver_name,
                stats.cells_explored,
                stats.cells_backtracked,
                stats.path_length,
                f"{stats.path_cost:.1f}",
                stats.total_steps,
                f"{stats.time_elapsed:.4f}",
            ),
        )

    def clear(self) -> None:
        """Remove all entries from the table."""
        for item in self._tree.get_children():
            self._tree.delete(item)

    def show(self) -> None:
        """Pack the comparison frame."""
        self._frame.pack(fill="x", padx=5, pady=2)

    def hide(self) -> None:
        """Unpack the comparison frame."""
        self._frame.pack_forget()
