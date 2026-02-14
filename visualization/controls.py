"""UI controls — dropdowns, sliders, buttons for maze configuration."""

from __future__ import annotations

from tkinter import Frame, StringVar, IntVar, BooleanVar, ttk
from typing import Callable

from config import DEFAULT_ANIMATION_DELAY_MS, MIN_ANIMATION_DELAY_MS, MAX_ANIMATION_DELAY_MS


class ControlPanel:
    """Top control bar with generator/solver selection, speed, and action buttons."""

    def __init__(self, parent: Frame) -> None:
        self._frame = ttk.Frame(parent)
        self._frame.pack(fill="x", padx=5, pady=5)

        self._callbacks: dict[str, Callable] = {}

        # --- Row 0: Generator and Solver selection ---
        row0 = ttk.Frame(self._frame)
        row0.pack(fill="x", pady=2)

        # Generator dropdown
        ttk.Label(row0, text="Generator:").pack(side="left", padx=(0, 5))
        self.generator_var = StringVar(value="DFS (Recursive Backtracker)")
        self._generator_combo = ttk.Combobox(
            row0,
            textvariable=self.generator_var,
            state="readonly",
            width=25,
        )
        self._generator_combo.pack(side="left", padx=(0, 15))

        # Solver dropdown
        ttk.Label(row0, text="Solver:").pack(side="left", padx=(0, 5))
        self.solver_var = StringVar(value="DFS")
        self._solver_combo = ttk.Combobox(
            row0,
            textvariable=self.solver_var,
            state="readonly",
            width=20,
        )
        self._solver_combo.pack(side="left", padx=(0, 15))

        # Seed
        ttk.Label(row0, text="Seed:").pack(side="left", padx=(0, 5))
        self.seed_var = StringVar(value="0")
        seed_entry = ttk.Entry(row0, textvariable=self.seed_var, width=6)
        seed_entry.pack(side="left", padx=(0, 15))

        # Grid size
        ttk.Label(row0, text="Cols:").pack(side="left", padx=(0, 3))
        self.cols_var = StringVar(value="30")
        ttk.Entry(row0, textvariable=self.cols_var, width=4).pack(side="left", padx=(0, 10))

        ttk.Label(row0, text="Rows:").pack(side="left", padx=(0, 3))
        self.rows_var = StringVar(value="20")
        ttk.Entry(row0, textvariable=self.rows_var, width=4).pack(side="left")

        # --- Row 1: Speed, buttons, toggles ---
        row1 = ttk.Frame(self._frame)
        row1.pack(fill="x", pady=2)

        # Speed slider
        ttk.Label(row1, text="Speed:").pack(side="left", padx=(0, 5))
        self.speed_var = IntVar(value=DEFAULT_ANIMATION_DELAY_MS)
        self._speed_scale = ttk.Scale(
            row1,
            from_=MIN_ANIMATION_DELAY_MS,
            to=MAX_ANIMATION_DELAY_MS,
            variable=self.speed_var,
            orient="horizontal",
            length=150,
            command=self._on_speed_change,
        )
        self._speed_scale.pack(side="left", padx=(0, 5))
        self._speed_label = ttk.Label(row1, text=f"{DEFAULT_ANIMATION_DELAY_MS}ms")
        self._speed_label.pack(side="left", padx=(0, 15))

        # Instant solve toggle
        self.instant_var = BooleanVar(value=False)
        ttk.Checkbutton(row1, text="Instant", variable=self.instant_var).pack(
            side="left", padx=(0, 15)
        )

        # Action buttons
        self._generate_btn = ttk.Button(row1, text="Generate", command=self._on_generate)
        self._generate_btn.pack(side="left", padx=3)

        self._solve_btn = ttk.Button(row1, text="Solve", command=self._on_solve)
        self._solve_btn.pack(side="left", padx=3)

        self._compare_btn = ttk.Button(row1, text="Compare All", command=self._on_compare)
        self._compare_btn.pack(side="left", padx=3)

        self._reset_btn = ttk.Button(row1, text="Reset", command=self._on_reset)
        self._reset_btn.pack(side="left", padx=3)

    def set_generators(self, names: list[str]) -> None:
        """Set available generator names in the dropdown."""
        self._generator_combo["values"] = names
        if names:
            self.generator_var.set(names[0])

    def set_solvers(self, names: list[str]) -> None:
        """Set available solver names in the dropdown."""
        self._solver_combo["values"] = names
        if names:
            self.solver_var.set(names[0])

    def on_generate(self, callback: Callable[[], None]) -> None:
        self._callbacks["generate"] = callback

    def on_solve(self, callback: Callable[[], None]) -> None:
        self._callbacks["solve"] = callback

    def on_compare(self, callback: Callable[[], None]) -> None:
        self._callbacks["compare"] = callback

    def on_reset(self, callback: Callable[[], None]) -> None:
        self._callbacks["reset"] = callback

    def on_speed_change(self, callback: Callable[[int], None]) -> None:
        self._callbacks["speed"] = callback

    def _on_generate(self) -> None:
        cb = self._callbacks.get("generate")
        if cb:
            cb()

    def _on_solve(self) -> None:
        cb = self._callbacks.get("solve")
        if cb:
            cb()

    def _on_compare(self) -> None:
        cb = self._callbacks.get("compare")
        if cb:
            cb()

    def _on_reset(self) -> None:
        cb = self._callbacks.get("reset")
        if cb:
            cb()

    def _on_speed_change(self, value: str) -> None:
        delay = int(float(value))
        self._speed_label.config(text=f"{delay}ms")
        cb = self._callbacks.get("speed")
        if cb:
            cb(delay)

    def set_buttons_state(self, generate: bool = True, solve: bool = True, compare: bool = True) -> None:
        """Enable/disable buttons during operations."""
        self._generate_btn.config(state="normal" if generate else "disabled")
        self._solve_btn.config(state="normal" if solve else "disabled")
        self._compare_btn.config(state="normal" if compare else "disabled")

    def get_seed(self) -> int | None:
        """Parse seed from the entry field."""
        text = self.seed_var.get().strip()
        if not text:
            return None
        try:
            return int(text)
        except ValueError:
            return None

    def get_grid_size(self) -> tuple[int, int]:
        """Parse (cols, rows) from entry fields."""
        try:
            cols = int(self.cols_var.get())
            rows = int(self.rows_var.get())
            return max(2, cols), max(2, rows)
        except ValueError:
            return 30, 20

    @property
    def frame(self) -> ttk.Frame:
        return self._frame
