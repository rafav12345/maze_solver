"""Sidewinder Algorithm for maze generation.

Similar to Binary Tree but creates horizontal runs before carving north.
Results in less extreme bias and more interesting patterns.
"""

from typing import Iterator, List
from core.events import AlgorithmEvent, EventType
from core.grid import Grid, GridPosition
from .base_generator import BaseGenerator


class SidewinderGenerator(BaseGenerator):
    """Maze generation using Sidewinder algorithm.
    
    Algorithm:
    1. Process grid row by row, left to right
    2. For each cell, either:
       a) Extend run eastward, OR
       b) Close run by carving north from random cell in run
    3. Top row always extends east (no north option)
    4. Rightmost cells always close run (no east option)
    
    Properties:
    - Like Binary Tree but less extreme bias
    - Creates horizontal "runs" of cells
    - Top row is always one long corridor
    - More balanced than Binary Tree
    - Still fast: O(n) single pass
    
    Comparison to Binary Tree:
    - Binary Tree: Every cell picks N or E
    - Sidewinder: Creates runs, then picks from run
    - Result: More interesting horizontal structure
    
    Best for:
    - Fast generation with less bias than Binary Tree
    - Horizontal corridor emphasis
    - Teaching algorithm variations
    """
    
    def generate(self) -> Iterator[AlgorithmEvent]:
        """Generate maze using Sidewinder algorithm."""
        for row in range(self._grid.num_rows):
            run: List[GridPosition] = []
            
            for col in range(self._grid.num_cols):
                pos = GridPosition(col, row)
                run.append(pos)
                
                # Determine if we should close the run
                at_east_boundary = (col == self._grid.num_cols - 1)
                at_north_boundary = (row == 0)
                
                # Must close run if at east edge OR randomly decide
                # Cannot close if at north edge (nowhere to carve)
                should_close_run = (
                    at_east_boundary or
                    (not at_north_boundary and self._rng.random() > 0.5)
                )
                
                if should_close_run:
                    # Close run by carving north from random cell in run
                    if not at_north_boundary:
                        # Pick random cell from current run
                        chosen = self._rng.choice(run)
                        above = GridPosition(chosen.col, chosen.row - 1)
                        
                        # Carve north
                        self._grid.remove_wall_between(chosen, above)
                        
                        yield AlgorithmEvent(
                            event_type=EventType.WALL_REMOVED,
                            position=chosen,
                            secondary_position=above,
                            metadata={
                                "type": "run_closed_north",
                                "run_length": len(run),
                                "row": row
                            }
                        )
                    
                    # Start new run
                    run = []
                else:
                    # Extend run eastward
                    next_pos = GridPosition(col + 1, row)
                    self._grid.remove_wall_between(pos, next_pos)
                    
                    yield AlgorithmEvent(
                        event_type=EventType.WALL_REMOVED,
                        position=pos,
                        secondary_position=next_pos,
                        metadata={
                            "type": "run_extended_east",
                            "run_length": len(run),
                            "row": row,
                            "progress": (row * self._grid.num_cols + col + 1) / 
                                      (self._grid.num_rows * self._grid.num_cols)
                        }
                    )
        
        # Break entrance and exit
        yield from self._break_entrance_and_exit()
        
        yield AlgorithmEvent(
            event_type=EventType.COMPLETE,
            position=self._grid.start,
            metadata={"algorithm": "Sidewinder"}
        )
