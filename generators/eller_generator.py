"""Eller's Algorithm for maze generation.

Efficient row-by-row generation with constant memory usage.
Can generate infinite mazes by processing one row at a time.
"""

from typing import Iterator, Dict
from core.events import AlgorithmEvent, EventType
from core.grid import Grid, GridPosition
from .base_generator import BaseGenerator


class EllerGenerator(BaseGenerator):
    """Maze generation using Eller's algorithm (row-by-row).
    
    Algorithm:
    1. Process maze row by row from top to bottom
    2. Maintain sets of connected cells in current row
    3. Randomly join some adjacent cells in same row
    4. Randomly connect current row to next row (at least one per set)
    5. Ensure each set has at least one connection downward
    
    Properties:
    - Memory efficient (only needs two rows in memory)
    - Can generate infinite mazes (streaming)
    - Distinctive horizontal bias (more corridors than DFS)
    - More predictable than pure random
    - Good for procedural generation
    
    Performance:
    - Time: O(n) - processes each cell once
    - Space: O(width) - only current row in memory
    - Fast and efficient
    
    Best for:
    - Memory-constrained environments
    - Streaming/infinite maze generation
    - Procedural level generation
    """
    
    def generate(self) -> Iterator[AlgorithmEvent]:
        """Generate maze using Eller's algorithm."""
        # Track which set each cell belongs to
        sets: Dict[GridPosition, int] = {}
        next_set_id = 0
        
        for row in range(self._grid.num_rows):
            # Initialize new cells in this row with unique sets
            for col in range(self._grid.num_cols):
                pos = GridPosition(col, row)
                if pos not in sets:
                    sets[pos] = next_set_id
                    next_set_id += 1
            
            # Phase 1: Randomly join adjacent cells in current row
            for col in range(self._grid.num_cols - 1):
                pos = GridPosition(col, row)
                next_pos = GridPosition(col + 1, row)
                
                # Only join if in different sets
                if sets[pos] != sets[next_pos]:
                    # For last row, join all cells to ensure connectivity
                    # For other rows, randomly decide
                    should_join = (row == self._grid.num_rows - 1) or (self._rng.random() > 0.5)
                    
                    if should_join:
                        # Join the sets
                        old_set = sets[next_pos]
                        new_set = sets[pos]
                        
                        # Remove wall
                        self._grid.remove_wall_between(pos, next_pos)
                        
                        # Update all cells in old set to new set
                        for p, s in list(sets.items()):
                            if s == old_set and p.row == row:
                                sets[p] = new_set
                        
                        yield AlgorithmEvent(
                            event_type=EventType.WALL_REMOVED,
                            position=pos,
                            secondary_position=next_pos,
                            metadata={
                                "type": "horizontal_join",
                                "row": row,
                                "set": new_set
                            }
                        )
            
            # Phase 2: Connect current row to next row
            if row < self._grid.num_rows - 1:
                # Track which sets have at least one connection down
                set_has_connection: Dict[int, bool] = {}
                
                # Get all unique sets in current row
                row_sets = {}
                for col in range(self._grid.num_cols):
                    pos = GridPosition(col, row)
                    cell_set = sets[pos]
                    if cell_set not in row_sets:
                        row_sets[cell_set] = []
                    row_sets[cell_set].append(pos)
                
                # For each set, ensure at least one downward connection
                for cell_set, cells_in_set in row_sets.items():
                    # Force at least one connection per set
                    must_connect = [cells_in_set[0]] if cells_in_set else []
                    
                    # Randomly connect others
                    for pos in cells_in_set:
                        below = GridPosition(pos.col, row + 1)
                        
                        should_connect = (
                            pos in must_connect or
                            self._rng.random() > 0.5
                        )
                        
                        if should_connect:
                            self._grid.remove_wall_between(pos, below)
                            sets[below] = cell_set
                            set_has_connection[cell_set] = True
                            
                            yield AlgorithmEvent(
                                event_type=EventType.WALL_REMOVED,
                                position=pos,
                                secondary_position=below,
                                metadata={
                                    "type": "vertical_connect",
                                    "row": row,
                                    "set": cell_set,
                                    "progress": (row + 1) / self._grid.num_rows
                                }
                            )
        
        # Break entrance and exit
        yield from self._break_entrance_and_exit()
        
        yield AlgorithmEvent(
            event_type=EventType.COMPLETE,
            position=self._grid.start,
            metadata={"algorithm": "Eller's (Row-by-Row)"}
        )
