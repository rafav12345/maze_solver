"""Binary Tree Algorithm for maze generation.

Extremely fast single-pass algorithm with strong diagonal bias.
"""

from typing import Iterator
from core.events import AlgorithmEvent, EventType
from core.grid import Grid, GridPosition
from .base_generator import BaseGenerator


class BinaryTreeGenerator(BaseGenerator):
    """Maze generation using Binary Tree algorithm.
    
    Algorithm:
    1. For each cell, randomly carve passage either north or east
    2. Edge cells only have one option (or none for corner)
    3. Creates strong diagonal bias from SW to NE
    
    Properties:
    - EXTREMELY fast (single pass, O(n) time)
    - Strong diagonal bias (NE to SW)
    - Long corridors along top and right edges
    - Predictable structure
    - Always generates perfect maze (no loops)
    - Simplest maze algorithm to understand
    
    Trade-offs:
    - Very biased (not random-looking)
    - Easy to solve visually (follow top/right walls)
    - Best for: performance benchmarks, teaching, quick generation
    """
    
    def generate(self) -> Iterator[AlgorithmEvent]:
        """Generate maze using Binary Tree algorithm."""
        # Process cells in any order (we'll go row by row)
        for row in range(self._grid.num_rows):
            for col in range(self._grid.num_cols):
                pos = GridPosition(col, row)
                
                # Determine available directions
                neighbors = []
                
                # Can go north?
                if row > 0:
                    neighbors.append((GridPosition(col, row - 1), "north"))
                
                # Can go east?
                if col < self._grid.num_cols - 1:
                    neighbors.append((GridPosition(col + 1, row), "east"))
                
                if neighbors:
                    # Randomly pick one direction
                    chosen_neighbor, direction = self._rng.choice(neighbors)
                    self._grid.remove_wall_between(pos, chosen_neighbor)
                    
                    yield AlgorithmEvent(
                        event_type=EventType.WALL_REMOVED,
                        position=pos,
                        secondary_position=chosen_neighbor,
                        metadata={
                            "direction": direction,
                            "progress": (row * self._grid.num_cols + col + 1) / 
                                      (self._grid.num_rows * self._grid.num_cols)
                        }
                    )
        
        # Break entrance and exit
        yield from self._break_entrance_and_exit()
        
        yield AlgorithmEvent(
            event_type=EventType.COMPLETE,
            position=self._grid.start
        )
