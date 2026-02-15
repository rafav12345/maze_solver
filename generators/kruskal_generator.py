"""Kruskal's Algorithm for maze generation.

Uses randomized minimum spanning tree approach with Union-Find data structure.
Creates mazes with uniform texture and good mix of passages.
"""

from typing import Iterator, Dict
import random

from core.events import AlgorithmEvent, EventType
from core.grid import Grid, GridPosition
from .base_generator import BaseGenerator


class KruskalGenerator(BaseGenerator):
    """Maze generation using Kruskal's algorithm (randomized MST).
    
    Algorithm:
    1. Start with grid where every cell is a separate set
    2. Create list of all possible walls between cells
    3. Randomly select walls and remove them if they connect different sets
    4. Use Union-Find data structure for efficient set operations
    
    Properties:
    - Creates uniform texture
    - Lots of short dead ends
    - Good mix of straight and winding passages
    - Unbiased (any maze equally likely)
    """
    
    def __init__(self, grid: Grid, seed: int | None = None):
        super().__init__(grid, seed)
        self._parent: Dict[GridPosition, GridPosition] = {}
    
    def _find(self, cell: GridPosition) -> GridPosition:
        """Find root of set containing cell (with path compression)."""
        if self._parent[cell] != cell:
            self._parent[cell] = self._find(self._parent[cell])
        return self._parent[cell]
    
    def _union(self, cell1: GridPosition, cell2: GridPosition) -> bool:
        """Unite sets containing cell1 and cell2.
        
        Returns:
            True if cells were in different sets (union performed)
            False if already in same set
        """
        root1 = self._find(cell1)
        root2 = self._find(cell2)
        
        if root1 != root2:
            self._parent[root2] = root1
            return True
        return False
    
    def generate(self) -> Iterator[AlgorithmEvent]:
        """Generate maze using Kruskal's algorithm."""
        # Initialize: every cell is its own set
        for row in range(self._grid.num_rows):
            for col in range(self._grid.num_cols):
                pos = GridPosition(col, row)
                self._parent[pos] = pos
        
        # Create list of all possible walls (edges)
        walls = []
        for row in range(self._grid.num_rows):
            for col in range(self._grid.num_cols):
                pos = GridPosition(col, row)
                
                # Add wall to right neighbor
                if col < self._grid.num_cols - 1:
                    neighbor = GridPosition(col + 1, row)
                    walls.append((pos, neighbor))
                
                # Add wall to bottom neighbor
                if row < self._grid.num_rows - 1:
                    neighbor = GridPosition(col, row + 1)
                    walls.append((pos, neighbor))
        
        # Shuffle walls for randomness
        self._rng.shuffle(walls)
        
        # Process walls in random order
        edges_added = 0
        total_edges_needed = self._grid.num_cols * self._grid.num_rows - 1
        
        for cell1, cell2 in walls:
            # Check if cells are in different sets
            if self._union(cell1, cell2):
                # Cells were in different sets, remove wall between them
                self._grid.remove_wall_between(cell1, cell2)
                edges_added += 1
                
                yield AlgorithmEvent(
                    event_type=EventType.VISIT,
                    position=cell1,
                    metadata={
                        "connecting_to": cell2,
                        "progress": edges_added / total_edges_needed
                    }
                )
                
                # Early termination: we have a spanning tree
                if edges_added >= total_edges_needed:
                    break
        
        # Break entrance and exit
        yield from self._break_entrance_and_exit()

        yield AlgorithmEvent(event_type=EventType.COMPLETE, position=self._grid.start)
