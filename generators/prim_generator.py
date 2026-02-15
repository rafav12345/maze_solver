"""Prim's Algorithm for maze generation.

Uses randomized version of Prim's minimum spanning tree algorithm.
Grows maze from a starting point by expanding frontier.
"""

from typing import Iterator
from core.events import AlgorithmEvent, EventType
from core.grid import Grid, GridPosition
from .base_generator import BaseGenerator


class PrimGenerator(BaseGenerator):
    """Maze generation using randomized Prim's algorithm.
    
    Algorithm:
    1. Start with random cell, mark as "in maze"
    2. Add all neighbors to frontier list
    3. Randomly pick cell from frontier
    4. Connect to random adjacent "in maze" cell
    5. Add new cell's neighbors to frontier
    6. Repeat until frontier empty
    
    Properties:
    - Creates more organic, flowing mazes
    - Fewer long corridors than DFS
    - Central area often more dense
    - Biased toward shorter dead ends
    - Good balance between randomness and structure
    """
    
    def generate(self) -> Iterator[AlgorithmEvent]:
        """Generate maze using Prim's algorithm."""
        # Start with random cell
        start = GridPosition(
            self._rng.randint(0, self._grid.num_cols - 1),
            self._rng.randint(0, self._grid.num_rows - 1)
        )
        
        in_maze = {start}
        # Frontier: list of (cell, connection_point) tuples
        # connection_point is the in_maze cell to connect to
        frontier = []
        
        # Add start's neighbors to frontier
        for neighbor, _ in self._grid.get_neighbors(start):
            frontier.append((neighbor, start))
        
        yield AlgorithmEvent(
            event_type=EventType.CELL_UPDATED,
            position=start,
            metadata={"type": "start"}
        )
        
        # Process frontier
        while frontier:
            # Pick random cell from frontier
            idx = self._rng.randint(0, len(frontier) - 1)
            current, connection = frontier.pop(idx)
            
            # Skip if already in maze (can happen if added multiple times)
            if current in in_maze:
                continue
            
            # Add to maze by removing wall to connection point
            self._grid.remove_wall_between(current, connection)
            in_maze.add(current)
            
            yield AlgorithmEvent(
                event_type=EventType.WALL_REMOVED,
                position=current,
                secondary_position=connection,
                metadata={
                    "frontier_size": len(frontier),
                    "progress": len(in_maze) / (self._grid.num_cols * self._grid.num_rows)
                }
            )
            
            # Add current's neighbors to frontier (if not in maze)
            for neighbor, _ in self._grid.get_neighbors(current):
                if neighbor not in in_maze:
                    # Check if already in frontier
                    already_in_frontier = any(f[0] == neighbor for f in frontier)
                    if not already_in_frontier:
                        frontier.append((neighbor, current))
        
        # Break entrance and exit
        yield from self._break_entrance_and_exit()
        
        yield AlgorithmEvent(
            event_type=EventType.COMPLETE,
            position=self._grid.start
        )
