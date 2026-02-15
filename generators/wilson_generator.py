"""Wilson's Algorithm for maze generation.

Uses loop-erased random walks to generate perfectly unbiased mazes.
Based on uniform spanning tree theory.
"""

from typing import Iterator
from core.events import AlgorithmEvent, EventType
from core.grid import Grid, GridPosition
from .base_generator import BaseGenerator


class WilsonGenerator(BaseGenerator):
    """Maze generation using Wilson's algorithm (loop-erased random walks).
    
    Algorithm:
    1. Start with one random cell in maze
    2. Pick random cell not in maze
    3. Perform random walk until hitting maze
    4. Erase any loops from the walk path
    5. Add loop-erased path to maze
    6. Repeat until all cells in maze
    
    Properties:
    - PERFECTLY unbiased (uniform distribution over all spanning trees)
    - Mathematically elegant (uses Markov chain theory)
    - Mesmerizing to watch (random walks + loop erasure)
    - Slower than other algorithms (but worth it!)
    - No structural bias whatsoever
    
    Performance:
    - Slow at start (small maze target to hit)
    - Fast at end (large maze target to hit)
    - Overall: O(n) expected time, but high constants
    
    Best for:
    - True randomness needed (research, fairness)
    - Visual demonstrations (beautiful to watch)
    - Baseline for comparing other algorithms
    - Understanding graph theory
    """
    
    def generate(self) -> Iterator[AlgorithmEvent]:
        """Generate maze using Wilson's algorithm."""
        # Start with one random cell
        start = GridPosition(
            self._rng.randint(0, self._grid.num_cols - 1),
            self._rng.randint(0, self._grid.num_rows - 1)
        )
        
        in_maze = {start}
        remaining = set()
        
        # Build set of remaining cells
        for row in range(self._grid.num_rows):
            for col in range(self._grid.num_cols):
                pos = GridPosition(col, row)
                if pos != start:
                    remaining.add(pos)
        
        yield AlgorithmEvent(
            event_type=EventType.CELL_UPDATED,
            position=start,
            metadata={"type": "maze_seed", "in_maze_count": 1}
        )
        
        # Process remaining cells
        while remaining:
            # Pick random unvisited cell
            current = self._rng.choice(list(remaining))
            
            # Perform loop-erased random walk
            path = []
            walk_cell = current
            walk_iterations = 0
            
            # Random walk until hitting maze
            while walk_cell not in in_maze:
                path.append(walk_cell)
                
                # Take random step to neighbor
                neighbors = [n for n, _ in self._grid.get_neighbors(walk_cell)]
                walk_cell = self._rng.choice(neighbors)
                walk_iterations += 1
                
                # Loop erasure: if we revisit a cell in our path, erase back to it
                if walk_cell in path:
                    # Find where we hit the loop
                    loop_start = path.index(walk_cell)
                    # Erase everything after loop start
                    erased_count = len(path) - loop_start
                    path = path[:loop_start]
                    
                    # Visualize loop erasure
                    yield AlgorithmEvent(
                        event_type=EventType.BACKTRACK,
                        position=walk_cell,
                        metadata={
                            "type": "loop_erased",
                            "erased_steps": erased_count,
                            "path_length": len(path),
                            "walk_iterations": walk_iterations
                        }
                    )
                else:
                    # Visualize random walk progress
                    if walk_iterations % 5 == 0:  # Don't spam events
                        yield AlgorithmEvent(
                            event_type=EventType.VISIT,
                            position=walk_cell,
                            metadata={
                                "type": "random_walk",
                                "path_length": len(path),
                                "remaining": len(remaining)
                            }
                        )
            
            # Now add the loop-erased path to maze
            # Connect path to the maze at walk_cell
            for i in range(len(path)):
                cell = path[i]
                
                # Determine next cell in path
                if i == len(path) - 1:
                    # Last cell in path connects to maze
                    next_cell = walk_cell
                else:
                    # Connect to next cell in path
                    next_cell = path[i + 1]
                
                # Remove wall between cells
                self._grid.remove_wall_between(cell, next_cell)
                in_maze.add(cell)
                remaining.discard(cell)
                
                yield AlgorithmEvent(
                    event_type=EventType.WALL_REMOVED,
                    position=cell,
                    secondary_position=next_cell,
                    metadata={
                        "type": "path_added",
                        "in_maze_count": len(in_maze),
                        "progress": len(in_maze) / (self._grid.num_cols * self._grid.num_rows)
                    }
                )
        
        # Break entrance and exit
        yield from self._break_entrance_and_exit()
        
        yield AlgorithmEvent(
            event_type=EventType.COMPLETE,
            position=self._grid.start,
            metadata={"algorithm": "Wilson's (Loop-Erased Random Walk)"}
        )
