"""Hunt-and-Kill Algorithm for maze generation.

DFS-like algorithm with a "hunt" phase to handle disconnected regions.
Creates more uniform mazes than pure DFS.
"""

from typing import Iterator
from core.events import AlgorithmEvent, EventType
from core.grid import Grid, GridPosition
from .base_generator import BaseGenerator


class HuntAndKillGenerator(BaseGenerator):
    """Maze generation using Hunt-and-Kill algorithm.
    
    Algorithm:
    1. Start random walk from random cell (like DFS)
    2. Continue until stuck (no unvisited neighbors)
    3. "Hunt" for first unvisited cell adjacent to visited cell
    4. Connect and continue walk from there
    5. Repeat until all cells visited
    
    Properties:
    - More uniform than pure DFS (fewer long corridors)
    - More organic looking than grid-based algorithms
    - Good balance between randomness and structure
    - No auxiliary data structures needed
    - Slightly slower than DFS due to hunt phase
    
    Comparison to DFS:
    - DFS: Backtracks when stuck, can create bias
    - Hunt-and-Kill: Scans grid for new starting point
    - Result: More balanced maze structure
    
    Best for:
    - Aesthetic mazes (good visual balance)
    - When DFS is too biased but Wilson's too slow
    - Teaching algorithm design patterns
    """
    
    def generate(self) -> Iterator[AlgorithmEvent]:
        """Generate maze using Hunt-and-Kill algorithm."""
        visited = set()
        
        # Start from random cell
        current = GridPosition(
            self._rng.randint(0, self._grid.num_cols - 1),
            self._rng.randint(0, self._grid.num_rows - 1)
        )
        visited.add(current)
        
        yield AlgorithmEvent(
            event_type=EventType.CELL_UPDATED,
            position=current,
            metadata={"type": "walk_start"}
        )
        
        total_cells = self._grid.num_cols * self._grid.num_rows
        
        while len(visited) < total_cells:
            # WALK phase: random walk until stuck
            unvisited_neighbors = [
                (n, direction) for n, direction in self._grid.get_neighbors(current)
                if n not in visited
            ]
            
            if unvisited_neighbors:
                # Continue walk: pick random unvisited neighbor
                next_cell, direction = self._rng.choice(unvisited_neighbors)
                self._grid.remove_wall_between(current, next_cell)
                visited.add(next_cell)
                current = next_cell
                
                yield AlgorithmEvent(
                    event_type=EventType.WALL_REMOVED,
                    position=current,
                    metadata={
                        "type": "walk",
                        "direction": direction,
                        "progress": len(visited) / total_cells
                    }
                )
            else:
                # HUNT phase: scan for unvisited cell next to visited cell
                found = False
                
                # Scan grid systematically
                for row in range(self._grid.num_rows):
                    for col in range(self._grid.num_cols):
                        pos = GridPosition(col, row)
                        
                        if pos not in visited:
                            # Check if adjacent to any visited cell
                            visited_neighbors = [
                                (n, direction) for n, direction in self._grid.get_neighbors(pos)
                                if n in visited
                            ]
                            
                            if visited_neighbors:
                                # Found unvisited cell next to visited region
                                # Connect to random visited neighbor
                                connection, direction = self._rng.choice(visited_neighbors)
                                self._grid.remove_wall_between(pos, connection)
                                visited.add(pos)
                                current = pos
                                found = True
                                
                                yield AlgorithmEvent(
                                    event_type=EventType.WALL_REMOVED,
                                    position=pos,
                                    secondary_position=connection,
                                    metadata={
                                        "type": "hunt",
                                        "direction": direction,
                                        "progress": len(visited) / total_cells
                                    }
                                )
                                break
                    
                    if found:
                        break
                
                # Should never happen, but safety check
                if not found and len(visited) < total_cells:
                    raise RuntimeError(
                        f"Hunt phase failed: {len(visited)}/{total_cells} cells visited"
                    )
        
        # Break entrance and exit
        yield from self._break_entrance_and_exit()
        
        yield AlgorithmEvent(
            event_type=EventType.COMPLETE,
            position=self._grid.start,
            metadata={"algorithm": "Hunt-and-Kill"}
        )
