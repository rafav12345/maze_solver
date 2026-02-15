# Implementation Plan: Maze Algorithms + Benchmarking Suite

## 🎯 Project Goal
Extend your maze solver with 7 new generation algorithms and build a comprehensive benchmarking suite to validate and compare all algorithms.

**Timeline:** 2 weeks
**Difficulty:** Low-Medium
**Impact:** Foundation for all future extensions

---

## 📅 Week 1: New Maze Generation Algorithms

### Overview
Add 7 new maze generation algorithms, each with unique characteristics that produce visually and structurally different mazes.

### Daily Breakdown

#### **Day 1: Kruskal's Algorithm**
*Creates mazes using randomized minimum spanning tree*

**Algorithm:**
1. Start with grid where every cell is a separate set
2. Create list of all possible walls
3. Randomly select walls and remove them if they connect different sets
4. Use Union-Find data structure for efficient set operations

**Properties:**
- Creates uniform texture
- Lots of short dead ends
- Good mix of straight and winding passages
- Unbiased (any maze equally likely)

**Implementation:**
```python
# generators/kruskal_generator.py
from typing import Iterator
from core.events import AlgorithmEvent, EventType
from core.grid import Grid, GridPosition
from .base_generator import BaseGenerator
import random

class KruskalGenerator(BaseGenerator):
    """Maze generation using Kruskal's algorithm (randomized MST)"""
    
    def generate(self) -> Iterator[AlgorithmEvent]:
        # Union-Find data structure
        parent = {}
        
        def find(cell):
            if parent[cell] != cell:
                parent[cell] = find(parent[cell])
            return parent[cell]
        
        def union(cell1, cell2):
            root1, root2 = find(cell1), find(cell2)
            if root1 != root2:
                parent[root2] = root1
                return True
            return False
        
        # Initialize: every cell is its own set
        for row in range(self._grid.rows):
            for col in range(self._grid.cols):
                pos = GridPosition(col, row)
                parent[pos] = pos
        
        # Create list of all walls
        walls = []
        for row in range(self._grid.rows):
            for col in range(self._grid.cols):
                pos = GridPosition(col, row)
                # Add wall to right neighbor
                if col < self._grid.cols - 1:
                    walls.append((pos, GridPosition(col + 1, row)))
                # Add wall to bottom neighbor
                if row < self._grid.rows - 1:
                    walls.append((pos, GridPosition(col, row + 1)))
        
        # Shuffle walls
        random.shuffle(walls)
        
        # Process walls
        for cell1, cell2 in walls:
            if union(cell1, cell2):
                # Cells were in different sets, remove wall
                self._grid.remove_wall(cell1, cell2)
                
                yield AlgorithmEvent(
                    type=EventType.VISIT,
                    position=cell1,
                    metadata={"connecting_to": cell2}
                )
        
        yield AlgorithmEvent(type=EventType.COMPLETE)
```

**Testing:**
```python
def test_kruskal_generator():
    grid = Grid(20, 20)
    gen = KruskalGenerator(grid, seed=42)
    
    # Run generation
    for event in gen.generate():
        pass
    
    # Verify maze properties
    assert grid.is_perfect_maze()  # No loops, fully connected
    assert grid.count_dead_ends() > 0
```

---

#### **Day 2: Prim's Algorithm**
*Growing tree algorithm that expands from a starting point*

**Algorithm:**
1. Start with random cell, mark as "in maze"
2. Add all neighbors to frontier list
3. Randomly pick cell from frontier
4. Connect to random adjacent "in maze" cell
5. Add new cell's neighbors to frontier
6. Repeat until frontier empty

**Properties:**
- Creates more organic, flowing mazes
- Fewer long corridors than DFS
- Central area often more dense
- Biased toward shorter dead ends

**Implementation:**
```python
# generators/prim_generator.py
class PrimGenerator(BaseGenerator):
    """Maze generation using randomized Prim's algorithm"""
    
    def generate(self) -> Iterator[AlgorithmEvent]:
        # Start with random cell
        start = GridPosition(
            self._random.randint(0, self._grid.cols - 1),
            self._random.randint(0, self._grid.rows - 1)
        )
        
        in_maze = {start}
        frontier = []
        
        # Add start's neighbors to frontier
        for neighbor in self._grid.get_neighbors(start):
            frontier.append((neighbor, start))
        
        yield AlgorithmEvent(type=EventType.VISIT, position=start)
        
        while frontier:
            # Pick random cell from frontier
            idx = self._random.randint(0, len(frontier) - 1)
            current, connection = frontier.pop(idx)
            
            if current in in_maze:
                continue
            
            # Add to maze by removing wall
            self._grid.remove_wall(current, connection)
            in_maze.add(current)
            
            yield AlgorithmEvent(
                type=EventType.VISIT,
                position=current,
                metadata={"from": connection}
            )
            
            # Add current's neighbors to frontier
            for neighbor in self._grid.get_neighbors(current):
                if neighbor not in in_maze:
                    frontier.append((neighbor, current))
        
        yield AlgorithmEvent(type=EventType.COMPLETE)
```

---

#### **Day 3: Binary Tree Algorithm**
*Simple, fast algorithm with distinctive bias*

**Algorithm:**
1. For each cell, randomly remove either north or east wall
2. Edge cells only have one option
3. Creates strong diagonal bias

**Properties:**
- EXTREMELY fast (single pass)
- Strong diagonal bias (NE to SW)
- Long corridors along edges
- Predictable structure
- Always generates perfect maze

**Implementation:**
```python
# generators/binary_tree_generator.py
class BinaryTreeGenerator(BaseGenerator):
    """Fast algorithm with strong NE bias"""
    
    def generate(self) -> Iterator[AlgorithmEvent]:
        for row in range(self._grid.rows):
            for col in range(self._grid.cols):
                pos = GridPosition(col, row)
                
                neighbors = []
                # Can go north?
                if row > 0:
                    neighbors.append(GridPosition(col, row - 1))
                # Can go east?
                if col < self._grid.cols - 1:
                    neighbors.append(GridPosition(col + 1, row))
                
                if neighbors:
                    # Pick random direction
                    chosen = self._random.choice(neighbors)
                    self._grid.remove_wall(pos, chosen)
                    
                    yield AlgorithmEvent(
                        type=EventType.VISIT,
                        position=pos,
                        metadata={"direction": chosen}
                    )
        
        yield AlgorithmEvent(type=EventType.COMPLETE)
```

---

#### **Day 4: Wilson's Algorithm**
*Loop-erased random walks - creates unbiased mazes*

**Algorithm:**
1. Start with random cell in maze
2. Pick random cell not in maze
3. Do random walk until hitting maze
4. Erase any loops from walk
5. Add loop-erased path to maze
6. Repeat until all cells in maze

**Properties:**
- Perfectly unbiased (uniform distribution)
- Slower than other algorithms
- Interesting to watch (random walks)
- No structural bias at all

**Implementation:**
```python
# generators/wilson_generator.py
class WilsonGenerator(BaseGenerator):
    """Unbiased maze generation via loop-erased random walks"""
    
    def generate(self) -> Iterator[AlgorithmEvent]:
        # Start with one random cell
        start = GridPosition(
            self._random.randint(0, self._grid.cols - 1),
            self._random.randint(0, self._grid.rows - 1)
        )
        
        in_maze = {start}
        remaining = set()
        
        for row in range(self._grid.rows):
            for col in range(self._grid.cols):
                pos = GridPosition(col, row)
                if pos != start:
                    remaining.add(pos)
        
        yield AlgorithmEvent(type=EventType.VISIT, position=start)
        
        while remaining:
            # Pick random cell not in maze
            current = self._random.choice(list(remaining))
            path = []
            walk_cell = current
            
            # Random walk until hitting maze
            while walk_cell not in in_maze:
                path.append(walk_cell)
                
                # Random step
                neighbors = list(self._grid.get_all_neighbors(walk_cell))
                walk_cell = self._random.choice(neighbors)
                
                # Erase loops: if we revisit a cell, erase back to it
                if walk_cell in path:
                    loop_start = path.index(walk_cell)
                    path = path[:loop_start + 1]
                
                yield AlgorithmEvent(
                    type=EventType.BACKTRACK,
                    position=walk_cell,
                    metadata={"path_length": len(path)}
                )
            
            # Add path to maze
            for i in range(len(path)):
                cell = path[i]
                if i == 0:
                    next_cell = walk_cell  # Connect to maze
                else:
                    next_cell = path[i - 1]
                
                self._grid.remove_wall(cell, next_cell)
                in_maze.add(cell)
                remaining.discard(cell)
                
                yield AlgorithmEvent(
                    type=EventType.VISIT,
                    position=cell
                )
        
        yield AlgorithmEvent(type=EventType.COMPLETE)
```

---

#### **Day 5: Eller's Algorithm**
*Efficient row-by-row generation*

**Algorithm:**
1. Process maze row by row
2. Maintain sets of connected cells
3. Randomly join some cells in current row
4. Randomly connect current row to next row
5. Ensures connectivity while allowing randomness

**Properties:**
- Memory efficient (only needs two rows)
- Can generate infinite mazes
- Distinctive horizontal corridors
- More predictable than DFS/Kruskal

**Implementation:**
```python
# generators/eller_generator.py
class EllerGenerator(BaseGenerator):
    """Row-by-row generation with set tracking"""
    
    def generate(self) -> Iterator[AlgorithmEvent]:
        # Track which set each cell belongs to
        sets = {}
        next_set_id = 0
        
        for row in range(self._grid.rows):
            # Initialize new cells
            for col in range(self._grid.cols):
                pos = GridPosition(col, row)
                if pos not in sets:
                    sets[pos] = next_set_id
                    next_set_id += 1
            
            # Join some cells in current row
            for col in range(self._grid.cols - 1):
                pos = GridPosition(col, row)
                next_pos = GridPosition(col + 1, row)
                
                if sets[pos] != sets[next_pos]:
                    if self._random.random() > 0.5:
                        # Join sets
                        old_set = sets[next_pos]
                        new_set = sets[pos]
                        self._grid.remove_wall(pos, next_pos)
                        
                        # Update all cells in old set
                        for p, s in sets.items():
                            if s == old_set:
                                sets[p] = new_set
                        
                        yield AlgorithmEvent(
                            type=EventType.VISIT,
                            position=pos
                        )
            
            # Connect to next row
            if row < self._grid.rows - 1:
                # For each set, connect at least one cell down
                set_has_connection = {}
                
                for col in range(self._grid.cols):
                    pos = GridPosition(col, row)
                    below = GridPosition(col, row + 1)
                    cell_set = sets[pos]
                    
                    should_connect = (
                        cell_set not in set_has_connection or
                        self._random.random() > 0.5
                    )
                    
                    if should_connect:
                        self._grid.remove_wall(pos, below)
                        sets[below] = cell_set
                        set_has_connection[cell_set] = True
                        
                        yield AlgorithmEvent(
                            type=EventType.VISIT,
                            position=below
                        )
        
        yield AlgorithmEvent(type=EventType.COMPLETE)
```

---

#### **Day 6: Sidewinder Algorithm**
*Binary tree variant with horizontal bias*

**Algorithm:**
1. Process row by row, left to right
2. Randomly extend run east or carve north
3. When carving north, pick random cell from current run
4. More controllable than binary tree

**Properties:**
- Like binary tree but less extreme bias
- Long horizontal corridors
- Top row is always one long corridor
- Fast and simple

**Implementation:**
```python
# generators/sidewinder_generator.py
class SidewinderGenerator(BaseGenerator):
    """Binary tree variant with horizontal runs"""
    
    def generate(self) -> Iterator[AlgorithmEvent]:
        for row in range(self._grid.rows):
            run = []
            
            for col in range(self._grid.cols):
                pos = GridPosition(col, row)
                run.append(pos)
                
                # At east boundary or randomly decide to close run
                at_east_boundary = (col == self._grid.cols - 1)
                at_north_boundary = (row == 0)
                
                should_close = (
                    at_east_boundary or
                    (not at_north_boundary and self._random.random() > 0.5)
                )
                
                if should_close:
                    # Pick random cell from run and go north
                    if not at_north_boundary:
                        chosen = self._random.choice(run)
                        above = GridPosition(chosen.col, chosen.row - 1)
                        self._grid.remove_wall(chosen, above)
                        
                        yield AlgorithmEvent(
                            type=EventType.VISIT,
                            position=chosen,
                            metadata={"run_length": len(run)}
                        )
                    
                    run = []
                else:
                    # Extend run east
                    next_pos = GridPosition(col + 1, row)
                    self._grid.remove_wall(pos, next_pos)
                    
                    yield AlgorithmEvent(
                        type=EventType.VISIT,
                        position=pos
                    )
        
        yield AlgorithmEvent(type=EventType.COMPLETE)
```

---

#### **Day 7: Hunt-and-Kill Algorithm**
*DFS-like but handles disconnected regions*

**Algorithm:**
1. Start random walk (like DFS)
2. When stuck, "hunt" for unvisited cell adjacent to visited
3. Connect and continue walk from there
4. More uniform than pure DFS

**Properties:**
- Fewer long corridors than DFS
- More organic looking
- Good balance of features
- Slightly slower than DFS

**Implementation:**
```python
# generators/hunt_kill_generator.py
class HuntAndKillGenerator(BaseGenerator):
    """DFS-like with hunt phase for better distribution"""
    
    def generate(self) -> Iterator[AlgorithmEvent]:
        visited = set()
        
        # Start random cell
        current = GridPosition(
            self._random.randint(0, self._grid.cols - 1),
            self._random.randint(0, self._grid.rows - 1)
        )
        visited.add(current)
        
        yield AlgorithmEvent(type=EventType.VISIT, position=current)
        
        while len(visited) < self._grid.cols * self._grid.rows:
            # Walk phase: random walk until stuck
            unvisited_neighbors = [
                n for n in self._grid.get_all_neighbors(current)
                if n not in visited
            ]
            
            if unvisited_neighbors:
                # Continue walk
                next_cell = self._random.choice(unvisited_neighbors)
                self._grid.remove_wall(current, next_cell)
                visited.add(next_cell)
                current = next_cell
                
                yield AlgorithmEvent(
                    type=EventType.VISIT,
                    position=current
                )
            else:
                # Hunt phase: find unvisited cell next to visited
                found = False
                
                for row in range(self._grid.rows):
                    for col in range(self._grid.cols):
                        pos = GridPosition(col, row)
                        
                        if pos not in visited:
                            # Check if adjacent to visited cell
                            visited_neighbors = [
                                n for n in self._grid.get_all_neighbors(pos)
                                if n in visited
                            ]
                            
                            if visited_neighbors:
                                # Connect to random visited neighbor
                                connection = self._random.choice(visited_neighbors)
                                self._grid.remove_wall(pos, connection)
                                visited.add(pos)
                                current = pos
                                found = True
                                
                                yield AlgorithmEvent(
                                    type=EventType.BACKTRACK,
                                    position=pos,
                                    metadata={"hunt_mode": True}
                                )
                                break
                    
                    if found:
                        break
        
        yield AlgorithmEvent(type=EventType.COMPLETE)
```

---

### Update main.py

```python
# main.py - Add to GENERATORS dict
GENERATORS: dict[str, type[BaseGenerator]] = {
    "DFS (Recursive Backtracker)": DFSGenerator,
    "Kruskal's (MST)": KruskalGenerator,
    "Prim's (Growing Tree)": PrimGenerator,
    "Wilson's (Loop-Erased Walk)": WilsonGenerator,
    "Eller's (Row-by-Row)": EllerGenerator,
    "Binary Tree": BinaryTreeGenerator,
    "Sidewinder": SidewinderGenerator,
    "Hunt and Kill": HuntAndKillGenerator,
}
```

---

## 📅 Week 2: Benchmarking Suite

### Overview
Build comprehensive performance testing and analysis framework.

### Daily Breakdown

#### **Day 1-2: Core Benchmark Framework**

**Create benchmark infrastructure:**

```python
# benchmarks/__init__.py
```

```python
# benchmarks/core.py
from dataclasses import dataclass, field
from typing import Dict, List, Any
import time
import tracemalloc
from datetime import datetime

@dataclass
class BenchmarkResult:
    """Results from a single benchmark run"""
    algorithm_name: str
    maze_size: tuple[int, int]
    maze_algorithm: str
    seed: int
    
    # Time metrics
    wall_clock_time: float  # seconds
    cpu_time: float  # seconds
    
    # Space metrics
    peak_memory_mb: float
    
    # Algorithm metrics
    steps_taken: int
    cells_visited: int
    path_length: int
    optimal_path_length: int
    
    # Success metrics
    found_solution: bool
    timed_out: bool
    
    # Metadata
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    
    @property
    def optimality_ratio(self) -> float:
        """How close to optimal? 1.0 = optimal"""
        if self.optimal_path_length == 0:
            return 0.0
        return self.optimal_path_length / self.path_length
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'algorithm': self.algorithm_name,
            'maze_size': f"{self.maze_size[0]}x{self.maze_size[1]}",
            'generator': self.maze_algorithm,
            'time_ms': round(self.wall_clock_time * 1000, 2),
            'memory_mb': round(self.peak_memory_mb, 2),
            'steps': self.steps_taken,
            'path_length': self.path_length,
            'optimality': round(self.optimality_ratio, 3),
            'success': self.found_solution,
        }


class BenchmarkRunner:
    """Runs benchmarks and collects results"""
    
    def __init__(self, timeout_seconds: float = 30.0):
        self.timeout = timeout_seconds
        self.results: List[BenchmarkResult] = []
    
    def benchmark_solver(
        self,
        solver_class,
        grid,
        start,
        end,
        solver_name: str,
        maze_config: dict
    ) -> BenchmarkResult:
        """Benchmark a single solver on a maze"""
        
        # Start memory tracking
        tracemalloc.start()
        
        # Create solver
        solver = solver_class(grid, start, end)
        
        # Time the solve
        start_time = time.perf_counter()
        start_cpu = time.process_time()
        
        # Run solver
        try:
            for event in solver.solve():
                # Check timeout
                if time.perf_counter() - start_time > self.timeout:
                    timed_out = True
                    break
            else:
                timed_out = False
        except Exception as e:
            print(f"Error in {solver_name}: {e}")
            timed_out = True
        
        end_time = time.perf_counter()
        end_cpu = time.process_time()
        
        # Get memory stats
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        
        # Get algorithm stats
        stats = solver.get_stats()
        
        # Calculate optimal path (using Dijkstra as ground truth)
        from solvers.dijkstra_solver import DijkstraSolver
        optimal_solver = DijkstraSolver(grid, start, end)
        optimal_solver.solve_complete()
        optimal_stats = optimal_solver.get_stats()
        
        return BenchmarkResult(
            algorithm_name=solver_name,
            maze_size=(grid.cols, grid.rows),
            maze_algorithm=maze_config.get('generator', 'unknown'),
            seed=maze_config.get('seed', 0),
            wall_clock_time=end_time - start_time,
            cpu_time=end_cpu - start_cpu,
            peak_memory_mb=peak / (1024 * 1024),
            steps_taken=stats.steps,
            cells_visited=stats.cells_visited,
            path_length=stats.path_length,
            optimal_path_length=optimal_stats.path_length,
            found_solution=stats.found_solution,
            timed_out=timed_out,
        )
    
    def run_benchmark_suite(
        self,
        solvers: Dict[str, type],
        generators: Dict[str, type],
        sizes: List[tuple[int, int]],
        trials_per_config: int = 3
    ):
        """Run comprehensive benchmark suite"""
        
        total_tests = len(solvers) * len(generators) * len(sizes) * trials_per_config
        completed = 0
        
        print(f"Running {total_tests} benchmark tests...")
        
        for size in sizes:
            for gen_name, gen_class in generators.items():
                for trial in range(trials_per_config):
                    # Generate maze
                    from core.grid import Grid
                    grid = Grid(size[0], size[1])
                    generator = gen_class(grid, seed=trial)
                    
                    # Run generation
                    for _ in generator.generate():
                        pass
                    
                    maze_config = {
                        'generator': gen_name,
                        'size': size,
                        'seed': trial,
                    }
                    
                    # Test each solver
                    for solver_name, solver_class in solvers.items():
                        # Reset grid
                        grid.reset_visited()
                        
                        result = self.benchmark_solver(
                            solver_class,
                            grid,
                            grid.start,
                            grid.end,
                            solver_name,
                            maze_config
                        )
                        
                        self.results.append(result)
                        completed += 1
                        
                        # Progress
                        if completed % 10 == 0:
                            print(f"Progress: {completed}/{total_tests} "
                                  f"({100*completed/total_tests:.1f}%)")
        
        print("Benchmarking complete!")
        return self.results
```

---

#### **Day 3: Visualization Module**

```python
# benchmarks/visualization.py
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from typing import List
import numpy as np

class BenchmarkVisualizer:
    """Create visualizations from benchmark results"""
    
    def __init__(self, results: List):
        self.df = pd.DataFrame([r.to_dict() for r in results])
    
    def plot_time_comparison(self, size_filter=None):
        """Bar chart comparing solve times"""
        df = self.df if size_filter is None else \
             self.df[self.df['maze_size'] == size_filter]
        
        plt.figure(figsize=(12, 6))
        
        # Group by algorithm and calculate mean
        grouped = df.groupby('algorithm')['time_ms'].mean().sort_values()
        
        grouped.plot(kind='bar')
        plt.title('Average Solve Time by Algorithm')
        plt.xlabel('Algorithm')
        plt.ylabel('Time (ms)')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig('benchmarks/output/time_comparison.png', dpi=150)
        plt.close()
    
    def plot_scaling(self):
        """Line plot showing how algorithms scale with size"""
        plt.figure(figsize=(14, 8))
        
        for algo in self.df['algorithm'].unique():
            algo_df = self.df[self.df['algorithm'] == algo]
            
            # Group by size and calculate mean
            grouped = algo_df.groupby('maze_size')['time_ms'].mean()
            
            plt.plot(range(len(grouped)), grouped.values, 
                    marker='o', label=algo, linewidth=2)
        
        plt.xlabel('Maze Size')
        plt.ylabel('Time (ms)')
        plt.title('Algorithm Scaling with Maze Size')
        plt.legend()
        plt.xticks(range(len(grouped)), grouped.index, rotation=45)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('benchmarks/output/scaling.png', dpi=150)
        plt.close()
    
    def plot_heatmap(self):
        """Heatmap: algorithm vs maze generator performance"""
        plt.figure(figsize=(12, 8))
        
        pivot = self.df.pivot_table(
            values='time_ms',
            index='algorithm',
            columns='generator',
            aggfunc='mean'
        )
        
        sns.heatmap(pivot, annot=True, fmt='.1f', cmap='YlOrRd')
        plt.title('Solve Time (ms) by Algorithm and Generator')
        plt.tight_layout()
        plt.savefig('benchmarks/output/heatmap.png', dpi=150)
        plt.close()
    
    def plot_memory_usage(self):
        """Memory consumption comparison"""
        plt.figure(figsize=(12, 6))
        
        grouped = self.df.groupby('algorithm')['memory_mb'].mean().sort_values()
        
        grouped.plot(kind='bar', color='steelblue')
        plt.title('Average Memory Usage by Algorithm')
        plt.xlabel('Algorithm')
        plt.ylabel('Peak Memory (MB)')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig('benchmarks/output/memory.png', dpi=150)
        plt.close()
    
    def plot_optimality(self):
        """Path optimality comparison"""
        plt.figure(figsize=(12, 6))
        
        grouped = self.df.groupby('algorithm')['optimality'].mean().sort_values(ascending=False)
        
        grouped.plot(kind='bar', color='green', alpha=0.7)
        plt.title('Path Optimality by Algorithm')
        plt.xlabel('Algorithm')
        plt.ylabel('Optimality Ratio (1.0 = optimal)')
        plt.axhline(y=1.0, color='r', linestyle='--', label='Perfect')
        plt.xticks(rotation=45, ha='right')
        plt.legend()
        plt.tight_layout()
        plt.savefig('benchmarks/output/optimality.png', dpi=150)
        plt.close()
    
    def plot_pareto(self):
        """Time vs Optimality tradeoff"""
        plt.figure(figsize=(10, 8))
        
        for algo in self.df['algorithm'].unique():
            algo_df = self.df[self.df['algorithm'] == algo]
            
            avg_time = algo_df['time_ms'].mean()
            avg_opt = algo_df['optimality'].mean()
            
            plt.scatter(avg_time, avg_opt, s=100, alpha=0.6, label=algo)
            plt.annotate(algo, (avg_time, avg_opt), 
                        fontsize=8, alpha=0.7)
        
        plt.xlabel('Average Time (ms)')
        plt.ylabel('Average Optimality')
        plt.title('Time vs Optimality Tradeoff')
        plt.grid(True, alpha=0.3)
        plt.legend(loc='best', fontsize=8)
        plt.tight_layout()
        plt.savefig('benchmarks/output/pareto.png', dpi=150)
        plt.close()
    
    def generate_all_plots(self):
        """Generate all visualization"""
        print("Generating visualizations...")
        
        self.plot_time_comparison()
        self.plot_scaling()
        self.plot_heatmap()
        self.plot_memory_usage()
        self.plot_optimality()
        self.plot_pareto()
        
        print("Visualizations saved to benchmarks/output/")
```

---

#### **Day 4-5: Report Generation**

```python
# benchmarks/report.py
from typing import List
import pandas as pd
from datetime import datetime

class BenchmarkReport:
    """Generate comprehensive benchmark reports"""
    
    def __init__(self, results: List):
        self.results = results
        self.df = pd.DataFrame([r.to_dict() for r in results])
    
    def generate_markdown(self) -> str:
        """Generate markdown report"""
        
        # Rankings
        fastest = self.df.groupby('algorithm')['time_ms'].mean().idxmin()
        fastest_time = self.df.groupby('algorithm')['time_ms'].mean().min()
        
        most_optimal = self.df.groupby('algorithm')['optimality'].mean().idxmax()
        optimal_score = self.df.groupby('algorithm')['optimality'].mean().max()
        
        most_efficient = self.df.groupby('algorithm')['memory_mb'].mean().idxmin()
        mem_score = self.df.groupby('algorithm')['memory_mb'].mean().min()
        
        report = f"""# Maze Solver Performance Report

**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  
**Total Tests:** {len(self.results)}  
**Algorithms Tested:** {len(self.df['algorithm'].unique())}  
**Maze Generators:** {len(self.df['generator'].unique())}  

---

## 🏆 Summary

- **Fastest Algorithm:** {fastest} ({fastest_time:.1f}ms avg)
- **Most Optimal:** {most_optimal} ({optimal_score:.3f} optimality)
- **Most Memory Efficient:** {most_efficient} ({mem_score:.2f}MB avg)

---

## 📊 Algorithm Rankings

### By Speed (Average Time)
"""
        
        # Speed rankings
        speed_ranking = self.df.groupby('algorithm')['time_ms'].mean().sort_values()
        for i, (algo, time) in enumerate(speed_ranking.items(), 1):
            report += f"{i}. **{algo}**: {time:.2f}ms\n"
        
        report += "\n### By Optimality\n"
        
        # Optimality rankings
        opt_ranking = self.df.groupby('algorithm')['optimality'].mean().sort_values(ascending=False)
        for i, (algo, opt) in enumerate(opt_ranking.items(), 1):
            report += f"{i}. **{algo}**: {opt:.3f}\n"
        
        report += "\n### By Memory Efficiency\n"
        
        # Memory rankings
        mem_ranking = self.df.groupby('algorithm')['memory_mb'].mean().sort_values()
        for i, (algo, mem) in enumerate(mem_ranking.items(), 1):
            report += f"{i}. **{algo}**: {mem:.2f}MB\n"
        
        report += """

---

## 📈 Detailed Analysis

### Performance by Maze Size

"""
        
        # Performance by size
        for size in sorted(self.df['maze_size'].unique()):
            size_df = self.df[self.df['maze_size'] == size]
            report += f"\n#### {size}\n"
            
            for algo in size_df['algorithm'].unique():
                algo_df = size_df[size_df['algorithm'] == algo]
                avg_time = algo_df['time_ms'].mean()
                avg_opt = algo_df['optimality'].mean()
                report += f"- {algo}: {avg_time:.2f}ms, {avg_opt:.3f} optimality\n"
        
        report += """

---

## 🎯 Recommendations

"""
        
        # Generate recommendations
        report += self._generate_recommendations()
        
        report += """

---

## 📊 Visualizations

See the following charts in `benchmarks/output/`:
- `time_comparison.png` - Bar chart of average solve times
- `scaling.png` - How algorithms scale with maze size  
- `heatmap.png` - Performance across generators
- `memory.png` - Memory usage comparison
- `optimality.png` - Path quality comparison
- `pareto.png` - Time vs optimality tradeoff

---

**Raw Data:** See `benchmarks/output/results.csv` for complete data
"""
        
        return report
    
    def _generate_recommendations(self) -> str:
        """Generate usage recommendations based on results"""
        
        fastest = self.df.groupby('algorithm')['time_ms'].mean().idxmin()
        most_optimal = self.df.groupby('algorithm')['optimality'].mean().idxmax()
        
        # Check which is best for different sizes
        small_df = self.df[self.df['maze_size'].str.contains('10x10|15x15')]
        large_df = self.df[self.df['maze_size'].str.contains('50x50|100x100')]
        
        best_small = small_df.groupby('algorithm')['time_ms'].mean().idxmin()
        best_large = large_df.groupby('algorithm')['time_ms'].mean().idxmin() if len(large_df) > 0 else fastest
        
        recs = f"""
### For Small Mazes (≤20x20)
**Use:** {best_small}  
Good balance of speed and simplicity.

### For Large Mazes (≥50x50)  
**Use:** {best_large}  
Best performance at scale.

### For Optimal Paths
**Use:** {most_optimal}  
Guaranteed shortest path.

### For Maximum Speed
**Use:** {fastest}  
Fastest on average across all sizes.

### For Educational Use
**Use:** BFS or DFS  
Easy to understand and visualize.
"""
        return recs
    
    def save_reports(self, output_dir: str = 'benchmarks/output'):
        """Save all reports"""
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        # Markdown report
        with open(f'{output_dir}/report.md', 'w') as f:
            f.write(self.generate_markdown())
        
        # CSV data
        self.df.to_csv(f'{output_dir}/results.csv', index=False)
        
        # JSON data
        import json
        with open(f'{output_dir}/results.json', 'w') as f:
            json.dump([r.__dict__ for r in self.results], f, indent=2)
        
        print(f"Reports saved to {output_dir}/")
```

---

#### **Day 6-7: CLI Tool & Integration**

```python
# benchmarks/cli.py
import argparse
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description='Maze Solver Benchmarking Suite')
    
    parser.add_argument(
        '--sizes',
        nargs='+',
        default=['10x10', '20x20', '30x30'],
        help='Maze sizes to test (e.g., 10x10 20x20)'
    )
    
    parser.add_argument(
        '--trials',
        type=int,
        default=3,
        help='Number of trials per configuration'
    )
    
    parser.add_argument(
        '--timeout',
        type=float,
        default=30.0,
        help='Timeout in seconds per solver'
    )
    
    parser.add_argument(
        '--output',
        default='benchmarks/output',
        help='Output directory'
    )
    
    parser.add_argument(
        '--skip-plots',
        action='store_true',
        help='Skip generating visualization plots'
    )
    
    args = parser.parse_args()
    
    # Parse sizes
    sizes = []
    for s in args.sizes:
        w, h = map(int, s.split('x'))
        sizes.append((w, h))
    
    # Import solvers and generators
    from main import SOLVERS, GENERATORS
    from benchmarks.core import BenchmarkRunner
    from benchmarks.visualization import BenchmarkVisualizer
    from benchmarks.report import BenchmarkReport
    
    # Run benchmarks
    print("=" * 60)
    print("MAZE SOLVER BENCHMARK SUITE")
    print("=" * 60)
    print(f"Algorithms: {len(SOLVERS)}")
    print(f"Generators: {len(GENERATORS)}")
    print(f"Sizes: {sizes}")
    print(f"Trials: {args.trials}")
    print("=" * 60)
    
    runner = BenchmarkRunner(timeout_seconds=args.timeout)
    results = runner.run_benchmark_suite(
        SOLVERS,
        GENERATORS,
        sizes,
        trials_per_config=args.trials
    )
    
    # Generate reports
    report = BenchmarkReport(results)
    report.save_reports(args.output)
    
    # Generate visualizations
    if not args.skip_plots:
        viz = BenchmarkVisualizer(results)
        viz.generate_all_plots()
    
    print("\n" + "=" * 60)
    print("BENCHMARKING COMPLETE!")
    print(f"Results saved to: {args.output}/")
    print("=" * 60)

if __name__ == '__main__':
    main()
```

**Usage:**
```bash
# Quick benchmark (3 sizes, 3 trials each)
python -m benchmarks.cli

# Comprehensive benchmark
python -m benchmarks.cli \
  --sizes 10x10 20x20 30x30 50x50 100x100 \
  --trials 5 \
  --timeout 60

# Fast benchmark without plots
python -m benchmarks.cli --sizes 15x15 --trials 1 --skip-plots
```

---

## 🎯 Testing & Validation

```python
# tests/test_generators.py - Add tests for new algorithms
import pytest
from core.grid import Grid
from generators import *

@pytest.mark.parametrize("generator_class", [
    KruskalGenerator,
    PrimGenerator,
    WilsonGenerator,
    EllerGenerator,
    BinaryTreeGenerator,
    SidewinderGenerator,
    HuntAndKillGenerator,
])
def test_generator_creates_perfect_maze(generator_class):
    """All generators should create perfect mazes (no loops, fully connected)"""
    grid = Grid(20, 20)
    gen = generator_class(grid, seed=42)
    
    # Generate maze
    for _ in gen.generate():
        pass
    
    # Verify properties
    assert grid.is_fully_connected(), f"{generator_class.__name__} didn't create connected maze"
    assert grid.is_tree(), f"{generator_class.__name__} created loops"

def test_generators_produce_different_mazes():
    """Different algorithms should produce different maze structures"""
    from generators.dfs_generator import DFSGenerator
    
    generators = [
        DFSGenerator,
        KruskalGenerator,
        PrimGenerator,
    ]
    
    mazes = []
    for gen_class in generators:
        grid = Grid(15, 15)
        gen = gen_class(grid, seed=42)
        for _ in gen.generate():
            pass
        mazes.append(grid.to_string())
    
    # At least some should be different
    assert len(set(mazes)) > 1, "All generators produced identical mazes"
```

---

## 📚 Documentation

```markdown
# Maze Generation Algorithms

## Overview
This project includes 8 different maze generation algorithms, each with unique characteristics.

## Algorithm Comparison

| Algorithm | Speed | Bias | Characteristics | Best For |
|-----------|-------|------|----------------|----------|
| **DFS** | Fast | High | Long corridors, low branching | General purpose |
| **Kruskal's** | Medium | None | Uniform texture, short dead ends | Unbiased mazes |
| **Prim's** | Medium | Low | Organic, flowing passages | Natural-looking mazes |
| **Wilson's** | Slow | None | Perfectly unbiased | Fair random mazes |
| **Eller's** | Fast | Medium | Horizontal corridors | Memory-efficient |
| **Binary Tree** | Very Fast | Very High | Diagonal bias, predicatable | Speed critical |
| **Sidewinder** | Fast | High | Horizontal runs | Simple, fast |
| **Hunt-and-Kill** | Medium | Low | Balanced, organic | Aesthetic mazes |

## Usage

```python
from core.grid import Grid
from generators import KruskalGenerator

# Create grid
grid = Grid(30, 30)

# Generate maze
generator = KruskalGenerator(grid, seed=42)
for event in generator.generate():
    # Process events (for visualization)
    pass

# Use the maze
# ... (solving, export, etc.)
```

## Benchmarking

Run comprehensive benchmarks:

```bash
python -m benchmarks.cli --sizes 10x10 20x20 50x50 --trials 5
```

View results in `benchmarks/output/report.md`
```

---

## 🚀 Deliverables

**Week 1:**
- ✅ 7 new maze generation algorithms
- ✅ Updated UI dropdown
- ✅ Tests for all generators
- ✅ Documentation

**Week 2:**
- ✅ Benchmark framework
- ✅ Visualization suite (6 chart types)
- ✅ Report generation (Markdown, CSV, JSON)
- ✅ CLI tool
- ✅ Integration with CI/CD ready

**Final Outputs:**
- `benchmarks/output/report.md` - Comprehensive analysis
- `benchmarks/output/*.png` - 6 visualization charts
- `benchmarks/output/results.csv` - Raw data
- `benchmarks/cli.py` - Runnable benchmark tool

---

## 🎉 Success Criteria

**Functional:**
- [ ] All 8 generators work correctly
- [ ] Benchmark suite runs without errors
- [ ] All visualizations generate successfully
- [ ] CLI tool is user-friendly

**Quality:**
- [ ] All generators create perfect mazes (no loops)
- [ ] Each generator produces visually distinct mazes
- [ ] Benchmarks cover all algorithms + generators
- [ ] Reports are clear and actionable

**Performance:**
- [ ] New generators complete in <1 second for 50x50
- [ ] Benchmark suite completes in <5 minutes
- [ ] Visualizations generate in <10 seconds

---

## 🤝 Next Steps After Completion

With this foundation, you're ready for:
1. **Web version** - Port to React
2. **AI playground** - RL training environments
3. **Competitive game** - User accounts + leaderboards
4. **Hybrid solver** - ML-based algorithm selection

**This is the perfect starting point. Let's build it! 🚀**

Ready to start? Which would you like to tackle first - Week 1 (algorithms) or Week 2 (benchmarks)?
