import unittest
from maze import Maze


class Tests(unittest.TestCase):
    def test_maze_create_cells(self):
        num_cols = 12
        num_rows = 10
        m1 = Maze(0, 0, num_rows, num_cols, 10, 10)
        self.assertEqual(len(m1._Maze__cells), num_cols)
        self.assertEqual(len(m1._Maze__cells[0]), num_rows)

    def test_maze_create_cells_small(self):
        m = Maze(0, 0, 2, 2, 10, 10)
        self.assertEqual(len(m._Maze__cells), 2)
        self.assertEqual(len(m._Maze__cells[0]), 2)

    def test_maze_create_cells_rectangular(self):
        m = Maze(0, 0, 5, 20, 10, 10)
        self.assertEqual(len(m._Maze__cells), 20)
        self.assertEqual(len(m._Maze__cells[0]), 5)
        
    def test_break_entrance_and_exit(self):
        m = Maze(0, 0, 10, 12, 10, 10)
        self.assertFalse(m._Maze__cells[0][0].has_top_wall)
        self.assertFalse(m._Maze__cells[11][9].has_bottom_wall)



if __name__ == "__main__":
    unittest.main()
