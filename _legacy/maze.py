import random
from time import sleep
from cell import Cell


class Maze:
    def __init__(
        self,
        x1,
        y1,
        num_rows,
        num_cols,
        cell_size_x,
        cell_size_y,
        win=None,
        seed=None,
    ):
        if seed is not None:
            random.seed(seed)

        self.__x1 = x1
        self.__y1 = y1
        self.__num_rows = num_rows
        self.__num_cols = num_cols
        self.__cell_size_x = cell_size_x
        self.__cell_size_y = cell_size_y
        self.__win = win

        self.__cells = []
        self.__create_cells()
        self.__break_entrance_and_exit()
        self.__break_walls_r(0, 0)
        self.__reset_cells_visited()


    def __create_cells(self):
        self.__cells = []
        for i in range(self.__num_cols):
            col = []
            for j in range(self.__num_rows):
                col.append(Cell(self.__win))
            self.__cells.append(col)

        for i in range(self.__num_cols):
            for j in range(self.__num_rows):
                self.__draw_cell(i, j)


    def __draw_cell(self, i, j):
        x1 = self.__x1 + i * self.__cell_size_x
        y1 = self.__y1 + j * self.__cell_size_y
        x2 = x1 + self.__cell_size_x
        y2 = y1 + self.__cell_size_y

        self.__cells[i][j].draw(x1, y1, x2, y2)
        self.__animate()


    def __animate(self):
        if self.__win is None:
            return
        self.__win.redraw()
        sleep(0.05)

    def __break_entrance_and_exit(self):
        # entrance: top-left cell
        self.__cells[0][0].has_top_wall = False
        self.__draw_cell(0, 0)

        # exit: bottom-right cell
        self.__cells[self.__num_cols - 1][self.__num_rows - 1].has_bottom_wall = False
        self.__draw_cell(self.__num_cols - 1, self.__num_rows - 1)
    
    def __reset_cells_visited(self):
        for i in range(self.__num_cols):
            for j in range(self.__num_rows):
                self.__cells[i][j].visited = False


    def __break_walls_r(self, i, j):
        self.__cells[i][j].visited = True

        while True:
            neighbors = []

            # left
            if i > 0 and not self.__cells[i - 1][j].visited:
                neighbors.append((i - 1, j))
            # right
            if i < self.__num_cols - 1 and not self.__cells[i + 1][j].visited:
                neighbors.append((i + 1, j))
            # up
            if j > 0 and not self.__cells[i][j - 1].visited:
                neighbors.append((i, j - 1))
            # down
            if j < self.__num_rows - 1 and not self.__cells[i][j + 1].visited:
                neighbors.append((i, j + 1))

            if len(neighbors) == 0:
                self.__draw_cell(i, j)
                return

            next_i, next_j = random.choice(neighbors)

            if next_i == i - 1:
                self.__cells[i][j].has_left_wall = False
                self.__cells[next_i][next_j].has_right_wall = False
            elif next_i == i + 1:
                self.__cells[i][j].has_right_wall = False
                self.__cells[next_i][next_j].has_left_wall = False
            elif next_j == j - 1:
                self.__cells[i][j].has_top_wall = False
                self.__cells[next_i][next_j].has_bottom_wall = False
            elif next_j == j + 1:
                self.__cells[i][j].has_bottom_wall = False
                self.__cells[next_i][next_j].has_top_wall = False

            self.__draw_cell(i, j)
            self.__draw_cell(next_i, next_j)

            self.__break_walls_r(next_i, next_j)





    def solve(self):
        self.__reset_cells_visited()
        return self.__solve_r(0, 0)


    def __solve_r(self, i, j):
        self.__animate()
        self.__cells[i][j].visited = True

        # goal check: bottom-right cell
        if i == self.__num_cols - 1 and j == self.__num_rows - 1:
            return True

        # Try directions in a fixed order: left, right, up, down

        # left
        if i > 0 and not self.__cells[i][j].has_left_wall and not self.__cells[i - 1][j].visited:
            self.__cells[i][j].draw_move(self.__cells[i - 1][j])
            if self.__solve_r(i - 1, j):
                return True
            self.__cells[i][j].draw_move(self.__cells[i - 1][j], undo=True)

        # right
        if i < self.__num_cols - 1 and not self.__cells[i][j].has_right_wall and not self.__cells[i + 1][j].visited:
            self.__cells[i][j].draw_move(self.__cells[i + 1][j])
            if self.__solve_r(i + 1, j):
                return True
            self.__cells[i][j].draw_move(self.__cells[i + 1][j], undo=True)

        # up
        if j > 0 and not self.__cells[i][j].has_top_wall and not self.__cells[i][j - 1].visited:
            self.__cells[i][j].draw_move(self.__cells[i][j - 1])
            if self.__solve_r(i, j - 1):
                return True
            self.__cells[i][j].draw_move(self.__cells[i][j - 1], undo=True)

        # down
        if j < self.__num_rows - 1 and not self.__cells[i][j].has_bottom_wall and not self.__cells[i][j + 1].visited:
            self.__cells[i][j].draw_move(self.__cells[i][j + 1])
            if self.__solve_r(i, j + 1):
                return True
            self.__cells[i][j].draw_move(self.__cells[i][j + 1], undo=True)

        return False
