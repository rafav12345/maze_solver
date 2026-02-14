from tkinter import Tk, BOTH, Canvas
from cell import Line  # not strictly needed, but ok
from maze import Maze


class Window:
    def __init__(self, width, height):
        self.__root = Tk()
        self.__root.title("Maze Solver")

        self.__canvas = Canvas(self.__root, width=width, height=height)
        self.__canvas.pack(fill=BOTH, expand=True)

        self.__running = False
        self.__root.protocol("WM_DELETE_WINDOW", self.close)

    def redraw(self):
        self.__root.update_idletasks()
        self.__root.update()

    def wait_for_close(self):
        self.__running = True
        while self.__running:
            self.redraw()

    def close(self):
        self.__running = False

    def draw_line(self, line, fill_color):
        line.draw(self.__canvas, fill_color)


def main():
    win = Window(800, 600)
    maze = Maze(50, 50, 20, 30, 20, 20, win, seed=0)
    maze.solve()
    win.wait_for_close()


if __name__ == "__main__":
    main()