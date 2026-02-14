class Point:
    def __init__(self, x, y):
        self.x = x
        self.y = y


class Line:
    def __init__(self, p1, p2):
        self.p1 = p1
        self.p2 = p2

    def draw(self, canvas, fill_color):
        canvas.create_line(
            self.p1.x,
            self.p1.y,
            self.p2.x,
            self.p2.y,
            fill=fill_color,
            width=2,
        )


class Cell:
    def __init__(self, win=None):
        self.has_left_wall = True
        self.has_right_wall = True
        self.has_top_wall = True
        self.has_bottom_wall = True

        self.__x1 = -1
        self.__x2 = -1
        self.__y1 = -1
        self.__y2 = -1

        self.__win = win
        self.visited = False


    def draw(self, x1, y1, x2, y2):
        self.__x1 = x1
        self.__y1 = y1
        self.__x2 = x2
        self.__y2 = y2

        if self.__win is None:
            return

        bg = "#d9d9d9"
        wall = "black"

        # left
        self.__win.draw_line(
            Line(Point(self.__x1, self.__y1), Point(self.__x1, self.__y2)),
            wall if self.has_left_wall else bg,
        )
        # top
        self.__win.draw_line(
            Line(Point(self.__x1, self.__y1), Point(self.__x2, self.__y1)),
            wall if self.has_top_wall else bg,
        )
        # right
        self.__win.draw_line(
            Line(Point(self.__x2, self.__y1), Point(self.__x2, self.__y2)),
            wall if self.has_right_wall else bg,
        )
        # bottom
        self.__win.draw_line(
            Line(Point(self.__x1, self.__y2), Point(self.__x2, self.__y2)),
            wall if self.has_bottom_wall else bg,
        )

    def draw_move(self, to_cell, undo=False):
        if self.__win is None:
            return

        x1 = (self.__x1 + self.__x2) / 2
        y1 = (self.__y1 + self.__y2) / 2
        x2 = (to_cell.__x1 + to_cell.__x2) / 2
        y2 = (to_cell.__y1 + to_cell.__y2) / 2

        color = "gray" if undo else "red"
        self.__win.draw_line(Line(Point(x1, y1), Point(x2, y2)), color)
