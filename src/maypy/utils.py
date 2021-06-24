# @do:format
from collections import namedtuple

Cell = namedtuple("Content", ["content", "orientation"])


class Row:
    def __init__(self, previous=None):
        self.previous = previous
        self.grid = {}

    def __getitem__(self, item: slice):
        if isinstance(item, tuple):
            for i in item:
                self.grid[i.start] = Cell(i.stop, i.step)
            return self

        self.grid[item.start] = Cell(item.stop, item.step)
        return self

    def __len__(self):
        return max(self.grid.keys()) + 1 if self.grid else 1

    def __iter__(self):
        return iter(self.grid.items())


class Document:
    def __init__(self, padding=5):
        self.rows = []
        self.padding = " " * padding

    @property
    def row(self):
        if not self.rows:
            self.rows.append(Row())
        else:
            self.rows.append(Row(self.rows[-1]))

        return self.rows[-1]

    @property
    def grid(self):
        height = len(self.rows)
        width = max([len(row) for row in self.rows])

        grid = [[Cell("", "left") for _ in range(width)] for _ in range(height)]
        column_size = [0 for _ in range(width)]

        for row_idx, row in enumerate(self.rows):
            for column_idx, cell in row:
                grid[row_idx][column_idx] = cell
                column_size[column_idx] = max(column_size[column_idx], len(str(cell.content)))

        for row_idx, row in enumerate(grid):
            for column_idx, cell in enumerate(row):
                content = str(cell.content)
                content_size = len(content)

                target_size = column_size[column_idx]

                cell_padding = " " * (target_size - content_size)

                if cell.orientation == "right":
                    grid[row_idx][column_idx] = cell_padding + content
                elif cell.orientation == "center":
                    division_point = len(cell_padding) // 2
                    padding_left, padding_right = cell_padding[:division_point], cell_padding[:division_point]
                    padding_right += " " * (len(cell_padding) - (len(padding_left) + len(padding_right)))
                    grid[row_idx][column_idx] = padding_left + content + padding_right
                else:
                    grid[row_idx][column_idx] = content + cell_padding

        return grid

    def __str__(self):
        return repr(self)

    def __repr__(self):
        lines = []
        wrapping = None
        for row in self.grid:
            line = self.padding.join(["#", *row, "#"])
            lines.append(line)

            if wrapping is None:
                wrapping = "#" * len(line)

        spaced_wrapping = f"#{' ' * (len(wrapping) - 2)}#"

        return "\n".join([wrapping, spaced_wrapping, *lines, spaced_wrapping, wrapping])
