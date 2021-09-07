"""
Grid related classes and functions
"""
import logging
import math
from collections import defaultdict

from pup.common.checkin import Checkin
from pup.common.datatypes import CheckinDataset, CheckinList
from pup.common.rectangle import Rectangle


logger = logging.getLogger(__name__)


class Grid(Rectangle):
    """ An grid on the space. Grid cells are indexed from the min value. This grid covers the entire space

    Attributes
    ----------
    min_x: float
        minimum x value
    min_y: float
        minimum y value
    max_x: float
        maximum x value
    max_y: float
        maximum y value
    cell_len_x: float
        length of a cell in x dimension
    cell_len_y: float
        length of a cell in y dimension
    data: defaultdict
        value
    """

    min_x = None
    min_y = None
    max_x = None
    max_y = None

    cell_len_x = None
    cell_len_y = None

    data = None

    def __init__(self, min_x, min_y, max_x, max_y, cell_len_x, cell_len_y):
        """
        Initialize a grid with given parameters.

        Parameters
        ----------
        min_x: float
            minimum x value
        min_y: float
            minimum y value
        max_x: float
            maximum x value
        max_y: float
            maximum y value
        cell_len_x: float
            length of a cell in x dimension
        cell_len_y: float
            length of a cell in y dimension
        """
        super().__init__(min_x, min_y, max_x, max_y)

        self.cell_len_x = float(cell_len_x)
        self.cell_len_y = float(cell_len_y)
        self.data = defaultdict(defaultdict)

    def get_shape(self) -> tuple:
        """
        Tuple of grid dimensions

        Returns
        -------
        tuple
            (shape_x, shape_y) as length in x and y dimensions
        """
        shape_x, shape_y = self.find_grid_index(self.max_x, self.max_y)

        # there are 2 cases:
        # case 1: when x size is divisible by the cell length, then nothing remains, so nothing needs to do
        # case 2: when x size is NOT divisible by the cell length, then there is still 1 remains, need to add 1 more
        if shape_x * self.cell_len_x < (self.max_x - self.min_x):
            shape_x += 1

        if shape_y * self.cell_len_y < (self.max_y - self.min_y):
            shape_y += 1

        return shape_x, shape_y

    def find_grid_index(self, x: float, y: float) -> tuple:
        """
        Find index in the grid of a position (x, y)

        Parameters
        ----------
        x: float
            x position
        y: float
            y position

        Returns
        -------
        tuple
            indexes in x and y dimensions
        """
        x_idx = find_grid_index_on_dimension(x, self.min_x, self.cell_len_x)
        y_idx = find_grid_index_on_dimension(y, self.min_y, self.cell_len_y)

        return x_idx, y_idx

    def find_cell_boundary(self, x: int, y: int) -> Rectangle:
        """ Find the boundary of the cell (x, y)

        Parameters
        ----------
        x
            index of cell in x dimension
        y
            index of cell in y dimension

        Returns
        -------
        Rectangle
            boundary of the cell or None if the cell index is out of the grid
        """
        max_x_idx, max_y_idx = self.get_shape()
        if 0 <= x < max_x_idx and 0 <= y < max_y_idx:
            cell_min_x = x * self.cell_len_x + self.min_x
            cell_min_y = y * self.cell_len_y + self.min_y

            return Rectangle(cell_min_x, cell_min_y, cell_min_x + self.cell_len_x, cell_min_y + self.cell_len_y)

        else:
            logger.error('Cell index out of bound: ({}, {}) out of ({}, {})'.format(x, y, max_x_idx, max_y_idx))
            return None

    def extend_boundary_to_nearest(self, k: int):
        """
        Extend the boundary to the nearest k

        Parameters
        ----------
        k
            The order to extend to
        """
        self.min_x = extend_to_nearest(self.min_x, k, True)
        self.min_y = extend_to_nearest(self.min_y, k, True)
        self.max_x = extend_to_nearest(self.max_x, k, False)
        self.max_y = extend_to_nearest(self.max_y, k, False)

    def __str__(self):
        return "Grid(min_x={min_x}, min_y={min_y}, max_x={max_x}, " \
               "max_y={max_y}, cell_len_x={cell_len_x}, cell_len_y={cell_len_y})".format(**vars(self))


def find_grid_index_on_dimension(pos: float, min_value: float, cell_len: float) -> int:
    """

    Parameters
    ----------
    pos: float
        position
    min_value: float
        min value of the dimension
    cell_len: float
        size of each grid cell of the dimension
    Returns
    -------
    int
        grid index for pos
    """
    return int((pos - min_value) / cell_len)


def create_grid_for_data(data: CheckinList, cell_len_x: float, cell_len_y: float, boundary_order: int = 0) -> Grid:
    """
    Create a grid that covers the entire area of the check-ins.
    The boundary of grid can be extended to the nearest order

    Parameters
    ----------
    data
        list of check-ins
    cell_len_x
        length of a cell in x dimension
    cell_len_y
        length of a cell in y dimension
    boundary_order: optional
        Extend the boundary to the nearest boundary_order, 0 if not extend

    Returns
    -------
    Grid
        the grid that covers the entire area of the check-ins.
    """
    x = [c.x for c in data]
    y = [c.y for c in data]
    grid = Grid(min(x), min(y), max(x), max(y), cell_len_x, cell_len_y)
    if boundary_order != 0:
        grid.extend_boundary_to_nearest(boundary_order)

    return grid


def extend_to_nearest(x: float, k: int, lower_bound: bool) -> float:
    """
    Extend a value x to the nearest lower/upper bound k.

    Parameters
    ----------
    x: float
        value to extend
    k: int
        order to extend to
    lower_bound: bool
        find lower bound or upper bound

    Returns
    -------
    float
        extended value of x

    Examples
    --------
    >>> extend_to_nearest(-24000, 5000, True)  # = -25000
    ... extend_to_nearest(24000, 5000, True)   # = 20000
    ... extend_to_nearest(0, 5000, True)       # = 0
    ... extend_to_nearest(-24000, 5000, False) # = -20000
    ... extend_to_nearest(24000, 5000, False)  # = 25000
    """
    v_floor = float(int(math.floor(abs(x) / float(k))) * k)
    v_ceil = float(int(math.ceil(abs(x) / float(k))) * k)
    if 0 <= x:
        if lower_bound:
            return v_floor
        else:
            return v_ceil
    else:
        if lower_bound:
            return -v_ceil
        else:
            return -v_floor
