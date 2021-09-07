"""
A rectangle object on the space
"""
from pup.common.constants import MIN_LON, MIN_LAT, MAX_LON, MAX_LAT, LOS_ANGELES_MIN_LON, LOS_ANGELES_MIN_LAT, \
    LOS_ANGELES_MAX_LON, LOS_ANGELES_MAX_LAT
from pup.common.enums import AreaCode


class Rectangle(object):
    """ Object represents a 2D rectangle.

    A 2D rectangle can be initialized by (x_min, y_min, x_max, y_max)

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

    Methods
    -------
    contain(float, float) -> bool:
        Whether a point (x, y) is contained by (i.e. inside) rectangle

    get_mid_point() -> tuple:
        Get middle point of this rectangle
    """

    def __init__(self, min_x: float, min_y: float, max_x: float, max_y: float):
        """
        Initialize a checkin with given values from datasets

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
        """
        self.min_x = float(min_x)
        self.min_y = float(min_y)
        self.max_x = float(max_x)
        self.max_y = float(max_y)

    def contain(self, x: float, y: float) -> bool:
        """
        Whether a point (x, y) is contained by (i.e. inside) rectangle

        Parameters
        ----------
        x: float
            x position
        y: float
            y position

        Returns
        -------
        bool
            Whether a point (x, y) is covered by (i.e. inside) rectangle
        """
        return self.min_x <= x <= self.max_x and self.min_y <= y <= self.max_y

    def get_mid_point(self) -> tuple:
        """
        Get middle point of this rectangle

        Returns
        -------
        tuple
            (x, y) as the middle point of this rectangle
        """
        x = self.min_x + (abs(self.max_x - self.min_x) / 2.0)
        y = self.min_y + (abs(self.max_y - self.min_y) / 2.0)
        return x, y

    def get_area(self) -> float:
        """ Get the size of the rectangle.

        Size = length_in_x_dimension * length_in_y_dimension

        Returns
        -------
        float
            the size of the rectangle
        """
        return (self.max_x - self.min_x) * (self.max_y - self.min_y)

    def __str__(self):
        return "Rectangle(min_x={min_x}, min_y={min_y}, max_x={max_x}, max_y={max_y}".format(**vars(self))


def prepare_global_rectangle():
    """
    Prepare a rectangle representing entire globe

    Returns
    -------
    Rectangle
        a rectangle representing entire globe
    """
    return Rectangle(min_x=MIN_LON, min_y=MIN_LAT, max_x=MAX_LON, max_y=MAX_LAT)


def prepare_la_rectangle():
    """
    Prepare a rectangle representing Los Angeles

    Returns
    -------
    Rectangle
        a rectangle representing Los Angeles
    """
    return Rectangle(min_x=LOS_ANGELES_MIN_LON, min_y=LOS_ANGELES_MIN_LAT,
                     max_x=LOS_ANGELES_MAX_LON, max_y=LOS_ANGELES_MAX_LAT)


def get_rectangle_for_area(area_code: AreaCode):
    """
    Get the rectangle the area

    Parameters
    ----------
    area_code: AreaCode
        area code

    Returns
    -------
    Rectangle
        the rectangle or global if area code does not match any known code
    """
    if area_code == AreaCode.LOS_ANGELES:
        return prepare_la_rectangle()
    else:
        return prepare_global_rectangle()

