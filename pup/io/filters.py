"""
Utility functions for data filtering
"""
from pup.common.checkin import Checkin
from pup.common.rectangle import Rectangle


def check_checkin_in_area(checkin: Checkin, rect: Rectangle) -> bool:
    """
    Check whether a checkin is inside a rectangle

    Parameters
    ----------
    checkin: Checkin
        the checkin
    rect: Rectangle
        rectangle

    Returns
    -------
    bool
        whether a checkin is inside the rectangle
    """
    return rect.contain(checkin.lon, checkin.lat)
