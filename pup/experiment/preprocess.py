"""
Utility function for data pre-processing
"""

import logging
import math
import random
from collections import defaultdict

from pup.common.constants import EARTH_RADIUS_METERS

logger = logging.getLogger(__name__)


def limit_checkins_per_user(checkins: list, num_checkins_per_user: int, random_seed=1):
    """
    Limit for each user a maximum number of check-ins by randomly select check-ins.

    Parameters
    ----------
    checkins: list
        list of check-ins
    num_checkins_per_user: int
        max number of check-ins per user, -1 for unlimited
    random_seed: int
        a random seed for random check-ins selection
    Returns
    -------
    list
        limited check-ins
    """
    if num_checkins_per_user < 0:
        return checkins

    # convert check-in list to dict per user
    checkins_per_user = defaultdict(list)
    for c in checkins:
        checkins_per_user[c.user_id].append(c)

    # randomly select check-ins of users
    limited_checkins = list()
    for user_id, v in checkins_per_user.items():
        if len(v) <= num_checkins_per_user:
            # there are not enough check-ins, so get them all
            limited_checkins.extend(v)
        else:
            # there are more check-ins than needed, so randomly choose some of them
            random.seed(random_seed)
            limited_checkins.extend(random.sample(v, k=num_checkins_per_user))

    return limited_checkins


def checkins_to_location_coordinates(checkins: list, orig_lat: float, orig_lon: float):
    """
    Convert lat/lon coordinates of check-ins to y/x in meters. These y/x values are updated to check-ins themselves

    Parameters
    ----------
    checkins: list
        list of check-ins
    orig_lat: float
        latitude of origin
    orig_lon: float
        longitude of origin
    """
    for c in checkins:
        c.y, c.x = to_local_coordinates(orig_lat, orig_lon, c.lat, c.lon)


def to_local_coordinates(orig_lat: float, orig_lon: float, lat: float, lon: float) -> tuple:
    """
    Convert (lat, lon) coordinates to coordinates in meters with (orig_lat, orig_lon) as the origin.

    Parameters
    ----------
    orig_lat: float
        latitude of origin
    orig_lon: float
        longitude of origin
    lat: float
        latitude of check-in
    lon: float
        longitude of check-in

    Returns
    -------
    tuple
        (vertical, horizontal) coordinates in meters (corresponding to latitude, longitude)
    """
    y_meters = cal_subtended_latitude_to_meters(lat - orig_lat)
    x_meters = cal_subtended_longitude_to_meters(lon - orig_lon, orig_lat)
    return y_meters, x_meters


def cal_subtended_latitude_to_meters(subtended_lat: float) -> float:
    """
    Calculate vertical coordinate in meters of a point at (subtended_lat, 0) with origin (0, 0).
    This is a rough estimate and should only be used when the `subtended_lat` is small enough.

    Parameters
    ----------
    subtended_lat: float
        coordinate in latitude dimension

    Returns
    -------
    float
        vertical coordinate in meters
    """
    return math.pi * EARTH_RADIUS_METERS * subtended_lat / 180.0


def cal_subtended_longitude_to_meters(subtended_lon: float, lat: float) -> float:
    """
    Calculate horizontal coordinate in meters of a point at (lat, subtended_lon) with origin at (lat, 0).
    This is a rough estimate and should only be used when the `subtended_lon` is small enough.

    Parameters
    ----------
    subtended_lon: float
        coordinate in longitude dimension
    lat: float
        latitude

    Returns
    -------
    float
        horizontal coordinate in meters
    """
    cos_lat = math.cos(math.pi * lat / 180.0)
    return math.pi * EARTH_RADIUS_METERS * subtended_lon * cos_lat / 180.0

