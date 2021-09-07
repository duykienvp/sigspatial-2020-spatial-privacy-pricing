"""
Utility functions for experiments
"""
import logging
from collections import defaultdict

import numpy as np

from pup.common.datatypes import CheckinList, CheckinDataset
from pup.common.grid import Grid

logging.getLogger(__name__)


def to_mapping(checkins: CheckinList) -> CheckinDataset:
    """
    Convert list of check-ins to dictionary of list of check-ins of each user

    Parameters
    ----------
    checkins
        list of check-ins

    Returns
    -------
    CheckinDataset
        dictionary of dict of check-ins of each user
    """
    result = defaultdict(defaultdict)

    for c in checkins:
        result[c.user_id][c.c_id] = c

    return result


def count_data(data: CheckinDataset, grid: Grid) -> np.ndarray:
    """ Calculate the number of check-ins inside each grid cell

    Parameters
    ----------
    data
        check-in data
    grid
        the grid

    Returns
    -------
    np.ndarray
        count for each cell of the grid
    """
    counts = np.zeros(grid.get_shape())
    for user, checkins in data.items():
        for c in checkins.values():
            x_idx, y_idx = grid.find_grid_index(c.x, c.y)
            counts[x_idx, y_idx] += 1

    return counts


def cal_num_data_points(data: dict) -> int:
    """ Calculate the number of data points in a dataset

    Parameters
    ----------
    data
        dataset

    Returns
    -------
    int
        the number of data points in a dataset
    """
    return sum([len(data_u) for u, data_u in data.items()])
