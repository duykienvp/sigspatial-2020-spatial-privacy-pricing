"""
The ground truth (or Oracle) baseline.
"""
from typing import Tuple, List

import numpy as np

from pup.algorithms import buyallaccurate
from pup.common.datatypes import CheckinDataset
from pup.common.grid import Grid


def exe_ground_truth(data: CheckinDataset, grid: Grid) -> Tuple[List[List], float, np.ndarray, float]:
    """ Calculate the ground truth: when the true counts are available not NO cost so all actions are the right actions

    Parameters
    ----------
    data
        check-in dataset
    grid
        the grid for experiment evaluation

    Returns
    -------
    typing.List[typing.List]
        the matrix of probability distributions of the number of users for each grid cell
    total_cost: float
        total cost spent on buying data
    costs: numpy.ndarray
        costs of each region
    exe_time: float
        execution time
    """
    # START Ground Truth ---------------------
    dists_of_num_users, total_cost, costs, exe_time = buyallaccurate.exe_buy_all_accurate(data, grid)
    costs = np.zeros(grid.get_shape())
    total_cost = 0

    # END Ground Truth ---------------------
    return dists_of_num_users, total_cost, costs, exe_time
