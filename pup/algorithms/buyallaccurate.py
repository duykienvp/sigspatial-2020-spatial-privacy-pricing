"""
The Buy-All-Accurate method: Buy all data points at their privacy valuation price point
which means we will have accurate data points.
This method is just to see how much it would cost us
if we simple buy all data points at their most expensive price point.
"""
import logging
from typing import List, Tuple
import time

import numpy as np

from pup.algorithms import util
from pup.common.datatypes import CheckinDataset
from pup.common.grid import Grid
from pup.experiment import exp_util

logger = logging.getLogger(__name__)


def exe_buy_all_accurate(data: CheckinDataset, grid: Grid) -> Tuple[List[List], float, np.ndarray, float]:
    """ Buy all data at without noise

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
    s_time = time.time()
    logger.info('Starting Buy All accurate method')

    # START Buy All Accurate ---------------------

    cost = 0
    for user, checkins in data.items():
        for c in checkins.values():
            cost += c.combined_privacy_value

    costs = np.zeros(grid.get_shape())
    costs.fill(cost)
    total_cost = cost * np.prod(grid.get_shape())
    logger.info('Total cost = {}'.format(total_cost))

    true_counts = exp_util.count_data(data, grid)  # checked for uniform

    # calculate probability of being popular for each grid cell
    dists_of_num_users = cal_ground_truth_prob_dists_num_users_for_grid(grid, true_counts)

    # END Buy All Accurate ---------------------
    exe_time = time.time() - s_time
    return dists_of_num_users, total_cost, costs, exe_time


def cal_ground_truth_prob_dists_num_users_for_grid(grid: Grid, true_counts: np.ndarray) -> List[List]:
    """ Calculate ground truth probability distributions of the number of users for each grid cell

    Parameters
    ----------
    grid
        the grid for experiment evaluation
    true_counts
        true counts of grid cells

    Returns
    -------
    typing.List[typing.List]
        the matrix of probability distributions of the number of users for each grid cell
    """
    grid_shape = grid.get_shape()

    dists_of_num_users = list()
    for x_idx in range(grid_shape[0]):
        dists_of_num_users_y = list()
        for y_idx in range(grid_shape[1]):
            # this is a grid cell (x_idx, y_idx)
            # get probabilities of being inside of each data point

            rv = util.create_dirac_delta_dist(true_counts[x_idx, y_idx])

            dists_of_num_users_y.append(rv)

        dists_of_num_users.append(dists_of_num_users_y)
    logger.info('Calculated probability distributions of the number of users for each grid cell')
    return dists_of_num_users
