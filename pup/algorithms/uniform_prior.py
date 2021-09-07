"""
Uniform Prior method
"""
import logging
from collections import defaultdict
from typing import List, Tuple
import time

import numpy as np
from scipy.stats import uniform

from pup.algorithms.util import filter_probs_by_threshold, cal_prod_dist_num_user
from pup.common.datatypes import CheckinDataset
from pup.common.enums import FinalProbsFilterType
from pup.common.grid import Grid
from pup.common.noisycheckin import NoisyCheckin

logger = logging.getLogger(__name__)


def exe_uniform_prior(data: CheckinDataset, grid: Grid) -> Tuple[List[List], float, np.ndarray, float]:
    """
    Execute Uniform Prior method

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
    logger.info('Starting Uniform Prior method')

    # START UNIFORM ---------------------
    # prepare noisy data
    noisy_data = defaultdict(defaultdict)
    scale_x = grid.max_x - grid.min_x
    scale_y = grid.max_y - grid.min_y
    for user, checkins in data.items():
        for c in checkins.values():
            noise_level = float('inf')
            rv_x = uniform(loc=grid.min_x, scale=scale_x)
            rv_y = uniform(loc=grid.min_y, scale=scale_y)
            noisy_c = NoisyCheckin(c, noise_level, rv_x, rv_y)

            noisy_data[user][noisy_c.c_id] = noisy_c

    costs = np.zeros(grid.get_shape())
    total_cost = 0
    logger.info('Prepare {} noisy data point with uniformly random variables'.format(len(noisy_data)))

    # Run experiment on the entire grid. One can run on single region by using 1x1 grid

    # Calculate the probability distributions of the number of each grid cell
    # Because this is uniform, we will have to keep all of those
    dists_of_num_users = cal_prob_dists_num_users_for_grid(grid, noisy_data, FinalProbsFilterType.ZERO)

    exe_time = time.time() - s_time
    return dists_of_num_users, total_cost, costs, exe_time


def cal_prob_dists_num_users_for_grid(grid: Grid, noisy_data: CheckinDataset,
                                      final_probs_filter_type: FinalProbsFilterType) -> List[List]:
    """ Calculate probability distributions of the number of users for each grid cell

    Parameters
    ----------
    grid
        the grid
    noisy_data
        the set of noisy data
    final_probs_filter_type
        the type of filter to remove probabilities after buying a set of data

    Returns
    -------
    typing.List[typing.List]
        the matrix of probability distributions of the number of users for each grid cell
    """
    # each prob grid for each data point  # checked for uniform
    prob_grids = defaultdict(defaultdict)
    for user, checkins in noisy_data.items():
        c: NoisyCheckin
        for c in checkins.values():
            prob_grids[user][c.c_id] = c.cal_prob_grid(grid)
    logger.info('Calculated prob grids for each data point')
    # each grid cell has its own approximate normal distribution for the sum of probabilities # checked for uniform

    grid_shape = grid.get_shape()
    dists_of_num_users = list()
    for x_idx in range(grid_shape[0]):
        dists_of_num_users_y = list()
        for y_idx in range(grid_shape[1]):
            # this is a grid cell (x_idx, y_idx)
            # get probabilities of being inside of each data point

            uniform_prob = grid.find_cell_boundary(x_idx, y_idx).get_area() / grid.get_area()
            # prob_threshold = util.get_prob_threshold_from_type(final_probs_filter_type, uniform_prob)

            # inside_probs = list()
            inside_probs = defaultdict(defaultdict)
            for user, user_prob_grids in prob_grids.items():
                user_prob_grid: np.ndarray
                for c_id, user_prob_grid in user_prob_grids.items():
                    inside_probs[user][c_id] = user_prob_grid[x_idx, y_idx]

            filtered_probs = filter_probs_by_threshold(final_probs_filter_type, inside_probs, uniform_prob)
            # count_better_than_uniform = sum([len(ps) for user, ps in filtered_probs.items()])
            # logger.info('({}, {}) count_better_than_uniform={}'.format(x_idx, y_idx, count_better_than_uniform))

            rv = cal_prod_dist_num_user(filtered_probs)

            dists_of_num_users_y.append(rv)

        dists_of_num_users.append(dists_of_num_users_y)
    logger.info('Calculated probability distributions of the number of users for each grid cell')
    return dists_of_num_users
