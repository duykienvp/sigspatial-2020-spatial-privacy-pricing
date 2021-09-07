"""
Fixed Maximum Cost (FMC) baseline
"""

import logging
from collections import defaultdict
from typing import Tuple, List
import time

import numpy as np

from pup.algorithms import privacy_helper
from pup.algorithms.uniform_prior import cal_prob_dists_num_users_for_grid
from pup.algorithms.util import get_linear_profit_fixed_cost
from pup.common.datatypes import CheckinDataset
from pup.common.enums import MethodType
from pup.common.grid import Grid
from pup.config import Config
from pup.experiment import exp_util
from pup.io import dataio

logger = logging.getLogger(__name__)


def exe_fixed_maximum_cost(data: CheckinDataset, grid: Grid) -> Tuple[List[List], float, np.ndarray, float]:
    """
    Execute Fixed Maximum Cost method

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
    logger.info('Starting FIXED MAXIMUM COST method')
    # Load config

    price_from_noise_rate = Config.price_from_noise_func_rate
    std_from_noise_initial_value = Config.standard_deviation_from_noise_func_initial_value
    std_from_noise_rate = Config.standard_deviation_from_noise_func_rate
    final_probs_filter_type = Config.final_probs_filter_type

    budget_per_region = get_fmc_budget()

    # START FMC ---------------------
    logger.info('Budget = {}'.format(budget_per_region))

    noisy_data, remain_budget_per_region = buy_data_with_budget(
        budget_per_region, data,
        price_from_noise_rate, std_from_noise_initial_value, std_from_noise_rate)
    logger.info('Prepare {} noisy data point with normal random variables'.format(len(noisy_data)))

    num_regions = np.prod(grid.get_shape())
    cost = budget_per_region - remain_budget_per_region
    costs = np.zeros(grid.get_shape())
    costs.fill(cost)
    total_cost = cost * num_regions
    logger.info('Total cost spent on buying data = {}'.format(total_cost))

    # Run experiment on the entire grid. One can run on single region by using 1x1 grid

    # Calculate the probability distributions of the number of each grid cell
    dists_of_num_users = cal_prob_dists_num_users_for_grid(grid, noisy_data, final_probs_filter_type)

    exe_time = time.time() - s_time
    return dists_of_num_users, total_cost, costs, exe_time

    # END FMC ---------------------


def get_fmc_budget() -> float:
    """ Get budget for FMC

    - First, get based on given percentage
    - Second, get based on probing costs if percentage is not given
    - Third, get based on a fixed budget if others are not available

    Returns
    -------
    float
        budget
    """
    fmc_budget_from_cost_percentage = Config.fmc_budget_from_cost_percentage

    if fmc_budget_from_cost_percentage <= 0:
        # we will not get budget from percentage of the fixed cost
        fmc_budget_from_probing = Config.fmc_budget_from_probing
        if fmc_budget_from_probing:
            # we get budget from costs of SIP
            costs = dataio.read_costs(MethodType.PROBING)
            budget = int(np.average(costs)) + 1
        else:
            # we used a fixed budget
            budget = Config.budget  # prepare budget
    else:
        # get budget from the percentage of the fixed cost
        budget = get_linear_profit_fixed_cost() * fmc_budget_from_cost_percentage / 100.0
    return budget


def buy_data_with_budget(budget: float, data: CheckinDataset,
                         price_from_noise_rate: float,
                         std_from_noise_initial_value: float,
                         std_from_noise_rate: float) -> Tuple[CheckinDataset, float]:
    """ Buy data points with a given total budget.

    Each data point would be given the same amount of budget.
    For a particular data point, the budget may be more than enough to buy it without perturbation.
    So there can be some budget left. This budget is not used for other data points.

    Parameters
    ----------
    budget
        maximum budget
    data
        the dataset to buy data from
    price_from_noise_rate
        rate of price from noise exponential function
    std_from_noise_initial_value
        initial value of standard deviation from noise exponential function, i.e. when input values is approx 0
    std_from_noise_rate
        rate of standard deviation from noise exponential function

    Returns
    -------
    noisy_data: CheckinDataset
        noisy data bought
    remain_budget: float
        remain budget
    """
    # calculate the price to pay for each data point
    num_data_points = exp_util.cal_num_data_points(data)
    price_per_data_point = budget / float(num_data_points)
    logger.info('Price per data point = {}'.format(price_per_data_point))

    # buy noisy data
    remain_budget = 0
    noisy_data = defaultdict(defaultdict)
    for user, checkins in data.items():
        for c_id, c in checkins.items():
            noisy_c = privacy_helper.buy_data_at_price(
                c, price_per_data_point, price_from_noise_rate, std_from_noise_initial_value, std_from_noise_rate)

            noisy_data[user][c_id] = noisy_c

            if c.combined_privacy_value < price_per_data_point:
                remain_budget += price_per_data_point - c.combined_privacy_value

    logger.info('Remain budget for region = {}'.format(remain_budget))
    return noisy_data, remain_budget


