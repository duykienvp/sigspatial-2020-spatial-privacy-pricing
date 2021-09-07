"""
Probing method (for POI and SIP and SIP-T)
"""

import logging
import multiprocessing as mp
from typing import Tuple, List

import numpy as np
from scipy.stats import rv_continuous

from pup.algorithms.probing_sip import probing_sip
from pup.algorithms.probing_poi_fast import probing_poi
from pup.common.datatypes import CheckinDataset
from pup.common.enums import ProbingAlgorithmType
from pup.common.grid import Grid
from pup.common.rectangle import Rectangle
from pup.config import Config
from pup.experiment import exp_util

logger = logging.getLogger(__name__)


def exe_probing(probing_algorithm: ProbingAlgorithmType, data: CheckinDataset, grid: Grid, budget_per_region: float
                ) -> Tuple[List[List], float, np.ndarray, float]:
    """
    Execute Probing method

    Parameters
    ----------
    probing_algorithm
        probing algorithm
    data
        check-in dataset
    grid
        the grid for experiment evaluation
    budget_per_region
        budget for buying data

    Returns
    -------
    typing.List[typing.List]
        the matrix of probability distributions of the number of users for each grid cell
    total_cost: float
        total cost spent on buying data
    costs: numpy.ndarray
        costs of each region
    exe_time: float
        average execution time
    """
    logger.info('Starting PROBING method')

    parallel = Config.parallel

    # calculate probability of being popular for each grid cell # checked for uniform
    if parallel:
        dists_of_num_users, total_cost, costs, exe_time = probing_parallel(
            probing_algorithm, data, grid, budget_per_region)

    else:
        dists_of_num_users, total_cost, costs, exe_time = probing_sequential(
            probing_algorithm, data, grid, budget_per_region)

    logger.info('Total cost = {}'.format(total_cost))

    return dists_of_num_users, total_cost, costs, exe_time


def probing_sequential(probing_algorithm: ProbingAlgorithmType,
                       data: CheckinDataset,
                       grid: Grid,
                       budget_per_region: float) -> Tuple[List[List], float, np.ndarray, float]:
    """
    Execute Probing method sequentially

    Parameters
    ----------
    probing_algorithm
        probing algorithm
    data
        check-in dataset
    grid
        the grid for experiment evaluation
    budget_per_region
        budget for buying data

    Returns
    -------
    typing.List[typing.List]
        the matrix of probability distributions of the number of users for each grid cell
    total_cost: float
        total cost spent on buying data
    costs: numpy.ndarray
        costs of each region
    exe_time: float
        average execution time
    """
    total_cost = 0
    dists_of_num_users = list()
    grid_shape = grid.get_shape()
    costs = np.zeros(grid_shape)
    exe_times = list()

    # True_counts here is only for debugging
    true_counts = exp_util.count_data(data, grid)  # checked for uniform

    for x_idx in range(grid_shape[0]):
        dists_of_num_users_y = list()
        for y_idx in range(grid_shape[1]):
            logger.info('---------region ({}, {})---------'.format(x_idx, y_idx))
            # Running for each cell separately
            # TEST_NOTE: testing single region. Note regions: (3, 6), (3, 7), (4, 6), (4, 8-10), (5, 5-7)
            # Sometimes switching condition is not right.
            # If decide to open, buy until having maximum popularity_thres points left? NO, if threshold is small
            if True:
                # (5, 8), (5, 9), (5, 10) = 382, 57, 36
                # if x_idx != 4 or y_idx != 7:
                if x_idx != 3 or y_idx != 3:
                # if x_idx != 6 or y_idx != 8:
                    dists_of_num_users_y.append(None)
                    continue

            logger.info('True count ({}, {}) = {}'.format(x_idx, y_idx, true_counts[x_idx, y_idx]))

            region = grid.find_cell_boundary(x_idx, y_idx)

            x, y, rv, cost, exe_time = probing_region(
                probing_algorithm,
                x_idx,
                y_idx,
                data,
                grid,
                budget_per_region,
                region)

            dists_of_num_users_y.append(rv)
            logger.info('({}, {}) Output mean={}, std={}'.format(x_idx, y_idx, rv.mean(), rv.std()))

            costs[x, y] = cost
            total_cost += cost
            logger.info('Current total cost = {}'.format(total_cost))

            logger.info('True count ({}, {}) = {}'.format(x_idx, y_idx, true_counts[x_idx, y_idx]))

            exe_times.append(exe_time)

        dists_of_num_users.append(dists_of_num_users_y)

    avg_exe_time = np.average(exe_times)
    return dists_of_num_users, total_cost, costs, avg_exe_time


def probing_parallel(probing_algorithm: ProbingAlgorithmType, data: CheckinDataset, grid: Grid, budget_per_region: float
                     ) -> Tuple[List[List], float, np.ndarray, float]:
    """
    Execute Probing method in parallel

    Parameters
    ----------
    probing_algorithm
        probing algorithm
    data
        check-in dataset
    grid
        the grid for experiment evaluation
    budget_per_region
        budget for buying data

    Returns
    -------
    typing.List[typing.List]
        the matrix of probability distributions of the number of users for each grid cell
    total_cost: float
        total cost spent on buying data
    costs: numpy.ndarray
        costs of each region
    exe_time: float
        average execution time
    """
    total_cost = 0
    dists_of_num_users = list()
    grid_shape = grid.get_shape()
    costs = np.zeros(grid_shape)
    exe_times = list()

    # Executing in parallel
    num_cpus = Config.parallel_num_cpu
    if num_cpus <= 0:
        num_cpus = mp.cpu_count()
    with mp.Pool(processes=num_cpus) as pool:
        results = pool.starmap(
            probing_region,
            [(probing_algorithm, x_idx, y_idx, data, grid, budget_per_region,
              grid.find_cell_boundary(x_idx, y_idx))
             for x_idx in range(grid_shape[0]) for y_idx in range(grid_shape[1])])

    # Collect results
    for x_idx in range(grid_shape[0]):
        dists_of_num_users_y = list()
        for y_idx in range(grid_shape[1]):
            for x, y, rv, cost, exe_time in results:
                if x == x_idx and y == y_idx:
                    dists_of_num_users_y.append(rv)
                    costs[x, y] = cost
                    total_cost += cost
                    exe_times.append(exe_time)
                    break
        dists_of_num_users.append(dists_of_num_users_y)

    avg_exe_time = np.average(exe_times)
    return dists_of_num_users, total_cost, costs, avg_exe_time


def probing_region(probing_algorithm: ProbingAlgorithmType,
                   x_idx: int,
                   y_idx: int,
                   data: CheckinDataset,
                   grid: Grid,
                   budget: float,
                   region: Rectangle) -> Tuple[int, int, rv_continuous, float, float]:
    """ Running the probing algorithm for a region

    Parameters
    ----------
    probing_algorithm
        probing algorithm
    x_idx
        x index of the region, used for parallelism only
    y_idx
        y index of the region, used for parallelism only
    data
        check-in dataset
    grid
        the grid for experiment evaluation
    budget
        budget for buying data
    region
        the region of interest

    Returns
    -------
    x_idx: int
        x index of the region, used for parallelism only
    y_idx: int
        y index of the region, used for parallelism only
    rv: rv_continuous
        the probability distribution of the number of data points inside a region
    cost: float
        total cost spent on buying data
    exe_time: float
        execution time
    """
    if probing_algorithm == ProbingAlgorithmType.POI:
        return probing_poi(x_idx, y_idx, data, grid, budget, region)
    elif probing_algorithm == ProbingAlgorithmType.SIP:
        return probing_sip(x_idx, y_idx, data, grid, budget, region)

    else:
        logger.info("Probing algorithm {} not implemented".format(probing_algorithm.name))
