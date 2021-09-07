import logging

import numpy as np

from pup.algorithms import uniform_prior, buyallaccurate, fixed_maximum_cost, probing, privacy_helper, groundtruth
from pup.evaluation import accuracy_evaluator, decisionmaking_linear_profit
from pup.common.enums import MethodType
from pup.config import Config
from pup.common.grid import create_grid_for_data
from pup.experiment import exp_util
from pup.io import dataio

logger = logging.getLogger(__name__)


def collect_data():
    """
    Collect data from raw data file that match some requirements and save to an output file
    """
    data = dataio.load_data_by_config(Config.checkin_selection_random_seed)

    output_file = Config.output_file
    dataio.save_checkins(data, output_file)

    logger.info('Saved {} check-ins to {}'.format(len(data), output_file))


def execute_method(method_type: MethodType, eval_only: bool):
    """ Run a method

    Parameters
    ----------
    method_type
        method to be executed
    eval_only:
        whether we only need to evaluate results without purchases, i.e. already had purchased data
    """
    grid_cell_len_x = Config.eval_grid_cell_len_x  # length of a cell in x dimension
    grid_cell_len_y = Config.eval_grid_cell_len_y  # length of a cell in y dimension
    grid_boundary_order = Config.eval_grid_boundary_order  # extend grid boundary to the nearest boundary_order

    # get threshold for being popular
    opening_threshold = Config.eval_opening_threshold
    if opening_threshold is None:
        opening_threshold = Config.linear_profit_fixed_cost / Config.linear_profit_profit_per_user

    # load data, also applied some filters based on configuration
    data_list = dataio.load_data_by_config(Config.checkin_selection_random_seed)

    # prepare grid for evaluation
    grid = create_grid_for_data(data_list, grid_cell_len_x, grid_cell_len_y, grid_boundary_order)

    # convert data from list to mapping
    data = exp_util.to_mapping(data_list)

    num_users = len(data)
    num_data_points = exp_util.cal_num_data_points(data)
    num_regions = np.prod(grid.get_shape())

    logger.info('Method: {}'.format(method_type))
    logger.info('Num users = {}'.format(num_users))
    logger.info('Num data points = {}'.format(num_data_points))
    logger.info('Grid shape = {}'.format(grid.get_shape()))
    logger.info('Num regions = {}'.format(num_regions))

    true_counts = exp_util.count_data(data, grid)  # checked for uniform
    for x_idx in range(grid.get_shape()[0]):
        for y_idx in range(grid.get_shape()[1]):
            if true_counts[x_idx, y_idx] > opening_threshold:
                logger.info('True count ({}, {}) = {} POPULAR'.format(x_idx, y_idx, true_counts[x_idx, y_idx]))
    logger.info('true_counts:\n{}'.format(true_counts))
    # assert (float(np.sum(true_counts) == float(len(data)))), 'Total counts should be equal to the number of users'

    # prepare privacy values
    privacy_helper.prepare_combined_privacy_values_for_data_points(data)

    dists_of_num_users = None
    total_cost = None
    costs = None
    exe_time = None

    if eval_only and (method_type == MethodType.PROBING or method_type == MethodType.FIXED_MAXIMUM_COST):
        # We do not calculate these values
        dists_of_num_users = dataio.read_distributions(method_type)
        costs = dataio.read_costs(method_type)
        total_cost = np.sum(costs)
        exe_time = 0
    else:
        # We need to calculate these values
        if method_type == MethodType.GROUND_TRUTH:
            dists_of_num_users, total_cost, costs, exe_time = groundtruth.exe_ground_truth(data, grid)

        elif method_type == MethodType.BUY_ALL_ACCURATE:
            dists_of_num_users, total_cost, costs, exe_time = buyallaccurate.exe_buy_all_accurate(data, grid)

        elif method_type == MethodType.UNIFORM_PRIOR:
            dists_of_num_users, total_cost, costs, exe_time = uniform_prior.exe_uniform_prior(data, grid)

        elif method_type == MethodType.FIXED_MAXIMUM_COST:
            dists_of_num_users, total_cost, costs, exe_time = fixed_maximum_cost.exe_fixed_maximum_cost(data, grid)

        elif method_type == MethodType.PROBING:
            probing_algorithm = Config.probing_algorithm
            budget = Config.budget  # prepare budget

            dists_of_num_users, total_cost, costs, exe_time = probing.exe_probing(probing_algorithm, data, grid, budget)
        else:
            logger.error('Invalid method type: {}'.format(method_type))

    if dists_of_num_users is not None:
        if not eval_only:
            dataio.write_distributions(method_type, dists_of_num_users)
            dataio.write_costs(method_type, costs)

        # pm_total_payoff, pm_total_adjusted_payoff, pm_avg_adjusted_payoff = \
        #     decisionmaking_payoffmatrix.eval_predictions(dists_of_num_users, grid, true_counts, total_cost)

        total_payoff, total_adjusted_payoff, avg_adjusted_payoff, median_adjusted_payoff, \
            true_positives, true_negatives, false_positives, false_negatives, precision, recall, f1_score = \
            decisionmaking_linear_profit.eval_predictions(dists_of_num_users, grid, true_counts, costs)

        logger.info('Total payoff = {}'.format(total_payoff))
        logger.info('Total adjusted payoff = {}'.format(total_adjusted_payoff))
        logger.info('Average adjusted payoff = {}'.format(avg_adjusted_payoff))
        logger.info('Median adjusted payoffalue = {}'.format(median_adjusted_payoff))

        logger.info('precision = {}'.format(precision))
        logger.info('recall = {}'.format(recall))
        logger.info('f1_score = {}'.format(f1_score))

        rmse = accuracy_evaluator.eval_rmse_means_vs_true_counts(dists_of_num_users, true_counts)
        logger.info('rmse = {}'.format(rmse))

        logger.info('exe_time = {}'.format(exe_time))

        dataio.write_output_evaluations(
            method_type, num_users, num_data_points, num_regions,
            total_cost,
            true_positives, true_negatives, false_positives, false_negatives, precision, recall, f1_score,
            total_payoff, total_adjusted_payoff, avg_adjusted_payoff, median_adjusted_payoff,
            rmse, exe_time)
