"""
Making decisions in linear profit evaluation model
"""
import logging
from typing import List, Tuple

import numpy as np

from pup.algorithms.util import get_linear_profit_fixed_cost
from pup.common.enums import FinalDecisionType
from pup.common.grid import Grid
from pup.config import Config
from pup.evaluation.util import calculate_result_summary, make_max_expected_utility_decision

logger = logging.getLogger(__name__)


def eval_predictions(dists_of_num_users: List[List],
                     grid: Grid,
                     true_counts: np.ndarray,
                     costs: np.ndarray):
    """ Evaluate the predicted probability distributions of the number of users for each grid cell

    Parameters
    ----------
    dists_of_num_users
        the predicted probability distributions of the number of users for each grid cell
    grid
        the grid for experiment evaluation
    true_counts
        true (i.e. ground truth) counts
    costs
        cost spent on buying data on each region

    Returns
    -------
    total_realized_payoff_value: float
        total realized payoff value
    total_adjusted_realized_payoff_value: float
        total realized payoff value adjusted for the cost
    average_adjusted_realized_payoff_value: float
        average realized payoff value per grid cell adjusted for the cost
    median_adjusted_realized_payoff_value: float
        average realized payoff value per grid cell adjusted for the cost
    true_positives: float
        true positives count
    true_negatives: float
        true negatives count
    false_positives: float
        false positives count
    false_negatives: float
        false negatives count
    precision: float
        precision
    recall: float
        recall
    f1_score: float
        f1 score
    """
    profit_per_user = Config.linear_profit_profit_per_user
    fixed_cost = get_linear_profit_fixed_cost()

    # calculate probability of being popular for each grid cell # checked for uniform
    expected_num_users = cal_expected_num_users_grid(grid, dists_of_num_users)

    # Make decisions
    decision_grid = make_opening_decision_multiple_regions(expected_num_users, profit_per_user, fixed_cost)

    # Evaluate decisions
    total_payoff, total_adjusted_payoff, average_adjusted_payoff, median_adjusted_payoff = \
        evaluate_decisions_with_utility(true_counts, decision_grid, profit_per_user, fixed_cost, costs)

    true_positives, true_negatives, false_positives, false_negatives, precision, recall, f1_score = \
        evaluate_decisions_confusion_matrix(true_counts, decision_grid, profit_per_user, fixed_cost)

    return total_payoff, total_adjusted_payoff, average_adjusted_payoff, median_adjusted_payoff, \
           true_positives, true_negatives, false_positives, false_negatives, precision, recall, f1_score


def make_opening_decision_multiple_regions(
        expected_num_users: np.ndarray,
        profit_per_user: float,
        fixed_cost: float) -> np.ndarray:
    """

    Parameters
    ----------
    expected_num_users
        the array of expected number of users
    profit_per_user
        profit per user
    fixed_cost
        the fixed cost

    Returns
    -------
    numpy.ndarray
        decision array
    """
    # Decide based on maximum expected utility and payoff matrix: # checked for uniform
    # action = 0 : open
    # action = 1 : cancel
    grid_shape = expected_num_users.shape

    decision_grid = np.zeros(grid_shape, dtype=np.int)
    for x_idx in range(grid_shape[0]):
        for y_idx in range(grid_shape[1]):
            decision, expected_value = make_opening_decision_single_region(
                expected_num_users[x_idx, y_idx], profit_per_user, fixed_cost)
            decision_grid[x_idx, y_idx] = decision

    logger.info('Decided based on linear profit model')

    return decision_grid


def make_opening_decision_single_region(
        expected_num_users: float,
        profit_per_user: float,
        fixed_cost: float) -> Tuple[int, float]:
    """ Make decision for one region

    Parameters
    ----------
    expected_num_users
        the expected number of users
    profit_per_user
        profit per user
    fixed_cost
        the fixed cost

    Returns
    -------
    decision: int
        action
    expected_value: float
        expected value
    """
    # action = 0 : open
    # action = 1 : cancel
    expected_value_open = expected_num_users * profit_per_user - fixed_cost
    expected_value_cancel = 0

    decision, expected_value = make_max_expected_utility_decision(expected_value_open, expected_value_cancel)

    return decision, expected_value


def evaluate_decisions_with_utility(true_counts: np.ndarray, decision_grid: np.ndarray,
                                    profit_per_user: float, fixed_cost: float,
                                    costs: np.ndarray) -> Tuple[float, float, float, float]:
    """ Evaluate decisions

    Parameters
    ----------
    true_counts
        true (i.e. ground truth) counts
    decision_grid
        decision array for grid cells
    profit_per_user
        profit per user
    fixed_cost
        the fixed cost
    costs
        cost spent on buying data on each region

    Returns
    -------
    total_realized_payoff_value: float
        total realized payoff value
    total_adjusted_realized_payoff_value: float
        total realized payoff value adjusted for the cost
    average_adjusted_realized_payoff_value: float
        average realized payoff value per grid cell adjusted for the cost
    median_adjusted_realized_payoff_value: float
        median realized payoff value per grid cell adjusted for the cost
    """
    grid_shape = true_counts.shape

    results_list = list()

    # calculate realized payoff grid
    realized_payoff_grid = np.zeros(grid_shape)
    for x_idx in range(grid_shape[0]):
        for y_idx in range(grid_shape[1]):
            action = decision_grid[x_idx, y_idx]
            realized_value = 0  # CANCEL action
            if action == FinalDecisionType.OPEN.value:
                realized_value = true_counts[x_idx, y_idx] * profit_per_user - fixed_cost

            realized_payoff_grid[x_idx, y_idx] = realized_value

            results_list.append((true_counts[x_idx, y_idx], x_idx, y_idx, realized_value, action))

    for t in sorted(results_list):
        print(t)

    logger.info('Calculated realized payoff grid for linear profit model')

    total_adjusted_realized_payoff_value, total_realized_payoff_value, \
        average_adjusted_realized_payoff_value, median_adjusted_realized_payoff_value = \
        calculate_result_summary(realized_payoff_grid, costs)

    return total_adjusted_realized_payoff_value, total_realized_payoff_value, \
           average_adjusted_realized_payoff_value, median_adjusted_realized_payoff_value


def evaluate_decisions_confusion_matrix(
        true_counts: np.ndarray,
        decision_grid: np.ndarray,
        profit_per_user: float,
        fixed_cost: float) -> Tuple[float, float, float, float, float, float, float]:
    """ Evaluate decisions with confusion matrix

    Handle special cases:
    https://github.com/dice-group/gerbil/wiki/Precision,-Recall-and-F1-measure

    In some rare cases, the calculation of Precision or Recall can cause a division by 0.
    Regarding the precision, this can happen if there are no results inside the answer of an annotator and,
    thus, the true as well as the false positives are 0.

    For these special cases, we have defined that if the true positives, false positives and false negatives are all 0,
    the precision, recall and F1-measure are 1.
    This might occur in cases in which the gold standard contains a document without any annotations and the annotator
    (correctly) returns no annotations.

    If true positives are 0 and one of the two other counters is >= 0, the precision, recall and F1-measure are 0.

    Parameters
    ----------
    true_counts
        true (i.e. ground truth) counts
    decision_grid
        decision array for grid cells
    profit_per_user
        profit per user
    fixed_cost
        the fixed cost

    Returns
    -------
    true_positives: float
        true positives count
    true_negatives: float
        true negatives count
    false_positives: float
        false positives count
    false_negatives: float
        false negatives count
    precision: float
        precision
    recall: float
        recall
    f1_score: float
        f1 score
    """
    grid_shape = true_counts.shape

    true_positives = true_negatives = false_positives = false_negatives = 0

    for x_idx in range(grid_shape[0]):
        for y_idx in range(grid_shape[1]):
            # find the correct action
            correct_action = FinalDecisionType.CANCEL
            if true_counts[x_idx, y_idx] * profit_per_user - fixed_cost > 0:
                correct_action = FinalDecisionType.OPEN

            # check if the action taken is the correct action, default is already NO
            action = decision_grid[x_idx, y_idx]
            if action == FinalDecisionType.OPEN.value:
                # positive case
                if action == correct_action.value:
                    true_positives += 1
                else:
                    false_positives += 1
            elif action == FinalDecisionType.CANCEL.value:
                # negative case
                if action == correct_action.value:
                    true_negatives += 1
                else:
                    false_negatives += 1

    if true_positives == 0:
        # handle special case
        if false_negatives == 0 and false_negatives == 0:
            precision = recall = f1_score = 1.0
        else:
            precision = recall = f1_score = 0.0
    else:
        precision = true_positives / (true_positives + false_positives)
        recall = true_positives / (true_positives + false_negatives)
        f1_score = 2 * precision * recall / (precision + recall)

    return true_positives, true_negatives, false_positives, false_negatives, precision, recall, f1_score


def cal_expected_num_users_grid(grid: Grid, dists_of_num_users: List[List]) -> np.ndarray:
    """ Calculate the expected number of users of all grid cells

    Parameters
    ----------
    grid
        the grid
    dists_of_num_users
        matrix of prob distributions of number of users for each grid cell

    Returns
    -------
    numpy.ndarray
        the array of expected number of users
    """
    grid_shape = grid.get_shape()
    expected_num_users = np.zeros(grid_shape)
    for x_idx in range(grid_shape[0]):
        for y_idx in range(grid_shape[1]):
            rv = dists_of_num_users[x_idx][y_idx]

            expected_num_users[x_idx, y_idx] = rv.mean()
    logger.info('Calculated the expected number of users of all grid cells')
    return expected_num_users
