import logging
from typing import Tuple

import numpy as np

from pup.common.enums import FinalDecisionType

logger = logging.getLogger(__name__)


def make_max_expected_utility_decision(expected_value_open: float, expected_value_cancel: float) -> Tuple[int, float]:
    """ Make decision based on Maximum Expected Utility principle

    Parameters
    ----------
    expected_value_open
        expected utility of open action
    expected_value_cancel
        expected utility of cancel action

    Returns
    -------
    decision: int
        int value of ActionType or ActionType
    expected_value: float
        expected value of the decision
    """
    if expected_value_cancel < expected_value_open:
        decision = FinalDecisionType.OPEN.value
        expected_value = expected_value_open
    else:
        decision = FinalDecisionType.CANCEL.value
        expected_value = expected_value_cancel
    return decision, expected_value


def calculate_result_summary(realized_payoff_grid: np.ndarray, costs: np.ndarray) -> Tuple[float, float, float, float]:
    """ Calculate result summary for decisions over grid

    Parameters
    ----------
    realized_payoff_grid
        realized payoff grid
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
    adjusted_realized_payoff_grid = realized_payoff_grid - costs

    total_realized_payoff_value = realized_payoff_grid.sum()

    total_adjusted_realized_payoff_value = adjusted_realized_payoff_grid.sum()

    average_adjusted_realized_payoff_value = np.average(adjusted_realized_payoff_grid)

    median_adjusted_realized_payoff_value = np.median(adjusted_realized_payoff_grid)

    return total_adjusted_realized_payoff_value, total_realized_payoff_value, \
           average_adjusted_realized_payoff_value, median_adjusted_realized_payoff_value
