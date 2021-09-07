"""
Probing using PoI algorithm in "The Price of Information in Combinatorial Optimization"
"""

import logging
from collections import defaultdict
from typing import Tuple

from scipy.stats import rv_continuous

from pup.algorithms.privacy_helper import buy_data_at_price
from pup.algorithms.util import cal_prod_dist_num_user, get_prob_threshold_from_type, get_linear_profit_fixed_cost
from pup.common.datatypes import CheckinDataset
from pup.common.grid import Grid
from pup.common.noisycheckin import NoisyCheckin
from pup.common.rectangle import Rectangle
from pup.config import Config

logger = logging.getLogger(__name__)


def probing_poi(
        x_idx: int,
        y_idx: int,
        data: CheckinDataset,
        grid: Grid,
        budget: float,
        region: Rectangle) -> Tuple[int, int, rv_continuous, float]:
    """ Running the faster PoI probing algorithm for a region

    Parameters
    ----------
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
    """
    # Load config for this algorithm
    price_from_noise_rate = Config.price_from_noise_func_rate
    std_from_noise_initial_value = Config.standard_deviation_from_noise_func_initial_value
    std_from_noise_rate = Config.standard_deviation_from_noise_func_rate
    # final_probs_filter_type = Config.final_probs_filter_type

    profit_per_user = Config.linear_profit_profit_per_user
    opening_threshold = Config.eval_opening_threshold
    fixed_cost = get_linear_profit_fixed_cost()

    cost = 0
    rv = None

    # Consider each data point equally regardless of user
    data_points = dict()
    for _, checkins in data.items():
        for checkin in checkins.values():
            data_points[checkin.c_id] = checkin

    # Calculate the uniform probability
    extended_area_size = float(grid.get_area())
    uniform_prob = region.get_area() / extended_area_size
    # prob_filter_threshold = get_prob_threshold_from_type(final_probs_filter_type, uniform_prob) # min prob to consider
    # logger.info('uniform_prob={}'.format(uniform_prob))

    # Calculate grade of each data points
    grades = dict()
    num_positive_grades = 0
    for checkin in data_points.values():
        if profit_per_user * uniform_prob < checkin.combined_privacy_value:
            grades[checkin.c_id] = profit_per_user * uniform_prob - checkin.combined_privacy_value
        else:
            grades[checkin.c_id] = profit_per_user - checkin.combined_privacy_value / uniform_prob
        num_positive_grades += 1 if grades[checkin.c_id] > 0 else 0
    logger.info('num_positive_grades = {}'.format(num_positive_grades))

    # Run algorithm 3 (PoI) in the paper:
    # Function g(Y, i, y) = y

    purchased_data = dict()

    # Step 1 of Algorithm 3 in their paper
    m_set = set()  # selected set
    non_m_set = set(data_points.keys())  # not selected set
    values = dict(grades)  # this way, the values only changes with a purchase and only the purchase

    current_profit = 0
    while True:
        # Step 2 of Algorithm 3 in their paper. No need to do this because we do it whenever we buy a new data point
        # for c_id in non_m_set:
        #     checkin = data_points[c_id]
        #
        #     if checkin.c_id in purchased_data:
        #         # value will be changed to this y^max value
        #         y_max = cal_y_max(region, purchased_data[checkin.c_id], profit_per_user, grades[checkin.c_id])
        #
        #         # value will change from a positive value to a non-positive value
        #         if y_max <= 0 < values[checkin.c_id]:
        #             num_positive_grades -= 1
        #
        #         values[checkin.c_id] = y_max

        if num_positive_grades <= 0:
            # no more positive value to buy later
            break

        # Step 3 of Algorithm 3 in their paper
        max_c_id = None
        max_value = 0
        for c_id in non_m_set:  # in not selected set
            if max_value < values[c_id]:
                max_value = values[c_id]
                max_c_id = c_id

        if max_c_id is None:
            break

        # print(max_c_id)
        # If we are here, max_value will be sure > 0 because it started with 0 and max_c_id = None
        if max_c_id in purchased_data:
            noisy_c = purchased_data[max_c_id]
            payoff = profit_per_user if region.contain(noisy_c.rv_x.mean(), noisy_c.rv_y.mean()) else 0

            # Step 3a
            m_set.add(max_c_id)
            non_m_set.remove(max_c_id)
            if 0 < values[max_c_id]:
                num_positive_grades -= 1
            values[max_c_id] = 0

            # update current profit and check
            current_profit += payoff
            if current_profit > fixed_cost:
                # able to decide to open
                break
        else:
            # Step 3b
            # probe it
            c = data_points[max_c_id]
            price_c = c.combined_privacy_value
            if budget < price_c:
                # not enough budget to buy more
                break
            noisy_c = buy_data_at_price(c, price_c,
                                        price_from_noise_rate, std_from_noise_initial_value, std_from_noise_rate)
            purchased_data[noisy_c.c_id] = noisy_c
            # print('Purchased {} data points'.format(len(purchased_data)))
            cost += price_c
            # print('Current cost = {}'.format(cost))
            budget -= price_c

            # check
            payoff = profit_per_user if region.contain(noisy_c.rv_x.mean(), noisy_c.rv_y.mean()) else 0
            if payoff > grades[noisy_c.c_id]:
                m_set.add(max_c_id)
                non_m_set.remove(max_c_id)
                if 0 < values[max_c_id]:
                    num_positive_grades -= 1
                values[max_c_id] = 0

                # update current profit and check
                current_profit += payoff
                if current_profit > fixed_cost:
                    # able to decide to open
                    break
            else:
                y_max = min(payoff, grades[noisy_c.c_id])

                # value will change from a positive value to a non-positive value
                if y_max <= 0 < values[noisy_c.c_id]:
                    num_positive_grades -= 1

                values[noisy_c.c_id] = y_max

        # Step 4 of Algorithm 3 in their paper. No need to do this because we break whenever num_positive_grades <= 0
        # should_return = True
        # for c_id in non_m_set:
        #     if values[c_id] != 0:
        #         # There is at least a data point with v != 0
        #         should_return = False
        #         break
        #
        # if should_return:
        #     # All data points have v == 0
        #     break

    inside_probs = defaultdict(defaultdict)
    for noisy_c in purchased_data.values():
        if noisy_c.c_id in m_set:
            if region.contain(noisy_c.rv_x.mean(), noisy_c.rv_y.mean()):
                inside_probs[noisy_c.user_id][noisy_c.c_id] = 1

    rv = cal_prod_dist_num_user(inside_probs)
    logger.info('({}, {}) Output mean={}, std={}'.format(x_idx, y_idx, rv.mean(), rv.std()))
    return x_idx, y_idx, rv, cost


def cal_y_max(region: Rectangle, c: NoisyCheckin, profit_per_user: float, grade: float) -> float:
    """ Calculate y^max value of PoI algorithm

    Parameters
    ----------
    region
    c
    profit_per_user
    grade

    Returns
    -------

    """
    # probed
    payoff = profit_per_user if region.contain(c.rv_x.mean(), c.rv_y.mean()) else 0
    y_max = min(payoff, grade)
    return y_max
