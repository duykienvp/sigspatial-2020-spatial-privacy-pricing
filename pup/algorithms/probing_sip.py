"""
Our probing algorithm SIP and SIP-T
"""
import logging
import time
from collections import defaultdict
from typing import Tuple

import numpy as np
import scipy.integrate as integrate
import scipy.special as special
from scipy.stats import rv_continuous

from pup.algorithms.privacy_helper import buy_data_at_price
from pup.algorithms.util import cal_prod_dist_num_user, get_linear_profit_fixed_cost
from pup.common.constants import SQRT_2
from pup.common.datatypes import CheckinDataset
from pup.common.grid import Grid
from pup.common.noisycheckin import NoisyCheckin
from pup.common.rectangle import Rectangle
from pup.config import Config

logger = logging.getLogger(__name__)


def probing_sip(
        x_idx: int,
        y_idx: int,
        data: CheckinDataset,
        grid: Grid,
        budget: float,
        region: Rectangle) -> Tuple[int, int, rv_continuous, float, float]:
    """ Running the basic probing algorithm for a region with singly buying

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
    exe_time: float
        execution time
    """

    # Load config for this algorithm
    price_from_noise_rate = Config.price_from_noise_func_rate
    std_from_noise_initial_value = Config.standard_deviation_from_noise_func_initial_value
    std_from_noise_rate = Config.standard_deviation_from_noise_func_rate
    price_increment_factor = Config.probing_price_increment_factor

    linear_profit_profit_per_user = Config.linear_profit_profit_per_user
    linear_profit_fixed_cost = get_linear_profit_fixed_cost()

    cost = 0

    # probabilities of each point being inside the region.
    # This is \mathbf{\tilde{D}} in the paper, but we only keep the probabilities so we do not need to recalculate it
    most_accurate_probs_inside = defaultdict(defaultdict)

    std_factor = Config.probing_extended_cell_sigma_factor

    find_actual_count = Config.find_actual_count

    data_points = dict()
    for _, checkins in data.items():
        for c in checkins.values():
            data_points[c.c_id] = c

    curr_total_profit = 0

    start_price = Config.start_price

    should_check_inout = Config.probing_should_check_inout

    s_time = time.time()

    # buy all data points at starting price
    logger.info('({}, {}) start buying at starting price {}'.format(x_idx, y_idx, start_price))
    current_prices = dict()  # storing the prices of each data point for illustration purpose
    eivs = dict()
    best_next_prices = dict()
    for c in data_points.values():
        price_c = start_price

        noisy_c = buy_data_at_price(
            c, price_c, price_from_noise_rate, std_from_noise_initial_value, std_from_noise_rate)

        cost += price_c
        budget -= price_c
        current_prices[c.c_id] = price_c

        # no matter what happen, this is the most accurate version, so update the inside probability
        prob_inside_region = noisy_c.cal_prob_inside_rect(region)
        most_accurate_probs_inside[c.user_id][c.c_id] = prob_inside_region

        if not find_actual_count:
            curr_point_profit = prob_inside_region * linear_profit_profit_per_user
            curr_total_profit += curr_point_profit
            if curr_total_profit > linear_profit_fixed_cost:
                break

        # check if we would continue to buy it at more accurate levels
        should_cal_eiv = True

        if should_check_inout:
            current_std = noisy_c.rv_x.std()

            std_threshold = std_factor * current_std
            is_inside = check_inside(region, noisy_c, std_threshold)
            is_outside = check_outside(region, noisy_c, std_threshold)

            should_cal_eiv = not is_inside and not is_outside  # unable to decide so we need to do more

        if should_cal_eiv:
            eiv, next_price = cal_eiv(noisy_c, prob_inside_region, price_c, price_increment_factor, region, linear_profit_profit_per_user)
            # logger.info('({}, {}) EIV of c {} = {}'.format(x_idx, y_idx, c.c_id, eiv))

            eivs[noisy_c.c_id] = eiv
            best_next_prices[noisy_c.c_id] = next_price

            # if np.isclose(next_price, 0):
            #     logger.info('({}, {}) EIV of c {} = {}, next price == 0, std={}, priv val = {}'.format(
            #         x_idx, y_idx, c.c_id, eiv, noisy_c.rv_x.std(), c.combined_privacy_value))
            #
            # if next_price > 0 and price_increment_factor < price_c / next_price:
            #     logger.info('Having price jump')

    logger.info('({}, {}) bought at starting price'.format(x_idx, y_idx))

    # We buy the one with the highest EIV dist
    while True:
        if len(eivs) == 0:
            break
        c_id = max(eivs, key=eivs.get)
        if not should_check_inout and eivs[c_id] <= 0:
            # No more point with positive EIV, so we stop
            # Only need to do this if we do not check inside/outside
            logger.info('({}, {}) No more point with positive EIV'.format(x_idx, y_idx))
            break

        # We are sure EIV of this point is positive, so buy it
        c = data_points[c_id]
        # price_c = current_prices[c.c_id] * price_increment_factor
        price_c = best_next_prices[c.c_id]

        noisy_c = buy_data_at_price(
            c, price_c, price_from_noise_rate, std_from_noise_initial_value, std_from_noise_rate)

        cost += price_c
        budget -= price_c
        current_prices[c.c_id] = price_c

        before_prob_inside = most_accurate_probs_inside[c.user_id][c.c_id]
        # no matter what happen, this is the most accurate version, so update the inside probability
        prob_inside_region = noisy_c.cal_prob_inside_rect(region)
        most_accurate_probs_inside[c.user_id][c.c_id] = prob_inside_region

        if not find_actual_count:
            # check profit
            prob_inside_change = prob_inside_region - before_prob_inside
            point_profit_change = prob_inside_change * linear_profit_profit_per_user
            curr_total_profit += point_profit_change
            if curr_total_profit > linear_profit_fixed_cost:
                break

        # check if we would continue to buy it at more accurate levels
        should_cal_eiv = True

        if should_check_inout:
            current_std = noisy_c.rv_x.std()

            std_threshold = std_factor * current_std
            is_inside = check_inside(region, noisy_c, std_threshold)
            is_outside = check_outside(region, noisy_c, std_threshold)

            should_cal_eiv = not is_inside and not is_outside  # unable to decide so we need to do more

        if should_cal_eiv:
            eiv, next_price = cal_eiv(noisy_c, prob_inside_region, price_c, price_increment_factor, region, linear_profit_profit_per_user)
            # logger.info('({}, {}) EIV of c {} = {}'.format(x_idx, y_idx, c.c_id, eiv))

            eivs[noisy_c.c_id] = eiv
            best_next_prices[noisy_c.c_id] = next_price

            # if next_price > 0 and price_increment_factor < price_c / next_price:
            #     logger.info('Having price jump')
        else:
            # We will not consider this point anymore
            # It happens only if we check inside/outside and it is either inside or outside
            del eivs[c.c_id]
            del best_next_prices[c.c_id]

    rv = cal_prod_dist_num_user(most_accurate_probs_inside)
    logger.info('({}, {}) Output mean={}, std={}'.format(x_idx, y_idx, rv.mean(), rv.std()))

    # dataio.write_prices(MethodType.PROBING, x_idx, y_idx, current_prices)
    exe_time = time.time() - s_time
    return x_idx, y_idx, rv, cost, exe_time


def cal_eiv(noisy_c, current_prob_inside, current_price, price_increment_factor, region, profit_per_user):
    """ Calculate the Expected Incremental Value of buying data point at a new price

    Parameters
    ----------
    noisy_c
        the current noisy data point
    current_prob_inside
        the current probability of this noisy data point being inside
    current_price
        the current price
    price_increment_factor
        the increment factor for price, which is also the decrement factor for std
    region
        the region
    profit_per_user
        profit per user

    Returns
    -------
    float
        the Expected Incremental Value of buying data point at a new price

    """
    prev_std = noisy_c.rv_x.std()
    prev_price = current_price
    best_price = 0
    eiv = -np.inf

    only_one_next_price = Config.probing_should_check_only_one_next_price
    while 1 <= prev_std:
        next_std = prev_std / price_increment_factor
        next_price = prev_price * price_increment_factor
        # Calculate EIV for this point
        # 1st, we calculate the current expected profit
        current_expected_profit = current_prob_inside * profit_per_user

        # 2nd, we calculate the next expected profit
        # start with x dimension
        prev_pos = noisy_c.rv_x.mean()
        r_min = region.min_x
        r_max = region.max_x
        integ_min = r_min - next_std * 3
        integ_max = r_max + next_std * 3
        expected_prob_inside_x = cal_expected_prob_inside(prev_pos, prev_std, next_std, r_min, r_max, integ_min, integ_max)

        # then with y dimension
        prev_pos = noisy_c.rv_y.mean()
        r_min = region.min_y
        r_max = region.max_y
        integ_min = r_min - next_std * 3
        integ_max = r_max + next_std * 3
        expected_prob_inside_y = cal_expected_prob_inside(prev_pos, prev_std, next_std, r_min, r_max, integ_min, integ_max)

        expected_prob_inside = expected_prob_inside_x * expected_prob_inside_y

        next_expected_profit = profit_per_user * expected_prob_inside - next_price
        new_eiv = next_expected_profit - current_expected_profit
        # eiv = next_expected_profit

        if eiv < new_eiv:
            eiv = new_eiv
            best_price = next_price

        prev_std = next_std
        prev_price = next_price

        if only_one_next_price:
            break

    return eiv, best_price


def cal_expected_prob_inside(x0, sigma0, sigma, r_min, r_max, integ_min, integ_max):
    """ Calculate the expected probability of being inside for a dimension

    Parameters
    ----------
    x0
        previous location
    sigma0
        previous sdv
    sigma
        current sigma
    r_min
        the min value of the region
    r_max
        the max value of the region
    integ_min
        the starting point of the integral
    integ_max
        the end point of the integral

    Returns
    -------
    float
        the expected probability of being inside for a dimension

    """
    # full_output = 1 in order to suppress warning
    integ = integrate.quad(cal_expected_prob_inside_integ, integ_min, integ_max, args=(x0, sigma0, sigma, r_min, r_max), full_output=1)
    expected_prob_inside = 0.5 * 1.0 / (np.sqrt(2 * np.pi * (sigma0*sigma0 + sigma*sigma))) * max(integ[0], 0)
    return expected_prob_inside


def cal_expected_prob_inside_integ(x, x0, sigma0, sigma, r_min, r_max):
    """ Calculate the part under the integral

    Parameters
    ----------
    x
        a possible location
    x0
        previous location
    sigma0
        previous sdv
    sigma
        current sigma
    r_min
        the min value of the region
    r_max
        the max value of the region

    Returns
    -------
    float
        the value of the part under the integral

    """
    v1 = np.exp(-0.5 * (x - x0) * (x - x0) / (sigma0*sigma0 + sigma*sigma))
    v2 = special.erf((r_max - x) / (sigma * SQRT_2)) - special.erf((r_min - x) / (sigma * SQRT_2))
    return v1 * v2


def check_inside(region: Rectangle, c: NoisyCheckin, distance_threshold: float) -> bool:
    """ Check if we can be confident that the data point is inside

    We can that we are confident if for either dimension, the mean is inside and the distance from the mean
    to the closest edge is no more than a threshold (which often be a factor of standard deviation)

    Parameters
    ----------
    region
        the region
    c
        noisy data point
    distance_threshold
        distance threshold to say if the data point is outside

    Returns
    -------
    bool
        whether or not we can be confident that it is inside
    """
    inside_x = region.min_x <= c.rv_x.mean() - distance_threshold and c.rv_x.mean() + distance_threshold <= region.max_x
    inside_y = region.min_y <= c.rv_y.mean() - distance_threshold and c.rv_y.mean() + distance_threshold <= region.max_y
    return inside_x and inside_y


def check_outside(region: Rectangle, c: NoisyCheckin, distance_threshold: float) -> bool:
    """ Check if we can be confident that data point is outside: on either side, the noisy mean is too far

    We can that we are confident if for either dimension, the mean is outside and the distance from the mean
    to the closest edge is more than a threshold (which often be a factor of standard deviation)

    Parameters
    ----------
    region
        the region
    c
        noisy data point
    distance_threshold
        distance threshold to say if the data point is outside

    Returns
    -------
    bool
        whether or not we can be confident that it is outside
    """
    dist_min_x = max(region.min_x - c.rv_x.mean(), 0)  # if the position is outside to the left or not
    dist_max_x = max(c.rv_x.mean() - region.max_x, 0)  # if the position is outside to the right or not
    dist_x = max(dist_min_x, dist_max_x)

    dist_min_y = max(region.min_y - c.rv_y.mean(), 0)  # if the position is outside to the lower or not
    dist_max_y = max(c.rv_y.mean() - region.max_y, 0)  # if the position is outside to the upper or not
    dist_y = max(dist_min_y, dist_max_y)

    outside_x = distance_threshold < dist_x
    outside_y = distance_threshold < dist_y
    return outside_x or outside_y


