"""
Utility helpers for privacy-related distributions and functions
"""
import logging
import math
from collections import defaultdict
from typing import Tuple

import numpy as np
from scipy.stats import rv_continuous, uniform, norm

from pup.algorithms.util import cal_exponential_function, cal_inverse_exponential_function
from pup.common.checkin import Checkin
from pup.common.datatypes import CheckinDataset
from pup.common.enums import DistributionType
from pup.common.noisycheckin import NoisyCheckin
from pup.config import Config

logger = logging.getLogger(__name__)


def prepare_probability_dist(dist_type: DistributionType, loc: float, scale: float) -> rv_continuous:
    """ Prepare a probability distribution

    Parameters
    ----------
    dist_type
        type of distribution
    loc
        loc parameter
    scale
        scale parameter

    Returns
    -------
    rv_continuous
        continuous random variable from the distribution
    """
    if dist_type == DistributionType.UNIFORM:
        return uniform(loc=loc, scale=scale)
    elif dist_type == DistributionType.NORMAL:
        return norm(loc=loc, scale=scale)
    else:
        logger.error('Unsupported distribution type {}'.format(dist_type))


def prepare_prob_dist_user_privacy_level() -> rv_continuous:
    """ Prepare probability distribution of users' privacy level based on configuration

    Returns
    -------
    rv_continuous
        continuous random variable from the distribution
    """
    dist_type = Config.dist_user_privacy_level_type
    loc = Config.dist_user_privacy_level_loc
    scale = Config.dist_user_privacy_level_scale

    return prepare_probability_dist(dist_type, loc, scale)


def prepare_prob_dist_user_loc_sensitivity() -> rv_continuous:
    """ Prepare probability distribution of sensitivity of a location based on configuration

    Returns
    -------
    rv_continuous
        continuous random variable from the distribution
    """
    dist_type = Config.dist_user_loc_sensitivity_type
    loc = Config.dist_user_loc_sensitivity_loc
    scale = Config.dist_user_loc_sensitivity_scale

    return prepare_probability_dist(dist_type, loc, scale)


def generate_privacy_values(item_ids: list, rv: rv_continuous, random_state=None) -> defaultdict:
    """ Generate privacy values from the given distribution.

    Privacy values are:
        - privacy levels for users
        - sensitivity for data points

    Parameters
    ----------
    item_ids
        user ids or data point ids
    rv
        distribution
    random_state: None, int, or numpy random state
        random state for generating random numbers

    Returns
    -------
    defaultdict
        dict of privacy values for each item: item_id => privacy value
    """
    num_items = len(item_ids)
    if Config.dist_privacy_should_random:
        levels = rv.rvs(size=num_items, random_state=random_state)
    else:
        levels = np.full((num_items,), rv.mean())

    privacy_values = defaultdict(float)
    for i in range(num_items):
        privacy_values[item_ids[i]] = levels[i]

    return privacy_values


def generate_privacy_values_for_users(data: dict, rv: rv_continuous, random_state=None) -> defaultdict:
    """ Generate privacy values for each user.

    Parameters
    ----------
    data
        data set: user_id => list of data points of each user
    rv
        distribution
    random_state: None, int, or numpy random state
        random state for generating random numbers

    Returns
    -------
    defaultdict
        dict of privacy value for each user: user_id => privacy value
    """
    return generate_privacy_values(list(data.keys()), rv, random_state)


def generate_privacy_values_for_data_points(data: CheckinDataset,
                                            rv: rv_continuous, random_state=None) -> CheckinDataset:
    """ Generate privacy values for each data point.

    Parameters
    ----------
    data
        check-in dataset
    rv
        distribution
    random_state: None, int, or numpy random state
        random state for generating random numbers

    Returns
    -------
    CheckinDataset
        dict of privacy value for each data point: user_id => list of privacy values of each data point of each user
    """
    total_num_data_points = 0  # this is the number of points we need to generate from the rv
    for user, data_points in data.items():
        total_num_data_points += len(data_points)

    # generate privacy values
    temp_data_point_ids = list(range(total_num_data_points))
    privacy_values = generate_privacy_values(temp_data_point_ids, rv, random_state)

    # assign those values to data points
    i = 0
    data_points_privacy_values = defaultdict(defaultdict)
    for user, data_points in data.items():
        for data_id in data_points.keys():
            data_points_privacy_values[user][data_id] = privacy_values[i]
            i += 1

    return data_points_privacy_values


def generate_combined_privacy_values_for_data_points(data: CheckinDataset,
                                                     user_rv: rv_continuous,
                                                     data_point_rv: rv_continuous,
                                                     user_random_state=None,
                                                     data_point_random_state=None):
    """ Generate combined privacy values for each data point.

    The combined value = value for user + value for data point. Each data point is assigned these 3 values

    Parameters
    ----------
    data
        check-in dataset
    user_rv
        distribution for users
    data_point_rv
        distribution for data points
    user_random_state: None, int, or numpy random state
        random state for generating random numbers for users
    data_point_random_state: None, int, or numpy random state
        random state for generating random numbers for data points
    """
    user_privacy_values = generate_privacy_values_for_users(data, user_rv, user_random_state)
    data_points_privacy_values = generate_privacy_values_for_data_points(data, data_point_rv, data_point_random_state)

    for user, data_points in data.items():
        user_privacy_value = user_privacy_values[user]
        for data_id, data_point in data_points.items():
            data_point.user_privacy_value = user_privacy_value
            data_point.sensitivity = data_points_privacy_values[user][data_id]
            data_point.combined_privacy_value = combine_privacy_values(
                data_point.user_privacy_value, data_point.sensitivity)


def combine_privacy_values(user_privacy_value: float, data_point_privacy_value: float) -> float:
    """ Combine privacy values of user and user for data point

    Parameters
    ----------
    user_privacy_value
        privacy value of user
    data_point_privacy_value
        privacy value of user for a data point

    Returns
    -------
    float
        the combined privacy value
    """
    # return user_privacy_value + data_point_privacy_value  # simple combination by addition
    return user_privacy_value * data_point_privacy_value  # simple combination by multiplication


def prepare_combined_privacy_values_for_data_points(data: CheckinDataset):
    """ Generate combined privacy values for each data point.

    The combined value = value for user + value for data point. Each data point is assigned these 3 values.
    Values are generated from configured distributions

    Parameters
    ----------
    data
        check-in dataset
    """
    user_rv = prepare_prob_dist_user_privacy_level()
    data_point_rv = prepare_prob_dist_user_loc_sensitivity()
    user_random_state = Config.dist_user_privacy_level_random_seed
    data_point_random_state = Config.dist_user_loc_sensitivity_random_seed

    generate_combined_privacy_values_for_data_points(
        data, user_rv, data_point_rv, user_random_state, data_point_random_state)


def price_to_noise_level(price, rate, privacy_value):
    """ Calculate noise level from a price and privacy level.

    Parameters
    ----------
    price
        price
    rate
        the rate k of noise level -> price function
    privacy_value
        the initial value

    Returns
    -------
    float
        the noise level
    """
    return cal_inverse_exponential_function(privacy_value, -rate, price)


def price_to_noise(price: float, privacy_value: float, price_from_noise_rate: float,
                   std_from_noise_initial_value: float, std_from_noise_rate: float) -> Tuple[float, float]:
    """ Calculate the parameters of noises based on a given price

    Parameters
    ----------
    price
        price
    privacy_value
        privacy value
    price_from_noise_rate
        rate of price from noise exponential function
    std_from_noise_initial_value
        initial value of standard deviation from noise exponential function, i.e. when input values is approx 0
    std_from_noise_rate
        rate of standard deviation from noise exponential function

    Returns
    -------
    noise_level: float
        noise level
    std: float
        standard deviation of noise distribution
    """
    if privacy_value < price or math.isclose(privacy_value, price):
        noise_level = 0
        std = 0.0001  # to avoid division by zero
    else:
        # noise_level = -(1.0 / price_from_noise_rate) * np.log(price / privacy_value)
        noise_level = price_to_noise_level(price, price_from_noise_rate, privacy_value)
        std = noise_level_to_noise_std(noise_level, std_from_noise_initial_value, std_from_noise_rate)
    return noise_level, std


def noise_level_to_noise_std(noise_level: float, initial_value: float, rate: float) -> float:
    """ Calculate noise standard deviation from noise level

    Parameters
    ----------
    noise_level
        noise level
    initial_value
        initial value of standard deviation from noise exponential function, i.e. when input values is approx 0
    rate
        rate of standard deviation from noise exponential function

    Returns
    -------
    float
        noise std
    """
    if math.isclose(noise_level, 0):
        return 0

    return cal_exponential_function(initial_value, rate, noise_level - 1)


def gen_gaussian_noisy_checkin(c: Checkin, noise_level: float, std: float, random_seed=None) -> NoisyCheckin:
    """ Generate a noisy check-in with Gaussian noise

    Parameters
    ----------
    c
        check-in
    noise_level
        noise level
    std
        standard deviation of noise distribution
    random_seed: None, int, or numpy random state
        random state for generating random numbers

    Returns
    -------
    NoisyCheckin
        the noisy check-in
    """
    noises = generate_noise(std, random_seed)

    rv_x = norm(loc=c.x + noises[0], scale=std)
    rv_y = norm(loc=c.y + noises[1], scale=std)
    noisy_c = NoisyCheckin(c, noise_level, rv_x, rv_y)

    return noisy_c


def generate_noise(std: float, random_seed=None) -> Tuple[float, float]:
    """ Generate noise given a standard deviation

    Parameters
    ----------
    std
        standard deviation
    random_seed
        random seed

    Returns
    -------
    tuple
        tuple of 2 values of noises
    """
    rv_noises = norm(loc=0, scale=std)
    noises = rv_noises.rvs(size=2, random_state=random_seed)

    return noises[0], noises[1]


def buy_data_at_price(c: Checkin, price: float, price_from_noise_rate: float,
                      std_from_noise_initial_value: float, std_from_noise_rate: float) -> NoisyCheckin:
    """ Buy a noisy version of a check-in at a price

    Parameters
    ----------
    c
        check-in
    price
        price to buy
    price_from_noise_rate
        rate of price from noise exponential function
    std_from_noise_initial_value
        initial value of standard deviation from noise exponential function, i.e. when input values is approx 0
    std_from_noise_rate
        rate of standard deviation from noise exponential function

    Returns
    -------
    NoisyCheckin
        the noisy check-in
    """
    noise_level, std = price_to_noise(price, c.combined_privacy_value,
                                      price_from_noise_rate,
                                      std_from_noise_initial_value, std_from_noise_rate)

    # logger.info('privacy_value={}, noise_level={}, std={}'.format(privacy_value, noise_level, std))

    # generate noisy check-in. For reproducibility, use user id as random state
    noisy_c = gen_gaussian_noisy_checkin(c, noise_level, std, c.user_id)

    return noisy_c
