import math
from collections import defaultdict
from typing import List, Dict

import numpy as np
from scipy.stats import rv_discrete, rv_continuous, norm

from pup.common.datatypes import UserId, CheckinId
from pup.common.enums import FinalProbsFilterType
from pup.common.rectangle import Rectangle
from pup.config import Config


def cal_exponential_function(initial_value: float, rate: float, x: float) -> float:
    """ Calculate value of an exponential grow/decay function:
        result = initial_value * exp(rate * x)

    Parameters
    ----------
    initial_value
        initial value of the function
    rate
        the exponential rate
    x
        input value

    Returns
    -------
    float
        the function value at x
    """
    return initial_value * np.exp(rate * float(x))


def cal_inverse_exponential_function(initial_value: float, rate: float, y: float) -> float:
    """ Calculate value of an exponential grow/decay function: y = initial_value * exp(rate * x)

    x = (1.0 / rate) * log(y / initial_value)

    Parameters
    ----------
    initial_value
        initial value of the function
    rate
        the exponential rate
    y
        out value

    Returns
    -------
    float
        the inverse function value at y
    """
    return (1.0 / rate) * np.log(float(y) / initial_value)


def create_dirac_delta_dist(x: float) -> rv_discrete:
    """ Create a discrete distribution that mimics Dirac delta function at x

    Parameters
    ----------
    x
        the value at with the probability is 1

    Returns
    -------
    rv_discrete
        the discrete distribution that mimics Dirac delta function at x
    """
    return rv_discrete(name='dirac_delta', values=([x], [1]))


def cal_prob_inside_rect(rect: Rectangle, rv_x: rv_continuous, rv_y: rv_continuous) -> float:
    """ Calculate the probability inside a rectangle of 2 distribution

    Parameters
    ----------
    rect
        rectangle of interest
    rv_x
        probability distribution for x dimension
    rv_y
        probability distribution for y dimension

    Returns
    -------
    float
        the probability that the original check-in of this noisy check-in is inside a rectangle
    """
    if rect is None:
        return 0

    prob_x = rv_x.cdf(rect.max_x) - rv_x.cdf(rect.min_x)
    prob_y = rv_y.cdf(rect.max_y) - rv_y.cdf(rect.min_y)

    prob = prob_x * prob_y

    return prob


def cal_prob_dist_num_users_inside(probs_inside: List[float]) -> rv_continuous:
    """ Calculate the probability distribution of the number of users inside the region.

    Parameters
    ----------
    probs_inside
        the probabilities of each user

    References
    ----------
    [1] https://en.wikipedia.org/wiki/Poisson_binomial_distribution

    Returns
    -------
    rv_continuous
        the probability distribution of the number of users inside the region
    """
    mean = sum(probs_inside)
    variance = sum([p * (1 - p) for p in probs_inside])
    if math.isclose(variance, 0):
        # avoid zero variance
        variance = 0.0000001
    std = np.sqrt(variance)
    rv = norm(mean, std)
    return rv


def cal_prod_dist_num_user(probs_inside: Dict[UserId, Dict[CheckinId, float]]) -> rv_continuous:
    """ Calculate the prob dist of the number of data points inside a region based on the probabilities of each point

    Parameters
    ----------
    probs_inside
        probabilities of each data point being inside the region

    Returns
    -------
    rv_continuous
        the probability distribution of the number of data points inside a region
    """
    # get probabilities of being inside of each data point
    probs = list()
    for user, use_probs_inside in probs_inside.items():
        for p in use_probs_inside.values():
            probs.append(p)

    # calculate the distribution
    rv = cal_prob_dist_num_users_inside(probs)
    return rv


def get_probs_higher_than(probs: Dict[UserId, Dict[CheckinId, float]],
                          threshold_prob: float) -> Dict[UserId, Dict[CheckinId, float]]:
    """ Get the probabilities that is higher than a threshold probability

    Parameters
    ----------
    probs
        the probabilities
    threshold_prob
        threshold probability

    Returns
    -------
    Dict[UserId, Dict[CheckinId, float]]
        the probabilities that is higher than a threshold probability
    """
    higher_probs = defaultdict(defaultdict)
    for user, user_probs in probs.items():
        for c_id, p in user_probs.items():
            if p > threshold_prob:
                higher_probs[user][c_id] = p

    return higher_probs


def filter_probs_by_threshold(final_probs_filter_type: FinalProbsFilterType,
                              probs: Dict[UserId, Dict[CheckinId, float]],
                              uniform_prob: float):
    """ Only consider probabilities of each point being inside the region that is higher than a threshold

    Parameters
    ----------
    final_probs_filter_type
        the type of filter to remove probabilities after buying a set of data
    probs
        the probabilities
    uniform_prob
        uniform probability

    Returns
    -------
    Dict[UserId, Dict[CheckinId, float]]
        the probabilities that is higher than a threshold probability

    Raises
    ------
    ValueError
        if the final probabilities filter type is not valid
    """
    final_probs_filter_threshold = get_prob_threshold_from_type(final_probs_filter_type, uniform_prob)

    filtered_probs = get_probs_higher_than(probs, final_probs_filter_threshold)
    return filtered_probs


def get_prob_threshold_from_type(final_probs_filter_type: FinalProbsFilterType, uniform_prob: float) -> float:
    """ Get the probability threshold from the type

    Parameters
    ----------
    final_probs_filter_type
        the type of filter to remove probabilities after buying a set of data
    uniform_prob
        uniform probability

    Returns
    -------
    float
        the probability threshold

    Raises
    ------
    ValueError
        if the final probabilities filter type is not valid
    """
    if final_probs_filter_type == FinalProbsFilterType.ZERO:
        final_probs_filter_threshold = 0
    elif final_probs_filter_type == FinalProbsFilterType.UNIFORM:
        final_probs_filter_threshold = uniform_prob
    else:
        raise ValueError('Invalid final probabilities filter type: {}'.format(final_probs_filter_type))

    return final_probs_filter_threshold


def get_linear_profit_fixed_cost() -> float:
    """ Get the fixed cost of linear profit model from configuration

    Returns
    -------
    float
        the fixed cost of linear profit model from configuration
    """
    opening_threshold = Config.eval_opening_threshold
    if opening_threshold is None:
        # opening_threshold = Config.linear_profit_fixed_cost / Config.linear_profit_profit_per_user
        fixed_cost = Config.linear_profit_fixed_cost
    else:
        # opening threshold is set => calculate fixed cost from profit per user and this threshold
        fixed_cost = opening_threshold * Config.linear_profit_profit_per_user
    return fixed_cost
