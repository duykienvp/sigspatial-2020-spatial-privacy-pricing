"""
Evaluation based on accuracy measurements
"""
import logging
from typing import List

import numpy as np

logger = logging.getLogger(__name__)


def eval_rmse_means_vs_true_counts(dists_of_num_users: List[List], true_counts: np.ndarray) -> float:
    """ Calculate the RMSE between the means of the predicted distributions and the true counts

    Parameters
    ----------
    dists_of_num_users
        the predicted distributions
    true_counts
        the true counts

    Returns
    -------
    float
        the RMSE
    """
    means = get_means_array(dists_of_num_users)
    return cal_rmse(means, true_counts)


def cal_rmse(predictions: np.ndarray, targets: np.ndarray) -> float:
    """ Calculate Root Mean Square Error

    Parameters
    ----------
    predictions
        the predictions
    targets
        the target

    Returns
    -------
    float
        the Root Mean Square Error
    """
    return np.sqrt(cal_mse(predictions, targets))


def cal_mse(predictions: np.ndarray, targets: np.ndarray) -> float:
    """ Calculate Mean Square Error

    Parameters
    ----------
    predictions
        the predictions
    targets
        the target

    Returns
    -------
    float
        the Mean Square Error
    """
    return np.mean(np.square((predictions - targets)))


def get_means_array(dists: List[List]) -> np.ndarray:
    """ Get the array of means of distributions, by x dimensions then y dimensions

    Parameters
    ----------
    dists
        the distributions

    Returns
    -------
    numpy.ndarray
        the array of the means of the distributions

    Raises
    ------
    ValueError
        if the array of distributions is empty in either dimension
    """
    x_size = len(dists)
    if 0 < x_size:
        y_size = len(dists[0])
        if 0 < y_size:
            means = np.zeros((x_size, y_size))
            for x in range(x_size):
                for y in range(y_size):
                    means[x, y] = dists[x][y].mean()

            return means
        else:
            raise ValueError('Empty x dimension')
    else:
        raise ValueError('Empty x dimension')
