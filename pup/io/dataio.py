"""
Utility functions for data input/output
"""

import csv
import logging
import ntpath
import os
from collections import defaultdict
from datetime import datetime
from typing import List

import numpy as np
from scipy.stats import rv_discrete, norm

from pup.algorithms import fixed_maximum_cost
from pup.algorithms.util import create_dirac_delta_dist, get_linear_profit_fixed_cost
from pup.common import rectangle
from pup.common.checkin import Checkin
from pup.common.constants import GOWALLA_TIME_PATTERN
from pup.common.enums import DatasetType, MethodType
from pup.config import Config
from pup.experiment import preprocess
from pup.io.filters import check_checkin_in_area

logger = logging.getLogger(__name__)


def load_data(filepath: str, dataset_type: DatasetType, areas=None):
    """
    Load data from file

    Parameters
    ----------
    filepath: str
        file path
    dataset_type: DatasetType
        type of dataset
    areas: list
        the data should belong to the intersection of these areas

    Returns
    -------
    list
        list of check-ins
    """
    if areas is None:
        areas = list()

    if dataset_type == DatasetType.GOWALLA:
        return load_dataset_gowalla(filepath, areas)
    else:
        logger.error("Invalid dataset type: {}".format(dataset_type))
        return None


def load_dataset_gowalla(filepath: str, areas=None) -> list:
    """
    Load a Gowalla dataset from file

    Parameters
    ----------
    filepath: str
        file path
    areas: list
        the data should belong to the intersection of these areas

    Returns
    -------
    list
        list of check-ins
    """
    if areas is None:
        areas = list()
    data = list()

    c_id = 0
    with open(filepath, 'r') as csv_file:
        gowalla_reader = csv.reader(csv_file, delimiter='\t')
        for row in gowalla_reader:
            user_id = int(row[0])
            checkin_time = datetime.strptime(row[1], GOWALLA_TIME_PATTERN)
            lat = float(row[2])
            lon = float(row[3])
            loc_id = int(row[4])

            c = Checkin(c_id=None,
                        user_id=user_id,
                        timestamp=int(datetime.timestamp(checkin_time)),
                        datetime=checkin_time,
                        lat=lat,
                        lon=lon,
                        location_id=loc_id)

            # check if check-in is in these areas
            ok = True
            for area in areas:
                ok = ok and check_checkin_in_area(c, area)

            if ok:
                c.c_id = c_id
                c_id += 1

                data.append(c)

    return data


def load_data_by_config(random_seed):
    """
    Load data from configuration.
    Also doing limiting number of check-ins per user and convert to local (x, y) coordinates

    Parameters
    ----------
    random_seed: int
        a random seed for random check-in selection

    Returns
    -------
    list
        list of check-ins
    """
    input_file = Config.data_file
    dataset_type = Config.dataset_type
    area_code = Config.eval_area_code
    num_checkins_per_user = Config.eval_num_checkins_per_user

    rect = rectangle.get_rectangle_for_area(area_code)

    areas = [rect]

    data: list = load_data(input_file, dataset_type, areas)

    logger.info('Loaded {} check-ins from {}'.format(len(data), input_file))

    # limit_num_checkins
    data = preprocess.limit_checkins_per_user(data, num_checkins_per_user, random_seed)

    # convert_to_local_coordinates
    orig_x, orig_y = rect.get_mid_point()
    preprocess.checkins_to_location_coordinates(data, orig_y, orig_x)  # latitude = y, longitude = x

    return data


def save_checkins(checkins: list, out_file: str):
    """
    Save checkins to file

    Parameters
    ----------
    checkins: list
        list of checkin
    out_file: str
        output file
    """
    with open(out_file, 'w') as f:
        csv_writer = csv.writer(f, delimiter='\t')
        for c in checkins:
            row = [str(c.user_id),
                   datetime.strftime(c.datetime, GOWALLA_TIME_PATTERN),
                   str(c.lat),
                   str(c.lon),
                   str(c.location_id)]
            csv_writer.writerow(row)


def prepare_content_dir() -> str:
    """ Prepare content directory which is a long path including various parameters from configuration
    in the name of directories in the path

    Returns
    -------
    str
        the path to directory
    """
    content_dir = Config.output_content_dir
    content_dir = os.path.join(content_dir,
                               'dataset_' + Config.dataset_type.name + '_area_' + Config.eval_area_code.name)
    content_dir = os.path.join(content_dir,
                               'n_checkins_' + str(Config.eval_num_checkins_per_user)
                               + '_len_x_' + str(Config.eval_grid_cell_len_x)
                               + '_len_y_' + str(Config.eval_grid_cell_len_y)
                               + '_boundary_' + str(Config.eval_grid_boundary_order))
    if not Config.dist_privacy_should_random:
        content_dir = os.path.join(content_dir, 'priv_fixed')
    content_dir = os.path.join(content_dir,
                               'priv_level_type_' + Config.dist_user_privacy_level_type.name
                               + '_loc_{0:.1f}'.format(Config.dist_user_privacy_level_loc)
                               + '_scale_{0:.1f}'.format(Config.dist_user_privacy_level_scale)
                               + '_seed_' + str(Config.dist_user_privacy_level_random_seed))
    content_dir = os.path.join(content_dir,
                               'sensitivity_type_' + Config.dist_user_loc_sensitivity_type.name
                               + '_loc_{0:.1f}'.format(Config.dist_user_loc_sensitivity_loc)
                               + '_scale_{0:.1f}'.format(Config.dist_user_loc_sensitivity_scale)
                               + '_seed_' + str(Config.dist_user_loc_sensitivity_random_seed))
    content_dir = os.path.join(content_dir,
                               'price_func_rate_{0:.1f}'.format(Config.price_from_noise_func_rate))
    content_dir = os.path.join(content_dir,
                               'std_init_value_{0:.1f}'.format(Config.standard_deviation_from_noise_func_initial_value)
                               + '_std_rate_{0:.1f}'.format(Config.standard_deviation_from_noise_func_rate))
    # get threshold for being popular
    opening_threshold = Config.eval_opening_threshold
    fixed_cost = get_linear_profit_fixed_cost()

    content_dir = os.path.join(content_dir,
                               'payoff_{}'.format(Config.payoff_matrix_type.name)
                               + '_profit_per_user_{}'.format(int(Config.linear_profit_profit_per_user))
                               + '_fixed_cost_{}'.format(int(fixed_cost))
                               + '_pop_threshold_' + str(int(opening_threshold)))

    os.makedirs(content_dir, exist_ok=True)

    return content_dir


def prepare_output_file_name(prefix: str, method: MethodType) -> str:
    """ Prepare output file name for distribution file

    Parameters
    ----------
    prefix
        file name prefix
    method
        method type

    Returns
    -------
    str
        output file name for distribution file
    """
    file_name = prefix + '_method_' + method.name + '_seed_' + str(Config.checkin_selection_random_seed)

    if method == MethodType.FIXED_MAXIMUM_COST or method == MethodType.PROBING:
        file_name += '_final_probs_filter_type_{}'.format(Config.final_probs_filter_type.name)
        budget = Config.budget
        if method == MethodType.FIXED_MAXIMUM_COST:
            budget = fixed_maximum_cost.get_fmc_budget()
        file_name += '_budget_{0:.1f}'.format(budget)
        file_name += '_find_count_{}'.format(Config.find_actual_count)
        file_name += '_algo_{}'.format(Config.probing_algorithm.name)

        if Config.start_price > 0:
            file_name += '_start_price_{0:.4f}'.format(Config.start_price)
        else:
            file_name += '_start_std_ratio_{0:.2f}'.format(Config.start_std_ratio)
        file_name += '_extended_std_factor_{0:.2f}'.format(Config.probing_extended_cell_sigma_factor)
        file_name += '_price_increment_factor_{0:.2f}'.format(Config.probing_price_increment_factor)
        file_name += '_check_inout_{}'.format(Config.probing_should_check_inout)
        file_name += '_only_one_next_price_{}'.format(Config.probing_should_check_only_one_next_price)

        # if Config.probing_buy_singly:
        #     file_name += '_SINGLY'
        #
        # if Config.probing_probability_stopping:
        #     file_name += '_PROB_STOPPING'

    file_name += '.csv'
    return file_name


def write_costs(method: MethodType, costs: np.ndarray):
    """ Write costs array to a output file

    Parameters
    ----------
    method
        method type
    costs
        costs of each region
    """
    dist_dir = prepare_content_dir()
    filename = prepare_output_file_name(Config.output_costs_prefix, method)

    outfile = os.path.join(dist_dir, filename)
    np.savetxt(outfile, costs, delimiter='\t', fmt='%5.3f')


def read_costs(method: MethodType) -> np.ndarray:
    """ Read costs array from an input file of a method

    Parameters
    ----------
    method
        method type

    Returns
    -------
    np.ndarray
        costs of each region
    """
    dist_dir = prepare_content_dir()
    filename = prepare_output_file_name(Config.output_costs_prefix, method)

    infile = os.path.join(dist_dir, filename)
    return read_costs_from_file(infile)


def read_costs_from_file(infile: str) -> np.ndarray:
    """ Read costs array from an input file

    Parameters
    ----------
    infile
        input file

    Returns
    -------
    np.ndarray
        costs of each region
    """
    with open(infile, 'r') as f:
        costs = np.loadtxt(f, delimiter='\t')
        return costs


def write_distributions(method: MethodType, dists: List[List]):
    """ Write distribution x, y, rv_type, mean, and std to a output file

    (x, y) is in the increasing order or x then in the increasing order of y.
    rv_type is 'rv_continuous' or 'rv_discrete' for normal distribution or dirac delta distribution in the list.

    Parameters
    ----------
    method
        method type
    dists
        distributions
    """
    dist_dir = prepare_content_dir()
    filename = prepare_output_file_name(Config.output_distributions_prefix, method)

    outfile = os.path.join(dist_dir, filename)

    rows = list()
    for x in range(len(dists)):
        for y in range(len(dists[x])):
            rv = dists[x][y]
            rv_type = 'rv_continuous'
            if isinstance(rv, rv_discrete):
                rv_type = 'rv_discrete'

            row = [str(x), str(y), rv_type, str(rv.mean()), str(rv.std())]

            rows.append(row)
    with open(outfile, 'w') as f:
        csv_writer = csv.writer(f, delimiter='\t')
        csv_writer.writerows(rows)


def read_distributions(method: MethodType) -> List[List]:
    """ Read distributions as x, y, type, mean, and std from input file

    Parameters
    ----------
    method
        method type

    Returns
    -------
    List[List]
        distributions
    """
    dist_dir = prepare_content_dir()
    filename = prepare_output_file_name(Config.output_distributions_prefix, method)

    infile = os.path.join(dist_dir, filename)
    return read_distributions_from_file(infile)


def read_distributions_from_file(infile: str) -> List[List]:
    """ Read distributions as x, y, type, mean, and std from input file

    Parameters
    ----------
    infile
        input file

    Returns
    -------
    List[List]
        distributions
    """
    dists = list()

    with open(infile, 'r') as f:
        csv_reader = csv.reader(f, delimiter='\t')

        prev_x = -1
        list_y = None
        for row in csv_reader:
            x = int(row[0])
            # y = int(row[1])
            rv_type = row[2]
            mean = float(row[3])
            std = float(row[4])

            if prev_x < x:
                # done with previous list
                if list_y is not None:
                    dists.append(list_y)
                # a new list needed
                list_y = list()

                prev_x = x

            if rv_type == 'rv_continuous':
                rv = norm(loc=mean, scale=std)
            else:
                rv = create_dirac_delta_dist(mean)

            list_y.append(rv)

        # still the last row left
        dists.append(list_y)

    return dists


def write_center_edge_probs(method: MethodType, center_probs: dict, edge_probs: dict):
    """ Write center and edger probabilities to file.
    Output format: Each line:
    std_step center_prob edge_prob

    Parameters
    ----------
    method
        method
    center_probs
        center probabilities
    edge_probs
        edge probabilities
    """
    dist_dir = prepare_content_dir()
    filename = prepare_output_file_name(Config.output_center_edge_probs_prefix, method)

    outfile = os.path.join(dist_dir, filename)
    with open(outfile, 'w') as f:
        csv_writer = csv.writer(f, delimiter='\t')
        max_std_step = max(center_probs.keys())
        for std_step in range(1, max_std_step + 1):
            row = [str(std_step), str(center_probs[std_step]), str(edge_probs[std_step])]

            csv_writer.writerow(row)


def read_center_edge_probs(method: MethodType) -> (dict, dict):
    """ Read center and edger probabilities from file.
    Input format: Each line:
    std_step center_prob edge_prob

    Parameters
    ----------
    method
        method

    Returns
    -------
    center_probs: dict
        center probabilities or None if error occurred
    edge_probs: dict
        edge probabilities or None if error occurred
    """
    dist_dir = prepare_content_dir()
    filename = prepare_output_file_name(Config.output_center_edge_probs_prefix, method)

    center_probs = defaultdict(float)
    edge_probs = defaultdict(float)

    infile = os.path.join(dist_dir, filename)
    try:

        with open(infile, 'r') as f:
            csv_reader = csv.reader(f, delimiter='\t')

            for row in csv_reader:
                std_step = int(row[0])

                center_probs[std_step] = float(row[1])
                edge_probs[std_step] = float(row[2])

        return center_probs, edge_probs
    except FileNotFoundError:
        return None, None


def write_output_evaluations(method,
                             num_users, num_data_points, num_regions,
                             total_cost,
                             true_positives, true_negatives, false_positives, false_negatives,
                             precision, recall, f1_score,
                             linear_total_payoff, linear_total_adjusted_payoff,
                             linear_avg_adjusted_payoff, linear_median_adjusted_payoff,
                             rmse,
                             exe_time,
                             note=''):
    """ Write a test result to file

    Parameters
    ----------
    method
        method
    num_users: int
        number of users
    num_data_points: int
        number of data points
    num_regions: int
        number of regions
    total_cost: float
        the total cost spent
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
    linear_total_payoff: float
        total realized payoff value of linear profit model
    linear_total_adjusted_payoff: float
        total realized payoff value adjusted for the cost of linear profit model
    linear_avg_adjusted_payoff: float
        average realized payoff value per grid cell adjusted for the cost of linear profit model
    linear_median_adjusted_payoff: float
        median adjusted payoff
    rmse: float
        RMSE
    exe_time: float
        execution time
    note: str
        some note, default is empty string
    """
    # write header
    if not os.path.isfile(Config.output_file):
        # file not exist, create it and add header
        with open(Config.output_file, 'a+') as f:
            csv_writer = csv.writer(f, delimiter='\t')
            header = [
                'time',
                'data_file', 'random_seed', 'payoff_matrix_type',
                'profit_per_user', 'fixed_cost',
                'area_code', 'num_checkins_per_user',
                'opening_threshold', 'grid_cell_len', 'grid_boundary_order',
                'dist_user_privacy_level_type', 'dist_user_privacy_level_loc',
                'dist_user_privacy_level_scale', 'dist_user_privacy_level_random_seed',
                'dist_user_loc_sensitivity_type', 'dist_user_loc_sensitivity_loc',
                'dist_user_loc_sensitivity_scale', 'dist_user_loc_sensitivity_random_seed',
                'price_from_noise_func_rate',
                'std_from_noise_func_initial_value', 'std_from_noise_func_rate',
                'find_actual_count',
                'free_data_price_threshold',
                'method',
                'fmc_budget_from_cost_percentage',
                'fmc_budget_from_probing',
                'budget',
                'probing_algorithm',
                'price_increment_factor',
                'should_check_inout',
                'should_check_only_one_next_price',
                'quantization_len',
                'extended_cell_sigma_factor',
                'start_std_ratio',
                'start_price',
                'probing_point_inside_stop_threshold',
                'final_probs_filter_type',
                'num_users', 'num_data_points', 'num_regions',
                'total_cost', 'avg_cost',
                'true_positives', 'true_negatives', 'false_positives', 'false_negatives',
                'precision', 'recall', 'f1_score',
                'linear_total_payoff', 'linear_total_adjusted_payoff', 'linear_avg_adjusted_payoff',
                'linear_median_adjusted_payoff',
                'rmse',
                'exe_time',
                'note']
            csv_writer.writerow(header)

    opening_threshold = Config.eval_opening_threshold
    fixed_cost = get_linear_profit_fixed_cost()

    # write result row
    with open(Config.output_file, 'a+') as f:
        csv_writer = csv.writer(f, delimiter='\t')
        row = [
            datetime.strftime(datetime.now(), '%Y-%m-%d %H:%M:%S.%f'),
            ntpath.basename(Config.data_file),
            '{}'.format(Config.checkin_selection_random_seed),
            '{}'.format(Config.payoff_matrix_type.name),
            '{}'.format(int(Config.linear_profit_profit_per_user)),
            '{}'.format(int(fixed_cost)),
            '{}'.format(Config.eval_area_code.name),
            '{}'.format(Config.eval_num_checkins_per_user),
            '{}'.format(int(opening_threshold)),
            '{}'.format(Config.eval_grid_cell_len_x),
            '{}'.format(Config.eval_grid_boundary_order),
            '{}'.format(Config.dist_user_privacy_level_type.name),
            '{}'.format(Config.dist_user_privacy_level_loc),
            '{}'.format(Config.dist_user_privacy_level_scale),
            '{}'.format(Config.dist_user_privacy_level_random_seed),
            '{}'.format(Config.dist_user_loc_sensitivity_type.name),
            '{}'.format(Config.dist_user_loc_sensitivity_loc),
            '{}'.format(Config.dist_user_loc_sensitivity_scale),
            '{}'.format(Config.dist_user_loc_sensitivity_random_seed),
            '{}'.format(Config.price_from_noise_func_rate),
            '{}'.format(Config.standard_deviation_from_noise_func_initial_value),
            '{}'.format(Config.standard_deviation_from_noise_func_rate),
            '{}'.format(Config.find_actual_count),
            '{0:.5f}'.format(Config.free_data_price_threshold),
            '{}'.format(method.name),
            '{}'.format(Config.fmc_budget_from_cost_percentage),
            '{}'.format(Config.fmc_budget_from_probing),
            '{0:.1f}'.format(Config.budget),
            '{}'.format(Config.probing_algorithm.name),
            '{0:.1f}'.format(Config.probing_price_increment_factor),
            '{}'.format(Config.probing_should_check_inout),
            '{}'.format(Config.probing_should_check_only_one_next_price),
            '{}'.format(Config.probing_quantization_len),
            '{}'.format(Config.probing_extended_cell_sigma_factor),
            '{0:.3f}'.format(Config.start_std_ratio),
            '{0:.4f}'.format(Config.start_price),
            '{0:.3f}'.format(Config.probing_point_inside_stop_threshold),
            '{}'.format(Config.final_probs_filter_type.name),
            '{}'.format(num_users),
            '{}'.format(num_data_points),
            '{}'.format(num_regions),
            '{0:.3f}'.format(total_cost),
            '{0:.3f}'.format(float(total_cost) / num_regions),
            '{}'.format(int(true_positives)),
            '{}'.format(int(true_negatives)),
            '{}'.format(int(false_positives)),
            '{}'.format(int(false_negatives)),
            '{0:.3f}'.format(precision),
            '{0:.3f}'.format(recall),
            '{0:.3f}'.format(f1_score),
            '{0:.3f}'.format(linear_total_payoff),
            '{0:.3f}'.format(linear_total_adjusted_payoff),
            '{0:.3f}'.format(linear_avg_adjusted_payoff),
            '{0:.3f}'.format(linear_median_adjusted_payoff),
            '{0:.3f}'.format(rmse),
            '{0:.3f}'.format(exe_time),
            '{}'.format(note)]
        csv_writer.writerow(row)
