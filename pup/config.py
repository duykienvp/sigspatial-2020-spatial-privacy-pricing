"""
Configuration handler: loading logging and program configuration
NOTE: This file must be included to be able to use the service correctly
"""

import logging
import logging.config
import os
import sys

import yaml

from pup.common.enums import DatasetType, AreaCode, PayoffMatrixType, FinalProbsFilterType, DistributionType, ProbingAlgorithmType

CONFIG_FILE = 'config.yml'
LOG_CONFIG_FILE = 'logging.yml'


def parse_dashed_list(value: str):
    """

    Parameters
    ----------
    value: str
        string with 2 numbers separated by 1 dash

    Returns
    -------
    list
        list from those 2 numbers or `None` if error occurred

    Examples
    --------
    >>> parse_dashed_list('1-5')
    [1, 2, 3, 4, 5]
    """
    dash_pos = value.find('-')
    if dash_pos != -1:
        s = int(value[:dash_pos])
        t = int(value[dash_pos + 1:])
        return list(range(s, t + 1))
    return None


class Config(object):
    """ Configuration
    """
    is_config_loaded = False

    project_dir = None

    # data
    data_dir = None
    data_file = None
    dataset_type = None

    output_dir = None
    output_file = None
    output_content_dir = None
    output_costs_prefix = None
    output_distributions_prefix = None
    output_prices_prefix = None
    output_center_edge_probs_prefix = None

    checkin_selection_random_seed = None
    parallel = None
    parallel_num_cpu = None

    # evaluation
    eval_area_code = None
    eval_num_checkins_per_user = None
    eval_opening_threshold = None
    eval_grid_cell_len_x = None
    eval_grid_cell_len_y = None
    eval_grid_boundary_order = None

    # payoff matrix
    linear_profit_profit_per_user = None
    linear_profit_fixed_cost = None

    # method specific
    find_actual_count = None
    final_probs_filter_type = None
    fmc_budget_from_cost_percentage = None
    fmc_budget_from_probing = None
    budget = None
    start_price = -1
    start_std_ratio = None
    probing_parallel_find_best_buying_action = None
    probing_algorithm = None
    probing_buy_singly = None
    probing_price_increment_factor = None
    probing_should_check_inout = None
    probing_should_check_only_one_next_price = None
    probing_probability_stopping = None
    probing_quantization_len = None
    probing_extended_cell_sigma_factor = None
    probing_focused_set_condition = None
    probing_point_inside_stop_threshold = None

    # privacy_distributions
    dist_privacy_should_random = None
    dist_user_privacy_level_type = None
    dist_user_privacy_level_loc = None
    dist_user_privacy_level_scale = None
    dist_user_privacy_level_random_seed = None

    dist_user_loc_sensitivity_type = None
    dist_user_loc_sensitivity_loc = None
    dist_user_loc_sensitivity_scale = None
    dist_user_loc_sensitivity_random_seed = None

    price_from_noise_func_rate = None

    standard_deviation_from_noise_func_initial_value = None
    standard_deviation_from_noise_func_rate = None

    free_data_price_threshold = None

    @staticmethod
    def load_config(file, reload_config=False):
        """ Load config from a file.
        Args:
            file: config file in YAML format
            reload_config: should we reload config
        """
        logger = logging.getLogger(__name__)
        # Load the configuration file
        if file is None:
            logger.error('No config file provided')
            return

        if not Config.is_config_loaded or reload_config:
            with open(file, 'r') as conf_file:
                cfg = yaml.safe_load(conf_file)
                # print(cfg)

                # Load config to variables

                if 'project_dir' in cfg:
                    Config.project_dir = cfg['project_dir']

                # data files
                if 'data' in cfg:
                    data_cfg = cfg['data']

                    if 'data_dir' in data_cfg:
                        Config.data_dir = os.path.join(Config.project_dir, data_cfg['data_dir'])

                    if 'data_file' in data_cfg:
                        Config.data_file = os.path.join(Config.data_dir, data_cfg['data_file'])

                    if 'dataset_type' in data_cfg:
                        Config.dataset_type = DatasetType[data_cfg['dataset_type'].upper()]

                # output
                if 'output' in cfg:
                    output_cfg = cfg['output']

                    if 'output_dir' in output_cfg:
                        Config.output_dir = os.path.join(Config.project_dir, output_cfg['output_dir'])
                    if 'output_file' in output_cfg:
                        Config.output_file = os.path.join(Config.output_dir, output_cfg['output_file'])

                    if 'content_dir' in output_cfg:
                        Config.output_content_dir = os.path.join(Config.output_dir, output_cfg['content_dir'])
                    if 'output_costs_prefix' in output_cfg:
                        Config.output_costs_prefix = output_cfg['output_costs_prefix']
                    if 'output_distributions_prefix' in output_cfg:
                        Config.output_distributions_prefix = output_cfg['output_distributions_prefix']
                    if 'output_prices_prefix' in output_cfg:
                        Config.output_prices_prefix = output_cfg['output_prices_prefix']
                    if 'output_center_edge_probs_prefix' in output_cfg:
                        Config.output_center_edge_probs_prefix = output_cfg['output_center_edge_probs_prefix']

                # random seed
                if 'checkin_selection_random_seed' in cfg:
                    Config.checkin_selection_random_seed = cfg['checkin_selection_random_seed']

                # parallel
                if 'parallel' in cfg:
                    Config.parallel = cfg['parallel']

                if 'parallel_num_cpu' in cfg:
                    Config.parallel_num_cpu = int(cfg['parallel_num_cpu'])

                # evaluation
                if 'evaluation' in cfg:
                    evaluation_cfg = cfg['evaluation']

                    if 'area_code' in evaluation_cfg:
                        Config.eval_area_code = AreaCode[evaluation_cfg['area_code'].upper()]

                    if 'num_checkins_per_user' in evaluation_cfg:
                        Config.eval_num_checkins_per_user = int(evaluation_cfg['num_checkins_per_user'])

                    if 'grid_cell_len_x' in evaluation_cfg:
                        Config.eval_grid_cell_len_x = evaluation_cfg['grid_cell_len_x']

                    if 'grid_cell_len_y' in evaluation_cfg:
                        Config.eval_grid_cell_len_y = evaluation_cfg['grid_cell_len_y']

                    if 'grid_boundary_order' in evaluation_cfg:
                        Config.eval_grid_boundary_order = int(evaluation_cfg['grid_boundary_order'])

                    if 'decision_making_eval' in evaluation_cfg:
                        decision_making_eval_cfg = evaluation_cfg['decision_making_eval']

                        if 'opening_threshold' in decision_making_eval_cfg:
                            Config.eval_opening_threshold = float(decision_making_eval_cfg['opening_threshold'])

                        # linear profit model
                        if 'linear_profit_profit_per_user' in decision_making_eval_cfg:
                            Config.linear_profit_profit_per_user = float(
                                decision_making_eval_cfg['linear_profit_profit_per_user'])
                        if 'linear_profit_fixed_cost' in decision_making_eval_cfg:
                            Config.linear_profit_fixed_cost = float(
                                decision_making_eval_cfg['linear_profit_fixed_cost'])

                if 'method_specific' in cfg:
                    method_specific_cfg = cfg['method_specific']

                    if 'find_actual_count' in method_specific_cfg:
                        Config.find_actual_count = method_specific_cfg['find_actual_count']

                    if 'fmc_budget_from_cost_percentage' in method_specific_cfg:
                        Config.fmc_budget_from_cost_percentage = float(method_specific_cfg[
                                                                         'fmc_budget_from_cost_percentage'])
                    if 'fmc_budget_from_probing' in method_specific_cfg:
                        Config.fmc_budget_from_probing = method_specific_cfg['fmc_budget_from_probing']

                    if 'final_probs_filter_type' in method_specific_cfg:
                        Config.final_probs_filter_type = FinalProbsFilterType[
                            method_specific_cfg['final_probs_filter_type'].upper()]

                    if 'budget' in method_specific_cfg:
                        Config.budget = float(method_specific_cfg['budget'])

                    if 'start_price' in method_specific_cfg:
                        Config.start_price = float(method_specific_cfg['start_price'])

                    if 'start_std_ratio' in method_specific_cfg:
                        Config.start_std_ratio = float(method_specific_cfg['start_std_ratio'])

                    if 'probing' in method_specific_cfg:
                        probing_cfg = method_specific_cfg['probing']

                        if 'parallel_find_best_buying_action' in probing_cfg:
                            Config.probing_parallel_find_best_buying_action = probing_cfg[
                                'parallel_find_best_buying_action']

                        if 'algorithm' in probing_cfg:
                            Config.probing_algorithm = ProbingAlgorithmType[
                                probing_cfg['algorithm'].upper()]

                        if 'buy_singly' in probing_cfg:
                            Config.probing_buy_singly = probing_cfg['buy_singly']

                        if 'probability_stopping' in probing_cfg:
                            Config.probing_probability_stopping = probing_cfg['probability_stopping']

                        if 'should_check_inout' in probing_cfg:
                            Config.probing_should_check_inout = probing_cfg['should_check_inout']

                        if 'should_check_only_one_next_price' in probing_cfg:
                            Config.probing_should_check_only_one_next_price = probing_cfg[
                                'should_check_only_one_next_price']

                        if 'price_increment_factor' in probing_cfg:
                            Config.probing_price_increment_factor = probing_cfg['price_increment_factor']

                        if 'quantization_len' in probing_cfg:
                            Config.probing_quantization_len = float(probing_cfg['quantization_len'])

                        if 'extended_cell_sigma_factor' in probing_cfg:
                            Config.probing_extended_cell_sigma_factor = float(probing_cfg['extended_cell_sigma_factor'])

                        if 'point_inside_stop_threshold' in probing_cfg:
                            Config.probing_point_inside_stop_threshold = float(
                                probing_cfg['point_inside_stop_threshold'])

                if 'privacy_distributions' in cfg:
                    privacy_distributions_cfg = cfg['privacy_distributions']

                    if 'should_random' in privacy_distributions_cfg:
                        Config.dist_privacy_should_random = privacy_distributions_cfg['should_random']

                    if 'dist_user_privacy_level' in privacy_distributions_cfg:
                        dist_user_privacy_level_cfg = privacy_distributions_cfg['dist_user_privacy_level']

                        if 'type' in dist_user_privacy_level_cfg:
                            Config.dist_user_privacy_level_type = DistributionType[
                                dist_user_privacy_level_cfg['type'].upper()]

                        if 'loc' in dist_user_privacy_level_cfg:
                            Config.dist_user_privacy_level_loc = float(dist_user_privacy_level_cfg['loc'])

                        if 'scale' in dist_user_privacy_level_cfg:
                            Config.dist_user_privacy_level_scale = float(dist_user_privacy_level_cfg['scale'])

                        if 'random_seed' in dist_user_privacy_level_cfg:
                            Config.dist_user_privacy_level_random_seed = dist_user_privacy_level_cfg['random_seed']

                    if 'dist_user_loc_sensitivity' in privacy_distributions_cfg:
                        dist_user_loc_sensitivity_cfg = privacy_distributions_cfg['dist_user_loc_sensitivity']

                        if 'type' in dist_user_loc_sensitivity_cfg:
                            Config.dist_user_loc_sensitivity_type = DistributionType[
                                dist_user_loc_sensitivity_cfg['type'].upper()]

                        if 'loc' in dist_user_loc_sensitivity_cfg:
                            Config.dist_user_loc_sensitivity_loc = float(dist_user_loc_sensitivity_cfg['loc'])

                        if 'scale' in dist_user_loc_sensitivity_cfg:
                            Config.dist_user_loc_sensitivity_scale = float(dist_user_loc_sensitivity_cfg['scale'])

                        if 'random_seed' in dist_user_loc_sensitivity_cfg:
                            Config.dist_user_loc_sensitivity_random_seed = dist_user_loc_sensitivity_cfg['random_seed']

                if 'price_from_noise_func' in cfg:
                    price_from_noise_func_cfg = cfg['price_from_noise_func']

                    if 'rate' in price_from_noise_func_cfg:
                        Config.price_from_noise_func_rate = float(price_from_noise_func_cfg['rate'])

                if 'standard_deviation_from_noise_func' in cfg:
                    standard_deviation_from_noise_func_cfg = cfg['standard_deviation_from_noise_func']

                    if 'initial_value' in standard_deviation_from_noise_func_cfg:
                        Config.standard_deviation_from_noise_func_initial_value = \
                            float(standard_deviation_from_noise_func_cfg['initial_value'])

                    if 'rate' in standard_deviation_from_noise_func_cfg:
                        Config.standard_deviation_from_noise_func_rate = \
                            float(standard_deviation_from_noise_func_cfg['rate'])

                if 'free_data_price_threshold' in cfg:
                    Config.free_data_price_threshold = float(cfg['free_data_price_threshold'])

    @staticmethod
    def get_config_str() -> str:
        """
        Get string representation of all configuration

        Returns
        -------
        str
            string representation of all the configurations
        """
        values = []
        for k, v in Config.__dict__.items():
            tmp = str(k) + '=' + str(v) + '\n'
            values.append(tmp)
        values.sort()
        res = ''.join(values)
        return res


def setup_logging(default_path='logging.yml', default_level=logging.INFO, env_key='LOG_CFG'):
    """ Setup logging configuration

    Parameters
    ----------
    default_path: str
        default logging file path
    default_level:
        default logging level
    env_key: str
        environment key to get logging config
    """
    path = default_path
    value = os.getenv(env_key, None)
    if value:
        path = value
    if os.path.exists(path):
        with open(path, 'rt', encoding=sys.getfilesystemencoding()) as f:
            config = yaml.safe_load(f.read())
        logging.config.dictConfig(config)
    else:
        logging.basicConfig(level=default_level)


dir_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
logging_config_file = os.path.join(dir_path, LOG_CONFIG_FILE)
setup_logging(default_path=logging_config_file)


Config.load_config(os.path.join(dir_path, CONFIG_FILE))
