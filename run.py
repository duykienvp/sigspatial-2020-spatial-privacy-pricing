import argparse
import sys
import logging

from pup.common.enums import MethodType
from pup.config import Config
from pup.experiment import exp_executor

logger = logging.getLogger(__name__)


TASK_BENCHMARK = 'benchmark'
TASK_COLLECT_DATA = 'collect'
TASKS = [TASK_BENCHMARK, TASK_COLLECT_DATA]

APPROACH_GT = 'gt'
APPROACH_BUY_ALL_ACCURATE = 'baa'
APPROACH_UP = 'up'
APPROACH_FMC = 'fmc'
APPROACH_PROBING = 'probing'
APPROACHES = [APPROACH_GT, APPROACH_BUY_ALL_ACCURATE, APPROACH_UP, APPROACH_FMC, APPROACH_PROBING]


class MyParser(argparse.ArgumentParser):
    """ An parse to print help whenever an error occurred to the parsing process
    """
    def error(self, message):
        sys.stderr.write('error: %s\n' % message)
        self.print_help()
        sys.exit(2)


if __name__ == '__main__':
    parser = MyParser(description='Execute a task')
    parser.add_argument('-t',
                        '--task',
                        help='Task name',
                        choices=TASKS,
                        required=True)

    parser.add_argument('-a',
                        '--approach',
                        help='Approach name',
                        choices=APPROACHES,
                        required=False)

    parser.add_argument('-v',
                        '--verbose',
                        help='increase output verbosity',
                        action='store_true')

    parser.add_argument("--eval-only",
                        default=False,
                        help="Flag to only run evaluation when we already have output costs and distributions",
                        action="store_true")

    args = parser.parse_args()

    logger.info('Started')
    logger.info(Config.get_config_str())

    if args.task == TASK_COLLECT_DATA:
        exp_executor.collect_data()

    if args.task == TASK_BENCHMARK:
        eval_only = args.eval_only
        if args.approach == APPROACH_GT:
            exp_executor.execute_method(MethodType.GROUND_TRUTH, eval_only)

        if args.approach == APPROACH_BUY_ALL_ACCURATE:
            exp_executor.execute_method(MethodType.BUY_ALL_ACCURATE, eval_only)

        if args.approach == APPROACH_UP:
            exp_executor.execute_method(MethodType.UNIFORM_PRIOR, eval_only)

        if args.approach == APPROACH_FMC:
            exp_executor.execute_method(MethodType.FIXED_MAXIMUM_COST, eval_only)

        if args.approach == APPROACH_PROBING:
            exp_executor.execute_method(MethodType.PROBING, eval_only)
