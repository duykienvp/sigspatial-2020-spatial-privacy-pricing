from enum import Enum


class MethodType(Enum):
    """
    Types of methods
    """
    GROUND_TRUTH = 'GT'
    BUY_ALL_ACCURATE = 'BAA'
    UNIFORM_PRIOR = 'UP'
    FIXED_MAXIMUM_COST = 'FMC'
    PROBING = 'PROBING'


class ActionType(Enum):
    """
    Types of action
    """
    BUYING = 0
    OPEN_CANCEL = 1


class FinalDecisionType(Enum):
    """
    Types of final decision
    """
    OPEN = 0
    CANCEL = 1


class ProbingAlgorithmType(Enum):
    """
    Types of probing algorithm
    """
    BASIC = 'BASIC'
    POI = 'POI'
    MPOI = 'MPOI'
    SIP = 'SIP'


class FinalProbsFilterType(Enum):
    """
    The type of filter to remove probabilities after buying a set of data
    """
    ZERO = 'ZERO'  # the probabilities has to be > 0
    UNIFORM = 'UNIFORM'  # the probabilities has to be > uniform probabilities


class DatasetType(Enum):
    """
    Types of datasets
    """
    GOWALLA = 'GOWALLA'


class AreaCode(Enum):
    """
    Area code for areas we experimented
    """
    LOS_ANGELES = 'LOS_ANGELES'


class PayoffMatrixType(Enum):
    """
    Type of payoff matrix
    """
    COUNTING_FOR_RESTAURANT = 'COUNTING_FOR_RESTAURANT'


class DistributionType(Enum):
    """
    Type of distributions
    """
    UNIFORM = 'UNIFORM'
    NORMAL = 'NORMAL'
