from functools import lru_cache

from maypy.distributions import Distribution
from maypy.experiment.report import Report
from pandas.core.dtypes.common import is_numeric_dtype
import numpy as np
import pandas as pd

PARAMETRIC = "PARAMETRIC"
NON_PARAMETRIC = "NON_PARAMETRIC"
MIXED = "MIXED"


@lru_cache
def is_parametric(P: Distribution, experiment=None):
    if "is_parametric" in P:
        return P["is_parametric"]
    else:
        result = P.distribution_name in ["norm", "lognorm"]
        report = Report("is_parametric",
                        "heuristic",
                        statistic=None,
                        p_value=None,
                        h0_rejected=lambda alpha: result,
                        interpretation=lambda alpha: result)

        P["is_parametric"] = report

    if experiment is not None:
        experiment[P] = report

    return report


@lru_cache
def is_non_parametric(P: Distribution, experiment=None):
    if "is_non_parametric" in P:
        report = P["is_non_parametric"]
    else:
        result = not is_parametric(P)
        report = Report("is_not_parametric",
                        "heuristic",
                        statistic=None,
                        p_value=None,
                        h0_rejected=lambda alpha: result,
                        interpretation=lambda alpha: result)

        P["is_non_parametric"] = report

    if experiment is not None:
        experiment[P] = report

    return report


@lru_cache
def is_continuous(P: Distribution, experiment=None):
    """
    Is is impossible to estimate whether numbers are continuous or discrete,
    but we can make a few checks to ensure no explicitly invalid data suchs as strings is used
    """
    if "is_continuous" in P:
        report = P["is_continuous"]
    else:
        checks = [
            is_numeric_dtype(P.data),  # Ensure all data is numeric
            len(np.unique(P.data)) > 2,  # Ensure there is no Boolean columns
            pd.isna(P.data).sum() == 0,  # Ensure no None values is used
        ]

        statistic = sum(checks) / len(checks)
        result = all(checks)

        report = Report("is_continous",
                        "heuristic",
                        statistic=statistic,
                        p_value=None,
                        h0_rejected=lambda alpha: result,
                        interpretation=lambda alpha: result)

        P["is_continuous"] = report

    if experiment is not None:
        experiment[P] = report

    return report


def minimum_sample_size(requirement, P: Distribution, experiment=None):
    if "minimum_sample_size" in P:
        report = P["minimum_sample_size"]
    else:
        result = requirement < len(P)
        report = Report("minimum_sample_size",
                        "measure",
                        statistic=None,
                        p_value=None,
                        h0_rejected=lambda alpha: result,
                        interpretation=lambda alpha: result)
        P["is_continuous"] = report

    if experiment is not None:
        experiment[P] = report

    return report


@lru_cache
def pair_type(P: Distribution, Q: Distribution, *args):
    if is_parametric(P) and is_parametric(Q):
        return PARAMETRIC
    elif is_non_parametric(P) and is_non_parametric(Q):
        return NON_PARAMETRIC
    else:
        return MIXED


def parametric_pair(P: Distribution, Q: Distribution, *args):
    return pair_type(P, Q) == PARAMETRIC


def non_parametric_pair(P: Distribution, Q: Distribution, *args):
    return pair_type(P, Q) == NON_PARAMETRIC


def mixed_pair(P: Distribution, Q: Distribution):
    return pair_type(P, Q) == MIXED
