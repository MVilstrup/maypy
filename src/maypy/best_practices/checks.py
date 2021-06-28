from functools import lru_cache

from maypy.distributions import Distribution
from maypy.experiment.report import Report

PARAMETRIC = "PARAMETRIC"
NON_PARAMETRIC = "NON_PARAMETRIC"
MIXED = "MIXED"


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
    if P.is_parametric and Q.is_parametric:
        return PARAMETRIC
    elif not P.is_parametric and not Q.is_parametric:
        return NON_PARAMETRIC
    else:
        return MIXED


def parametric_pair(P: Distribution, Q: Distribution, *args):
    return pair_type(P, Q) == PARAMETRIC


def non_parametric_pair(P: Distribution, Q: Distribution, *args):
    return pair_type(P, Q) == NON_PARAMETRIC


def mixed_pair(P: Distribution, Q: Distribution):
    return pair_type(P, Q) == MIXED
