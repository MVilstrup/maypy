from functools import lru_cache

from maypy.distributions import Distribution
from maypy.experiment.report import Report

PARAMETRIC = "PARAMETRIC"
NON_PARAMETRIC = "NON_PARAMETRIC"
MIXED = "MIXED"

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
