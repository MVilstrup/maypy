from maypy import ALPHA
from maypy.best_practices.checks import parametric_pair
from maypy.distributions import Distribution
import scipy.stats as st
import numpy as np
from maypy.hypothesis_testing.non_parametric.confidence.different_medians import \
    unpaired_non_parametric_different_medians


def unpaired_difference_confidence(P: Distribution, Q: Distribution):
    if not parametric_pair(P, Q):
        return unpaired_non_parametric_different_medians(P, Q)
    else:
        raise NotImplementedError()
