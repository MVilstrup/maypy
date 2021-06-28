from functools import lru_cache

from maypy.best_practices.checks import parametric_pair
from maypy.distributions import Distribution
from maypy.hypothesis_testing.parametric.correlation.pearson import PearsonCorrelationTest
from maypy.hypothesis_testing.non_parametric.correlation.spearman import SpearmanCorrelationTest


@lru_cache
def is_correlated(P: Distribution, Q: Distribution, experiment=None):
    if parametric_pair(P, Q):
        return PearsonCorrelationTest(experiment).correlated(P, Q)
    else:
        return SpearmanCorrelationTest(experiment).correlated(P, Q)


@lru_cache
def not_correlated(P: Distribution, Q: Distribution, experiment=None):
    if parametric_pair(P, Q):
        return PearsonCorrelationTest(experiment).not_correlated(P, Q)
    else:
        return SpearmanCorrelationTest(experiment).not_correlated(P, Q)
