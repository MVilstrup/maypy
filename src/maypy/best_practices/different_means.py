from maypy.best_practices.checks import parametric_pair
from maypy.distributions import Distribution
from maypy.experiment.experiment import Experiment
from maypy.hypothesis_testing.non_parametric.goodness_of_fit.kolmogorov import KolmogorovSmirnovTest


def greater_mean(P: Distribution, Q: Distribution):
    if not parametric_pair(P, Q):
        return KolmogorovSmirnovTest(None).greater_than(P, Q)
    else:
        raise NotImplementedError("Can currently only two non parametric distributions")


def lesser_mean(P: Distribution, Q: Distribution):
    if not parametric_pair(P, Q):
        return KolmogorovSmirnovTest(None).less_than(P, Q)
    else:
        raise NotImplementedError("Can currently only two non parametric distributions")


def different_mean(P: Distribution, Q: Distribution):
    if not parametric_pair(P, Q):
        return KolmogorovSmirnovTest(None).not_equal(P, Q)
    else:
        raise NotImplementedError("Can currently only two non parametric distributions")
