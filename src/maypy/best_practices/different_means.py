from maypy.best_practices.checks import parametric_pair
from maypy.distributions import Distribution
from maypy.experiment.experiment import Experiment
from maypy.hypothesis_testing.non_parametric.goodness_of_fit.kolmogorov import KolmogorovSmirnovTest


def greater_mean(P: Distribution, Q: Distribution, experiment: Experiment):
    if not parametric_pair(P, Q):
        return KolmogorovSmirnovTest(experiment).greater_than(P, Q)
    else:
        raise NotImplementedError("Can currently only two non parametric distributions")


def lesser_mean(P: Distribution, Q: Distribution, experiment: Experiment):
    if not parametric_pair(P, Q):
        return KolmogorovSmirnovTest(experiment).less_than(P, Q)
    else:
        raise NotImplementedError("Can currently only two non parametric distributions")


def different_mean(P: Distribution, Q: Distribution, experiment: Experiment):
    if not parametric_pair(P, Q):
        return KolmogorovSmirnovTest(experiment).not_equal(P, Q)
    else:
        raise NotImplementedError("Can currently only two non parametric distributions")
