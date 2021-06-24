from functools import lru_cache

from maypy.best_practices.checks import pair_type, non_parametric_pair
from maypy.distributions import Distribution
from maypy.experiment.experiment import Experiment


class DistributionPair:

    @staticmethod
    def mean_greater_than(P: Distribution, Q: Distribution, experiment: Experiment):
        from maypy.hypothesis_testing.non_parametric.goodness_of_fit.kolmogorov import KolmogorovSmirnovTest

        if pair_type(P, Q) == "non_parametric":
            test = KolmogorovSmirnovTest(experiment).greater_than
        else:
            raise NotImplementedError("Can currently only two non parametric distributions")

        return test(P, Q)

    @staticmethod
    def mean_less_than(P: Distribution, Q: Distribution, experiment: Experiment):
        from maypy.hypothesis_testing.non_parametric.goodness_of_fit.kolmogorov import KolmogorovSmirnovTest

        if non_parametric_pair(P, Q):
            test = KolmogorovSmirnovTest(experiment).less_than
        else:
            raise NotImplementedError("Can currently only two non parametric distributions")

        return test(P, Q)

    @staticmethod
    def mean_not_equal(P: Distribution, Q: Distribution, experiment: Experiment):
        from maypy.hypothesis_testing.non_parametric.goodness_of_fit.kolmogorov import KolmogorovSmirnovTest

        if non_parametric_pair(P, Q):
            test = KolmogorovSmirnovTest(experiment).not_equal
        else:
            raise NotImplementedError("Can currently only two non parametric distributions")

        return test(P, Q)
