from typing import Optional

from maypy.best_practices.checks import is_continuous
from maypy.distributions import Distribution
from maypy.experiment.report import Report
from maypy.experiment.test import Test
from statsmodels.stats.diagnostic import lilliefors


class LilleforsNormality(Test):

    def check_assumptions(self, P: Distribution, Q: Optional[Distribution] = None):
        helper = "The Lillefors Test can only handle continuous distributions"
        self.distribution_assumption(is_continuous, P, helper)

    def __call__(self, P: Distribution):
        statistic, p_value = lilliefors(P.data, pvalmethod="approx")
        report = Report("Lillefors Normality",
                        "statistic",
                        statistic=statistic,
                        p_value=p_value,
                        h0_rejected=lambda alpha: p_value > alpha,
                        interpretation=lambda alpha: p_value > alpha)

        self.experiment[P] = report
        return report
