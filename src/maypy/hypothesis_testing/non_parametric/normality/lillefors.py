from typing import Optional

from maypy.distributions import Distribution
from maypy.experiment.interpretation import Interpretation
from maypy.experiment.report import Report
from maypy.experiment.test import Test
from statsmodels.stats.diagnostic import lilliefors


class LilleforsNormality(Test):

    def check_assumptions(self, report, P: Distribution, Q=None):
        """

        :param report:
        :param P:
        :param Q:
        :return:
        """
        report.add_assumption("P is continuous", P.is_continuous)
        return report

    def __call__(self, P: Distribution):
        statistic, p_value = lilliefors(P.data, pvalmethod="approx")
        report = Report("Lillefors Normality",
                        "statistic",
                        statistic=statistic,
                        p_value=p_value,
                        h0_rejected=lambda alpha: p_value > alpha,
                        interpretation=lambda alpha: Interpretation(p_value > alpha))

        self.experiment[P] = report
        return self.check_assumptions(report, P)
