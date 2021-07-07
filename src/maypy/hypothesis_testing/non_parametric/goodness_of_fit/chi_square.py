from typing import Optional

from maypy.best_practices.correlation import not_correlated
from maypy.distributions import Distribution
from maypy.experiment.interpretation import Interpretation
from maypy.experiment.report import Report
from maypy.experiment.test import Test
import scipy.stats as st


class ChiSquareTest(Test):
    NULL_HYPOTHESIS = "The elements of P and Q *ARE* from the same underlying distribution"
    ALTERNATE_HYPOTHESIS = "The elements of P and Q *ARE NOT* from the same underlying distribution"

    def check_assumptions(self, report, P: Distribution, Q: Optional[Distribution] = None):
        """

        :param report:
        :param P:
        :param Q:
        :return:
        """
        report.add_assumption("P is continuous", P.is_continuous)

        if Q is not None:
            report.add_assumption("Q is continuous", Q.is_continuous)
            report.add_assumption("P & Q not correlated", bool(not_correlated(P, Q)))

        return report

    def normality(self, P):
        """

        :param P:
        :return:
        """
        statistic, p_value = st.chisquare(P.data)

        report = Report("Chi Square Normality",
                        "chi-statistic",
                        conclusion="Is Normal",
                        statistic=statistic,
                        p_value=p_value,
                        h0_rejected=lambda alpha: p_value < alpha,
                        interpretation=lambda alpha: Interpretation(p_value > alpha))

        self.experiment[P] = report
        return self.check_assumptions(report, P)

    def goodness_of_fit(self, P: Distribution, Q: Distribution):
        """

        :param P:
        :param Q:
        :return:
        """
        statistic, p_value = st.chisquare(P.data, Q.data)
        report = Report("goodness_of_fit",
                        "chi-statistic",
                        conclusion="P resembles Q",
                        statistic=statistic,
                        p_value=p_value,
                        h0_rejected=lambda alpha: p_value < alpha,
                        interpretation=lambda alpha: Interpretation(p_value > alpha))

        self.experiment[P] = report
        return self.check_assumptions(report, P, Q)
