from typing import Optional

from maypy.best_practices.checks import is_continuous
from maypy.best_practices.correlation import is_correlated
from maypy.distributions import Distribution
from maypy.experiment.report import Report
from maypy.experiment.test import Test
import scipy.stats as st


class ChiSquareTest(Test):
    NULL_HYPOTHESIS = "The elements of P and Q *ARE* from the same underlying distribution"
    ALTERNATE_HYPOTHESIS = "The elements of P and Q *ARE NOT* from the same underlying distribution"

    def check_assumptions(self, P: Distribution, Q: Optional[Distribution] = None):
        helper = "The ChiSquare Test can only handle continuous distributions"
        self.distribution_assumption(is_continuous, P, helper)
        self.distribution_assumption(is_continuous, Q, helper)

        helper = "The ChiSquare Test cannot handle two correlated distributions"
        self.pair_assumption(is_correlated, P, Q, helper)

    def normality(self, P, test_sample=False):
        self.check_assumptions(P)

        if test_sample:
            statistic, p_value = st.chisquare(P.data, P.sample())
        else:
            statistic, p_value = st.chisquare(P.data)

        report = Report("Chi Square Normality",
                        "chi-statistic",
                        statistic=statistic,
                        p_value=p_value,
                        h0_rejected=lambda alpha: p_value < alpha,
                        interpretation=lambda alpha: p_value > alpha)

        self.experiment[P] = report
        return report

    def goodness_of_fit(self, P: Distribution, Q: Distribution):
        self.check_assumptions(P, Q)
        statistic, p_value = st.chisquare(P.data, Q.data)
        report = Report("goodness_of_fit",
                        "chi-statistic",
                        statistic=statistic,
                        p_value=p_value,
                        h0_rejected=lambda alpha: p_value < alpha,
                        interpretation=lambda alpha: p_value > alpha)

        self.experiment[P] = report
        return report
