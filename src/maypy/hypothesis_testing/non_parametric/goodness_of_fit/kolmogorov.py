from typing import Optional

from maypy import ALPHA
from maypy.best_practices.correlation import not_correlated
from maypy.distributions import Distribution
from maypy.experiment.report import Report
from maypy.experiment.test import Test
import scipy.stats as st
import numpy as np


class KolmogorovSmirnovTest(Test):
    ALPHA_CRIT_TABLE = dict(zip(ALPHA.ALLOWED, [1.22, 1.36, 1.48, 1.63, 1.73, 1.95]))

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

    @staticmethod
    def critical_d(P_size: int, Q_size: int, alpha):
        """

        :param P_size:
        :param Q_size:
        :param alpha:
        :return:
        """
        crit_component = KolmogorovSmirnovTest.ALPHA_CRIT_TABLE[alpha]
        return crit_component * np.sqrt((P_size + Q_size) / (P_size * Q_size))

    def _ks(self, P: Distribution, Q: Distribution, alternative: str, conclusion):
        """

        :param P:
        :param Q:
        :param alternative:
        :param conclusion:
        :return:
        """
        statistic, p_value = st.ks_2samp(P.data, Q.data, alternative=alternative)

        critical_d = lambda alpha: KolmogorovSmirnovTest.critical_d(len(P), len(Q), alpha)

        sided = "one-sided" if alternative != "two-sided" else alternative
        report = Report(f"Kolmogorov-Smirnov",
                        "ks-statistic",
                        test_description=f"({sided.capitalize()}) Different Mean",
                        conclusion=conclusion,
                        statistic=statistic,
                        p_value=p_value,
                        h0_rejected=lambda alpha: p_value < alpha,
                        # The p_value should be less than alpha to be significant
                        interpretation=lambda alpha: p_value < alpha)

        report.at_alpha(critical_d=critical_d, below_critical_d=lambda alpha: statistic < critical_d(alpha))

        self.experiment[(P, Q)] = report

        return report

    def less_than(self, P: Distribution, Q: Distribution):
        """

        :param P:
        :param Q:
        :return:
        """
        report = self._ks(P, Q, "less", "Q-Mean Smaller")
        return self.check_assumptions(report, P, Q)

    def greater_than(self, P: Distribution, Q: Distribution):
        """

        :param P:
        :param Q:
        :return:
        """
        report = self._ks(P, Q, "greater", "Q-Mean Greater")
        return self.check_assumptions(report, P, Q)

    def not_equal(self, P: Distribution, Q: Distribution):
        """

        :param P:
        :param Q:
        :return:
        """
        report = self._ks(P, Q, "two-sided", "Means Different")
        return self.check_assumptions(report, P, Q)

    def normality(self, P):
        """

        :param P:
        :return:
        """
        statistic, p_value = st.kstest(P.data, 'norm')
        report = Report("Kolmogorov-Smirnov ",
                        "ks-statistic",
                        test_description=f"Normality",
                        conclusion="Is Normal",
                        statistic=statistic,
                        p_value=p_value,
                        h0_rejected=lambda alpha: p_value < alpha,
                        interpretation=lambda alpha: p_value > alpha)

        self.experiment[P] = report
        return self.check_assumptions(report, P)
