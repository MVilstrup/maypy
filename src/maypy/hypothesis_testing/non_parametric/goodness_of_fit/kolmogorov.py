from typing import Optional

from maypy import ALPHA
from maypy.best_practices.checks import is_continuous
from maypy.best_practices.correlation import is_correlated
from maypy.distributions import Distribution
from maypy.experiment.report import Report
from maypy.experiment.test import Test
import scipy.stats as st
import numpy as np


class KolmogorovSmirnovTest(Test):
    ALPHA_CRIT_TABLE = dict(zip(ALPHA.ALLOWED, [1.22, 1.36, 1.48, 1.63, 1.73, 1.95]))

    def check_assumptions(self, P: Distribution, Q: Optional[Distribution] = None):
        helper = "The Kolmogorov-Smirnov (KS)-Test Test can only handle continuous distributions"
        self.distribution_assumption(is_continuous, P, helper)
        self.distribution_assumption(is_continuous, Q, helper)

        helper = "The Kolmogorov-Smirnov (KS)-Test cannot handle two correlated distributions"
        self.pair_assumption(is_correlated, P, Q, helper)

    @staticmethod
    def critical_d(P_size: int, Q_size: int, alpha):
        crit_component = KolmogorovSmirnovTest.ALPHA_CRIT_TABLE[alpha]
        return crit_component * np.sqrt((P_size + Q_size) / (P_size * Q_size))

    def _ks(self, P: Distribution, Q: Distribution, alternative: str):
        statistic, p_value = st.ks_2samp(P.data, Q.data, alternative=alternative)

        critical_d = lambda alpha: KolmogorovSmirnovTest.critical_d(len(P), len(Q), alpha)

        sided = "one-sided" if alternative != "two-sided" else alternative
        report = Report(f"Kolmogorov-Smirnov({sided.capitalize()}) Different Mean",
                        "ks-statistic",
                        statistic=statistic,
                        p_value=p_value,
                        h0_rejected=lambda alpha: p_value < alpha,
                        # The p_value should be less than alpha to be significant
                        interpretation=lambda alpha: p_value < alpha)

        report.at_alpha(critical_d=critical_d, below_critical_d=lambda alpha: statistic < critical_d(alpha))

        self.experiment[(P, Q)] = report

        return report

    def less_than(self, P: Distribution, Q: Distribution):
        self.check_assumptions(P, Q)
        return self._ks(P, Q, "less").set_conclusion("Q-Mean Smaller")

    def greater_than(self, P: Distribution, Q: Distribution):
        self.check_assumptions(P, Q)
        return self._ks(P, Q, "greater").set_conclusion("Q-Mean Greater")

    def not_equal(self, P: Distribution, Q: Distribution):
        self.check_assumptions(P, Q)
        return self._ks(P, Q, "two-sided").set_conclusion("Means Different")

    def normality(self, P):
        self.check_assumptions(P)

        statistic, p_value = st.kstest(P.data, 'norm')
        report = Report("Kolmogorov-Smirnov Normality",
                        "ks-statistic",
                        statistic=statistic,
                        p_value=p_value,
                        h0_rejected=lambda alpha: p_value < alpha,
                        interpretation=lambda alpha: p_value > alpha)

        self.experiment[P] = report
        return report.set_conclusion("Is Normal")
