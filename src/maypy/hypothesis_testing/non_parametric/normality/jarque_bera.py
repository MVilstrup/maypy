from typing import Optional

from maypy.distributions import Distribution
from maypy.experiment.report import Report
from maypy.experiment.test import Test
import scipy.stats as st


class JarqueBeraNormality(Test):
    def check_assumptions(self, report, P: Distribution, Q=None):
        """

        :param report:
        :param P:
        :param Q:
        :return:
        """
        report.add_assumption("P is continuous", P.is_continuous)
        report.add_assumption("Data Sample > 2000", len(P) > 2000)
        return report

    def __call__(self, P: Distribution):
        statistic, p_value = st.jarque_bera(P.data)
        report = Report("Jarque Bera Normality",
                        "statistic",
                        statistic=statistic,
                        p_value=p_value,
                        h0_rejected=lambda alpha: p_value > alpha,
                        interpretation=lambda alpha: p_value > alpha)

        self.experiment[P] = report
        return self.check_assumptions(report, P)
