from typing import Optional

from maypy.distributions import Distribution
from maypy.experiment.interpretation import Interpretation
from maypy.experiment.report import Report
from maypy.experiment.test import Test
import scipy.stats as st
import numpy as np


class AndersonDarlingNormality(Test):

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
        result = st.anderson(P.data)

        test_alphas = np.array(result.significance_level) / 100

        valid_alphas = test_alphas[result.statistic < np.array(result.critical_values)]

        report = Report("Anderson-Darling Normality",
                        "AD-statistic",
                        statistic=None,
                        p_value=None,
                        h0_rejected=lambda alpha: len(valid_alphas) > 0 and valid_alphas.max() >= alpha,
                        interpretation=lambda alpha: Interpretation(len(valid_alphas) > 0 and valid_alphas.max() >= alpha))

        self.experiment[P] = report
        return self.check_assumptions(report, P)
