from typing import Optional

from maypy.best_practices.checks import is_continuous
from maypy.distributions import Distribution
from maypy.experiment.report import Report
from maypy.experiment.test import Test
import scipy.stats as st
import numpy as np


class AndersonDarlingNormality(Test):

    def check_assumptions(self, P: Distribution, Q: Optional[Distribution] = None):
        helper = "The Anderson-Darling Test can only handle continuous distributions"
        self.distribution_assumption(is_continuous, P, helper)

    def __call__(self, P: Distribution):
        self.check_assumptions(P)

        result = st.anderson(P.data)

        test_alphas = np.array(result.significance_level) / 100

        valid_alphas = test_alphas[result.statistic < np.array(result.critical_values)]

        report = Report("Anderson-Darling Normality",
                        "AD-statistic",
                        statistic=None,
                        p_value=None,
                        h0_rejected=lambda alpha: len(valid_alphas) > 0 and valid_alphas.max() >= alpha,
                        interpretation=lambda alpha: len(valid_alphas) > 0 and valid_alphas.max() >= alpha)

        self.experiment[P] = report
        return report
