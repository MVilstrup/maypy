from typing import Optional

from maypy.best_practices.checks import is_continuous, minimum_sample_size
from maypy.distributions import Distribution
from maypy.experiment.report import Report
from maypy.experiment.test import Test
import scipy.stats as st


class JarqueBeraNormality(Test):
    def check_assumptions(self, P: Distribution, Q: Optional[Distribution] = None):
        helper = "The Jarque Bera Normality Test can only handle continuous distributions"
        self.distribution_assumption(is_continuous, P, helper)

        helper = "The Jarque Bera Normality Test requires more than 2000 samples"
        self.distribution_assumption(minimum_sample_size, P, helper, requirement=2000)

    def __call__(self, P: Distribution):
        statistic, p_value = st.jarque_bera(P.data)
        report = Report("Jarque Bera Normality",
                        "statistic",
                        statistic=statistic,
                        p_value=p_value,
                        h0_rejected=lambda alpha: p_value > alpha,
                        interpretation=lambda alpha: p_value > alpha)

        self.experiment[P] = report
        return report
