from typing import Optional

from maypy.best_practices.checks import is_continuous
from maypy.distributions import Distribution
from maypy.experiment.report import Report
from maypy.experiment.test import Test
import scipy.stats as st


class PearsonCorrelationTest(Test):

    def check_assumptions(self, P: Distribution, Q: Optional[Distribution] = None):
        helper = "The Pearson Correlation Test can only handle continuous distributions"
        self.distribution_assumption(is_continuous, P, helper)
        self.distribution_assumption(is_continuous, Q, helper)

    @staticmethod
    def interpretation(coefficient):
        type = "Negative" if coefficient < 0 else "Positive"

        strength_values = [
            (0.1, "Trivial"),
            (0.3, "Moderate"),
            (0.5, "Strong"),
            (1.0, "Perfect"),
        ]

        for min_coef, strength in strength_values:
            if abs(coefficient) <= min_coef:
                return {"correlation_type": type, "correlation_strenght": strength}

        return {"correlation_type": type, "correlation_strenght": "Error"}

    def correlated(self, P: Distribution, Q: Distribution):
        self.check_assumptions(P, Q)

        correlation_coefficient, p_value = st.pearsonr(P.data, Q.data)

        # The correlation should be above 0.6 for the null hypothesis to be rejected
        # and p_value should be less than alpha to be significant
        report = Report("Pearson Correlation",
                        "correlation_coefficient",
                        statistic=correlation_coefficient,
                        p_value=p_value,
                        h0_rejected=lambda alpha: p_value < alpha,
                        interpretation=lambda alpha: (p_value < alpha) and (correlation_coefficient > 0.6))

        self.experiment[(P, Q)] = report
        return report

    def not_correlated(self, P: Distribution, Q: Distribution):
        self.check_assumptions(P, Q)

        correlation_coefficient, p_value = st.pearsonr(P.data, Q.data)

        # The correlation should be above 0.6 for the null hypothesis to be rejected
        # and p_value should be less than alpha to be significant
        report = Report("Pearson Correlation",
                        "correlation_coefficient",
                        statistic=correlation_coefficient,
                        p_value=p_value,
                        h0_rejected=lambda alpha: p_value > alpha,
                        interpretation=lambda alpha: (p_value > alpha) or (correlation_coefficient < 0.6))

        self.experiment[(P, Q)] = report
        return report
