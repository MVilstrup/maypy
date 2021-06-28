from typing import Optional

from maypy.distributions import Distribution
from maypy.experiment.report import Report
from maypy.experiment.test import Test
import scipy.stats as st


class PearsonCorrelationTest(Test):

    def check_assumptions(self, report, P: Distribution, Q: Optional[Distribution] = None):
        """

        :param report:
        :param P:
        :param Q:
        :return:
        """
        report.add_assumption("P is continuous", P.is_continuous)
        report.add_assumption("Q is continuous", Q.is_continuous)
        return report

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
        return self.check_assumptions(report, P, Q)

    def not_correlated(self, P: Distribution, Q: Distribution):
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
        return self.check_assumptions(report, P, Q)
