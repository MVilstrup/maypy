from typing import Tuple
from maypy.distributions import Distribution
from maypy.experiment.interpretation import Interpretation
from maypy.experiment.report import Report
from maypy.experiment.test import Test
import scipy.stats as st
from collections import Counter


class LeveneTest(Test):
    null_hypothesis = "The Levene test tests the null hypothesis that all input samples are from populations with equal variances"

    def __call__(self, *distributions: Tuple[Distribution]):
        """
        Sources:
        - https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.levene.html
        - https://en.wikipedia.org/wiki/Kurtosis

        :param P:
        :return:
        """
        most_common_skewness = Counter([P.skewness for P in distributions]).most_common(1)[0]
        most_common_kurtosis = Counter([P.kurtosis for P in distributions]).most_common(1)[0]

        if most_common_skewness != "normal":
            center = "median"
        else:
            if most_common_kurtosis == "peaked":
                center = "trimmed"
            else:
                center = "mean"

        statistic, p_value = st.levene(*[P.data for P in distributions], center=center)
        report = Report(f"Levene ({center})",
                        "statistic",
                        test_description=f"Equality of Variance",
                        conclusion="Equal Variance",
                        statistic=statistic,
                        p_value=p_value,
                        h0_rejected=lambda alpha: p_value > alpha,
                        interpretation=lambda alpha: Interpretation(p_value > alpha))

        return report
