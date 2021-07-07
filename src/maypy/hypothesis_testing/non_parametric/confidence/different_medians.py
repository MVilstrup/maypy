from maypy.distributions import Distribution
import scipy.stats as st
import numpy as np
from maypy.experiment.result import Result


def unpaired_non_parametric_different_medians(P: Distribution, Q: Distribution):
    R = lambda x: round(x, 1)
    P_size, Q_size = len(P), len(Q)

    # The confidence interval for the difference between the two population
    # medians is derived through the n x m differences.
    differences = P - Q

    def difference_at_alpha(alpha):
        np.random.seed(34)

        N = st.norm.ppf(1 - (alpha / 2))
        # the Kth smallest to the Kth largest of the n x m differences then determine
        # the confidence interval, where K is:
        k = np.math.ceil(P_size * Q_size / 2 - (N * (P_size * Q_size * (P_size + Q_size + 1) / 12) ** 0.5))
        lower, upper = differences[k - 1], differences[-k]
        return lower, upper

    def overview(alpha):
        lower, upper = difference_at_alpha(alpha)
        if lower < 0 and upper < 0:
            return "P << Q"
        if lower > 0 and upper > 0:
            return "P >> Q"
        if lower < 0 and abs(lower) > abs(upper):
            return "P < Q"
        if lower < 0 and abs(lower) < abs(upper):
            return "P > Q"
        else:
            return "P ≈ Q"

    def explanation(alpha):
        sign = lambda result: "+" if result > 0 else "-"
        lower, upper = map(R, difference_at_alpha(alpha))

        return f"P|{R((1 - alpha) * 100)}% ≈ (Q{sign(lower)}{abs(lower)}, Q{sign(upper)}{abs(upper)})"

    return Result("Different Medians", "Kth largest difference",
                  explanation=explanation,
                  overview=overview,
                  confidence_interval=lambda alpha: tuple(map(R, difference_at_alpha(alpha))))
