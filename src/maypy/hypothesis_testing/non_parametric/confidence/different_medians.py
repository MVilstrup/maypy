from maypy import ALPHA
from maypy.distributions import Distribution
import scipy.stats as st
import numpy as np


def unpaired_non_parametric_different_medians(P: Distribution, Q: Distribution):
    P_size, Q_size = len(P), len(Q)
    N = st.norm.ppf(1 - (float(ALPHA) / 2))

    # The confidence interval for the difference between the two population
    # medians is derived through the n x m differences.
    differences = P - Q

    # the Kth smallest to the Kth largest of the n x m differences then determine
    # the confidence interval, where K is:
    k = np.math.ceil(P_size * Q_size / 2 - (N * (P_size * Q_size * (P_size + Q_size + 1) / 12) ** 0.5))

    return differences[k - 1], differences[len(differences) - k]
