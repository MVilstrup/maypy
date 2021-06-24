import warnings

from maypy import ALPHA
from maypy.best_practices.normality import is_normal
from maypy.distributions import Normal
from maypy.distributions import ALL_NON_PARAMETRIC_DISTRIBUTIONS, COMMON_NON_PARAMETRIC
import numpy as np
from maypy.experiment.experiment import Experiment


def fit(data, num_samples=5000, alpha=ALPHA.DEFAULT, measure="rss", only_common=False):
    # Find the correct size of the sub_sample (To reduce computation time in case of major samples)
    num_samples = min(num_samples, len(data)) if num_samples > 0 else len(data)

    # Make a subsample if necessary
    if len(data) > num_samples:
        data = np.random.choice(data, num_samples)

    with warnings.catch_warnings():
        warnings.filterwarnings('ignore')

        measure = measure.lower()
        if measure not in ['rss', 'wasserstein', 'energy']:
            raise ValueError("measure should be one of ['rss', 'wasserstein', 'energy']")

        with Experiment("Find Distribution for Sample", "Distribution Search") as exp:
            with ALPHA(alpha):
                # Check if the subsample is Gaussian distributed
                as_normal = Normal(data, experiment=exp)
                if is_normal(as_normal, exp):
                    return as_normal

                DISTRIBUTIONS = COMMON_NON_PARAMETRIC if only_common else ALL_NON_PARAMETRIC_DISTRIBUTIONS

                # If it is not normal, run the sample through all Non Parametric Distributions
                distributions = [lazy_dist(data) for lazy_dist in DISTRIBUTIONS]

                # Filter all distributions which are not valid given a Chi Squared Goodness of Fit Test
                distributions = [lazy_dist for lazy_dist in distributions if lazy_dist.is_valid(alpha)]

                if only_common and not distributions:
                    return fit(data, num_samples, alpha, measure, only_common=False)

                # Return the distribution with the lowest value of the chosen measure
                if measure == "rss":
                    method = lambda lazy_dist: lazy_dist.rss
                elif measure == "wasserstein":
                    method = lambda lazy_dist: lazy_dist.wasserstein
                else:
                    method = lambda lazy_dist: lazy_dist.energy

                return min(distributions, key=method)
