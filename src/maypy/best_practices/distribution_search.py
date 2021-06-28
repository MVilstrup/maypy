import warnings

from maypy import ALPHA
from maypy.best_practices.normality import is_normal
from maypy.distributions import Normal
from maypy.distributions import NP_DISTRIBUTIONS
import numpy as np
from maypy.experiment.experiment import Experiment


def fit(data, num_samples=5000, alpha=ALPHA.DEFAULT, measure="rss"):
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

                # If it is not normal, run the sample through all Non Parametric Distributions
                distributions = [as_normal] + [lazy_dist(data, experiment=exp) for lazy_dist in NP_DISTRIBUTIONS]

                # Filter all distributions which are not valid given a Chi Squared Goodness of Fit Test
                distributions = [lazy_dist for lazy_dist in distributions if lazy_dist.is_valid]

                # Return the distribution with the lowest value of the chosen measure
                if measure == "rss":
                    method = lambda lazy_dist: lazy_dist.rss
                elif measure == "wasserstein":
                    method = lambda lazy_dist: lazy_dist.wasserstein
                else:
                    method = lambda lazy_dist: lazy_dist.energy

                distributions = sorted(distributions, key=method)

                best_fit = distributions[0]
                best_fit.set_alternatives(distributions)
                return best_fit
