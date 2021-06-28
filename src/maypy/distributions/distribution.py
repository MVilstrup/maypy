from itertools import product

import scipy.stats as st
import numpy as np
from functools import lru_cache
from collections import namedtuple

from maypy import ALPHA
from maypy.distributions.properties import DistributionProperties

from maypy.experiment.experiment import Experiment
from maypy.utils import Document


class Distribution(DistributionProperties):

    def __init__(self, data, distribution, num_samples=-1, args=None, loc=None, scale=None, experiment=None,
                 alternatives=None):
        super().__init__(data, distribution, num_samples)
        self.name = None
        self.distribution_name = distribution.name.capitalize()

        self.args = args
        self.loc = loc
        self.scale = scale
        self._dist = None
        self._experiment = experiment
        self._alternatives = alternatives if alternatives is not None else []

    @staticmethod
    def example():
        raise NotImplementedError()

    @property
    def class_name(self):
        return type(self).__name__

    @property
    def CLT(self):
        from maypy.distributions import Normal

        num_samples = 10000
        sample_size = len(self.data)

        population_mean = np.array([np.random.choice(self.data, sample_size).mean() for _ in range(num_samples)]).mean()
        sample_std = np.std(self.data) / np.sqrt(sample_size)

        return self.copy(distribution=Normal, args=(population_mean, sample_std))

    @property
    @lru_cache
    def chi_square(self):
        Result = namedtuple("Result", ["statistic", "p_value"])
        statistic, p_value = st.chisquare(f_obs=self.data, f_exp=self.sample())
        return Result(statistic, p_value)

    @property
    def chi_statistic(self):
        return self.chi_square.statistic

    def is_valid(self):
        return self.chi_square.p_value < ALPHA

    @property
    @lru_cache
    def rss(self):
        score = np.power(self.data_observations - self.pdf(self.data_bins), 2.0).sum()
        return score if score > 0 else np.inf

    @property
    @lru_cache
    def wasserstein(self):
        score = st.wasserstein_distance(self.data_observations, self.pdf(self.data_bins))
        return score if score > 0 else np.inf

    @property
    @lru_cache
    def energy(self):
        score = st.energy_distance(self.data_observations, self.pdf(self.data_bins))
        return score if score > 0 else np.inf

    def set_alternatives(self, alternatives):
        for alt in alternatives:
            if hasattr(self, f"as_{alt.class_name}") or alt.class_name == self.class_name:
                continue

            setattr(self, f"as_{alt.class_name}", alt)
            self._alternatives.append(alt)
            alt.set_alternatives(alternatives)

    def pdf(self, x=None):
        if x is None:
            x = len(self)
        return self.dist.pdf(x)

    def ppf(self, value):
        return self.dist.ppf(value)

    def cdf(self, x=None):
        if x is None:
            x = sorted(self.data)
        return self.dist.cdf(x)

    def sf(self, x=None):
        if x is None:
            x = self.data
        return self.dist.sf(x)

    def probability_densities(self, x=None):
        return self.pdf(x)

    def cummulative_densities(self, x=None):
        return self.cdf(x)

    def survival(self, x=None):
        return self.sf(x)

    def copy(self, distribution=None, args=None, loc=None, scale=None):
        distribution = type(self) if distribution is None else distribution
        return distribution(data=self.data,
                            num_samples=-1,
                            args=self.args if args is None else args,
                            loc=self.loc if loc is None else loc,
                            scale=self.scale if scale is None else scale)

    def sample(self, sample_size=None, max_value=None):
        if sample_size is None:
            sample_size = len(self)
        estimated = self.dist.rvs(sample_size)
        return estimated[estimated <= max_value] if max_value else estimated

    def __len__(self):
        return len(self.data)

    def __sub__(self, other):
        # When subtracting two distribution we return the n x m differences of the two underlying samples
        return np.sort([i - j for i, j in product(self.data, other.data)])

    def __ge__(self, other):
        from maypy.best_practices.different_means import greater_mean
        self.name, other.name = "P", "Q"
        experiment = Experiment("Distributions has different mean", self, other)
        greater_mean(self, other, experiment)
        return experiment

    def __le__(self, other):
        from maypy.best_practices.different_means import lesser_mean
        self.name, other.name = "P", "Q"
        experiment = Experiment("Distributions has different mean", self, other)
        lesser_mean(self, other, experiment)
        return experiment

    def __ne__(self, other):
        from maypy.best_practices.different_means import different_mean
        self.name, other.name = "P", "Q"
        experiment = Experiment("Distributions has different mean", self, other)
        different_mean(self, other, experiment)
        return experiment

    def __iter__(self):
        return iter(self.data)

    @property
    def summary(self):
        R = lambda x, d=3: round(x, d) if isinstance(x, float) else x
        doc = Document(3)

        doc.row[0: f"Distribution: {self.name}({self.class_name})" if self.name else f"Distribution: {self.class_name}",
                1: f"RSS: {R(self.rss, 5)}"]

        doc.row[1:f"Sample Mean: {R(self.sample_mean)}",
                2:f"Sample Variance: {R(self.sample_variance)}",
                3: f"Sample Median: {R(self.sample_median)}"]
        doc.row[1: f"Mean: {R(self.mean)}",
                2: f"Variance: {R(self.variance)}",
                3: f"Median: {R(self.median)}"]


        # Distribution Properties
        doc.row
        doc.row[0: "Continuous": "right"][1: self.is_continuous]
        doc.row[0: "Parametric": "right"][1: self.is_parametric]

        return repr(doc)

    def __repr__(self):
        R = lambda x: round(x, 3) if isinstance(x, float) else x
        doc = Document(3)

        # @no:format Sample information
        doc.row[0: f"Sample"]
        doc.row
        doc.row[0: f"Size: {len(self.data)}",
                1:f"Mean: {R(self.sample_mean)}",
                2:f"Variance: {R(self.sample_variance)}",
                3: f"Median: {R(self.sample_median)}"]
        doc.row
        doc.row
        # @do:format

        # @no:format Distribution information
        doc.row[0: f"Distribution: {self.name}({self.class_name})" if self.name else f"Distribution: {self.class_name}",
                1: f"RSS: {R(self.rss)}",
                2: f"Energy: {R(self.energy)}"]

        doc.row[1: f"Mean: {R(self.mean)}",
                2: f"Variance: {R(self.variance)}",
                3: f"Median: {R(self.median)}"]
        doc.row
        # @do:format

        # Found Parameters
        doc.row[0: "Parameters":"right"][2:f"Central Limit":"right"]

        rows = []
        for key, value in self.kwargs.items():
            rows.append(doc.row[0: key: "right"][1: value])
        rows.append(doc.row[0: "loc": "right"][1: self.loc])
        rows.append(doc.row[0: "scale": "right"][1: self.loc])

        clt = self.CLT
        rows[0][2: f"Mean":"right"][3: R(clt.dist.mean())]
        rows[1][2: f"Variance":"right"][3: R(clt.dist.var())]
        rows[2][2: f"Median":"right"][3: R(clt.dist.median())]
        doc.row
        doc.row

        # Alternative Distributions
        if self._alternatives:
            doc.row[0: "Alternatives"]
            for i, alt in enumerate(self._alternatives[:3]):
                doc.row[0:i: "right", 1:type(alt).__name__, 2:f"RSS: {R(alt.rss)}"]

        doc.row

        # Distribution Properties
        doc.row[0: "Distribution Properties"]
        doc.row
        doc.row[0: "Continuous": "right"][1: self.is_continuous]
        doc.row[0: "Parametric": "right"][1: self.is_parametric]

        return repr(doc)
