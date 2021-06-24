import scipy.stats as st
import warnings
import pandas as pd
import numpy as np
from functools import lru_cache
import inspect
from collections import namedtuple, defaultdict

from maypy.best_practices.sample_processing import prepare_data, sample_bins
from maypy.experiment.experiment import Experiment
from maypy.utils import Document


class Distribution:

    def __init__(self, data, distribution, num_samples=-1, args=None, loc=None, scale=None, experiment=None):
        self.name = None
        self.distribution_name = distribution.name.capitalize()
        self.data = prepare_data(data, max_size=num_samples)
        self.data_bins, self.data_observations = sample_bins(self.data)

        self.args = args
        self.loc = loc
        self.scale = scale
        self.base = distribution
        self._dist = None
        self._experiment = experiment
        self.properties = defaultdict(dict)

    @staticmethod
    def example():
        raise NotImplementedError()

    @property
    def dist(self):
        if self._dist is None:
            with warnings.catch_warnings():
                warnings.filterwarnings('ignore')
                if self.args is None:
                    self.args = self.base.fit(self.data)
                    self.loc, self.scale = self.args[-2], self.args[-1]

                self._dist = self.base(*self.args)
        return self._dist

    @property
    def kwargs(self):
        if self.args:
            return dict(zip(inspect.signature(self.base.__init__).parameters.keys(), self.args))

        return {}

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

    @property
    def stats(self):
        return self.dist.stats()

    @property
    def entropy(self):
        return self.dist.entropy()

    @property
    @lru_cache
    def chi_square(self):
        Result = namedtuple("Result", ["statistic", "p_value"])
        statistic, p_value = st.chisquare(f_obs=self.data, f_exp=self.sample())
        return Result(statistic, p_value)

    @property
    def chi_statistic(self):
        return self.chi_square.statistic

    def is_valid(self, alpha=0.05):
        return self.chi_square.p_value < alpha

    def pdf(self, x=None):
        if x is None:
            x = len(self)
        return self.dist.pdf(x)

    def cdf(self, x=None):
        if x is None:
            x = len(self)
        return self.dist.cdf(x)

    def sf(self, x=None):
        if x is None:
            x = len(self)
        return self.dist.sf(x)

    def probability_densities(self, x=None):
        return self.pdf(x)

    def cummulative_densities(self, x=None):
        return self.cdf(x)

    def survival(self, x=None):
        return self.sf(x)

    @property
    def plot(self):
        from maypy.plot import Plot
        return Plot(self)

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

    def __setitem__(self, key, value):
        self.properties[key] = value

    def __contains__(self, item):
        return item in self.properties

    def __getitem__(self, item):
        return self.properties[item]

    def __len__(self):
        return len(self.data)

    def __ge__(self, other):
        from maypy.distributions.distribution_pair import DistributionPair
        self.name = "P"
        other.name = "Q"
        experiment = Experiment("Distributions has different mean", self, other)
        DistributionPair.mean_greater_than(self, other, experiment)
        return experiment

    def __le__(self, other):
        from maypy.distributions.distribution_pair import DistributionPair
        self.name = "P"
        other.name = "Q"
        experiment = Experiment("Distributions has different mean", self, other)
        DistributionPair.mean_less_than(self, other, experiment)
        return experiment

    def __ne__(self, other):
        from maypy.distributions.distribution_pair import DistributionPair
        self.name = "P"
        other.name = "Q"
        experiment = Experiment("Distributions has different mean", self, other)
        DistributionPair.mean_not_equal(self, other, experiment)
        return experiment

    def __iter__(self):
        return iter(self.data)

    def __repr__(self):
        R = lambda x: round(x, 3) if isinstance(x, float) else x
        doc = Document(3)
        T = type(self).__name__

        # Sample information
        doc.row[0: f"Sample Size: {len(self.data)}"]
        doc.row[1:f"Mean: {R(np.mean(self.data))}",
                2:f"Variance: {R(np.var(self.data))}",
                3: f"Median: {R(np.median(self.data))}"]
        doc.row

        # Distribution information
        doc.row[0: f"Distribution: {self.name}({T})" if self.name else f"Distribution: {T}",
                1: f"RSS: {R(self.rss)}",
                2: f"Energy: {R(self.energy)}",]
        doc.row[1: f"Mean: {R(self.dist.mean())}",
                2: f"Variance: {R(self.dist.var())}",
                3: f"Median: {R(self.dist.median())}"]
        doc.row

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


        # Discovered Properties
        if self.properties:
            doc.row[0: "Tested Properties"]
            for name, value in self.properties.items():
                doc.row[0: name: "right"][1: bool(value)]

        return repr(doc)






























