import inspect
import warnings
from collections import namedtuple
from functools import lru_cache

from maypy.best_practices.sample_processing import prepare_data, sample_bins
from pandas.core.dtypes.common import is_numeric_dtype
import numpy as np
import pandas as pd


class DistributionProperties:
    def __init__(self, data, distribution, num_samples):
        self.data = prepare_data(data, max_size=num_samples)
        self.base = distribution
        self.data_bins, self.data_observations = sample_bins(self.data)

    @property
    def sample_mean(self):
        return np.mean(self.data)

    @property
    def sample_variance(self):
        return np.var(self.data)

    @property
    def sample_median(self):
        return np.median(self.data)

    @property
    def sample_std(self):
        return np.std(self.data)

    @property
    def mean(self):
        return self.dist.mean()

    @property
    def variance(self):
        return self.dist.var()

    @property
    def median(self):
        return self.dist.median()

    @property
    def std(self):
        return self.dist.std()

    @property
    def stats(self):
        return self.dist.stats()

    @property
    def entropy(self):
        return self.dist.entropy()

    @property
    @lru_cache
    def is_continuous(self):
        """
        Is is impossible to estimate whether numbers are continuous or discrete,
        but we can make a few checks to ensure no explicitly invalid data suchs as strings is used
        """
        return all([
            is_numeric_dtype(self.data),  # Ensure all data is numeric
            len(np.unique(self.data)) > 2,  # Ensure there is no Boolean columns
            pd.isna(self.data).sum() == 0,  # Ensure no None values is used
        ])

    @property
    @lru_cache
    def is_parametric(self):
        return self.base.name == "norm"

    @property
    def CI(self):
        CIValues = namedtuple("CIValues", ["lower", "upper"])

        class ConfidenceIntervals:
            def __init__(self, props):
                self.props = props

            @property
            def parametric(self):
                return CIValues(lambda alpha: self.props.ppf(1 - alpha), lambda alpha: self.props.ppf(alpha))

            @property
            def quantile(self):
                X = self.props.sample()
                return CIValues(lambda alpha: np.quantile(X, 1 - alpha), lambda alpha: np.quantile(X, alpha))

            @property
            def percentile(self):
                X = self.props.sample()
                return CIValues(lambda alpha: np.percentile(X, (1 - (alpha / 2)) * 100),
                                lambda alpha: np.percentile(X, (0 + (alpha / 2)) * 100))

        return ConfidenceIntervals(self)

    @property
    def plot(self):
        from maypy.plot import Plot
        return Plot(self)

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
