import pandas as pd
from maypy.distributions import Distribution
import numpy as np
import matplotlib.pyplot as plt


class Plot:
    def __init__(self, distribution: Distribution):
        self._dist = distribution

    def cdf(self):
        def _cdf(x):
            # getting data of the histogram
            count, bins_count = np.histogram(x, bins=100)
            pdf = count / sum(count)
            return np.arange(len(pdf)), np.cumsum(pdf)

        # ax = sns.lineplot(x=self._dist.data, y=st.norm.cdf(self._dist.data), label="Observed")
        plt.plot(*_cdf(self._dist.data), label="Observed")
        plt.plot(*_cdf(self._dist.sample()), label="Estimated")
        plt.legend()

    def qq(self):
        import statsmodels.api as sm
        sm.qqplot(self._dist.sample(10000))

    def hist(self, bins=100, **kwargs):
        if "alpha" not in kwargs:
            kwargs["alpha"] = 0.6
            kwargs["label"] = "Observed"

        ax = pd.Series(self._dist.data).hist(bins=bins, **kwargs)
        kwargs.update({"ax": ax, "label": "Estimated"})
        pd.Series(self._dist.sample(max_value=max(self._dist.data))).hist(bins=bins, **kwargs)
        plt.legend()
