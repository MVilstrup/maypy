import pandas as pd
from maypy.distributions import Distribution


class Plot:
    def __init__(self, distribution: Distribution):
        self._dist = distribution

    def cdp(self):
        import numpy as np
        import matplotlib.pyplot as plt

        sample = np.sort(self._dist.data)  # Or data.sort(), if data can be modified
        distribution = np.sort(self._dist.sample(max_value=max(self._dist.data)))

        # Cumulative counts:
        plt.step(sample, np.arange(sample.size), label="Observed")
        plt.step(distribution, np.arange(distribution.size), label="Estimated")
        plt.legend()
        plt.show()

    def qq(self):
        import statsmodels.api as sm
        sm.qqplot(self._dist.sample(10000))

    def hist(self, bins=100, **kwargs):
        import matplotlib.pyplot as plt
        if "alpha" not in kwargs:
            kwargs["alpha"] = 0.6
            kwargs["label"] = "Observed"

        ax = pd.Series(self._dist.data).hist(bins=bins, **kwargs)
        kwargs.update({"ax": ax, "label": "Estimated"})
        pd.Series(self._dist.sample(max_value=max(self._dist.data))).hist(bins=bins, **kwargs)
        plt.legend()
