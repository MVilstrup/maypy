from maypy.distributions.distribution import Distribution
import scipy.stats as st


class DGamma(Distribution):
    def __init__(self, data, num_samples=-1, args=None, loc=None, scale=None, experiment=None):
        Distribution.__init__(self, data, st.dgamma, num_samples, args, loc, scale, experiment)

    @staticmethod
    def example():
        return DGamma(st.dgamma(a=1.1).rvs(1000))
