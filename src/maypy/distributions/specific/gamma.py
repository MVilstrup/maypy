from maypy.distributions.distribution import Distribution
import scipy.stats as st


class Gamma(Distribution):
    def __init__(self, data, num_samples=-1, args=None, loc=None, scale=None, experiment=None):
        Distribution.__init__(self, data, st.gamma, num_samples, args, loc, scale, experiment)

    @staticmethod
    def example():
        return Gamma(st.gamma(a=1.99).rvs(1000))