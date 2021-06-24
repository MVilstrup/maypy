from maypy.distributions.distribution import Distribution
import scipy.stats as st

class ExponentialNorm(Distribution):
    def __init__(self, data, num_samples=-1, args=None, loc=None, scale=None, experiment=None):
        Distribution.__init__(self, data, st.exponnorm, num_samples, args, loc, scale, experiment)

    @staticmethod
    def example():
        return ExponentialNorm(st.exponnorm(K=1.5).rvs(1000))
