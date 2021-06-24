from maypy.distributions.distribution import Distribution
import scipy.stats as st

class LogGamma(Distribution):
    def __init__(self, data, num_samples=-1, args=None, loc=None, scale=None, experiment=None):
        Distribution.__init__(self, data, st.loggamma, num_samples, args, loc, scale, experiment)

    @staticmethod
    def example():
        return LogGamma(st.loggamma(c=0.414).rvs(1000))