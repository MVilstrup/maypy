from maypy.distributions.distribution import Distribution
import scipy.stats as st

class LogNorm(Distribution):
    def __init__(self, data, num_samples=-1, args=None, loc=None, scale=None, experiment=None):
        Distribution.__init__(self, data, st.lognorm, num_samples, args, loc, scale, experiment)

    @staticmethod
    def example():
        return LogNorm(st.lognorm(s=0.954).rvs(1000))
