from maypy.distributions.distribution import Distribution
import scipy.stats as st

class BetaPrime(Distribution):
    def __init__(self, data, num_samples=-1, args=None, loc=None, scale=None, experiment=None):
        Distribution.__init__(self, data, st.betaprime, num_samples, args, loc, scale, experiment)

    @staticmethod
    def example():
        return BetaPrime(st.beta(a=5, b=5).rvs(1000))
