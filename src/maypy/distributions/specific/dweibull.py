from maypy.distributions.distribution import Distribution
import scipy.stats as st

class DWeibull(Distribution):
    def __init__(self, data, num_samples=-1, args=None, loc=None, scale=None, experiment=None):
        Distribution.__init__(self, data, st.dweibull, num_samples, args, loc, scale, experiment)

    @staticmethod
    def example():
        return DWeibull(st.dweibull(c=2.07).rvs(1000))
