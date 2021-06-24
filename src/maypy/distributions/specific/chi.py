from maypy.distributions.distribution import Distribution
import scipy.stats as st


class Chi(Distribution):
    def __init__(self, data, num_samples=-1, args=None, loc=None, scale=None, experiment=None):
        Distribution.__init__(self, data, st.chi, num_samples, args, loc, scale, experiment)

    @staticmethod
    def example():
        return Chi(st.chi(df=78).rvs(1000))
