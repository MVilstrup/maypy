from maypy.distributions.distribution import Distribution
import scipy.stats as st


class Logistic(Distribution):
    def __init__(self, data, num_samples=-1, args=None, loc=None, scale=None, experiment=None):
        Distribution.__init__(self, data, st.logistic, num_samples, args, loc, scale, experiment)

    @staticmethod
    def example():
        return Logistic(st.logistic().rvs(1000))
