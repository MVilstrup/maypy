from maypy.distributions.distribution import Distribution
import scipy.stats as st


class TurkeyLambda(Distribution):
    def __init__(self, data, num_samples=-1, args=None, loc=None, scale=None, experiment=None):
        Distribution.__init__(self, data, st.tukeylambda, num_samples, args, loc, scale, experiment)

    @staticmethod
    def example():
        return TurkeyLambda(st.tukeylambda(lam=3.13).rvs(1000))
