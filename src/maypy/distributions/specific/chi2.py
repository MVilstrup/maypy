from maypy.distributions.distribution import Distribution
import scipy.stats as st


class Chi2(Distribution):
    def __init__(self, data, num_samples=-1, args=None, loc=None, scale=None, experiment=None):
        Distribution.__init__(self, data, st.chi2, num_samples, args, loc, scale, experiment)

    @staticmethod
    def example():
        return Chi2(st.chi2(df=55).rvs(1000))
