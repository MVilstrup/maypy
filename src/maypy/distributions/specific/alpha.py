from maypy.distributions.distribution import Distribution
import scipy.stats as st


class Alpha(Distribution):
    def __init__(self, data, num_samples=-1, args=None, loc=None, scale=None, experiment=None):
        Distribution.__init__(self, data, st.alpha, num_samples, args, loc, scale, experiment)

    @staticmethod
    def example():
        return Alpha(st.alpha(a=3.57).rvs(1000))
