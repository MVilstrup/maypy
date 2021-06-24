from maypy.distributions.distribution import Distribution
import scipy.stats as st


class PowerLogNorm(Distribution):
    def __init__(self, data, num_samples=-1, args=None, loc=None, scale=None, experiment=None):
        Distribution.__init__(self, data, st.powerlognorm, num_samples, args, loc, scale, experiment)

    @staticmethod
    def example():
        return PowerLogNorm(st.powerlognorm(c=2.14, s=0.446).rvs(1000))
