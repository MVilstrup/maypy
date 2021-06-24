from maypy.distributions.specific.pareto import Pareto
from maypy.distributions.specific.turkey_lambda import TurkeyLambda
from maypy.distributions.specific.alpha import Alpha
from maypy.distributions.specific.gamma import Gamma
from maypy.distributions.specific.exponential_norm import ExponentialNorm
from maypy.distributions.specific.exponential import Exponential
from maypy.distributions.specific.logistic import Logistic
from maypy.distributions.specific.power_norm import PowerNorm
from maypy.distributions.specific.power_log_norm import PowerLogNorm
from maypy.distributions.specific.lognorm import LogNorm
from maypy.distributions.specific.dweibull import DWeibull
from maypy.distributions.specific.d_gamma import DGamma
from maypy.distributions.specific.cosine import Cosine
from maypy.distributions.specific.chi import Chi
from maypy.distributions.specific.chi2 import Chi2
from maypy.distributions.specific.uniform import Uniform
from maypy.distributions.specific.beta import Beta
from maypy.distributions.specific.beta_prime import BetaPrime
from maypy.distributions.specific.log_gamma import LogGamma

from maypy.distributions.specific.normal import Normal

from maypy.distributions.distribution import Distribution
from maypy.distributions.distribution_pair import DistributionPair

ALL_NON_PARAMETRIC_DISTRIBUTIONS = [
    Pareto, TurkeyLambda, Alpha, Gamma, ExponentialNorm,
    Exponential, Logistic, PowerNorm, PowerLogNorm, LogNorm, DWeibull,
    DGamma, Cosine, Chi2, Chi, Uniform, BetaPrime, Beta, LogGamma
]

COMMON_NON_PARAMETRIC = [Exponential, Logistic, Uniform, DWeibull, LogNorm, Pareto]
