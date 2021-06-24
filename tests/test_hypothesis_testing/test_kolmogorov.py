from maypy.distributions import ALL_NON_PARAMETRIC_DISTRIBUTIONS
from maypy.hypothesis_testing.non_parametric.goodness_of_fit.kolmogorov import KolmogorovSmirnovTest

def test_on_identical_distributions():
    for distribution in ALL_NON_PARAMETRIC_DISTRIBUTIONS:
        P = distribution.example()
        result = KolmogorovSmirnovTest().not_equal(P, P)
        print(result)
        assert not result["test"]