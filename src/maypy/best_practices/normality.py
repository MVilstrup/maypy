from functools import lru_cache

from maypy.distributions import Distribution


@lru_cache
def is_normal(P: Distribution, experiment):
    from maypy.hypothesis_testing.non_parametric.normality.anderson_darling import AndersonDarlingNormality
    from maypy.hypothesis_testing.non_parametric.goodness_of_fit.chi_square import ChiSquareTest
    from maypy.hypothesis_testing.non_parametric.normality.d_argustino import DArgustinoNormality
    from maypy.hypothesis_testing.non_parametric.normality.lillefors import LilleforsNormality
    from maypy.hypothesis_testing.non_parametric.normality.shapiro_wilk import ShapiroWilkNormality
    from maypy.hypothesis_testing.non_parametric.goodness_of_fit.kolmogorov import KolmogorovSmirnovTest

    normality_tests = [
        AndersonDarlingNormality(experiment),
        DArgustinoNormality(experiment),
        LilleforsNormality(experiment),
        ShapiroWilkNormality(experiment),
        ChiSquareTest(experiment).normality,
        KolmogorovSmirnovTest(experiment).normality,
    ]

    normality_results = [test(P) for test in normality_tests]  # Run all normality tests
    normal_votes = sum(normality_results)  # The amount of tests deeming the data as normally distributed
    non_normal_votes = len(normality_results) - normal_votes  # The amount of tests which did not

    # If most tests find the data to be normal distributed, the data is deemed gaussian
    return normal_votes >= non_normal_votes
