from maypy.best_practices.different_means import different_mean, greater_mean, lesser_mean
from maypy.distributions import NP_DISTRIBUTIONS
from random import choice


def test_on_identical_distributions():
    for distribution in NP_DISTRIBUTIONS:
        P = distribution.example()

        # Ensure idential distributions do not differ in mean
        if not P.class_name == "Normal":
            assert not different_mean(P, P)
            assert not greater_mean(P, P)
            assert not lesser_mean(P, P)

        # Ensure idential distributions are correlated
        assert P & P

        # Ensure different distributions are not correlated
        Q = choice(NP_DISTRIBUTIONS).example()
        while P.class_name == Q.class_name:
            Q = choice(NP_DISTRIBUTIONS).example()

        assert P | Q
