# import maypy.distributions as lazy
#
#
#
# def test_fit_finds_the_right_distribution():
#     for distribution in lazy.ALL_DISTRIBUTIONS:
#         size = 10000
#         sample = distribution.example().rvs(size)
#         estimated = lazy.fit(sample, size)
#         assert isinstance(estimated, distribution), f"{type(estimated)} != {distribution}"