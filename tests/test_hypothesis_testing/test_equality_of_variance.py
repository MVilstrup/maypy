from maypy import fit
from maypy.hypothesis_testing.universal.equality_of_variance.levenes import LeveneTest


def test_levenes_method():
    a = fit([8.88, 9.12, 9.04, 8.98, 9.00, 9.08, 9.01, 8.85, 9.06, 8.99])
    b = fit([8.88, 8.95, 9.29, 9.44, 9.15, 9.58, 8.36, 9.18, 8.67, 9.05])
    c = fit([8.95, 9.12, 8.95, 8.85, 9.03, 8.84, 9.07, 8.98, 8.86, 8.98])

    assert not LeveneTest()(a, b, c)

