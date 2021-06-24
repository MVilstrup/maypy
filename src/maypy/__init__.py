import sys

if sys.version_info[:2] >= (3, 8):
    # TODO: Import directly (no need for conditional) when `python_requires = >= 3.8`
    from importlib.metadata import PackageNotFoundError, version  # pragma: no cover
else:
    from importlib_metadata import PackageNotFoundError, version  # pragma: no cover

try:
    # Change here if project is renamed and does not equal the package name
    dist_name = "maypy"
    __version__ = version(dist_name)
except PackageNotFoundError:  # pragma: no cover
    __version__ = "unknown"
finally:
    del version, PackageNotFoundError


class __ALPHA__:
    ALLOWED = [0.1, 0.05, 0.025, 0.01, 0.005, 0.001]
    DEFAULT = 0.005

    def __init__(self, initial_alpha):
        self.alpha = initial_alpha
        self.prev_alpha = initial_alpha

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.alpha = self.prev_alpha

    def __call__(self, alpha):
        if not alpha in self.ALLOWED:
            raise ValueError(f"alpha should be one of {self.ALLOWED}")

        self.prev_alpha = self.alpha
        self.alpha = alpha
        return self

    def __float__(self):
        return self.alpha


ALPHA = __ALPHA__(__ALPHA__.DEFAULT)

from maypy.best_practices.distribution_search import fit
