from typing import Optional

from maypy.distributions import Distribution
from maypy.experiment.experiment import Experiment


class Test:
    def __init__(self, experiment=None):
        self.experiment = experiment if experiment is not None else Experiment(None, None)

    def distribution_assumption(self, function, P, helper, **kwargs):
        if P is not None:
            report = function(P=P, experiment=self.experiment, **kwargs)
            if not report:
                self.experiment.failure(helper, report, P)

    def pair_assumption(self, function, P, Q, helper, **kwargs):
        if P is not None and Q is not None:
            report = function(P=P, Q=Q, experiment=self.experiment, **kwargs)
            if not report:
                self.experiment.failure(helper, report, (P, Q))

    def check_assumptions(self, P: Distribution, Q: Optional[Distribution] = None):
        raise NotImplementedError()
