from typing import Optional
from uuid import uuid4

from collections import defaultdict

from maypy.distributions import Distribution


class Experiment:

    def __init__(self, name, P: Distribution, Q:Distribution=None):
        self.id = uuid4()
        self.name = name

        self.P = P
        self.Q = Q

        self.reports = defaultdict(dict)

        self.failure_explanation = None
        self.failure_report = None
        self.failure_distribution = None

        self.active = False

    def failure(self, explanation, report, distribution=Optional[Distribution]):
        self.failure_explanation = explanation
        self.failure_report = report
        self.failure_distribution = distribution

    def __setitem__(self, distributions, report):
        if isinstance(distributions, (list, tuple)):
            P, Q = distributions
            self.reports[(P.name, Q.name)][report.name] = report
        else:
            P = distributions
            self.reports[P.name][report.name] = report

    def __getitem__(self, info):
        distributions, report = info
        return self.reports[distributions][report.name]

    def __enter__(self):
        self.active = True

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.active = False

    def __repr__(self):
        repr_P = repr(self.P)
        repr_Q = repr(self.Q)
        tests = map(repr, self.reports[(self.P.name, self.Q.name)].values())
        return "\n\n".join([self.name, repr_P, repr_Q, *tests])