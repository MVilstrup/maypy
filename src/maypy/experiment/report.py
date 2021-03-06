from collections import defaultdict, Counter

from maypy import ALPHA
from maypy.experiment.interpretation import Interpretation
from maypy.utils import Document


class Report:
    def __init__(self, test_name, statistic_name, statistic, p_value, h0_rejected: callable, interpretation: callable,
                 conclusion="Conclusion", test_description="", **extra):
        self.name = test_name
        self.description = test_description
        self.statistic_name = statistic_name
        self.statistic = statistic
        self.p_value = p_value
        self.h0_rejected_at_alpha = {a: h0_rejected(a) for a in ALPHA.ALLOWED}
        self.interpretation_at_alpha = {a: interpretation(a) for a in ALPHA.ALLOWED}
        self.conclusion = conclusion
        self.assumptions = {}
        self.extra_at_alpha = {}

        self.extra = defaultdict(dict)

        for key, value in extra:
            if isinstance(value, dict):
                self.extra[key] = value
            else:
                extra["Extra Info"][key] = value

    def interpretation(self, alpha=None):
        return self.interpretation_at_alpha[alpha if alpha is not None else float(ALPHA)]

    def all_interpretations(self, alpha):
        yield "Result", self.interpretation(alpha)
        for key, value in self.extra_at_alpha.items():
            yield key, value[alpha]

    def set_conclusion(self, header):
        self.conclusion = header
        return self

    def add_assumption(self, name, check):
        self.assumptions[name] = check

    def at_alpha(self, **kwargs):
        for name, callable_value in kwargs.items():
            self.extra_at_alpha[name] = {a: callable_value(a) for a in ALPHA.ALLOWED}
        return self

    def __bool__(self):
        return bool(self.interpretation_at_alpha[float(ALPHA)])

    def __int__(self):
        return int(bool(self))

    def __add__(self, other):
        return int(self) + other

    def __radd__(self, other):
        return int(self) + other

    def __str__(self):
        return f"{self.name} - {self.description}"

    def __repr__(self):
        # @no:format
        R = lambda x, d=3: round(x, d) if isinstance(x, float) else x

        doc = Document()
        doc.row[0: self.name][1:self.description]
        doc.row
        doc.row[1: f"{self.statistic_name}: {R(self.statistic)}"][2: f"p-value: {R(self.p_value, 5)}"]
        doc.row

        if self.assumptions:
            doc.row
            doc.row[0: "Test Assumptions"]
            for key, result in self.assumptions.items():
                if isinstance(result, Report):
                    doc.row[0: key: "right"][1: result.interpretation()]
                else:
                    doc.row[0: key: "right"][1: Interpretation(result)]

            doc.row


        doc.row[0: "Result at Significance"]
        header_row = doc.row[1: "H0 Rejected"][2: self.conclusion]
        for i, key in enumerate(self.extra_at_alpha.keys()):
            header_row[i+3:" ".join([k.capitalize() for k in key.split("_")])]

        for alpha in ALPHA.ALLOWED:
            row = doc.row[0: f"alpha: {alpha}": "right",
                          1: self.h0_rejected_at_alpha[alpha],
                          2: self.interpretation(alpha)]
            for i, value_dict in enumerate(self.extra_at_alpha.values()):
                value = value_dict[alpha]
                if isinstance(value, float):
                    value = R(value)

                row[i+3:value:"center"]


        if self.extra:
            for key, values in self.extra.items():
                doc.row[0: key]
                for param, value in values.items():
                    doc.row[0: param: "right"][1: value]
                doc.row

        # @do:format
        return repr(doc)
