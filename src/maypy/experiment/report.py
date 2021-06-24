from maypy import ALPHA

# @no:format
from maypy.utils import Document
class Report:
    def __init__(self, test_name, statistic_name, statistic, p_value, h0_rejected: callable, interpretation: callable, **extra):
        self.name = test_name
        self.statistic_name = statistic_name
        self.statistic = statistic
        self.p_value = p_value
        self.h0_rejected_at_alpha = {a: h0_rejected(a) for a in ALPHA.ALLOWED}
        self.interpretation_at_alpha = {a: interpretation(a) for a in ALPHA.ALLOWED}
        self.extra = extra
        self.conclusion = "Conclusion"

    def set_conclusion(self, header):
        self.conclusion = header
        return self

    def at_alpha(self, **kwargs):
        for name, callable_value in kwargs.items():
            self.extra[name] = {a: callable_value(a) for a in ALPHA.ALLOWED}
        return self

    def __bool__(self):
        return bool(self.interpretation_at_alpha[float(ALPHA)])

    def __int__(self):
        return int(bool(self))

    def __add__(self, other):
        return int(self) + other

    def __radd__(self, other):
        return int(self) + other

    def __repr__(self):
        R = lambda x: round(x, 3) if isinstance(x, float) else x

        extra_at_alpha = {key: value for key, value in self.extra.items() if  isinstance(value, (dict))}
        extra_constants = {key: value for key, value in self.extra.items() if not isinstance(value, (dict, list, tuple))}

        doc = Document()
        doc.row[0: self.name]
        doc.row
        doc.row[0: f"{self.statistic_name}: {self.statistic}"][1: f"p-value: {R(self.p_value)}"]
        doc.row

        doc.row[0: "Result at Significance"]
        header_row = doc.row[1: "H0 Rejected"][2: self.conclusion]
        for i, key in enumerate(extra_at_alpha.keys()):
            header_row[i+3:key]

        for alpha in ALPHA.ALLOWED:
            row = doc.row[0: f"alpha: {alpha}": "right",
                          1: self.h0_rejected_at_alpha[alpha]: "center",
                          2: self.interpretation_at_alpha[alpha]: "center"]

            for i, value_dict in enumerate(extra_at_alpha.values()):
                value = value_dict[alpha]
                if isinstance(value, float):
                    value = R(value)

                row[i+3:value:"center"]

        if extra_constants:
            doc.row[0: "Extra Info"]
            for name, value in extra_constants.items():
                doc.row[0: name: "right"][1: value]

        return repr(doc)

# @do:format
