from collections import defaultdict

from maypy import ALPHA
from maypy.utils import Document


class Result:
    def __init__(self, name=None, description=None, **extra):
        self.name = name
        self.description = description
        self.extra = {}
        self.extra_at_alpha = {}

        for key, value in extra.items():
            if callable(value):
                self.extra_at_alpha[key] = {a: value(a) for a in ALPHA.ALLOWED}
            else:
                self.extra[key] = value

    def __repr__(self):
        # @no:format
        R = lambda x, d=3: round(x, d) if isinstance(x, float) else x

        doc = Document()
        doc.row[0: self.name][1:self.description]
        doc.row

        doc.row[0: "Result at Significance"]
        header_row = doc.row
        for i, key in enumerate(self.extra_at_alpha.keys()):
            header_row[i+1:key]

        for alpha in ALPHA.ALLOWED:
            row = doc.row[0: f"alpha: {alpha}": "right"]
            for i, value_dict in enumerate(self.extra_at_alpha.values()):
                row[i + 1 : R(value_dict[alpha])]

        if self.extra:
            doc.row
            for key, value in self.extra.items():
                doc.row[0: key: "right"][1: value]

        # @do:format
        return repr(doc)
