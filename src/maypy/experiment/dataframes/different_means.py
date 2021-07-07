from collections import defaultdict
from maypy import ALPHA, fit
from maypy.experiment.report import Report
import numpy as np
import pandas as pd


def different_mean(df, group_col, alpha=ALPHA.DEFAULT, subset=None, ignore=None, P=None, Q=None, kind="two-sided"):
    keys = np.unique(df[group_col])
    assert len(keys) == 2
    assert alpha in ALPHA.ALLOWED

    P_key = P if P is not None else keys[0]
    Q_key = Q if Q is not None else keys[1]

    P_data, Q_data = df[df[group_col] == P_key], df[df[group_col] == Q_key]

    columns = set(subset) if subset is not None else set(df.columns)
    columns = columns - set([*ignore, group_col] if ignore is not None else [group_col])

    results = defaultdict(list)
    index = []
    for column in columns:
        if sum(P_data[column].notna()) == 0 or sum(Q_data[column].notna()) == 0:
            continue
        index.append(column)

        P, Q = fit(P_data[column]), fit(Q_data[column])

        if kind == "less":
            experiment = P < Q
        elif kind == "greater":
            experiment = P > Q
        else:
            experiment = P != Q

        results[f"Distribution({P_key}, {Q_key})"].append((f"{P.class_name}", f"{Q.class_name}"))
        for i, (report_name, report) in enumerate(experiment.reports[(P.name, Q.name)].items()):
            for key, result in report.all_interpretations(alpha):
                results[f"{report_name} {key}"].append(result)

    return pd.DataFrame(results, index=index).transpose()
