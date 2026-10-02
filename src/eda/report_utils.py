"""Small helpers for generating markdown reports from the output tables."""

from __future__ import annotations

import numpy as np
import pandas as pd


def fmt(v, nd: int = 4) -> str:
    if isinstance(v, (float, np.floating)):
        if np.isnan(v):
            return ""
        if v != 0 and (abs(v) < 10 ** -(nd) or abs(v) >= 1e6):
            return f"{v:.3g}"
        return f"{v:.{nd}f}"
    if isinstance(v, (bool, np.bool_)):
        return str(bool(v))
    return str(v)


def md_table(df: pd.DataFrame, nd: int = 4, max_rows: int | None = None) -> str:
    d = df.copy()
    if max_rows is not None:
        d = d.head(max_rows)
    cols = [str(c) for c in d.columns]
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    for _, r in d.iterrows():
        lines.append("| " + " | ".join(fmt(v, nd) for v in r.values) + " |")
    return "\n".join(lines)
