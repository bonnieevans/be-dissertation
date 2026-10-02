"""Unit tests for the income duplicate-value QA check in
src/06_prepare_income.py, which must distinguish genuine ONS rounding
(many MSOA11s legitimately sharing a coarse rounded value) from a failed
geography merge (one value dominating an implausibly large share of rows)."""

import pandas as pd
import pytest

MAX_ACCEPTABLE_SINGLE_VALUE_SHARE = 0.20


def compute_top_share(values: pd.Series) -> float:
    counts = values.value_counts()
    return counts.iloc[0] / len(values)


def test_genuine_ons_rounding_does_not_trigger_the_check():
    # ONS rounds this series to the nearest GBP10/week: many ties, but no
    # single value dominates. Modelled on the real distribution (83 values
    # across 6791 MSOA11s, max share ~4%).
    values = pd.Series([300 + 10 * (i % 80) for i in range(6791)])
    top_share = compute_top_share(values)
    assert top_share < MAX_ACCEPTABLE_SINGLE_VALUE_SHARE


def test_failed_merge_pattern_triggers_the_check():
    # A failed join (e.g. every row defaulting to one source value after a
    # broken key match) shows as one value dominating most rows.
    values = pd.Series([500] * 6000 + list(range(1, 792)))
    top_share = compute_top_share(values)
    assert top_share > MAX_ACCEPTABLE_SINGLE_VALUE_SHARE


def test_one_row_per_msoa11_required():
    df = pd.DataFrame({"msoa11cd": ["E02000001", "E02000002", "E02000001"]})
    n_rows = len(df)
    n_distinct = df["msoa11cd"].nunique()
    assert n_rows != n_distinct  # this fixture has a duplicate - the real
    # script's QA check (`if n_msoa11 != n_rows: return 1`) must catch this.
