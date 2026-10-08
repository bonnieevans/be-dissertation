"""Tests for the Census 2011 age-structure preparation (src/11b_prepare_census2011_age.py)."""

import importlib.util
import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
spec = importlib.util.spec_from_file_location("age_prep", ROOT / "src" / "11b_prepare_census2011_age.py")
age = importlib.util.module_from_spec(spec)
spec.loader.exec_module(age)


def test_groups_partition_the_sixteen_bands_exactly():
    grouped = [b for g in age.GROUPS.values() for b in g]
    assert sorted(grouped) == sorted(age.BANDS)          # every band used once, none left out
    assert len(grouped) == len(set(grouped)) == 16


def test_five_groups():
    assert list(age.GROUPS) == ["under16", "16_24", "25_44", "45_64", "65plus"]


OUT = ROOT / "data" / "interim" / "geography_crosswalks" / "msoa11_census2011_age.parquet"


@pytest.mark.skipif(not OUT.exists(), reason="age structure table not built")
def test_built_table_shares_sum_to_one_and_z_scores_standardised():
    df = pd.read_parquet(OUT)
    shares = [c for c in df.columns if c.startswith("age_share_") and c.endswith("_2011")]
    assert len(df) == 6791 and df.msoa11cd.is_unique
    assert ((df[shares].sum(axis=1) - 1).abs() < 1e-9).all()
    for c in shares:
        z = df[c.replace("_2011", "_z")]
        assert z.mean() == pytest.approx(0, abs=1e-9) and z.std(ddof=0) == pytest.approx(1, abs=1e-9)
