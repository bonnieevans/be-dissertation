"""Tests for the 2011 accessibility preparation helpers (src/11d_prepare_accessibility2011.py)."""

import importlib.util
import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
spec = importlib.util.spec_from_file_location("acc11", ROOT / "src" / "11d_prepare_accessibility2011.py")
acc = importlib.util.module_from_spec(spec)
spec.loader.exec_module(acc)


def test_num_strips_thousands_commas():
    assert acc.num(pd.Series(["1,680", " 920 ", "5"])).tolist() == [1680, 920, 5]


def test_col_matches_prefix_despite_footnote_digits_and_requires_unique():
    df = pd.DataFrame(columns=["LSOA_code1", "Pop_age5to101", "psPTtime1", "pscarnewtime1,2"])
    assert acc.col(df, "Pop_age5to10") == "Pop_age5to101"
    assert acc.col(df, "pscarnewtime") == "pscarnewtime1,2"
    with pytest.raises(KeyError):
        acc.col(df, "missing")
    with pytest.raises(KeyError):
        acc.col(pd.DataFrame(columns=["abc1", "abc2"]), "abc")


OUT = ROOT / "data" / "interim" / "geography_crosswalks" / "msoa11_accessibility2011.parquet"


@pytest.mark.skipif(not OUT.exists(), reason="2011 accessibility table not built")
def test_built_table_covers_all_msoa11_and_matches_dft_2011_headline():
    d = pd.read_parquet(OUT)
    assert len(d) == 6791 and d.msoa11cd.is_unique and not d.isna().any().any()
    assert 12 < d["access2011_keyservices7_pt_min"].mean() < 16        # DfT 2011: about 14 minutes
    assert 4.5 < d["access2011_keyservices7_car_min"].mean() < 8       # DfT 2011: about 6 minutes
