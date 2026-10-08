"""Tests for the DfT accessibility preparation (src/11c_prepare_accessibility.py)."""

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
spec = importlib.util.spec_from_file_location("acc_prep", ROOT / "src" / "11c_prepare_accessibility.py")
acc = importlib.util.module_from_spec(spec)
spec.loader.exec_module(acc)


def test_weighted_mean_is_population_weighted_not_simple():
    df = pd.DataFrame({"m": ["A", "A", "B"], "t": [10.0, 30.0, 5.0], "w": [900.0, 100.0, 50.0]})
    r = acc.weighted_mean(df, "t", "w", "m")
    assert r["A"] == pytest.approx((10 * 900 + 30 * 100) / 1000) and r["A"] != 20.0
    assert r["B"] == 5.0


def test_zero_weight_lsoa_is_ignored_but_msoa_survives():
    df = pd.DataFrame({"m": ["A", "A"], "t": [120.0, 8.0], "w": [0.0, 40.0]})
    assert acc.weighted_mean(df, "t", "w", "m")["A"] == 8.0


OUT = ROOT / "data" / "interim" / "geography_crosswalks" / "msoa11_accessibility2014.parquet"


@pytest.mark.skipif(not OUT.exists(), reason="accessibility table not built")
def test_built_table_structure_and_ranges():
    d = pd.read_parquet(OUT)
    assert len(d) == 6791 and d.msoa11cd.is_unique
    times = [c for c in d.columns if c.startswith("access_") and c.endswith("_min_2014")]
    assert len(times) == 18                       # 8 services x 2 modes + 2 composites
    assert (d[times] > 0).all().all() and (d[times] <= 120).all().all()
    assert (d["dist_town_centre_km"] >= 0).all() and d["dist_town_centre_km"].notna().all()
    svc_pt = [c for c in times if c.endswith("_pt_min_2014") and "keyservices" not in c]
    assert np.allclose(d[svc_pt].mean(axis=1), d["access_keyservices_pt_min_2014"])
    for z in ("access_keyservices_pt_z", "dist_town_centre_z"):
        assert d[z].mean() == pytest.approx(0, abs=1e-9) and d[z].std(ddof=0) == pytest.approx(1, abs=1e-9)


@pytest.mark.skipif(not OUT.exists(), reason="accessibility table not built")
def test_england_average_close_to_published_dft_figure():
    d = pd.read_parquet(OUT)
    assert 14 < d["access_keyservices_pt_min_2014"].mean() < 20      # DfT: about 17 minutes
    assert 8 < d["access_keyservices_car_min_2014"].mean() < 13      # DfT: about 10 minutes
