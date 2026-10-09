"""Tests for the green belt preparation (src/11e_prepare_greenbelt.py)."""

import importlib.util
import sys
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
import shapely

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
spec = importlib.util.spec_from_file_location("gb", ROOT / "src" / "11e_prepare_greenbelt.py")
gb = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gb)


def test_tiling_preserves_area_and_pieces_are_disjoint():
    poly = shapely.union_all([shapely.box(0, 0, 25_000, 7_000), shapely.Point(12_000, 3_000).buffer(5_000)])
    pieces = gb.tile_polygons(poly, cell=10_000)
    assert sum(p.area for p in pieces) == pytest.approx(poly.area, rel=1e-9)
    assert shapely.union_all(pieces).area == pytest.approx(poly.area, rel=1e-9)     # no overlap double counting


def test_share_from_overlay_matches_hand_calculation():
    msoa = gpd.GeoDataFrame({"msoa11cd": ["A", "B"]}, geometry=[shapely.box(0, 0, 100, 100), shapely.box(100, 0, 200, 100)], crs=27700)
    pieces = gpd.GeoDataFrame({"gb": [1]}, geometry=[shapely.box(0, 0, 150, 40)], crs=27700)     # all of the bottom 40% of A and half of B's 40% band
    inter = gpd.overlay(msoa, pieces, how="intersection")
    share = (inter.geometry.area.groupby(inter.msoa11cd).sum() / msoa.set_index("msoa11cd").geometry.area).fillna(0)
    assert share["A"] == pytest.approx(0.4) and share["B"] == pytest.approx(0.2)


OUT = ROOT / "data" / "interim" / "geography_crosswalks" / "msoa11_greenbelt2011.parquet"


@pytest.mark.skipif(not OUT.exists(), reason="green belt table not built")
def test_built_table_ranges_and_zero_inflation():
    d = pd.read_parquet(OUT)
    assert len(d) == 6791 and d.msoa11cd.is_unique and not d.isna().any().any()
    assert d.greenbelt_share_2011.between(0, 1).all()
    assert d.greenbelt_any_2011.equals((d.greenbelt_share_2011 >= 0.01).astype(int))
    assert d.greenbelt_share_z.mean() == pytest.approx(0, abs=1e-9)
    assert 0.5 < (d.greenbelt_share_2011 == 0).mean() < 0.7          # about 61% of MSOA11s have no green belt
