"""Tests for the spatial spillover exposure functions (src/spillover_methods.py) on small synthetic graphs with known answers,
plus structural checks on the built tables when they exist."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import spillover_methods as sm  # noqa: E402

# Synthetic geography: five MSOAs on a line at x = 0, 1, 2, 3, 10 km. Queen-style neighbours: 0-1, 1-2, 2-3 (4 is isolated).
HH = np.array([1000.0, 2000.0, 3000.0, 4000.0, 500.0])
CNT = np.array([[10, 0, 4], [20, 5, 8], [60, 5, 12], [40, 0, 16], [7, 7, 7]], dtype=float)      # 3 years
EDGES = [(0, 1), (1, 0), (1, 2), (2, 1), (2, 3), (3, 2)]
W = sm.edges_to_matrix([a for a, _ in EDGES], [b for _, b in EDGES], 5)


def test_pooled_rate_is_total_over_total_households_not_mean_of_rates():
    r = sm.pooled_rate(W, CNT, HH)
    assert r[0, 0] == pytest.approx(1000 * 20 / 2000)                    # neighbour {1}
    assert r[1, 0] == pytest.approx(1000 * (10 + 60) / (1000 + 3000))    # neighbours {0, 2}: 17.5
    assert r[2, 1] == pytest.approx(1000 * (5 + 0) / (2000 + 4000))      # neighbours {1, 3}
    mean_of_rates = np.mean([1000 * CNT[0, 0] / HH[0], 1000 * CNT[2, 0] / HH[2]])   # (10 + 20) / 2 = 15
    assert mean_of_rates == pytest.approx(15.0) and r[1, 0] != pytest.approx(mean_of_rates)   # the two definitions differ


def test_focal_msoa_is_excluded():
    s = sm.neighbour_sum(W, CNT)
    assert s[1, 0] == 10 + 60                                            # not 10 + 20 + 60
    assert W.diagonal().sum() == 0


def test_no_neighbour_gives_nan_not_zero():
    r = sm.pooled_rate(W, CNT, HH)
    assert np.isnan(r[4]).all()
    assert np.isnan(sm.neighbour_mean(W, CNT)[4]).all()
    assert (sm.neighbour_sum(W, CNT)[4] == 0).all()                      # the raw sum is 0, which is why rates (not totals) carry NaN


def test_neighbour_mean_matches_row_standardised_lag():
    Wr = sm.row_normalise(W)
    rates = 1000 * CNT / HH[:, None]
    assert np.allclose(sm.neighbour_mean(W, rates)[:4], (Wr @ rates)[:4])
    assert np.allclose(np.asarray(Wr.sum(axis=1)).ravel()[:4], 1.0)


def test_lagged_exposure_uses_the_preceding_year():
    lag1 = np.hstack([np.full((5, 1), 3.0), CNT[:, :-1]])                # first-year lag comes from the lookback (here 3 everywhere)
    r_lag = sm.pooled_rate(W, lag1, HH)
    r_prev = sm.pooled_rate(W, CNT, HH)
    assert np.allclose(r_lag[:4, 1:], r_prev[:4, :-1])                   # year t lag1 equals year t-1 same-year exposure
    assert not np.isnan(r_lag[0, 0])                                     # first year defined through the lookback


def test_cross_lad_restriction():
    lad = np.array(["A", "A", "B", "B", "C"])
    cross = [(a, b) for a, b in EDGES if lad[a] != lad[b]]
    Wc = sm.edges_to_matrix([a for a, _ in cross], [b for _, b in cross], 5)
    r = sm.pooled_rate(Wc, CNT, HH)
    assert r[1, 0] == pytest.approx(1000 * 60 / 3000)                    # only neighbour 2 is in another LAD
    assert np.isnan(r[0, 0])                                             # MSOA 0 has no cross-LAD neighbour


def test_distance_pairs_thresholds_and_exclusion_of_self():
    xy = np.array([[0, 0], [1000, 0], [2000, 0], [3000, 0], [10000, 0]], dtype=float)
    p = sm.distance_pairs(xy, 2000.0)
    assert (p.i != p.j).all()
    assert p.distance_m.max() <= 2000
    pairs = set(zip(p.i, p.j))
    assert (0, 2) in pairs and (2, 0) in pairs and (0, 3) not in pairs and (3, 4) not in pairs
    assert len(p) == 10


def test_distance_weighted_rate_matches_hand_calculation():
    xy = np.array([[0, 0], [1000, 0], [2000, 0], [3000, 0], [10000, 0]], dtype=float)
    p = sm.distance_pairs(xy, 3000.0)
    w = sm.exp_decay(p.distance_m, 3000.0)
    Wexp = sm.edges_to_matrix(p.i, p.j, 5, w)
    r = sm.pooled_rate(Wexp, CNT, HH)
    e1, e2 = np.exp(-1000 / 3000), np.exp(-2000 / 3000)
    # focal 1: neighbours 0 (1 km), 2 (1 km), 3 (2 km)
    num = e1 * CNT[0, 0] + e1 * CNT[2, 0] + e2 * CNT[3, 0]
    den = e1 * HH[0] + e1 * HH[2] + e2 * HH[3]
    assert r[1, 0] == pytest.approx(1000 * num / den)
    assert np.isnan(r[4, 0])                                             # nothing within 3 km of the remote MSOA


def test_income_is_household_weighted_level_and_log_of_mean_differs_from_mean_of_logs():
    inc = np.array([20000.0, 30000.0, 60000.0, 25000.0, 40000.0])
    m = sm.household_weighted_mean(W, inc, HH)
    assert m[1] == pytest.approx((20000 * 1000 + 60000 * 3000) / 4000)   # 50,000
    assert np.log(m[1]) != pytest.approx((np.log(20000) * 1000 + np.log(60000) * 3000) / 4000)
    assert np.isnan(m[4])


def test_neighbour_price_is_sale_weighted_and_ignores_missing():
    lp = np.array([[7.0], [8.0], [np.nan], [9.0], [8.5]])
    sales = np.array([[10.0], [20.0], [30.0], [0.0], [5.0]])
    mean, nvalid, tot = sm.sale_weighted_price(W, lp, sales)
    assert mean[1, 0] == pytest.approx(7.0)                              # neighbours 0 (valid) and 2 (missing price): only 0 counts
    assert nvalid[1, 0] == 1 and tot[1, 0] == 10
    assert mean[2, 0] == pytest.approx(8.0)                              # neighbours 1 (valid) and 3 (zero sales -> excluded)
    assert np.isnan(mean[4, 0])                                          # no neighbours -> NaN, not zero


def test_second_order_excludes_focal_and_first_order():
    W2 = sm.second_order(W)
    assert W2[0, 2] == 1 and W2[1, 3] == 1 and W2[3, 1] == 1
    assert W2[0, 1] == 0 and W2[0, 0] == 0 and W2[0, 3] == 0


def test_invalid_edge_lists_are_rejected():
    with pytest.raises(ValueError):
        sm.edges_to_matrix([0, 1], [0, 2], 3)                            # self neighbour
    with pytest.raises(ValueError):
        sm.edges_to_matrix([0, 0], [1, 1], 3)                            # duplicated directed pair


G = ROOT / "data" / "interim" / "geography_crosswalks"


@pytest.mark.skipif(not (G / "msoa11_spatial_neighbors_queen.parquet").exists(), reason="spatial tables not built")
def test_built_queen_lookup_properties():
    q = pd.read_parquet(G / "msoa11_spatial_neighbors_queen.parquet")
    assert (q.focal_msoa11cd != q.neighbour_msoa11cd).all()
    assert not q.duplicated(["focal_msoa11cd", "neighbour_msoa11cd"]).any()
    g = q[q.edge_type == "queen"]
    rev = set(zip(g.neighbour_msoa11cd, g.focal_msoa11cd))
    assert set(zip(g.focal_msoa11cd, g.neighbour_msoa11cd)) == rev       # genuine contiguity is reciprocal
    assert (q.distance_m >= 0).all() and set(q.edge_type) <= {"queen", "island_fallback"}
    assert set(q[q.edge_type == "island_fallback"].focal_msoa11cd) | set(q[q.edge_type == "island_fallback"].neighbour_msoa11cd) >= {"E02006781"}
    assert q.focal_msoa11cd.nunique() == 6791


@pytest.mark.skipif(not (G / "msoa11_spatial_neighbors_distance.parquet").exists(), reason="spatial tables not built")
def test_built_distance_lookup_properties():
    d = pd.read_parquet(G / "msoa11_spatial_neighbors_distance.parquet")
    assert (d.focal_msoa11cd != d.neighbour_msoa11cd).all()
    assert d.distance_m.between(0, 10000).all()
    assert d.loc[d.within_5km, "distance_m"].max() <= 5000 and (d.loc[~d.within_5km, "distance_m"] > 5000).all()
    assert np.allclose(d.weight_exp, np.exp(-d.distance_m / 3000))
    assert np.allclose(d.groupby("focal_msoa11cd").weight_exp_rownorm.sum(), 1.0)


@pytest.mark.skipif(not (G / "msoa_year_spillover_exposure.parquet").exists(), reason="spatial tables not built")
def test_built_exposure_table_structure():
    e = pd.read_parquet(G / "msoa_year_spillover_exposure.parquet")
    assert len(e) == 81492 and not e.duplicated(["msoa11cd", "year"]).any()
    assert (e.nbr_queen_nb_total >= 0).all()
    assert e.nbr_queen_nb_lag1_rate_per1000.notna().all()                # every MSOA has a queen neighbour
    assert e.loc[e.nbr_5km_none, "nbr_5km_nb_lag1_rate_per1000"].isna().all()
    assert e.loc[~e.nbr_5km_none, "nbr_5km_nb_lag1_rate_per1000"].notna().all()
    assert e[e.year == 2012].nbr_queen_nb_lag1_rate_per1000.notna().all()    # lookback lags keep 2012 defined
