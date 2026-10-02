"""Tests for the panel time-series tests (src/eda/panel_tests.py) and the model-lab helpers (src/models)."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
for sub in ("src", "src/eda", "src/models"):
    sys.path.insert(0, str(ROOT / sub))

import panel_tests as pt  # noqa: E402
from m03_moderators import holm, within  # noqa: E402
from m06_clustering_inference import group_means_demean, wild_cluster_bootstrap  # noqa: E402
import model_common as MC  # noqa: E402

rng = np.random.default_rng(0)


def simulate(N, T, rho, reps):
    out = []
    for _ in range(reps):
        e = rng.standard_normal((N, T))
        y = np.zeros((N, T))
        y[:, 0] = e[:, 0]
        for t in range(1, T):
            y[:, t] = rho * y[:, t - 1] + e[:, t]
        out.append(y)
    return out


def test_harris_tzavalis_size_and_power():
    rw = [pt.harris_tzavalis(y)["p_value"] < .05 for y in simulate(1500, 12, 1.0, 120)]
    ar = [pt.harris_tzavalis(y)["p_value"] < .05 for y in simulate(1500, 12, 0.5, 30)]
    assert 0.0 <= np.mean(rw) <= 0.12          # nominal 5%
    assert np.mean(ar) > 0.95


def test_harris_tzavalis_trend_size():
    rej = []
    for y in simulate(1500, 12, 1.0, 120):
        yt = y + 0.3 * np.arange(12)[None, :] + rng.standard_normal((1500, 1))
        rej.append(pt.harris_tzavalis(yt, trend=True)["p_value"] < .05)
    assert np.mean(rej) <= 0.12


def test_hadri_size_and_power():
    iid = [pt.hadri_lm(y)["p_value"] < .05 for y in simulate(1500, 12, 0.0, 120)]
    rw = [pt.hadri_lm(y)["p_value"] < .05 for y in simulate(1500, 12, 1.0, 30)]
    assert np.mean(iid) <= 0.12
    assert np.mean(rw) > 0.95


def test_pesaran_cd_independent_and_common_factor():
    E = rng.standard_normal((600, 10))
    assert abs(pt.pesaran_cd(E)["CD"]) < 3.5
    common = rng.standard_normal((1, 10))
    E2 = rng.standard_normal((600, 10)) + 1.5 * common * rng.standard_normal((600, 1))
    assert abs(pt.pesaran_cd(E2)["CD"]) > 3.5 or abs(pt.pesaran_cd(E2)["mean_abs_pairwise_corr"]) > 0.1
    # a time-specific shock common to every unit makes the CD statistic large
    E3 = rng.standard_normal((600, 10)) + rng.standard_normal((1, 10)) * 2
    assert abs(pt.pesaran_cd(E3)["CD"]) > 3.5


def test_pesaran_cd_ignores_constant_units():
    E = rng.standard_normal((50, 8))
    E[:5] = 1.0
    assert np.isfinite(pt.pesaran_cd(E)["CD"])


def test_holm_adjustment_known_values():
    p = np.array([0.01, 0.04, 0.03, 0.20])
    adj = holm(p)
    assert np.allclose(adj, [0.04, 0.09, 0.09, 0.20])


def _synthetic_panel(beta=0.5, n_lad=20, per_lad=6, T=8, seed=3):
    r = np.random.default_rng(seed)
    rows = []
    for l in range(n_lad):
        for m in range(per_lad):
            ae = r.standard_normal()
            for t in range(T):
                rows.append((f"L{l}", f"M{l}_{m}", 2016 + t, ae, r.standard_normal() * 2))
    d = pd.DataFrame(rows, columns=["lad23cd_analysis", "msoa11cd", "year", "ae", "x"])
    lt = {(l, t): r.standard_normal() for l in d.lad23cd_analysis.unique() for t in d.year.unique()}
    d["lad_year"] = d.lad23cd_analysis + "_" + d.year.astype(str)
    d["y"] = beta * d.x + d.ae + [lt[(a, b)] for a, b in zip(d.lad23cd_analysis, d.year)] + r.standard_normal(len(d)) * 0.5
    return d


def test_within_transformation_matches_fixest_for_both_fe_structures():
    import pyfixest as pf
    d = _synthetic_panel()
    for fe_id, fe in (("M3", "| msoa11cd + year"), ("M6", "| msoa11cd + lad_year")):
        W = within(d, ["x", "y"], fe_id)
        b = (W.x * W.y).sum() / (W.x ** 2).sum()
        f = pf.feols(f"y ~ x {fe}", data=d)
        assert b == pytest.approx(float(f.coef().iloc[0]), rel=1e-6)


def test_wild_cluster_bootstrap_detects_effect_and_not_noise():
    d = _synthetic_panel(beta=0.5)
    d = d.rename(columns={"y": MC.OUTCOME, "x": MC.TREATMENT})
    strong = wild_cluster_bootstrap(d, MC.TREATMENT, "M3", reps=199)
    assert strong["wild_bootstrap_p"] < 0.05
    d2 = _synthetic_panel(beta=0.0, seed=11).rename(columns={"y": MC.OUTCOME, "x": MC.TREATMENT})
    null = wild_cluster_bootstrap(d2, MC.TREATMENT, "M3", reps=199)
    assert null["wild_bootstrap_p"] > 0.01


def test_group_means_demean_is_idempotent():
    codes = np.array([0, 0, 1, 1, 2, 2])
    v = np.array([1., 3., 2., 4., 5., 9.])
    once = group_means_demean(v, [codes])
    assert np.allclose(group_means_demean(once, [codes]), once)
    assert np.allclose(once[:2].sum(), 0)


def test_units_sentence_uses_log_point_conversion():
    s = MC.units_sentence(0.01)
    assert "1.0050%" in s
