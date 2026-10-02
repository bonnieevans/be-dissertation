"""Unit tests for the IMD 2015 revision (src/imd_methods.py, src/07_prepare_imd2015.py).

Includes the reproduction test required by the revision brief: recombining
all seven domains with the Technical Report method must reproduce the
published IMD 2015 score (rank correlation > 0.999). There are no tests of
rank averaging because rank averaging was removed.
"""

import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import imd_methods as im  # noqa: E402

CFG = yaml.safe_load(open(ROOT / "config" / "config.yaml"))
IMD = CFG["imd2015"]
FILE7 = ROOT / "data" / "raw" / "imd" / "imd2015" / "File_7_ID2015_ranks_deciles_scores_population.csv"


# --------------------------------------------------------------------------
# Transform (Technical Report Appendix F)
# --------------------------------------------------------------------------
def test_scaled_rank_is_one_for_most_deprived_and_one_over_n_for_least():
    n = 32844
    r = im.scaled_rank(pd.Series([1, n]), n)
    assert r.iloc[0] == pytest.approx(1.0)
    assert r.iloc[1] == pytest.approx(1.0 / n)


def test_exp_transform_endpoints_and_median():
    r = pd.Series([1.0, 1e-12, 0.1])
    x = im.exp_transform(r)
    assert x.iloc[0] == pytest.approx(100.0)          # most deprived -> 100
    assert x.iloc[1] == pytest.approx(0.0, abs=1e-9)   # least deprived -> ~0
    # Appendix F: the most deprived 10% (R >= 0.9) cover scores 50-100
    assert im.exp_transform(pd.Series([0.9])).iloc[0] == pytest.approx(
        -23 * math.log(1 - 0.9 * (1 - math.exp(-100 / 23))))
    assert im.exp_transform(pd.Series([0.9])).iloc[0] == pytest.approx(50.0, abs=0.5)


def test_exp_transform_is_monotone_in_rank():
    r = pd.Series(np.linspace(0.001, 1, 500))
    assert im.exp_transform(r).is_monotonic_increasing


# --------------------------------------------------------------------------
# Weights and variants
# --------------------------------------------------------------------------
def test_published_weights_sum_to_0_999():
    assert sum(IMD["domain_weights"].values()) == pytest.approx(0.999)


def test_reduced_variants_exclude_the_right_domains():
    v = IMD["variants"]
    assert "barriers" not in v["imd_ex_housing"] and len(v["imd_ex_housing"]) == 6
    assert "barriers" not in v["imd_ex_housing_living"] and "living" not in v["imd_ex_housing_living"]
    assert len(v["imd_ex_housing_living"]) == 5
    assert "barriers" not in v["imd_ex_income_housing"] and "income" not in v["imd_ex_income_housing"]
    assert len(v["imd_ex_income_housing"]) == 5


def test_rescaled_weights_sum_to_one_and_unrescaled_do_not():
    t = pd.DataFrame({d: [50.0] for d in im.OVERALL_DOMAINS})
    full_published = im.combine(t, im.OVERALL_DOMAINS, IMD["domain_weights"], rescale=False).iloc[0]
    assert full_published == pytest.approx(50.0 * 0.999)       # published weights AS published
    reduced = im.combine(t, IMD["variants"]["imd_ex_housing"], IMD["domain_weights"], rescale=True).iloc[0]
    assert reduced == pytest.approx(50.0)                       # all domains equal -> rescaled mean unchanged


# --------------------------------------------------------------------------
# REPRODUCTION TEST (required): 7 recombined domains vs published IMD score
# --------------------------------------------------------------------------
@pytest.mark.skipif(not FILE7.exists(), reason="IMD 2015 File 7 not downloaded")
def test_seven_domain_recombination_reproduces_published_imd_score():
    df = im.strip_columns(pd.read_csv(FILE7))
    im.assert_columns_present(df, IMD)
    assert len(df) == IMD["expected_n_lsoa"]
    df = df.set_index(IMD["columns"]["lsoa"])
    tx = im.transformed_domains(df, IMD, IMD["rank_source"])
    recombined = im.combine(tx, im.OVERALL_DOMAINS, IMD["domain_weights"], rescale=False)
    published = df[IMD["columns"]["overall_score"]].astype(float)
    rho = spearmanr(recombined, published)[0]
    max_abs_diff = (recombined - published).abs().max()
    assert rho > IMD["min_spearman_reproduction"]
    assert max_abs_diff < 0.01      # observed 0.0025 (rounding of the published score)


@pytest.mark.skipif(not FILE7.exists(), reason="IMD 2015 File 7 not downloaded")
def test_decile_function_reproduces_published_deciles():
    df = im.strip_columns(pd.read_csv(FILE7))
    dec = im.decile_from_values(df[IMD["columns"]["overall_score"]].astype(float))
    assert (dec == df[IMD["columns"]["overall_decile"]]).mean() > 0.999


# --------------------------------------------------------------------------
# Population-weighted aggregation of VALUES (never ranks / deciles)
# --------------------------------------------------------------------------
def test_pop_weighted_mean_uses_population_not_lsoa_count():
    df = pd.DataFrame({"m": ["A", "A"], "v": [0.50, 0.10], "pop": [1000, 9000]})
    out = im.pop_weighted_mean(df, "m", "v", "pop")
    assert out["A"] == pytest.approx((0.5 * 1000 + 0.1 * 9000) / 10000)   # 0.14, not 0.30


def test_income_rate_weighted_by_total_population_equals_exact_msoa_rate():
    # income score = (income-deprived people) / population, so the population-weighted
    # mean of the LSOA rates equals total income-deprived / total population exactly
    pop = np.array([1500.0, 1200.0, 1800.0])
    rate = np.array([0.30, 0.05, 0.12])
    df = pd.DataFrame({"m": "A", "rate": rate, "pop": pop})
    exact = (rate * pop).sum() / pop.sum()
    assert im.pop_weighted_mean(df, "m", "rate", "pop")["A"] == pytest.approx(exact)


def test_employment_rate_uses_its_own_working_age_denominator():
    df = pd.DataFrame({"m": "A", "rate": [0.2, 0.1], "pop_total": [1000, 1000], "pop_working": [100, 900]})
    assert im.pop_weighted_mean(df, "m", "rate", "pop_working")["A"] == pytest.approx(0.11)
    assert im.pop_weighted_mean(df, "m", "rate", "pop_total")["A"] == pytest.approx(0.15)


def test_zero_weight_total_fails_loudly():
    df = pd.DataFrame({"m": ["A"], "v": [1.0], "pop": [0.0]})
    with pytest.raises(ValueError):
        im.pop_weighted_mean(df, "m", "v", "pop")


def test_bottom20_share_is_population_weighted():
    df = pd.DataFrame({"m": ["A", "A"], "flag": [1, 0], "pop": [1000, 9000]})
    assert im.pop_share(df, "m", "flag", "pop")["A"] == pytest.approx(0.10)   # not 0.5


# --------------------------------------------------------------------------
# Moderator switches
# --------------------------------------------------------------------------
def _cfg(dep, inc):
    return {"moderators": {"deprivation_moderator": dep, "income_moderator": inc, "high_deprivation_quartile": 4}}


def test_default_switches_are_valid():
    assert im.resolve_moderators(CFG) == ("imd_ex_housing", "saie")


def test_same_measure_for_both_switches_is_refused():
    with pytest.raises(ValueError, match="Same underlying measure"):
        im.resolve_moderators(_cfg("imd_income_rate", "imd_income_rate"))


def test_unknown_choices_are_refused():
    with pytest.raises(ValueError):
        im.resolve_moderators(_cfg("imd_overall", "saie"))
    with pytest.raises(ValueError):
        im.resolve_moderators(_cfg("imd_ex_housing", "census"))


def test_overlap_warning_when_income_domain_is_double_counted():
    assert im.moderator_overlap_warnings("imd_ex_housing", "imd_income_rate")
    assert not im.moderator_overlap_warnings("imd_ex_income_housing", "imd_income_rate")
    assert not im.moderator_overlap_warnings("imd_ex_housing", "saie")


def test_build_moderators_signs_and_quartiles():
    n = 8
    base = pd.DataFrame({
        "imd_ex_housing": np.arange(n, dtype=float),                       # higher = more deprived
        "imd_ex_housing_living": np.arange(n, dtype=float),
        "imd_ex_income_housing": np.arange(n, dtype=float),
        "imd2015_income_rate_msoa": np.arange(n, dtype=float),
        "baseline_income_bhc_2011_12": np.arange(n, 0, -1, dtype=float) * 1000,   # richer for low index
    }, index=[f"E0{i}" for i in range(n)])
    m = im.build_moderators(base, _cfg("imd_ex_housing", "saie"))
    assert m["deprivation_quartile"].iloc[-1] == 4 and m["deprivation_quartile"].iloc[0] == 1   # 4 = most deprived
    assert m["high_deprivation_q4"].sum() == n // 4
    assert m["deprivation_z"].mean() == pytest.approx(0, abs=1e-12)
    assert m["deprivation_z"].std(ddof=1) == pytest.approx(1)
    assert m["income_quartile"].iloc[-1] == 1 and m["low_income_q1"].iloc[-1] == 1        # SAIE: 1 = poorest
    m2 = im.build_moderators(base, _cfg("imd_ex_income_housing", "imd_income_rate"))
    assert m2["income_moderator_direction"].iloc[0] == "higher = more deprived"
    assert m2["low_income_q1"].iloc[-1] == 1     # highest income deprivation is still flagged 'poor'
