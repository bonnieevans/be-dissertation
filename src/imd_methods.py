"""Pure functions for the IMD 2015 revision (no file I/O, so they are unit
tested directly: tests/test_imd_population_weighting.py).

Method source: English Indices of Deprivation 2015 Technical Report,
Section 3.6 (Stage 5), Section 3.7 (Stage 6, weights, Table 3.1) and
Appendix F (exponential transformation):

    R = (N - rank + 1) / N        # scaled rank, R = 1 for the MOST deprived
    X = -23 * ln(1 - R * (1 - exp(-100/23)))

Population-weighted means are taken of LSOA *values* (rates, scores,
recombined indices) - never of ranks or deciles, which are not additive.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

OVERALL_DOMAINS = ["income", "employment", "education", "health", "crime", "barriers", "living"]
DEPRIVATION_CHOICES = ["imd_ex_housing", "imd_ex_housing_living", "imd_ex_income_housing", "imd_income_rate"]
INCOME_CHOICES = ["saie", "imd_income_rate"]


# --------------------------------------------------------------------------
# Loading / column mapping
# --------------------------------------------------------------------------
def strip_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Strip stray whitespace from headers (File 7's working-age header has a
    trailing space). Names are mapped from config, never guessed."""
    out = df.copy()
    out.columns = [str(c).strip() for c in out.columns]
    return out


def required_columns(imd_cfg: dict) -> list[str]:
    cols = imd_cfg["columns"]
    names = [cols[k] for k in ("lsoa", "overall_score", "overall_rank", "overall_decile",
                               "population_total", "population_working_age")]
    for d in cols["domains"].values():
        names += [d["score"], d["rank"]]
    return [n.strip() for n in names]


def assert_columns_present(df: pd.DataFrame, imd_cfg: dict) -> None:
    missing = [c for c in required_columns(imd_cfg) if c not in df.columns]
    if missing:
        raise KeyError(f"Configured IMD columns not found in File 7: {missing}\n"
                       f"Actual headers: {list(df.columns)}")


# --------------------------------------------------------------------------
# Transform and recombination
# --------------------------------------------------------------------------
def scaled_rank(rank: pd.Series, n: int) -> pd.Series:
    """R in (0,1]; rank 1 (most deprived) -> R = 1; rank N -> R = 1/N."""
    return (n - rank.astype(float) + 1.0) / n


def exp_transform(r: pd.Series, constant: float = 23.0, scale: float = 100.0) -> pd.Series:
    """Appendix F: X = -23 ln(1 - R (1 - exp(-100/23))); X in (0, 100]."""
    return -constant * np.log(1.0 - r * (1.0 - math.exp(-scale / constant)))


def domain_ranks(df: pd.DataFrame, imd_cfg: dict, source: str) -> pd.DataFrame:
    """Rank (1 = most deprived) per domain, from published ranks or by
    re-ranking the (rounded) published scores."""
    out = {}
    for name, d in imd_cfg["columns"]["domains"].items():
        if source == "published":
            out[name] = df[d["rank"].strip()].astype(float)
        elif source == "score":
            out[name] = df[d["score"].strip()].rank(ascending=False, method="min")
        else:
            raise ValueError(f"rank_source must be 'published' or 'score', got {source!r}")
    return pd.DataFrame(out, index=df.index)


def transformed_domains(df: pd.DataFrame, imd_cfg: dict, source: str | None = None) -> pd.DataFrame:
    source = source or imd_cfg["rank_source"]
    n = len(df)
    ranks = domain_ranks(df, imd_cfg, source)
    return ranks.apply(lambda s: exp_transform(scaled_rank(s, n),
                                               imd_cfg["exp_transform_constant"],
                                               imd_cfg["exp_transform_scale"]))


def combine(transformed: pd.DataFrame, domains: list[str], weights: dict, rescale: bool) -> pd.Series:
    """Weighted sum of transformed domains. rescale=True makes the weights of
    the included domains sum to 1 (used for the reduced variants); the full
    7-domain IMD uses the published weights as published (they sum to 0.999)."""
    w = pd.Series({d: weights[d] for d in domains}, dtype=float)
    if rescale:
        w = w / w.sum()
    return sum(transformed[d] * w[d] for d in domains)


def decile_from_values(values: pd.Series, n: int | None = None) -> pd.Series:
    """National decile, 1 = most deprived 10%, for a 'higher = more deprived'
    series: ceil(10 * rank / N) with rank 1 = highest value, ties share the
    best rank. (Applied to the published overall IMD score this reproduces
    99.99% of the published deciles; applied to published ranks, 100%.)"""
    n = n or len(values)
    rank = values.rank(ascending=False, method="min")
    return np.ceil(rank * 10.0 / n).astype(int)


# --------------------------------------------------------------------------
# Aggregation
# --------------------------------------------------------------------------
def pop_weighted_mean(df: pd.DataFrame, group: str, value: str, weight: str) -> pd.Series:
    """sum(w * v) / sum(w) within group. Fails loudly on a zero denominator."""
    num = (df[value] * df[weight]).groupby(df[group]).sum()
    den = df[weight].groupby(df[group]).sum()
    if (den <= 0).any():
        raise ValueError(f"Non-positive weight total for {value} in {(den <= 0).sum()} groups")
    return num / den


def pop_share(df: pd.DataFrame, group: str, flag: str, weight: str) -> pd.Series:
    """Share of the group's weight carried by rows with flag == 1."""
    num = (df[flag] * df[weight]).groupby(df[group]).sum()
    den = df[weight].groupby(df[group]).sum()
    return num / den


# --------------------------------------------------------------------------
# Standardising helpers (England-only, unweighted, deterministic ties)
# --------------------------------------------------------------------------
def zscore(s: pd.Series) -> pd.Series:
    return (s - s.mean()) / s.std(ddof=1)


def ntile(s: pd.Series, k: int = 4) -> pd.Series:
    """Equal-count groups 1..k in ascending order of s. Ties are broken by
    row order (stable), so results are reproducible; quartile membership of
    exactly tied values is therefore arbitrary - flagged where it matters."""
    order = s.rank(method="first")
    return np.ceil(order * k / len(s)).astype(int)


# --------------------------------------------------------------------------
# Moderator switches
# --------------------------------------------------------------------------
def resolve_moderators(cfg: dict) -> tuple[str, str]:
    """Validate the two config switches. Returns (deprivation, income).
    Refuses (ValueError) if the same underlying measure is chosen for both;
    logs nothing itself - callers log the returned choices."""
    m = cfg["moderators"]
    dep, inc = m["deprivation_moderator"], m["income_moderator"]
    if dep not in DEPRIVATION_CHOICES:
        raise ValueError(f"deprivation_moderator {dep!r} not in {DEPRIVATION_CHOICES}")
    if inc not in INCOME_CHOICES:
        raise ValueError(f"income_moderator {inc!r} not in {INCOME_CHOICES}")
    if dep == "imd_income_rate" and inc == "imd_income_rate":
        raise ValueError("Same underlying measure (imd_income_rate) selected for both "
                         "deprivation_moderator and income_moderator - choose different measures.")
    return dep, inc


def moderator_overlap_warnings(dep: str, inc: str) -> list[str]:
    """Non-fatal: the IMD indices that still contain the Income domain overlap
    with imd_income_rate when that is the income moderator."""
    warn = []
    if inc == "imd_income_rate" and dep in ("imd_ex_housing", "imd_ex_housing_living"):
        warn.append(f"{dep} contains the Income domain (weight 22.5%), which overlaps the income "
                    "moderator imd_income_rate; consider imd_ex_income_housing.")
    return warn


def build_moderators(baseline: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """One row per MSOA11 in, same index out, adding the selected moderators:
    deprivation_* (always higher = more deprived) and income_* (direction
    follows the source - see income_moderator_direction)."""
    dep, inc = resolve_moderators(cfg)
    q_top = cfg["moderators"]["high_deprivation_quartile"]
    out = pd.DataFrame(index=baseline.index)

    dep_col = {"imd_ex_housing": "imd_ex_housing", "imd_ex_housing_living": "imd_ex_housing_living",
               "imd_ex_income_housing": "imd_ex_income_housing", "imd_income_rate": "imd2015_income_rate_msoa"}[dep]
    v = baseline[dep_col]
    out["deprivation_moderator_used"] = dep
    out["deprivation_moderator_value"] = v
    out["deprivation_z"] = zscore(v)
    out["deprivation_quartile"] = ntile(v, 4)               # 4 = most deprived
    out["high_deprivation_q4"] = (out["deprivation_quartile"] == q_top).astype(int)

    if inc == "saie":
        iv = baseline["baseline_income_bhc_2011_12"]
        out["income_moderator_value"] = iv
        out["income_moderator_direction"] = "higher = richer"
        out["income_z"] = zscore(iv)
        out["income_quartile"] = ntile(iv, 4)                # 1 = poorest
        out["low_income_q1"] = (out["income_quartile"] == 1).astype(int)
        out["below_median_income"] = (iv < iv.median()).astype(int)
    else:
        iv = baseline["imd2015_income_rate_msoa"]
        out["income_moderator_value"] = iv
        out["income_moderator_direction"] = "higher = more deprived"
        out["income_z"] = zscore(iv)
        out["income_quartile"] = ntile(iv, 4)                # 4 = most income-deprived
        out["low_income_q1"] = (out["income_quartile"] == 4).astype(int)   # 'poor' flag keeps its meaning
        out["below_median_income"] = (iv > iv.median()).astype(int)
    out["income_moderator_used"] = inc
    return out
