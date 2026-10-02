"""Shared helpers for the baseline specification lab (src/models).

These are DIAGNOSTIC regressions used to choose a specification; they are not the dissertation's
reported results. Every specification that is run is written to a registry
(outputs/models/tables/spec_registry.csv), including ones that do not behave, so the number of
specifications tried is on record (relevant to multiple-testing adjustments).

Conventions
- Outcome y = log median NOMINAL price per m2 (`log_median_ppsqm`).
- Treatment x = new-builds per 1,000 baseline (2011) households completed in the previous year
  (`newbuilds_lag1_per_1000`).
- Moderators enter as England-wide z-scores (mean 0, SD 1 across the 6,791 MSOAs), interacted with x
  at regression time. Moderator main effects are absorbed by MSOA fixed effects.
- Estimation sample: main sample 2016-2023, restricted to LADs with at least 2 MSOAs (a LAD with one
  MSOA cannot be separated from its LAD x year effect).
- Default standard errors: clustered by LAD (296 clusters).
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import pyfixest as pf
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "eda"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import eda_common as EC  # noqa: E402
from eda_common import (BASELINE, MODEL_DIR, MODERATORS, OPTIONAL_MODERATORS, OUTCOME, PRIMARY_MODERATORS,  # noqa: E402,F401
                        TREATMENT, Z_COL, get_logger, load_panel, out_dir, plt, savefig, style)

VCOV = {"lad": {"CRV1": "lad23cd_analysis"}, "msoa": {"CRV1": "msoa11cd"}, "region": {"CRV1": "region_code"},
        "lad_and_year": {"CRV1": "lad23cd_analysis + year"}, "hc1": "hetero", "iid": "iid"}
REGISTRY: list[dict] = []


def estimation_sample(df: pd.DataFrame, full: bool = False) -> pd.DataFrame:
    """Main sample (2016-2023) or the full panel, dropping LADs with a single MSOA."""
    d = df if full else df[df.main_sample]
    n = d.groupby("lad23cd_analysis")["msoa11cd"].transform("nunique")
    return d[n >= 2].copy()


def fit(formula: str, data: pd.DataFrame, spec_id: str, vcov: str = "lad", weights: str | None = None,
        note: str = "", sample: str = "main_2016_2023"):
    """Estimate with pyfixest and record the specification in the registry."""
    kwargs = {"vcov": VCOV[vcov]}
    if weights:
        kwargs["weights"] = weights
    try:
        f = pf.feols(formula, data=data, **kwargs)
    except Exception as e:  # recorded, not hidden
        REGISTRY.append({"spec_id": spec_id, "formula": formula, "sample": sample, "vcov": vcov, "weights": weights,
                         "n_obs": np.nan, "r2_within": np.nan, "status": f"FAILED: {type(e).__name__}: {e}", "note": note})
        raise
    REGISTRY.append({"spec_id": spec_id, "formula": formula, "sample": sample, "vcov": vcov, "weights": weights,
                     "n_obs": f._N, "r2_within": getattr(f, "_r2_within", np.nan), "status": "ok", "note": note})
    return f


def save_registry(name: str) -> None:
    out_dir(MODEL_DIR, "tables")
    pd.DataFrame(REGISTRY).to_csv(MODEL_DIR / "tables" / f"spec_registry_{name}.csv", index=False)


def coef_table(f, spec_id: str, label: str = "", terms: list[str] | None = None, x_sd: float | None = None) -> pd.DataFrame:
    t = f.tidy().reset_index().rename(columns={"Coefficient": "term"})
    if terms is not None:
        t = t[t["term"].isin(terms)]
    out = pd.DataFrame({"spec_id": spec_id, "label": label, "term": t["term"], "coef": t["Estimate"], "se": t["Std. Error"],
                        "t": t["t value"], "p": t["Pr(>|t|)"], "ci_low": t["2.5%"], "ci_high": t["97.5%"],
                        "n_obs": f._N, "n_clusters": (f._G[0] if getattr(f, "_G", None) is not None and len(f._G) > 0 else np.nan),
                        "r2_within": getattr(f, "_r2_within", np.nan)})
    out["pct_change_per_unit"] = 100 * (np.exp(out["coef"]) - 1)         # for the treatment term
    if x_sd is not None:
        out["pct_change_per_1sd_x"] = 100 * (np.exp(out["coef"] * x_sd) - 1)
    return out


def wald(f, terms: list[str]) -> dict:
    """Joint Wald test that the named coefficients are all zero, using the model's own (clustered) vcov.
    Reports the chi-square statistic and the F version with (q, G-1) degrees of freedom."""
    names = list(f._coefnames)
    idx = [names.index(t) for t in terms]
    b = np.asarray(f._beta_hat)[idx]
    V = np.asarray(f._vcov)[np.ix_(idx, idx)]
    W = float(b @ np.linalg.solve(V, b))
    q = len(idx)
    G = f._G[0] if getattr(f, "_G", None) is not None and len(f._G) > 0 else np.inf
    F = W / q
    return {"wald_chi2": W, "q": q, "p_chi2": 1 - stats.chi2.cdf(W, q), "F": F,
            "p_F": 1 - stats.f.cdf(F, q, G - 1) if np.isfinite(G) else np.nan}


def ssr(f) -> float:
    return float(np.sum(np.asarray(f._u_hat) ** 2))


def nested_f(f_restricted, f_full, extra_params: int, k_full_params: int) -> dict:
    """Classical F test for the extra parameters in f_full (iid errors; reported for reference only,
    the cluster-robust tests are the ones to rely on). k_full_params = total estimated parameters of f_full."""
    N = f_full._N
    s_r, s_u = ssr(f_restricted), ssr(f_full)
    df2 = N - k_full_params
    F = ((s_r - s_u) / extra_params) / (s_u / df2)
    return {"F": F, "df1": extra_params, "df2": df2, "p": 1 - stats.f.cdf(F, extra_params, df2)}


def units_sentence(coef: float, per: str = "one additional new-build per 1,000 households (previous year)") -> str:
    return (f"A coefficient of {coef:.5f} means that {per} is associated with a "
            f"{100 * (np.exp(coef) - 1):.4f}% difference in the median nominal price per m2 (log-point coefficient x 100, "
            f"approximately), holding the fixed effects in the model constant.")


def add_moderator_interactions(d: pd.DataFrame, mods: list[str]) -> pd.DataFrame:
    """Build x * z interaction columns in memory (never stored in the permanent datasets)."""
    d = d.copy()
    for m in mods:
        d[f"x_{m}"] = d[TREATMENT] * d[Z_COL[m]]
    return d


def twoway_demean(M: pd.DataFrame, entity: pd.Series, time: pd.Series) -> pd.DataFrame:
    """Within transformation for a balanced panel (exact for MSOA + year)."""
    X = M - M.groupby(entity.values).transform("mean")
    return X - X.groupby(time.values).transform("mean")
