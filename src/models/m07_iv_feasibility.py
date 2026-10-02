"""M07 - what variation could an instrument have at MSOA scale? (mechanical check, no instrument is estimated)

This does NOT estimate an IV model. It shows, for the fixed-effects structures used here, how much variation of
different TYPES of candidate variable survives the fixed effects. An instrument must (i) move the treatment and
(ii) vary within the fixed-effects cells; this checks (ii) mechanically, and (i)'s raw material: how much of the
treatment's own variation lies within the cells.

Types checked (synthetic stand-ins; the real variables are discussed in docs/iv_literature_note.md):
  1. LAD-level, constant over time
  2. LAD-level, changing over time (LAD x year)
  3. LAD-level constant characteristic x national year shock
  4. MSOA-level constant characteristic x national year shock (a "shift-share"-type structure; the stand-in for
     the characteristic is the baseline log density, used only to show the algebra)
  5. MSOA-level, changing over time
The same arithmetic applies to any LAD x year variable, including a place-based funding indicator defined at LAD-year
level (the policy registry is not part of this build).
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from model_common import *  # noqa: F401,F403
import model_common as MC
from m03_moderators import within

log = get_logger("m07_iv_feasibility")
T = out_dir(MODEL_DIR, "tables")


def main() -> int:
    df = load_panel()
    d = estimation_sample(df).reset_index(drop=True)
    rng = np.random.default_rng(7)
    lad_codes = d.lad23cd_analysis.unique()
    lad_const = d.lad23cd_analysis.map(dict(zip(lad_codes, rng.standard_normal(len(lad_codes)))))
    ly = d.lad_year.unique()
    lad_time = d.lad_year.map(dict(zip(ly, rng.standard_normal(len(ly)))))
    yrs = np.sort(d.year.unique())
    shock = d.year.map(dict(zip(yrs, rng.standard_normal(len(yrs)))))
    z_dens = d["density_z_eda"]
    cand = {"1. LAD-level, constant over time": lad_const,
            "2. LAD-level, changing over time (LAD x year)": lad_time,
            "3. LAD-level constant x national year shock": lad_const * shock,
            "4. MSOA-level constant (baseline density) x national year shock": z_dens * shock,
            "5. MSOA-level, changing over time (random)": pd.Series(rng.standard_normal(len(d)), index=d.index)}

    # ---- 1. How much of each candidate's variation survives the fixed effects? ----------------------------------------
    # WHAT: the share of each stand-in variable's variance that remains after the within transformation for M3 (MSOA + year) and M6
    #       (MSOA + LAD x year). Also the same for the treatment itself.
    # LOOK FOR: types whose remaining share is zero (they are absorbed by the fixed effects and cannot be separately
    #           estimated or used as instruments in that model), and the remaining share of the treatment.
    rows = []
    allv = {**cand, "treatment x (new-builds per 1,000, lag 1)": d[TREATMENT]}
    for nm, s in allv.items():
        D = pd.DataFrame({"v": s.values}, index=d.index)
        row = {"variable": nm, "total_variance": float(D.v.var())}
        for fe in ("M3", "M6"):
            w = within(d.assign(_v=D.v.values), ["_v"], fe)["_v"]
            row[f"share_variance_left_{fe}"] = float(w.var() / D.v.var()) if D.v.var() > 0 else np.nan
        rows.append(row)
    res = pd.DataFrame(rows)
    res.to_csv(T / "m07_variation_surviving_fixed_effects.csv", index=False)
    log.info("Variation surviving the fixed effects:\n" + res.round(4).to_string())

    # ---- 2. Rank check ----------------------------------------------------------------------------------------------------------
    # WHAT: is the LAD x year stand-in (type 2) linearly dependent on the LAD x year fixed effects? Computed as the R2 of regressing
    #       it on LAD x year dummies (via group means) - an R2 of 1 means perfect collinearity.
    # LOOK FOR: R2 of exactly 1 (the variable would be dropped by the estimator as collinear).
    r2 = {}
    for nm in ("2. LAD-level, changing over time (LAD x year)", "3. LAD-level constant x national year shock"):
        s = cand[nm]
        gm = s.groupby(d.lad_year.values).transform("mean")
        r2[nm] = float(1 - ((s - gm) ** 2).sum() / ((s - s.mean()) ** 2).sum())
    pd.Series(r2, name="R2_on_LADxYear_dummies").to_csv(T / "m07_collinearity_with_lad_year_effects.csv")
    MC.save_registry("m07")
    log.info("M07 complete.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
