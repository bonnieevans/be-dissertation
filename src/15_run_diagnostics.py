"""Diagnostics for the final panel's socioeconomic moderators: a correlation
matrix across the six z-scores, and VIF for the joint interaction model
(Model 6: all four required interactions together). Report only - the
brief is explicit that a variable is never dropped solely for being
correlated with another.

Per the user's instruction, interaction terms are NOT stored in the
permanent panel - they are computed here, in-memory, purely for this
diagnostic check.
"""

from __future__ import annotations

import sys

import pandas as pd
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.tools import add_constant

from utils import PROJECT_ROOT, get_logger, load_config

log = get_logger("15_run_diagnostics")

FINAL_PANEL = PROJECT_ROOT / "data" / "processed" / "final_msoa_year_dissertation_panel.parquet"
QA_DIR = PROJECT_ROOT / "outputs" / "qa"


def main() -> int:
    cfg = load_config()
    zscore_vars = cfg["diagnostics"]["zscore_variables"]

    if not FINAL_PANEL.exists():
        log.error(f"{FINAL_PANEL} not found - run src/14_merge_final_panel.py first.")
        return 1

    cols = list(set(zscore_vars + ["newbuilds_lag1_per_1000"]))
    df = pd.read_parquet(FINAL_PANEL, columns=cols)

    # --- Correlation matrix across the six baseline z-scores ---
    corr_df = df[zscore_vars].dropna()
    corr = corr_df.corr()
    corr.to_csv(QA_DIR / "baseline_characteristic_correlations.csv")
    log.info(f"Correlation matrix across {len(corr_df):,} complete MSOA-year rows:\n{corr}")

    # --- VIF for the joint model's regressors: primary treatment + the four
    # required interaction terms, computed on the fly (not stored) ---
    df["newbuild_x_income"] = df["newbuilds_lag1_per_1000"] * df["income_z"]
    df["newbuild_x_deprivation"] = df["newbuilds_lag1_per_1000"] * df["deprivation_z"]
    df["newbuild_x_socialrent"] = df["newbuilds_lag1_per_1000"] * df["social_rent_z"]
    df["newbuild_x_density"] = df["newbuilds_lag1_per_1000"] * df["density_z"]

    joint_vars = ["newbuilds_lag1_per_1000", "newbuild_x_income", "newbuild_x_deprivation",
                  "newbuild_x_socialrent", "newbuild_x_density"]
    vif_df = df[joint_vars].dropna()
    X = add_constant(vif_df)
    vif_rows = []
    for i, col in enumerate(X.columns):
        if col == "const":
            continue
        vif = variance_inflation_factor(X.values, i)
        vif_rows.append({"variable": col, "vif": vif})
    vif_result = pd.DataFrame(vif_rows)
    vif_result.to_csv(QA_DIR / "interaction_vif.csv", index=False)
    log.info(f"VIF for the joint-model regressors (n={len(vif_df):,}):\n{vif_result}")

    high_vif = vif_result[vif_result["vif"] > 10]
    if not high_vif.empty:
        log.warning(
            f"VIF > 10 for: {high_vif['variable'].tolist()} - reported per the brief, NOT dropped. "
            "Consider mean-centering or excluding overlapping interactions at the modelling stage if needed."
        )
    else:
        log.info("No VIF exceeds 10 among the joint-model regressors.")

    log.info("Diagnostics complete.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
