"""Diagnostics for the IMD 2015 revision. Writes everything to
outputs/qa/imd_revision/:

  - Pearson and Spearman correlations between log SAIE income, the IMD
    income rate, imd_ex_housing, imd_ex_income_housing, the overall IMD
    (robustness only) and the Barriers domain score; scatter of SAIE vs the
    IMD income rate.
  - z-score correlation matrix and joint-model VIFs under
      (a) SAIE + imd_ex_housing
      (b) SAIE + imd_ex_income_housing
    (treatment + the four interactions; interactions built in memory only).
  - Before/after deprivation-quartile membership when moving from the
    overall IMD to imd_ex_housing.
  - IMD_REVISION_REPORT.md assembled from those results.
Interaction terms are never written to the panel.
"""

from __future__ import annotations

import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.tools import add_constant

import imd_methods as im
from utils import PROJECT_ROOT, get_logger, load_config

log = get_logger("15b_imd_revision_diagnostics")

BASELINE = PROJECT_ROOT / "data" / "processed" / "msoa_baseline_characteristics.parquet"
PANEL = PROJECT_ROOT / "data" / "processed" / "final_msoa_year_dissertation_panel.parquet"
QA = PROJECT_ROOT / "outputs" / "qa" / "imd_revision"

CORR_VARS = {
    "log_saie_income": "log_baseline_income",
    "imd_income_rate": "imd2015_income_rate_msoa",
    "imd_ex_housing": "imd_ex_housing",
    "imd_ex_income_housing": "imd_ex_income_housing",
    "imd_overall_score_pw (robustness only)": "imd2015_overall_score_pw",
    "imd_barriers_domain_score": "imd2015_barriers_score_pw",
}


def corr_tables(b: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    d = pd.DataFrame({k: b[v] for k, v in CORR_VARS.items()})
    return d.corr(method="pearson"), d.corr(method="spearman")


def vif_table(df: pd.DataFrame, label: str, sample: str) -> pd.DataFrame:
    X = add_constant(df.dropna())
    rows = [{"model": label, "sample": sample, "variable": c, "vif": variance_inflation_factor(X.values, i),
             "n_obs": len(X)} for i, c in enumerate(X.columns) if c != "const"]
    return pd.DataFrame(rows)


def main() -> int:
    cfg = load_config()
    for p in (BASELINE, PANEL):
        if not p.exists():
            log.error(f"{p} not found - run 14_merge_final_panel.py first.")
            return 1
    QA.mkdir(parents=True, exist_ok=True)
    b = pd.read_parquet(BASELINE).set_index("msoa11cd")
    dep_choice, inc_choice = im.resolve_moderators(cfg)

    # ---------------------------------------------------------------- 1. correlations
    pear, spear = corr_tables(b)
    pear.to_csv(QA / "corr_pearson.csv")
    spear.to_csv(QA / "corr_spearman.csv")
    long = []
    keys = list(CORR_VARS)
    for i, a in enumerate(keys):
        for c in keys[i + 1:]:
            long.append({"var_a": a, "var_b": c, "pearson": pear.loc[a, c], "spearman": spear.loc[a, c]})
    long = pd.DataFrame(long)
    long.to_csv(QA / "corr_pairs_long.csv", index=False)
    log.info(f"Pearson correlations (n={len(b):,} MSOA11s):\n{pear.round(3)}")
    log.info(f"Spearman correlations:\n{spear.round(3)}")

    # scatter SAIE vs IMD income rate
    r_p = pearsonr(b["baseline_income_bhc_2011_12"], b["imd2015_income_rate_msoa"])[0]
    r_plog = pearsonr(b["log_baseline_income"], b["imd2015_income_rate_msoa"])[0]
    r_s = spearmanr(b["baseline_income_bhc_2011_12"], b["imd2015_income_rate_msoa"])[0]
    fig, ax = plt.subplots(figsize=(6.5, 5))
    ax.scatter(b["baseline_income_bhc_2011_12"] / 1000, b["imd2015_income_rate_msoa"], s=5, alpha=0.35)
    ax.set_xlabel("SAIE net household income before housing costs, FYE2012 (GBP '000 per year; higher = richer)")
    ax.set_ylabel("IMD 2015 income deprivation rate (MSOA; higher = more deprived)")
    ax.set_title(f"SAIE income vs IMD income rate, 6,791 MSOA11s\nPearson {r_p:.3f} (raw SAIE) / {r_plog:.3f} (log SAIE); Spearman {r_s:.3f}")
    fig.savefig(QA / "scatter_saie_vs_imd_income_rate.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # ---------------------------------------------------------------- 2. z-score corr + VIF under (a) and (b)
    pan = pd.read_parquet(PANEL, columns=["msoa11cd", "year", "main_sample", "newbuilds_lag1_per_1000",
                                          "social_rent_z", "density_z", "degree_share_z", "unemployment_z"])
    z_inc = im.zscore(b["baseline_income_bhc_2011_12"]).rename("income_z")           # SAIE for both (a) and (b)
    variants = {"a_saie_plus_imd_ex_housing": "imd_ex_housing",
                "b_saie_plus_imd_ex_income_housing": "imd_ex_income_housing"}
    corr_out, vif_out = [], []
    for label, dep_col in variants.items():
        z_dep = im.zscore(b[dep_col]).rename("deprivation_z")
        zs = pd.concat([z_inc, z_dep], axis=1)
        cm = (b[["social_rent_z", "density_z", "degree_share_z", "unemployment_z"]].join(zs)
              [["income_z", "deprivation_z", "social_rent_z", "density_z", "degree_share_z", "unemployment_z"]]
              .corr())
        cm.insert(0, "model", label)
        corr_out.append(cm.reset_index().rename(columns={"index": "variable"}))
        d = pan.join(zs, on="msoa11cd")
        d["x_income"] = d["newbuilds_lag1_per_1000"] * d["income_z"]
        d["x_deprivation"] = d["newbuilds_lag1_per_1000"] * d["deprivation_z"]
        d["x_socialrent"] = d["newbuilds_lag1_per_1000"] * d["social_rent_z"]
        d["x_density"] = d["newbuilds_lag1_per_1000"] * d["density_z"]
        reg = ["newbuilds_lag1_per_1000", "x_income", "x_deprivation", "x_socialrent", "x_density"]
        vif_out.append(vif_table(d[reg], label, "all_years_2012_2023"))
        vif_out.append(vif_table(d.loc[d["main_sample"], reg], label, "main_sample_2016_2023"))
    corr_z = pd.concat(corr_out, ignore_index=True)
    vif = pd.concat(vif_out, ignore_index=True)
    corr_z.to_csv(QA / "zscore_correlation_matrix_by_model.csv", index=False)
    vif.to_csv(QA / "joint_model_vif_by_model.csv", index=False)
    log.info(f"Joint-model VIFs:\n{vif.round(3).to_string(index=False)}")
    if (vif["vif"] > 10).any():
        log.warning("A VIF exceeds 10 - reported, not dropped, per the brief.")

    # ---------------------------------------------------------------- 3. quartile movement
    q = pd.DataFrame(index=b.index)
    q["overall_imd"] = im.ntile(b["imd2015_overall_score_pw"], 4)
    q["imd_ex_housing"] = im.ntile(b["imd_ex_housing"], 4)
    q["imd_ex_income_housing"] = im.ntile(b["imd_ex_income_housing"], 4)
    q["legacy_bottom20_share"] = im.ntile(b["imd2015_overall_bottom20_popshare_legacy"], 4)
    comparisons = [("overall_imd", "imd_ex_housing", "PRIMARY: overall IMD score -> imd_ex_housing"),
                   ("overall_imd", "imd_ex_income_housing", "overall IMD score -> imd_ex_income_housing"),
                   ("legacy_bottom20_share", "imd_ex_housing",
                    "previous moderator (quartile of old bottom-20 share; tied zeros, order arbitrary) -> imd_ex_housing")]
    summ, tables = [], {}
    for before, after, label in comparisons:
        ct = pd.crosstab(q[before], q[after])
        ct.index.name, ct.columns.name = f"quartile_before ({before})", f"quartile_after ({after})"
        tables[(before, after)] = ct
        ct.to_csv(QA / f"quartile_transition_{before}_to_{after}.csv")
        changed = float((q[before] != q[after]).mean())
        moved2 = float(((q[before] - q[after]).abs() >= 2).mean())
        top_before = q[before] == 4
        top_after = q[after] == 4
        summ.append({"comparison": label, "n_msoa11": len(q), "share_changing_quartile": changed,
                     "share_moving_2plus_quartiles": moved2,
                     "share_top_quartile_before": float(top_before.mean()),
                     "of_top_before_share_still_top": float((top_before & top_after).sum() / top_before.sum()),
                     "spearman_between_measures": spearmanr(b[{'overall_imd': 'imd2015_overall_score_pw',
                                                                'legacy_bottom20_share': 'imd2015_overall_bottom20_popshare_legacy'}[before]],
                                                            b[{'imd_ex_housing': 'imd_ex_housing',
                                                               'imd_ex_income_housing': 'imd_ex_income_housing'}[after]])[0]})
    summ = pd.DataFrame(summ)
    summ.to_csv(QA / "quartile_movement_summary.csv", index=False)
    log.info(f"Quartile movement:\n{summ.round(4).to_string(index=False)}")

    # ---------------------------------------------------------------- 4. report
    val = pd.read_csv(QA / "imd_reproduction_validation.csv")
    vcfg = val[val["is_configured_source"]].iloc[0]
    vscore = val[~val["is_configured_source"]].iloc[0]
    f9 = pd.read_csv(QA / "imd_transform_vs_file9.csv") if (QA / "imd_transform_vs_file9.csv").exists() else None
    agree = pd.read_csv(QA / "imd_rank_source_agreement.csv")

    def md_table(df: pd.DataFrame, floatfmt: str = ".3f") -> str:
        d = df.copy()
        for c in d.columns:
            if pd.api.types.is_float_dtype(d[c]):
                d[c] = d[c].map(lambda x: format(x, floatfmt))
        head = "| " + " | ".join(map(str, d.columns)) + " |\n|" + "---|" * len(d.columns) + "\n"
        return head + "\n".join("| " + " | ".join(map(str, r)) + " |" for r in d.values) + "\n"

    pc = pear.round(3).reset_index().rename(columns={"index": ""})
    sc = spear.round(3).reset_index().rename(columns={"index": ""})
    vif_wide = vif.pivot_table(index=["sample", "variable"], columns="model", values="vif").reset_index()
    prim = tables[("overall_imd", "imd_ex_housing")]
    prim_pct = prim.div(prim.sum(axis=1), axis=0).round(3)
    prim_pct.columns = [f"ex_housing Q{c}" for c in prim_pct.columns]
    prim_pct.index = [f"overall IMD Q{i}" for i in prim_pct.index]

    report = f"""# IMD 2015 revision: results

Generated by `src/15b_imd_revision_diagnostics.py` (reproduction numbers from
`src/07_prepare_imd2015.py`). Moderators in the current build:
deprivation = `{dep_choice}`, income = `{inc_choice}`.
Sign convention: SAIE income higher = richer; every IMD measure higher = more deprived.

## 1. Reproduction test (hard stop in 07; unit test in tests/)

Recombining all 7 domains with the Technical Report method (Appendix F transform,
published weights as published, which sum to 0.999) against the published IMD 2015 score,
{int(vcfg['n_lsoa']):,} LSOAs:

{md_table(val[['rank_source','spearman','pearson','max_abs_diff','mean_abs_diff','is_configured_source']], '.6f')}
**Result: PASSED.** With the configured `rank_source = published`: Spearman
{vcfg['spearman']:.6f}, Pearson {vcfg['pearson']:.6f}, **maximum absolute difference
{vcfg['max_abs_diff']:.4f}** index points (mean {vcfg['mean_abs_diff']:.5f}); the threshold was Spearman > 0.999.

Deviation from the brief, flagged: step 3a says to rank each domain score. The published
scores in File 7 are rounded to 3 decimals, so re-ranking them creates large tie blocks
(Income: {agree.loc[agree.domain=='income','share_ranks_identical'].iloc[0]:.1%} of re-ranked ranks equal the
published ranks). The published ranks were computed by MHCLG from unrounded scores, and
using them reproduces the official transformed domain scores in File 9
(max difference {f9[f9.rank_source=='published'].max_abs_diff_vs_file9.max():.4f}, i.e. rounding) where re-ranking does not
(max {f9[f9.rank_source=='score'].max_abs_diff_vs_file9.max():.3f}). Re-ranking still passes the test (Spearman
{vscore['spearman']:.6f}, max difference {vscore['max_abs_diff']:.3f}). `imd2015.rank_source` in config.yaml switches between them.

## 2. Correlations (6,791 MSOA11s)

Pearson:

{md_table(pc)}
Spearman:

{md_table(sc)}
Scatter: `scatter_saie_vs_imd_income_rate.png` plots raw SAIE against the income rate:
Pearson {r_p:.3f} on the raw SAIE level, {r_plog:.3f} on log SAIE (the table above), Spearman {r_s:.3f}.

## 3. Joint-model VIFs (treatment + four interactions, interactions built in memory)

(a) = SAIE + imd_ex_housing; (b) = SAIE + imd_ex_income_housing.

{md_table(vif_wide)}
Largest VIF in any cell: {vif['vif'].max():.3f} (threshold for concern 10).
The z-score correlation matrices for each model are in `zscore_correlation_matrix_by_model.csv`.

## 4. Quartile membership: overall IMD -> imd_ex_housing

{summ.iloc[0]['share_changing_quartile']:.1%} of MSOA11s change deprivation quartile when the overall IMD score is replaced by
imd_ex_housing ({summ.iloc[0]['share_moving_2plus_quartiles']:.1%} move by two or more quartiles);
{summ.iloc[0]['of_top_before_share_still_top']:.1%} of the most-deprived quartile stays in the top quartile.
Spearman between the two measures: {summ.iloc[0]['spearman_between_measures']:.4f}.
Row shares (rows = overall IMD quartile, columns = imd_ex_housing quartile):

{md_table(prim_pct.reset_index().rename(columns={'index':''}))}
All three comparisons:

{md_table(summ, '.4f')}
Note on the third row: the previous moderator was a quartile split of a share that is zero for
{(b['imd2015_overall_bottom20_popshare_legacy']==0).mean():.1%} of MSOAs, so its quartiles were arbitrary within the zero block; treat that row as indicative only.
"""
    (QA / "IMD_REVISION_REPORT.md").write_text(report)
    log.info(f"Wrote {QA / 'IMD_REVISION_REPORT.md'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
