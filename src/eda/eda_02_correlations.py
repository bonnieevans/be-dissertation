"""EDA 02 - correlations among all variables, at several levels.

Why this matters in an econometric context
- A pooled correlation mixes three things: differences between places (between), movement
  over time within a place (within), and the common time pattern. A fixed-effects regression
  uses only the within part (after year effects, the two-way within part), so the
  correlation that matters for identification is the two-way-within correlation, which can
  differ in size and sign from the pooled one.
- Correlations among the moderators determine multicollinearity in the interaction terms.
- Spearman vs Pearson differences flag non-linearity / outlier influence.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from eda_common import *  # noqa: F401,F403
import seaborn as sns

log = get_logger("eda_02_correlations")
style()
T = out_dir(EDA_DIR, "tables", "correlations")
F = out_dir(EDA_DIR, "figures", "correlations")

TV = [OUTCOME, "mean_log_ppsqm", "d_log_price", "newbuilds_per_1000", "newbuilds_lag1_per_1000",
      "newbuilds_lag2_per_1000", "newbuilds_lag3_per_1000", "newbuilds_prev3yr_per_1000", "nb_lead1", "nb_lead2",
      "log_sale_count", "median_floor_area", "detached_sale_share", "semidetached_sale_share",
      "terraced_sale_share", "flat_sale_share", "leasehold_sale_share"]


def heat(mat: pd.DataFrame, title: str, path: Path, annot: bool = True, size: float = 0.42) -> None:
    n = len(mat)
    fig, ax = plt.subplots(figsize=(max(6, n * size + 2), max(5, n * size + 1)))
    sns.heatmap(mat, cmap="RdBu_r", vmin=-1, vmax=1, center=0, annot=annot and n <= 24, fmt=".2f",
                annot_kws={"size": 7}, square=True, cbar_kws={"shrink": .6}, ax=ax)
    ax.set_title(title, fontsize=10)
    plt.setp(ax.get_xticklabels(), rotation=60, ha="right", fontsize=7)
    plt.setp(ax.get_yticklabels(), fontsize=7)
    savefig(fig, path)


def main() -> int:
    df = load_panel()
    base = load_baseline()
    ms = df[df.main_sample].copy()

    # ---- 1. Pooled correlations of every numeric variable (CSV) --------------------------
    # WHAT: Pearson and Spearman correlations of all numeric panel columns, pooled over all rows.
    # LOOK FOR: pairs that are mechanically related (a count and its per-1,000 rate, a z-score and
    #           its raw variable, quartile indicators); these are listed in the file but are not
    #           informative about relationships between distinct concepts.
    num = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c]) and df[c].dtype != bool and df[c].nunique() > 2
           and c not in ("year",)]
    df[num].corr().to_csv(T / "pooled_pearson_all_numeric.csv")
    df[num].corr(method="spearman").to_csv(T / "pooled_spearman_all_numeric.csv")
    pc = df[num].corr()
    pairs = pc.where(np.triu(np.ones(pc.shape, bool), 1)).stack().rename("pearson").reset_index()
    pairs.columns = ["var1", "var2", "pearson"]
    pairs["abs"] = pairs.pearson.abs()
    pairs.sort_values("abs", ascending=False).drop(columns="abs").to_csv(T / "pooled_pearson_pairs_ranked.csv", index=False)

    # ---- 2. Time-varying variables: pooled, between, within, two-way within -------------------
    # WHAT: correlations among the time-varying variables on the main sample (2016-2023), computed
    #       (a) pooled, (b) between MSOAs (MSOA means), (c) within MSOAs (deviations from MSOA mean),
    #       (d) two-way within (deviations from MSOA and year means) - the variation a two-way fixed
    #       effects regression uses.
    # LOOK FOR: how the correlation between the treatment and the outcome changes from (a) to (d),
    #           including its sign; compare the lagged and lead versions of the treatment in (d).
    d = ms[TV].dropna()
    ids = ms.loc[d.index, ["msoa11cd", "year"]]
    out = {}
    out["pooled"] = d.corr()
    out["between"] = d.groupby(ids.msoa11cd).mean().corr()
    w = d - d.groupby(ids.msoa11cd).transform("mean")
    out["within_msoa"] = w.corr()
    tw = w - w.groupby(ids.year).transform("mean")
    out["two_way_within"] = tw.corr()
    for k, m in out.items():
        m.to_csv(T / f"timevarying_{k}_pearson.csv")
        heat(m, f"Time-varying variables, {k.replace('_', ' ')} correlation (main sample 2016-2023)", F / f"timevarying_{k}.png")
    key = pd.DataFrame({k: m[OUTCOME] for k, m in out.items()}).loc[
        ["newbuilds_per_1000", "newbuilds_lag1_per_1000", "newbuilds_lag2_per_1000", "newbuilds_lag3_per_1000",
         "newbuilds_prev3yr_per_1000", "nb_lead1", "nb_lead2", "d_log_price"]]
    key.to_csv(T / "treatment_outcome_correlation_by_level.csv")
    log.info("Treatment-outcome correlation by level:\n" + key.round(3).to_string())

    # ---- 3. Treatment-outcome correlation within each year (cross-section) ------------------------
    # WHAT: year-by-year cross-sectional correlation between the lagged treatment and the outcome level
    #       and growth.
    # LOOK FOR: whether the relationship is stable across years or changes in sign/size.
    rows = []
    for yr, g in df.groupby("year"):
        g = g.dropna(subset=[TREATMENT])
        rows.append({"year": yr, "corr_with_log_price": g[TREATMENT].corr(g[OUTCOME]),
                     "spearman_with_log_price": g[TREATMENT].corr(g[OUTCOME], method="spearman"),
                     "corr_with_price_growth": g[TREATMENT].corr(g["d_log_price"])})
    yr = pd.DataFrame(rows)
    yr.to_csv(T / "treatment_outcome_correlation_by_year.csv", index=False)
    fig, ax = plt.subplots(figsize=(7, 4))
    yr.set_index("year")[["corr_with_log_price", "spearman_with_log_price", "corr_with_price_growth"]].plot(marker="o", ax=ax)
    ax.axhline(0, color="k", lw=.6)
    ax.set_title("Cross-sectional correlation of new-build intensity (lag 1) with price, by year")
    savefig(fig, F / "treatment_outcome_correlation_by_year.png")

    # ---- 4. Baseline characteristics (one row per MSOA) ----------------------------------------------
    # WHAT: Pearson and Spearman correlations among the six moderators, then among all baseline
    #       variables including the IMD domain scores and tenure shares.
    # LOOK FOR: moderator pairs with |r| close to 0.8 or above (collinear interactions);
    #           Pearson-Spearman gaps (non-linear or outlier-driven association); whether the
    #           deprivation index is closer to the income or the unemployment variable.
    mod_cols = {k: v for k, v in MODERATORS.items()}
    mm = base.rename(columns={v: k for k, v in mod_cols.items()})[list(mod_cols)]
    mm.corr().to_csv(T / "moderators_pearson.csv")
    mm.corr(method="spearman").to_csv(T / "moderators_spearman.csv")
    heat(mm.corr(), "Moderators: Pearson (MSOA level, n=6,791)", F / "moderators_pearson.png")
    heat(mm.corr(method="spearman"), "Moderators: Spearman (MSOA level, n=6,791)", F / "moderators_spearman.png")
    # partial correlations from the precision matrix of the standardised moderators
    prec = np.linalg.inv(mm.corr().values)
    d_ = np.sqrt(np.diag(prec))
    pcor = pd.DataFrame(-prec / np.outer(d_, d_), index=mm.columns, columns=mm.columns)
    np.fill_diagonal(pcor.values, 1.0)
    pcor.to_csv(T / "moderators_partial_correlations.csv")
    heat(pcor, "Moderators: partial correlations (each pair, holding the other four constant)", F / "moderators_partial.png")

    bcols = [c for c in base.columns if pd.api.types.is_numeric_dtype(base[c]) and base[c].nunique() > 4
             and not c.endswith("_z") and "population" not in c and c not in ("region_code",)]
    base[bcols].corr().to_csv(T / "baseline_all_pearson.csv")
    base[bcols].corr(method="spearman").to_csv(T / "baseline_all_spearman.csv")
    cg = sns.clustermap(base[bcols].corr(method="spearman"), cmap="RdBu_r", vmin=-1, vmax=1, center=0,
                        figsize=(max(9, len(bcols) * .38), max(9, len(bcols) * .38)),
                        xticklabels=True, yticklabels=True)
    cg.ax_heatmap.tick_params(labelsize=6)
    cg.fig.suptitle("Baseline characteristics, Spearman, clustered", y=1.01)
    savefig(cg.fig, F / "baseline_all_spearman_clustered.png")

    # ---- 5. Moderators vs. average outcome and treatment (MSOA level) ------------------------
    # WHAT: scatter matrix of the moderators with each MSOA's mean log price and mean new-build intensity
    #       over the main sample; and binned scatters of the outcome on the treatment (raw and two-way
    #       demeaned) overall and by moderator quartile.
    # LOOK FOR: curvature, clusters and outliers not visible in a correlation coefficient; whether the
    #           slope of the demeaned binned scatter differs across the moderator quartiles.
    avg = ms.groupby("msoa11cd")[[OUTCOME, "newbuilds_per_1000"]].mean()
    sm = mm.copy()
    sm["mean_log_price"] = avg.reindex(base["msoa11cd"])[OUTCOME].values
    sm["mean_newbuilds_per_1000"] = avg.reindex(base["msoa11cd"])["newbuilds_per_1000"].values
    g = sns.pairplot(sm, corner=True, plot_kws={"s": 3, "alpha": .3}, diag_kws={"bins": 40}, height=1.6)
    g.fig.suptitle("Moderators with MSOA mean log price and new-build intensity (2016-2023 means)", y=1.01)
    savefig(g.fig, F / "scatter_matrix_moderators_means.png", dpi=90)

    dd = ms.dropna(subset=[TREATMENT, OUTCOME]).copy()
    for nm, col in (("raw", None), ("two_way_within", "tw")):
        x, y = dd[TREATMENT], dd[OUTCOME]
        if col:
            def tw_demean(s):
                s1 = s - s.groupby(dd.msoa11cd).transform("mean")
                return s1 - s1.groupby(dd.year).transform("mean")
            x, y = tw_demean(x), tw_demean(y)
        dd[f"x_{nm}"], dd[f"y_{nm}"] = x, y
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for ax, nm in zip(axes, ("raw", "two_way_within")):
        b = dd.assign(bin=pd.qcut(dd[f"x_{nm}"], 40, duplicates="drop")).groupby("bin", observed=True)[[f"x_{nm}", f"y_{nm}"]].mean()
        ax.scatter(b.iloc[:, 0], b.iloc[:, 1], s=18)
        ax.set_title(f"Binned scatter ({nm.replace('_', '-')}): log price vs new-build intensity (lag 1)")
        ax.set_xlabel(TREATMENT)
        ax.set_ylabel(OUTCOME)
    savefig(fig, F / "binned_scatter_outcome_treatment.png")
    qmaps = {"income": "income_quartile", "deprivation": "deprivation_quartile",
             "social_rent": "social_rent_quartile", "density": "density_quartile"}
    fig, axes = plt.subplots(1, 4, figsize=(17, 4), sharey=False)
    rows = []
    for ax, (mname, qcol) in zip(axes, qmaps.items()):
        for qv in sorted(dd[qcol].dropna().unique()):
            s = dd[dd[qcol] == qv]
            b = s.assign(bin=pd.qcut(s["x_two_way_within"], 20, duplicates="drop")).groupby("bin", observed=True)[
                ["x_two_way_within", "y_two_way_within"]].mean()
            ax.plot(b.iloc[:, 0], b.iloc[:, 1], marker="o", ms=3, label=f"Q{int(qv)}")
            slope = np.polyfit(s["x_two_way_within"], s["y_two_way_within"], 1)[0]
            rows.append({"moderator": mname, "quartile": qv, "n": len(s), "two_way_within_slope_no_controls": slope})
        ax.set_title(f"By {mname} quartile (two-way within)")
        ax.legend(fontsize=7)
    savefig(fig, F / "binned_scatter_by_moderator_quartile.png")
    pd.DataFrame(rows).to_csv(T / "two_way_within_slope_by_moderator_quartile.csv", index=False)
    log.info("EDA 02 complete.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
