"""EDA 10 - the spatial spillover exposure variables (distributions, maps, correlations, zero-neighbour and island cases).

Reads the persisted variables from the final panel (built by src/13b_prepare_spatial_spillovers.py and merged by src/14_merge_final_panel.py).
Exposures are new builds per 1,000 households, lag 1, unless stated. Completions in 2022-2023 are incomplete (many EPCs lack a UPRN), so
time averages below use the main window 2016-2022.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from eda_common import *  # noqa: F401,F403
from report_utils import md_table
import seaborn as sns

log = get_logger("eda_10_spillover_exposure")
style()
T = out_dir(EDA_DIR, "tables", "spillover")
F = out_dir(EDA_DIR, "figures", "spatial")
Q, K5, EXP, K10, MEAN = ("nbr_queen_nb_lag1_rate_per1000", "nbr_5km_nb_lag1_rate_per1000", "nbr_exp10km_nb_lag1_rate_per1000",
                         "nbr_10km_nb_lag1_rate_per1000", "nbr_queen_nb_mean_rate_lag1")
LAB = {Q: "queen pooled", MEAN: "queen mean of rates", K5: "5 km pooled", K10: "10 km pooled", EXP: "distance-weighted (3 km decay)"}


def main() -> int:
    df = load_panel()
    base = df.drop_duplicates("msoa11cd").set_index("msoa11cd")
    win = df[(df.year >= 2016) & (df.year <= 2022)]
    msoa_avg = win.groupby("msoa11cd")[[TREATMENT, Q, MEAN, K5, K10, EXP]].mean()

    # ---- 1. neighbour-count distributions ----
    # WHAT: number of neighbours per MSOA under each definition. LOOK FOR: very few or very many neighbours (the pooled rates are less stable with few).
    fig, ax = plt.subplots(1, 3, figsize=(15, 4))
    for a, (c, t) in zip(ax, [("nbr_queen_n", "Queen contiguity"), ("nbr_5km_n", "Centroids within 5 km"), ("nbr_10km_n", "Centroids within 10 km")]):
        a.hist(base[c], bins=40, color="#3b6ea5")
        a.set_title(f"{t}: {int((base[c] == 0).sum())} MSOAs with none")
        a.set_xlabel("number of neighbouring MSOAs")
    savefig(fig, F / "spillover_neighbour_count_distributions.png")
    cnt = pd.DataFrame({c: base[c].describe() for c in ("nbr_queen_n", "nbr_5km_n", "nbr_10km_n")}).T
    cnt.to_csv(T / "10_neighbour_counts.csv")

    # ---- 2. maps of neighbouring construction intensity ----
    # WHAT: mean 2016-2022 pooled neighbouring construction (queen and 5 km) per 1,000 neighbouring households, and the own rate for comparison.
    # LOOK FOR: whether high-construction areas cluster (neighbouring intensity resembles own intensity) and where the 5 km measure is undefined.
    gdf = msoa_gdf().merge(msoa_avg.reset_index(), on="msoa11cd", how="left")
    fig, axes = plt.subplots(1, 3, figsize=(20, 8))
    for ax, (c, t) in zip(axes, [(TREATMENT, "Own construction"), (Q, "Neighbouring construction (queen, pooled)"), (K5, "Neighbouring construction (5 km, pooled)")]):
        v = gdf[c]
        gdf.plot(column=c, ax=ax, cmap="viridis", linewidth=0, legend=True, vmin=0, vmax=v.quantile(.98), legend_kwds={"shrink": .45},
                 missing_kwds={"color": "lightgrey", "label": "undefined"})
        ax.set_title(f"{t}\nnew builds per 1,000 households, mean 2016-2022", fontsize=10)
        ax.set_axis_off()
    savefig(fig, F / "maps_neighbouring_construction.png", dpi=85)

    # ---- 3. own vs neighbouring construction ----
    # WHAT: correlation between an MSOA's lagged construction rate and the exposure measures: pooled, between MSOAs (2016-2022 means), and two-way within
    #       (MSOA and year means removed, the variation a fixed-effects model uses).
    # LOOK FOR: the within correlation (the collinearity that matters in regression) against the pooled and between correlations.
    d = win.dropna(subset=[Q, K5, EXP, MEAN, K10, TREATMENT]).copy()
    cols = [TREATMENT, Q, MEAN, K5, K10, EXP]
    dm = d[cols] - d.groupby("msoa11cd")[cols].transform("mean")
    tw = dm - dm.groupby(d.year.values).transform("mean")
    corr = pd.DataFrame({"pooled": d[cols].corr()[TREATMENT], "between_msoa_means": msoa_avg[cols].corr()[TREATMENT], "two_way_within": tw.corr()[TREATMENT]}).drop(index=TREATMENT)
    corr.index = [LAB[i] for i in corr.index]
    corr.to_csv(T / "10_own_vs_neighbouring_correlation.csv")
    allc = pd.concat([tw.add_suffix("__tw")], axis=1).corr()
    allc.index = allc.columns = [LAB.get(c.replace("__tw", ""), "own") for c in allc.columns]
    allc.to_csv(T / "10_two_way_within_correlations_all_measures.csv")

    # ---- 4. queen vs distance-based exposures ----
    # WHAT: scatter of the queen pooled rate against the 5 km pooled and decay-weighted rates (MSOA means 2016-2022), and the correlation matrix.
    # LOOK FOR: how closely the definitions agree (the 5 km pool averages over many more MSOAs in cities).
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))
    for ax, c in zip(axes, [K5, K10, EXP]):
        ax.hexbin(msoa_avg[Q], msoa_avg[c], gridsize=45, mincnt=1, cmap="Blues", bins="log")
        ax.plot([0, 60], [0, 60], "r", lw=.7)
        ax.set_xlim(0, 60); ax.set_ylim(0, 60)
        ax.set_title(f"Queen pooled vs {LAB[c]}\nr = {msoa_avg[Q].corr(msoa_avg[c]):.2f}", fontsize=10)
        ax.set_xlabel("queen pooled rate"); ax.set_ylabel(LAB[c])
    savefig(fig, F / "spillover_queen_vs_distance_exposures.png")
    msoa_avg[cols].corr().to_csv(T / "10_between_correlations_exposure_measures.csv")

    # ---- 5. neighbouring construction by baseline income quartile ----
    # WHAT: mean pooled neighbouring construction (queen and 5 km) by quartile of the focal MSOA's baseline income and by year, and neighbouring income against own.
    # LOOK FOR: whether richer areas are surrounded by more construction (spillover exposure is not random with respect to income).
    q = df.groupby(["income_quartile", "year"])[[TREATMENT, Q, K5]].mean().reset_index()
    q.to_csv(T / "10_exposure_by_income_quartile_year.csv", index=False)
    tab = win.groupby("income_quartile")[[TREATMENT, Q, K5, EXP]].mean()
    tab.to_csv(T / "10_exposure_by_income_quartile_2016_2022.csv")
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6))
    for qv, g in q.groupby("income_quartile"):
        axes[0].plot(g.year, g[Q], marker="o", label=f"Q{int(qv)}")
        axes[1].plot(g.year, g[TREATMENT], marker="o", label=f"Q{int(qv)}")
    axes[0].set_title("Neighbouring construction (queen pooled), by focal income quartile (Q1 = poorest)")
    axes[1].set_title("Own construction, by income quartile")
    axes[0].legend()
    for a in axes:
        a.axvspan(2021.5, 2023.5, color="grey", alpha=.15)
    savefig(fig, F / "spillover_exposure_by_income_quartile.png")
    inc = pd.DataFrame({"corr_own_income_vs_neighbour_income_queen": [base.log_baseline_income.corr(base.nbr_queen_log_income_hhmean)],
                        "corr_own_income_vs_neighbour_income_5km": [base.log_baseline_income.corr(base.nbr_5km_log_income_hhmean)],
                        "sd_income_gap_queen": [base.income_gap_own_minus_nbr_queen_log.std()],
                        "share_own_income_above_neighbours_queen": [(base.income_gap_own_minus_nbr_queen_log > 0).mean()]})
    inc.to_csv(T / "10_neighbour_income_summary.csv", index=False)

    # ---- 6. zero-neighbour and island cases ----
    # WHAT: MSOAs with no centroid within 5 km or 10 km (their distance exposures are undefined, NaN, not zero), the island fallback, and the share of
    #       MSOAs with no cross-LAD queen neighbour.
    # LOOK FOR: that undefined cases are rural/remote and concentrated in a few regions, so dropping them changes the sample composition.
    z = base[base.nbr_5km_n == 0]
    zr = base.assign(none5=(base.nbr_5km_n == 0), none10=(base.nbr_10km_n == 0)).groupby("region_code").agg(n_msoa=("msoa11cd" if "msoa11cd" in base else "nbr_queen_n", "size"),
                                                                                                          share_none_5km=("none5", "mean"), share_none_10km=("none10", "mean"))
    zr.to_csv(T / "10_zero_neighbour_by_region.csv")
    prof = pd.DataFrame({"zero_5km": z[["population_density_2011", "log_baseline_income", "imd_ex_housing", "social_rent_share_2011"]].mean(),
                         "all_msoa": base[["population_density_2011", "log_baseline_income", "imd_ex_housing", "social_rent_share_2011"]].mean()})
    prof.to_csv(T / "10_zero_5km_neighbour_profile.csv")
    isl = base[base.nbr_queen_has_island_fallback][["msoa11nm", "lad23cd_analysis", "nbr_queen_n"]]
    isl.to_csv(T / "10_island_fallback_msoas.csv")
    gdf2 = msoa_gdf().merge(base.reset_index()[["msoa11cd", "nbr_5km_n", "nbr_10km_n", "nbr_queen_has_island_fallback"]], on="msoa11cd")
    fig, ax = plt.subplots(figsize=(7, 8))
    gdf2.plot(color="#dddddd", ax=ax, linewidth=0)
    gdf2[gdf2.nbr_5km_n == 0].plot(color="#d95f02", ax=ax, linewidth=0)
    gdf2[gdf2.nbr_10km_n == 0].plot(color="#7a0177", ax=ax, linewidth=0)
    gdf2[gdf2.nbr_queen_has_island_fallback].plot(color="black", ax=ax, linewidth=0)
    ax.set_title("No centroid within 5 km (orange), within 10 km (purple); island fallback (black)", fontsize=10)
    ax.set_axis_off()
    savefig(fig, F / "spillover_zero_neighbour_and_island_cases.png")

    # ---- report ----
    ann = pd.read_csv(PROJECT_ROOT / "outputs" / "qa" / "spatial_spillovers" / "annual_exposure_summary.csv")
    L = ["# Spatial spillover exposure: exploratory report\n",
         "Generated by `src/eda/eda_10_spillover_exposure.py` from the variables built by `src/13b_prepare_spatial_spillovers.py`. Exposures are new builds per 1,000 households (lag 1) in "
         "neighbouring MSOAs unless stated. Completions in 2022-2023 are incomplete (many EPCs lack a UPRN), so time averages use 2016-2022.\n",
         "## Neighbour counts\n", md_table(cnt.reset_index().rename(columns={"index": "network"}), nd=2), "",
         "## Own vs neighbouring construction (correlation of the own rate with each exposure)\n", md_table(corr.reset_index().rename(columns={"index": "exposure"}), nd=3), "",
         "## Mean exposure by focal baseline income quartile, 2016-2022\n", md_table(tab.reset_index(), nd=2), "",
         "## Neighbouring and own income\n", md_table(inc, nd=3), "",
         "## Zero-neighbour cases by region\n", md_table(zr.reset_index(), nd=3), "",
         "Profile of MSOAs with no centroid within 5 km:\n", md_table(prof.reset_index().rename(columns={"index": "variable"}), nd=3), "",
         "Island fallback (nearest-neighbour edge, not genuine contiguity):\n", md_table(isl.reset_index(), nd=3), "",
         "## Annual summary\n", md_table(ann[["year", "nbr_queen_nb_total_mean", "nbr_queen_nb_total_median", "queen_nb_total_zero_share", "nbr_queen_nb_lag1_rate_per1000_mean",
                                              "nbr_5km_nb_lag1_rate_per1000_mean", "own_vs_queen_nb_lag1_rate_corr", "upstream_note"]], nd=3), "",
         "Figures: `outputs/eda/figures/spatial/spillover_*.png`, `maps_neighbouring_construction.png`.\n"]
    (EDA_DIR / "SPILLOVER_EXPOSURE_REPORT.md").write_text("\n".join(L))
    log.info("EDA 10 complete.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
