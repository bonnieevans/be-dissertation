"""EDA 03 - trends over time, common shocks, persistence, maps.

Why this matters in an econometric context
- A strong common time pattern (national house-price cycle) is absorbed by year fixed
  effects, but if the pattern differs by region or LAD, year effects are not enough and
  region x year or LAD x year effects are needed.
- Divergence of price paths across moderator quartiles before the treatment period is the
  first visual check for differential pre-trends.
- Persistence (autocorrelation) of the treatment and of price growth drives serial
  correlation in the errors and hence the standard errors.
- The outcome in the panel is the log of the median NOMINAL price per m2; the trend plots are
  therefore nominal.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from eda_common import *  # noqa: F401,F403

log = get_logger("eda_03_trends")
style()
T = out_dir(EDA_DIR, "tables", "trends")
F = out_dir(EDA_DIR, "figures", "trends")
QMAP = {"income": "income_quartile", "deprivation": "deprivation_quartile",
        "social_rent": "social_rent_quartile", "density": "density_quartile"}
QLAB = {"income": "1 = poorest", "deprivation": "4 = most deprived", "social_rent": "4 = highest share",
        "density": "4 = densest"}


def acf_within(df: pd.DataFrame, col: str, lags: int = 5) -> pd.DataFrame:
    """Within-MSOA autocorrelation: demean by MSOA, then correlate with own lags."""
    x = df[col] - df.groupby("msoa11cd")[col].transform("mean")
    out = []
    for k in range(1, lags + 1):
        lag = x.groupby(df["msoa11cd"]).shift(k)
        m = x.notna() & lag.notna()
        out.append({"variable": col, "lag": k, "within_autocorrelation": np.corrcoef(x[m], lag[m])[0, 1]})
    return pd.DataFrame(out)


def r2_of_group_year_means(df: pd.DataFrame, col: str, group: str) -> float:
    """Share of within-MSOA variance of `col` explained by group x year means (balanced panel,
    MSOA nested in group, so MSOA and group-year effects are orthogonal)."""
    x = df[col] - df.groupby("msoa11cd")[col].transform("mean")
    ok = x.notna()
    x = x[ok]
    gm = x.groupby([df.loc[ok, group], df.loc[ok, "year"]]).transform("mean")
    return float(1 - ((x - gm) ** 2).sum() / (x ** 2).sum())


def main() -> int:
    df = load_panel()
    base = load_baseline()

    # ---- 1. National series ----------------------------------------------------------
    # WHAT: cross-MSOA median, mean and 10th-90th percentile band of the outcome and of the treatment,
    #       total sales and total new-builds, by year.
    # LOOK FOR: the shape of the common cycle; whether the cross-sectional spread widens or narrows over
    #           time; breaks in 2020-2021 and the low 2023 new-build total (the latest completions are
    #           the least complete).
    g = df.groupby("year")
    nat = pd.DataFrame({"median_log_price": g[OUTCOME].median(), "mean_log_price": g[OUTCOME].mean(),
                        "p10_log_price": g[OUTCOME].quantile(.1), "p90_log_price": g[OUTCOME].quantile(.9),
                        "sd_log_price": g[OUTCOME].std(), "median_ppsqm": g["median_ppsqm"].median(),
                        "total_sales": g["sale_count"].sum(), "total_newbuilds": g["newbuild_total"].sum(),
                        "mean_newbuilds_per_1000": g["newbuilds_per_1000"].mean(),
                        "median_newbuilds_per_1000": g["newbuilds_per_1000"].median(),
                        "share_msoa_any_newbuild": g["newbuild_total"].apply(lambda s: (s > 0).mean()),
                        "top10pct_share_of_newbuilds": g["newbuild_total"].apply(
                            lambda s: s.nlargest(int(len(s) * .1)).sum() / s.sum())})
    nat.to_csv(T / "03_national_series.csv")
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    ax = axes[0, 0]
    ax.plot(nat.index, nat.median_log_price, marker="o")
    ax.fill_between(nat.index, nat.p10_log_price, nat.p90_log_price, alpha=.2)
    ax.set_title("Log median nominal price/m2: MSOA median and p10-p90")
    nat["sd_log_price"].plot(ax=axes[0, 1], marker="o", title="Cross-MSOA sd of log price (sigma divergence/convergence)")
    nat["total_sales"].plot(ax=axes[0, 2], marker="o", title="Total sales in the price sample")
    nat[["total_newbuilds"]].plot(ax=axes[1, 0], marker="o", title="Total new-builds (all England)")
    nat[["mean_newbuilds_per_1000", "median_newbuilds_per_1000"]].plot(ax=axes[1, 1], marker="o", title="New-builds per 1,000 households")
    nat[["share_msoa_any_newbuild", "top10pct_share_of_newbuilds"]].plot(ax=axes[1, 2], marker="o", title="Spread of construction")
    fig.tight_layout()
    savefig(fig, F / "national_series.png")

    # ---- 2. Paths by moderator quartile --------------------------------------------------
    # WHAT: mean log price indexed to 2016 (=0), and mean construction intensity, by quartile of each
    #       moderator; also the 2012-2015 paths (before the main sample).
    # LOOK FOR: whether paths fan out or converge before 2016 (visual pre-trend check); differences in the
    #           level and timing of construction across quartiles.
    rows = []
    fig, axes = plt.subplots(3, 4, figsize=(19, 11))
    for j, (mod, qcol) in enumerate(QMAP.items()):
        t = df.groupby([qcol, "year"]).agg(mean_log_price=(OUTCOME, "mean"), mean_nb=("newbuilds_per_1000", "mean"),
                                           median_nb=("newbuilds_per_1000", "median"),
                                           mean_growth=("d_log_price", "mean")).reset_index()
        ref = t[t.year == 2016].set_index(qcol)["mean_log_price"]
        t["log_price_index_2016"] = t["mean_log_price"] - t[qcol].map(ref)
        t.insert(0, "moderator", mod)
        rows.append(t.rename(columns={qcol: "quartile"}))
        for q, s in t.groupby(qcol):
            axes[0, j].plot(s.year, s.log_price_index_2016, marker=".", label=f"Q{int(q)}")
            axes[1, j].plot(s.year, s.mean_nb, marker=".", label=f"Q{int(q)}")
            axes[2, j].plot(s.year, s.mean_growth, marker=".", label=f"Q{int(q)}")
        axes[0, j].set_title(f"{mod} ({QLAB[mod]}): log price, 2016 = 0")
        axes[1, j].set_title(f"{mod}: mean new-builds per 1,000")
        axes[2, j].set_title(f"{mod}: mean annual log price growth")
        axes[0, j].axvline(2016, color="k", lw=.5, ls="--")
        axes[0, j].legend(fontsize=7)
    fig.tight_layout()
    savefig(fig, F / "paths_by_moderator_quartile.png")
    pd.concat(rows).to_csv(T / "03_paths_by_moderator_quartile.csv", index=False)

    # ---- 3. Paths by region ------------------------------------------------------------------
    # WHAT: price index (2016 = 0), price growth and construction intensity by region.
    # LOOK FOR: regional differences in timing and size of the price cycle (these are what region x year or
    #           LAD x year fixed effects would absorb).
    r = df.groupby(["region_code", "year"]).agg(mean_log_price=(OUTCOME, "mean"), mean_nb=("newbuilds_per_1000", "mean"),
                                                mean_growth=("d_log_price", "mean")).reset_index()
    r["log_price_index_2016"] = r["mean_log_price"] - r.groupby("region_code")["mean_log_price"].transform(
        lambda s: s.loc[r.loc[s.index, "year"] == 2016].iloc[0])
    r.to_csv(T / "03_paths_by_region.csv", index=False)
    fig, axes = plt.subplots(1, 3, figsize=(17, 4.8))
    for rc, s in r.groupby("region_code"):
        axes[0].plot(s.year, s.log_price_index_2016, label=rc[-2:])
        axes[1].plot(s.year, s.mean_growth, label=rc[-2:])
        axes[2].plot(s.year, s.mean_nb, label=rc[-2:])
    axes[0].set_title("Log price by region, 2016 = 0 (regions E12000001-09)")
    axes[1].set_title("Mean annual log price growth by region")
    axes[2].set_title("Mean new-builds per 1,000 by region")
    axes[0].legend(fontsize=6, ncol=3)
    savefig(fig, F / "paths_by_region.png")

    # ---- 4. How much of the within-MSOA variation is common to a year, region-year, LAD-year? -----------
    # WHAT: R-squared of the within-MSOA variation of the outcome and of annual price growth from year
    #       means, region x year means and LAD x year means (nested groups, balanced panel).
    # LOOK FOR: how much each layer of common shocks adds; the remainder is the idiosyncratic MSOA-level
    #           variation left for the treatment to explain. The LAD x year number is an upper bound on
    #           the variation absorbed (it also removes sampling noise within each LAD-year).
    df["all"] = "ENG"
    rows = []
    for col in (OUTCOME, "d_log_price", "newbuilds_per_1000"):
        for grp, lab in (("all", "year"), ("region_code", "region x year"), ("lad23cd_analysis", "LAD x year")):
            rows.append({"variable": col, "common_component": lab,
                         "share_of_within_msoa_variance": r2_of_group_year_means(df, col, grp)})
    pd.DataFrame(rows).to_csv(T / "03_common_shock_variance_shares.csv", index=False)
    log.info("Common shock shares:\n" + pd.DataFrame(rows).round(3).to_string())

    # ---- 5. Persistence -------------------------------------------------------------------------------
    # WHAT: within-MSOA autocorrelation (lags 1-5) of the outcome level, price growth, and the treatment,
    #       and the share of zero-construction years that follow a zero year.
    # LOOK FOR: high first-order autocorrelation (serial correlation in errors; clustering needed) and the
    #           persistence of construction (leads and lags of the treatment are correlated).
    ac = pd.concat([acf_within(df, c) for c in (OUTCOME, "d_log_price", "newbuilds_per_1000")])
    ac.to_csv(T / "03_within_autocorrelation.csv", index=False)
    fig, ax = plt.subplots(figsize=(7, 4))
    for c, s in ac.groupby("variable"):
        ax.plot(s.lag, s.within_autocorrelation, marker="o", label=c)
    ax.axhline(0, color="k", lw=.5)
    ax.set_title("Within-MSOA autocorrelation")
    ax.set_xlabel("lag (years)")
    ax.legend()
    savefig(fig, F / "within_autocorrelation.png")
    z = df.assign(z=df.newbuild_total == 0)
    z["z_lag"] = z.groupby("msoa11cd")["z"].shift(1)
    pd.DataFrame({"P(zero | zero last year)": [z.loc[z.z_lag == True, "z"].mean()],
                  "P(zero | positive last year)": [z.loc[z.z_lag == False, "z"].mean()]}).to_csv(T / "03_zero_persistence.csv", index=False)

    # ---- 6. Convergence ---------------------------------------------------------------------------------
    # WHAT: regress each MSOA's 2016-2023 log price growth on its 2016 log price (beta convergence).
    # LOOK FOR: the sign of the slope; if initial price level predicts later growth, the price level
    #           carries information about trends (relevant to fixed effects vs. trend controls).
    a = df[df.year == 2016].set_index("msoa11cd")[OUTCOME]
    b = df[df.year == 2023].set_index("msoa11cd")[OUTCOME]
    cv = pd.DataFrame({"p2016": a, "growth_2016_2023": b - a})
    slope, icpt = np.polyfit(cv.p2016, cv.growth_2016_2023, 1)
    pd.DataFrame({"slope": [slope], "intercept": [icpt], "corr": [cv.p2016.corr(cv.growth_2016_2023)]}).to_csv(
        T / "03_beta_convergence.csv", index=False)
    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.scatter(cv.p2016, cv.growth_2016_2023, s=3, alpha=.3)
    ax.plot(cv.p2016.sort_values(), icpt + slope * cv.p2016.sort_values(), color="r")
    ax.set_xlabel("log median price/m2, 2016")
    ax.set_ylabel("log growth 2016-2023")
    ax.set_title(f"Beta convergence (slope {slope:.3f})")
    savefig(fig, F / "beta_convergence.png")

    # ---- 7. MSOA x year heatmaps ---------------------------------------------------------------------------
    # WHAT: every MSOA as a row, sorted by its mean, years as columns.
    # LOOK FOR: whether paths are parallel (similar colour progression across rows) and the pattern of
    #           construction (clusters of consecutive building years, long runs of zeros).
    piv = df.pivot(index="msoa11cd", columns="year", values=OUTCOME)
    piv = piv.loc[piv.mean(axis=1).sort_values().index]
    pn = df.pivot(index="msoa11cd", columns="year", values="newbuilds_per_1000")
    pn = pn.loc[pn.mean(axis=1).sort_values().index]
    fig, axes = plt.subplots(1, 2, figsize=(13, 8))
    im = axes[0].imshow(piv.values, aspect="auto", cmap="viridis")
    axes[0].set_title("log price: MSOAs sorted by mean (rows)")
    axes[0].set_xticks(range(12))
    axes[0].set_xticklabels(piv.columns, rotation=60)
    fig.colorbar(im, ax=axes[0], shrink=.6)
    im = axes[1].imshow(np.log1p(pn.values), aspect="auto", cmap="magma")
    axes[1].set_title("log(1 + new-builds per 1,000): MSOAs sorted by mean")
    axes[1].set_xticks(range(12))
    axes[1].set_xticklabels(pn.columns, rotation=60)
    fig.colorbar(im, ax=axes[1], shrink=.6)
    savefig(fig, F / "heatmap_msoa_by_year.png")

    # ---- 8. Maps ----------------------------------------------------------------------------------------------
    # WHAT: choropleths of the moderators, the mean outcome, price growth, and cumulative construction.
    # LOOK FOR: geographic clustering (neighbouring areas looking alike) and regional blocks, which is
    #           the visual counterpart of spatial autocorrelation.
    gdf = msoa_gdf().merge(base[["msoa11cd"] + list(MODERATORS.values())], on="msoa11cd", how="left")
    avg = df[df.main_sample].groupby("msoa11cd").agg(mean_log_price=(OUTCOME, "mean"),
                                                     cum_newbuilds_per_1000=("newbuilds_per_1000", "sum")).reset_index()
    gr = (b - a).rename("log_price_growth_2016_2023").reset_index()
    gdf = gdf.merge(avg, on="msoa11cd").merge(gr, on="msoa11cd")
    maps = {**{k: v for k, v in MODERATORS.items()}, "mean_log_price": "mean_log_price",
            "growth_2016_2023": "log_price_growth_2016_2023", "cum_newbuilds_per_1000": "cum_newbuilds_per_1000"}
    fig, axes = plt.subplots(3, 3, figsize=(17, 20))
    for ax, (nm, col) in zip(axes.ravel(), maps.items()):
        s = gdf[col]
        gdf.plot(column=col, ax=ax, cmap="viridis", linewidth=0, legend=True, vmin=s.quantile(.02), vmax=s.quantile(.98),
                 legend_kwds={"shrink": .5})
        ax.set_title(nm)
        ax.set_axis_off()
    fig.tight_layout()
    savefig(fig, F / "maps_moderators_outcome_treatment.png", dpi=90)
    log.info("EDA 03 complete.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
