"""EDA 01 - data structure, quality, distributions, sales-count profile.

Why this matters for the panel regressions
- Balance / missingness: fixed-effects estimators drop incomplete groups; unbalanced
  panels change what is identified.
- Within vs between variance: with MSOA fixed effects only WITHIN-MSOA variation in the
  treatment identifies the coefficient, and the time-invariant baseline moderators are
  absorbed (only their interactions with the treatment are identified).
- Distributions / outliers: OLS-type estimators are sensitive to heavy tails and
  high-leverage points; skewness guides log transforms and winsorising.
- Sales counts: the outcome is a median over a small number of sales per MSOA-year, so
  noise (and hence heteroskedasticity) depends on sale_count.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import eda_common as C
from eda_common import *  # noqa: F401,F403
from scipy import stats

log = get_logger("eda_01_data_structure")
style()
T = out_dir(EDA_DIR, "tables")
F = out_dir(EDA_DIR, "figures", "distributions")


def main() -> int:
    df = load_panel()
    base_cols = df.drop_duplicates("msoa11cd")
    num = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c]) and df[c].dtype != bool]

    # ---- 1. Balance and missingness -------------------------------------------------
    # WHAT: confirm one row per MSOA-year, balanced years, and list every column with missing values.
    # LOOK FOR: any MSOA with fewer than 12 years; which variables have missing values and
    #           whether the missingness is concentrated in particular years or areas.
    per_msoa = df.groupby("msoa11cd").size()
    bal = pd.DataFrame({"n_rows": [len(df)], "n_msoa": [df.msoa11cd.nunique()], "n_years": [df.year.nunique()],
                        "n_lad": [df.lad23cd_analysis.nunique()], "n_region": [df.region_code.nunique()],
                        "min_years_per_msoa": [per_msoa.min()], "max_years_per_msoa": [per_msoa.max()],
                        "duplicate_keys": [int(df.duplicated(["msoa11cd", "year"]).sum())]})
    bal.to_csv(T / "01_panel_balance.csv", index=False)
    miss = df.isna().sum().rename("n_missing").to_frame()
    miss["share_missing"] = miss["n_missing"] / len(df)
    miss[miss.n_missing > 0].to_csv(T / "01_missingness.csv")
    log.info(f"Balance: {bal.iloc[0].to_dict()}")

    # ---- 2. Within / between variance decomposition ----------------------------------
    # WHAT: split each variable's variance into between-MSOA (differences in MSOA means) and
    #       within-MSOA (movement over time around the MSOA's own mean). Done on the main
    #       sample (2016-2023), the estimation window, and on the full panel.
    # LOOK FOR: the within-MSOA SD of the treatment and of the outcome (this is the variation
    #           the fixed-effects estimator actually uses); variables with zero within-variation
    #           are the baseline moderators (absorbed by MSOA fixed effects).
    rows = []
    vd_vars = [OUTCOME, "d_log_price", TREATMENT, "newbuilds_per_1000", "newbuild_total", "sale_count",
               "median_ppsqm"] + list(MODERATORS.values())
    for label, d in (("full_2012_2023", df), ("main_2016_2023", df[df.main_sample])):
        for v in vd_vars:
            s = d[v].dropna()
            g = d.loc[s.index].groupby("msoa11cd")[v]
            within = (d.loc[s.index, v] - g.transform("mean"))
            rows.append({"sample": label, "variable": v, "overall_sd": s.std(), "between_sd": g.mean().std(),
                         "within_sd": within.std(), "within_share_of_variance": within.var() / s.var()})
    pd.DataFrame(rows).to_csv(T / "01_within_between_variance.csv", index=False)

    # ---- 3. Distribution summary ----------------------------------------------------
    # WHAT: moments, quantiles, share of zeros, and Jarque-Bera normality for every numeric variable.
    # LOOK FOR: strong right skew (candidates for log transform), heavy tails (kurtosis), zero
    #           inflation in the treatment. With ~80,000 rows the JB test rejects normality for almost
    #           any variable, so read the skewness/kurtosis magnitudes and not the p-value.
    rows = []
    for c in num:
        s = df[c].dropna()
        if s.nunique() < 2:
            continue
        jb = stats.jarque_bera(s) if len(s) > 7 else (np.nan, np.nan)
        rows.append({"variable": c, "n": len(s), "mean": s.mean(), "sd": s.std(), "min": s.min(),
                     "p1": s.quantile(.01), "p25": s.quantile(.25), "median": s.median(), "p75": s.quantile(.75),
                     "p99": s.quantile(.99), "max": s.max(), "skew": stats.skew(s), "excess_kurtosis": stats.kurtosis(s),
                     "share_zero": (s == 0).mean(), "jb_stat": jb[0], "jb_p": jb[1]})
    dist = pd.DataFrame(rows)
    dist.to_csv(T / "01_distribution_summary.csv", index=False)

    # ---- 3b. Histograms of every numeric variable (pages of 6 x 5) ----------------------
    plot_vars = [c for c in num if df[c].nunique() > 2 and c not in ("year",)]
    per_page = 30
    for i in range(0, len(plot_vars), per_page):
        chunk = plot_vars[i:i + per_page]
        fig, axes = plt.subplots(6, 5, figsize=(17, 15))
        for ax, c in zip(axes.ravel(), chunk):
            s = df[c].dropna()
            lo, hi = s.quantile(.001), s.quantile(.999)
            ax.hist(s.clip(lo, hi), bins=50, color="#3b6ea5")
            ax.set_title(c, fontsize=8)
            ax.tick_params(labelsize=7)
        for ax in axes.ravel()[len(chunk):]:
            ax.axis("off")
        fig.suptitle("Distributions (clipped at the 0.1th / 99.9th percentile for display)", y=1.0)
        fig.tight_layout()
        savefig(fig, F / f"hist_all_variables_p{i // per_page + 1}.png")

    # ---- 3c. QQ plots and log transformation of key variables ---------------------------
    # WHAT: compare the shape of the outcome/treatment in levels and logs against the normal.
    # LOOK FOR: whether the log transform removes the right skew (outcome) and how heavy the
    #           upper tail of the treatment remains after the transform.
    key = {"median_ppsqm": df["median_ppsqm"], "log_median_ppsqm": df[OUTCOME],
           "newbuilds_per_1000": df["newbuilds_per_1000"], "log1p(newbuilds_per_1000)": np.log1p(df["newbuilds_per_1000"]),
           "sale_count": df["sale_count"], "log(sale_count)": df["log_sale_count"]}
    fig, axes = plt.subplots(2, 3, figsize=(13, 8))
    for ax, (k, s) in zip(axes.ravel(), key.items()):
        stats.probplot(s.dropna().sample(min(20000, s.notna().sum()), random_state=1), dist="norm", plot=ax)
        ax.set_title(f"QQ: {k}  (skew {stats.skew(s.dropna()):.2f})")
    fig.tight_layout()
    savefig(fig, F / "qq_key_variables.png")

    # ---- 3d. Zero inflation of the treatment by year -------------------------------------
    zero = df.groupby("year").agg(share_zero_newbuilds=("newbuild_total", lambda s: (s == 0).mean()),
                                  mean_newbuilds_per_1000=("newbuilds_per_1000", "mean"),
                                  median_newbuilds_per_1000=("newbuilds_per_1000", "median"),
                                  p99_newbuilds_per_1000=("newbuilds_per_1000", lambda s: s.quantile(.99)),
                                  total_newbuilds=("newbuild_total", "sum"))
    zero.to_csv(T / "01_treatment_zero_share_by_year.csv")

    # ---- 4. Outliers in the treatment ---------------------------------------------------
    # WHAT: list the largest values of new-builds per 1,000 households and the baseline household
    #       counts behind them (small denominators inflate the rate).
    # LOOK FOR: whether extreme rates sit in MSOAs with very small baseline_households_2011 or a single
    #           very large development; these are high-leverage points for a continuous treatment.
    top = df.nlargest(25, "newbuilds_per_1000")[["msoa11cd", "msoa11nm", "year", "newbuild_total", "baseline_households_2011",
                                                 "newbuilds_per_1000", "sale_count"]]
    top.to_csv(T / "01_treatment_top25_outliers.csv", index=False)
    q = df["newbuilds_per_1000"].quantile([.9, .95, .99, .999, 1.0])
    pd.DataFrame({"quantile": q.index, "newbuilds_per_1000": q.values}).to_csv(T / "01_treatment_upper_tail.csv", index=False)
    hh = base_cols["baseline_households_2011"]
    pd.DataFrame({"stat": ["min", "p1", "p5", "median", "p95", "max"],
                  "baseline_households_2011": [hh.min(), hh.quantile(.01), hh.quantile(.05), hh.median(), hh.quantile(.95), hh.max()]}
                 ).to_csv(T / "01_baseline_households_distribution.csv", index=False)

    # ---- 5. Transaction counts by MSOA-year -----------------------------------------------
    # WHAT: distribution of sale_count (the number of sales behind each median).
    # LOOK FOR: how many MSOA-years fall under each threshold and in which years; a thick left tail
    #           means noisier medians and a case for weighting or a minimum-sales rule.
    sc = df["sale_count"]
    qs = sc.quantile([.01, .05, .1, .25, .5, .75, .9, .99])
    summ = pd.DataFrame({"stat": ["mean", "median", "sd", "min", "max"] + [f"p{int(p * 100)}" for p in qs.index],
                         "sale_count": [sc.mean(), sc.median(), sc.std(), sc.min(), sc.max()] + list(qs.values)})
    summ.to_csv(T / "01_sale_count_distribution.csv", index=False)
    thr = []
    for k in (10, 20, 30, 50):
        under = df["sale_count"] < k
        thr.append({"threshold": k, "obs_below": int(under.sum()), "share_obs_below": under.mean(),
                    "msoas_with_any_year_below": df.loc[under, "msoa11cd"].nunique(),
                    "share_below_main_sample": under[df.main_sample].mean()})
    pd.DataFrame(thr).to_csv(T / "01_sale_count_threshold_exposure.csv", index=False)
    by_year = df.groupby("year")["sale_count"].agg(["mean", "median", "min", lambda s: s.quantile(.05)])
    by_year.columns = ["mean", "median", "min", "p5"]
    by_year.to_csv(T / "01_sale_count_by_year.csv")
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    ax[0].hist(sc.clip(upper=sc.quantile(.995)), bins=60, color="#3b6ea5")
    ax[0].set_title("Sales per MSOA-year")
    for k in (10, 20, 30, 50):
        ax[0].axvline(k, color="k", lw=.6, ls="--")
    by_year.plot(ax=ax[1])
    ax[1].set_title("Sales per MSOA-year by year")
    savefig(fig, EDA_DIR / "figures" / "sale_count_profile.png")

    # ---- 6. Are low-transaction areas different? -----------------------------------------
    # WHAT: compare the baseline characteristics of MSOA-years under each sales threshold with the rest,
    #       as standardised mean differences, and report the share of each moderator quartile that a
    #       threshold would remove.
    # LOOK FOR: a standardised difference that is not near zero for deprivation, social-rent share, income
    #           or density, and whether the share removed differs across moderator quartiles; either would
    #           change who remains in the heterogeneity sample.
    rows, qrows = [], []
    chars = {"deprivation": "deprivation_moderator_value", "social_rent_share": "social_rent_share_2011",
             "log_income": "log_baseline_income", "log_density": "log_population_density_2011",
             "degree_share": "degree_share_2011", "unemployment": "unemployment_rate_2011",
             "baseline_households": "baseline_households_2011"}
    for k in (10, 20, 30, 50):
        low = df["sale_count"] < k
        if low.sum() < 5:
            continue
        for name, col in chars.items():
            a, b = df.loc[low, col], df.loc[~low, col]
            pooled = np.sqrt((a.var() + b.var()) / 2)
            rows.append({"threshold": k, "characteristic": name, "mean_low": a.mean(), "mean_rest": b.mean(),
                         "std_mean_diff": (a.mean() - b.mean()) / pooled, "n_low": int(low.sum())})
        for mod, qcol in (("deprivation", "deprivation_quartile"), ("income", "income_quartile"),
                          ("social_rent", "social_rent_quartile"), ("density", "density_quartile")):
            t = df.assign(low=low).groupby(qcol)["low"].mean()
            for qv, sh in t.items():
                qrows.append({"threshold": k, "moderator": mod, "quartile": qv, "share_obs_below_threshold": sh})
    pd.DataFrame(rows).to_csv(T / "01_low_sales_characteristics.csv", index=False)
    pd.DataFrame(qrows).to_csv(T / "01_low_sales_share_by_moderator_quartile.csv", index=False)
    msoa_mean = df.groupby("msoa11cd")["sale_count"].mean().rename("mean_sale_count").reset_index().merge(
        base_cols[["msoa11cd"] + list(chars.values())], on="msoa11cd")
    msoa_mean["decile_of_mean_sales"] = pd.qcut(msoa_mean["mean_sale_count"], 10, labels=False) + 1
    msoa_mean.groupby("decile_of_mean_sales")[["mean_sale_count"] + list(chars.values())].mean().to_csv(
        T / "01_characteristics_by_decile_of_mean_sales.csv")
    log.info("EDA 01 complete.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
