"""M04 - endogeneity checks: leads and lags, reverse direction, pre-period balance.

Concern: new-builds are not randomly located or timed. If developers build where prices are about to rise (or
completions follow earlier price rises), the coefficient on construction picks up the reverse direction. Planning
permission precedes completion by one to three years, so future completions may also reflect anticipation of
construction. The checks below describe the data; they cannot separate these explanations on their own.

Data note: the 2023 new-build total is incomplete (EPC lodgement and UPRN matching lag), so completions dated 2023
are under-counted. Lead terms that reach into 2023 therefore have a separate sample (leads using only completions
up to 2022).
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from model_common import *  # noqa: F401,F403
import model_common as MC
from scipy import stats

log = get_logger("m04_endogeneity_pretrends")
style()
T = out_dir(MODEL_DIR, "tables")
F = out_dir(MODEL_DIR, "figures")
Y, X = OUTCOME, TREATMENT
FE = {"M3": "| msoa11cd + year", "M6": "| msoa11cd + lad_year"}


def lincom(f, weights: dict[str, float]) -> tuple[float, float, float]:
    names = list(f._coefnames)
    g = np.zeros(len(names))
    for k, v in weights.items():
        g[names.index(k)] = v
    b, V = np.asarray(f._beta_hat), np.asarray(f._vcov)
    est, se = float(g @ b), float(np.sqrt(g @ V @ g))
    return est, se, 2 * (1 - stats.norm.cdf(abs(est / se)))


def main() -> int:
    df = load_panel()
    # leads / lags of the SAME-year construction rate N_t = newbuilds_per_1000, built in memory
    df = df.rename(columns={"newbuilds_per_1000": "N_t", "nb_lead1": "N_lead1", "nb_lead2": "N_lead2", "nb_lag1": "N_lag1",
                            "newbuilds_lag2_per_1000": "N_lag2", "newbuilds_lag3_per_1000": "N_lag3"})
    base_all = estimation_sample(df, full=True)           # full years 2012-2023 so lags/leads exist
    x_sd = df["N_t"].std()

    # ---- 1. Leads and lags of construction --------------------------------------------------------------------------
    # WHAT: regress log price in year t on construction in t+2, t+1, t (same year), t-1, t-2 and t-3, with M3 and M6
    #       fixed effects. Three samples: (A) two leads, years 2016-2020 (no lead reaches 2023); (B) one lead, years
    #       2016-2021 (lead reaches 2022 at most); (C) one lead with the lead reaching 2023 (years 2016-2022), shown to
    #       illustrate the effect of the incomplete 2023 count. The joint test is that all lead coefficients are zero
    #       (LAD-clustered Wald).
    # LOOK FOR: the sign, size and significance of the lead coefficients against those of the lags; the joint
    #           lead test; the cumulative lag effect (sum of coefficients on t to t-3); how M3 and M6 differ; and
    #           whether results differ when the lead reaches into 2023.
    # UNITS:    each coefficient is the change in log median nominal price per m2 in year t for one additional new-build
    #           per 1,000 households completed in the stated year, holding the other leads and lags fixed.
    specs = {"A_two_leads_2016_2020": (["N_lead2", "N_lead1", "N_t", "N_lag1", "N_lag2", "N_lag3"], 2016, 2020, ["N_lead2", "N_lead1"]),
             "B_one_lead_2016_2021": (["N_lead1", "N_t", "N_lag1", "N_lag2", "N_lag3"], 2016, 2021, ["N_lead1"]),
             "C_one_lead_reaching_2023_2016_2022": (["N_lead1", "N_t", "N_lag1", "N_lag2", "N_lag3"], 2016, 2022, ["N_lead1"])}
    rows, jt, cum = [], [], []
    es_store = {}
    for sname, (regs, y0, y1, leads) in specs.items():
        d = base_all[(base_all.year >= y0) & (base_all.year <= y1)].dropna(subset=regs)
        for fe_id, fe in FE.items():
            f = fit(f"{Y} ~ {' + '.join(regs)} {fe}", d, f"{fe_id}_eventstudy_{sname}", note=sname, sample=sname)
            rows.append(coef_table(f, f"{fe_id}_{sname}", sname, regs, x_sd).assign(fe=fe_id, sample=sname))
            wj = wald(f, leads)
            jt.append({"fe": fe_id, "sample": sname, "leads_tested": "+".join(leads), **wj, "n_obs": f._N})
            lags = [r for r in regs if r in ("N_t", "N_lag1", "N_lag2", "N_lag3")]
            est, se, p = lincom(f, {r: 1.0 for r in lags})
            cum.append({"fe": fe_id, "sample": sname, "sum_of_coefficients_t_to_t-3": est, "se": se, "p": p})
            es_store[(sname, fe_id)] = f
    pd.concat(rows, ignore_index=True).to_csv(T / "m04_leads_lags_coefficients.csv", index=False)
    pd.DataFrame(jt).to_csv(T / "m04_leads_joint_tests.csv", index=False)
    pd.DataFrame(cum).to_csv(T / "m04_cumulative_lag_effect.csv", index=False)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    order = ["N_lead2", "N_lead1", "N_t", "N_lag1", "N_lag2", "N_lag3"]
    xpos = {"N_lead2": 2, "N_lead1": 1, "N_t": 0, "N_lag1": -1, "N_lag2": -2, "N_lag3": -3}
    for ax, fe_id in zip(axes, FE):
        f = es_store[("A_two_leads_2016_2020", fe_id)]
        t = f.tidy()
        for r in order:
            ax.errorbar(-xpos[r], t.loc[r, "Estimate"], yerr=1.96 * t.loc[r, "Std. Error"], fmt="o", color="C0", capsize=3)
        ax.axhline(0, color="k", lw=.6)
        ax.set_xticks([-2, -1, 0, 1, 2, 3])
        ax.set_xticklabels(["t+2", "t+1", "t", "t-1", "t-2", "t-3"])
        ax.set_title(f"{fe_id}: construction in year (rel. to price year t), 95% CI")
    savefig(fig, F / "m04_leads_lags_event_study.png")

    # ---- 2. Reverse direction (Granger-type) -----------------------------------------------------------------------------
    # WHAT: regress construction in year t on annual log price growth in t-1 and t-2, with M3 / M6 fixed effects (LAD-
    #       clustered), with and without last year's construction. With MSOA effects and a lagged dependent variable the
    #       estimates carry a Nickell bias of order 1/T (T = 7-8 here), so the version without the lag is the cleaner one.
    # LOOK FOR: whether past price growth is associated with later construction (the feedback the lead test cannot isolate)
    #           and its size relative to the within standard deviation of construction.
    # UNITS:    the coefficient is the change in new-builds per 1,000 households in year t associated with a one-unit
    #           (100 log points) higher price growth in the stated earlier year; multiply by 0.01 for a one-percentage-point
    #           higher growth rate.
    d = base_all[(base_all.year >= 2016)].copy()
    d["dlp_l1"] = d.groupby("msoa11cd")["d_log_price"].shift(1)
    d["dlp_l2"] = d.groupby("msoa11cd")["d_log_price"].shift(2)
    d = d.dropna(subset=["dlp_l1", "dlp_l2", "N_lag1"])
    rows = []
    for fe_id, fe in FE.items():
        for lab, rhs in (("no lagged construction", "dlp_l1 + dlp_l2"), ("with lagged construction", "dlp_l1 + dlp_l2 + N_lag1")):
            f = fit(f"N_t ~ {rhs} {fe}", d, f"{fe_id}_reverse_{lab}", note="reverse direction", sample="2016-2023")
            rows.append(coef_table(f, f"{fe_id}_reverse_{lab}", lab, ["dlp_l1", "dlp_l2"]).assign(fe=fe_id, spec=lab))
    pd.concat(rows, ignore_index=True).to_csv(T / "m04_reverse_direction.csv", index=False)

    # ---- 3. Pre-period balance and a pre-trend event-study ------------------------------------------------------------
    # WHAT: (a) correlation and regression (with region fixed effects, LAD-clustered) of cumulative construction in
    #       2016-2023 on price growth in the earlier period 2012-2015, which is before the main sample; (b) mean 2012-2015 price
    #       growth by quartile of cumulative construction, with moderator quartile breakdown; (c) an event-study: log price
    #       on year dummies interacted with an indicator for the top quartile of cumulative construction (against the rest),
    #       with MSOA fixed effects and (i) year effects or (ii) LAD x year effects, relative to 2015.
    # LOOK FOR: (a)-(b) whether areas that later build more already had faster or slower price growth before 2016;
    #           (c) whether the coefficients for 2012-2015 are close to zero and trending, or already departing from zero;
    #           differences between (i) and (ii).
    # UNITS:    (a) the slope is the change in log(1 + cumulative new-builds per 1,000) per 1 log-point higher 2012-2015
    #           price growth; (c) each coefficient is the difference in log median nominal price per m2 between top-quartile
    #           and other MSOAs in that year, relative to 2015.
    P = df.pivot(index="msoa11cd", columns="year", values=Y)
    pre = (P[2015] - P[2012]).rename("pre_growth_2012_2015")
    cum = df[df.main_sample].groupby("msoa11cd")["N_t"].sum().rename("cum_construction_2016_2023")
    cs = pd.concat([pre, cum], axis=1).join(df.drop_duplicates("msoa11cd").set_index("msoa11cd")[
        ["lad23cd_analysis", "region_code", "income_quartile", "deprivation_quartile", "social_rent_quartile", "density_quartile"]]).reset_index()
    cs["log1p_cum"] = np.log1p(cs.cum_construction_2016_2023)
    cs["cum_quartile"] = pd.qcut(cs.cum_construction_2016_2023, 4, labels=[1, 2, 3, 4])
    corr = pd.DataFrame({"pearson": [cs.pre_growth_2012_2015.corr(cs.cum_construction_2016_2023)],
                         "spearman": [cs.pre_growth_2012_2015.corr(cs.cum_construction_2016_2023, method="spearman")],
                         "pearson_with_log1p": [cs.pre_growth_2012_2015.corr(cs.log1p_cum)]})
    corr.to_csv(T / "m04_pre_period_correlation.csv", index=False)
    rows = []
    for lab, rhs in (("region FE only", "| region_code"), ("no FE", "")):
        f = fit(f"log1p_cum ~ pre_growth_2012_2015 {rhs}".strip(), cs, f"pre_period_balance_{lab}", note=lab, sample="cross-section")
        rows.append(coef_table(f, lab, lab, ["pre_growth_2012_2015"]))
    pd.concat(rows, ignore_index=True).to_csv(T / "m04_pre_period_regression.csv", index=False)
    cs.groupby("cum_quartile", observed=True).agg(n=("msoa11cd", "size"), mean_pre_growth=("pre_growth_2012_2015", "mean"),
                                                  sd_pre_growth=("pre_growth_2012_2015", "std"),
                                                  mean_cum_construction=("cum_construction_2016_2023", "mean")).to_csv(
        T / "m04_pre_growth_by_construction_quartile.csv")
    for m, qc in (("income", "income_quartile"), ("deprivation", "deprivation_quartile"),
                  ("social_rent", "social_rent_quartile"), ("density", "density_quartile")):
        cs.groupby([qc, "cum_quartile"], observed=True)["pre_growth_2012_2015"].mean().unstack().to_csv(T / f"m04_pre_growth_by_{m}_and_construction_quartile.csv")
    # (c) event study on the full 2012-2023 panel
    e = base_all.merge(cs[["msoa11cd", "cum_quartile"]], on="msoa11cd")
    e["top_q"] = (e.cum_quartile == 4).astype(int)
    rows = []
    fig, ax = plt.subplots(figsize=(7.5, 4.3))
    for fe_id, fe in FE.items():
        f = fit(f"{Y} ~ i(year, top_q, ref=2015) {fe}", e, f"{fe_id}_pretrend_event_study", note="top-quartile construction x year", sample="2012-2023")
        t = coef_table(f, f"{fe_id}_event", "top-quartile cumulative construction x year (ref 2015)")
        t["year"] = t["term"].str.extract(r"year::(\d+)")[0].astype(int)
        rows.append(t.assign(fe=fe_id))
        tt = t.sort_values("year")
        ax.errorbar(tt.year + (0.1 if fe_id == "M6" else 0), tt.coef, yerr=1.96 * tt.se, fmt="o-", capsize=2, label=fe_id)
        wj = wald(f, [c for c in f._coefnames if int(c.split("::")[1].split(":")[0]) in (2012, 2013, 2014)])
        pd.DataFrame([{"fe": fe_id, "pre_years_tested": "2012-2014", **wj}]).to_csv(T / f"m04_pretrend_event_joint_{fe_id}.csv", index=False)
    ax.axhline(0, color="k", lw=.6)
    ax.axvline(2015.5, color="k", lw=.6, ls="--")
    ax.set_title("Log price: top-quartile construction MSOAs vs others, by year (ref. 2015)")
    ax.legend()
    savefig(fig, F / "m04_pretrend_event_study.png")
    pd.concat(rows, ignore_index=True).to_csv(T / "m04_pretrend_event_study_coefficients.csv", index=False)
    MC.save_registry("m04")
    log.info("M04 complete.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
