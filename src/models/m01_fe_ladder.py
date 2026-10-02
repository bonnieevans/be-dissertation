"""M01 - fixed-effects ladder for the baseline specification: y on x only, no controls.

Question: how does the coefficient on the treatment change as the fixed-effects structure changes,
and which structure do the specification tests support?

  M0  pooled OLS, no fixed effects
  M1  year fixed effects
  M2  MSOA fixed effects
  M3  MSOA + year fixed effects (two-way)
  M4  LAD + year fixed effects (tests whether MSOA effects add anything beyond LAD effects)
  M5  MSOA + region x year fixed effects
  M6  MSOA + LAD x year fixed effects (compares an MSOA with other MSOAs in the same LAD and year)

Estimand (Priority 1): with MSOA + year effects (M3) the coefficient uses variation across all of
England, so it can include effects that operate at the housing-market level (a LAD-wide supply
effect on every MSOA in the LAD). With LAD x year effects (M6) any shock or effect that is common to
all MSOAs of a LAD in a year is removed, so the coefficient compares MSOAs within the same LAD.
The difference between M3 and M6 shows how much of the M3 association comes from between-LAD variation.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from model_common import *  # noqa: F401,F403
import model_common as MC
from scipy import stats
import statsmodels.api as sm
import statsmodels.stats.api as sms
import esda

log = get_logger("m01_fe_ladder")
style()
T = out_dir(MODEL_DIR, "tables")
F = out_dir(MODEL_DIR, "figures")
Y, X = OUTCOME, TREATMENT

SPECS = [("M0", "pooled OLS", ""), ("M1", "year FE", "| year"), ("M2", "MSOA FE", "| msoa11cd"),
         ("M3", "MSOA + year FE", "| msoa11cd + year"), ("M4", "LAD + year FE", "| lad23cd_analysis + year"),
         ("M5", "MSOA + region x year FE", "| msoa11cd + region_year"), ("M6", "MSOA + LAD x year FE", "| msoa11cd + lad_year")]


def main() -> int:
    df = load_panel()
    d = estimation_sample(df)
    x_sd = d[X].std()
    n_msoa, n_lad, Tn = d.msoa11cd.nunique(), d.lad23cd_analysis.nunique(), d.year.nunique()
    log.info(f"Estimation sample: {len(d):,} rows, {n_msoa} MSOAs, {n_lad} LADs, {Tn} years; SD of x = {x_sd:.2f}")

    # ---- 1. The ladder ------------------------------------------------------------------------------------------
    # WHAT: regress log median price per m2 on the lagged new-build rate with progressively richer fixed effects, with
    #       no other controls. Standard errors clustered by LAD (primary) and by MSOA (comparison).
    # LOOK FOR: how the coefficient and its standard error move from M0 to M6; the sign of the pooled estimate against
    #           the within estimates; the within-R2 (share of the within-MSOA variation explained); and the difference
    #           between M3 and M6.
    # UNITS:    the coefficient on newbuilds_lag1_per_1000 is the change in log median nominal price per m2 for one
    #           additional new-build per 1,000 baseline households in the previous year; x 100 is approximately the
    #           percentage change in the median price per m2. `pct_change_per_1sd_x` rescales it to one SD of x.
    fits, rows = {}, []
    for sid, label, fe in SPECS:
        for vc in ("lad", "msoa"):
            f = fit(f"{Y} ~ {X} {fe}".strip(), d, f"{sid}_{vc}", vcov=vc, note=label)
            rows.append(coef_table(f, f"{sid}_{vc}", f"{sid}: {label} (SE clustered by {vc.upper()})", [X], x_sd).assign(model=sid, cluster=vc))
            if vc == "lad":
                fits[sid] = f
    ladder = pd.concat(rows)
    ladder.to_csv(T / "m01_fe_ladder_coefficients.csv", index=False)
    log.info("Ladder (LAD-clustered):\n" + ladder[ladder.cluster == "lad"][["model", "label", "coef", "se", "p", "n_obs", "r2_within"]].round(6).to_string())
    lad_rows = ladder[ladder.cluster == "lad"]
    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.errorbar(range(len(lad_rows)), lad_rows.coef, yerr=1.96 * lad_rows.se, fmt="o", capsize=3)
    ax.axhline(0, color="k", lw=.6)
    ax.set_xticks(range(len(lad_rows)))
    ax.set_xticklabels(lad_rows.model)
    ax.set_title("Coefficient on new-builds (lag 1) per 1,000 households, 95% CI (LAD-clustered)")
    ax.set_ylabel("log points per unit")
    savefig(fig, F / "m01_fe_ladder_coefficients.png")

    # ---- 2. Do the fixed effects matter? Nested F-tests ---------------------------------------------------------------
    # WHAT: classical F-tests (iid errors, for reference) that (a) year effects are jointly zero (M2 vs M3), (b) the MSOA
    #       effects add nothing beyond LAD effects (M4 vs M3), (c) LAD x year effects add nothing beyond year effects
    #       (M3 vs M6), (d) region x year effects add nothing beyond year effects (M3 vs M5). The cluster-robust Wald
    #       test of the year effects is also reported, because the iid F-test is over-confident under serial and
    #       spatial correlation.
    # LOOK FOR: the F statistics and p-values; whether a restriction is rejected under both the iid and the cluster-robust
    #           version.
    k = 1   # one slope coefficient
    n_year = Tn
    P = {"M2": n_msoa, "M3": n_msoa + n_year - 1, "M4": n_lad + n_year - 1, "M5": n_msoa + d.region_year.nunique() - d.region_code.nunique(),
         "M6": n_msoa + n_lad * n_year - n_lad}
    tests = [("year effects (M2 -> M3)", fits["M2"], fits["M3"], n_year - 1, P["M3"] + k),
             ("MSOA effects beyond LAD effects (M4 -> M3)", fits["M4"], fits["M3"], n_msoa - n_lad, P["M3"] + k),
             ("LAD x year effects beyond year effects (M3 -> M6)", fits["M3"], fits["M6"], P["M6"] - P["M3"], P["M6"] + k),
             ("region x year effects beyond year effects (M3 -> M5)", fits["M3"], fits["M5"], P["M5"] - P["M3"], P["M5"] + k)]
    rows = [{"test": nm, **nested_f(fr, fu, ex, kp)} for nm, fr, fu, ex, kp in tests]
    # cluster-robust joint test of the year effects: add explicit year dummies to the MSOA-FE model
    fy = fit(f"{Y} ~ {X} + i(year) | msoa11cd", d, "M2_year_dummies_wald", vcov="lad", note="cluster-robust test of year effects")
    wy = wald(fy, [c for c in fy._coefnames if "year" in c])
    rows.append({"test": "year effects, cluster-robust Wald (LAD)", "F": wy["F"], "df1": wy["q"], "df2": np.nan, "p": wy["p_F"]})
    pd.DataFrame(rows).to_csv(T / "m01_fixed_effects_f_tests.csv", index=False)
    log.info("FE F-tests:\n" + pd.DataFrame(rows).round(4).to_string())

    # ---- 3. Fixed vs random effects (Mundlak / correlated random effects) ---------------------------------------------
    # WHAT: add each MSOA's own mean of x to a model with year effects but no MSOA effects. The coefficient on the MSOA mean is
    #       the difference between the between-MSOA and within-MSOA effects of x; testing it against zero is the
    #       cluster-robust version of the Hausman test (the classical Hausman test is not valid with clustered errors).
    # LOOK FOR: whether the coefficient on the MSOA mean differs from zero. A non-zero value means the MSOA effects are
    #           correlated with x, so a random-effects estimator would not recover the within effect.
    d["x_msoa_mean"] = d.groupby("msoa11cd")[X].transform("mean")
    fm = fit(f"{Y} ~ {X} + x_msoa_mean | year", d, "mundlak_year", vcov="lad", note="Mundlak / CRE test")
    coef_table(fm, "mundlak", "Mundlak: coefficient on the MSOA mean of x tests FE vs RE", [X, "x_msoa_mean"], x_sd).to_csv(
        T / "m01_mundlak_test.csv", index=False)

    # ---- 4. Residual diagnostics for M3 and M6 --------------------------------------------------------------------------
    # WHAT: using the residuals of M3 and M6: (a) Wooldridge (2002) serial-correlation test from a first-difference regression
    #       (under no serial correlation the residual's own first-order autocorrelation is -0.5); (b) Breusch-Pagan test and
    #       a table of residual variance by sales-count decile (heteroskedasticity linked to the number of sales behind a
    #       median); (c) Pesaran CD test and neighbour correlation of the residuals; (d) Moran's I of the residuals by year;
    #       (e) share of the residual variance that is common within a LAD-year.
    # LOOK FOR: (a) an estimated autocorrelation far from -0.5; (b) residual variance falling as sale_count rises;
    #           (c)-(d) correlation of residuals across areas, larger among neighbours; (e) the share of residual variance
    #           that sits at LAD-year level, which is what the choice of clustering unit must cover.
    # (a) Wooldridge test
    d = d.sort_values(["msoa11cd", "year"])
    d["dy"] = d.groupby("msoa11cd")[Y].diff()
    d["dx"] = d.groupby("msoa11cd")[X].diff()
    fd = d.dropna(subset=["dy", "dx"]).copy()
    f_fd = fit("dy ~ dx | year", fd, "wooldridge_first_difference", vcov="msoa", note="first-difference residuals for serial correlation")
    fd["u"] = np.asarray(f_fd._u_hat)
    fd["u_lag"] = fd.groupby("msoa11cd")["u"].shift(1)
    z = fd.dropna(subset=["u_lag"])
    ols = sm.OLS(z["u"], z[["u_lag"]]).fit(cov_type="cluster", cov_kwds={"groups": z["msoa11cd"]})
    rho, se = float(ols.params["u_lag"]), float(ols.bse["u_lag"])
    wt = (rho + 0.5) / se
    pd.DataFrame({"rho_hat": [rho], "se_msoa_cluster": [se], "H0_rho": [-0.5], "t_stat": [wt], "p": [2 * (1 - stats.norm.cdf(abs(wt)))],
                  "n": [len(z)]}).to_csv(T / "m01_wooldridge_serial_correlation.csv", index=False)
    log.info(f"Wooldridge: rho_hat {rho:.3f} (H0 -0.5), t {wt:.1f}")
    # (b)-(e) on the M3 and M6 residuals
    w, _ = EC.queen_weights()
    w.transform = "r"
    ids = list(w.id_order)
    rows_h, rows_cd, rows_mo, rows_icc, vb = [], [], [], [], []
    for sid in ("M3", "M6"):
        f = fits[sid]
        e = pd.Series(np.asarray(f._u_hat), index=d.index) if len(f._u_hat) == len(d) else None
        if e is None:
            d_used = d.loc[~d.index.isin(f._na_index)] if hasattr(f, "_na_index") else d
            e = pd.Series(np.asarray(f._u_hat), index=d_used.index[:len(f._u_hat)])
        dd = d.assign(e=e).dropna(subset=["e"])
        # (b) heteroskedasticity
        exog = sm.add_constant(pd.DataFrame({"inv_sales": 1 / dd.sale_count, X: dd[X]}))
        bp = sms.het_breuschpagan(dd["e"], exog)
        rows_h.append({"model": sid, "BP_LM": bp[0], "BP_p": bp[1], "BP_F": bp[2], "BP_F_p": bp[3]})
        v = dd.assign(e2=dd.e ** 2, bin=pd.qcut(dd.sale_count, 10, duplicates="drop")).groupby("bin", observed=True).agg(
            mean_sales=("sale_count", "mean"), resid_variance=("e2", "mean"), n=("e", "size")).reset_index(drop=True)
        v.insert(0, "model", sid)
        vb.append(v)
        # (c) CD on residuals, neighbour correlation
        Pe = dd.pivot(index="msoa11cd", columns="year", values="e").reindex(ids).dropna(axis=0, how="any")
        cd = MC.__dict__.get("pt") or None
        import panel_tests as pt
        r_cd = pt.pesaran_cd(Pe.values)
        pos = {m: i for i, m in enumerate(Pe.index)}
        pi = [(pos[a], pos[b]) for a, nb in w.neighbors.items() if a in pos for b in nb if b in pos and pos[a] < pos[b]]
        Xn = Pe.values - Pe.values.mean(axis=1, keepdims=True)
        Xn = Xn / np.sqrt((Xn ** 2).sum(axis=1, keepdims=True))
        nbc = float(np.mean([Xn[i] @ Xn[j] for i, j in pi]))
        rows_cd.append({"model": sid, "CD": r_cd["CD"], "CD_p": r_cd["p_value"], "mean_pairwise_corr": r_cd["mean_pairwise_corr"],
                        "mean_abs_pairwise_corr": r_cd["mean_abs_pairwise_corr"], "mean_corr_neighbours": nbc})
        # (d) Moran's I by year
        from libpysal.weights import w_subset
        present = sorted(dd.msoa11cd.unique())
        ws = w_subset(w, present, silence_warnings=True)          # weights among the MSOAs in the estimation sample
        ws.transform = "r"
        for yr, g in dd.groupby("year"):
            s = g.set_index("msoa11cd")["e"].reindex(ws.id_order)
            mi = esda.Moran(s.values, ws, permutations=199)
            rows_mo.append({"model": sid, "year": yr, "morans_I": mi.I, "p_sim": mi.p_sim})
        # (e) variance shares
        tot = (dd.e ** 2).sum()
        for gname, gcol in (("MSOA", "msoa11cd"), ("LAD", "lad23cd_analysis"), ("LAD x year", "lad_year"), ("region x year", "region_year"), ("year", "year")):
            gm = dd.groupby(gcol)["e"].transform("mean")
            rows_icc.append({"model": sid, "grouping": gname, "share_of_residual_variance_in_group_means": float((gm ** 2).sum() / tot)})
    pd.DataFrame(rows_h).to_csv(T / "m01_heteroskedasticity_bp.csv", index=False)
    pd.concat(vb).to_csv(T / "m01_residual_variance_by_sales_decile.csv", index=False)
    pd.DataFrame(rows_cd).to_csv(T / "m01_residual_cross_sectional_dependence.csv", index=False)
    pd.DataFrame(rows_mo).to_csv(T / "m01_residual_morans_I_by_year.csv", index=False)
    pd.DataFrame(rows_icc).to_csv(T / "m01_residual_variance_by_group.csv", index=False)
    log.info("Residual CD:\n" + pd.DataFrame(rows_cd).round(4).to_string())
    log.info("Residual variance in group means:\n" + pd.DataFrame(rows_icc).round(4).to_string())

    # ---- 5. Sensitivity of the baseline coefficient to sample and to how the treatment is measured -------------------
    # WHAT: re-estimate M3 and M6 on the full panel (2012-2023), with alternative treatment timing (same-year, lag 2, lag 3,
    #       previous-three-year total), a log(1+x) transform, and with the treatment winsorised at the 99th percentile or
    #       trimmed above the 99.9th.
    # LOOK FOR: whether the sign and magnitude of the coefficient are stable across these definitions; the influence of
    #           the extreme construction rates (listed in outputs/eda/tables/01_treatment_top25_outliers.csv).
    # UNITS:    for the log(1+x) version the coefficient is the change in log price for a one-unit change in log(1+x), i.e.
    #           roughly the elasticity of price with respect to new-build intensity at larger values of x.
    rows = []
    full = estimation_sample(df, full=True)
    variants = {"main sample, lag 1 (baseline)": (d, X), "full panel 2012-2023, lag 1": (full, X),
                "same-year construction": (d, "newbuilds_per_1000"), "lag 2": (d, "newbuilds_lag2_per_1000"),
                "lag 3": (d, "newbuilds_lag3_per_1000"), "previous 3 years total": (d, "newbuilds_prev3yr_per_1000")}
    d["x_log1p"] = np.log1p(d[X])
    d["x_wins99"] = d[X].clip(upper=d[X].quantile(.99))
    variants["log(1 + x)"] = (d, "x_log1p")
    variants["winsorised at p99"] = (d, "x_wins99")
    variants["trimmed above p99.9"] = (d[d[X] <= d[X].quantile(.999)], X)
    for nm, (dat, xv) in variants.items():
        for sid, fe in (("M3", "| msoa11cd + year"), ("M6", "| msoa11cd + lad_year")):
            f = fit(f"{Y} ~ {xv} {fe}", dat, f"{sid}_robust_{nm}", note=nm, sample=nm)
            rows.append(coef_table(f, f"{sid}_{nm}", nm, [xv], dat[xv].std()).assign(model=sid, treatment_variable=xv))
    pd.concat(rows).to_csv(T / "m01_treatment_definition_sensitivity.csv", index=False)
    MC.save_registry("m01")
    log.info("M01 complete.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
