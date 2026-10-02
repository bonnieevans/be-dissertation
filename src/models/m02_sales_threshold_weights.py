"""M02 - transaction counts: minimum-sales thresholds and weighting.

Why: the outcome is a median over `sale_count` sales. Cells with few sales are noisy, and residual
variance falls with sale_count (see M01 heteroskedasticity output). Options are to drop thin cells,
to weight cells by transaction volume, or both. Thin cells are concentrated in deprived, high-social-rent,
dense areas (EDA 01), so a restriction can change who is in the heterogeneity sample; that composition is
checked here.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from model_common import *  # noqa: F401,F403
import model_common as MC

log = get_logger("m02_sales_threshold_weights")
style()
T = out_dir(MODEL_DIR, "tables")
F = out_dir(MODEL_DIR, "figures")
Y, X = OUTCOME, TREATMENT


def main() -> int:
    df = load_panel()
    d = estimation_sample(df)
    x_sd = d[X].std()

    # ---- 1. Baseline coefficient under thresholds and weights -------------------------------------------------------
    # WHAT: M3 (MSOA + year FE) and M6 (MSOA + LAD x year FE) re-estimated (a) with no restriction, (b) dropping MSOA-years
    #       with fewer than 10, 20, 30 or 50 sales, (c) weighting by sale_count (analytic weights), (d) both: threshold 30
    #       plus weights. SEs clustered by LAD.
    # LOOK FOR: whether the coefficient and its standard error move with the restriction or the weights; the number of
    #           observations lost; how the within-R2 changes.
    # UNITS:    as in M01: change in log median nominal price per m2 per additional new-build per 1,000 households (lag 1).
    variants = [("no restriction", 0, None), ("sales >= 10", 10, None), ("sales >= 20", 20, None), ("sales >= 30", 30, None),
                ("sales >= 50", 50, None), ("weighted by sale_count", 0, "sale_count"),
                ("sales >= 30 and weighted", 30, "sale_count")]
    rows = []
    for nm, k, w in variants:
        dd = d[d.sale_count >= k]
        n = dd.groupby("lad23cd_analysis")["msoa11cd"].transform("nunique")
        dd = dd[n >= 2]
        for sid, fe in (("M3", "| msoa11cd + year"), ("M6", "| msoa11cd + lad_year")):
            f = fit(f"{Y} ~ {X} {fe}", dd, f"{sid}_sales_{nm}", weights=w, note=nm, sample=nm)
            rows.append(coef_table(f, f"{sid}_{nm}", nm, [X], x_sd).assign(model=sid, threshold=k, weights=w or "none", n_dropped=len(d) - len(dd)))
    res = pd.concat(rows)
    res.to_csv(T / "m02_sales_threshold_weights_coefficients.csv", index=False)
    log.info("Thresholds and weights:\n" + res[["model", "label", "coef", "se", "p", "n_obs", "r2_within"]].round(6).to_string())

    # ---- 2. Composition of the sample after each restriction -----------------------------------------------------
    # WHAT: for each restriction, the share of MSOA-years retained in each quartile of each moderator, and (for the
    #       weighted version) the share of total weight in each quartile; plus the mean and SD of each moderator z-score
    #       among retained observations and the SD of the treatment within quartile.
    # LOOK FOR: quartiles that lose a much larger share of observations than others (the restriction removes part of
    #           the support of the heterogeneity analysis); shifts in the mean z-score; whether the within-quartile
    #           variation of the treatment is preserved.
    qcols = {"income": "income_quartile", "deprivation": "deprivation_quartile",
             "social_rent": "social_rent_quartile", "density": "density_quartile"}
    rows, rows2 = [], []
    for nm, k, w in variants:
        dd = d[d.sale_count >= k]
        wt = dd[w] if w else pd.Series(1.0, index=dd.index)
        for m, qc in qcols.items():
            tot_all = d.groupby(qc).size()
            kept = dd.groupby(qc).size() / tot_all
            wshare = wt.groupby(dd[qc]).sum() / wt.sum()
            sd_x = dd.groupby(qc)[X].std()
            for q in sorted(tot_all.index):
                rows.append({"restriction": nm, "moderator": m, "quartile": q, "share_of_cells_retained": kept[q],
                             "share_of_total_weight": wshare[q], "sd_treatment_in_quartile": sd_x[q]})
        for m in MODERATORS:
            z = dd[Z_COL[m]]
            mu = np.average(z, weights=wt)
            rows2.append({"restriction": nm, "moderator": m, "weighted_mean_z": mu,
                          "weighted_sd_z": np.sqrt(np.average((z - mu) ** 2, weights=wt))})
    comp = pd.DataFrame(rows)
    comp.to_csv(T / "m02_sample_composition_by_quartile.csv", index=False)
    pd.DataFrame(rows2).to_csv(T / "m02_sample_moderator_moments.csv", index=False)
    fig, axes = plt.subplots(1, 4, figsize=(17, 3.8), sharey=True)
    for ax, (m, qc) in zip(axes, qcols.items()):
        s = comp[(comp.moderator == m) & comp.restriction.isin(["sales >= 10", "sales >= 20", "sales >= 30", "sales >= 50"])]
        for r, g in s.groupby("restriction"):
            ax.plot(g.quartile, g.share_of_cells_retained, marker="o", label=r)
        ax.set_title(f"Share retained by {m} quartile")
        ax.set_xticks([1, 2, 3, 4])
    axes[0].legend(fontsize=7)
    savefig(fig, F / "m02_share_retained_by_quartile.png")

    # ---- 3. Residual variance against sales count (to justify the weighting) -------------------------------------------
    # WHAT: fit M3 and regress the squared residual on 1 / sale_count.
    # LOOK FOR: whether the variance is proportional to 1/sale_count (the pattern analytic weights by sale_count assume);
    #           the slope and the R2 of this regression.
    f = fit(f"{Y} ~ {X} | msoa11cd + year", d, "M3_for_variance", note="variance model")
    e2 = np.asarray(f._u_hat) ** 2
    A = np.column_stack([np.ones(len(d)), 1 / d.sale_count.values])
    beta = np.linalg.lstsq(A, e2, rcond=None)[0]
    r2 = 1 - ((e2 - A @ beta) ** 2).sum() / ((e2 - e2.mean()) ** 2).sum()
    pd.DataFrame({"intercept": [beta[0]], "slope_on_inverse_sales": [beta[1]], "r2": [r2]}).to_csv(T / "m02_variance_vs_inverse_sales.csv", index=False)
    MC.save_registry("m02")
    log.info("M02 complete.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
