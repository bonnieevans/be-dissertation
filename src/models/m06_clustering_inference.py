"""M06 - standard errors: clustering level and wild-cluster bootstrap.

Why: the M01 residual diagnostics show serial correlation within MSOAs and cross-sectional dependence among
neighbouring MSOAs. Conventional (iid) and MSOA-clustered standard errors assume independence across MSOAs, which
the spatial correlation contradicts. Clustering by LAD allows arbitrary correlation among all MSOAs of a LAD (and
over time). With 296 LAD clusters, asymptotic cluster-robust inference is reasonable, but a wild-cluster bootstrap is
run as a check. Region (9 clusters) is reported only to show why it is too coarse to rely on.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from model_common import *  # noqa: F401,F403
import model_common as MC

log = get_logger("m06_clustering_inference")
style()
T = out_dir(MODEL_DIR, "tables")
Y, X = OUTCOME, TREATMENT
FE = {"M3": "| msoa11cd + year", "M6": "| msoa11cd + lad_year"}


def group_means_demean(v: np.ndarray, codes_list: list[np.ndarray]) -> np.ndarray:
    for codes in codes_list:
        s = np.bincount(codes, weights=v)
        n = np.bincount(codes)
        v = v - (s / n)[codes]
    return v


def wild_cluster_bootstrap(d: pd.DataFrame, x: str, fe_id: str, reps: int = 999, seed: int = 123) -> dict:
    """Wild-cluster (Rademacher) bootstrap-t for H0: coefficient on x = 0, clusters = LAD, FE structure M3 or M6.
    Imposes the null (restricted residuals = within-transformed y) and works in the within space (balanced panel).
    The small-sample factor is applied identically to the actual and bootstrap statistics."""
    rng = np.random.default_rng(seed)
    codes_m = pd.factorize(d.msoa11cd)[0]
    codes_t = pd.factorize(d.year if fe_id == "M3" else d.lad_year)[0]
    cl = pd.factorize(d.lad23cd_analysis)[0]
    G = cl.max() + 1
    xt = group_means_demean(d[x].values.astype(float), [codes_m, codes_t])
    yt = group_means_demean(d[Y].values.astype(float), [codes_m, codes_t])
    sxx = (xt ** 2).sum()

    def t_stat(yv):
        b = (xt * yv).sum() / sxx
        e = yv - b * xt
        s = np.bincount(cl, weights=xt * e)
        se = np.sqrt((s ** 2).sum()) / sxx
        return b, b / se

    b0, t0 = t_stat(yt)
    tb = np.empty(reps)
    for r in range(reps):
        w = rng.choice([-1.0, 1.0], size=G)
        ystar = group_means_demean(w[cl] * yt, [codes_m, codes_t])
        tb[r] = t_stat(ystar)[1]
    return {"coef": b0, "t_cluster_raw": t0, "wild_bootstrap_p": float((np.abs(tb) >= abs(t0)).mean()),
            "reps": reps, "n_clusters": int(G)}


def main() -> int:
    df = load_panel()
    d = estimation_sample(df)
    for m, c in Z_COL.items():
        d[f"x_{m}"] = d[X] * d[c]
    ints = [f"x_{m}" for m in PRIMARY_MODERATORS]

    # ---- 1. Standard errors of the baseline coefficient under different clustering choices -----------------------------------
    # WHAT: the M3 and M6 baseline coefficient with default (iid), heteroskedasticity-robust (HC1), MSOA-clustered, LAD-
    #       clustered, region-clustered, and LAD-and-year two-way clustered standard errors.
    # LOOK FOR: how much the standard error grows from iid to MSOA to LAD clustering (it measures how much correlation
    #           the coarser unit allows for); the number of clusters behind each; and the instability of the region version
    #           (9 clusters).
    rows = []
    for fe_id, fe in FE.items():
        for vc in ("iid", "hc1", "msoa", "lad", "region", "lad_and_year"):
            f = fit(f"{Y} ~ {X} {fe}", d, f"{fe_id}_se_{vc}", vcov=vc, note=f"SE: {vc}")
            t = coef_table(f, f"{fe_id}_{vc}", vc, [X]).assign(fe=fe_id, vcov=vc)
            rows.append(t)
    se = pd.concat(rows, ignore_index=True)
    base_se = se[se.vcov == "iid"].set_index("fe")["se"]
    se["se_ratio_to_iid"] = se.apply(lambda r: r.se / base_se[r.fe], axis=1)
    se.to_csv(T / "m06_baseline_se_by_clustering.csv", index=False)

    # ---- 2. The same for the four-moderator interaction model (joint test of the interactions) ---------------------------------
    # WHAT: the joint Wald test that the four interactions are zero, and each interaction's standard error, under MSOA, LAD
    #       and region clustering.
    # LOOK FOR: whether the conclusion of the joint test depends on the clustering level.
    rows, jt = [], []
    for fe_id, fe in FE.items():
        for vc in ("msoa", "lad", "region"):
            f = fit(f"{Y} ~ {X} + {' + '.join(ints)} {fe}", d, f"{fe_id}_se_mods_{vc}", vcov=vc, note=f"SE: {vc}")
            rows.append(coef_table(f, f"{fe_id}_{vc}", vc, [X] + ints).assign(fe=fe_id, vcov=vc))
            jt.append({"fe": fe_id, "vcov": vc, **wald(f, ints)})
    pd.concat(rows, ignore_index=True).to_csv(T / "m06_moderator_se_by_clustering.csv", index=False)
    pd.DataFrame(jt).to_csv(T / "m06_moderator_joint_by_clustering.csv", index=False)

    # ---- 3. Wild-cluster bootstrap (LAD clusters) for the baseline coefficient and each single interaction ---------------------
    # WHAT: bootstrap-t p-values with 999 Rademacher draws, imposing the null, for (a) the baseline coefficient on x and
    #       (b) each interaction x*z entered alone (coefficient on the interaction in a model with x and that interaction).
    #       Run for M3 and M6. Implemented directly (balanced-panel within transformation) and checked against the model fit.
    # LOOK FOR: whether the bootstrap p-value agrees with the asymptotic LAD-clustered p-value (large differences point to
    #           too few or too unbalanced clusters).
    rows = []
    for fe_id, fe in FE.items():
        for xv in [X] + ints:
            if xv == X:
                wb = wild_cluster_bootstrap(d, X, fe_id)
                f = fit(f"{Y} ~ {X} {fe}", d, f"{fe_id}_wcb_{xv}", vcov="lad")
                p_asym = float(f.pvalue().iloc[0])
            else:
                # Frisch-Waugh: partial x out of both the outcome and the interaction (within the FE space), then the
                # single-regressor bootstrap applies to the interaction coefficient
                codes_m = pd.factorize(d.msoa11cd)[0]
                codes_t = pd.factorize(d.year if fe_id == "M3" else d.lad_year)[0]
                a = group_means_demean(d[X].values.astype(float), [codes_m, codes_t])
                bz = group_means_demean(d[xv].values.astype(float), [codes_m, codes_t])
                yt = group_means_demean(d[Y].values.astype(float), [codes_m, codes_t])
                tmp = d.assign(**{Y: yt - a * ((a * yt).sum() / (a ** 2).sum()),
                                  X: bz - a * ((a * bz).sum() / (a ** 2).sum())})
                wb = wild_cluster_bootstrap(tmp, X, fe_id)
                f = fit(f"{Y} ~ {X} + {xv} {fe}", d, f"{fe_id}_wcb_{xv}", vcov="lad")
                p_asym = float(f.pvalue().loc[xv])
            rows.append({"fe": fe_id, "term": xv, "coef_wcb_space": wb["coef"], "t_cluster": wb["t_cluster_raw"],
                         "p_wild_cluster_bootstrap": wb["wild_bootstrap_p"], "p_asymptotic_lad_cluster": p_asym,
                         "reps": wb["reps"], "n_clusters": wb["n_clusters"]})
    pd.DataFrame(rows).to_csv(T / "m06_wild_cluster_bootstrap.csv", index=False)
    log.info("Wild-cluster bootstrap:\n" + pd.DataFrame(rows).round(4).to_string())
    MC.save_registry("m06")
    log.info("M06 complete.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
