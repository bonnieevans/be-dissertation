"""M05 - spillovers from construction in neighbouring MSOAs, using the PERSISTED exposure variables (src/13b_prepare_spatial_spillovers.py).

Concern: if new housing in one MSOA affects prices in adjacent MSOAs (the same local market), the own-MSOA coefficient mixes a direct effect and
the effect of neighbouring construction that is correlated with it. The check adds neighbouring construction (a spatial-lag-of-X, SLX, term) and
compares estimates with and without it. Neighbouring prices are NOT used as a regressor (simultaneity / reflection problem).

Order of work
  1. Reproduce the preliminary results (own + queen-neighbour MEAN of rates, own + cross-LAD mean, four moderators) from the persisted
     variables and compare them with the results saved before the exposure variables were persisted (hard check).
  2. Specification comparison on identical samples: own only; + queen mean rate; + pooled queen rate; + pooled 5 km rate; + distance-weighted rate;
     + pooled queen rate and its interaction with focal baseline income.
All exposures are in new builds per 1,000 households (lag 1), the same units as own construction. These are exploratory associations, not causal estimates.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from model_common import *  # noqa: F401,F403
import model_common as MC
import scipy.sparse as sp

log = get_logger("m05_spillovers")
style()
T = out_dir(MODEL_DIR, "tables")
Y, X = OUTCOME, TREATMENT
FE = {"M3": "| msoa11cd + year", "M6": "| msoa11cd + lad_year"}
G = MC.EC.PROJECT_ROOT / "data" / "interim" / "geography_crosswalks"
PRELIM = {"queen_mean": "nbr_queen_nb_mean_rate_lag1", "queen_pooled": "nbr_queen_nb_lag1_rate_per1000",
          "km5_pooled": "nbr_5km_nb_lag1_rate_per1000", "exp10": "nbr_exp10km_nb_lag1_rate_per1000"}


def legacy_crosslad_mean(df: pd.DataFrame) -> pd.Series:
    """Row-standardised mean of neighbours' lag-1 RATES over cross-LAD queen neighbours (the preliminary model-lab definition), rebuilt
    from the saved queen lookup. Used only to check reproduction; the persisted cross-LAD variable is the pooled rate."""
    q = pd.read_parquet(G / "msoa11_spatial_neighbors_queen.parquet")
    q = q[~q.same_lad]
    ids = sorted(df.msoa11cd.unique())
    pos = {m: i for i, m in enumerate(ids)}
    W = sp.csr_matrix((np.ones(len(q)), (q.focal_msoa11cd.map(pos), q.neighbour_msoa11cd.map(pos))), shape=(len(ids), len(ids)))
    n = np.asarray(W.sum(axis=1)).ravel()
    P = df.pivot(index="msoa11cd", columns="year", values=X).loc[ids]
    m = (W @ P.values) / np.where(n > 0, n, 1)[:, None]
    m[n == 0] = np.nan
    return pd.DataFrame(m, index=ids, columns=P.columns).stack().rename("nb_mean_cross_lad_legacy")


def main() -> int:
    df = load_panel()
    needed = list(PRELIM.values())
    if any(c not in df.columns for c in needed):
        log.error("persisted spillover variables are missing from the final panel - run 13b and 14 first.")
        return 1
    leg = legacy_crosslad_mean(df).reset_index()
    leg.columns = ["msoa11cd", "year", "nb_mean_cross_lad_legacy"]
    df = df.merge(leg, on=["msoa11cd", "year"], how="left")
    df["nb_mean_all"] = df[PRELIM["queen_mean"]]                       # same name as in the preliminary script
    d = estimation_sample(df)
    x_sd = d[X].std()
    d_cross = estimation_sample(df[df.nb_mean_cross_lad_legacy.notna()].copy())
    pd.DataFrame({"n_msoa": [df.msoa11cd.nunique()], "share_with_cross_lad_neighbour": [df.drop_duplicates("msoa11cd").nb_mean_cross_lad_legacy.notna().mean()
                                                                                           if False else (df.groupby("msoa11cd").nb_mean_cross_lad_legacy.apply(lambda s: s.notna().any()).mean())],
                  "mean_n_neighbours": [df.drop_duplicates("msoa11cd").nbr_queen_n.mean()]}).to_csv(T / "m05_neighbour_variable_info.csv", index=False)

    # ---- 1. Reproduce the preliminary specifications from the persisted variables ----
    # WHAT: the preliminary spillover regressions (own only; own + mean of neighbours' rates; own + cross-LAD neighbour mean), M3 and M6, LAD-clustered.
    # LOOK FOR: that every coefficient, standard error and sample size equals the value saved before the variables were persisted (differences > 1e-9 stop the run).
    rows = []
    for fe_id, fe in FE.items():
        for lab, dat, rhs in (("own only (full sample)", d, X), ("own + all-neighbour mean", d, f"{X} + nb_mean_all"),
                              ("own only (cross-LAD-neighbour sample)", d_cross, X),
                              ("own + cross-LAD neighbour mean", d_cross, f"{X} + nb_mean_cross_lad_legacy")):
            f = fit(f"{Y} ~ {rhs} {fe}", dat, f"{fe_id}_prelim_{lab}", note=lab)
            terms = [t for t in (X, "nb_mean_all", "nb_mean_cross_lad_legacy") if t in f._coefnames]
            rows.append(coef_table(f, f"{fe_id}_{lab}", lab, terms, x_sd).assign(fe=fe_id, spec=lab))
    new = pd.concat(rows, ignore_index=True)
    new["term"] = new["term"].replace({"nb_mean_cross_lad_legacy": "nb_mean_cross_lad"})
    new.to_csv(T / "m05_spillover_baseline.csv", index=False)
    old = pd.read_csv(MC.EC.PROJECT_ROOT / "outputs" / "models" / "tables" / "m05_spillover_baseline_PRELIMINARY_SAVED.csv") \
        if (T / "m05_spillover_baseline_PRELIMINARY_SAVED.csv").exists() else None
    if old is not None:
        mg = new.merge(old, on=["fe", "spec", "term"], suffixes=("_new", "_old"))
        mg["abs_diff_coef"] = (mg.coef_new - mg.coef_old).abs()
        mg["abs_diff_se"] = (mg.se_new - mg.se_old).abs()
        mg[["fe", "spec", "term", "coef_old", "coef_new", "abs_diff_coef", "se_old", "se_new", "abs_diff_se", "n_obs_old", "n_obs_new"]].to_csv(
            T / "m05_reproduction_check.csv", index=False)
        worst = float(max(mg.abs_diff_coef.max(), mg.abs_diff_se.max()))
        log.info(f"Reproduction of {len(mg)} preliminary coefficients: max abs difference {worst:.2e}; n_obs equal: {bool((mg.n_obs_old == mg.n_obs_new).all())}")
        if len(mg) != len(old) or worst > 1e-9 or not (mg.n_obs_old == mg.n_obs_new).all():
            log.error("STOP: the persisted variables do not reproduce the preliminary spillover results.")
            return 1
    else:
        log.warning("no saved preliminary results to compare with (m05_spillover_baseline_PRELIMINARY_SAVED.csv).")

    # four-moderator model with and without the neighbour variable (as before)
    for m, c in Z_COL.items():
        d[f"x_{m}"] = d[X] * d[c]
    ints = [f"x_{m}" for m in PRIMARY_MODERATORS]
    rows, jt = [], []
    for fe_id, fe in FE.items():
        for lab, extra in (("four moderators", ""), ("four moderators + neighbour mean", " + nb_mean_all")):
            f = fit(f"{Y} ~ {X} + {' + '.join(ints)}{extra} {fe}", d, f"{fe_id}_prelim_mods_{lab}", note=lab)
            rows.append(coef_table(f, f"{fe_id}_{lab}", lab, [X] + ints + (["nb_mean_all"] if extra else []), x_sd).assign(fe=fe_id, spec=lab))
            jt.append({"fe": fe_id, "spec": lab, **wald(f, ints)})
    pd.concat(rows, ignore_index=True).to_csv(T / "m05_spillover_with_moderators.csv", index=False)
    pd.DataFrame(jt).to_csv(T / "m05_spillover_with_moderators_joint.csv", index=False)

    # ---- 2. Specification comparison on identical samples ----
    # WHAT: six specifications, each for M3 (MSOA + year FE) and M6 (MSOA + LAD x year FE), LAD-clustered:
    #   (1) own construction only; (2) + queen mean of neighbours' rates (preliminary measure); (3) + pooled queen rate (total completions / total households);
    #   (4) + pooled 5 km rate; (5) + distance-weighted rate (exp decay, 3 km parameter, within 10 km); (6) + pooled queen rate and its interaction with focal baseline income (z).
    #   All regressions use the SAME rows: MSOA-years where every exposure is defined (MSOAs with no centroid within 5 km are excluded), and the main estimation sample.
    # LOOK FOR: the own-construction coefficient across (1)-(6) (does adding a neighbour term change it?); the neighbour coefficient and its standard error
    #           under M3 vs M6; whether the choice of neighbour definition (adjacent MSOAs, 5 km pool, decay-weighted) changes the sign or size;
    #           in (6), the sign and size of the interaction and the joint Wald test of the neighbour term and its interaction.
    # UNITS:    own and neighbour rates are both new builds per 1,000 households (lag 1). The neighbour coefficient is the change in log median nominal price per
    #           m2 associated with one more completion per 1,000 households in the neighbouring MSOAs (pooled), holding own construction and the fixed effects constant.
    #           x 100 is approximately a percentage change. The interaction is the change in that coefficient for a focal MSOA one England-wide SD higher in income.
    common = d.dropna(subset=list(PRELIM.values())).copy()
    common["x_nbq_inc"] = common[PRELIM["queen_pooled"]] * common[Z_COL["income"]]
    log.info(f"Common sample: {len(common):,} MSOA-years ({common.msoa11cd.nunique():,} MSOAs) of {len(d):,}")
    specs = [("1 own only", X, None), ("2 + queen mean of rates", f"{X} + {PRELIM['queen_mean']}", PRELIM["queen_mean"]),
             ("3 + queen pooled rate", f"{X} + {PRELIM['queen_pooled']}", PRELIM["queen_pooled"]),
             ("4 + 5 km pooled rate", f"{X} + {PRELIM['km5_pooled']}", PRELIM["km5_pooled"]),
             ("5 + distance-weighted rate", f"{X} + {PRELIM['exp10']}", PRELIM["exp10"]),
             ("6 + queen pooled rate x focal income", f"{X} + {PRELIM['queen_pooled']} + x_nbq_inc", PRELIM["queen_pooled"])]
    rows, jts = [], []
    for fe_id, fe in FE.items():
        for lab, rhs, nbv in specs:
            f = fit(f"{Y} ~ {rhs} {fe}", common, f"{fe_id}_spillcomp_{lab}", note=lab, sample="common spillover sample")
            terms = [t for t in f._coefnames]
            rows.append(coef_table(f, f"{fe_id}_{lab}", lab, terms, x_sd).assign(fe=fe_id, spec=lab))
            if "x_nbq_inc" in f._coefnames:
                jts.append({"fe": fe_id, "spec": lab, "test": "neighbour term and its interaction with income jointly zero", **wald(f, [nbv, "x_nbq_inc"])})
    comp = pd.concat(rows, ignore_index=True)
    comp["term"] = comp["term"].replace({v: k for k, v in {"queen mean rate": PRELIM["queen_mean"], "queen pooled rate": PRELIM["queen_pooled"],
                                                            "5 km pooled rate": PRELIM["km5_pooled"], "distance-weighted rate": PRELIM["exp10"],
                                                            "neighbour construction x focal income z": "x_nbq_inc"}.items()})
    comp.to_csv(T / "m05_spillover_specification_comparison.csv", index=False)
    pd.DataFrame(jts).to_csv(T / "m05_spillover_specification_comparison_joint_tests.csv", index=False)
    log.info("Specification comparison (neighbour terms):\n" + comp[comp.term != X][["fe", "spec", "term", "coef", "se", "p"]].round(6).to_string(index=False))
    MC.save_registry("m05")
    log.info("M05 complete.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
