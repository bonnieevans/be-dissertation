"""M05 - spillovers from construction in neighbouring MSOAs.

Concern: if new housing in one MSOA affects prices in adjacent MSOAs (the same local market), the own-MSOA
coefficient mixes a direct effect and the effect of neighbouring construction that is correlated with it
(own and neighbour construction are positively correlated, EDA 06). The check adds the construction rate of
neighbouring MSOAs and compares estimates with and without it.

Neighbours: queen contiguity on MSOA11 boundaries (islands attached to their nearest neighbour). The neighbour
variable is the average lag-1 new-build rate of an MSOA's neighbours (row-standardised weights), built in memory.
Two versions: all neighbours, and only neighbours in a DIFFERENT LAD. With LAD x year effects (M6) the
variation in the all-neighbours variable that is common within a LAD-year is removed, whereas the
cross-LAD version keeps variation across LAD boundaries.
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


def main() -> int:
    df = load_panel()
    w, islands = MC.EC.queen_weights()
    ids = list(w.id_order)
    pos = {m: i for i, m in enumerate(ids)}
    lad = df.drop_duplicates("msoa11cd").set_index("msoa11cd").loc[ids, "lad23cd_analysis"].values
    rows_i, rows_j = [], []
    for a, nb in w.neighbors.items():
        for b in nb:
            rows_i.append(pos[a])
            rows_j.append(pos[b])
    ri, rj = np.array(rows_i), np.array(rows_j)
    Wb = sp.csr_matrix((np.ones(len(ri)), (ri, rj)), shape=(len(ids), len(ids)))
    cross = lad[ri] != lad[rj]
    Wc = sp.csr_matrix((np.ones(cross.sum()), (ri[cross], rj[cross])), shape=(len(ids), len(ids)))

    def row_std(W):
        s = np.asarray(W.sum(axis=1)).ravel()
        inv = np.where(s > 0, 1 / np.where(s > 0, s, 1), 0)
        return sp.diags(inv) @ W, s > 0

    Wr, _ = row_std(Wb)
    Wcr, has_cross = row_std(Wc)
    P = df.pivot(index="msoa11cd", columns="year", values=X).loc[ids]
    nb_all = pd.DataFrame(Wr @ P.values, index=ids, columns=P.columns).stack().rename("nb_mean_all").reset_index()
    nb_cross = pd.DataFrame(Wcr @ P.values, index=ids, columns=P.columns).stack().rename("nb_mean_cross_lad").reset_index()
    nb_all.columns = nb_cross.columns = ["msoa11cd", "year", nb_all.columns[-1]]
    nb_cross.columns = ["msoa11cd", "year", "nb_mean_cross_lad"]
    d0 = df.merge(nb_all, on=["msoa11cd", "year"]).merge(nb_cross, on=["msoa11cd", "year"])
    d0["has_cross_lad_neighbour"] = d0.msoa11cd.map(pd.Series(has_cross, index=ids))
    d0 = d0.merge(pd.read_csv(MC.EC.EDA_DIR / "tables" / "pca" / "04_scores_four_primary_moderators.csv")[["msoa11cd"]], on="msoa11cd")
    d = estimation_sample(d0)
    x_sd = d[X].std()
    pd.DataFrame({"n_msoa": [len(ids)], "share_with_cross_lad_neighbour": [has_cross.mean()],
                  "mean_n_neighbours": [np.asarray(Wb.sum(axis=1)).mean()]}).to_csv(T / "m05_neighbour_variable_info.csv", index=False)

    # ---- 1. Baseline coefficient with and without the neighbours' construction ------------------------------------------------
    # WHAT: M3 and M6 with (i) own construction only, (ii) own plus the mean of neighbours' construction (all neighbours),
    #       (iii) own plus the mean of neighbours in other LADs, on the sample of MSOAs that have such neighbours (the
    #       own-only model is re-estimated on that same sample for a like-for-like comparison).
    # LOOK FOR: the change in the own-MSOA coefficient and its standard error when the neighbour variable is added; the
    #           size and sign of the neighbour coefficient; whether the neighbour variable keeps any variation under LAD x year
    #           effects (compare its standard error in M3 and M6).
    # UNITS:    the neighbour coefficient is the change in log median nominal price per m2 in an MSOA associated with one more
    #           new-build per 1,000 households (lag 1), on average across its neighbouring MSOAs.
    rows = []
    d_cross = estimation_sample(d0[d0.has_cross_lad_neighbour].copy())
    for fe_id, fe in FE.items():
        for lab, dat, rhs in (("own only (full sample)", d, X), ("own + all-neighbour mean", d, f"{X} + nb_mean_all"),
                              ("own only (cross-LAD-neighbour sample)", d_cross, X),
                              ("own + cross-LAD neighbour mean", d_cross, f"{X} + nb_mean_cross_lad")):
            f = fit(f"{Y} ~ {rhs} {fe}", dat, f"{fe_id}_spill_{lab}", note=lab)
            terms = [t for t in (X, "nb_mean_all", "nb_mean_cross_lad") if t in f._coefnames]
            rows.append(coef_table(f, f"{fe_id}_{lab}", lab, terms, x_sd).assign(fe=fe_id, spec=lab))
    pd.concat(rows, ignore_index=True).to_csv(T / "m05_spillover_baseline.csv", index=False)

    # ---- 2. Do the moderator interactions change when the neighbour variable is added? ---------------------------------------
    # WHAT: the four-moderator interaction model (M03) with and without the all-neighbour variable.
    # LOOK FOR: changes in the interaction coefficients and the joint test of the interactions.
    for m, c in Z_COL.items():
        d[f"x_{m}"] = d[X] * d[c]
    ints = [f"x_{m}" for m in PRIMARY_MODERATORS]
    rows, jt = [], []
    for fe_id, fe in FE.items():
        for lab, extra in (("four moderators", ""), ("four moderators + neighbour mean", " + nb_mean_all")):
            f = fit(f"{Y} ~ {X} + {' + '.join(ints)}{extra} {fe}", d, f"{fe_id}_spill_mods_{lab}", note=lab)
            rows.append(coef_table(f, f"{fe_id}_{lab}", lab, [X] + ints + (["nb_mean_all"] if extra else []), x_sd).assign(fe=fe_id, spec=lab))
            jt.append({"fe": fe_id, "spec": lab, **wald(f, ints)})
    pd.concat(rows, ignore_index=True).to_csv(T / "m05_spillover_with_moderators.csv", index=False)
    pd.DataFrame(jt).to_csv(T / "m05_spillover_with_moderators_joint.csv", index=False)
    MC.save_registry("m05")
    log.info("M05 complete.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
