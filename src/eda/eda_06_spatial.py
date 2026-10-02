"""EDA 06 - spatial structure and spatial autocorrelation.

Why this matters in an econometric context
- Neighbouring MSOAs share housing markets, so prices, and the shocks to them, are correlated across
  space. Positive spatial autocorrelation in regression errors makes conventional and MSOA-clustered
  standard errors too small, and motivates clustering at a larger unit (LAD) or spatial HAC errors.
- The neighbour structure built here (queen contiguity on the MSOA11 boundaries) is also what defines
  the 'construction in neighbouring MSOAs' spillover variable used in the model lab.
- Moran's I measures the global correlation of a variable with the average of its neighbours;
  local Moran's I (LISA) shows where the clusters are.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from eda_common import *  # noqa: F401,F403
import esda

log = get_logger("eda_06_spatial")
style()
T = out_dir(EDA_DIR, "tables", "spatial")
F = out_dir(EDA_DIR, "figures", "spatial")


def main() -> int:
    df = load_panel()
    base = load_baseline().set_index("msoa11cd")

    # ---- 1. Neighbour structure ---------------------------------------------------------------------------
    # WHAT: queen-contiguity weights from the MSOA11 boundaries; island MSOAs are attached to their nearest
    #       neighbour so every area has at least one. Weights are row-standardised for Moran's I.
    # LOOK FOR: the distribution of the number of neighbours (very small or very large counts affect how
    #           comparable the neighbour averages are) and the number of islands attached.
    w, islands = queen_weights()
    ids = list(w.id_order)
    card = pd.Series(w.cardinalities).reindex(ids)
    pd.DataFrame({"n_areas": [w.n], "islands_attached": [len(islands)], "mean_neighbours": [card.mean()],
                  "min": [card.min()], "median": [card.median()], "max": [card.max()]}).to_csv(T / "06_neighbour_structure.csv", index=False)
    w.transform = "r"

    # ---- 2. Global Moran's I by year ------------------------------------------------------------------------
    # WHAT: Moran's I (999 permutations) of the outcome level, annual price growth and the treatment, for each
    #       year, and for the baseline moderators (one value each).
    # LOOK FOR: the size of I (0 = no spatial autocorrelation, 1 = perfect clustering) and whether it is stable
    #           across years; the permutation p-value; comparison between price levels (very clustered), growth
    #           (less so) and the treatment.
    rows = []
    for var in (OUTCOME, "d_log_price", "newbuilds_per_1000", "newbuilds_lag1_per_1000"):
        for yr, g in df.groupby("year"):
            s = g.set_index("msoa11cd")[var].reindex(ids)
            if s.isna().any():
                continue
            mi = esda.Moran(s.values, w, permutations=999)
            rows.append({"variable": var, "year": yr, "morans_I": mi.I, "z_sim": mi.z_sim, "p_sim": mi.p_sim})
    for k, col in MODERATORS.items():
        mi = esda.Moran(base.loc[ids, col].values, w, permutations=999)
        rows.append({"variable": f"baseline_{k}", "year": np.nan, "morans_I": mi.I, "z_sim": mi.z_sim, "p_sim": mi.p_sim})
    mo = pd.DataFrame(rows)
    mo.to_csv(T / "06_morans_I.csv", index=False)
    fig, ax = plt.subplots(figsize=(7, 4))
    for v, s in mo[mo.variable.isin([OUTCOME, "d_log_price", "newbuilds_per_1000"])].groupby("variable"):
        ax.plot(s.year, s.morans_I, marker="o", label=v)
    ax.set_title("Global Moran's I by year (queen contiguity)")
    ax.legend()
    savefig(fig, F / "morans_I_by_year.png")
    log.info("Moran's I:\n" + mo.round(3).to_string())

    # ---- 3. Local Moran's I (LISA) ---------------------------------------------------------------------------------
    # WHAT: local indicators of spatial association for baseline deprivation, cumulative construction per 1,000 households
    #       (2016-2023) and 2016-2023 price growth; High-High / Low-Low are clusters, High-Low / Low-High are spatial outliers.
    # LOOK FOR: where the clusters sit (for example whether construction clusters in particular regions or in
    #           particular deprivation clusters), which indicates how much of the variation is shared within a local area.
    gdf = msoa_gdf().set_index("msoa11cd").loc[ids]
    cum = df[df.main_sample].groupby("msoa11cd")["newbuilds_per_1000"].sum().reindex(ids)
    growth = (df[df.year == 2023].set_index("msoa11cd")[OUTCOME] - df[df.year == 2016].set_index("msoa11cd")[OUTCOME]).reindex(ids)
    vars_ = {"deprivation (baseline)": base.loc[ids, MODERATORS["deprivation"]], "cumulative new-builds per 1,000 (2016-23)": cum,
             "log price growth 2016-23": growth}
    fig, axes = plt.subplots(1, 3, figsize=(19, 8))
    names = {1: "High-High", 2: "Low-High", 3: "Low-Low", 4: "High-Low"}
    cols = {0: "#eeeeee", 1: "#d7191c", 2: "#abd9e9", 3: "#2c7bb6", 4: "#fdae61"}
    summ = []
    for ax, (nm, s) in zip(axes, vars_.items()):
        lm = esda.Moran_Local(s.values, w, permutations=999, seed=7)
        q = np.where(lm.p_sim < .05, lm.q, 0)
        gdf["lisa"] = q
        gdf.assign(c=[cols[v] for v in q]).plot(color=gdf.assign(c=[cols[v] for v in q])["c"], ax=ax, linewidth=0)
        ax.set_title(f"LISA: {nm}")
        ax.set_axis_off()
        vc = pd.Series(q).map({0: "not significant", **names}).value_counts()
        for k_, v_ in vc.items():
            summ.append({"variable": nm, "cluster_type": k_, "n_msoa": int(v_)})
    handles = [plt.Rectangle((0, 0), 1, 1, color=cols[k]) for k in cols]
    fig.legend(handles, ["not significant"] + list(names.values()), loc="lower center", ncol=5)
    savefig(fig, F / "lisa_maps.png", dpi=90)
    pd.DataFrame(summ).to_csv(T / "06_lisa_cluster_counts.csv", index=False)

    # ---- 4. Spatial correlation of the treatment with its neighbours ----------------------------------------------------
    # WHAT: correlation between each MSOA's construction intensity and the average of its neighbours', pooled over years
    #       and within MSOA after year effects are removed (the variation the spillover regression would use).
    # LOOK FOR: how strongly an MSOA's construction moves with its neighbours' (relevant to whether the spillover variable
    #           is collinear with the own-MSOA treatment).
    P = df.pivot(index="msoa11cd", columns="year", values="newbuilds_per_1000").loc[ids]
    NB = pd.DataFrame(w.sparse @ P.values, index=ids, columns=P.columns)       # row-standardised: neighbours' mean
    pooled = np.corrcoef(P.values.ravel(), NB.values.ravel())[0, 1]
    def tw(M):
        M = M - M.mean(axis=1, keepdims=True)
        return M - M.mean(axis=0, keepdims=True)
    within = np.corrcoef(tw(P.values).ravel(), tw(NB.values).ravel())[0, 1]
    pd.DataFrame({"pooled_corr_own_vs_neighbour_mean": [pooled], "two_way_within_corr_own_vs_neighbour_mean": [within]}).to_csv(
        T / "06_treatment_own_vs_neighbour_correlation.csv", index=False)
    log.info(f"Own vs neighbour-mean construction: pooled {pooled:.3f}, two-way within {within:.3f}")
    log.info("EDA 06 complete.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
