"""Spatial spillover exposure: explicit neighbour networks and neighbouring-construction / income / price measures at MSOA11.

Runs after 13_build_msoa_newbuild_panel.py and before 14_merge_final_panel.py. Reads INTERMEDIATE tables only (never the final panel),
so the whole pipeline can be rebuilt from scratch.

Networks (fixed 2011 MSOA geography)
  A  queen contiguity (primary): shared boundary or point. Island MSOAs are attached to their nearest neighbour exactly as in the EDA and
     model lab (queen_attach_islands); those edges are flagged edge_type = 'island_fallback' and are not genuine contiguity.
  B  centroid distance <= 5 km and <= 10 km (ONS population-weighted centroids, British National Grid; centroid-to-centroid, not boundary-to-boundary)
  C  exponential distance decay w_ij = exp(-d_ij / decay_m), pairs within 10 km
  (optional) second-order queen neighbours

Measures: pooled neighbouring construction rates (sum of neighbours' completions per 1,000 of the neighbours' combined baseline households),
the existing unweighted mean of neighbours' rates (to reproduce m05), cross-LAD exposure, distance-based exposures, household-weighted
neighbouring baseline income, and sale-count-weighted neighbouring log prices. Rates with no eligible neighbours are NaN, not zero.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent / "eda"))
import spillover_methods as sm  # noqa: E402
from utils import PROJECT_ROOT, get_logger, load_config  # noqa: E402

log = get_logger("13b_prepare_spatial_spillovers")
G = PROJECT_ROOT / "data" / "interim" / "geography_crosswalks"
QA = PROJECT_ROOT / "outputs" / "qa" / "spatial_spillovers"


def fail(msg: str) -> int:
    log.error(f"STOP: {msg}")
    return 1


def main() -> int:
    cfg = load_config()["spatial_spillovers"]
    if not cfg["enabled"]:
        log.info("spatial_spillovers.enabled is false - nothing to do.")
        return 0
    QA.mkdir(parents=True, exist_ok=True)
    import eda_common as C  # noqa: E402  (reuses queen_weights / queen_edge_table)

    nb = pd.read_parquet(G / "msoa_year_newbuild_panel.parquet")
    pr = pd.read_parquet(G / "msoa_year_price_panel_trimmed.parquet")
    inc = pd.read_parquet(G / "msoa11_baseline_income.parquet")
    geo = pd.read_parquet(G / "msoa11_geography_reference.parquet")
    years = sorted(nb.year.unique())
    ids = sorted(nb.msoa11cd.unique())
    N, T = len(ids), len(years)
    if N != 6791 or T != 12 or len(nb) != N * T or nb.duplicated(["msoa11cd", "year"]).any():
        return fail(f"new-build panel is not a balanced 6,791 x 12 panel (N={N}, T={T}, rows={len(nb)}).")
    pos = {m: i for i, m in enumerate(ids)}

    def mat(df: pd.DataFrame, col: str) -> np.ndarray:
        return df.pivot(index="msoa11cd", columns="year", values=col).reindex(index=ids, columns=years).to_numpy(dtype=float)

    hh = nb.drop_duplicates("msoa11cd").set_index("msoa11cd").loc[ids, "baseline_households_2011"].to_numpy(dtype=float)
    if (hh <= 0).any() or np.isnan(hh).any():
        return fail("non-positive or missing baseline households.")
    cnt, lag1, lag2, lag3, prev3 = (mat(nb, c) for c in ("newbuild_total", "newbuilds_lag1", "newbuilds_lag2", "newbuilds_lag3", "newbuilds_prev3yr"))
    rate_lag1 = mat(nb, "newbuilds_lag1_per_1000")
    if np.isnan(lag1).any() or np.isnan(prev3).any() or (cnt < 0).any():
        return fail("missing or negative construction counts (the upstream lookback lags must be complete).")
    if not np.allclose(prev3, lag1 + lag2 + lag3):
        return fail("newbuilds_prev3yr is not lag1 + lag2 + lag3.")
    log.info(f"2012 lag1 total (= 2011 completions from the lookback): {lag1[:, 0].sum():,.0f}")

    # ---------------- geography ----------------
    bnd = pd.Series(sorted(C.msoa_gdf().msoa11cd))
    if list(bnd) != ids:
        return fail("MSOA11 boundary identifiers differ from the panel's MSOA11 universe.")
    pwc = C.gpd.read_file(PROJECT_ROOT / cfg["centroids_geojson"])
    # the saved file holds British National Grid coordinates (requested with outSR=27700) but carries no CRS tag, so the CRS is set, not converted
    pwc = pwc.set_crs(27700, allow_override=True).sort_values("msoa11cd").reset_index(drop=True)
    if not (np.isfinite(pwc.geometry.x).all() and pwc.geometry.x.between(0, 700000).all() and pwc.geometry.y.between(0, 1300000).all()):
        return fail("population-weighted centroids are not valid British National Grid coordinates.")
    centroid_note = "ONS 2011 population-weighted centroids (EPSG:27700)"
    if list(pwc.msoa11cd) != ids:
        return fail("population-weighted centroid file does not match the MSOA11 universe.")
    xy = np.column_stack([pwc.geometry.x.values, pwc.geometry.y.values])
    lad = geo.set_index("msoa11cd").loc[ids, "lad23cd_analysis"].to_numpy()

    # ---- Network A: queen ----
    qe = C.queen_edge_table() if cfg["queen_attach_islands"] else None
    qe["fi"] = qe.focal_msoa11cd.map(pos)
    qe["ni"] = qe.neighbour_msoa11cd.map(pos)
    qe["same_lad"] = lad[qe.fi.values] == lad[qe.ni.values]
    d_ = xy[qe.fi.values] - xy[qe.ni.values]
    qe["distance_m"] = np.hypot(d_[:, 0], d_[:, 1])
    Wq = sm.edges_to_matrix(qe.fi, qe.ni, N)
    gen = qe[qe.edge_type == "queen"]
    rev = set(zip(gen.ni, gen.fi))
    nonrecip = int((~pd.Series(list(zip(gen.fi, gen.ni))).isin(rev)).sum())
    if nonrecip:
        return fail(f"{nonrecip} genuine queen-contiguity edges are not reciprocal.")
    fb = qe[qe.edge_type == "island_fallback"]
    fb_rev = set(zip(fb.ni, fb.fi))
    fb_asym = int((~pd.Series(list(zip(fb.fi, fb.ni))).isin(fb_rev)).sum()) if len(fb) else 0
    log.info(f"Queen network: {len(qe):,} directed edges ({len(gen):,} genuine, {len(fb)} island fallback; fallback asymmetry {fb_asym}); "
             f"cross-LAD edges {int((~qe.same_lad).sum()):,}")
    # consistency with the existing EDA
    ref = pd.read_csv(PROJECT_ROOT / "outputs" / "eda" / "tables" / "spatial" / "06_neighbour_structure.csv")
    n_q = np.asarray(Wq.sum(axis=1)).ravel()
    if abs(float(ref.mean_neighbours.iloc[0]) - n_q.mean()) > 1e-9 or int(ref["max"].iloc[0]) != int(n_q.max()) or int(ref["min"].iloc[0]) != int(n_q.min()):
        return fail(f"queen neighbour counts differ from the EDA (EDA mean {ref.mean_neighbours.iloc[0]}, here {n_q.mean()}).")
    log.info(f"Queen neighbours per MSOA: mean {n_q.mean():.3f}, min {n_q.min():.0f}, max {n_q.max():.0f} (matches 06_neighbour_structure.csv)")

    # ---- Networks B and C: distance (population-weighted centroids) ----
    dmax = float(cfg["distance_max_m"])
    radii = sorted(float(r) for r in cfg["distance_radii_m"])
    if max(radii) > dmax:
        return fail("distance_radii_m exceeds distance_max_m.")
    dp = sm.distance_pairs(xy, dmax)
    dp["weight_exp"] = sm.exp_decay(dp.distance_m, float(cfg["distance_decay_m"]))
    dp["weight_exp_rownorm"] = dp.weight_exp / dp.groupby("i").weight_exp.transform("sum")
    for r in radii:
        dp[f"within_{int(r / 1000)}km"] = dp.distance_m <= r
    if (dp.distance_m < 0).any() or (dp.distance_m > dmax).any():
        return fail("distance values outside [0, distance_max_m].")
    for r in radii:
        if not (dp.loc[dp[f"within_{int(r / 1000)}km"], "distance_m"] <= r).all():
            return fail(f"within_{int(r / 1000)}km flag inconsistent with distance.")
    Wd = {int(r / 1000): sm.edges_to_matrix(dp.i[dp[f"within_{int(r / 1000)}km"]], dp.j[dp[f"within_{int(r / 1000)}km"]], N) for r in radii}
    Wexp = sm.edges_to_matrix(dp.i, dp.j, N, dp.weight_exp)
    n_d = {k: np.asarray(W.sum(axis=1)).ravel() for k, W in Wd.items()}
    log.info("Distance networks: " + "; ".join(f"<= {k} km: {len(dp[dp[f'within_{k}km']]):,} directed pairs, mean {n_d[k].mean():.1f}, "
                                              f"zero-neighbour MSOAs {int((n_d[k] == 0).sum())}" for k in n_d)
             + f"; weights up to {dmax / 1000:.0f} km: {len(dp):,} pairs")

    # ---------------- construction exposure (time-varying) ----------------
    out = {}
    Wcross = sm.edges_to_matrix(qe.fi[~qe.same_lad], qe.ni[~qe.same_lad], N) if cfg["compute_cross_lad"] else None
    out["nbr_queen_nb_total"] = sm.neighbour_sum(Wq, cnt)
    out["nbr_queen_nb_rate_per1000"] = sm.pooled_rate(Wq, cnt, hh)
    out["nbr_queen_nb_lag1_total"] = sm.neighbour_sum(Wq, lag1)
    out["nbr_queen_nb_lag1_rate_per1000"] = sm.pooled_rate(Wq, lag1, hh)
    out["nbr_queen_nb_prev3yr_rate_per1000"] = sm.pooled_rate(Wq, prev3, hh)
    out["nbr_queen_nb_mean_rate_lag1"] = sm.neighbour_mean(Wq, rate_lag1)
    if Wcross is not None:
        out["nbr_queen_crosslad_nb_lag1_rate_per1000"] = sm.pooled_rate(Wcross, lag1, hh)
    out["nbr_5km_nb_total"] = np.where((n_d[5] > 0)[:, None], sm.neighbour_sum(Wd[5], cnt), np.nan)   # undefined (NaN), not zero, with no neighbours
    out["nbr_5km_nb_rate_per1000"] = sm.pooled_rate(Wd[5], cnt, hh)
    out["nbr_5km_nb_lag1_rate_per1000"] = sm.pooled_rate(Wd[5], lag1, hh)
    out["nbr_10km_nb_lag1_rate_per1000"] = sm.pooled_rate(Wd[10], lag1, hh)
    out["nbr_exp10km_nb_lag1_rate_per1000"] = sm.pooled_rate(Wexp, lag1, hh)
    if cfg["compute_second_order"]:
        W2 = sm.second_order(Wq)
        out["nbr_queen2_nb_lag1_rate_per1000"] = sm.pooled_rate(W2, lag1, hh)
    # neighbouring prices (descriptive; never summed, missing never zero)
    if cfg["compute_neighbour_prices"]:
        lp, sales = mat(pr, "log_median_ppsqm"), mat(pr, "sale_count")
        mean, nv, tot = sm.sale_weighted_price(Wq, lp, sales)
        out["nbr_queen_price_log_ppsqm_salewmean"] = mean
        out["nbr_queen_price_n_valid"] = nv
        out["nbr_queen_price_total_sales"] = tot
        lagged = np.full_like(mean, np.nan)
        lagged[:, 1:] = mean[:, :-1]
        out["nbr_queen_price_log_ppsqm_salewmean_lag1"] = lagged     # previous year's value; for descriptive use

    expo = pd.DataFrame({"msoa11cd": np.repeat(ids, T), "year": np.tile(years, N)})
    for k, v in out.items():
        expo[k] = v.reshape(-1)
    expo["year"] = expo["year"].astype(int)
    # honest flags: where the exposure is undefined versus genuinely zero
    expo["nbr_5km_none"] = np.repeat(n_d[5] == 0, T)
    expo["nbr_10km_none"] = np.repeat(n_d[10] == 0, T)
    if Wcross is not None:
        expo["nbr_queen_crosslad_none"] = np.repeat(np.asarray(Wcross.sum(axis=1)).ravel() == 0, T)

    # ---------------- static characteristics (one row per MSOA) ----------------
    inc_v = inc.set_index("msoa11cd").loc[ids, "baseline_income_bhc_2011_12"].to_numpy(dtype=float)
    own_log = np.log(inc_v)
    st = pd.DataFrame({"msoa11cd": ids})
    st["nbr_queen_n"] = n_q.astype(int)
    st["nbr_queen_same_lad_n"] = np.asarray(sm.edges_to_matrix(qe.fi[qe.same_lad], qe.ni[qe.same_lad], N).sum(axis=1)).ravel().astype(int)
    st["nbr_queen_crosslad_n"] = (st.nbr_queen_n - st.nbr_queen_same_lad_n).astype(int)
    st["nbr_queen_has_island_fallback"] = np.isin(np.arange(N), np.unique(np.r_[fb.fi.values, fb.ni.values]))
    st["nbr_queen_households_2011"] = np.asarray(Wq @ hh)
    st["nbr_5km_n"] = n_d[5].astype(int)
    st["nbr_10km_n"] = n_d[10].astype(int)
    st["nbr_5km_households_2011"] = np.asarray(Wd[5] @ hh)
    st["nbr_exp10km_weight_sum"] = np.asarray(Wexp.sum(axis=1)).ravel()
    for nm, W in (("queen", Wq), ("5km", Wd[5])):
        m = sm.household_weighted_mean(W, inc_v, hh)
        st[f"nbr_{nm}_income_bhc2012_hhmean"] = m
        st[f"nbr_{nm}_log_income_hhmean"] = np.log(m)                 # log of the weighted mean level, NOT the mean of logs
        st[f"income_gap_own_minus_nbr_{nm}_log"] = own_log - np.log(m)
    if cfg["compute_second_order"]:
        st["nbr_queen2_n"] = np.asarray(W2.sum(axis=1)).ravel().astype(int)

    # ---------------- QA hard stops ----------------
    if set(geo.msoa11cd) != set(ids):
        return fail("some panel MSOAs have no spatial record.")
    if expo.duplicated(["msoa11cd", "year"]).any() or len(expo) != N * T:
        return fail("exposure table keys are not unique or not N x T.")
    for c in [c for c in expo.columns if c.endswith("_total") or "rate" in c]:
        if (expo[c].dropna() < 0).any():
            return fail(f"negative values in {c}.")
        if np.isinf(expo[c].to_numpy(dtype=float)).any():
            return fail(f"infinite values in {c}.")
    if np.isinf(st.select_dtypes("number").to_numpy(dtype=float)).any() or st.drop(columns=[c for c in st if c.startswith("nbr_5km")]).isna().any().any():
        bad = st.columns[st.isna().any()].tolist()
        # 5 km income measures may be NaN where an MSOA has no 5 km neighbour; anything else is an error
        if [b for b in bad if not b.startswith(("nbr_5km", "income_gap_own_minus_nbr_5km"))]:
            return fail(f"unexpected missing values in static characteristics: {bad}")
    # 2012 lagged exposures come from the upstream lookback and must not be missing where the neighbourhood is defined
    y12 = expo[expo.year == years[0]]
    for c in ("nbr_queen_nb_lag1_rate_per1000", "nbr_queen_nb_prev3yr_rate_per1000", "nbr_queen_nb_mean_rate_lag1"):
        if y12[c].isna().any():
            return fail(f"{c} is missing in {years[0]} (the lookback lags should make it defined).")

    # (1) reproduction of the existing preliminary exposure measure (m05_spillovers.py): row-standardised queen lag of newbuilds_lag1_per_1000
    w_old, _ = C.queen_weights()
    w_old.transform = "r"
    old = pd.DataFrame(np.asarray(w_old.sparse @ pd.DataFrame(rate_lag1, index=ids).loc[list(w_old.id_order)].to_numpy()), index=list(w_old.id_order)).loc[ids].to_numpy()
    gap = float(np.abs(old - out["nbr_queen_nb_mean_rate_lag1"]).max())
    log.info(f"Reproduction of the m05 row-standardised neighbour mean: max abs difference {gap:.2e}")
    if gap > 1e-9:
        return fail("nbr_queen_nb_mean_rate_lag1 does not reproduce the existing model-lab measure.")

    # (2) independent manual re-computation for >= 30 random MSOAs (plain pandas loops over the saved edge table)
    rng = np.random.default_rng(20261009)
    sample = rng.choice(N, size=40, replace=False)
    nbrs = qe.groupby("fi").ni.apply(list)
    checks = []
    for i in sample:
        js = nbrs[i]
        t = int(rng.integers(0, T))
        tot = sum(cnt[j, t] for j in js)
        den = sum(hh[j] for j in js)
        manual = {"nbr_queen_nb_total": tot, "nbr_queen_nb_rate_per1000": 1000 * tot / den,
                  "nbr_queen_nb_lag1_rate_per1000": 1000 * sum(lag1[j, t] for j in js) / den,
                  "nbr_queen_nb_prev3yr_rate_per1000": 1000 * sum(prev3[j, t] for j in js) / den,
                  "nbr_queen_nb_mean_rate_lag1": float(np.mean([rate_lag1[j, t] for j in js]))}
        s_ = sum(sales[j, t] for j in js) if cfg["compute_neighbour_prices"] else None
        if s_ is not None and s_ > 0:
            manual["nbr_queen_price_log_ppsqm_salewmean"] = sum(sales[j, t] * lp[j, t] for j in js) / s_
        manual["_inc"] = sum(inc_v[j] * hh[j] for j in js) / den
        row = expo[(expo.msoa11cd == ids[i]) & (expo.year == years[t])].iloc[0]
        for k, v in manual.items():
            got = st.loc[i, "nbr_queen_income_bhc2012_hhmean"] if k == "_inc" else row[k]
            ok = np.isclose(got, v, rtol=1e-10, atol=1e-10)
            checks.append({"msoa11cd": ids[i], "year": years[t], "variable": k.replace("_inc", "nbr_queen_income_bhc2012_hhmean"), "manual": v, "stored": got, "ok": bool(ok),
                           "focal_construction_excluded": ids[i] not in [ids[j] for j in js]})
        # lagged exposure must use the preceding year's construction
        if t > 0 and not np.isclose(sum(lag1[j, t] for j in js), sum(cnt[j, t - 1] for j in js)):
            return fail(f"lag1 exposure for {ids[i]} in {years[t]} is not the preceding year's neighbouring construction.")
    chk = pd.DataFrame(checks)
    chk.to_csv(QA / "manual_recomputation_checks.csv", index=False)
    if not chk.ok.all() or not chk.focal_construction_excluded.all():
        return fail(f"manual recomputation disagrees for {int((~chk.ok).sum())} checks.")
    log.info(f"Manual re-computation: {len(chk)} checks on 40 random MSOA-years all agree (rtol 1e-10).")

    # ---------------- write outputs ----------------
    qe_out = qe[["focal_msoa11cd", "neighbour_msoa11cd", "edge_type", "same_lad", "distance_m"]].copy()
    qe_out.to_parquet(G / "msoa11_spatial_neighbors_queen.parquet", compression="zstd", index=False)
    qe_out.to_csv(G / "msoa11_spatial_neighbors_queen.csv", index=False)
    qe_out.to_csv(QA / "queen_neighbour_lookup.csv", index=False)      # a trackable copy (data/interim is git-ignored) so the relationships can be inspected
    dpo = pd.DataFrame({"focal_msoa11cd": np.array(ids)[dp.i.values], "neighbour_msoa11cd": np.array(ids)[dp.j.values], "distance_m": dp.distance_m.values,
                        **{f"within_{int(r / 1000)}km": dp[f"within_{int(r / 1000)}km"].values for r in radii},
                        "weight_exp": dp.weight_exp.values, "weight_exp_rownorm": dp.weight_exp_rownorm.values})
    dpo.to_parquet(G / "msoa11_spatial_neighbors_distance.parquet", compression="zstd", index=False)
    st.to_parquet(G / "msoa11_spatial_characteristics.parquet", compression="zstd", index=False)
    expo.to_parquet(G / "msoa_year_spillover_exposure.parquet", compression="zstd", index=False)
    log.info(f"Wrote queen lookup ({len(qe_out):,} rows), distance lookup ({len(dpo):,} rows), characteristics ({len(st):,} x {st.shape[1]}), "
             f"exposure ({len(expo):,} x {expo.shape[1]}). Centroids: {centroid_note}.")

    # ---------------- QA tables ----------------
    pd.Series(n_q).describe().to_frame("queen_neighbours").join(pd.Series(n_d[5]).describe().to_frame("n_within_5km")).join(
        pd.Series(n_d[10]).describe().to_frame("n_within_10km")).to_csv(QA / "neighbour_count_distribution.csv")
    pd.DataFrame({"network": ["queen genuine", "queen island fallback", "queen cross-LAD edges", "queen directed edges", "5 km directed pairs", "10 km directed pairs",
                              "5 km zero-neighbour MSOAs", "10 km zero-neighbour MSOAs", "queen MSOAs with no cross-LAD neighbour"],
                  "count": [len(gen), len(fb), int((~qe.same_lad).sum()), len(qe), int(dp.within_5km.sum()), int(dp.within_10km.sum()),
                            int((n_d[5] == 0).sum()), int((n_d[10] == 0).sum()), int((st.nbr_queen_crosslad_n == 0).sum())]}).to_csv(QA / "network_summary.csv", index=False)
    ann = []
    for t, y in enumerate(years):
        r = {"year": y}
        for k in ("nbr_queen_nb_total", "nbr_queen_nb_lag1_rate_per1000", "nbr_5km_nb_lag1_rate_per1000", "nbr_exp10km_nb_lag1_rate_per1000"):
            v = out[k][:, t]
            r.update({f"{k}_mean": np.nanmean(v), f"{k}_median": np.nanmedian(v), f"{k}_share_zero": float(np.nanmean(v == 0))})
        r["queen_nb_total_zero_share"] = float((out["nbr_queen_nb_total"][:, t] == 0).mean())
        r["own_vs_queen_nb_lag1_rate_corr"] = float(np.corrcoef(rate_lag1[:, t], out["nbr_queen_nb_lag1_rate_per1000"][:, t])[0, 1])
        r["upstream_note"] = "completions incomplete (many EPCs lack a UPRN): 2022 partly, 2023 severely" if y >= 2022 else ""
        ann.append(r)
    pd.DataFrame(ann).to_csv(QA / "annual_exposure_summary.csv", index=False)
    miss = pd.DataFrame({"variable": expo.columns, "n_missing": expo.isna().sum().values, "share_missing": expo.isna().mean().values})
    miss.to_csv(QA / "missingness_by_variable.csv", index=False)
    cols = ["nbr_queen_nb_lag1_rate_per1000", "nbr_queen_nb_mean_rate_lag1", "nbr_5km_nb_lag1_rate_per1000", "nbr_10km_nb_lag1_rate_per1000", "nbr_exp10km_nb_lag1_rate_per1000"]
    expo.merge(nb[["msoa11cd", "year", "newbuilds_lag1_per_1000"]], on=["msoa11cd", "year"])[cols + ["newbuilds_lag1_per_1000"]].corr().to_csv(QA / "correlations_exposure_measures.csv")
    log.info("Spatial spillover preparation complete.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
