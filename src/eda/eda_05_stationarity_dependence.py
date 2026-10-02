"""EDA 05 - stationarity and cross-sectional dependence.

Why this matters in an econometric context
- If log prices are non-stationary (contain a unit root), a regression of price levels on the
  treatment can be spurious. The tests tell whether to model levels, or growth / first
  differences.
- With T = 12 (8 in the main sample) and N = 6,791, only fixed-T, large-N panel tests are
  appropriate (Harris-Tzavalis, Hadri). Both are implemented in panel_tests.py and validated
  below on simulated data (random walks and AR(0.5) series) to confirm size and power in
  this exact (N, T) configuration.
- Cross-sectional dependence (a shock hitting many MSOAs at once, or neighbouring MSOAs moving
  together) violates the independence assumed by conventional and MSOA-clustered standard errors.
  The Pesaran CD test measures it; the neighbour comparison shows whether it is spatial.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from eda_common import *  # noqa: F401,F403
import panel_tests as pt
from statsmodels.tsa.stattools import adfuller
from scipy import stats

log = get_logger("eda_05_stationarity_dependence")
style()
T = out_dir(EDA_DIR, "tables", "tests")
rng = np.random.default_rng(42)


def simulate_validation(N: int, Tn: int, reps: int = 300) -> pd.DataFrame:
    """Empirical rejection rates at the 5% level under a unit root (size) and under AR(0.5) (power)."""
    rows = []
    for name, rho in (("iid noise (rho=0)", 0.0), ("AR(0.5)", 0.5), ("random walk (rho=1)", 1.0)):
        rej_ht, rej_h, rej_htt = [], [], []
        for _ in range(reps):
            e = rng.standard_normal((N, Tn))
            y = np.zeros((N, Tn))
            y[:, 0] = e[:, 0]
            for t in range(1, Tn):
                y[:, t] = rho * y[:, t - 1] + e[:, t]
            rej_ht.append(pt.harris_tzavalis(y)["p_value"] < .05)
            yt = y + 0.3 * np.arange(Tn)[None, :] + rng.standard_normal((N, 1))        # unit effects + common drift
            rej_htt.append(pt.harris_tzavalis(yt, trend=True)["p_value"] < .05)
            rej_h.append(pt.hadri_lm(y)["p_value"] < .05)
        rows.append({"DGP": name, "N": N, "T_obs": Tn, "HT_rejection_rate_5pct": np.mean(rej_ht),
                     "HT_trend_rejection_rate_5pct (series has drift)": np.mean(rej_htt),
                     "Hadri_rejection_rate_5pct": np.mean(rej_h)})
    return pd.DataFrame(rows)


def main() -> int:
    df = load_panel()

    # ---- 0. Validate the implementations by simulation ------------------------------------------------
    # WHAT: simulate N = 2,000 units for T = 12 and T = 8 time periods. HT should reject a true unit root (random
    #       walk) about 5% of the time and reject stationary series often; Hadri should reject iid noise about 5% of the
    #       time and a random walk often. Hadri is NOT designed for serially correlated stationary series, so its
    #       rejection of AR(0.5) is expected and shown deliberately.
    # LOOK FOR: rejection rates for the true nulls near 0.05. Rates far from that mean the implementation or the
    #           finite-T approximation cannot be relied on for the real-data tests below.
    val = pd.concat([simulate_validation(2000, 12), simulate_validation(2000, 8)])
    val.to_csv(T / "05_simulation_validation.csv", index=False)
    log.info("Simulation validation:\n" + val.round(3).to_string())

    # ---- 1. Stationarity tests ----------------------------------------------------------------------------
    # WHAT: Harris-Tzavalis (H0: unit root), Hadri (H0: stationary), and a Fisher-type combination of
    #       per-MSOA ADF tests, applied to log price, annual price growth and the treatment; for the full panel
    #       (T = 12) and the main sample (T = 8); each with and without removing cross-sectional (year) means,
    #       which reduces the effect of a common shock on the tests.
    # LOOK FOR: whether HT rejects the unit root and Hadri fails to reject (agreement => stationary) or the
    #           reverse; differences once year means are removed; the ADF/Fisher result is shown for completeness but has
    #           low power and approximate p-values at T = 8-12 and should be given the least weight.
    rows = []
    series = {"log_median_ppsqm": OUTCOME, "d_log_price": "d_log_price", "newbuilds_per_1000": "newbuilds_per_1000"}
    for label, d in (("full_2012_2023", df), ("main_2016_2023", df[df.main_sample])):
        for sname, col in series.items():
            P = d.pivot(index="msoa11cd", columns="year", values=col).dropna(axis=1, how="any").dropna(axis=0, how="any")
            for yearmean in (False, True):
                Y = P.values - (P.values.mean(axis=0, keepdims=True) if yearmean else 0)
                ht = pt.harris_tzavalis(Y)
                htt = pt.harris_tzavalis(Y, trend=True)
                had = pt.hadri_lm(Y)
                rows.append({"sample": label, "series": sname, "cross_section_means_removed": yearmean, "N": Y.shape[0],
                             "T_obs": Y.shape[1], "HT_rho": ht["rho"], "HT_z": ht["z"], "HT_p(H0 unit root)": ht["p_value"],
                             "HTtrend_rho": htt["rho"], "HTtrend_z": htt["z"], "HTtrend_p(H0 unit root)": htt["p_value"],
                             "Hadri_z": had["z"], "Hadri_p(H0 stationary)": had["p_value"],
                             "Hadri_N_non_constant": had["N"]})
    st = pd.DataFrame(rows)
    # Fisher-type (Choi inverse-normal) combination of per-MSOA ADF with 0 augmenting lags, constant
    ff = []
    for label, d in (("full_2012_2023", df), ("main_2016_2023", df[df.main_sample])):
        for sname, col in series.items():
            P = d.pivot(index="msoa11cd", columns="year", values=col).dropna(axis=0, how="any")
            n_const = int((P.nunique(axis=1) == 1).sum())          # constant series cannot be ADF-tested
            P = P[P.nunique(axis=1) > 1]
            pv = np.array([adfuller(r, maxlag=0, regression="c", autolag=None)[1] for r in P.values])
            pv = np.clip(pv, 1e-10, 1 - 1e-10)
            z = stats.norm.ppf(pv).sum() / np.sqrt(len(pv))
            ff.append({"sample": label, "series": sname, "N": len(pv), "n_constant_series_excluded": n_const, "ADF_share_reject_5pct": (pv < .05).mean(),
                       "Fisher_inverse_normal_Z": z, "Fisher_p": stats.norm.cdf(z)})
    st.to_csv(T / "05_stationarity_tests.csv", index=False)
    pd.DataFrame(ff).to_csv(T / "05_stationarity_fisher_adf.csv", index=False)
    log.info("Stationarity tests:\n" + st.round(4).to_string())

    # ---- 2. Cross-sectional dependence (Pesaran CD) ---------------------------------------------------------------
    # WHAT: Pesaran CD statistic and the average pairwise correlation across the 6,791 MSOAs for the demeaned series
    #       (MSOA means removed), then also with year means removed, region x year means removed and LAD x year means
    #       removed. The average correlation among geographic NEIGHBOURS is compared with that among all pairs.
    # LOOK FOR: how much cross-sectional dependence is left after each layer of common effects (CD near 0 = independent);
    #           whether neighbouring MSOAs are more correlated than random pairs after those effects are removed (spatial
    #           dependence that LAD x year effects may not fully remove).
    w, _ = queen_weights()
    ids = sorted(df.msoa11cd.unique())
    pos = {m: i for i, m in enumerate(ids)}
    pair_i = np.array([pos[a] for a, nbrs in w.neighbors.items() for b in nbrs if pos[a] < pos[b]])
    pair_j = np.array([pos[b] for a, nbrs in w.neighbors.items() for b in nbrs if pos[a] < pos[b]])
    rows = []
    for label, d in (("main_2016_2023", df[df.main_sample]), ("full_2012_2023", df)):
        for sname, col in series.items():
            M = d.pivot(index="msoa11cd", columns="year", values=col).loc[ids].dropna(axis=1, how="any")
            Mv = M.values
            lad = df.drop_duplicates("msoa11cd").set_index("msoa11cd").loc[ids]
            variants = {"MSOA means removed": Mv - Mv.mean(axis=1, keepdims=True)}
            e = variants["MSOA means removed"]
            variants["+ year means removed"] = e - e.mean(axis=0, keepdims=True)
            for gname, gcol in (("+ region x year", "region_code"), ("+ LAD x year", "lad23cd_analysis")):
                codes = lad[gcol].values
                r = e.copy()
                for g in np.unique(codes):
                    m = codes == g
                    r[m] = e[m] - e[m].mean(axis=0, keepdims=True)
                if gcol == "lad23cd_analysis":
                    # in a LAD with k MSOAs, demeaning forces the k residuals to sum to zero (mechanically negative
                    # correlation, very strong for k = 2), so only LADs with at least 10 MSOAs are used here
                    size = pd.Series(codes).map(pd.Series(codes).value_counts()).values
                    r[size < 10] = 0.0
                variants[gname + " means removed (LADs with >= 10 MSOAs)" if gcol == "lad23cd_analysis"
                         else gname + " means removed"] = r
            for vname, E in variants.items():
                cd = pt.pesaran_cd(E)
                X = E - E.mean(axis=1, keepdims=True)
                nrm = np.sqrt((X ** 2).sum(axis=1, keepdims=True))
                valid = nrm[:, 0] > 1e-10
                X = X / np.where(nrm > 1e-10, nrm, 1.0)
                both = valid[pair_i] & valid[pair_j]
                nb = float((X[pair_i[both]] * X[pair_j[both]]).sum(axis=1).mean())
                rows.append({"sample": label, "series": sname, "effects_removed": vname, "N": cd["N"], "T": E.shape[1],
                             "CD": cd["CD"], "CD_p": cd["p_value"], "mean_pairwise_corr_all": cd["mean_pairwise_corr"],
                             "mean_abs_pairwise_corr_all": cd["mean_abs_pairwise_corr"], "mean_corr_neighbours": nb})
    cdr = pd.DataFrame(rows)
    cdr.to_csv(T / "05_cross_sectional_dependence.csv", index=False)
    log.info("CD tests (main sample):\n" + cdr[cdr["sample"] == "main_2016_2023"].round(4).to_string())
    log.info("EDA 05 complete.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
