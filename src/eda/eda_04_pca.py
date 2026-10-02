"""EDA 04 - principal component analysis.

Why this matters in an econometric context
- PCA of the baseline moderators shows how many distinct dimensions of neighbourhood
  socioeconomic status the six moderators contain. Strongly overlapping moderators give
  highly correlated interaction terms (multicollinearity); PCA quantifies this and gives
  a fallback of orthogonal composite moderators.
- PCA of the 12-year price paths (6,791 x 12) shows how many common time factors drive
  prices. A single dominant factor is what year fixed effects remove; further factors
  with heterogeneous loadings across MSOAs are what interactive fixed effects or
  region x year effects would be needed for. The number of factors is also estimated with
  the Bai-Ng (2002) information criteria.
- PCA does not use the outcome or the treatment, so it describes structure only.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from eda_common import *  # noqa: F401,F403
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

log = get_logger("eda_04_pca")
style()
T = out_dir(EDA_DIR, "tables", "pca")
F = out_dir(EDA_DIR, "figures", "pca")


def scree(ev: np.ndarray, title: str, path: Path, n: int | None = None) -> None:
    n = n or len(ev)
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.8))
    ax[0].bar(range(1, n + 1), ev[:n])
    ax[0].axhline(1, color="r", lw=.7, ls="--")
    ax[0].set_title(f"{title}: eigenvalues (red line = 1)")
    ax[1].plot(range(1, n + 1), np.cumsum(ev[:n]) / ev.sum(), marker="o")
    ax[1].set_title("Cumulative share of variance")
    ax[1].set_ylim(0, 1.02)
    savefig(fig, path)


def biplot(scores, loadings, names, evr, path, title, color=None) -> None:
    fig, ax = plt.subplots(figsize=(7.5, 6.5))
    ax.scatter(scores[:, 0], scores[:, 1], s=3, alpha=.25, c=color if color is not None else "grey")
    sc = np.abs(scores[:, :2]).max() / np.abs(loadings[:, :2]).max() * .8
    for (x, y), nm in zip(loadings[:, :2], names):
        ax.arrow(0, 0, x * sc, y * sc, color="r", head_width=.1)
        ax.text(x * sc * 1.08, y * sc * 1.08, nm, fontsize=7, color="r")
    ax.set_xlabel(f"PC1 ({evr[0]:.0%})")
    ax.set_ylabel(f"PC2 ({evr[1]:.0%})")
    ax.set_title(title)
    savefig(fig, path)


def bai_ng(X: np.ndarray, kmax: int = 6) -> pd.DataFrame:
    """Bai & Ng (2002) IC_p1, IC_p2, IC_p3 for the number of factors in a T x N panel X (N units,
    T periods), after removing unit and time means."""
    X = X - X.mean(axis=0, keepdims=True) - X.mean(axis=1, keepdims=True) + X.mean()
    T_, N = X.shape
    U, s, Vt = np.linalg.svd(X, full_matrices=False)
    rows = []
    for k in range(0, kmax + 1):
        Xhat = (U[:, :k] * s[:k]) @ Vt[:k] if k else np.zeros_like(X)
        V = ((X - Xhat) ** 2).sum() / (N * T_)
        pen1 = k * (N + T_) / (N * T_) * np.log(N * T_ / (N + T_))
        pen2 = k * (N + T_) / (N * T_) * np.log(min(N, T_))
        pen3 = k * np.log(min(N, T_)) / min(N, T_)
        rows.append({"k_factors": k, "V(k)": V, "IC_p1": np.log(V) + pen1, "IC_p2": np.log(V) + pen2, "IC_p3": np.log(V) + pen3})
    return pd.DataFrame(rows)


def main() -> int:
    df = load_panel()
    base = load_baseline()

    # ---- 1. PCA of the six baseline moderators -----------------------------------------------
    # WHAT: PCA on the standardised moderators (MSOA level, n = 6,791); eigenvalues, loadings, and a biplot
    #       coloured by the deprivation moderator. Run for the four primary moderators and for all six.
    # LOOK FOR: the number of components with eigenvalue above 1 and the cumulative variance explained; which
    #           variables load on the same component with the same sign (they carry overlapping information);
    #           the loadings of the "income" and "deprivation" variables on PC1.
    mod_names = list(MODERATORS)
    X6 = base[[MODERATORS[k] for k in mod_names]].copy()
    X6.columns = mod_names
    for label, cols in (("six_moderators", mod_names), ("four_primary_moderators", PRIMARY_MODERATORS)):
        Xs = StandardScaler().fit_transform(X6[cols])
        p = PCA().fit(Xs)
        ev = p.explained_variance_
        pd.DataFrame({"component": [f"PC{i + 1}" for i in range(len(ev))], "eigenvalue": ev,
                      "share_variance": p.explained_variance_ratio_, "cumulative": np.cumsum(p.explained_variance_ratio_)}
                     ).to_csv(T / f"04_eigenvalues_{label}.csv", index=False)
        load = pd.DataFrame(p.components_.T, index=cols, columns=[f"PC{i + 1}" for i in range(len(cols))])
        load.to_csv(T / f"04_loadings_{label}.csv")
        scree(ev, label.replace("_", " "), F / f"scree_{label}.png")
        scores = p.transform(Xs)
        biplot(scores, p.components_.T * np.sqrt(ev), cols, p.explained_variance_ratio_, F / f"biplot_{label}.png",
               f"Moderators PCA ({label.replace('_', ' ')}), coloured by deprivation", color=X6["deprivation"])
        pd.DataFrame(scores[:, :3], columns=["PC1", "PC2", "PC3"]).assign(msoa11cd=base["msoa11cd"].values
                                                                          ).to_csv(T / f"04_scores_{label}.csv", index=False)
        log.info(f"{label}: eigenvalues {np.round(ev, 2).tolist()}; PC1 loadings {load['PC1'].round(2).to_dict()}")

    # ---- 2. PCA of all baseline socioeconomic variables including IMD domains -------------------------
    # WHAT: PCA on moderators, the seven IMD domain scores, the IMD index variants, and the Census tenure shares.
    # LOOK FOR: how many dimensions all of these reduce to; whether income, employment, education and health
    #           domains load together (they sit in the same index) and where social-rent share and density fall.
    dom = [c for c in base.columns if c.startswith("imd2015_") and c.endswith(("_pw", "_msoa")) and "legacy" not in c
           and "bottom20" not in c and "overall" not in c]
    cols = list(MODERATORS.values()) + dom + ["imd_ex_housing", "owner_share_2011", "private_rent_share_2011"]
    cols = [c for c in dict.fromkeys(cols) if c in base.columns]
    Xa = StandardScaler().fit_transform(base[cols])
    pa = PCA().fit(Xa)
    ev = pa.explained_variance_
    pd.DataFrame({"component": [f"PC{i + 1}" for i in range(len(ev))], "eigenvalue": ev, "share_variance": pa.explained_variance_ratio_,
                  "cumulative": np.cumsum(pa.explained_variance_ratio_)}).to_csv(T / "04_eigenvalues_all_baseline.csv", index=False)
    pd.DataFrame(pa.components_.T, index=cols, columns=[f"PC{i + 1}" for i in range(len(cols))]).to_csv(T / "04_loadings_all_baseline.csv")
    scree(ev, "all baseline socioeconomic variables", F / "scree_all_baseline.png")
    biplot(pa.transform(Xa), pa.components_.T * np.sqrt(ev), cols, pa.explained_variance_ratio_, F / "biplot_all_baseline.png",
           "All baseline variables PCA, coloured by deprivation", color=base["deprivation_moderator_value"])

    # ---- 3. PCA of the price paths (common time factors) ----------------------------------------------------
    # WHAT: principal components of the 6,791 x 12 matrix of log prices (and of annual price growth), after removing
    #       each MSOA's own mean, with and without also removing the year means. Plots the factor time paths (loadings
    #       on years), correlates the MSOA scores on the remaining factors with the moderators, and reports Bai-Ng
    #       information criteria.
    # LOOK FOR: the share of variance in the first factor before and after removing year means (removing year means
    #           is what year fixed effects do), the shape of the factor time path, and the number of factors the
    #           information criteria select; whether the MSOA scores on later factors line up with the moderators.
    P = df.pivot(index="year", columns="msoa11cd", values=OUTCOME)       # T x N
    G = df.pivot(index="year", columns="msoa11cd", values="d_log_price").iloc[1:]
    rows = []

    def svd_factors(X: pd.DataFrame):
        """Uncentred SVD of an N x T matrix (MSOAs x years). Uncentred so that a factor common to every
        MSOA (the year effect) shows up as a factor, unlike sklearn's PCA which removes year means."""
        U, sv, Vt = np.linalg.svd(X.values, full_matrices=False)
        return sv ** 2 / (sv ** 2).sum(), Vt, U * sv

    for nm, M in (("log_price", P), ("price_growth", G)):
        Xm = (M - M.mean(axis=0)).T                         # N x T, own mean removed
        Xy = Xm.sub(Xm.mean(axis=0), axis=1)                # also year means removed
        for tag, X in (("demean_msoa", Xm), ("demean_msoa_and_year", Xy)):
            share, Vt, sc = svd_factors(X)
            rows += [{"series": nm, "preprocessing": tag, "component": i + 1, "share_variance": v}
                     for i, v in enumerate(share[:6])]
            if tag == "demean_msoa":
                fig, ax = plt.subplots(figsize=(7, 4))
                for i in range(3):
                    ax.plot(X.columns, Vt[i], marker="o", label=f"factor {i + 1} ({share[i]:.0%})")
                ax.set_title(f"Common time factors of {nm} (MSOA means removed, year means kept)")
                ax.legend()
                savefig(fig, F / f"price_paths_factors_{nm}.png")
            if tag == "demean_msoa_and_year":
                fig, ax = plt.subplots(figsize=(7, 4))
                for i in range(3):
                    ax.plot(X.columns, Vt[i], marker="o", label=f"factor {i + 1} ({share[i]:.0%})")
                ax.set_title(f"Remaining factors of {nm} after removing MSOA and year means")
                ax.legend()
                savefig(fig, F / f"price_paths_factors_after_year_means_{nm}.png")
                sdf = pd.DataFrame({"msoa11cd": X.index, "f1": sc[:, 0], "f2": sc[:, 1], "f3": sc[:, 2]})
                sdf.to_csv(T / f"04_price_factor_scores_after_year_means_{nm}.csv", index=False)
                mm = sdf.merge(base[["msoa11cd"] + list(MODERATORS.values())], on="msoa11cd")
                cor = pd.DataFrame({k: mm[["f1", "f2", "f3"]].corrwith(mm[col]) for k, col in MODERATORS.items()})
                cor.to_csv(T / f"04_price_factor_score_correlation_with_moderators_{nm}.csv")
    pd.DataFrame(rows).to_csv(T / "04_price_path_pca_variance.csv", index=False)
    pd.concat([bai_ng(P.values).assign(series="log_price"), bai_ng(G.values).assign(series="price_growth")]).to_csv(
        T / "04_bai_ng_number_of_factors.csv", index=False)
    log.info("EDA 04 complete.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
