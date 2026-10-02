"""Panel time-series tests implemented from the published formulas (no standard Python
implementation exists for balanced short-T panels). Each is validated by simulation in
eda_05_stationarity_dependence.py before use.

- Harris & Tzavalis (1999) unit-root test: H0 all panels have a unit root; fixed T, N -> inf.
- Hadri (2000) LM test: H0 all panels are (level) stationary.
- Pesaran (2004) CD test: H0 cross-sectional independence of the errors.
"""

from __future__ import annotations

import numpy as np
from scipy import stats


def harris_tzavalis(Y: np.ndarray, trend: bool = False) -> dict:
    """Y is N x (T+1): observations t = 0..T for each of N units (balanced).
    trend=False: unit-specific intercepts. Statistic (HT 1999):
        sqrt(N) (rho_hat - (1 - 3/(T+1))) / sqrt(3 (17 T^2 - 20 T + 17) / (5 (T-1) (T+1)^3)).
    trend=True: unit-specific intercepts and linear trends (needed when the series drifts, as log prices do):
        mean 1 - 15/(2 (T+2)), variance 15 (193 T^2 - 728 T + 1147) / (112 (T-2) (T+2)^3).
    Both are checked against simulated random walks (with drift for the trend case) in eda_05."""
    N, T1 = Y.shape
    T = T1 - 1
    y, ylag = Y[:, 1:], Y[:, :-1]
    if not trend:
        yd = y - y.mean(axis=1, keepdims=True)
        ld = ylag - ylag.mean(axis=1, keepdims=True)
        mean = 1 - 3 / (T + 1)
        var = 3 * (17 * T ** 2 - 20 * T + 17) / (5 * (T - 1) * (T + 1) ** 3)
    else:
        t = np.arange(1, T + 1, dtype=float)
        Z = np.column_stack([np.ones(T), t])
        Pz = np.eye(T) - Z @ np.linalg.inv(Z.T @ Z) @ Z.T          # residual-maker for constant + trend
        yd, ld = y @ Pz, ylag @ Pz
        mean = 1 - 15 / (2 * (T + 2))
        var = 15 * (193 * T ** 2 - 728 * T + 1147) / (112 * (T - 2) * (T + 2) ** 3)
    rho = (yd * ld).sum() / (ld ** 2).sum()
    z = np.sqrt(N) * (rho - mean) / np.sqrt(var)
    return {"rho": rho, "z": z, "p_value": stats.norm.cdf(z), "N": N, "T_transitions": T}


_HADRI_CACHE: dict = {}


def _hadri_null_moments(T: int, reps: int = 400_000) -> tuple[float, float]:
    """Mean and variance of the unit-level Hadri LM statistic under H0 (iid deviations around a unit mean,
    unit-specific variance estimate) for a finite T, by Monte Carlo. The asymptotic moments (1/6, 1/45)
    are for T -> infinity and are not accurate at T = 8-12."""
    if T not in _HADRI_CACHE:
        r = np.random.default_rng(12345)
        e = r.standard_normal((reps, T))
        e = e - e.mean(axis=1, keepdims=True)
        S = np.cumsum(e, axis=1)
        lm_i = (S ** 2).sum(axis=1) / T ** 2 / ((e ** 2).sum(axis=1) / T)
        _HADRI_CACHE[T] = (lm_i.mean(), lm_i.var())
    return _HADRI_CACHE[T]


def hadri_lm(Y: np.ndarray) -> dict:
    """Hadri (2000) LM test of H0: every unit is stationary around its own mean, with iid deviations
    (unit-specific variances). Y is N x T. Uses finite-T null moments from simulation instead of the
    asymptotic 1/6 and 1/45. NOT robust to serial correlation in the deviations: a persistent but stationary
    series (e.g. AR(0.5)) is rejected, so rejection should be read together with the Harris-Tzavalis result."""
    N, T = Y.shape
    e = Y - Y.mean(axis=1, keepdims=True)
    sig2 = (e ** 2).sum(axis=1, keepdims=True) / T
    ok = sig2[:, 0] > 0
    S = np.cumsum(e[ok], axis=1)
    lm_i = ((S ** 2 / sig2[ok]).sum(axis=1)) / T ** 2
    m, v = _hadri_null_moments(T)
    z = np.sqrt(ok.sum()) * (lm_i.mean() - m) / np.sqrt(v)
    return {"LM": lm_i.mean(), "null_mean": m, "z": z, "p_value": 1 - stats.norm.cdf(z), "N": int(ok.sum()), "T": T}


def pesaran_cd(E: np.ndarray, chunk: int = 800) -> dict:
    """Pesaran (2004) CD test. E is N x T residuals (unit x time). CD = sqrt(2T / (N (N-1))) * sum_{i<j} rho_ij,
    where rho_ij is the pairwise correlation of the T residuals. Also returns the average pairwise correlation
    and the average absolute correlation."""
    X = E - E.mean(axis=1, keepdims=True)
    keep = (X ** 2).sum(axis=1) > 1e-20                 # units with no variation have no correlation
    X = X[keep]
    N, T = X.shape
    X = X / np.sqrt((X ** 2).sum(axis=1, keepdims=True))
    tot, tot_abs = 0.0, 0.0
    for i in range(0, N, chunk):
        C = X[i:i + chunk] @ X.T                      # correlations of this block with all units
        idx = np.arange(i, min(i + chunk, N))
        C[np.arange(len(idx)), idx] = 0.0             # drop self-correlation
        # upper triangle only: columns j > row index
        mask = np.arange(N)[None, :] > idx[:, None]
        tot += C[mask].sum()
        tot_abs += np.abs(C[mask]).sum()
    npairs = N * (N - 1) / 2
    cd = np.sqrt(2 * T / (N * (N - 1))) * tot
    return {"CD": cd, "p_value": 2 * (1 - stats.norm.cdf(abs(cd))), "mean_pairwise_corr": tot / npairs,
            "mean_abs_pairwise_corr": tot_abs / npairs, "N": N, "T": T}
