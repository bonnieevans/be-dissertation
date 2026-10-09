"""Pure functions for the spatial spillover exposure measures (no file I/O), so they can be tested on small synthetic graphs.

Conventions
- MSOAs are indexed 0..N-1 in a fixed order; W is an N x N sparse matrix with W[i, j] > 0 if j is a neighbour of focal MSOA i.
  The focal MSOA is never its own neighbour (zero diagonal).
- Counts and rates are N x T arrays (MSOA x year). Denominators are per-MSOA baseline households (length N).
- A rate whose neighbour denominator is zero (no eligible neighbours) is NaN: it is "not defined", never a genuine zero.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.spatial import cKDTree


def edges_to_matrix(focal_idx, neigh_idx, n: int, weights=None) -> sp.csr_matrix:
    """Sparse N x N matrix from directed edges (focal row, neighbour column). Duplicated directed pairs are an error."""
    focal_idx, neigh_idx = np.asarray(focal_idx), np.asarray(neigh_idx)
    if (focal_idx == neigh_idx).any():
        raise ValueError("an MSOA is listed as its own neighbour")
    pairs = pd.MultiIndex.from_arrays([focal_idx, neigh_idx])
    if pairs.duplicated().any():
        raise ValueError("duplicated directed focal-neighbour pairs")
    w = np.ones(len(focal_idx)) if weights is None else np.asarray(weights, dtype=float)
    return sp.csr_matrix((w, (focal_idx, neigh_idx)), shape=(n, n))


def neighbour_sum(W: sp.spmatrix, X: np.ndarray) -> np.ndarray:
    """sum_j W_ij * X_j for each focal i (X is N or N x T)."""
    return np.asarray(W @ X)


def pooled_rate(W: sp.spmatrix, counts: np.ndarray, households: np.ndarray, per: float = 1000.0) -> np.ndarray:
    """per * sum_j W_ij counts_j / sum_j W_ij households_j. NaN where the weighted denominator is zero.
    counts: N or N x T; households: N."""
    num = neighbour_sum(W, counts)
    den = np.asarray(W @ households, dtype=float)
    den_b = den.reshape(-1, *([1] * (num.ndim - 1)))
    with np.errstate(divide="ignore", invalid="ignore"):
        out = np.where(den_b > 0, per * num / np.where(den_b > 0, den_b, 1.0), np.nan)
    return out


def neighbour_mean(W: sp.spmatrix, X: np.ndarray) -> np.ndarray:
    """Unweighted (binary W) or W-weighted arithmetic mean of X over neighbours; NaN where there are none. Equals the row-standardised
    spatial lag used by the preliminary model-lab spillover regression when W is the binary queen matrix."""
    s = np.asarray(W.sum(axis=1)).ravel()
    num = np.asarray(W @ X)
    sb = s.reshape(-1, *([1] * (num.ndim - 1)))
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(sb > 0, num / np.where(sb > 0, sb, 1.0), np.nan)


def household_weighted_mean(W: sp.spmatrix, values: np.ndarray, households: np.ndarray) -> np.ndarray:
    """sum_j W_ij v_j h_j / sum_j W_ij h_j (NaN if no neighbours). The household-weighted mean of the LEVEL (e.g. income); take logs
    afterwards if the log of the weighted mean is wanted (not the weighted mean of logs)."""
    num = np.asarray(W @ (values * households), dtype=float)
    den = np.asarray(W @ households, dtype=float)
    return np.where(den > 0, num / np.where(den > 0, den, 1.0), np.nan)


def sale_weighted_price(W: sp.spmatrix, log_price: np.ndarray, sales: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Sale-count-weighted mean of MSOA log median prices over neighbours with a valid price and a positive sale count.
    Returns (mean, n_valid_neighbours, total_sales). Missing prices are excluded, never treated as zero. N x T inputs."""
    valid = np.isfinite(log_price) & np.isfinite(sales) & (sales > 0)
    s = np.where(valid, sales, 0.0)
    lp = np.where(valid, log_price, 0.0)
    num = np.asarray(W @ (s * lp))
    den = np.asarray(W @ s)
    n_valid = np.asarray(W @ valid.astype(float))        # counts neighbours when W is binary (queen network)
    mean = np.where(den > 0, num / np.where(den > 0, den, 1.0), np.nan)
    return mean, n_valid, den


def distance_pairs(xy: np.ndarray, radius: float) -> pd.DataFrame:
    """All ordered pairs (i, j), i != j, whose Euclidean distance (projected metres) is <= radius, via a KD-tree (no N x N matrix)."""
    tree = cKDTree(xy)
    coo = tree.sparse_distance_matrix(tree, radius, output_type="coo_matrix")
    keep = coo.row != coo.col
    return pd.DataFrame({"i": coo.row[keep].astype(int), "j": coo.col[keep].astype(int), "distance_m": coo.data[keep]})


def exp_decay(distance_m, decay_m: float) -> np.ndarray:
    return np.exp(-np.asarray(distance_m, dtype=float) / decay_m)


def row_normalise(W: sp.csr_matrix) -> sp.csr_matrix:
    s = np.asarray(W.sum(axis=1)).ravel()
    inv = np.where(s > 0, 1.0 / np.where(s > 0, s, 1.0), 0.0)
    return sp.diags(inv) @ W


def second_order(W1: sp.csr_matrix) -> sp.csr_matrix:
    """Binary matrix of neighbours of neighbours, excluding the focal MSOA and all first-order neighbours."""
    A = (W1 > 0).astype(int)
    A2 = (A @ A > 0).astype(int)
    out = A2.tolil()
    out.setdiag(0)
    out = out.tocsr()
    out = out - out.multiply(A)
    out.eliminate_zeros()
    return (out > 0).astype(float).tocsr()
