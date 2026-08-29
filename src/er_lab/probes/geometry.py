"""Embedding-geometry panel: alignment, uniformity, effective rank, hubness (PLAN.md notebook 08).

These are the measurable embedding properties the lab's pressure -> property ->
metric mediation chains run through (PLAN §5): each training pressure is a
pre-registered conjecture about moving one of these numbers. All functions are
pure numpy on 2-d arrays (torch tensors are accepted and detached to numpy);
deterministic, no randomness.

References: alignment/uniformity — Wang & Isola, ICML 2020 ("Understanding
Contrastive Representation Learning through Alignment and Uniformity on the
Hypersphere"); RankMe — Garrido et al., ICML 2023; hubness — Radovanović,
Nanopoulos & Ivanović, JMLR 2010 (k-occurrence skewness); SCL-02 uses the
hubness panel for the frequent-name "wolf record" question.
"""

from __future__ import annotations

import numpy as np
from scipy.special import logsumexp
from scipy.stats import skew

__all__ = ["alignment", "hubness", "rankme", "uniformity"]

#: hard row cap for the exact brute-force hubness computation.
HUBNESS_MAX_ROWS = 20_000


def _as_2d(x: object, name: str) -> np.ndarray:
    if hasattr(x, "detach"):  # torch tensor (torch optional: never imported here)
        x = x.detach().cpu().numpy()
    arr = np.asarray(x, dtype=np.float64)
    if arr.ndim != 2:
        raise ValueError(f"{name} must be a 2-d array (n_rows, dim), got shape {arr.shape}")
    if arr.shape[0] == 0:
        raise ValueError(f"{name} must have at least one row")
    if not np.isfinite(arr).all():
        raise ValueError(f"{name} contains non-finite values")
    return arr


def _normalize_rows(arr: np.ndarray) -> np.ndarray:
    return arr / np.maximum(np.linalg.norm(arr, axis=1, keepdims=True), 1e-12)


def alignment(pos_a: np.ndarray, pos_b: np.ndarray) -> float:
    """Wang-Isola alignment: mean squared distance between normalized positives.

    ``pos_a[i]`` and ``pos_b[i]`` are the embeddings of the i-th positive pair
    (two serializations of the same person). Rows are L2-normalized, then
    ``mean_i || a_i - b_i ||^2`` (the alpha=2 case of Wang & Isola 2020, eq. 3).
    Range [0, 4]: 0 = every positive pair maps to the identical direction (perfect
    invariance); lower is better-aligned.
    """
    a = _normalize_rows(_as_2d(pos_a, "pos_a"))
    b = _normalize_rows(_as_2d(pos_b, "pos_b"))
    if a.shape != b.shape:
        raise ValueError(f"positive pair arrays must share a shape, got {a.shape} vs {b.shape}")
    return float(np.mean(np.sum((a - b) ** 2, axis=1)))


def uniformity(X: np.ndarray, t: float = 2.0) -> float:
    """Wang-Isola uniformity: log mean Gaussian potential over normalized rows.

    ``log E_{i != j} exp(-t || x_i - x_j ||^2)`` with rows L2-normalized
    (Wang & Isola 2020, eq. 4), computed in log-space (logsumexp) over all
    unordered pairs. Always <= 0; 0 when all points collapse to one direction,
    more negative the more uniformly the points spread on the hypersphere
    (minimized by the uniform distribution). O(n^2) memory — intended for probe
    samples, not full corpora.
    """
    x = _normalize_rows(_as_2d(X, "X"))
    n = x.shape[0]
    if n < 2:
        raise ValueError("uniformity needs at least 2 rows")
    if t <= 0:
        raise ValueError(f"t must be positive, got {t}")
    gram = x @ x.T
    sq_dist = np.clip(2.0 - 2.0 * gram, 0.0, None)  # ||u - v||^2 = 2 - 2 u.v on the sphere
    iu, ju = np.triu_indices(n, k=1)
    return float(logsumexp(-t * sq_dist[iu, ju]) - np.log(iu.size))


def rankme(X: np.ndarray) -> float:
    """RankMe effective rank: exp of the entropy of the normalized singular values.

    With singular values sigma_k of ``X`` and p_k = sigma_k / sum(sigma),
    ``rankme = exp(-sum_k p_k log p_k)`` (Garrido et al., ICML 2023). Range
    [1, min(n, d)]: ~1 for a rank-1 (collapsed) embedding, ~min(n, d) when the
    spectrum is flat (e.g. i.i.d. Gaussian rows with n >> d). Rows are used as
    given (no centering or normalization), matching the RankMe definition.
    """
    x = _as_2d(X, "X")
    sigma = np.linalg.svd(x, compute_uv=False)
    total = float(sigma.sum())
    if total <= 0.0:
        raise ValueError("rankme is undefined for an all-zero matrix")
    p = sigma / total
    nz = p[p > 0]
    return float(np.exp(-np.sum(nz * np.log(nz))))


def hubness(X: np.ndarray, k: int = 10) -> dict[str, float]:
    """k-occurrence hubness of the cosine k-NN graph (Radovanović et al. 2010).

    ``N_k(j)`` counts how many other rows have row j among their k nearest
    neighbors by cosine similarity (self excluded). Returns::

        {"skewness": sample skewness of the N_k distribution,
         "max_k_occurrence": max_j N_k(j)}

    Positive skew means hub records exist — rows that appear in many neighbor
    lists (mean N_k is always exactly k, so a heavy right tail is the signature);
    for i.i.d. low-dimensional data skew is mild, and it grows with intrinsic
    dimensionality and with planted hub geometry. A constant N_k distribution
    (every row equally k-occurring — e.g. a corpus made of same-size groups of
    exact-duplicate serializations) carries no hubness signal, so its skewness
    is reported as 0.0 by definition (scipy's sample skewness would emit a
    precision-loss warning and return NaN there). Exact brute-force (sklearn
    NearestNeighbors, algorithm='brute'): allowed only up to
    ``HUBNESS_MAX_ROWS`` (20k) rows, raises above — use a sampled subset at scale.
    """
    x = _as_2d(X, "X")
    n = x.shape[0]
    if n > HUBNESS_MAX_ROWS:
        raise ValueError(
            f"hubness is exact brute-force and capped at {HUBNESS_MAX_ROWS} rows; "
            f"got {n} — pass a sampled subset"
        )
    if not 0 < k < n:
        raise ValueError(f"k must satisfy 0 < k < n_rows ({n}), got {k}")
    from sklearn.neighbors import NearestNeighbors

    nn = NearestNeighbors(n_neighbors=k + 1, algorithm="brute", metric="cosine").fit(x)
    neighbor_idx = nn.kneighbors(x, return_distance=False)
    counts = np.zeros(n, dtype=np.int64)
    for i, row in enumerate(neighbor_idx):
        kept = 0
        for j in row:  # drop self; with exact duplicates self may not be first
            if j == i:
                continue
            counts[j] += 1
            kept += 1
            if kept == k:
                break
    if np.ptp(counts) == 0:
        # constant N_k: the most hub-free outcome possible — report 0 skew
        # instead of scipy's NaN-with-catastrophic-cancellation-warning
        skewness = 0.0
    else:
        skewness = float(skew(counts))
    return {"skewness": skewness, "max_k_occurrence": int(counts.max())}
