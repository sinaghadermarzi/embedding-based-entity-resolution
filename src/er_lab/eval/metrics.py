"""Entity-level and pairwise ER metrics (PLAN.md §5; lit_review §4, MET-01/02).

Shared representation: a clustering is a ``pandas.Series`` indexed by record_id
(strings) with hashable cluster labels; truth is a same-shape Series of entity
ids. Every metric aligns pred and truth on their index and raises on any
mismatch (different id sets, duplicate ids, missing labels) — silent alignment
bugs are how evaluations lie.

Frequency weights: every partition metric takes an optional keyword-only
``weights`` Series (record_id -> non-negative count). A record with weight w
behaves exactly as if it were physically duplicated w times (weight 0 = drop);
integer weights therefore reproduce the metric on a physically-duplicated
frame bit-for-bit in exact arithmetic. This is the hook
``er_lab.eval.bootstrap`` uses to resample entities without copying frames.
An unweighted call is the all-ones special case.

References: Bagga & Baldwin 1998 (B-cubed); Meila 2003 (VI); Menestrina,
Whang & Garcia-Molina, VLDB 2010 (GMD; metrics disagree in rankings);
Papadakis et al., ACM CSUR 2020 (RR/PC and their limitations); Binette &
Steorts 2023 (incomplete-ground-truth bias).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

__all__ = [
    "bcubed",
    "pairwise",
    "variation_of_information",
    "generalized_merge_distance",
    "cluster_f",
    "blocking_metrics",
    "pair_completeness_bounds",
]


# ---------------------------------------------------------------- alignment


def _aligned(
    pred: pd.Series, truth: pd.Series, weights: pd.Series | None = None
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Validate and align pred/truth (and weights) -> (p_codes, t_codes, w).

    Returns integer cluster codes in pred's index order with zero-weight
    records dropped. Raises on: non-Series input, length mismatch, duplicate
    record ids, differing id sets, missing labels, bad weights.
    """
    for name, s in (("pred", pred), ("truth", truth)):
        if not isinstance(s, pd.Series):
            raise TypeError(f"{name} must be a pandas Series, got {type(s).__name__}")
    if len(pred) == 0:
        raise ValueError("empty clustering: nothing to evaluate")
    if len(pred) != len(truth):
        raise ValueError(f"pred has {len(pred)} records but truth has {len(truth)}")
    if not pred.index.is_unique:
        raise ValueError("pred index has duplicate record ids")
    if pred.index is truth.index or pred.index.equals(truth.index):
        t = truth
    else:
        if not truth.index.is_unique:
            raise ValueError("truth index has duplicate record ids")
        missing = pred.index.difference(truth.index)
        if len(missing):
            raise ValueError(f"record ids in pred but not in truth: {list(missing[:5])}")
        # same length + both unique + pred ⊆ truth  =>  same id set
        t = truth.reindex(pred.index)

    pv = pred.to_numpy()
    tv = t.to_numpy()
    if pd.isna(pv).any():
        raise ValueError("pred contains missing cluster labels")
    if pd.isna(tv).any():
        raise ValueError("truth contains missing entity ids")
    p_codes = pd.factorize(pv)[0]
    t_codes = pd.factorize(tv)[0]

    if weights is None:
        return p_codes, t_codes, np.ones(len(pred), dtype=float)
    if not isinstance(weights, pd.Series):
        raise TypeError(f"weights must be a pandas Series, got {type(weights).__name__}")
    if weights.index is pred.index or weights.index.equals(pred.index):
        w = weights.to_numpy(dtype=float)
    else:
        w = weights.reindex(pred.index).to_numpy(dtype=float)
    if np.isnan(w).any():
        raise ValueError("weights missing for some record ids")
    if not np.isfinite(w).all() or (w < 0).any():
        raise ValueError("weights must be finite and >= 0")
    if w.sum() == 0:
        raise ValueError("all weights are zero: nothing to evaluate")
    keep = w > 0
    if not keep.all():
        p_codes, t_codes, w = p_codes[keep], t_codes[keep], w[keep]
    return p_codes, t_codes, w


def _contingency(
    p_codes: np.ndarray, t_codes: np.ndarray, w: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Weighted pred x truth contingency over occupied cells.

    Returns (ci, cj, Wc, Wp, Wt): for each occupied cell, its pred code ci,
    truth code cj and mass Wc; plus per-pred-cluster masses Wp and
    per-truth-cluster masses Wt (indexed by code).
    """
    nt = int(t_codes.max()) + 1
    cell = p_codes.astype(np.int64) * nt + t_codes
    uc, inv = np.unique(cell, return_inverse=True)
    wc = np.bincount(inv, weights=w)
    ci = (uc // nt).astype(np.int64)
    cj = (uc % nt).astype(np.int64)
    wp = np.bincount(p_codes, weights=w, minlength=int(p_codes.max()) + 1)
    wt = np.bincount(t_codes, weights=w, minlength=nt)
    return ci, cj, wc, wp, wt


# ------------------------------------------------------------------ metrics


def bcubed(
    pred: pd.Series, truth: pd.Series, *, weights: pd.Series | None = None
) -> dict[str, float]:
    """B-cubed precision/recall/F1 (Bagga & Baldwin 1998), record-averaged.

    Definition: for record r with predicted cluster C_p(r) and truth entity
    C_t(r), precision_r = |C_p(r) ∩ C_t(r)| / |C_p(r)| and recall_r =
    |C_p(r) ∩ C_t(r)| / |C_t(r)|; precision/recall are the (weight-)averages
    of these over records, and f1 is their harmonic mean. Both components are
    strictly positive (a record always intersects its own cell), so no zero
    conventions arise. With weights, set sizes become summed weights and the
    record average is weight-averaged — exactly the metric on a frame where
    record r is duplicated weights[r] times.

    Assumptions: pred and truth are full partitions of the same record set.
    B-cubed rewards correct co-membership per record, so it is size-sensitive:
    one bad mega-merge costs far more than many small ones (the MET-01 lever).
    """
    p_codes, t_codes, w = _aligned(pred, truth, weights)
    ci, cj, wc, wp, wt = _contingency(p_codes, t_codes, w)
    total = w.sum()
    precision = float((wc * wc / wp[ci]).sum() / total)
    recall = float((wc * wc / wt[cj]).sum() / total)
    f1 = 2 * precision * recall / (precision + recall)
    return {"precision": precision, "recall": recall, "f1": float(f1)}


def pairwise(
    pred: pd.Series, truth: pd.Series, *, weights: pd.Series | None = None
) -> dict[str, float]:
    """Pairwise precision/recall/F1 over unordered co-clustered record pairs.

    Definition: tp = pairs co-clustered in both pred and truth, fp = in pred
    only, fn = in truth only (true negatives are ignored — they are ~n² and
    uninformative; lit_review §4). precision = tp/(tp+fp), recall =
    tp/(tp+fn), f1 = 2PR/(P+R).

    Zero-denominator convention (documented, deliberate): if pred co-clusters
    no pairs (all singletons), precision = NaN — not 1.0, which would reward
    refusing to link; if truth has no co-clustered pairs, recall = NaN. f1 is
    NaN whenever precision or recall is NaN, and 0.0 when both are defined and
    tp == 0 (the standard F convention for P = R = 0).

    With weights, a cluster of summed weight W contributes W(W-1)/2 pairs, so
    integer weights reproduce physical duplication exactly (duplicated copies
    of a record do pair with each other, as they would in a real frame). tp,
    fp, fn are returned as floats (weighted pair masses).
    """
    p_codes, t_codes, w = _aligned(pred, truth, weights)
    _, _, wc, wp, wt = _contingency(p_codes, t_codes, w)

    def npairs(x: np.ndarray) -> float:
        return float((x * (x - 1)).sum() / 2.0)

    tp = npairs(wc)
    pred_pairs = npairs(wp)
    truth_pairs = npairs(wt)
    fp = pred_pairs - tp
    fn = truth_pairs - tp
    precision = tp / pred_pairs if pred_pairs > 0 else float("nan")
    recall = tp / truth_pairs if truth_pairs > 0 else float("nan")
    if np.isnan(precision) or np.isnan(recall):
        f1 = float("nan")
    elif precision + recall == 0:
        f1 = 0.0
    else:
        f1 = 2 * precision * recall / (precision + recall)
    return {
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "tp": tp,
        "fp": fp,
        "fn": fn,
    }


def variation_of_information(
    pred: pd.Series, truth: pd.Series, *, weights: pd.Series | None = None
) -> float:
    """Variation of information between the two partitions, in bits.

    Definition (Meila 2003): VI = H(pred|truth) + H(truth|pred)
    = Σ_ij p_ij (log2 p_i + log2 p_j − 2 log2 p_ij), summing over occupied
    contingency cells with p_ij the joint cell mass fraction. VI is a true
    metric on partitions (symmetric, triangle inequality) and is 0 iff the
    partitions are identical. Computed in log space over occupied cells only,
    so no 0·log 0 terms arise; tiny negative rounding is clipped to 0.

    With weights, masses are summed weights — the VI of the duplicated frame.
    """
    p_codes, t_codes, w = _aligned(pred, truth, weights)
    ci, cj, wc, wp, wt = _contingency(p_codes, t_codes, w)
    total = w.sum()
    pij = wc / total
    pi = wp[ci] / total
    pj = wt[cj] / total
    vi = float((pij * (np.log2(pi) + np.log2(pj) - 2.0 * np.log2(pij))).sum())
    return max(vi, 0.0)


def generalized_merge_distance(
    pred: pd.Series,
    truth: pd.Series,
    *,
    split_cost: float = 1.0,
    merge_cost: float = 1.0,
    weights: pd.Series | None = None,
) -> float:
    """Generalized merge distance with constant per-operation costs.

    Definition (Menestrina, Whang & Garcia-Molina, VLDB 2010): the minimum
    cost of transforming pred into truth using cluster splits and merges. With
    a constant cost per operation, their linear-time "Slice" algorithm reduces
    to counting operations from the intersection pattern: a pred cluster
    intersecting k distinct truth entities needs k−1 splits (into its pure
    parts), and a truth entity intersecting m distinct pred clusters needs
    m−1 merges (of those parts), so

        GMD = split_cost · Σ_pred (k−1)  +  merge_cost · Σ_truth (m−1).

    Properties (verified in tests): GMD(pred, truth) = 0 iff identical;
    GMD(pred, truth; s, m) = GMD(truth, pred; m, s) — the reverse
    transformation swaps splits and merges — so unit costs make it symmetric.
    Note this constant-cost variant is the edit-operation count; Menestrina's
    pairwise-metric specialization uses size-dependent costs instead and is
    not what this function computes.

    Weights only drop zero-weight records: with constant costs the distance
    depends on the intersection pattern, not on masses, so positive weights
    (duplication) leave it unchanged — consistent with physical duplication.
    """
    if not (np.isfinite(split_cost) and np.isfinite(merge_cost)):
        raise ValueError("split_cost and merge_cost must be finite")
    if split_cost < 0 or merge_cost < 0:
        raise ValueError("split_cost and merge_cost must be >= 0")
    p_codes, t_codes, w = _aligned(pred, truth, weights)
    ci, cj, _, _, _ = _contingency(p_codes, t_codes, w)
    parts_per_pred = np.bincount(ci)  # occupied cells per pred cluster
    parts_per_truth = np.bincount(cj)
    splits = int((parts_per_pred[parts_per_pred > 0] - 1).sum())
    merges = int((parts_per_truth[parts_per_truth > 0] - 1).sum())
    return float(split_cost * splits + merge_cost * merges)


def cluster_f(
    pred: pd.Series, truth: pd.Series, *, weights: pd.Series | None = None
) -> dict[str, float]:
    """Closest-cluster F1 (as used by Barnes / PatentsView inventor evaluation).

    Exact definition implemented: for each truth entity T, take the maximum
    over predicted clusters P of the record-overlap F1

        F1(T, P) = 2 |T ∩ P| / (|T| + |P|),

    then average these per-entity maxima over truth entities, weighting each
    entity by its size: f1 = Σ_T (|T|/N) · max_P F1(T, P). ``macro_f1`` is the
    same with equal entity weights. This is NOT a one-to-one assignment:
    several truth entities may claim the same predicted cluster, so
    closest-cluster F1 is optimistic relative to matching-based cluster
    measures (CEAF) — one reason it is never this lab's sole headline metric.

    With weights, sizes become summed weights and the entity weighting uses
    weighted sizes (physical-duplication semantics).
    """
    p_codes, t_codes, w = _aligned(pred, truth, weights)
    ci, cj, wc, wp, wt = _contingency(p_codes, t_codes, w)
    f_cell = 2.0 * wc / (wt[cj] + wp[ci])
    best = np.zeros(len(wt))
    np.maximum.at(best, cj, f_cell)
    present = wt > 0
    total = w.sum()
    f1 = float((wt[present] * best[present]).sum() / total)
    macro = float(best[present].mean())
    return {
        "f1": f1,
        "macro_f1": macro,
        "n_truth_clusters": int(present.sum()),
        "n_pred_clusters": int((wp > 0).sum()),
    }


# ----------------------------------------------------------------- blocking


def _check_truth(truth: pd.Series) -> None:
    if not isinstance(truth, pd.Series):
        raise TypeError(f"truth must be a pandas Series, got {type(truth).__name__}")
    if len(truth) == 0:
        raise ValueError("truth is empty")
    if not truth.index.is_unique:
        raise ValueError("truth index has duplicate record ids")
    if truth.isna().any():
        raise ValueError("truth contains missing entity ids")


def _candidate_pair_set(candidate_pairs: pd.DataFrame) -> set[tuple[str, str]]:
    """Canonicalize a candidate-pair frame: unordered, deduplicated, no self-pairs."""
    if not isinstance(candidate_pairs, pd.DataFrame):
        raise TypeError("candidate_pairs must be a DataFrame with columns 'a' and 'b'")
    if not {"a", "b"}.issubset(candidate_pairs.columns):
        raise ValueError("candidate_pairs needs columns 'a' and 'b'")
    if candidate_pairs[["a", "b"]].isna().any().any():
        raise ValueError("candidate_pairs contains missing record ids")
    out: set[tuple[str, str]] = set()
    for x, y in zip(candidate_pairs["a"], candidate_pairs["b"]):
        x, y = str(x), str(y)
        if x == y:
            continue  # self-pair carries no linkage information
        out.add((x, y) if x < y else (y, x))
    return out


def _true_pair_set(truth: pd.Series) -> set[tuple[str, str]]:
    """All unordered same-entity record pairs implied by a truth Series.

    Quadratic in entity size — fine at evaluation scale (entities are small);
    the 1e8-scale pair space is exactly what blocking exists to avoid.
    """
    _check_truth(truth)
    out: set[tuple[str, str]] = set()
    for _, grp in truth.groupby(truth, sort=False):
        ids = sorted(str(i) for i in grp.index)
        for i in range(len(ids) - 1):
            for j in range(i + 1, len(ids)):
                out.add((ids[i], ids[j]))
    return out


def blocking_metrics(
    candidate_pairs: pd.DataFrame, truth: pd.Series, *, n_records: int
) -> dict[str, float]:
    """Reduction ratio and pair completeness of a blocking scheme.

    Definitions (Papadakis et al., ACM CSUR 2020): with candidate pairs
    canonicalized to distinct unordered pairs (self-pairs dropped),

        reduction_ratio  = 1 − pairs_kept / C(n_records, 2)
        pair_completeness = |candidate ∩ true pairs| / |true pairs|

    where true pairs are all unordered same-entity pairs implied by ``truth``.
    ``n_records`` is passed explicitly because candidate pairs need not
    mention every record. pair_completeness is NaN when truth implies no
    co-entity pairs (nothing to find). Remember RR is quality-blind and PC
    assumes complete ground truth — for incomplete truth use
    :func:`pair_completeness_bounds`.
    """
    if n_records < 2:
        raise ValueError("n_records must be >= 2")
    cand = _candidate_pair_set(candidate_pairs)
    total = n_records * (n_records - 1) // 2
    if len(cand) > total:
        raise ValueError(f"{len(cand)} distinct candidate pairs but only {total} possible pairs")
    true_pairs = _true_pair_set(truth)
    found = len(cand & true_pairs)
    pc = found / len(true_pairs) if true_pairs else float("nan")
    return {
        "reduction_ratio": 1.0 - len(cand) / total,
        "pair_completeness": pc,
        "pairs_kept": len(cand),
        "n_true_pairs": len(true_pairs),
        "n_true_pairs_found": found,
        "n_comparisons_total": total,
    }


def pair_completeness_bounds(
    candidate_pairs: pd.DataFrame, labeled_truth: pd.Series, *, coverage: float
) -> dict[str, float]:
    """Optimistic/pessimistic pair-completeness bounds under incomplete truth.

    Bound logic. Let coverage c ∈ (0, 1] be the assumed fraction of ALL true
    pairs that ``labeled_truth`` captures (a *pair*-level coverage: if whole
    entities are labeled, c is the labeled share of the corpus's within-entity
    pair mass, not the share of records). With T_l labeled true pairs, H_l of
    them hit by the blocker, the estimated total is T = T_l / c, of which
    U = T − T_l are unlabeled with unknown hit count h ∈ [0, U]. True PC is
    (H_l + h)/T, so:

        pc_pessimistic = H_l / T           = c · pc_observed      (h = 0)
        pc_optimistic  = (H_l + U) / T     = c · pc_observed + (1 − c)

    These bounds are sharp given exact c and need NO representativeness
    assumption about which pairs got labeled; pc_observed = H_l / T_l is
    additionally an unbiased point estimate only when labeled pairs are a
    random sample of true pairs (Binette & Steorts 2023). All quantities NaN
    when the labeled truth implies no pairs.
    """
    if not (0 < coverage <= 1):
        raise ValueError("coverage must be in (0, 1]")
    cand = _candidate_pair_set(candidate_pairs)
    true_pairs = _true_pair_set(labeled_truth)
    t_l = len(true_pairs)
    if t_l == 0:
        nan = float("nan")
        return {
            "pc_observed": nan,
            "pc_pessimistic": nan,
            "pc_optimistic": nan,
            "coverage": coverage,
            "n_true_pairs_labeled": 0,
            "n_true_pairs_estimated": nan,
        }
    h_l = len(cand & true_pairs)
    pc_obs = h_l / t_l
    return {
        "pc_observed": pc_obs,
        "pc_pessimistic": coverage * pc_obs,
        "pc_optimistic": coverage * pc_obs + (1.0 - coverage),
        "coverage": coverage,
        "n_true_pairs_labeled": t_l,
        "n_true_pairs_estimated": t_l / coverage,
    }
