"""Operating-point selection: fixed precision, explicit cost, FP budget.

PLAN.md §5 locks the lab-wide operating points: fixed entity-precision
{0.99, 0.995} plus a cost grid {1:1, 1:10, 1:100 FP:FN}, with a fixed
FP-budget-per-record as the mandatory secondary at scale tiers — and NEVER
per-method best-F1. Hand & Christen (Statistics and Computing 2018) showed
best-F1 comparisons weight precision vs recall by each method's own output,
so every method is scored at a different tradeoff; MET-02 demonstrates the
resulting rank reversals on this lab's outputs. These helpers pin the
tradeoff *before* looking at any method's sweet spot.

Threshold grid semantics (shared by both sweep functions): thresholds are the
unique score values in ``scored_pairs`` — the points where the clustering can
actually change — subsampled evenly to at most ``grid`` values when there are
more. No threshold above the maximum score is added: the all-singleton output
is not a real operating point of a linkage system, and admitting it would let
"link nothing" satisfy any precision target. The smoke-fallback rail
(PLAN §5) is explicit instead: when the target precision is unattainable on
the grid, the best attainable point is returned with ``fallback`` set —
never a silent protocol switch.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd

from er_lab.eval.metrics import bcubed, pairwise

__all__ = [
    "cost_optimal_threshold",
    "find_threshold_for_precision",
    "fp_budget_threshold",
]


def _threshold_grid(scored_pairs: pd.DataFrame, grid: int) -> np.ndarray:
    if not isinstance(scored_pairs, pd.DataFrame) or not {"a", "b", "score"}.issubset(
        scored_pairs.columns
    ):
        raise ValueError("scored_pairs must be a DataFrame with columns 'a', 'b', 'score'")
    if len(scored_pairs) == 0:
        raise ValueError("scored_pairs is empty")
    if grid < 1:
        raise ValueError("grid must be >= 1")
    scores = scored_pairs["score"].to_numpy(dtype=float)
    if not np.isfinite(scores).all():
        raise ValueError("scores must be finite")
    uniq = np.unique(scores)  # ascending
    if len(uniq) > grid:
        idx = np.unique(np.round(np.linspace(0, len(uniq) - 1, grid)).astype(int))
        uniq = uniq[idx]
    return uniq[::-1]  # descending: tightest threshold first


def _sweep(
    scored_pairs: pd.DataFrame,
    clusterer: Callable[[float], pd.Series],
    truth: pd.Series,
    grid: int,
) -> pd.DataFrame:
    """Evaluate B-cubed and pairwise counts at every grid threshold."""
    rows = []
    for t in _threshold_grid(scored_pairs, grid):
        pred = clusterer(float(t))
        b3 = bcubed(pred, truth)
        pw = pairwise(pred, truth)
        rows.append(
            {
                "threshold": float(t),
                "precision": b3["precision"],
                "recall": b3["recall"],
                "f1": b3["f1"],
                "pair_fp": pw["fp"],
                "pair_fn": pw["fn"],
            }
        )
    return pd.DataFrame(rows)


def find_threshold_for_precision(
    scored_pairs: pd.DataFrame,
    clusterer: Callable[[float], pd.Series],
    truth: pd.Series,
    *,
    target: float,
    grid: int = 200,
) -> dict:
    """Loosest threshold whose clustering meets a B-cubed precision target.

    Precision here is entity-level B-cubed precision (the lab's primary
    precision notion, PLAN §5); the clusterer is called once per grid
    threshold and must return a clustering Series over truth's records. Among
    thresholds attaining ``target``, the one with maximum recall is returned
    (ties: higher precision, then higher threshold); precision need not be
    monotone in the threshold (transitive closure can make it dip), which is
    why the sweep is exhaustive over the grid rather than bisecting.

    Smoke-fallback rail: when no grid threshold attains the target (common on
    small eval sets where one bad merge sinks precision below 0.99), the
    highest-attainable point is returned with ``attained=False`` and
    ``fallback='highest_attainable'`` — the caller must surface this, never
    silently report a different protocol. Returns {threshold,
    attained_precision, attained, fallback, recall_at, f1_at} plus ``target``
    and the full ``curve`` DataFrame for plotting.
    """
    if not (0 < target <= 1):
        raise ValueError("target precision must be in (0, 1]")
    curve = _sweep(scored_pairs, clusterer, truth, grid)
    feasible = curve[curve["precision"] >= target]
    if len(feasible):
        pick = feasible.sort_values(
            ["recall", "precision", "threshold"], ascending=False
        ).iloc[0]
        attained, fallback = True, None
    else:
        pick = curve.sort_values(
            ["precision", "recall", "threshold"], ascending=False
        ).iloc[0]
        attained, fallback = False, "highest_attainable"
    return {
        "threshold": float(pick["threshold"]),
        "attained_precision": float(pick["precision"]),
        "attained": attained,
        "fallback": fallback,
        "recall_at": float(pick["recall"]),
        "f1_at": float(pick["f1"]),
        "target": float(target),
        "curve": curve,
    }


def cost_optimal_threshold(
    scored_pairs: pd.DataFrame,
    clusterer: Callable[[float], pd.Series],
    truth: pd.Series,
    *,
    fp_cost: float,
    fn_cost: float,
    grid: int = 200,
) -> dict:
    """Threshold minimizing the explicit pairwise error cost fp_cost·FP + fn_cost·FN.

    FP/FN are counted over unordered record pairs (a false merge of clusters
    sized j and k contributes j·k FPs — chain merges are charged what they
    cost), making the {1:1, 1:10, 1:100} cost grid of PLAN §5 concrete.
    Entity-level costing is a modeling decision left to the notebooks. Ties
    break toward the higher (more conservative) threshold. Returns
    {threshold, cost, fp, fn, precision, recall, f1, fp_cost, fn_cost, curve}
    with precision/recall/f1 the B-cubed values at the chosen threshold.
    """
    if fp_cost < 0 or fn_cost < 0:
        raise ValueError("fp_cost and fn_cost must be >= 0")
    if fp_cost == 0 and fn_cost == 0:
        raise ValueError("at least one of fp_cost / fn_cost must be > 0")
    curve = _sweep(scored_pairs, clusterer, truth, grid)
    curve = curve.assign(cost=fp_cost * curve["pair_fp"] + fn_cost * curve["pair_fn"])
    pick = curve.sort_values(["cost", "threshold"], ascending=[True, False]).iloc[0]
    return {
        "threshold": float(pick["threshold"]),
        "cost": float(pick["cost"]),
        "fp": float(pick["pair_fp"]),
        "fn": float(pick["pair_fn"]),
        "precision": float(pick["precision"]),
        "recall": float(pick["recall"]),
        "f1": float(pick["f1"]),
        "fp_cost": float(fp_cost),
        "fn_cost": float(fn_cost),
        "curve": curve,
    }


def fp_budget_threshold(
    scores: np.ndarray | pd.Series, *, budget_per_record: float, n_records: int
) -> float:
    """Score threshold that caps accepted candidate pairs at a per-record budget.

    Pure score-tail computation, no clustering and no truth: in the scale
    regime (SCL-01/02) candidate pairs are overwhelmingly non-matches, so
    pairs accepted above the threshold are, to first order, false positives —
    capping their count caps the FP load per record. The returned t is the
    lowest threshold (most pairs kept) such that

        #{scores >= t}  <=  floor(budget_per_record * n_records).

    Ties at the cut are handled conservatively: if keeping every score equal
    to the k-th largest would blow the budget, t is nudged just above that
    value (np.nextafter), dropping the whole tie group. Edge cases: a budget
    of >= len(scores) returns min(scores) (keep everything); a budget < 1
    returns just above max(scores) (keep nothing).
    """
    s = np.asarray(scores, dtype=float)
    if s.ndim != 1 or len(s) == 0:
        raise ValueError("scores must be a non-empty 1-d array")
    if not np.isfinite(s).all():
        raise ValueError("scores must be finite")
    if n_records < 1:
        raise ValueError("n_records must be >= 1")
    if budget_per_record < 0:
        raise ValueError("budget_per_record must be >= 0")
    k = int(np.floor(budget_per_record * n_records + 1e-12))
    if k <= 0:
        return float(np.nextafter(s.max(), np.inf))
    if k >= len(s):
        return float(s.min())
    top = np.sort(s)[::-1]
    t = float(top[k - 1])
    if int((s >= t).sum()) > k:
        t = float(np.nextafter(t, np.inf))
    return t
