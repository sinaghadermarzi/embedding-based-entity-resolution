"""er_lab.eval.operating_points — exact thresholds on hand-computable toys."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd
import pytest
from pytest import approx

from er_lab.eval.operating_points import (
    cost_optimal_threshold,
    find_threshold_for_precision,
    fp_budget_threshold,
)


def scored(pairs: list[tuple[str, str, float]]) -> pd.DataFrame:
    return pd.DataFrame(pairs, columns=["a", "b", "score"])


def cc_clusterer(pairs: pd.DataFrame, records: list[str]) -> Callable[[float], pd.Series]:
    """Connected components over pairs with score >= threshold (monotone)."""

    def cluster(threshold: float) -> pd.Series:
        parent = {r: r for r in records}

        def find(x: str) -> str:
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        for a, b, s in zip(pairs["a"], pairs["b"], pairs["score"]):
            if s >= threshold:
                ra, rb = find(a), find(b)
                if ra != rb:
                    parent[ra] = rb
        return pd.Series({r: find(r) for r in records})

    return cluster


# Toy 1: truth {r1,r2},{r3,r4},{r5},{r6}; true pairs score .9/.8, false .6/.4.
# Exact B-cubed precision/recall curve of the CC clusterer (by hand):
#   t=.9 : merge {r1,r2}            P=1              R=(1+1+.5+.5+1+1)/6=5/6
#   t=.8 : + {r3,r4}                P=1              R=1
#   t=.6 : + {r5,r6} (false)        P=(4·1+2·.5)/6=5/6   R=1
#   t=.4 : {r1..r4},{r5,r6}         P=(4·.5+2·.5)/6=.5   R=1
RECORDS = ["r1", "r2", "r3", "r4", "r5", "r6"]
TRUTH = pd.Series(["A", "A", "B", "B", "C", "D"], index=RECORDS)
PAIRS = scored([("r1", "r2", 0.9), ("r3", "r4", 0.8), ("r5", "r6", 0.6), ("r1", "r3", 0.4)])


def test_find_threshold_hits_exact_operating_point() -> None:
    out = find_threshold_for_precision(
        PAIRS, cc_clusterer(PAIRS, RECORDS), TRUTH, target=0.99
    )
    # feasible thresholds {.9, .8}; max recall picks the looser one
    assert out["threshold"] == approx(0.8)
    assert out["attained"] is True and out["fallback"] is None
    assert out["attained_precision"] == 1.0
    assert out["recall_at"] == 1.0 and out["f1_at"] == 1.0
    # the swept curve matches the hand computation above
    curve = out["curve"].set_index("threshold")
    assert curve.loc[0.9, "precision"] == 1.0
    assert curve.loc[0.9, "recall"] == approx(5 / 6)
    assert curve.loc[0.6, "precision"] == approx(5 / 6)
    assert curve.loc[0.4, "precision"] == approx(0.5)


def test_find_threshold_lower_target_prefers_recall_among_feasible() -> None:
    # target .8: thresholds {.9, .8, .6} are feasible (P >= 5/6 > .8); recall
    # ties at 1.0 for {.8, .6} -> higher precision breaks the tie -> .8
    out = find_threshold_for_precision(
        PAIRS, cc_clusterer(PAIRS, RECORDS), TRUTH, target=0.8
    )
    assert out["threshold"] == approx(0.8)
    assert out["attained_precision"] == 1.0


def test_find_threshold_unattainable_sets_fallback_rail() -> None:
    # top-scored pair is FALSE: truth {q1,q2},{q3},{q4}; (q3,q4)=.95 false,
    # (q1,q2)=.9 true. Precision is .75 at both thresholds -> target .99 is
    # unattainable; tie on precision -> max recall -> t=.9. The fallback flag
    # MUST be set (PLAN §5: never silently switch protocol).
    records = ["q1", "q2", "q3", "q4"]
    truth = pd.Series(["X", "X", "Y", "Z"], index=records)
    pairs = scored([("q3", "q4", 0.95), ("q1", "q2", 0.9)])
    out = find_threshold_for_precision(pairs, cc_clusterer(pairs, records), truth, target=0.99)
    assert out["attained"] is False
    assert out["fallback"] == "highest_attainable"
    assert out["threshold"] == approx(0.9)
    assert out["attained_precision"] == approx(0.75)
    assert out["recall_at"] == approx(1.0)


def test_find_threshold_validation() -> None:
    with pytest.raises(ValueError, match="target precision"):
        find_threshold_for_precision(PAIRS, cc_clusterer(PAIRS, RECORDS), TRUTH, target=1.5)
    with pytest.raises(ValueError, match="columns"):
        find_threshold_for_precision(
            PAIRS.rename(columns={"score": "s"}), cc_clusterer(PAIRS, RECORDS), TRUTH, target=0.9
        )


# Cost toy: truth pairs score .9/.55, false pairs .6/.4 -> pairwise (fp, fn):
#   t=.9 : (0, 1)   t=.6 : (1, 1)   t=.55 : (1, 0)   t=.4 : (5, 0)
COST_PAIRS = scored(
    [("r1", "r2", 0.9), ("r3", "r4", 0.55), ("r5", "r6", 0.6), ("r1", "r3", 0.4)]
)


def test_cost_optimal_threshold_known_answers() -> None:
    clusterer = cc_clusterer(COST_PAIRS, RECORDS)
    # 1:1 -> costs [1, 2, 1, 5]; tie between .9 and .55 breaks conservative (.9)
    out = cost_optimal_threshold(COST_PAIRS, clusterer, TRUTH, fp_cost=1.0, fn_cost=1.0)
    assert out["threshold"] == approx(0.9)
    assert out["cost"] == approx(1.0) and out["fp"] == 0 and out["fn"] == 1
    # FN 5x costlier -> costs [5, 6, 1, 5] -> accept the false merge risk at .55
    out = cost_optimal_threshold(COST_PAIRS, clusterer, TRUTH, fp_cost=1.0, fn_cost=5.0)
    assert out["threshold"] == approx(0.55)
    assert out["cost"] == approx(1.0) and out["fp"] == 1 and out["fn"] == 0
    # FP 10x costlier -> costs [1, 11, 10, 50] -> stay conservative at .9
    out = cost_optimal_threshold(COST_PAIRS, clusterer, TRUTH, fp_cost=10.0, fn_cost=1.0)
    assert out["threshold"] == approx(0.9)
    assert out["cost"] == approx(1.0)
    with pytest.raises(ValueError, match="at least one"):
        cost_optimal_threshold(COST_PAIRS, clusterer, TRUTH, fp_cost=0.0, fn_cost=0.0)


def test_fp_budget_threshold_known_answers() -> None:
    scores = np.array([0.99, 0.95, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2])
    # budget = 0.2 * 10 = 2 pairs -> threshold = 2nd largest score
    t = fp_budget_threshold(scores, budget_per_record=0.2, n_records=10)
    assert t == approx(0.95)
    assert (scores >= t).sum() == 2
    # budget covers everything -> min score
    assert fp_budget_threshold(scores, budget_per_record=2.0, n_records=10) == approx(0.2)
    # budget < 1 pair -> just above the max: nothing kept
    t = fp_budget_threshold(scores, budget_per_record=0.05, n_records=10)
    assert t > 0.99 and (scores >= t).sum() == 0


def test_fp_budget_threshold_tie_handling_is_conservative() -> None:
    # k=2 but three scores tie at .8: keeping the tie group would blow the
    # budget, so the whole group is dropped
    scores = np.array([0.9, 0.8, 0.8, 0.8, 0.5])
    t = fp_budget_threshold(scores, budget_per_record=2.0, n_records=1)
    assert t > 0.8
    assert (scores >= t).sum() == 1  # only the .9 survives
    with pytest.raises(ValueError, match="budget_per_record"):
        fp_budget_threshold(scores, budget_per_record=-1.0, n_records=1)
