"""er_lab.eval.metrics — known-answer tests, worked by hand in comments."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from pytest import approx

from er_lab.eval.metrics import (
    bcubed,
    blocking_metrics,
    cluster_f,
    generalized_merge_distance,
    pair_completeness_bounds,
    pairwise,
    variation_of_information,
)

# ---------------------------------------------------------------------------
# The classic toy example used throughout:
#   truth: T1={a,b,c}  T2={d,e}
#   pred:  P1={a,b}    P2={c,d,e}
# ---------------------------------------------------------------------------
TRUTH = pd.Series({"a": "T1", "b": "T1", "c": "T1", "d": "T2", "e": "T2"})
PRED = pd.Series({"a": "P1", "b": "P1", "c": "P2", "d": "P2", "e": "P2"})


def test_bcubed_classic_toy_hand_computed() -> None:
    # Per-record precision |Cp∩Ct|/|Cp|:
    #   a: |P1∩T1|/|P1| = 2/2 = 1        b: 1
    #   c: |P2∩T1|/|P2| = 1/3
    #   d: |P2∩T2|/|P2| = 2/3            e: 2/3
    #   P = (1 + 1 + 1/3 + 2/3 + 2/3)/5 = (11/3)/5 = 11/15
    # Per-record recall |Cp∩Ct|/|Ct|:
    #   a: 2/3   b: 2/3   c: 1/3   d: 2/2=1   e: 1
    #   R = (2/3 + 2/3 + 1/3 + 1 + 1)/5 = (11/3)/5 = 11/15
    # F1 = harmonic(11/15, 11/15) = 11/15
    out = bcubed(PRED, TRUTH)
    assert out["precision"] == approx(11 / 15)
    assert out["recall"] == approx(11 / 15)
    assert out["f1"] == approx(11 / 15)


def test_bcubed_perfect_and_alignment_reorder() -> None:
    assert bcubed(TRUTH, TRUTH) == {"precision": 1.0, "recall": 1.0, "f1": 1.0}
    # index order must not matter — alignment is on record ids
    shuffled = PRED.loc[["e", "a", "d", "b", "c"]]
    assert bcubed(shuffled, TRUTH)["f1"] == approx(11 / 15)


def test_pairwise_classic_toy_hand_computed() -> None:
    # Pred co-clustered pairs: {a,b} + {c,d},{c,e},{d,e}  -> 4 pairs
    # Truth co-clustered pairs: {a,b},{a,c},{b,c} + {d,e} -> 4 pairs
    # tp = {a,b},{d,e} = 2;  fp = {c,d},{c,e} = 2;  fn = {a,c},{b,c} = 2
    # P = 2/4 = 0.5, R = 2/4 = 0.5, F1 = 0.5
    out = pairwise(PRED, TRUTH)
    assert out["tp"] == 2 and out["fp"] == 2 and out["fn"] == 2
    assert out["precision"] == approx(0.5)
    assert out["recall"] == approx(0.5)
    assert out["f1"] == approx(0.5)


def test_pairwise_zero_denominator_convention_is_nan() -> None:
    singletons = pd.Series({"a": 1, "b": 2, "c": 3, "d": 4, "e": 5})
    # pred all singletons: no predicted pairs -> precision NaN (never 1.0), f1 NaN
    out = pairwise(singletons, TRUTH)
    assert np.isnan(out["precision"]) and np.isnan(out["f1"])
    assert out["recall"] == 0.0 and out["fn"] == 4
    # truth all singletons: no true pairs -> recall NaN
    out = pairwise(PRED, singletons)
    assert np.isnan(out["recall"]) and np.isnan(out["f1"])
    assert out["precision"] == 0.0


def test_variation_of_information_known_answers() -> None:
    # identical clusterings -> 0 bits
    assert variation_of_information(TRUTH, TRUTH) == 0.0
    # documented small case: truth {a,b},{c,d} vs pred all-in-one:
    # H(pred)=0, I=0 -> VI = H(truth) = 1 bit
    t = pd.Series({"a": 1, "b": 1, "c": 2, "d": 2})
    p = pd.Series({"a": 0, "b": 0, "c": 0, "d": 0})
    assert variation_of_information(p, t) == approx(1.0)
    # classic toy, by hand: cells (P1,T1)=2, (P2,T1)=1, (P2,T2)=2 of N=5
    #   VI = Σ p_ij [log2 p_i + log2 p_j − 2 log2 p_ij]
    #      = .4·log2(1.5) + .4·log2(3) + .4·log2(1.5) = 1.101955...
    expected = 0.8 * np.log2(1.5) + 0.4 * np.log2(3.0)
    assert variation_of_information(PRED, TRUTH) == approx(expected)
    # VI is symmetric
    assert variation_of_information(PRED, TRUTH) == approx(
        variation_of_information(TRUTH, PRED)
    )


def test_gmd_known_answers() -> None:
    # identical -> 0
    assert generalized_merge_distance(TRUTH, TRUTH) == 0.0
    # one split: pred merges everything, truth = {a,b,c},{d,e} -> 1 split
    allone = pd.Series({"a": 0, "b": 0, "c": 0, "d": 0, "e": 0})
    assert generalized_merge_distance(allone, TRUTH, split_cost=3.0, merge_cost=100.0) == 3.0
    # one merge: reverse direction -> 1 merge
    assert generalized_merge_distance(TRUTH, allone, split_cost=100.0, merge_cost=3.0) == 3.0
    # classic toy: split P2={c,d,e} into {c},{d,e} (1 split), then merge {c}
    # into {a,b} (1 merge) -> split_cost + merge_cost
    assert generalized_merge_distance(PRED, TRUTH) == 2.0
    assert generalized_merge_distance(PRED, TRUTH, split_cost=5.0, merge_cost=0.5) == 5.5


def test_gmd_documented_symmetry_property() -> None:
    # GMD(pred, truth; s, m) == GMD(truth, pred; m, s): the reverse
    # transformation swaps splits and merges. Unit costs -> symmetric.
    rng = np.random.default_rng(0)
    ids = [f"r{i}" for i in range(40)]
    p = pd.Series(rng.integers(0, 7, 40), index=ids)
    t = pd.Series(rng.integers(0, 5, 40), index=ids)
    assert generalized_merge_distance(
        p, t, split_cost=2.0, merge_cost=7.0
    ) == generalized_merge_distance(t, p, split_cost=7.0, merge_cost=2.0)
    assert generalized_merge_distance(p, t) == generalized_merge_distance(t, p)


def test_cluster_f_documented_example() -> None:
    # T1={a,b,c}: F1 vs P1={a,b} is 2·2/(3+2)=0.8, vs P2={c,d,e} is 2·1/(3+3)=1/3
    #   -> best 0.8
    # T2={d,e}:   F1 vs P2 is 2·2/(2+3)=0.8 -> best 0.8
    # size-weighted: (3·0.8 + 2·0.8)/5 = 0.8 ; macro: (0.8+0.8)/2 = 0.8
    out = cluster_f(PRED, TRUTH)
    assert out["f1"] == approx(0.8)
    assert out["macro_f1"] == approx(0.8)
    assert out["n_truth_clusters"] == 2 and out["n_pred_clusters"] == 2
    perfect = cluster_f(TRUTH, TRUTH)
    assert perfect["f1"] == 1.0 and perfect["macro_f1"] == 1.0


def test_alignment_errors() -> None:
    with pytest.raises(ValueError, match="pred has 4 records"):
        bcubed(PRED.iloc[:4], TRUTH)
    with pytest.raises(ValueError, match="not in truth"):
        bcubed(PRED.rename(index={"a": "zz"}), TRUTH)
    dup = pd.concat([PRED, PRED.iloc[:1]])
    with pytest.raises(ValueError, match="duplicate record ids"):
        bcubed(dup, pd.concat([TRUTH, TRUTH.iloc[:1]]))
    with pytest.raises(ValueError, match="missing cluster labels"):
        bcubed(PRED.mask(PRED == "P1"), TRUTH)
    with pytest.raises(TypeError, match="pandas Series"):
        bcubed(PRED.to_dict(), TRUTH)  # type: ignore[arg-type]


def test_weights_match_physical_duplication() -> None:
    # weight 2 on record 'a', 0 on 'e'  ==  duplicated frame with a copy of
    # 'a' and 'e' dropped, for every partition metric
    w = pd.Series({"a": 2, "b": 1, "c": 1, "d": 1, "e": 0})
    dup_ids = ["a", "a2", "b", "c", "d"]
    pred_dup = pd.Series([PRED["a"], PRED["a"], PRED["b"], PRED["c"], PRED["d"]], index=dup_ids)
    truth_dup = pd.Series(
        [TRUTH["a"], TRUTH["a"], TRUTH["b"], TRUTH["c"], TRUTH["d"]], index=dup_ids
    )
    assert bcubed(PRED, TRUTH, weights=w) == approx(bcubed(pred_dup, truth_dup))
    assert pairwise(PRED, TRUTH, weights=w) == approx(pairwise(pred_dup, truth_dup))
    assert variation_of_information(PRED, TRUTH, weights=w) == approx(
        variation_of_information(pred_dup, truth_dup)
    )
    got = cluster_f(PRED, TRUTH, weights=w)
    want = cluster_f(pred_dup, truth_dup)
    assert got["f1"] == approx(want["f1"]) and got["macro_f1"] == approx(want["macro_f1"])
    assert generalized_merge_distance(PRED, TRUTH, weights=w) == generalized_merge_distance(
        pred_dup, truth_dup
    )


def test_weights_validation() -> None:
    with pytest.raises(ValueError, match="finite"):
        bcubed(PRED, TRUTH, weights=pd.Series(-1.0, index=PRED.index))
    with pytest.raises(ValueError, match="all weights are zero"):
        bcubed(PRED, TRUTH, weights=pd.Series(0.0, index=PRED.index))
    with pytest.raises(ValueError, match="weights missing"):
        bcubed(PRED, TRUTH, weights=pd.Series({"a": 1.0}))
    # Fractional weights would give W*(W-1)/2 < 0 pair masses (negative tp,
    # NaN precision) — rejected up front rather than silently returned.
    with pytest.raises(ValueError, match="integer-valued"):
        pairwise(PRED, TRUTH, weights=pd.Series(0.5, index=PRED.index))


# ------------------------------------------------------------------ blocking


def cand(pairs: list[tuple[str, str]]) -> pd.DataFrame:
    return pd.DataFrame(pairs, columns=["a", "b"])


def test_blocking_metrics_known_case() -> None:
    # truth pairs: {a,b},{a,c},{b,c},{d,e} -> 4 true pairs; n_records=5 ->
    # C(5,2)=10 comparisons. Candidates {ab, ac, de, ad}: kept=4,
    # RR = 1 - 4/10 = 0.6; hits = ab, ac, de -> PC = 3/4.
    out = blocking_metrics(
        cand([("a", "b"), ("a", "c"), ("d", "e"), ("a", "d")]), TRUTH, n_records=5
    )
    assert out["pairs_kept"] == 4
    assert out["reduction_ratio"] == approx(0.6)
    assert out["pair_completeness"] == approx(0.75)
    assert out["n_true_pairs"] == 4 and out["n_true_pairs_found"] == 3


def test_blocking_metrics_canonicalizes_pairs() -> None:
    # duplicates, reversed order, and self-pairs must not inflate pairs_kept
    messy = cand([("a", "b"), ("b", "a"), ("a", "b"), ("c", "c"), ("d", "e")])
    out = blocking_metrics(messy, TRUTH, n_records=5)
    assert out["pairs_kept"] == 2
    assert out["pair_completeness"] == approx(0.5)  # ab, de found of 4


def test_blocking_metrics_no_true_pairs_is_nan() -> None:
    singles = pd.Series({"a": 1, "b": 2, "c": 3})
    out = blocking_metrics(cand([("a", "b")]), singles, n_records=3)
    assert np.isnan(out["pair_completeness"])
    assert out["reduction_ratio"] == approx(1 - 1 / 3)


def test_pair_completeness_bounds_known_case() -> None:
    # labeled truth = the toy: 4 true pairs, candidates hit 3 -> pc_obs = .75.
    # coverage=0.5 -> estimated total 8 true pairs, 4 unlabeled:
    #   pessimistic (blocker misses all unlabeled): 3/8 = .375
    #   optimistic (blocker catches all unlabeled): (3+4)/8 = .875
    out = pair_completeness_bounds(
        cand([("a", "b"), ("a", "c"), ("d", "e"), ("a", "d")]), TRUTH, coverage=0.5
    )
    assert out["pc_observed"] == approx(0.75)
    assert out["pc_pessimistic"] == approx(0.375)
    assert out["pc_optimistic"] == approx(0.875)
    assert out["n_true_pairs_labeled"] == 4
    assert out["n_true_pairs_estimated"] == approx(8.0)
    # full coverage: bounds collapse onto the observed value
    full = pair_completeness_bounds(cand([("a", "b")]), TRUTH, coverage=1.0)
    assert full["pc_pessimistic"] == approx(full["pc_observed"])
    assert full["pc_optimistic"] == approx(full["pc_observed"])
    with pytest.raises(ValueError, match="coverage"):
        pair_completeness_bounds(cand([("a", "b")]), TRUTH, coverage=0.0)
