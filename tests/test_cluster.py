"""Tests for er_lab.cluster: partition schemes and graph QA on hand-built graphs."""

from __future__ import annotations

import pandas as pd
import pytest

from er_lab.cluster.graph_qa import component_stats, suspicious_clusters
from er_lab.cluster.schemes import (
    capped_agglomerative,
    greedy_correlation,
    star_clustering,
    transitive_closure,
)


def pairs_frame(edges: list[tuple[str, str, float]]) -> pd.DataFrame:
    return pd.DataFrame(edges, columns=["a", "b", "prob"])


def clusters_of(pred: pd.Series) -> set[frozenset]:
    """Partition as a set of frozensets — label-invariant comparison."""
    return {frozenset(g.index) for _, g in pred.groupby(pred)}


RECORDS = pd.Index(["a", "b", "c", "d", "e"])


# ---------------------------------------------------------------- transitive closure


def test_transitive_closure_chain_merges_all_three():
    # a-b and b-c above threshold: closure merges all three even though a-c
    # was never scored — the chain-merge mechanism itself
    pairs = pairs_frame([("a", "b", 0.9), ("b", "c", 0.9), ("c", "d", 0.5)])
    pred = transitive_closure(pairs, threshold=0.8, records=RECORDS)
    assert list(pred.index) == list(RECORDS)
    assert clusters_of(pred) == {frozenset("abc"), frozenset("d"), frozenset("e")}
    # labels are the min member id
    assert pred["a"] == pred["b"] == pred["c"] == "a"
    assert pred["d"] == "d" and pred["e"] == "e"


def test_transitive_closure_all_below_threshold_gives_singletons():
    pairs = pairs_frame([("a", "b", 0.3)])
    pred = transitive_closure(pairs, threshold=0.8, records=RECORDS)
    assert clusters_of(pred) == {frozenset(x) for x in RECORDS}


def test_transitive_closure_orientation_and_row_order_invariant():
    fwd = pairs_frame([("a", "b", 0.9), ("b", "c", 0.9)])
    rev = pairs_frame([("c", "b", 0.9), ("b", "a", 0.9)])  # reversed + reordered
    pd.testing.assert_series_equal(
        transitive_closure(fwd, threshold=0.8, records=RECORDS),
        transitive_closure(rev, threshold=0.8, records=RECORDS),
    )


# ---------------------------------------------------------------- capped agglomerative


def test_capped_agglomerative_refuses_cap_violation_on_chain():
    # the chain-merge demo mechanism (notebook 07): closure would fuse the whole
    # chain; the cap refuses the middle merge
    chain = pairs_frame([("a", "b", 0.95), ("b", "c", 0.90), ("c", "d", 0.93)])
    closed = transitive_closure(chain, threshold=0.8, records=RECORDS)
    assert clusters_of(closed) == {frozenset("abcd"), frozenset("e")}
    capped = capped_agglomerative(chain, threshold=0.8, cap=2, records=RECORDS)
    # best edge a-b merges first, c-d merges next; b-c is refused (size 4 > cap)
    assert clusters_of(capped) == {frozenset("ab"), frozenset("cd"), frozenset("e")}
    assert capped.groupby(capped).size().max() <= 2


def test_capped_agglomerative_greedy_order_is_by_score():
    # b-c is the strongest edge, so it merges first and locks b; with cap=2 both
    # weaker edges are then refused
    pairs = pairs_frame([("a", "b", 0.85), ("b", "c", 0.99), ("c", "d", 0.85)])
    pred = capped_agglomerative(pairs, threshold=0.8, cap=2, records=RECORDS)
    assert clusters_of(pred) == {frozenset("bc"), frozenset("a"), frozenset("d"), frozenset("e")}


def test_capped_agglomerative_cap_validation():
    with pytest.raises(ValueError, match="cap"):
        capped_agglomerative(pairs_frame([]), threshold=0.5, cap=0, records=RECORDS)


# ---------------------------------------------------------------------------- star


def test_star_clustering_hub_becomes_center():
    # h is a 4-degree hub; s1..s4 attach to it; s1-s2 edge does not chain further
    records = pd.Index(["h", "s1", "s2", "s3", "s4", "x"])
    pairs = pairs_frame(
        [("h", "s1", 0.9), ("h", "s2", 0.9), ("h", "s3", 0.9), ("h", "s4", 0.9), ("s1", "s2", 0.9)]
    )
    pred = star_clustering(pairs, threshold=0.8, records=records)
    assert clusters_of(pred) == {frozenset({"h", "s1", "s2", "s3", "s4"}), frozenset({"x"})}


def test_star_clustering_radius_one():
    # path a-b-c-d: no cluster may span more than one hop from its center
    pairs = pairs_frame([("a", "b", 0.9), ("b", "c", 0.9), ("c", "d", 0.9)])
    pred = star_clustering(pairs, threshold=0.8, records=RECORDS)
    # b and c tie on degree 2; b wins by id -> star {a,b,c}, then d is a leftover singleton
    assert clusters_of(pred) == {frozenset("abc"), frozenset("d"), frozenset("e")}


# ------------------------------------------------------------- greedy correlation


def test_greedy_correlation_recovers_cliques_any_seed():
    records = pd.Index(["a", "b", "c", "x", "y", "z"])
    pairs = pairs_frame(
        [
            ("a", "b", 0.9), ("a", "c", 0.9), ("b", "c", 0.9),
            ("x", "y", 0.9), ("x", "z", 0.9), ("y", "z", 0.9),
            ("c", "x", 0.2),  # negative log-odds: never attracts
        ]
    )
    expected = {frozenset("abc"), frozenset("xyz")}
    for seed in (0, 1, 17):
        pred = greedy_correlation(pairs, records=records, seed=seed)
        assert clusters_of(pred) == expected


def test_greedy_correlation_uses_log_odds_sign_not_a_threshold():
    pairs = pairs_frame([("a", "b", 0.51), ("c", "d", 0.49)])
    pred = greedy_correlation(pairs, records=RECORDS, seed=0)
    assert pred["a"] == pred["b"]  # prob > 0.5 -> positive log-odds -> merged
    assert pred["c"] != pred["d"]  # prob < 0.5 -> negative -> apart


def test_greedy_correlation_deterministic_under_seed():
    records = pd.Index([f"r{i}" for i in range(12)])
    edges = [(f"r{i}", f"r{j}", 0.8) for i in range(12) for j in range(i + 1, 12) if j - i <= 2]
    pairs = pairs_frame(edges)
    pd.testing.assert_series_equal(
        greedy_correlation(pairs, records=records, seed=3),
        greedy_correlation(pairs, records=records, seed=3),
    )


# ------------------------------------------------------------------- validation


def test_schemes_validate_inputs():
    with pytest.raises(ValueError, match="absent from records"):
        transitive_closure(pairs_frame([("a", "ghost", 0.9)]), threshold=0.5, records=RECORDS)
    with pytest.raises(ValueError, match="self-pairs"):
        transitive_closure(pairs_frame([("a", "a", 0.9)]), threshold=0.5, records=RECORDS)
    with pytest.raises(ValueError, match="within"):
        transitive_closure(pairs_frame([("a", "b", 1.5)]), threshold=0.5, records=RECORDS)
    with pytest.raises(KeyError, match="missing columns"):
        transitive_closure(pd.DataFrame({"a": [], "b": []}), threshold=0.5, records=RECORDS)


# --------------------------------------------------------------------- graph QA


def barbell() -> tuple[pd.DataFrame, pd.Index]:
    """Two triangles {a1,a2,a3} and {b1,b2,b3} joined by the single bridge a3-b1."""
    edges = [
        ("a1", "a2", 0.9), ("a1", "a3", 0.9), ("a2", "a3", 0.9),
        ("b1", "b2", 0.9), ("b1", "b3", 0.9), ("b2", "b3", 0.9),
        ("a3", "b1", 0.9),
    ]
    return pairs_frame(edges), pd.Index(["a1", "a2", "a3", "b1", "b2", "b3", "solo"])


def test_component_stats_on_barbell():
    pairs, records = barbell()
    stats = component_stats(pairs, 0.8, records)
    assert list(stats.columns) == ["cluster_id", "size", "density", "n_bridges"]
    assert len(stats) == 2  # the barbell + the singleton
    bar = stats[stats["cluster_id"] == "a1"].iloc[0]
    assert bar["size"] == 6
    assert bar["density"] == pytest.approx(7 / 15)  # 7 edges of C(6,2)=15
    assert bar["n_bridges"] == 1  # exactly the a3-b1 connector
    solo = stats[stats["cluster_id"] == "solo"].iloc[0]
    assert solo["size"] == 1 and solo["density"] == 1.0 and solo["n_bridges"] == 0


def test_component_stats_pure_chain_is_all_bridges():
    pairs = pairs_frame([("a", "b", 0.9), ("b", "c", 0.9), ("c", "d", 0.9)])
    stats = component_stats(pairs, 0.8, pd.Index(["a", "b", "c", "d"]))
    chain = stats.iloc[0]
    assert chain["size"] == 4
    assert chain["n_bridges"] == 3  # every edge of a path is a bridge
    assert chain["density"] == pytest.approx(3 / 6)


def test_component_stats_threshold_filters_edges():
    pairs = pairs_frame([("a", "b", 0.9), ("b", "c", 0.3)])
    stats = component_stats(pairs, 0.8, pd.Index(["a", "b", "c"]))
    assert sorted(stats["size"].tolist()) == [1, 2]


def test_suspicious_clusters_rules():
    stats = pd.DataFrame(
        {
            "cluster_id": ["chain", "dense", "mega", "pair"],
            "size": [4, 4, 50, 2],
            "density": [3 / 6, 1.0, 0.05, 1.0],
            "n_bridges": [3, 0, 10, 1],
        }
    )
    flagged = suspicious_clusters(stats, size_p99=40)
    reasons = dict(zip(flagged["cluster_id"], flagged["reasons"]))
    assert "chain" in reasons and "bridges" in reasons["chain"]  # 3/3 bridges > 0.5
    assert "mega" in reasons and "size" in reasons["mega"] and "density" in reasons["mega"]
    assert "dense" not in reasons  # small complete cluster is fine
    assert "pair" not in reasons  # a pair is never suspicious by size


def test_suspicious_clusters_on_barbell_stats_and_overrides():
    pairs, records = barbell()
    stats = component_stats(pairs, 0.8, records)
    # permissive rules (size rule silenced by an explicit high cut): nothing flagged
    calm = suspicious_clusters(
        stats, size_p99=100, rules={"min_density": 0.1, "bridge_frac": 1.0}
    )
    assert len(calm) == 0
    # a strict density rule catches the barbell as a sparse-merge suspect
    strict = suspicious_clusters(stats, size_p99=100, rules={"min_density": 0.6})
    assert strict["cluster_id"].tolist() == ["a1"]
    assert strict["reasons"].tolist() == ["density"]
    with pytest.raises(KeyError, match="unknown rule"):
        suspicious_clusters(stats, rules={"nope": 1})
