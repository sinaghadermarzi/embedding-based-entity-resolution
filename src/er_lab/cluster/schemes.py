"""Pair scores -> entity partitions: the CLU-01 clustering-scheme family (PLAN §3).

All schemes share one signature family — ``(pairs: DataFrame[a, b, prob], *,
..., records: Index) -> pd.Series`` — where the returned Series is a full
partition: indexed by *records* (every record, including ones no pair
mentions, which become singletons), valued by a cluster label. Labels are the
lexicographically smallest member record_id of each cluster: deterministic,
stable under record order, and directly usable as ``pred`` by
:mod:`er_lab.eval.metrics`.

Why four schemes: transitive closure is what naive pipelines do and is the
chain-merge catastrophe's enabling mechanism (notebook 07); the capped and
star variants are the standard cheap guards; greedy correlation clustering
(KwikCluster on log-odds sign) is the principled reference that needs no
threshold at all. CLU-01 measures what each buys at each score-quality level.

Deterministic: fixed tie-breaking everywhere; the one randomized scheme
(:func:`greedy_correlation`'s pivot order) is driven only by its ``seed``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

__all__ = [
    "capped_agglomerative",
    "greedy_correlation",
    "star_clustering",
    "transitive_closure",
]


class UnionFind:
    """Union-find with path compression + union by size (shared with graph_qa)."""

    def __init__(self, items: list[str]):
        self.parent = {x: x for x in items}
        self.size = {x: 1 for x in items}

    def find(self, x: str) -> str:
        root = x
        while self.parent[root] != root:
            root = self.parent[root]
        while self.parent[x] != root:  # path compression
            self.parent[x], x = root, self.parent[x]
        return root

    def union(self, a: str, b: str) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return
        if self.size[ra] < self.size[rb]:
            ra, rb = rb, ra
        self.parent[rb] = ra
        self.size[ra] += self.size[rb]


def check_pairs(pairs: pd.DataFrame, records: pd.Index) -> pd.DataFrame:
    """Validate a scored-pairs frame against the record universe.

    Requires columns a, b, prob; probs in [0, 1]; no self-pairs; every id in
    *records* (an unknown id is a wiring bug upstream, never silently dropped).
    Returns a copy with a/b as plain str and canonical a<b orientation.
    """
    missing_cols = {"a", "b", "prob"} - set(pairs.columns)
    if missing_cols:
        raise KeyError(f"pairs frame missing columns {sorted(missing_cols)}")
    if not records.is_unique:
        raise ValueError("records index has duplicate ids")
    out = pd.DataFrame(
        {
            "a": pairs["a"].astype(str).to_numpy(),
            "b": pairs["b"].astype(str).to_numpy(),
            "prob": pairs["prob"].astype(float).to_numpy(),
        }
    )
    if ((out["prob"] < 0) | (out["prob"] > 1) | out["prob"].isna()).any():
        raise ValueError("pairs prob values must be within [0, 1]")
    if (out["a"] == out["b"]).any():
        raise ValueError("pairs contains self-pairs (a == b)")
    known = {str(r) for r in records}
    unknown = (set(out["a"]) | set(out["b"])) - known
    if unknown:
        raise ValueError(f"pairs reference ids absent from records: {sorted(unknown)[:5]}")
    swap = out["a"] > out["b"]
    out.loc[swap, ["a", "b"]] = out.loc[swap, ["b", "a"]].to_numpy()
    return out


def _labels_from_uf(uf: UnionFind, records: pd.Index) -> pd.Series:
    """Partition Series from a union-find: label = min member id per cluster."""
    ids = [str(r) for r in records]
    label_of_root: dict[str, str] = {}
    for x in sorted(ids):  # ascending, so the first member seen per root is the min
        root = uf.find(x)
        label_of_root.setdefault(root, x)
    return pd.Series([label_of_root[uf.find(x)] for x in ids], index=records, name="cluster")


def _labels_from_clusters(assign: dict[str, str], records: pd.Index) -> pd.Series:
    """Partition Series from an id->cluster-key dict; labels re-canonicalized to min member."""
    members: dict[str, list[str]] = {}
    for x, c in assign.items():
        members.setdefault(c, []).append(x)
    label = {c: min(xs) for c, xs in members.items()}
    ids = [str(r) for r in records]
    return pd.Series([label[assign[x]] for x in ids], index=records, name="cluster")


def transitive_closure(
    pairs: pd.DataFrame, *, threshold: float, records: pd.Index
) -> pd.Series:
    """Union every pair with prob >= threshold; connected components are clusters.

    The naive default — and the mechanism behind chain-merges: one bad edge
    fuses two real entities irreversibly (notebook 07 demonstrates the
    catastrophe this invites at scale). Singletons for untouched records.
    """
    p = check_pairs(pairs, records)
    uf = UnionFind([str(r) for r in records])
    keep = p[p["prob"] >= threshold].sort_values(["a", "b"])  # order irrelevant, fixed anyway
    for a, b in zip(keep["a"], keep["b"]):
        uf.union(a, b)
    return _labels_from_uf(uf, records)


def capped_agglomerative(
    pairs: pd.DataFrame, *, threshold: float, cap: int, records: pd.Index
) -> pd.Series:
    """Greedy best-score-first merging that refuses any cluster larger than *cap*.

    Edges with prob >= threshold are processed in descending-prob order (ties
    by (a, b)); a merge that would produce a cluster of size > cap is skipped
    — the direct guard against chain-merges: a chain a-b-c stops at the cap
    instead of percolating (the notebook-07 demo depends on this refusal).
    Greedy, no backtracking: an early merge can block a later, better one.
    """
    if cap < 1:
        raise ValueError(f"cap must be >= 1, got {cap}")
    p = check_pairs(pairs, records)
    uf = UnionFind([str(r) for r in records])
    keep = p[p["prob"] >= threshold].sort_values(
        ["prob", "a", "b"], ascending=[False, True, True]
    )
    for a, b in zip(keep["a"], keep["b"]):
        ra, rb = uf.find(a), uf.find(b)
        if ra != rb and uf.size[ra] + uf.size[rb] <= cap:
            uf.union(a, b)
    return _labels_from_uf(uf, records)


def star_clustering(
    pairs: pd.DataFrame, *, threshold: float, records: pd.Index
) -> pd.Series:
    """Center-picking clustering: high-degree records become star centers.

    Over the prob >= threshold graph, records are visited by (degree desc, id
    asc); an unassigned record becomes a center and claims its still-unassigned
    neighbors. Every cluster therefore has radius <= 1 around a center — chains
    cannot form, at the price of splitting genuine sprawling entities. The
    standard hub-resistant scheme CLU-01 pits against closure.
    """
    p = check_pairs(pairs, records)
    keep = p[p["prob"] >= threshold]
    ids = [str(r) for r in records]
    neighbors: dict[str, set[str]] = {x: set() for x in ids}
    for a, b in zip(keep["a"], keep["b"]):
        neighbors[a].add(b)
        neighbors[b].add(a)
    order = sorted(ids, key=lambda x: (-len(neighbors[x]), x))
    assign: dict[str, str] = {}
    for center in order:
        if center in assign:
            continue
        assign[center] = center
        for nb in sorted(neighbors[center]):
            if nb not in assign:
                assign[nb] = center
    return _labels_from_clusters(assign, records)


def greedy_correlation(
    pairs: pd.DataFrame, *, records: pd.Index, seed: int = 0
) -> pd.Series:
    """Greedy correlation clustering (KwikCluster / pivot) on log-odds sign.

    No threshold: an edge attracts iff its log-odds log(p/(1-p)) is positive,
    i.e. prob > 0.5 — the sign the FS model itself assigns. Pivots are drawn
    in a seed-determined random order (the KwikCluster 3-approximation
    guarantee is in expectation over this order); each unassigned pivot claims
    its unassigned positive neighbors. Deterministic for a fixed *seed*.
    """
    p = check_pairs(pairs, records)
    keep = p[p["prob"] > 0.5]  # log-odds > 0
    ids = sorted(str(r) for r in records)
    neighbors: dict[str, set[str]] = {x: set() for x in ids}
    for a, b in zip(keep["a"], keep["b"]):
        neighbors[a].add(b)
        neighbors[b].add(a)
    rng = np.random.default_rng(seed)
    order = [ids[i] for i in rng.permutation(len(ids))]
    assign: dict[str, str] = {}
    for pivot in order:
        if pivot in assign:
            continue
        assign[pivot] = pivot
        for nb in sorted(neighbors[pivot]):
            if nb not in assign:
                assign[nb] = pivot
    return _labels_from_clusters(assign, records)
