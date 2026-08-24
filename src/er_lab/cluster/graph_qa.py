"""Cluster-graph quality assurance: structural red flags for bad merges (CLU-01/07).

A predicted cluster is a connected component of the thresholded pair graph.
Its *shape* betrays how it formed: a genuine duplicate group is small and
dense (most pairs scored as matches), while a chain-merge accident is big,
sparse, and held together by bridges — single edges whose removal would split
the component. :func:`component_stats` measures exactly those three signals;
:func:`suspicious_clusters` turns them into a flagged worklist for the
adjudication widget.

The heuristics are deliberately simple and *documented, not validated* —
their empirical hit-rate against ground truth is a notebook question
(notebooks 07/13), not something this module asserts.
"""

from __future__ import annotations

import pandas as pd

from er_lab.cluster.schemes import UnionFind, check_pairs

__all__ = ["component_stats", "suspicious_clusters"]

#: Default flagging rules (override any subset via ``rules=``):
#: - size_quantile: flag clusters above this empirical size quantile (and > 2),
#: - min_density:   flag clusters of size >= min_density_size sparser than this,
#: - bridge_frac:   flag clusters of size >= min_density_size where more than
#:                  this fraction of a spanning tree's edges are bridges
#:                  (a pure chain has (size-1) bridges -> fraction 1.0).
DEFAULT_RULES = {
    "size_quantile": 0.99,
    "min_density": 0.4,
    "min_density_size": 4,
    "bridge_frac": 0.5,
}


def _bridges(nodes: list[str], adj: dict[str, list[str]]) -> int:
    """Count bridge edges in one component — iterative Tarjan low-link DFS.

    Assumes a simple graph (edges deduped by the caller), so the single
    ``child == parent`` occurrence per adjacency list is always the tree edge.
    """
    disc: dict[str, int] = {}
    low: dict[str, int] = {}
    bridges = 0
    counter = 0
    for start in nodes:
        if start in disc:
            continue
        # stack entries: (node, parent, next index into adj[node])
        stack: list[tuple[str, str | None, int]] = [(start, None, 0)]
        disc[start] = low[start] = counter
        counter += 1
        while stack:
            node, parent, i = stack.pop()
            if i < len(adj[node]):
                stack.append((node, parent, i + 1))
                child = adj[node][i]
                if child == parent:
                    continue  # the tree edge back to the parent
                if child in disc:
                    low[node] = min(low[node], disc[child])  # back edge
                else:
                    disc[child] = low[child] = counter
                    counter += 1
                    stack.append((child, node, 0))
            elif parent is not None:  # node's subtree is complete
                low[parent] = min(low[parent], low[node])
                if low[node] > disc[parent]:
                    bridges += 1
    return bridges


def component_stats(pairs: pd.DataFrame, threshold: float, records: pd.Index) -> pd.DataFrame:
    """Per-component ``DataFrame[cluster_id, size, density, n_bridges]``.

    Components of the prob >= threshold graph over *records* (singletons
    included, density 1.0 by convention — a singleton is trivially complete).
    ``cluster_id`` is the min member record_id, matching the labels produced
    by :mod:`er_lab.cluster.schemes`; ``density`` = edges / (size choose 2);
    ``n_bridges`` counts edges whose removal disconnects the component.
    Rows are sorted by (size desc, cluster_id) — worst suspects first.
    """
    p = check_pairs(pairs, records)
    keep = p[p["prob"] >= threshold].drop_duplicates(["a", "b"])
    ids = [str(r) for r in records]
    uf = UnionFind(ids)
    adj: dict[str, list[str]] = {x: [] for x in ids}
    for a, b in zip(keep["a"], keep["b"]):
        uf.union(a, b)
        adj[a].append(b)
        adj[b].append(a)
    for nbrs in adj.values():  # deterministic DFS order
        nbrs.sort()

    members: dict[str, list[str]] = {}
    for x in ids:
        members.setdefault(uf.find(x), []).append(x)
    n_edges: dict[str, int] = {}
    for a in keep["a"]:
        root = uf.find(a)
        n_edges[root] = n_edges.get(root, 0) + 1

    rows = []
    for root, xs in members.items():
        size = len(xs)
        edges = n_edges.get(root, 0)
        density = 1.0 if size == 1 else edges / (size * (size - 1) / 2)
        rows.append(
            {
                "cluster_id": min(xs),
                "size": size,
                "density": density,
                "n_bridges": _bridges(sorted(xs), adj) if size > 1 else 0,
            }
        )
    out = pd.DataFrame(rows, columns=["cluster_id", "size", "density", "n_bridges"])
    return out.sort_values(["size", "cluster_id"], ascending=[False, True], ignore_index=True)


def suspicious_clusters(
    stats: pd.DataFrame,
    *,
    size_p99: float | None = None,
    rules: dict | None = None,
) -> pd.DataFrame:
    """Flag structurally suspect clusters; returns the flagged subset + a 'reasons' column.

    Heuristics (each independently sufficient; see :data:`DEFAULT_RULES`):

    - ``size``:    size exceeds *size_p99* when given, else the empirical
      ``size_quantile`` of the observed sizes — and is > 2 (pairs are never
      suspicious by size alone),
    - ``density``: size >= min_density_size and density < min_density —
      a big sparse component means most member pairs were NOT scored as
      matches: the chain-merge signature,
    - ``bridges``: size >= min_density_size and n_bridges / (size - 1)
      > bridge_frac — the component hangs together by cut edges.

    Validation of these heuristics against truth is deferred to the notebooks.
    """
    cfg = dict(DEFAULT_RULES)
    if rules:
        unknown = set(rules) - set(DEFAULT_RULES)
        if unknown:
            raise KeyError(f"unknown rule keys {sorted(unknown)}; expected {sorted(DEFAULT_RULES)}")
        cfg.update(rules)
    required = {"cluster_id", "size", "density", "n_bridges"}
    if not required <= set(stats.columns):
        raise KeyError(f"stats frame missing columns {sorted(required - set(stats.columns))}")

    if len(stats) == 0:
        out = stats.copy()
        out["reasons"] = pd.Series([], dtype=object)
        return out

    size_cut = (
        float(size_p99)
        if size_p99 is not None
        else float(stats["size"].quantile(cfg["size_quantile"]))
    )
    reasons: list[list[str]] = []
    for row in stats.itertuples(index=False):
        why = []
        if row.size > max(size_cut, 2):
            why.append("size")
        if row.size >= cfg["min_density_size"] and row.density < cfg["min_density"]:
            why.append("density")
        if row.size >= cfg["min_density_size"] and row.n_bridges / (row.size - 1) > cfg[
            "bridge_frac"
        ]:
            why.append("bridges")
        reasons.append(why)
    out = stats.copy()
    out["reasons"] = [", ".join(w) for w in reasons]
    return out[out["reasons"] != ""].reset_index(drop=True)
