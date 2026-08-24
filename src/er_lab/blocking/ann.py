"""Dense ANN blocking over record embeddings — the embeddings arm of BAS-02 (PLAN §3).

Each record's embedding queries a faiss index over all records; the top-k
cosine neighbors become candidate pairs, with the same ``DataFrame[a, b]``
shape and per-record budget-k semantics as the matchkeys and BM25 arms, so the
three are comparable at matched candidate budget.

Index variants (the EFF-01 / SCL axis of the head-to-head):

- ``'flat'``  — exact inner-product search (IndexFlatIP); the recall ceiling,
- ``'hnsw'``  — graph ANN (IndexHNSWFlat); the practical 1e7-scale choice,
- ``'ivfpq'`` — inverted lists + product quantization (IndexIVFPQ); the
  memory-compressed regime whose *ER-metric* cost EFF-01 measures.

Embeddings are expected L2-normalized (the encoder contract) so inner product
= cosine; they are defensively re-normalized here, which is a no-op on already
normalized inputs. faiss is pinned to a single OpenMP thread for the build and
search: HNSW graph construction and IVF k-means are otherwise thread-order
dependent, and blocking output must be deterministic (cross-cutting rule).
The thread cap is restored afterwards.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

__all__ = ["candidates"]

_INDEX_KINDS = ("flat", "hnsw", "ivfpq")


def _build_index(kind: str, x: np.ndarray):
    """Construct (and train, where needed) the faiss index for *kind* over *x*."""
    import faiss

    n, d = x.shape
    if kind == "flat":
        index = faiss.IndexFlatIP(d)
    elif kind == "hnsw":
        index = faiss.IndexHNSWFlat(d, 32, faiss.METRIC_INNER_PRODUCT)
        index.hnsw.efConstruction = 80
    elif kind == "ivfpq":
        # Lab-scale parameter policy, documented rather than tuned: ~one list per
        # 39 vectors (faiss's k-means guidance), sub-quantizer count = the largest
        # power of two <= 8 dividing d, and codebooks no bigger than the data.
        nlist = max(1, min(256, n // 39))
        m = next(m for m in (8, 4, 2, 1) if d % m == 0)
        nbits = int(min(8, max(2, np.floor(np.log2(max(n, 4))) - 1)))
        quantizer = faiss.IndexFlatIP(d)
        index = faiss.IndexIVFPQ(quantizer, d, nlist, m, nbits, faiss.METRIC_INNER_PRODUCT)
        index.train(x)
        index.nprobe = min(nlist, 8)
    else:
        raise ValueError(f"unknown index kind {kind!r}: expected one of {_INDEX_KINDS}")
    index.add(x)
    return index


def candidates(
    df: pd.DataFrame,
    embeddings: np.ndarray,
    *,
    k: int,
    index: str = "flat",
) -> pd.DataFrame:
    """Top-k cosine neighbor pairs as ``DataFrame[a, b]``, a<b, deduped, no self-pairs.

    *embeddings* aligns positionally with *df* (row i embeds record i).
    Deterministic for a given (embeddings, k, index) triple — see the module
    docstring for how. ``.attrs['stats']`` records the index kind and budget.
    """
    import faiss

    if "record_id" not in df.columns:
        raise KeyError("candidates: frame has no 'record_id' column")
    rid = df["record_id"].astype("string")
    if rid.isna().any() or not rid.is_unique:
        raise ValueError("candidates: record_id must be unique and non-null")
    if k < 1:
        raise ValueError(f"k must be >= 1, got {k}")
    x = np.ascontiguousarray(np.asarray(embeddings, dtype=np.float32))
    if x.ndim != 2 or x.shape[0] != len(df):
        raise ValueError(
            f"embeddings shape {x.shape} does not align with frame of {len(df)} records"
        )
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    norms[norms == 0.0] = 1.0  # zero vectors stay zero rather than becoming NaN
    x = x / norms

    ids = rid.tolist()
    n = len(ids)
    if n < 2:
        out = pd.DataFrame(columns=["a", "b"], dtype="string")
        out.attrs["stats"] = {"method": f"ann_{index}", "k": k, "n_pairs": 0, "n_records": n}
        return out

    prev_threads = faiss.omp_get_max_threads()
    faiss.omp_set_num_threads(1)
    try:
        ix = _build_index(index, x)
        if index == "hnsw":
            ix.hnsw.efSearch = max(50, 2 * (k + 1))
        _, nbr = ix.search(x, min(k + 1, n))
    finally:
        faiss.omp_set_num_threads(prev_threads)

    seen: set[tuple[str, str]] = set()
    for i in range(n):
        kept = 0
        for j in np.asarray(nbr[i]).tolist():
            if j == i or j < 0:  # self-hit / "no result" slot
                continue
            if kept >= k:
                break
            kept += 1
            a, b = (ids[i], ids[j]) if ids[i] < ids[j] else (ids[j], ids[i])
            seen.add((a, b))

    out = pd.DataFrame(sorted(seen), columns=["a", "b"], dtype="string")
    out.attrs["stats"] = {
        "method": f"ann_{index}",
        "k": k,
        "n_pairs": len(out),
        "n_records": n,
    }
    return out
