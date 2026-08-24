"""BM25 sparse-retrieval blocking — the lexical arm of BAS-02 (PLAN §3).

Every record's serialized text (see :mod:`er_lab.serialize`) queries a BM25
index over all records; its top-k neighbors become candidate pairs. This is
the classic "cheap lexical recall" baseline that dense ANN blocking must beat
at a *matched* per-record budget k — the same k semantics as
:func:`er_lab.blocking.matchkeys.candidates` and :func:`er_lab.blocking.ann.candidates`.

Honesty details: the self-hit is removed *before* the budget is applied (so k
means k genuine neighbors), and neighbors with BM25 score 0 are dropped — a
zero score means no token overlap at all, and keeping such rows would pad the
candidate set with arbitrary index-order records rather than real candidates.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd

__all__ = ["candidates"]


def candidates(df: pd.DataFrame, texts: Sequence[str] | pd.Series, *, k: int) -> pd.DataFrame:
    """Top-k BM25 neighbor pairs as ``DataFrame[a, b]``, a<b, deduped, no self-pairs.

    *texts* must align positionally with *df* (one serialized string per
    record). Deterministic: bm25s scoring and top-k selection have no random
    state. ``.attrs['stats']`` records the budget and drop counts.
    """
    import bm25s  # heavy-ish import kept local to the blocking call

    if "record_id" not in df.columns:
        raise KeyError("candidates: frame has no 'record_id' column")
    rid = df["record_id"].astype("string")
    if rid.isna().any() or not rid.is_unique:
        raise ValueError("candidates: record_id must be unique and non-null")
    if k < 1:
        raise ValueError(f"k must be >= 1, got {k}")
    if isinstance(texts, pd.Series):
        texts = texts.tolist()
    texts = ["" if t is None or pd.isna(t) else str(t) for t in texts]
    if len(texts) != len(df):
        raise ValueError(f"texts has {len(texts)} entries but frame has {len(df)} records")

    ids = rid.tolist()
    n = len(ids)
    if n < 2:
        out = pd.DataFrame(columns=["a", "b"], dtype="string")
        out.attrs["stats"] = {"method": "bm25", "k": k, "n_pairs": 0, "n_records": n}
        return out

    tokens = bm25s.tokenize(texts, stopwords=None, show_progress=False)
    retriever = bm25s.BM25()
    retriever.index(tokens, show_progress=False)
    # k+1 because each record retrieves itself (usually rank 1); capped at n.
    idx, scores = retriever.retrieve(tokens, k=min(k + 1, n), show_progress=False)

    seen: set[tuple[str, str]] = set()
    zero_dropped = 0
    for i in range(n):
        kept = 0
        for j, s in zip(np.asarray(idx[i]).tolist(), np.asarray(scores[i]).tolist()):
            if j == i:
                continue
            if kept >= k:
                break
            if s <= 0.0:
                zero_dropped += 1
                continue
            kept += 1
            a, b = (ids[i], ids[j]) if ids[i] < ids[j] else (ids[j], ids[i])
            seen.add((a, b))

    out = pd.DataFrame(sorted(seen), columns=["a", "b"], dtype="string")
    out.attrs["stats"] = {
        "method": "bm25",
        "k": k,
        "n_pairs": len(out),
        "n_records": n,
        "zero_score_dropped": zero_dropped,
    }
    return out
