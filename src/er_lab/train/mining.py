"""Positive-pair building and hard-negative mining, plus the TRN-02 instruments.

The TRN-02 story: in a dedup-dense corpus, "hard negatives" mined by lexical or
embedding similarity are exactly where the unlabeled duplicates live, so mining
contaminates training with false negatives. This module supplies the miners,
the *measurement* (:func:`fn_contamination` — computable only where truth
exists, i.e. on the synthetic corpora), and the *mitigation*
(:func:`cluster_aware_filter`, which drops mined pairs that a truth-proxy
clustering already believes co-refer).

All frames are canonical (``er_lab.data.schema``): a unique ``record_id``
column, ``entity_id`` as the truth key (NA when unknown). All functions are
deterministic given their ``seed``.
"""

from __future__ import annotations

from itertools import combinations
from typing import Protocol

import numpy as np
import pandas as pd

__all__ = [
    "MinedBatches",
    "ann_hard_negatives",
    "bm25_hard_negatives",
    "build_pairs_inbatch",
    "cluster_aware_filter",
    "fn_contamination",
]

MINED_COLUMNS = ["anchor_id", "neg_id", "rank"]


class MinedBatches(Protocol):
    """A hard-negative miner: (corpus_df, features, *, k, seed) -> mined frame.

    ``features`` is whatever similarity substrate the miner ranks by (serialized
    texts for BM25, embedding rows for ANN). The mined frame has columns
    ``anchor_id, neg_id, rank`` with rank 1 = the hardest (most similar)
    negative; ``anchor_id != neg_id`` always.
    """

    def __call__(self, corpus_df, features, /, *, k: int, seed: int) -> pd.DataFrame: ...


def build_pairs_inbatch(corpus_df: pd.DataFrame, *, n_pairs: int, seed: int) -> pd.DataFrame:
    """Sample ``n_pairs`` positive record pairs (same entity_id) -> DataFrame[a_id, b_id].

    Enumerates every within-entity pair (records with NA entity_id contribute
    none), then samples uniformly with a seeded rng — without replacement when
    enough pairs exist, with replacement otherwise (small corpora still fill a
    training run). Raises when the corpus has no duplicated entity at all:
    contrastive training without a single positive is a data bug, not a run.
    """
    ids = corpus_df.loc[corpus_df["entity_id"].notna(), ["record_id", "entity_id"]]
    all_pairs: list[tuple[str, str]] = []
    for _, grp in ids.groupby("entity_id", sort=True):
        rids = grp["record_id"].tolist()
        all_pairs.extend(combinations(rids, 2))
    if not all_pairs:
        raise ValueError(
            "build_pairs_inbatch: corpus has no within-entity positive pairs "
            "(no entity_id appears on more than one record)"
        )
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(all_pairs), size=n_pairs, replace=len(all_pairs) < n_pairs)
    rows = [all_pairs[i] for i in idx]
    return pd.DataFrame(rows, columns=["a_id", "b_id"]).astype("string")


def _neighbors_to_frame(record_ids: list[str], indices: np.ndarray, k: int) -> pd.DataFrame:
    """Shared tail of both miners: neighbor index matrix -> mined frame, self excluded."""
    rows: list[tuple[str, str, int]] = []
    for i, row in enumerate(indices):
        rank = 0
        for j in row:
            j = int(j)
            if j == i or j < 0:  # self, or faiss padding when n < k+1
                continue
            rank += 1
            rows.append((record_ids[i], record_ids[j], rank))
            if rank == k:
                break
    out = pd.DataFrame(rows, columns=MINED_COLUMNS)
    out[["anchor_id", "neg_id"]] = out[["anchor_id", "neg_id"]].astype("string")
    return out


def bm25_hard_negatives(corpus_df: pd.DataFrame, texts, *, k: int, seed: int = 0) -> pd.DataFrame:
    """Mine each record's top-``k`` BM25-most-similar other records as negatives.

    ``texts`` is the serialized text per record, positionally aligned to
    ``corpus_df``. Tokenized with ``stopwords=None`` — person records are short
    and name tokens ('will', 'may') must never be stopworded away. BM25
    retrieval is deterministic; ``seed`` is accepted for the
    :class:`MinedBatches` signature and reserved for sampled variants.
    """
    import bm25s  # deferred: numba-backed import is slow

    del seed  # deterministic miner; see docstring
    corpus_texts = [str(t) for t in (texts.tolist() if hasattr(texts, "tolist") else texts)]
    if len(corpus_texts) != len(corpus_df):
        raise ValueError(f"{len(corpus_texts)} texts for {len(corpus_df)} records")
    tokens = bm25s.tokenize(corpus_texts, stopwords=None, show_progress=False)
    retriever = bm25s.BM25()
    retriever.index(tokens, show_progress=False)
    depth = min(k + 1, len(corpus_texts))
    indices, _scores = retriever.retrieve(tokens, k=depth, show_progress=False)
    return _neighbors_to_frame(corpus_df["record_id"].astype(str).tolist(), indices, k)


def ann_hard_negatives(
    corpus_df: pd.DataFrame, embeddings: np.ndarray, *, k: int, seed: int = 0
) -> pd.DataFrame:
    """Mine each record's top-``k`` nearest neighbors (cosine, exact faiss) as negatives.

    ``embeddings`` is (n_records, dim) positionally aligned to ``corpus_df``;
    rows are L2-normalized here before an exact IndexFlatIP search, so ranking
    is by cosine. Exact search is deterministic; ``seed`` is accepted for the
    :class:`MinedBatches` signature and reserved for sampled variants.
    """
    import faiss  # deferred: keeps import cost off non-ANN paths

    del seed  # deterministic miner; see docstring
    x = np.ascontiguousarray(np.asarray(embeddings, dtype=np.float32))
    if x.shape[0] != len(corpus_df):
        raise ValueError(f"{x.shape[0]} embedding rows for {len(corpus_df)} records")
    x = x / np.clip(np.linalg.norm(x, axis=1, keepdims=True), 1e-12, None)
    index = faiss.IndexFlatIP(x.shape[1])
    index.add(x)
    _sims, indices = index.search(x, min(k + 1, x.shape[0]))
    return _neighbors_to_frame(corpus_df["record_id"].astype(str).tolist(), indices, k)


def _same_entity_mask(mined: pd.DataFrame, mapping: pd.Series) -> pd.Series:
    """Boolean mask: mined rows whose anchor and negative share a non-NA mapped id."""
    a = mined["anchor_id"].map(mapping)
    b = mined["neg_id"].map(mapping)
    return ((a == b) & a.notna() & b.notna()).fillna(False).astype(bool)


def fn_contamination(mined: pd.DataFrame, truth: pd.Series) -> float:
    """Fraction of mined negatives that are actually same-entity — the TRN-02 measurement.

    ``truth`` maps record_id (index) -> entity_id. Rows whose anchor or negative
    has no known entity count as clean (contamination is only what truth can
    prove). NaN for an empty mined frame — no negatives means nothing measured,
    not a clean miner.
    """
    if len(mined) == 0:
        return float("nan")
    return float(_same_entity_mask(mined, truth).mean())


def cluster_aware_filter(mined: pd.DataFrame, truth_proxy: pd.Series) -> pd.DataFrame:
    """Drop mined negatives whose anchor and negative share a truth-proxy entity.

    The TRN-02 mitigation: ``truth_proxy`` maps record_id -> proxy cluster id
    (in production a cheap clustering or a prior model's output — never the
    held-out truth; with synthetic truth passed in it becomes the oracle
    upper-bound arm). Rows with NA proxy on either side are kept.
    """
    keep = ~_same_entity_mask(mined, truth_proxy)
    return mined.loc[keep.to_numpy()].reset_index(drop=True)
