"""Tests for er_lab.train.mining (positives, hard-negative miners, TRN-02 instruments)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from er_lab.train.mining import (
    ann_hard_negatives,
    bm25_hard_negatives,
    build_pairs_inbatch,
    cluster_aware_filter,
    fn_contamination,
)


def make_corpus(n_ent: int = 30, copies: int = 2) -> pd.DataFrame:
    """Planted-duplicate corpus: every entity has `copies` near-identical records."""
    rows = []
    for e in range(n_ent):
        for c in range(copies):
            rows.append(
                {
                    "record_id": f"r{e:03d}_{c}",
                    "entity_id": f"e{e:03d}",
                    "given_name": f"NAME{e}",
                    "family_name": f"FAM{e % 7}",
                }
            )
    return pd.DataFrame(rows, dtype="string")


def truth_of(df: pd.DataFrame) -> pd.Series:
    return df.set_index("record_id")["entity_id"]


# --- build_pairs_inbatch ----------------------------------------------------


def test_positive_pairs_are_same_entity_and_sized():
    df = make_corpus()
    truth = truth_of(df)
    pairs = build_pairs_inbatch(df, n_pairs=40, seed=3)
    assert list(pairs.columns) == ["a_id", "b_id"]
    assert len(pairs) == 40
    assert (pairs["a_id"] != pairs["b_id"]).all()
    assert (pairs["a_id"].map(truth) == pairs["b_id"].map(truth)).all()


def test_positive_pairs_deterministic():
    df = make_corpus()
    p1 = build_pairs_inbatch(df, n_pairs=25, seed=11)
    p2 = build_pairs_inbatch(df, n_pairs=25, seed=11)
    pd.testing.assert_frame_equal(p1, p2)


def test_positive_pairs_oversampling_with_replacement():
    df = make_corpus(n_ent=3, copies=2)  # only 3 distinct pairs exist
    pairs = build_pairs_inbatch(df, n_pairs=20, seed=0)
    assert len(pairs) == 20


def test_no_duplicated_entity_raises():
    df = make_corpus(copies=1)
    with pytest.raises(ValueError, match="positive pairs"):
        build_pairs_inbatch(df, n_pairs=5, seed=0)


def test_na_entities_contribute_no_pairs():
    df = make_corpus(n_ent=2)
    df.loc[df["entity_id"] == "e001", "entity_id"] = pd.NA
    pairs = build_pairs_inbatch(df, n_pairs=10, seed=0)
    assert set(pairs["a_id"].str[:5]) == {"r000_"}


# --- fn_contamination / cluster_aware_filter -------------------------------


def _hand_built_mined() -> pd.DataFrame:
    # 8 mined negatives; rows 0 and 4 are same-entity (planted false negatives)
    return pd.DataFrame(
        {
            "anchor_id": [
                "r000_0",
                "r000_0",
                "r001_0",
                "r001_0",
                "r002_0",
                "r002_0",
                "r003_0",
                "r003_0",
            ],
            "neg_id": [
                "r000_1",
                "r005_0",
                "r002_0",
                "r003_0",
                "r001_1",
                "r002_1",
                "r004_0",
                "r005_1",
            ],
            "rank": [1, 2, 1, 2, 1, 2, 1, 2],
        }
    ).astype({"anchor_id": "string", "neg_id": "string"})


def test_fn_contamination_exact_on_planted_dups():
    mined = _hand_built_mined()
    # r000_0~r000_1 (row 0) and r002_0~r002_1 (row 5) share an entity -> 2/8
    assert fn_contamination(mined, truth_of(make_corpus())) == pytest.approx(0.25)


def test_fn_contamination_na_truth_counts_clean():
    mined = _hand_built_mined()
    truth = truth_of(make_corpus())
    truth.loc["r002_1"] = pd.NA  # removes one provable contamination
    assert fn_contamination(mined, truth) == pytest.approx(0.125)


def test_fn_contamination_empty_is_nan():
    empty = _hand_built_mined().iloc[:0]
    assert np.isnan(fn_contamination(empty, truth_of(make_corpus())))


def test_cluster_aware_filter_drops_exactly_the_planted_rows():
    mined = _hand_built_mined()
    truth = truth_of(make_corpus())
    filtered = cluster_aware_filter(mined, truth)
    assert len(filtered) == 6
    assert fn_contamination(filtered, truth) == 0.0
    # only same-proxy rows dropped, everything else kept in order
    kept = list(zip(filtered["anchor_id"], filtered["neg_id"]))
    assert ("r000_0", "r000_1") not in kept and ("r002_0", "r002_1") not in kept


def test_cluster_aware_filter_keeps_na_proxy_rows():
    mined = _hand_built_mined()
    proxy = truth_of(make_corpus())
    proxy[:] = pd.NA  # proxy knows nothing -> nothing is dropped
    assert len(cluster_aware_filter(mined, proxy)) == len(mined)


# --- miners -----------------------------------------------------------------


def _check_miner_frame(mined: pd.DataFrame, df: pd.DataFrame, k: int) -> None:
    assert list(mined.columns) == ["anchor_id", "neg_id", "rank"]
    assert (mined["anchor_id"] != mined["neg_id"]).all()  # never mines self
    per_anchor = mined.groupby("anchor_id", sort=False)
    assert len(per_anchor) == len(df)  # every record is an anchor
    assert (per_anchor.size() == k).all()  # exactly k negatives each (n > k here)
    expected_ranks = list(range(1, k + 1))
    assert all(ranks == expected_ranks for ranks in per_anchor["rank"].apply(list))


def test_bm25_miner_shape_and_no_self():
    df = make_corpus()
    texts = df["given_name"] + " " + df["family_name"]
    mined = bm25_hard_negatives(df, texts, k=3, seed=0)
    _check_miner_frame(mined, df, k=3)


def test_bm25_miner_finds_planted_dups_and_is_deterministic():
    df = make_corpus()
    texts = df["given_name"] + " " + df["family_name"]
    m1 = bm25_hard_negatives(df, texts, k=3, seed=0)
    m2 = bm25_hard_negatives(df, texts, k=3, seed=0)
    pd.testing.assert_frame_equal(m1, m2)
    # the duplicate record has identical text -> mined as a top negative
    assert fn_contamination(m1, truth_of(df)) > 0.2


def test_ann_miner_shape_no_self_and_contamination():
    df = make_corpus()
    rng = np.random.default_rng(0)
    emb = rng.normal(size=(len(df), 16)).astype(np.float32)
    emb[1::2] = emb[0::2] + 0.01  # duplicates land next to their entity-mates
    mined = ann_hard_negatives(df, emb, k=4, seed=0)
    _check_miner_frame(mined, df, k=4)
    truth = truth_of(df)
    assert fn_contamination(mined, truth) > 0.2  # the TRN-02 trap, planted
    assert fn_contamination(cluster_aware_filter(mined, truth), truth) == 0.0  # mitigated


def test_ann_miner_k_capped_by_corpus_size():
    df = make_corpus(n_ent=2)  # 4 records: at most 3 non-self neighbors
    emb = np.eye(4, dtype=np.float32)
    mined = ann_hard_negatives(df, emb, k=10, seed=0)
    assert mined.groupby("anchor_id").size().max() <= 3


def test_miners_reject_misaligned_features():
    df = make_corpus(n_ent=4)
    with pytest.raises(ValueError, match="records"):
        bm25_hard_negatives(df, ["only one text"], k=2, seed=0)
    with pytest.raises(ValueError, match="records"):
        ann_hard_negatives(df, np.zeros((2, 8), dtype=np.float32), k=2, seed=0)
