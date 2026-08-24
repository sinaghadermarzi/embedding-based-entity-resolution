"""Tests for er_lab.blocking: matchkeys OR-union, BM25, and ANN candidates."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from er_lab.blocking import ann, matchkeys, sparse


def make_frame() -> pd.DataFrame:
    """Ten records with known key collisions (soundex, zip+initial, phone)."""
    rows = [
        # r00/r01: same soundex(SMITH/SMYTH)+dob year AND same zip+initial -> both passes
        ("r00", "SARAH", "SMITH", "1960-01-02", "27701", "9195550000"),
        ("r01", "SALLY", "SMYTH", "1960-05-06", "27701", "9195550001"),
        # r02/r03: same phone only
        ("r02", "MARK", "GARCIA", "1970-01-01", "27510", "9195559999"),
        ("r03", "NINA", "OBRIEN", "1980-02-02", "27607", "9195559999"),
        # r04/r05: same soundex+year only (different zip)
        ("r04", "PAUL", "MCDONALD", "1955-03-03", "27001", "9195550004"),
        ("r05", "PETE", "MACDONALD", "1955-09-09", "27002", "9195550005"),
        # r06: missing zip and phone -> only pass 1 can touch it; unique family
        ("r06", "IRIS", "ZBIGNIEW", "1990-04-04", None, None),
        # r07/r08/r09: shared soundex(L000)+year block of size 3
        ("r07", "ANNA", "LEE", "1985-01-01", "27801", "9195550007"),
        ("r08", "ABEL", "LEA", "1985-06-06", "27802", "9195550008"),
        ("r09", "AMOS", "LOWE", "1985-12-12", "27803", "9195550009"),
    ]
    return pd.DataFrame(
        rows,
        columns=["record_id", "given_name", "family_name", "dob", "zip", "phone"],
    ).astype("string")


PASSES = [["soundex(family_name)", "year(dob)"], ["zip", "initial(given_name)"], ["phone"]]


def test_matchkeys_or_union_dedup_and_canonical():
    out = matchkeys.candidates(make_frame(), passes=PASSES, budget_k=None)
    assert list(out.columns) == ["a", "b"]
    assert (out["a"] < out["b"]).all()
    pairs = set(zip(out["a"], out["b"]))
    assert len(pairs) == len(out)  # OR-union deduped
    # (r00, r01) matches pass 1 (S530+1960) AND pass 2 (27701+S) -> exactly once
    assert ("r00", "r01") in pairs
    stats = out.attrs["stats"]
    p1, p2, _ = stats["passes"]
    assert p1["pairs_new"] >= 1 and p2["pairs_generated"] >= 1
    assert p2["pairs_new"] == p2["pairs_generated"] - 1  # the (r00, r01) repeat deduped
    # known memberships
    assert ("r02", "r03") in pairs  # phone pass only
    assert ("r04", "r05") in pairs  # soundex M235 + year 1955
    assert {("r07", "r08"), ("r07", "r09"), ("r08", "r09")} <= pairs  # L block of 3
    # r06 has unique soundex and no zip/phone -> appears nowhere
    assert not any("r06" in p for p in pairs)
    # stats totals agree
    assert stats["n_pairs"] == len(out)
    assert stats["pairs_dropped_by_budget"] == 0


def test_matchkeys_budget_k_honored():
    out = matchkeys.candidates(make_frame(), passes=PASSES, budget_k=1)
    counts: dict[str, int] = {}
    for a, b in zip(out["a"], out["b"]):
        counts[a] = counts.get(a, 0) + 1
        counts[b] = counts.get(b, 0) + 1
    assert max(counts.values()) <= 1
    full = matchkeys.candidates(make_frame(), passes=PASSES, budget_k=None)
    assert out.attrs["stats"]["pairs_dropped_by_budget"] > 0
    assert len(out) < len(full)


def test_matchkeys_default_passes_and_determinism():
    df = make_frame()
    assert matchkeys.default_passes(df)[0] == ["soundex(family_name)", "year(dob)"]
    a = matchkeys.candidates(df, budget_k=None)
    b = matchkeys.candidates(df, budget_k=None)
    pd.testing.assert_frame_equal(a, b)
    assert len(a) > 0


def test_matchkeys_missing_keys_sit_out():
    # whitespace-only zip must not form a block with another blank zip
    df = make_frame()
    df.loc[df["record_id"] == "r06", "zip"] = "   "
    df.loc[df["record_id"] == "r09", "zip"] = "   "
    out = matchkeys.candidates(df, passes=[["zip"]], budget_k=None)
    assert not any("r06" in p or "r09" in p for p in zip(out["a"], out["b"]))


def test_matchkeys_errors():
    df = make_frame()
    with pytest.raises(ValueError, match="unknown matchkey function"):
        matchkeys.candidates(df, passes=[["metaphone(family_name)"]])
    with pytest.raises(KeyError, match="not in frame"):
        matchkeys.candidates(df, passes=[["soundex(nope)"]])
    dup = pd.concat([df, df.iloc[[0]]], ignore_index=True)
    with pytest.raises(ValueError, match="unique"):
        matchkeys.candidates(dup, passes=PASSES)
    with pytest.raises(ValueError, match="budget_k"):
        matchkeys.candidates(df, passes=PASSES, budget_k=0)


# ---------------------------------------------------------------- bm25 sparse


def sparse_frame() -> tuple[pd.DataFrame, list[str]]:
    df = pd.DataFrame({"record_id": pd.array(["r0", "r1", "r2", "r3", "r4"], dtype="string")})
    texts = [
        "william smith oakwood avenue durham",
        "william smith oakwood avenue durham",
        "elizabeth garcia main street raleigh",
        "elizabeth garcia main street raleigh",
        "zzyzx qwerty plugh",  # shares no token with anyone
    ]
    return df, texts


def test_sparse_known_neighbors():
    df, texts = sparse_frame()
    out = sparse.candidates(df, texts, k=1)
    assert list(out.columns) == ["a", "b"]
    assert (out["a"] < out["b"]).all()
    assert set(zip(out["a"], out["b"])) == {("r0", "r1"), ("r2", "r3")}
    # r4 only ever matches at score 0 -> dropped, never padded in
    assert out.attrs["stats"]["zero_score_dropped"] > 0


def test_sparse_budget_and_no_self_pairs():
    df, texts = sparse_frame()
    for k in (1, 2, 3):
        out = sparse.candidates(df, texts, k=k)
        assert (out["a"] != out["b"]).all()
        assert len(out) <= len(df) * k
        assert len(out) == len(set(zip(out["a"], out["b"])))


def test_sparse_determinism_and_errors():
    df, texts = sparse_frame()
    pd.testing.assert_frame_equal(sparse.candidates(df, texts, k=2), sparse.candidates(df, texts, k=2))
    with pytest.raises(ValueError, match="k must be"):
        sparse.candidates(df, texts, k=0)
    with pytest.raises(ValueError, match="entries"):
        sparse.candidates(df, texts[:-1], k=1)


# ------------------------------------------------------------------ dense ann


def ann_embeddings() -> tuple[pd.DataFrame, np.ndarray]:
    """Four records forming two tight cosine pairs: (r0, r1) and (r2, r3)."""
    df = pd.DataFrame({"record_id": pd.array(["r0", "r1", "r2", "r3"], dtype="string")})
    x = np.array(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.99, 0.14, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.99, 0.14],
        ],
        dtype=np.float32,
    )
    return df, x


@pytest.mark.parametrize("kind", ["flat", "hnsw"])
def test_ann_known_neighbors(kind: str):
    df, x = ann_embeddings()
    out = ann.candidates(df, x, k=1, index=kind)
    assert list(out.columns) == ["a", "b"]
    assert (out["a"] < out["b"]).all()
    assert set(zip(out["a"], out["b"])) == {("r0", "r1"), ("r2", "r3")}


@pytest.mark.parametrize("kind", ["flat", "hnsw", "ivfpq"])
def test_ann_budget_no_self_deterministic(kind: str):
    rng = np.random.default_rng(3)
    n, d = 80, 8
    x = rng.normal(size=(n, d)).astype(np.float32)
    df = pd.DataFrame({"record_id": pd.array([f"r{i:03d}" for i in range(n)], dtype="string")})
    out1 = ann.candidates(df, x, k=3, index=kind)
    out2 = ann.candidates(df, x, k=3, index=kind)
    pd.testing.assert_frame_equal(out1, out2)  # deterministic
    assert len(out1) > 0
    assert (out1["a"] != out1["b"]).all()
    assert (out1["a"] < out1["b"]).all()
    assert len(out1) <= n * 3
    assert len(out1) == len(set(zip(out1["a"], out1["b"])))
    assert out1.attrs["stats"]["method"] == f"ann_{kind}"


def test_ann_errors():
    df, x = ann_embeddings()
    with pytest.raises(ValueError, match="unknown index kind"):
        ann.candidates(df, x, k=1, index="lsh")
    with pytest.raises(ValueError, match="align"):
        ann.candidates(df, x[:2], k=1)
    with pytest.raises(ValueError, match="k must be"):
        ann.candidates(df, x, k=0)
