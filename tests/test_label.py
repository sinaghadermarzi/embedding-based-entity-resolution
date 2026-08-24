"""Tests for er_lab.label.widget: queue determinism, JSONL persistence, kappa."""

from __future__ import annotations

import pandas as pd
import pytest

from er_lab.label.widget import (
    LABELS,
    AdjudicationWidget,
    PairQueue,
    cohens_kappa,
    latest_labels,
    read_labels,
)

RECORDS = pd.DataFrame(
    {
        "record_id": ["r1", "r2", "r3", "r4"],
        "given_name": ["ann", "anne", "bob", "rob"],
        "family_name": ["li", "li", "moss", "moss"],
    }
)


def queue(tmp_path=None, pairs=None) -> PairQueue:
    if pairs is None:
        # deliberately unsorted, mis-oriented, and with a duplicate
        pairs = pd.DataFrame(
            {"a": ["r3", "r2", "r1", "r2"], "b": ["r4", "r1", "r2", "r1"],
             "prob": [0.7, 0.9, 0.9, 0.9]}
        )
    path = None if tmp_path is None else tmp_path / "labels.jsonl"
    return PairQueue(pairs, RECORDS, path=path)


# -------------------------------------------------------------------- queue


def test_queue_iteration_order_deterministic():
    q = queue()
    ids = [item["pair_id"] for item in q]
    # canonical a<b orientation, sorted, deduped — independent of input order
    assert ids == ["r1||r2", "r3||r4"]
    assert len(q) == 2
    shuffled = pd.DataFrame({"a": ["r4", "r1"], "b": ["r3", "r2"]})
    assert queue(pairs=shuffled).pair_ids() == ["r1||r2", "r3||r4"]


def test_queue_items_carry_both_records_and_extras():
    item = next(iter(queue()))
    assert item["record_a"]["given_name"] == "ann"
    assert item["record_b"]["given_name"] == "anne"
    assert item["prob"] == 0.9


def test_queue_validation():
    with pytest.raises(ValueError, match="self-pairs"):
        PairQueue(pd.DataFrame({"a": ["r1"], "b": ["r1"]}), RECORDS)
    with pytest.raises(ValueError, match="absent"):
        PairQueue(pd.DataFrame({"a": ["r1"], "b": ["zz"]}), RECORDS)
    with pytest.raises(KeyError, match="missing"):
        PairQueue(pd.DataFrame({"a": ["r1"]}), RECORDS)


# ---------------------------------------------------------- JSONL round-trip


def test_record_label_jsonl_round_trip(tmp_path):
    q = queue(tmp_path)
    q.record_label("r1||r2", "match", "alice")
    q.record_label("r3||r4", "nonmatch", "alice")
    q.record_label("r1||r2", "unsure", "bob")
    labels = read_labels(q.path)
    assert list(labels.columns) == ["pair_id", "label", "annotator", "at"]
    assert len(labels) == 3  # append-only: full history
    assert labels.iloc[0]["pair_id"] == "r1||r2"
    assert labels.iloc[0]["label"] == "match"
    assert labels.iloc[2]["annotator"] == "bob"
    # change of mind: new line appended, latest_labels resolves it
    q.record_label("r1||r2", "nonmatch", "alice")
    labels = read_labels(q.path)
    assert len(labels) == 4
    last = latest_labels(labels)
    alice = last[(last["annotator"] == "alice") & (last["pair_id"] == "r1||r2")]
    assert alice["label"].tolist() == ["nonmatch"]


def test_record_label_validation(tmp_path):
    q = queue(tmp_path)
    with pytest.raises(ValueError, match="unknown label"):
        q.record_label("r1||r2", "maybe", "alice")
    with pytest.raises(KeyError, match="not in this queue"):
        q.record_label("r1||r9", "match", "alice")
    with pytest.raises(ValueError, match="no path"):
        queue().record_label("r1||r2", "match", "alice")


def test_pending_resumes_per_annotator(tmp_path):
    q = queue(tmp_path)
    assert q.pending("alice") == ["r1||r2", "r3||r4"]
    q.record_label("r1||r2", "match", "alice")
    assert q.pending("alice") == ["r3||r4"]
    assert q.pending("bob") == ["r1||r2", "r3||r4"]  # bob starts fresh


# -------------------------------------------------------------------- kappa


def series(vals, idx=None) -> pd.Series:
    return pd.Series(vals, index=idx or [f"p{i}" for i in range(len(vals))])


def test_kappa_perfect_agreement_is_one():
    a = series(["match", "nonmatch", "unsure", "match"])
    assert cohens_kappa(a, a.copy()) == pytest.approx(1.0)


def test_kappa_independent_hand_computed_zero():
    # 2x2 with po == pe == 0.5 -> kappa exactly 0
    a = series(["match", "match", "nonmatch", "nonmatch"])
    b = series(["match", "nonmatch", "match", "nonmatch"])
    assert cohens_kappa(a, b) == pytest.approx(0.0)


def test_kappa_partial_agreement_hand_computed():
    # confusion counts: mm=5, nn=3, mn=1, nm=1 over 10 pairs
    # po = 0.8; pe = 0.6*0.6 + 0.4*0.4 = 0.52; kappa = 0.28/0.48 = 7/12
    a = series(["m"] * 6 + ["n"] * 4)
    b = series(["m"] * 5 + ["n"] + ["m"] + ["n"] * 3)
    assert cohens_kappa(a, b) == pytest.approx(7 / 12)


def test_kappa_aligns_on_common_index_and_degenerate_case():
    a = series(["m", "m", "n"], idx=["p1", "p2", "p3"])
    b = series(["m", "n"], idx=["p2", "p9"])  # only p2 in common; they agree
    assert cohens_kappa(a, b) == pytest.approx(1.0)  # pe=1 degenerate -> 1.0
    with pytest.raises(ValueError, match="no common"):
        cohens_kappa(a, series(["m"], idx=["zz"]))


# ------------------------------------------------------------------- widget


def test_adjudication_widget_constructs(tmp_path):
    q = queue(tmp_path)
    w = AdjudicationWidget(q, annotator="alice")
    assert len(w._buttons) == len(LABELS)
    assert "0/2" in w._progress.value
    # clicking a button persists + advances (button behavior, no display needed)
    w._on_click(w._buttons[0])
    assert q.pending("alice") == ["r3||r4"]
    assert "1/2" in w._progress.value


def test_adjudication_widget_requires_persisting_queue():
    with pytest.raises(ValueError, match="path"):
        AdjudicationWidget(queue())
