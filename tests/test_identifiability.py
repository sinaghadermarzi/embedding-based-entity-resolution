import pandas as pd
import pytest

from er_lab.eval.identifiability import conditioned, unresolvable_mask, unresolvable_report

FIELDS = ["given_name", "family_name", "birth_year"]


def _frame():
    """r1/r2: exact twins across entities; r3: near-miss (one field differs);
    r4: unique; r5/r6: same tuple but SAME entity (ordinary duplicates)."""
    return pd.DataFrame(
        {
            "record_id": ["r1", "r2", "r3", "r4", "r5", "r6"],
            "entity_id": ["e1", "e2", "e1", "e3", "e4", "e4"],
            "given_name": ["ANNA", "ANNA", "ANNA", "BEN", "CARL", "CARL"],
            "family_name": ["LEE", "LEE", "LEE", "LEE", "MOSS", "MOSS"],
            "birth_year": ["1990", "1990", "1991", "1990", "1970", "1970"],
            "county": ["A", "A", "A", "A", "B", "B"],
        }
    ).astype("string")


def test_unresolvable_mask_twins_and_near_miss():
    mask = unresolvable_mask(_frame(), fields=FIELDS)
    assert mask.dtype == bool
    assert list(mask.index) == ["r1", "r2", "r3", "r4", "r5", "r6"]
    assert bool(mask["r1"]) and bool(mask["r2"])  # same fields, different entity
    assert not bool(mask["r3"])  # near-miss: birth_year differs
    assert not bool(mask["r4"])  # unique tuple
    # same tuple within ONE entity is resolvable (ordinary duplicate, not a twin)
    assert not bool(mask["r5"]) and not bool(mask["r6"])


def test_unresolvable_mask_missing_values_compare_equal():
    df = pd.DataFrame(
        {
            "record_id": ["a", "b", "c"],
            "entity_id": ["e1", "e2", "e3"],
            "given_name": ["JO", "JO", "JO"],
            "family_name": ["KIM", "KIM", "KIM"],
            "birth_year": [pd.NA, pd.NA, "1950"],
        }
    ).astype("string")
    mask = unresolvable_mask(df, fields=FIELDS)
    # NA birth_year == NA birth_year: a and b present identical information
    assert bool(mask["a"]) and bool(mask["b"])
    assert not bool(mask["c"])


def test_unresolvable_mask_validation():
    df = _frame()
    with pytest.raises(KeyError, match="missing from frame"):
        unresolvable_mask(df, fields=["not_a_field"])
    with pytest.raises(ValueError, match="at least one column"):
        unresolvable_mask(df, fields=[])
    with pytest.raises(ValueError, match="entity_id has missing"):
        unresolvable_mask(df.assign(entity_id=pd.NA), fields=FIELDS)
    dup = pd.concat([df, df.iloc[[0]]], ignore_index=True)
    with pytest.raises(ValueError, match="record_id values must be unique"):
        unresolvable_mask(dup, fields=FIELDS)


def test_unresolvable_report_overall_and_per_group():
    report = unresolvable_report(_frame(), FIELDS, by=["county"])
    assert list(report.columns) == ["county", "n", "n_unresolvable", "rate"]
    overall = report.iloc[0]
    assert pd.isna(overall["county"])
    assert overall["n"] == 6
    assert overall["n_unresolvable"] == 2
    assert overall["rate"] == pytest.approx(2 / 6)
    by_group = report.iloc[1:].set_index("county")
    assert by_group.loc["A", "n"] == 4
    assert by_group.loc["A", "n_unresolvable"] == 2
    assert by_group.loc["A", "rate"] == pytest.approx(0.5)
    assert by_group.loc["B", "n"] == 2
    assert by_group.loc["B", "n_unresolvable"] == 0
    assert by_group.loc["B", "rate"] == 0.0


def test_unresolvable_report_no_groups():
    report = unresolvable_report(_frame(), FIELDS)
    assert list(report.columns) == ["n", "n_unresolvable", "rate"]
    assert len(report) == 1
    assert report.loc[0, "rate"] == pytest.approx(2 / 6)
    with pytest.raises(KeyError, match="'by' columns missing"):
        unresolvable_report(_frame(), FIELDS, by=["nope"])


def test_conditioned_drops_unresolvable_records():
    df = _frame()
    mask = unresolvable_mask(df, fields=FIELDS)
    ids = list(df["record_id"])
    pred = pd.Series(["c1", "c1", "c2", "c3", "c4", "c4"], index=ids)
    truth = pd.Series(df["entity_id"].tolist(), index=ids)
    pred_sub, truth_sub = conditioned(pred, truth, mask)
    assert list(pred_sub.index) == ["r3", "r4", "r5", "r6"]
    assert list(truth_sub.index) == ["r3", "r4", "r5", "r6"]
    assert (truth_sub == truth.loc[truth_sub.index]).all()
    # differently-ordered truth still aligns by index
    _pred_sub2, truth_sub2 = conditioned(pred, truth.iloc[::-1], mask)
    pd.testing.assert_series_equal(truth_sub2, truth_sub)


def test_conditioned_raises_on_mismatch():
    df = _frame()
    mask = unresolvable_mask(df, fields=FIELDS)
    ids = list(df["record_id"])
    pred = pd.Series("c", index=ids)
    truth = pd.Series("e", index=ids)
    with pytest.raises(ValueError, match="same record ids"):
        conditioned(pred, truth.iloc[:-1], mask)
    with pytest.raises(KeyError, match="mask does not cover"):
        conditioned(pred, truth, mask.iloc[:-1])
    with pytest.raises(TypeError, match="mask must be boolean"):
        conditioned(pred, truth, mask.astype(int))
    dup_pred = pd.concat([pred, pred.iloc[[0]]])
    with pytest.raises(ValueError, match="duplicate record ids"):
        conditioned(dup_pred, truth, mask)
