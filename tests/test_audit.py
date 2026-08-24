"""er_lab.noise.audit — diff taxonomy + Wilson prevalence contract tests."""

from __future__ import annotations

import ast
import inspect

import pandas as pd
import pytest
from scipy.stats import binomtest

import er_lab.noise.audit as audit_module
from er_lab.noise.audit import audit_report, classify_pair_diffs, prevalence

FIELDS = ["given_name", "family_name", "phone"]

#: tiny in-test lexicon: canonical_lower -> variants_lower (load_lexicon's shape)
LEXICON = {"william": {"bill", "will"}, "elizabeth": {"liz", "beth"}}

NA = pd.NA


def aligned() -> pd.DataFrame:
    """11 pairs covering every category (fields: given_name, family_name, phone)."""
    return pd.DataFrame(
        {
            "ncid": [f"r{i:02d}" for i in range(1, 12)],
            "given_name_a": ["MARIA", NA, "JOHN", "JOHN", "JOHN", "Bill", "MARIA", "ANA", "ANA", "RALEIGH", "smith"],
            "given_name_b": ["MARIA", "ANN", "JOHN", "JOHN", "JOHN", "William", "GARCIA", "ANA", "ANA", "DURHAM", "SMYTH"],
            "family_name_a": ["DIAZ", "LEE", "SMITH", "SMITH", "SMITH", "POE", "GARCIA", "LI", "OBRIEN", "KIM", "DOE"],
            "family_name_b": ["DIAZ", "", "SMYTH", "SMYTHE", "SMYTHES", "POE", "MARIA", "LI", "O'Brien", "KIM", "DOE"],
            "phone_a": [NA, "5551", "111", "111", "111", "222", "333", "(919) 555-1212", "444", "555", "666"],
            "phone_b": [NA, "5551", "111", "111", "111", "222", "333", "9195551212", "444", "555", "666"],
        }
    )


EXPECTED = {
    "given_name": {
        "r01": "identical",     # verbatim equal
        "r02": "missing_gain",  # NA -> value
        "r03": "identical",
        "r04": "identical",
        "r05": "identical",
        "r06": "nickname",      # Bill <-> William via lexicon
        "r07": "swap",          # given<->family exchanged
        "r08": "identical",
        "r09": "identical",
        "r10": "wholesale",     # unrelated values
        "r11": "typo",          # smith vs SMYTH: case-insensitive distance 1
    },
    "family_name": {
        "r01": "identical",
        "r02": "missing_loss",  # value -> '' (blank counts as missing)
        "r03": "typo",          # SMITH vs SMYTH: distance 1
        "r04": "typo",          # SMITH vs SMYTHE: distance exactly 2 (boundary in)
        "r05": "wholesale",     # SMITH vs SMYTHES: distance 3 (boundary out)
        "r06": "identical",
        "r07": "swap",
        "r08": "identical",
        "r09": "format_drift",  # OBRIEN vs O'Brien: same alnum skeleton, not typo
        "r10": "identical",
        "r11": "identical",
    },
    "phone": {
        "r01": "identical",     # missing on both sides
        "r02": "identical",
        "r03": "identical",
        "r04": "identical",
        "r05": "identical",
        "r06": "identical",
        "r07": "identical",
        "r08": "format_drift",  # (919) 555-1212 vs 9195551212
        "r09": "identical",
        "r10": "identical",
        "r11": "identical",
    },
}


def as_map(long_df: pd.DataFrame) -> dict[tuple[str, str], str]:
    return {(r.field, r.key): r.category for r in long_df.itertuples()}


def test_classify_covers_every_category() -> None:
    long_df = classify_pair_diffs(aligned(), fields=FIELDS, lexicon=LEXICON)
    assert list(long_df.columns) == ["key", "field", "category"]
    assert len(long_df) == len(FIELDS) * 11
    got = as_map(long_df)
    for field, per_key in EXPECTED.items():
        for key, category in per_key.items():
            assert got[(field, key)] == category, (field, key)
    # the fixture exercises the full taxonomy
    assert set(long_df["category"]) == set(audit_module.CATEGORIES)


def test_classify_row_order_is_field_blocks_in_aligned_order() -> None:
    long_df = classify_pair_diffs(aligned(), fields=FIELDS, lexicon=LEXICON)
    for i, field in enumerate(FIELDS):
        block = long_df.iloc[i * 11 : (i + 1) * 11]
        assert (block["field"] == field).all()
        assert block["key"].tolist() == aligned()["ncid"].tolist()


def test_nickname_needs_lexicon() -> None:
    got = as_map(classify_pair_diffs(aligned(), fields=["given_name"]))
    # Bill vs William: distance > 2 and no lexicon -> wholesale, never nickname
    assert got[("given_name", "r06")] == "wholesale"


def test_classify_is_deterministic() -> None:
    pd.testing.assert_frame_equal(
        classify_pair_diffs(aligned(), fields=FIELDS, lexicon=LEXICON),
        classify_pair_diffs(aligned(), fields=FIELDS, lexicon=LEXICON),
    )


def test_classify_validation_errors() -> None:
    with pytest.raises(ValueError, match="fields"):
        classify_pair_diffs(aligned(), fields=[])
    with pytest.raises(KeyError, match="dob"):
        classify_pair_diffs(aligned(), fields=["dob"])
    with pytest.raises(KeyError, match="key"):
        classify_pair_diffs(aligned().drop(columns=["ncid"]), fields=["phone"])


def test_prevalence_wilson_known_answer() -> None:
    long_df = pd.DataFrame(
        {
            "key": [str(i) for i in range(10)],
            "field": ["given_name"] * 10,
            "category": ["typo"] * 3 + ["identical"] * 7,
        }
    )
    prev = prevalence(long_df)
    assert list(prev.columns) == ["field", "category", "n", "rate", "ci_low", "ci_high"]
    row = prev[prev["category"] == "typo"].iloc[0]
    assert row["n"] == 3
    assert row["rate"] == pytest.approx(0.3)
    ci = binomtest(3, 10).proportion_ci(confidence_level=0.95, method="wilson")
    assert row["ci_low"] == pytest.approx(ci.low, abs=1e-12)
    assert row["ci_high"] == pytest.approx(ci.high, abs=1e-12)


def test_prevalence_by_group_sums_to_totals() -> None:
    long_df = pd.DataFrame(
        {
            "key": [str(i) for i in range(12)],
            "field": ["phone"] * 12,
            "category": ["typo"] * 4 + ["identical"] * 2 + ["typo"] * 1 + ["identical"] * 5,
            "g": ["x"] * 6 + ["y"] * 6,
        }
    )
    total = prevalence(long_df).set_index(["field", "category"])["n"]
    grouped = prevalence(long_df, by=["g"])
    summed = grouped.groupby(["field", "category"], observed=True)["n"].sum()
    for idx, n in summed.items():
        assert n == total[idx]
    # rates within each (field, group) cell sum to 1
    for _, cell in grouped.groupby(["field", "g"], observed=True):
        assert cell["rate"].sum() == pytest.approx(1.0)
    # known cells: group x has 4/6 typo, group y has 1/6 typo
    by_cell = grouped.set_index(["category", "g"])
    assert by_cell.loc[("typo", "x"), "n"] == 4
    assert by_cell.loc[("typo", "y"), "rate"] == pytest.approx(1 / 6)


def test_prevalence_missing_columns_raise() -> None:
    long_df = classify_pair_diffs(aligned(), fields=["phone"])
    with pytest.raises(KeyError, match="race"):
        prevalence(long_df, by=["race"])


def test_audit_report_ties_classify_and_prevalence() -> None:
    report = audit_report(aligned(), FIELDS, lexicon=LEXICON)
    expected = prevalence(classify_pair_diffs(aligned(), fields=FIELDS, lexicon=LEXICON))
    pd.testing.assert_frame_equal(report, expected)


def test_audit_report_by_group_breakdown() -> None:
    df = aligned()
    df["county_a"] = pd.Series(["07"] * 6 + ["92"] * 5, dtype="string")
    report = audit_report(df, FIELDS, lexicon=LEXICON, by=["county_a"])
    assert list(report.columns) == [
        "field", "category", "county_a", "n", "rate", "ci_low", "ci_high",
    ]
    # every pair lands in exactly one category per field, per group
    per_group = report.groupby(["field", "county_a"], observed=True)["n"].sum()
    for field in FIELDS:
        assert per_group[(field, "07")] == 6
        assert per_group[(field, "92")] == 5
    # grouped counts sum to the ungrouped counts
    total = audit_report(aligned(), FIELDS, lexicon=LEXICON).set_index(["field", "category"])["n"]
    summed = report.groupby(["field", "category"], observed=True)["n"].sum()
    for idx, n in summed.items():
        assert n == total[idx]


def test_audit_is_independent_of_the_generator() -> None:
    # NSE-02 scores the generator against this taxonomy: audit must not import it
    tree = ast.parse(inspect.getsource(audit_module))
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names = [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom):
            names = [node.module or ""] + [alias.name for alias in node.names]
        else:
            continue
        assert not any("channels" in name for name in names), ast.dump(node)
