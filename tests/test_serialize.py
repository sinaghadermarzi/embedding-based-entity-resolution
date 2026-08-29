"""Tests for er_lab.serialize (TRN-05's serialization factor)."""

from __future__ import annotations

import pandas as pd
import pytest

from er_lab.serialize import MISSING_TOKEN, serialize_frame, serialize_record

ROW = {"given_name": "ANA", "family_name": pd.NA, "zip": "27510"}
TEXT_ROLES = ["given_name", "family_name", "zip"]


def test_missing_token_literal():
    assert MISSING_TOKEN == "[MISSING]"


# --- exact known answers, scheme x missing ---------------------------------


def test_colval_token_exact():
    assert (
        serialize_record(ROW, text_roles=TEXT_ROLES)
        == "[COL] given_name [VAL] ANA [COL] family_name [VAL] [MISSING] [COL] zip [VAL] 27510"
    )


def test_colval_drop_exact():
    assert (
        serialize_record(ROW, text_roles=TEXT_ROLES, missing="drop")
        == "[COL] given_name [VAL] ANA [COL] zip [VAL] 27510"
    )


def test_template_token_exact():
    assert (
        serialize_record(ROW, text_roles=TEXT_ROLES, scheme="template")
        == "given_name: ANA | family_name: [MISSING] | zip: 27510"
    )


def test_template_drop_exact():
    assert (
        serialize_record(ROW, text_roles=TEXT_ROLES, scheme="template", missing="drop")
        == "given_name: ANA | zip: 27510"
    )


def test_json_token_exact():
    assert (
        serialize_record(ROW, text_roles=TEXT_ROLES, scheme="json")
        == '{"given_name":"ANA","family_name":"[MISSING]","zip":"27510"}'
    )


def test_json_drop_exact():
    assert (
        serialize_record(ROW, text_roles=TEXT_ROLES, scheme="json", missing="drop")
        == '{"given_name":"ANA","zip":"27510"}'
    )


def test_bare_token_exact():
    assert serialize_record(ROW, text_roles=TEXT_ROLES, scheme="bare") == "ANA [MISSING] 27510"


def test_bare_drop_exact():
    assert (
        serialize_record(ROW, text_roles=TEXT_ROLES, scheme="bare", missing="drop") == "ANA 27510"
    )


# --- missing definition -----------------------------------------------------


@pytest.mark.parametrize("missing_value", [pd.NA, None, float("nan"), ""])
def test_na_none_nan_and_empty_all_count_as_missing(missing_value):
    row = {"given_name": missing_value, "family_name": "LEE", "zip": "27510"}
    assert (
        serialize_record(row, text_roles=TEXT_ROLES, scheme="bare", missing="drop") == "LEE 27510"
    )


def test_role_absent_from_row_is_missing():
    assert (
        serialize_record({"zip": "27510"}, text_roles=TEXT_ROLES, scheme="template")
        == "given_name: [MISSING] | family_name: [MISSING] | zip: 27510"
    )


def test_values_rendered_verbatim_no_cleaning():
    row = {"given_name": "  ANA ", "family_name": "o'brien", "zip": "27510"}
    assert serialize_record(row, text_roles=TEXT_ROLES, scheme="bare") == "  ANA  o'brien 27510"


# --- field order ------------------------------------------------------------


def test_field_order_permutation_exact():
    out = serialize_record(
        ROW, text_roles=TEXT_ROLES, field_order=["zip", "given_name", "family_name"]
    )
    assert out == (
        "[COL] zip [VAL] 27510 [COL] given_name [VAL] ANA [COL] family_name [VAL] [MISSING]"
    )


def test_field_order_changes_json_key_order():
    out = serialize_record(
        ROW,
        text_roles=TEXT_ROLES,
        scheme="json",
        missing="drop",
        field_order=["zip", "given_name", "family_name"],
    )
    assert out == '{"zip":"27510","given_name":"ANA"}'


@pytest.mark.parametrize(
    "bad_order",
    [["given_name", "zip"], ["given_name", "family_name", "zip", "city"], ["a", "b", "c"]],
)
def test_field_order_must_be_permutation(bad_order):
    with pytest.raises(ValueError, match="permutation"):
        serialize_record(ROW, text_roles=TEXT_ROLES, field_order=bad_order)


def test_unknown_scheme_and_missing_raise():
    with pytest.raises(ValueError, match="scheme"):
        serialize_record(ROW, text_roles=TEXT_ROLES, scheme="xml")
    with pytest.raises(ValueError, match="missing"):
        serialize_record(ROW, text_roles=TEXT_ROLES, missing="skip")


# --- serialize_frame --------------------------------------------------------


def test_frame_alignment_and_values():
    df = pd.DataFrame(
        {"given_name": ["ANA", ""], "zip": ["27510", "10001"]}, dtype="string", index=[5, 9]
    )
    out = serialize_frame(df, text_roles=TEXT_ROLES, scheme="bare")
    assert list(out.index) == [5, 9]
    assert out.dtype == "string"
    # family_name column is absent entirely; row 9's given_name is '' -> both missing
    assert out.loc[5] == "ANA [MISSING] 27510"
    assert out.loc[9] == "[MISSING] [MISSING] 10001"


def test_frame_matches_record_serialization():
    df = pd.DataFrame(
        {"given_name": ["ANA", "BO"], "family_name": [pd.NA, "LEE"], "zip": ["27510", pd.NA]},
        dtype="string",
    )
    out = serialize_frame(df, text_roles=TEXT_ROLES, scheme="template", missing="drop")
    expected = [
        serialize_record(rec, text_roles=TEXT_ROLES, scheme="template", missing="drop")
        for rec in df.to_dict("records")
    ]
    assert out.tolist() == expected


def test_frame_field_order_validated():
    df = pd.DataFrame({"given_name": ["ANA"]}, dtype="string")
    with pytest.raises(ValueError, match="permutation"):
        serialize_frame(df, text_roles=TEXT_ROLES, field_order=["given_name"])
