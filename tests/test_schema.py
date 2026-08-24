"""er_lab.data.schema — DeclaredSchema contract tests."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from er_lab.data.schema import NON_TEXT_ROLES, ROLES, ROW_SENTINEL, DeclaredSchema

REPO_ROOT = Path(__file__).resolve().parents[1]
SCHEMAS_DIR = REPO_ROOT / "configs" / "schemas"

TOY_YAML = """\
name: toy
record_id: rid
entity_id: eid
roles:
  given_name: first
  family_name: last
  street_address: [house, direction, street]
  snapshot_date: snap
extra_keep: [county_id]
"""


@pytest.fixture()
def toy_schema(tmp_path: Path) -> DeclaredSchema:
    p = tmp_path / "toy.yaml"
    p.write_text(TOY_YAML)
    return DeclaredSchema.from_yaml(p)


def toy_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "rid": ["r1", "r2", "r3"],
            "eid": ["e1", "e1", "e2"],
            "first": ["  MARIA ", "Ann", None],
            "last": ["Diaz", "Lee", "Poe"],
            "house": ["12", "", None],
            "direction": ["N", None, ""],
            "street": ["ELM ST", "OAK AVE", ""],
            "snap": ["2024-01-01", "2024-01-01", "2024-01-01"],
            "county_id": [7, 7, 9],
            "ignored": ["x", "y", "z"],
        }
    )


def test_from_yaml_roundtrip(toy_schema: DeclaredSchema) -> None:
    s = toy_schema
    assert s.name == "toy"
    assert s.record_id == "rid"
    assert s.entity_id == "eid"
    assert s.roles["given_name"] == "first"
    assert s.roles["street_address"] == ["house", "direction", "street"]
    assert s.extra_keep == ["county_id"]
    assert s.has("given_name") and s.has("street_address")
    assert not s.has("dob")


def test_from_yaml_null_entity_id(tmp_path: Path) -> None:
    p = tmp_path / "nokey.yaml"
    p.write_text("name: nokey\nrecord_id: rid\nentity_id: null\nroles:\n  full_name: name\n")
    s = DeclaredSchema.from_yaml(p)
    assert s.entity_id is None
    assert s.extra_keep == []


def test_unknown_role_rejected(tmp_path: Path) -> None:
    p = tmp_path / "bad.yaml"
    p.write_text("name: bad\nrecord_id: rid\nroles:\n  shoe_size: shoes\n")
    with pytest.raises(ValueError, match="shoe_size"):
        DeclaredSchema.from_yaml(p)


def test_to_canonical_columns_and_dtypes(toy_schema: DeclaredSchema) -> None:
    out = toy_schema.to_canonical(toy_frame())
    assert list(out.columns) == [
        "given_name",
        "family_name",
        "street_address",
        "snapshot_date",
        "record_id",
        "entity_id",
        "source",
        "county_id",
    ]
    for col in ("given_name", "family_name", "street_address", "record_id", "entity_id", "source"):
        assert out[col].dtype == "string", col
    assert list(out["source"]) == ["toy", "toy", "toy"]
    assert list(out["record_id"]) == ["r1", "r2", "r3"]
    assert list(out["entity_id"]) == ["e1", "e1", "e2"]
    # extra_keep passes through unchanged, original dtype intact
    assert list(out["county_id"]) == [7, 7, 9]
    assert "ignored" not in out.columns


def test_to_canonical_is_verbatim(toy_schema: DeclaredSchema) -> None:
    out = toy_schema.to_canonical(toy_frame())
    # no trimming, no casefolding, no standardization — PRS-01 owns that upstream
    assert out["given_name"][0] == "  MARIA "
    assert out["given_name"][2] is pd.NA


def test_multicolumn_join_order_and_empty_parts(toy_schema: DeclaredSchema) -> None:
    out = toy_schema.to_canonical(toy_frame())
    # yaml list order: house, direction, street — joined with single spaces
    assert out["street_address"][0] == "12 N ELM ST"
    # empty-string and NA parts are dropped, not joined as blanks
    assert out["street_address"][1] == "OAK AVE"
    # all parts empty/missing -> NA, not ""
    assert out["street_address"][2] is pd.NA


def test_join_keeps_whitespace_only_parts_verbatim() -> None:
    # NC snapshots space-pad blank fields (half_code=' ', street_dir=' ', ...);
    # whitespace-only parts are deliberately KEPT — the padding is part of the
    # noise under study, so the join must not quietly clean it away.
    s = DeclaredSchema(
        name="t",
        record_id=ROW_SENTINEL,
        entity_id=None,
        roles={"street": ["street_dir", "street_name", "street_sufx_cd"]},
    )
    df = pd.DataFrame({"street_dir": [" "], "street_name": [" WARD ST "], "street_sufx_cd": [" "]})
    out = s.to_canonical(df)
    assert out["street"][0] == "   WARD ST   "  # ' ' + ' ' + ' WARD ST ' + ' ' + ' '


def test_extra_keep_collision_with_reserved_rejected() -> None:
    for taken in ("record_id", "entity_id", "source"):
        with pytest.raises(ValueError, match=taken):
            DeclaredSchema(name="t", record_id="rid", entity_id=None, roles={}, extra_keep=[taken])


def test_extra_keep_collision_with_role_rejected() -> None:
    with pytest.raises(ValueError, match="given_name"):
        DeclaredSchema(
            name="t",
            record_id="rid",
            entity_id=None,
            roles={"given_name": "first"},
            extra_keep=["given_name"],
        )


def test_extra_keep_duplicates_rejected() -> None:
    with pytest.raises(ValueError, match="duplicate"):
        DeclaredSchema(
            name="t", record_id="rid", entity_id=None, roles={}, extra_keep=["county", "county"]
        )


def test_row_sentinel_record_id() -> None:
    s = DeclaredSchema(name="t", record_id=ROW_SENTINEL, entity_id=None, roles={"full_name": "n"})
    out = s.to_canonical(pd.DataFrame({"n": ["a", "b", "c"]}))
    assert list(out["record_id"]) == ["0", "1", "2"]
    assert out["record_id"].dtype == "string"


def test_entity_id_na_when_null() -> None:
    s = DeclaredSchema(name="t", record_id="rid", entity_id=None, roles={"full_name": "n"})
    out = s.to_canonical(pd.DataFrame({"rid": [1, 2], "n": ["a", "b"]}))
    assert out["entity_id"].isna().all()
    assert out["entity_id"].dtype == "string"


def test_missing_column_raises(toy_schema: DeclaredSchema) -> None:
    with pytest.raises(KeyError, match="street"):
        toy_schema.to_canonical(toy_frame().drop(columns=["street"]))


def test_text_roles_order(toy_schema: DeclaredSchema) -> None:
    # yaml order, minus non-text roles (snapshot_date is metadata, not person content)
    assert toy_schema.text_roles() == ["given_name", "family_name", "street_address"]
    assert NON_TEXT_ROLES <= ROLES


@pytest.mark.parametrize("path", sorted(SCHEMAS_DIR.glob("*.yaml")), ids=lambda p: p.stem)
def test_repo_schema_yamls_load(path: Path) -> None:
    s = DeclaredSchema.from_yaml(path)
    assert s.name == path.stem
    assert set(s.roles) <= ROLES
    assert s.record_id  # non-empty (a column name or __row__)


@pytest.mark.network
def test_fake_1000_header_matches_yaml() -> None:
    import requests

    url = (
        "https://raw.githubusercontent.com/moj-analytical-services/"
        "splink_datasets/master/data/fake_1000.csv"
    )
    resp = requests.get(url, headers={"Range": "bytes=0-256"}, timeout=30)
    resp.raise_for_status()
    header = resp.text.splitlines()[0].split(",")

    s = DeclaredSchema.from_yaml(SCHEMAS_DIR / "fake_1000.yaml")
    mapped = {s.record_id, s.entity_id}
    for spec in s.roles.values():
        mapped.update([spec] if isinstance(spec, str) else spec)
    assert mapped == set(header)
