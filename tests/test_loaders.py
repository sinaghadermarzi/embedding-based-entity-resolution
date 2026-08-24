"""er_lab.data.loaders: ONC path discovery, checksum gating, error quality, live fake_1000."""

from __future__ import annotations

import pytest

from er_lab.data import loaders
from er_lab.data.loaders import (
    _checksum_gate,
    _onc_select,
    load_bpid,
    load_fake_1000,
    load_ohio,
    load_onc,
)

# ------------------------------------------------------------------ ONC discovery (canned API JSON)

_ONC_STEM = "ONC Patient Matching Algorithm Challenge Test Dataset"
_ONC_SEGMENTS = ["A-C", "D-F", "G-I", "J-mid L", "Null", "O-R", "S", "T-Z", "mid L - N"]


def _entry(name: str) -> dict:
    return {
        "name": name,
        "type": "file",
        "download_url": (
            "https://raw.githubusercontent.com/onc-healthit/patient-matching/master/"
            + name.replace(" ", "%20")
        ),
    }


ONC_CONTENTS = [_entry(f"{_ONC_STEM}.{seg}.csv") for seg in _ONC_SEGMENTS] + [
    _entry("README.md")
]


def test_onc_select_all_files():
    selected = _onc_select(ONC_CONTENTS, None)
    names = [e["name"] for e in selected]
    assert len(names) == 9  # 8 alphabetical segments + the null-last-name file
    assert f"{_ONC_STEM}.Null.csv" in names
    assert "README.md" not in names


def test_onc_select_subset_and_case():
    selected = _onc_select(ONC_CONTENTS, ["a-c", "Null"])
    assert [e["name"] for e in selected] == [
        f"{_ONC_STEM}.A-C.csv",
        f"{_ONC_STEM}.Null.csv",
    ]
    assert selected[0]["download_url"].endswith("Dataset.A-C.csv")


def test_onc_select_multiword_segment():
    selected = _onc_select(ONC_CONTENTS, ["mid L - N", "J-mid L"])
    assert [e["name"] for e in selected] == [
        f"{_ONC_STEM}.mid L - N.csv",
        f"{_ONC_STEM}.J-mid L.csv",
    ]


def test_onc_select_unknown_segment_lists_available():
    with pytest.raises(ValueError, match="A-C"):
        _onc_select(ONC_CONTENTS, ["Q"])


# ------------------------------------------------------------------ checksum gate

def test_checksum_gate_records_then_verifies(tmp_path):
    path = tmp_path / "profiles.csv"
    path.write_text("a,b\n1,2\n")
    _checksum_gate(path)
    sidecar = tmp_path / "profiles.csv.sha256"
    assert sidecar.exists()
    _checksum_gate(path)  # unchanged file passes
    path.write_text("tampered\n")
    with pytest.raises(ValueError, match="sha256 mismatch"):
        _checksum_gate(path)


# ------------------------------------------------------------------ gated-local error quality

def test_load_bpid_missing_gives_download_instructions(tmp_path):
    with pytest.raises(FileNotFoundError) as excinfo:
        load_bpid(tmp_path)
    msg = str(excinfo.value)
    assert str(tmp_path / "raw" / "bpid") in msg  # names the exact expected directory
    assert "zenodo.org/records/13932202" in msg  # names the source
    assert "user-side" in msg  # says why the loader cannot fetch it
    assert "DATA_GOVERNANCE.md" in msg


def test_load_ohio_missing_gives_download_instructions(tmp_path):
    with pytest.raises(FileNotFoundError) as excinfo:
        load_ohio(tmp_path)
    msg = str(excinfo.value)
    assert str(tmp_path / "raw" / "ohio") in msg
    assert "ohiosos.gov" in msg
    assert "user-side" in msg
    assert "DATA_GOVERNANCE.md" in msg


# ------------------------------------------------------------------ provenance notes

def test_provenance_note_prints_once_per_process(capsys):
    loaders._provenance_note("test_source_xyz", "terms text here")
    loaders._provenance_note("test_source_xyz", "terms text here")
    out = capsys.readouterr().out
    assert out.count("terms text here") == 1


# ------------------------------------------------------------------ live round-trip (network)

FAKE_1000_FIXTURE_YAML = """\
# test fixture — used only when configs/schemas/fake_1000.yaml does not exist yet
name: fake_1000
record_id: unique_id
entity_id: cluster
roles:
  given_name: first_name
  family_name: surname
  dob: dob
  city: city
  email: email
extra_keep: []
"""


@pytest.mark.network
def test_load_fake_1000_live_round_trip(tmp_path):
    schema_path = loaders.SCHEMA_DIR / "fake_1000.yaml"
    if not schema_path.exists():
        schema_path = tmp_path / "fake_1000.yaml"
        schema_path.write_text(FAKE_1000_FIXTURE_YAML)

    df, schema = load_fake_1000(tmp_path / "data", schema_path=schema_path)

    assert len(df) == 1000
    assert schema.name == "fake_1000"
    for role in schema.roles:
        assert role in df.columns
    for col in ("record_id", "entity_id", "source"):
        assert col in df.columns
        assert str(df[col].dtype) == "string"
    assert df["record_id"].is_unique
    assert (df["source"] == "fake_1000").all()
    # entity_id (the 'cluster' truth key) is populated on every row
    assert df["entity_id"].notna().all()
    assert (df["entity_id"].str.len() > 0).all()
    # verbatim spot check against the known first row of the csv
    first = df[df["record_id"] == "0"].iloc[0]
    assert first["given_name"] == "Robert"
    # a second call must reuse the downloaded file (and not re-print the note)
    df2, _ = load_fake_1000(tmp_path / "data", schema_path=schema_path)
    assert len(df2) == 1000


@pytest.mark.network
def test_load_onc_null_segment_live_dob_verbatim(tmp_path, monkeypatch):
    # Discovery via api.github.com is unit-tested against canned JSON above; the build
    # sandbox reaches raw.githubusercontent.com only, so pin the smallest segment here.
    monkeypatch.setattr(loaders, "_onc_entries", lambda: [_entry(f"{_ONC_STEM}.Null.csv")])
    schema_path = loaders.SCHEMA_DIR / "onc.yaml"
    if not schema_path.exists():
        pytest.skip("configs/schemas/onc.yaml not built yet")

    df, schema = load_onc(tmp_path / "data", letters=["Null"], schema_path=schema_path)

    assert schema.name == "onc"
    assert len(df) > 1000
    assert df["entity_id"].notna().all()  # EnterpriseID, the truth key
    assert df["record_id"].is_unique  # synthesized __row__ ids
    # THE domain rule: DOB arrives as Excel-style serial integers and must stay
    # verbatim strings — never parsed into dates (the noise is the object of study).
    assert str(df["dob"].dtype) == "string"
    assert df["dob"].str.fullmatch(r"\d*").all()
    assert (df["dob"].str.len() > 0).mean() > 0.5  # mostly populated, plain serials
