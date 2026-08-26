"""Tests for er_lab.noise.channels (notebook 05's calibrated dirt machine)."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from er_lab.noise import channels as C

FIXTURE_LEXICON = {"william": {"bill"}, "robert": {"bob", "rob"}}

EXPECTED_CHANNELS = {
    "typo",
    "ocr",
    "phonetic_spelling",
    "nickname",
    "name_order_swap",
    "field_dropout",
    "field_format_drift",
    "suffix_confusion",
    "hub_value",
    "raw_unparse",
}


def make_frame(n: int = 60) -> pd.DataFrame:
    # compact fixture data
    # fmt: off
    given = ["WILLIAM", "Robert", "ELIZABETH", "MARGARET", "JAMES", "Katherine", "MICHAEL", "PATRICIA"]
    fam = ["SMITH", "GARCIA LOPEZ", "JOHNSON", "MCDONALD", "OBRIEN", "LEE", "MARTINEZ", "BROWN"]
    # fmt: on
    rows = {
        "record_id": [f"r{i:05d}" for i in range(n)],
        "entity_id": [pd.NA] * n,
        "given_name": [given[i % 8] for i in range(n)],
        "middle_name": [
            ("JOSEFA" if i % 3 == 0 else ("ANN" if i % 3 == 1 else pd.NA)) for i in range(n)
        ],
        "family_name": [fam[i % 8] for i in range(n)],
        "name_suffix": [("JR" if i % 5 == 0 else pd.NA) for i in range(n)],
        "dob": [f"19{60 + i % 40:02d}-{1 + i % 12:02d}-{1 + i % 28:02d}" for i in range(n)],
        "sex": ["F" if i % 2 else "M" for i in range(n)],
        "race": ["W" if i % 2 else "B" for i in range(n)],
        "house_number": [str(100 + i) for i in range(n)],
        "street": ["MAIN STREET" if i % 2 else "OAK AVENUE" for i in range(n)],
        "unit": [("APT 2" if i % 4 == 0 else pd.NA) for i in range(n)],
        "city": ["DURHAM"] * n,
        "state": ["NC"] * n,
        "zip": [f"275{i % 100:02d}" for i in range(n)],
        "phone": [f"919555{i % 10000:04d}" for i in range(n)],
        "email": [f"p{i}@example.com" for i in range(n)],
        "source": ["test"] * n,
    }
    return pd.DataFrame(rows, dtype="string")


def eq(a, b) -> bool:
    a_na, b_na = bool(pd.isna(a)), bool(pd.isna(b))
    if a_na or b_na:
        return a_na and b_na
    return a == b


CHANNEL_BUILDS = [
    ("typo", C.typo),
    ("ocr", C.ocr),
    ("phonetic_spelling", C.phonetic_spelling),
    ("nickname", lambda: C.nickname(lexicon=FIXTURE_LEXICON)),
    ("name_order_swap", C.name_order_swap),
    ("field_dropout", C.field_dropout),
    ("field_format_drift", C.field_format_drift),
    ("suffix_confusion", C.suffix_confusion),
    ("hub_value", C.hub_value),
    ("raw_unparse", C.raw_unparse),
]


def test_registry_names_and_channel_protocol():
    assert set(C.CHANNELS) == EXPECTED_CHANNELS
    for name, factory in C.CHANNELS.items():
        ch = factory()
        assert ch.name == name
        assert callable(ch.apply)


@pytest.mark.parametrize("name,build", CHANNEL_BUILDS)
def test_seed_determinism(name, build):
    df = make_frame()
    runs = []
    for _ in range(2):
        out, ops = build().apply(df, np.random.default_rng(42), pd.Series(0.5, index=df.index))
        runs.append((out, ops))
    pd.testing.assert_frame_equal(runs[0][0], runs[1][0])
    pd.testing.assert_frame_equal(runs[0][1], runs[1][1])


@pytest.mark.parametrize("name,build", CHANNEL_BUILDS)
def test_ops_log_exactness(name, build):
    """Every changed cell appears in ops with correct before/after; unchanged absent."""
    df = make_frame()
    pristine = df.copy()
    out, ops = build().apply(df, np.random.default_rng(7), pd.Series(0.8, index=df.index))

    pd.testing.assert_frame_equal(df, pristine)  # input never mutated
    assert list(ops.columns) == C.OPS_COLUMNS
    assert (ops["channel"] == name).all()

    diffs: dict[tuple[str, str], tuple] = {}
    cols = set(df.columns) | set(out.columns)
    for i in range(len(df)):
        rid = str(df["record_id"].iloc[i])
        for col in cols:
            if col == "record_id":
                continue
            before = df[col].iloc[i] if col in df.columns else pd.NA
            after = out[col].iloc[i]
            if not eq(before, after):
                diffs[(rid, col)] = (before, after)

    logged = {
        (str(r.record_id), str(r.field)): (r.before, r.after) for r in ops.itertuples(index=False)
    }
    assert set(diffs) == set(logged)
    for key, (before, after) in diffs.items():
        assert eq(before, logged[key][0]), (key, before, logged[key])
        assert eq(after, logged[key][1]), (key, after, logged[key])


def test_rate_honored_binomial_dropout():
    n, rate = 2000, 0.2
    df = make_frame(n)
    _, ops = C.field_dropout().apply(df, np.random.default_rng(5), pd.Series(rate, index=df.index))
    observed = ops["record_id"].nunique() / n
    tol = 4 * math.sqrt(rate * (1 - rate) / n)  # ~0.036
    assert abs(observed - rate) < tol, (observed, rate, tol)


def test_rate_honored_binomial_typo():
    n, rate = 2000, 0.2
    df = make_frame(n)
    _, ops = C.typo().apply(df, np.random.default_rng(6), pd.Series(rate, index=df.index))
    observed = ops["record_id"].nunique() / n
    tol = 4 * math.sqrt(rate * (1 - rate) / n) + 0.01  # + slack for rare no-op edits
    assert abs(observed - rate) < tol, (observed, rate, tol)


def test_typo_edits_are_small():
    jellyfish = pytest.importorskip("jellyfish")
    df = make_frame(200)
    _, ops = C.typo().apply(df, np.random.default_rng(3), pd.Series(0.5, index=df.index))
    assert len(ops) > 20
    for r in ops.itertuples(index=False):
        assert 1 <= jellyfish.levenshtein_distance(str(r.before), str(r.after)) <= 2


def test_ocr_hits_digit_fields():
    df = make_frame(200)
    _, ops = C.ocr().apply(df, np.random.default_rng(4), pd.Series(0.8, index=df.index))
    assert len(ops) > 20
    assert (ops["before"] != ops["after"]).all()


def test_phonetic_spelling_changes_values():
    df = make_frame(200)
    _, ops = C.phonetic_spelling().apply(
        df, np.random.default_rng(8), pd.Series(0.9, index=df.index)
    )
    assert len(ops) > 10
    assert (ops["before"] != ops["after"]).all()


def test_nickname_case_pattern_preserved():
    df = pd.DataFrame(
        {
            "record_id": ["a", "b", "c"],
            "given_name": ["WILLIAM", "William", "william"],
            "family_name": ["SMITH", "Smith", "smith"],
        },
        dtype="string",
    )
    ch = C.nickname(lexicon={"william": {"bill"}})
    out, ops = ch.apply(df, np.random.default_rng(0), pd.Series(1.0, index=df.index))
    assert out["given_name"].tolist() == ["BILL", "Bill", "bill"]
    assert len(ops) == 3
    assert set(zip(ops["before"], ops["after"])) == {
        ("WILLIAM", "BILL"),
        ("William", "Bill"),
        ("william", "bill"),
    }


def test_nickname_reverse_direction():
    ch = C.nickname(lexicon={"william": {"bill"}})
    df = pd.DataFrame({"record_id": ["a"], "given_name": ["BILL"]}, dtype="string")
    out, _ops = ch.apply(df, np.random.default_rng(0), pd.Series(1.0, index=df.index))
    assert out["given_name"].tolist() == ["WILLIAM"]


def test_name_order_swap_logs_both_cells():
    df = make_frame(20)
    out, ops = C.name_order_swap().apply(
        df, np.random.default_rng(2), pd.Series(1.0, index=df.index)
    )
    assert len(ops) == 40  # two rows per record
    for i in range(len(df)):
        assert out["given_name"].iloc[i] == df["family_name"].iloc[i]
        assert out["family_name"].iloc[i] == df["given_name"].iloc[i]


def test_field_dropout_sets_na():
    df = make_frame(50)
    out, ops = C.field_dropout().apply(df, np.random.default_rng(9), pd.Series(1.0, index=df.index))
    assert len(ops) == 50
    assert ops["after"].isna().all()
    for r in ops.itertuples(index=False):
        i = df.index[df["record_id"] == r.record_id][0]
        assert pd.isna(out[str(r.field)].iloc[i])


def test_format_drift_dob_preserves_date():
    n = 30
    df = pd.DataFrame(
        {
            "record_id": [f"r{i}" for i in range(n)],
            "dob": ["1985-03-04"] * n,
            "street": ["ELM"] * n,  # no abbreviable word
            "phone": [pd.NA] * n,
        },
        dtype="string",
    )
    _out, ops = C.field_format_drift().apply(
        df, np.random.default_rng(1), pd.Series(1.0, index=df.index)
    )
    assert len(ops) == n
    assert (ops["field"] == "dob").all()
    for r in ops.itertuples(index=False):
        dt, fmt = C.parse_dob(str(r.after))
        assert dt is not None and fmt != "%Y-%m-%d"
        assert (dt.year, dt.month, dt.day) == (1985, 3, 4)


def test_format_drift_phone_preserves_digits():
    n = 20
    df = pd.DataFrame(
        {
            "record_id": [f"r{i}" for i in range(n)],
            "dob": [pd.NA] * n,
            "street": ["ELM"] * n,
            "phone": ["9195551234"] * n,
        },
        dtype="string",
    )
    _, ops = C.field_format_drift().apply(
        df, np.random.default_rng(1), pd.Series(1.0, index=df.index)
    )
    assert len(ops) == n
    for r in ops.itertuples(index=False):
        assert "".join(c for c in str(r.after) if c.isdigit()) == "9195551234"
        assert str(r.after) != "9195551234"


def test_format_drift_address_abbreviation():
    n = 20
    df = pd.DataFrame(
        {
            "record_id": [f"r{i}" for i in range(n)],
            "dob": [pd.NA] * n,
            "street": ["MAIN STREET"] * n,
            "phone": [pd.NA] * n,
        },
        dtype="string",
    )
    _out, ops = C.field_format_drift().apply(
        df, np.random.default_rng(1), pd.Series(1.0, index=df.index)
    )
    assert len(ops) == n
    assert set(ops["after"]) == {"MAIN ST"}


def test_suffix_confusion_add_drop_swap():
    df = make_frame(50)
    _out, ops = C.suffix_confusion().apply(
        df, np.random.default_rng(10), pd.Series(1.0, index=df.index)
    )
    assert len(ops) == 50
    assert (ops["field"] == "name_suffix").all()
    had = {str(r): v for r, v in zip(df["record_id"], df["name_suffix"])}
    for r in ops.itertuples(index=False):
        if pd.isna(had[str(r.record_id)]):  # added
            assert str(r.after) in {"JR", "SR", "II"}
        else:  # dropped or confused, never unchanged
            assert pd.isna(r.after) or str(r.after) in {"SR", "II"}


def test_hub_value_injects_placeholders():
    df = make_frame(60)
    df["dob"] = pd.Series(["03/04/1985"] * 60, dtype="string")  # %m/%d/%Y per-dataset style
    _out, ops = C.hub_value().apply(df, np.random.default_rng(12), pd.Series(1.0, index=df.index))
    assert len(ops) == 60
    for r in ops.itertuples(index=False):
        fld = str(r.field)
        if fld == "dob":
            assert str(r.after) == "01/01/1900"  # matches the record's own format
        elif fld == "phone":
            assert str(r.after) == "0000000"
        elif fld == "street":
            assert str(r.after) == "GENERAL DELIVERY"
        else:
            pytest.fail(f"unexpected hub field {fld}")
    assert {"dob", "phone", "street"} == set(ops["field"].unique())


def test_raw_unparse_composes_and_blanks():
    df = pd.DataFrame(
        {
            "record_id": ["r0"],
            "given_name": ["MARIA"],
            "middle_name": ["JOSEFA"],
            "family_name": ["GARCIA LOPEZ"],
            "name_suffix": [pd.NA],
            "house_number": ["1204"],
            "street": ["MAIN STREET"],
            "unit": [pd.NA],
        },
        dtype="string",
    )
    out, ops = C.raw_unparse().apply(df, np.random.default_rng(0), pd.Series(1.0, index=df.index))
    assert out["full_name"].iloc[0] == "GARCIA LOPEZ, MARIA J"
    assert out["street_address"].iloc[0] == "1204 MAIN STREET"
    for part in ("given_name", "middle_name", "family_name", "house_number", "street"):
        assert pd.isna(out[part].iloc[0])
    logged = {(str(r.field)): (r.before, r.after) for r in ops.itertuples(index=False)}
    assert logged["full_name"][1] == "GARCIA LOPEZ, MARIA J"
    assert logged["given_name"][0] == "MARIA" and pd.isna(logged["given_name"][1])
    assert logged["street_address"][1] == "1204 MAIN STREET"
    assert logged["street"][0] == "MAIN STREET" and pd.isna(logged["street"][1])
    assert "unit" not in logged and "name_suffix" not in logged  # were already missing


def test_scalar_rate_accepted():
    df = make_frame(20)
    _out, ops = C.field_dropout().apply(df, np.random.default_rng(1), 1.0)
    assert len(ops) == 20


def _write_lexicon(tmp_path, text: str) -> None:
    lex_dir = tmp_path / "lexicons"
    lex_dir.mkdir()
    (lex_dir / "names.csv").write_text(text)


def test_load_lexicon_legacy_format(tmp_path):
    _write_lexicon(tmp_path, "william,bill,will\nrobert,bob\nsolo\n")
    lex = C.load_lexicon(tmp_path)
    assert lex == {"william": {"bill", "will"}, "robert": {"bob"}}


def test_load_lexicon_legacy_single_row_not_misdetected_as_triple(tmp_path):
    # one legacy row with exactly two nicknames must NOT be read as a triple
    _write_lexicon(tmp_path, "william,bill,will\n")
    assert C.load_lexicon(tmp_path) == {"william": {"bill", "will"}}


@pytest.mark.parametrize("header", ["name1,relationship,name2", "name1,has_nickname,name2"])
def test_load_lexicon_triple_format_with_header(tmp_path, header):
    _write_lexicon(
        tmp_path,
        f"{header}\n"
        "william,has_nickname,bill\n"
        "william,has_nickname,will\n"
        "robert,has_nickname,bob\n",
    )
    lex = C.load_lexicon(tmp_path)
    assert lex == {"william": {"bill", "will"}, "robert": {"bob"}}
    assert "name1" not in lex  # header row skipped
    assert all("has_nickname" not in v for v in lex.values())  # no spurious variants


def test_load_lexicon_triple_format_headerless(tmp_path):
    _write_lexicon(tmp_path, "william,has_nickname,bill\nrobert,has_nickname,bob\n")
    assert C.load_lexicon(tmp_path) == {"william": {"bill"}, "robert": {"bob"}}


def test_load_lexicon_triple_format_drops_self_links(tmp_path):
    _write_lexicon(
        tmp_path,
        "name1,relationship,name2\nwilliam,has_nickname,william\nrobert,has_nickname,bob\n",
    )
    assert C.load_lexicon(tmp_path) == {"robert": {"bob"}}


def test_load_lexicon_missing_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        C.load_lexicon(tmp_path)


@pytest.mark.network
def test_fetch_lexicon_real(tmp_path):
    path = C.fetch_lexicon(tmp_path)
    assert path.exists() and path == tmp_path / "lexicons" / "names.csv"
    lex = C.load_lexicon(tmp_path)
    assert len(lex) > 500
    assert "bill" in lex["william"]
