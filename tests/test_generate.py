"""Tests for er_lab.noise.generate (corpus generation + household confusables)."""

from __future__ import annotations

import math

import pandas as pd
import pytest

from er_lab.noise.exposure import ExposureModel
from er_lab.noise.generate import generate_corpus, household_confusables


def make_base(n: int = 300) -> pd.DataFrame:
    # compact fixture data
    # fmt: off
    given = ["WILLIAM", "ROBERT", "ELIZABETH", "MARGARET", "JAMES", "KATHERINE", "MICHAEL", "PATRICIA"]
    fam = ["SMITH", "GARCIA LOPEZ", "JOHNSON", "MCDONALD", "OBRIEN", "LEE", "MARTINEZ", "BROWN"]
    # fmt: on
    rows = {
        "record_id": [f"r{i:05d}" for i in range(n)],
        "entity_id": [pd.NA] * n,
        "given_name": [given[i % 8] for i in range(n)],
        "middle_name": [("JOSEFA" if i % 3 == 0 else pd.NA) for i in range(n)],
        "family_name": [fam[i % 8] for i in range(n)],
        "name_suffix": [("JR" if i % 7 == 0 else pd.NA) for i in range(n)],
        "dob": [f"19{60 + i % 40:02d}-{1 + i % 12:02d}-{1 + i % 28:02d}" for i in range(n)],
        "sex": ["F" if i % 2 else "M" for i in range(n)],
        "race": ["W" if i % 2 else "B" for i in range(n)],
        "house_number": [str(100 + i) for i in range(n)],
        "street": ["MAIN STREET" if i % 2 else "OAK AVENUE" for i in range(n)],
        "city": ["DURHAM"] * n,
        "state": ["NC"] * n,
        "zip": [f"275{i % 100:02d}" for i in range(n)],
        "phone": [f"919555{i % 10000:04d}" for i in range(n)],
        "email": [f"p{i}@example.com" for i in range(n)],
        "source": ["test"] * n,
    }
    return pd.DataFrame(rows, dtype="string")


RATES = {"typo": 0.5, "nickname": 0.4, "name_order_swap": 0.3, "field_dropout": 0.4}


def eq(a, b) -> bool:
    a_na, b_na = bool(pd.isna(a)), bool(pd.isna(b))
    if a_na or b_na:
        return a_na and b_na
    return a == b


def test_seed_determinism():
    base = make_base(150)
    c1, o1 = generate_corpus(base, channel_rates=RATES, seed=7)
    c2, o2 = generate_corpus(base, channel_rates=RATES, seed=7)
    pd.testing.assert_frame_equal(c1, c2)
    pd.testing.assert_frame_equal(o1, o2)
    c3, _ = generate_corpus(base, channel_rates=RATES, seed=8)
    assert not c3.equals(c1)


def test_entity_id_integrity():
    base = make_base(300)
    corpus, _ = generate_corpus(base, channel_rates=RATES, seed=11)
    base_entity = dict(zip(base["record_id"], base["record_id"]))  # entity_id NA -> own rid
    for rid, ent in zip(corpus["record_id"], corpus["entity_id"]):
        base_rid = str(rid).split("#dup")[0]
        assert str(ent) == base_entity[base_rid], (rid, ent)


def test_entity_id_preserved_when_present():
    base = make_base(50)
    base["entity_id"] = pd.Series([f"e{i // 2}" for i in range(50)], dtype="string")
    corpus, _ = generate_corpus(base, channel_rates={}, seed=3)
    ent = dict(zip(base["record_id"], base["entity_id"]))
    for rid, e in zip(corpus["record_id"], corpus["entity_id"]):
        assert str(e) == str(ent[str(rid).split("#dup")[0]])


def test_zipf_cluster_shape():
    base = make_base(2000)
    corpus, ops = generate_corpus(base, channel_rates={}, dup_params={"a": 2.5, "max": 20}, seed=5)
    assert len(ops) == 0
    sizes = corpus.groupby("entity_id").size()
    assert len(sizes) == 2000  # every base entity present (keep_original_rate=1.0)
    assert sizes.max() <= 20  # max respected
    assert (sizes == 1).mean() > 0.6  # most entities singletons (P[s=1] ~= 0.75)
    assert (sizes > 1).any()  # but duplicates do exist
    # duplicate ids are well-formed
    dups = corpus[corpus["record_id"].str.contains("#dup")]
    assert len(dups) > 0
    assert dups["record_id"].str.match(r"r\d{5}#dup\d+").all()
    assert dups["record_id"].is_unique


def test_keep_original_rate_zero_drops_originals():
    base = make_base(200)
    corpus, _ = generate_corpus(base, channel_rates={}, keep_original_rate=0.0, seed=2)
    assert corpus["record_id"].str.contains("#dup").all()


def test_ops_log_replays_base_to_corpus():
    """The ops log is exact ground truth: replaying it turns base cells into corpus cells."""
    base = make_base(200)
    corpus, ops = generate_corpus(base, channel_rates=RATES, seed=13)
    assert list(ops.columns) == ["record_id", "channel", "field", "before", "after"]
    assert set(ops["channel"].unique()) <= set(RATES)

    chains: dict[tuple[str, str], list[tuple]] = {}
    for r in ops.itertuples(index=False):  # frame order == application order
        chains.setdefault((str(r.record_id), str(r.field)), []).append((r.before, r.after))

    base_rows = {str(r): base.iloc[i] for i, r in enumerate(base["record_id"])}
    skip = {"record_id", "entity_id"}
    for i in range(len(corpus)):
        rid = str(corpus["record_id"].iloc[i])
        base_row = base_rows[rid.split("#dup")[0]]
        for col in corpus.columns:
            if col in skip:
                continue
            expected = base_row[col] if col in base_row.index else pd.NA
            for before, after in chains.get((rid, col), []):
                assert eq(expected, before), (rid, col, expected, before)
                expected = after
            assert eq(corpus[col].iloc[i], expected), (rid, col)
    # unchanged cells absent: chains only exist for cells that actually differ or chain back
    for (rid, col), chain in chains.items():
        assert not eq(chain[0][0], chain[-1][1]) or len(chain) > 1


def test_exposure_model_drives_rates():
    class GroupExposure:
        """race B -> full rate, race W -> zero (duck-typed ExposureModel)."""

        def per_record_rates(self, df: pd.DataFrame, base_rate: float) -> pd.Series:
            mult = (df["race"] == "B").astype("float64")
            return pd.Series(base_rate, index=df.index) * mult

    base = make_base(300)
    corpus, ops = generate_corpus(
        base, channel_rates={"field_dropout": 1.0}, exposure=GroupExposure(), seed=17
    )
    assert len(ops) > 0
    race = dict(zip(base["record_id"], base["race"]))
    dup_rids = set(corpus.loc[corpus["record_id"].str.contains("#dup"), "record_id"])
    mutated = set(ops["record_id"])
    for rid in dup_rids:
        if race[str(rid).split("#dup")[0]] == "B":
            assert rid in mutated  # dropout at rate 1.0 always applies
        else:
            assert rid not in mutated


def test_exposure_rates_computed_from_pristine_dup_frame():
    """Exposure is generative: a dup whose race cell was dropped by an earlier
    channel must still get the exposure-multiplied rate for later channels."""
    base = make_base(1500)
    base["race"] = pd.Series(["B"] * len(base), dtype="string")
    exposure = ExposureModel.from_table(pd.DataFrame({"race": ["B"], "multiplier": [8.0]}))
    _corpus, ops = generate_corpus(
        base,
        channel_rates={"field_dropout": 0.9, "typo": 0.1},
        exposure=exposure,
        seed=23,
    )
    dropped = set(
        ops.loc[(ops["channel"] == "field_dropout") & (ops["field"] == "race"), "record_id"]
    )
    assert len(dropped) > 100  # race-dropped dups form a real subpopulation
    typo_rids = set(ops.loc[ops["channel"] == "typo", "record_id"])
    nominal = min(0.1 * 8.0, 1.0)  # multiplied typo rate, regardless of the dropout
    observed = len(dropped & typo_rids) / len(dropped)
    tol = 4 * math.sqrt(nominal * (1 - nominal) / len(dropped)) + 0.02  # + no-op slack
    assert abs(observed - nominal) < tol, (observed, nominal, tol)


def test_unknown_channel_raises():
    with pytest.raises(KeyError):
        generate_corpus(make_base(10), channel_rates={"nope": 0.1}, seed=1)


def test_unsupported_dup_dist_raises():
    with pytest.raises(ValueError):
        generate_corpus(make_base(10), channel_rates={}, dup_dist="uniform", seed=1)


def test_household_confusables_distinct_entities_shared_household():
    base = make_base(200)
    hh, ops = household_confusables(base, rate=0.5, seed=3)
    assert len(hh) > 20
    assert hh["record_id"].str.contains("#hh").all()
    assert hh["entity_id"].is_unique
    assert (hh["entity_id"] == hh["record_id"]).all()  # fresh singleton entities
    assert set(hh["entity_id"]).isdisjoint(set(base["record_id"]))

    base_rows = {str(r): base.iloc[i] for i, r in enumerate(base["record_id"])}
    kinds = {
        str(r.record_id): str(r.after)
        for r in ops.itertuples(index=False)
        if str(r.field) == "__kind__"
    }
    assert set(kinds) == set(hh["record_id"])
    assert set(kinds.values()) <= {"sibling", "jr_sr", "twin"}
    assert (ops["channel"] == "household").all()

    for i in range(len(hh)):
        rid = str(hh["record_id"].iloc[i])
        b = base_rows[rid.split("#hh")[0]]
        # the co-resident trap: same family name, same address
        assert hh["family_name"].iloc[i] == b["family_name"]
        assert hh["street"].iloc[i] == b["street"]
        assert hh["house_number"].iloc[i] == b["house_number"]
        kind = kinds[rid]
        if kind in ("sibling", "twin"):
            assert hh["given_name"].iloc[i] != b["given_name"]
        if kind == "twin":
            assert hh["dob"].iloc[i] == b["dob"]  # shared birth date
        if kind == "jr_sr":
            assert hh["given_name"].iloc[i] == b["given_name"]
            assert not eq(hh["name_suffix"].iloc[i], b["name_suffix"])


def test_household_ops_match_field_changes():
    base = make_base(120)
    hh, ops = household_confusables(base, rate=0.6, seed=9)
    base_rows = {str(r): base.iloc[i] for i, r in enumerate(base["record_id"])}
    field_ops = ops[ops["field"] != "__kind__"]
    logged = {
        (str(r.record_id), str(r.field)): (r.before, r.after)
        for r in field_ops.itertuples(index=False)
    }
    for i in range(len(hh)):
        rid = str(hh["record_id"].iloc[i])
        b = base_rows[rid.split("#hh")[0]]
        for col in base.columns:
            if col in ("record_id", "entity_id"):
                continue
            before, after = b[col], hh[col].iloc[i]
            if eq(before, after):
                assert (rid, col) not in logged
            else:
                assert logged[(rid, col)] == (before, after) or (
                    eq(logged[(rid, col)][0], before) and eq(logged[(rid, col)][1], after)
                )


def test_household_schema_identical_between_empty_and_populated():
    base = make_base(40).drop(columns=["entity_id"])
    populated, _ = household_confusables(base, rate=1.0, seed=1)
    empty, _ = household_confusables(base, rate=0.0, seed=1)
    assert len(populated) > 0 and len(empty) == 0
    assert "entity_id" in populated.columns
    assert list(empty.columns) == list(populated.columns)
    assert (empty.dtypes == populated.dtypes).all()


def test_household_unshiftable_sibling_logged_as_twin():
    """A drawn sibling whose dob cannot be shifted is structurally a twin and
    must be labeled as one in the __kind__ ops row."""
    base = make_base(80)
    base["dob"] = pd.Series(["UNKNOWN"] * len(base), dtype="string")
    hh, ops = household_confusables(base, rate=1.0, seed=5)
    kinds = {
        str(r.record_id): str(r.after)
        for r in ops.itertuples(index=False)
        if str(r.field) == "__kind__"
    }
    assert len(hh) > 0
    assert "twin" in kinds.values()
    assert "sibling" not in kinds.values()  # every drawn sibling was relabeled
    # relabeled twins really are twins: identical birth field, different given name
    base_rows = {str(r): base.iloc[i] for i, r in enumerate(base["record_id"])}
    for i in range(len(hh)):
        rid = str(hh["record_id"].iloc[i])
        if kinds[rid] == "twin":
            b = base_rows[rid.split("#hh")[0]]
            assert hh["dob"].iloc[i] == b["dob"] == "UNKNOWN"
            assert hh["given_name"].iloc[i] != b["given_name"]


def test_household_determinism():
    base = make_base(150)
    h1, o1 = household_confusables(base, rate=0.4, seed=21)
    h2, o2 = household_confusables(base, rate=0.4, seed=21)
    pd.testing.assert_frame_equal(h1, h2)
    pd.testing.assert_frame_equal(o1, o2)
