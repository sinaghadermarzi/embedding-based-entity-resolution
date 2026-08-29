import zlib

import numpy as np
import pandas as pd
import pytest

from er_lab.probes.battery import (
    MUST_NOT_HOLD,
    PROBE_COLUMNS,
    SHOULD_HOLD,
    battery_report,
    build_probe_set,
)

REPORT_COLUMNS = ["kind", "n", "mean_cos", "p05_cos", "p95_cos", "auc_vs_true"]


def _ngram_encode(texts: list[str]) -> np.ndarray:
    """FAKE deterministic encoder: hashed char-3-gram counts (no learning, no seed)."""
    dim = 512
    out = np.zeros((len(texts), dim), dtype=float)
    for i, text in enumerate(texts):
        padded = f"^^{text.lower()}$$"
        for j in range(len(padded) - 2):
            out[i, zlib.crc32(padded[j : j + 3].encode("utf-8")) % dim] += 1.0
    return out


def _base_texts(n: int = 24) -> list[str]:
    rng = np.random.default_rng(11)
    givens = ["maria", "james", "elena", "robert", "keisha", "daniel"]
    familys = ["hernandez", "osei", "nguyen", "kowalski", "smith", "abara"]
    texts = []
    for i in range(n):
        g = givens[int(rng.integers(len(givens)))]
        f = familys[int(rng.integers(len(familys)))]
        year = 1950 + int(rng.integers(50))
        texts.append(f"given={g} | family={f} | dob={year}-03-1{i % 10} | city=durham")
    return texts


def _hand_built_probes():
    """should_hold: one-char typo pairs; must_not_hold: scrambled-text control."""
    rng = np.random.default_rng(7)
    rows = []
    true_rows = []
    for text in _base_texts():
        # one-character typo (deterministic position/replacement)
        pos = 8
        typo = text[:pos] + "x" + text[pos + 1 :]
        rows.append(("typo_one_char", SHOULD_HOLD, text, typo))
        true_rows.append((text, typo))
        # scrambled control: same characters, order destroyed
        scrambled = "".join(rng.permutation(list(text)))
        rows.append(("scrambled_control", MUST_NOT_HOLD, text, scrambled))
    probes = pd.DataFrame(rows, columns=PROBE_COLUMNS)
    true_pairs = pd.DataFrame(true_rows, columns=["text_a", "text_b"])
    return probes, true_pairs


def test_report_shape_and_columns():
    probes, true_pairs = _hand_built_probes()
    report = battery_report(_ngram_encode, probes, true_pairs=true_pairs)
    assert list(report.columns) == REPORT_COLUMNS
    assert set(report.index) == {"typo_one_char", "scrambled_control"}
    assert (report["n"] == 24).all()
    assert report.loc["typo_one_char", "kind"] == SHOULD_HOLD
    assert report.loc["scrambled_control", "kind"] == MUST_NOT_HOLD
    for col in ("mean_cos", "p05_cos", "p95_cos"):
        assert report[col].between(-1.0 - 1e-9, 1.0 + 1e-9).all()
    assert (report["p05_cos"] <= report["mean_cos"] + 1e-12).all()
    assert (report["mean_cos"] <= report["p95_cos"] + 1e-12).all()


def test_should_hold_scores_above_scrambled_control():
    probes, _ = _hand_built_probes()
    report = battery_report(_ngram_encode, probes)
    assert (
        report.loc["typo_one_char", "mean_cos"]
        > report.loc["scrambled_control", "mean_cos"] + 0.2
    )


def test_auc_vs_true_bounds_and_direction():
    probes, true_pairs = _hand_built_probes()
    report = battery_report(_ngram_encode, probes, true_pairs=true_pairs)
    auc = report.loc["scrambled_control", "auc_vs_true"]
    assert 0.0 <= auc <= 1.0
    # constructed separable case: true pairs (one-char typos, high cosine) vs
    # scrambled confusables (low cosine) -> near-perfect separation
    assert auc > 0.9
    # should_hold slices never get an AUC
    assert np.isnan(report.loc["typo_one_char", "auc_vs_true"])


def test_auc_nan_without_true_pairs():
    probes, _ = _hand_built_probes()
    report = battery_report(_ngram_encode, probes)
    assert report["auc_vs_true"].isna().all()


def test_report_validation_errors():
    probes, _ = _hand_built_probes()
    with pytest.raises(KeyError, match="missing columns"):
        battery_report(_ngram_encode, probes.drop(columns=["kind"]))
    mixed = probes.copy()
    mixed.loc[mixed.index[0], "slice"] = "scrambled_control"  # now mixes kinds
    with pytest.raises(ValueError, match="mixes kinds"):
        battery_report(_ngram_encode, mixed)
    bad_kind = probes.copy()
    bad_kind["kind"] = "maybe"
    with pytest.raises(ValueError, match="unknown kind"):
        battery_report(_ngram_encode, bad_kind)
    with pytest.raises(ValueError, match="one embedding row per text"):
        battery_report(lambda ts: np.zeros((1, 4)), probes)


# ---------------------------------------------------------------------------
# build_probe_set — active once the concurrent er_lab.noise.channels build lands
# ---------------------------------------------------------------------------


def _canonical_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "record_id": [f"r{i}" for i in range(8)],
            "entity_id": [f"e{i}" for i in range(8)],
            "given_name": [
                "William", "Robert", "Elizabeth", "James",
                "Margaret", "John", "Katherine", "Michael",
            ],
            "family_name": [
                "Harris", "Okafor", "Nguyen", "Sanchez",
                "Kowalski", "Smith", "Abara", "Lindqvist",
            ],
            "middle_name": ["A", "B", pd.NA, "D", "E", pd.NA, "G", "H"],
            "name_suffix": [pd.NA] * 8,
            "dob": [
                "1984-03-12", "1990-07-01", "1975-11-30", "1988-02-29",
                "1969-05-17", "2001-12-25", "1955-08-08", "1979-01-19",
            ],
            "street": [
                "12 Oak Street", "480 Maple Avenue", "7 Pine Road", "33 Elm Drive",
                "901 Cedar Lane", "5 Birch Court", "62 Walnut Place", "18 Ash Boulevard",
            ],
            "city": [
                "Durham", "Raleigh", "Cary", "Wilson",
                "Apex", "Boone", "Monroe", "Shelby",
            ],
            "zip": ["27701", "27601", "27511", "27893", "27502", "28607", "28110", "28150"],
        }
    ).astype("string")


def _serialize(row_dict: dict) -> str:
    parts = []
    for key, value in row_dict.items():
        if value is None or pd.isna(value):
            continue
        parts.append(f"{key}={value}")
    return " | ".join(parts)  # order-sensitive on purpose


def test_build_probe_set_with_channels():
    pytest.importorskip("er_lab.noise.channels")
    df = _canonical_frame()
    probes = build_probe_set(df, serialize=_serialize, seed=7, n_per_slice=8)
    assert list(probes.columns) == PROBE_COLUMNS
    assert len(probes) > 0
    assert (probes["text_a"] != probes["text_b"]).all()

    kind_by_slice = probes.groupby("slice")["kind"].unique()
    assert all(len(k) == 1 for k in kind_by_slice)
    slices = set(probes["slice"])
    should = {
        "typo@0.02", "typo@0.05", "typo@0.1", "typo@0.2",
        "nickname_swap", "field_order_permutation", "format_drift",
    }
    must_not = {"different_birth_year", "suffix_jr_sr", "twin_household"}
    assert should <= slices  # all applicable on this frame
    assert must_not <= slices
    for s in should:
        assert kind_by_slice[s][0] == SHOULD_HOLD
    for s in must_not:
        assert kind_by_slice[s][0] == MUST_NOT_HOLD

    # heavier dial -> at least as many accumulated edits; texts still change
    assert (probes[probes["slice"] == "typo@0.2"]["text_a"] != "").all()


def test_build_probe_set_deterministic_under_seed():
    pytest.importorskip("er_lab.noise.channels")
    df = _canonical_frame()
    p1 = build_probe_set(df, serialize=_serialize, seed=123, n_per_slice=6)
    p2 = build_probe_set(df, serialize=_serialize, seed=123, n_per_slice=6)
    pd.testing.assert_frame_equal(p1, p2)


def test_build_probe_set_feeds_battery_report():
    pytest.importorskip("er_lab.noise.channels")
    df = _canonical_frame()
    probes = build_probe_set(df, serialize=_serialize, seed=5, n_per_slice=8)
    true_pairs = probes.loc[probes["slice"] == "typo@0.02", ["text_a", "text_b"]]
    report = battery_report(_ngram_encode, probes, true_pairs=true_pairs)
    assert list(report.columns) == REPORT_COLUMNS
    aucs = report.loc[report["kind"] == MUST_NOT_HOLD, "auc_vs_true"].dropna()
    assert ((aucs >= 0.0) & (aucs <= 1.0)).all()


def test_build_probe_set_validation():
    pytest.importorskip("er_lab.noise.channels")
    df = _canonical_frame()
    with pytest.raises(KeyError, match="record_id"):
        build_probe_set(df.drop(columns=["record_id"]), serialize=_serialize, seed=0)
    with pytest.raises(ValueError, match="n_per_slice"):
        build_probe_set(df, serialize=_serialize, seed=0, n_per_slice=0)
