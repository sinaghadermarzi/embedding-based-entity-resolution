"""Tests for er_lab.score.fs — the tuned Splink 4 Fellegi–Sunter baseline adapter."""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import pytest

from er_lab.data.schema import DeclaredSchema
from er_lab.score.fs import fit_fs, fs_scores

logging.getLogger("splink").setLevel(logging.ERROR)

GIVEN = [
    "WILLIAM", "ROBERT", "ELIZABETH", "MARGARET", "JAMES", "KATHERINE", "MICHAEL",
    "PATRICIA", "DAVID", "BARBARA", "RICHARD", "SUSAN", "JOSEPH", "JESSICA", "THOMAS",
    "SARAH", "CHARLES", "KAREN", "CHRISTOPHER", "LISA",
]
FAMILY = [
    "SMITH", "GARCIA", "JOHNSON", "MCDONALD", "OBRIEN", "LEE", "MARTINEZ", "BROWN",
    "DAVIS", "WILSON", "ANDERSON", "THOMAS", "TAYLOR", "MOORE", "JACKSON", "MARTIN",
    "THOMPSON", "WHITE", "LOPEZ", "CLARK",
]

N_ENTITIES = 170  # every even entity gets a corrupted duplicate -> 255 records


def make_fs_frame() -> pd.DataFrame:
    """~300 records with planted duplicates (patterned after tests/test_generate.py,
    but self-contained — the noise modules are deliberately not imported here)."""
    rows = []
    for e in range(N_ENTITIES):
        g, f = GIVEN[e % 20], FAMILY[(e * 7) % 20]
        dob = f"19{60 + e % 15:02d}-{1 + e % 12:02d}-{1 + e % 25:02d}"
        zipc = f"275{e % 20:02d}"
        rows.append(
            {"record_id": f"r{e:04d}", "entity_id": f"e{e:03d}",
             "given_name": g, "family_name": f, "dob": dob, "zip": zipc}
        )
        if e % 2 == 0:  # planted duplicate with mild, varied corruption
            g2 = g[:2] + g[3:] if len(g) > 3 else g  # char deletion typo
            dob2 = dob[:-1] + ("2" if dob[-1] != "2" else "3") if e % 4 == 0 else dob
            zip2 = zipc if e % 6 else "27599"
            rows.append(
                {"record_id": f"r{e:04d}d", "entity_id": f"e{e:03d}",
                 "given_name": g2, "family_name": f, "dob": dob2, "zip": zip2}
            )
    return pd.DataFrame(rows).astype("string")


def make_pairs(df: pd.DataFrame, n_random: int = 85, seed: int = 5):
    """All planted duplicate pairs + n_random non-duplicate pairs, with truth dict."""
    truth: dict[tuple[str, str], int] = {}
    for e in range(0, N_ENTITIES, 2):
        truth[(f"r{e:04d}", f"r{e:04d}d")] = 1
    rng = np.random.default_rng(seed)
    ids = df["record_id"].tolist()
    entity = dict(zip(df["record_id"], df["entity_id"]))
    while sum(v == 0 for v in truth.values()) < n_random:
        a, b = (str(x) for x in rng.choice(ids, 2, replace=False))
        a, b = min(a, b), max(a, b)
        if entity[a] != entity[b] and (a, b) not in truth:
            truth[(a, b)] = 0
    pairs = pd.DataFrame(list(truth), columns=["a", "b"])
    return pairs, truth


SCHEMA = DeclaredSchema(
    name="fs_synth",
    record_id="record_id",
    entity_id="entity_id",
    roles={r: r for r in ("given_name", "family_name", "dob", "zip")},
)


@pytest.fixture(scope="module")
def linker():
    return fit_fs(make_fs_frame(), schema=SCHEMA)


def test_em_converges(linker):
    sessions = linker._er_lab_training["em_sessions"]
    assert len(sessions) == 2  # names trained blocked-on-dob, dob/zip blocked-on-names
    for s in sessions:
        assert s["iterations"] >= 1
        assert s["iterations"] < s["max_iterations"]
        assert s["final_change"] < s["em_convergence"]
        assert s["converged"] is True


def test_planted_dups_rank_above_random_nondups(linker):
    df = make_fs_frame()
    pairs, truth = make_pairs(df)
    scored = fs_scores(linker, pairs=pairs)
    scored["is_dup"] = [truth[(a, b)] for a, b in zip(scored["a"], scored["b"])]
    dup = scored.loc[scored["is_dup"] == 1, "prob"].to_numpy()
    non = scored.loc[scored["is_dup"] == 0, "prob"].to_numpy()
    assert len(dup) == 85 and len(non) == 85
    # rank-order assertion (not absolute probabilities): pairwise AUC
    auc = (dup[:, None] > non[None, :]).mean() + 0.5 * (dup[:, None] == non[None, :]).mean()
    assert auc > 0.95
    assert np.quantile(dup, 0.10) > np.quantile(non, 0.90)


def test_pairs_mode_scores_exactly_the_requested_pairs(linker):
    df = make_fs_frame()
    pairs, _ = make_pairs(df, n_random=20, seed=11)
    # hand pairs over in reversed orientation + a duplicate row: fs_scores must
    # canonicalize, dedupe, and return exactly the requested set — nothing else
    scrambled = pd.concat(
        [pairs.rename(columns={"a": "b", "b": "a"}), pairs.iloc[[0]]], ignore_index=True
    )
    scored = fs_scores(linker, pairs=scrambled)
    assert list(scored.columns) == ["a", "b", "prob"]
    assert (scored["a"] < scored["b"]).all()
    assert set(zip(scored["a"], scored["b"])) == set(zip(pairs["a"], pairs["b"]))
    assert len(scored) == len(pairs)
    assert scored["prob"].between(0.0, 1.0).all()


def test_pairs_mode_determinism(linker):
    df = make_fs_frame()
    pairs, _ = make_pairs(df, n_random=10, seed=13)
    pd.testing.assert_frame_equal(
        fs_scores(linker, pairs=pairs), fs_scores(linker, pairs=pairs)
    )


def test_pairs_mode_rejects_unknown_ids_and_self_pairs(linker):
    with pytest.raises(ValueError, match="absent"):
        fs_scores(linker, pairs=pd.DataFrame({"a": ["r0000"], "b": ["ghost"]}))
    with pytest.raises(ValueError, match="self-pairs"):
        fs_scores(linker, pairs=pd.DataFrame({"a": ["r0000"], "b": ["r0000"]}))
    with pytest.raises(KeyError, match="columns 'a' and 'b'"):
        fs_scores(linker, pairs=pd.DataFrame({"x": ["r0000"], "y": ["r0002"]}))


def test_blocked_predict_mode(linker):
    scored = fs_scores(linker)
    assert list(scored.columns) == ["a", "b", "prob"]
    assert len(scored) > 0
    assert (scored["a"] < scored["b"]).all()
    assert scored["prob"].between(0.0, 1.0).all()
    assert not scored.duplicated(["a", "b"]).any()
    # a mildly-corrupted planted duplicate (given-name typo only; e=2 keeps
    # dob and zip intact) shares family_name -> blocked in and scored high
    row = scored[(scored["a"] == "r0002") & (scored["b"] == "r0002d")]
    assert len(row) == 1
    assert float(row["prob"].iloc[0]) > 0.5


def test_fit_rejects_frames_without_usable_roles():
    df = pd.DataFrame(
        {"record_id": ["x1", "x2"], "given_name": [pd.NA, pd.NA], "family_name": ["A", "A"]}
    ).astype("string")
    schema = DeclaredSchema(
        name="degenerate", record_id="record_id", entity_id=None,
        roles={"given_name": "given_name", "family_name": "family_name"},
    )
    with pytest.raises(ValueError):
        fit_fs(df, schema=schema)
