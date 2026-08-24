import math

import numpy as np
import pandas as pd
import pytest

from er_lab.eval.truth_model import corrected_precision_recall, disagreement_sample

# ---------------------------------------------------------------------------
# corrected_precision_recall
# ---------------------------------------------------------------------------


def test_zero_rates_are_identity():
    out = corrected_precision_recall(
        0.93, 0.81, truth_dup_rate=0.0, truth_overlay_rate=0.0
    )
    assert out["precision"] == pytest.approx(0.93)
    assert out["recall"] == pytest.approx(0.81)
    # with exact rates and no CI, the bands collapse onto the points
    assert out["precision_low"] == pytest.approx(0.93)
    assert out["precision_high"] == pytest.approx(0.93)
    assert out["recall_low"] == pytest.approx(0.81)
    assert out["recall_high"] == pytest.approx(0.81)


def test_two_percent_duplication_known_answer():
    # Constructed case: the truth key is duplicated at rate d = 0.02 (2% of truly
    # co-referent pairs are keyed apart), no overlays (o = 0). Observed precision
    # is 0.98. The correction algebra gives
    #   P = (P_obs - o) / (1 - d - o) = (0.98 - 0) / (1 - 0.02) = 0.98 / 0.98 = 1.0
    # i.e. the 2% of predicted links scored as FPs are exactly the true links that
    # duplicated keys made LOOK like FPs — the system was actually perfect.
    out = corrected_precision_recall(
        0.98, 0.90, truth_dup_rate=0.02, truth_overlay_rate=0.0
    )
    assert out["precision"] == pytest.approx(1.0)
    # recall's point estimate is untouched by dup alone (o = 0): R = 0.90 / 1 = 0.90
    assert out["recall"] == pytest.approx(0.90)


def test_general_known_answers():
    # P = (0.90 - 0) / (1 - 0.02) = 0.90 / 0.98
    out = corrected_precision_recall(
        0.90, 0.80, truth_dup_rate=0.02, truth_overlay_rate=0.0
    )
    assert out["precision"] == pytest.approx(0.90 / 0.98)

    # overlays inflate observed precision: P = (0.98 - 0.03) / (1 - 0.03) = 0.95/0.97
    out = corrected_precision_recall(
        0.98, 0.80, truth_dup_rate=0.0, truth_overlay_rate=0.03
    )
    assert out["precision"] == pytest.approx(0.95 / 0.97)
    # overlays deflate observed recall (spurious truth pairs): R = 0.80 / 0.97
    assert out["recall"] == pytest.approx(0.80 / 0.97)
    # recall band under o=0.03, d=0: low = (0.80-0.03)/0.97, high = 0.80/0.97
    assert out["recall_low"] == pytest.approx(0.77 / 0.97)
    assert out["recall_high"] == pytest.approx(0.80 / 0.97)

    # dup widens the recall band by d on the high side: high = 0.9*0.8 + 0.1
    out = corrected_precision_recall(
        0.98, 0.80, truth_dup_rate=0.1, truth_overlay_rate=0.0
    )
    assert out["recall_low"] == pytest.approx(0.9 * 0.8)
    assert out["recall_high"] == pytest.approx(0.9 * 0.8 + 0.1)


def test_point_estimates_lie_within_bands():
    out = corrected_precision_recall(
        0.95,
        0.85,
        truth_dup_rate=0.03,
        truth_overlay_rate=0.02,
        dup_rate_ci=(0.01, 0.05),
        overlay_rate_ci=(0.01, 0.04),
    )
    assert out["precision_low"] <= out["precision"] <= out["precision_high"]
    assert out["recall_low"] <= out["recall"] <= out["recall_high"]
    for v in out.values():
        assert 0.0 <= v <= 1.0


def test_bands_widen_with_ci_inputs():
    kwargs = {"truth_dup_rate": 0.03, "truth_overlay_rate": 0.02}
    no_ci = corrected_precision_recall(0.95, 0.85, **kwargs)
    with_ci = corrected_precision_recall(
        0.95, 0.85, **kwargs, dup_rate_ci=(0.01, 0.05), overlay_rate_ci=(0.005, 0.04)
    )
    width = lambda o, m: o[f"{m}_high"] - o[f"{m}_low"]
    assert width(no_ci, "precision") == pytest.approx(0.0)  # exact algebra, no CI
    assert width(with_ci, "precision") > 0.0
    assert width(with_ci, "recall") > width(no_ci, "recall")
    # points are unchanged by adding CIs
    assert with_ci["precision"] == pytest.approx(no_ci["precision"])
    assert with_ci["recall"] == pytest.approx(no_ci["recall"])


def test_point_clipping_to_unit_interval():
    # observed precision below the overlay rate would invert negative: clipped to 0
    out = corrected_precision_recall(
        0.01, 0.99, truth_dup_rate=0.0, truth_overlay_rate=0.05
    )
    assert out["precision"] == 0.0
    # recall inflating past 1 is clipped
    assert out["recall"] == 1.0


def test_nan_observations_pass_through():
    out = corrected_precision_recall(
        float("nan"), 0.8, truth_dup_rate=0.02, truth_overlay_rate=0.01
    )
    assert math.isnan(out["precision"])
    assert math.isnan(out["precision_low"]) and math.isnan(out["precision_high"])
    assert out["recall"] == pytest.approx(0.8 / 0.99)


def test_validation_errors():
    with pytest.raises(ValueError, match="below 1"):
        corrected_precision_recall(0.9, 0.9, truth_dup_rate=0.6, truth_overlay_rate=0.5)
    with pytest.raises(ValueError, match="below 1"):
        corrected_precision_recall(
            0.9, 0.9, truth_dup_rate=0.1, truth_overlay_rate=0.1, dup_rate_ci=(0.1, 0.95)
        )
    with pytest.raises(ValueError, match="rate"):
        corrected_precision_recall(0.9, 0.9, truth_dup_rate=-0.1, truth_overlay_rate=0.0)
    with pytest.raises(ValueError, match="obs_precision"):
        corrected_precision_recall(1.2, 0.9, truth_dup_rate=0.0, truth_overlay_rate=0.0)
    with pytest.raises(ValueError, match="low <= high"):
        corrected_precision_recall(
            0.9, 0.9, truth_dup_rate=0.1, truth_overlay_rate=0.0, dup_rate_ci=(0.2, 0.1)
        )


# ---------------------------------------------------------------------------
# disagreement_sample
# ---------------------------------------------------------------------------


def _pair_frames():
    """Two labelings over shared pairs, with a 'county' stratum on side a.

    County X: 12 shared pairs, 8 disagreements; county Y: 6 shared pairs, 4
    disagreements; plus one a-only pair in X (counts as a disagreement).
    Total disagreements: X = 9, Y = 4, N = 13.
    """
    rows_a, rows_b = [], []
    i = 0

    def shared(county, n, n_disagree):
        nonlocal i
        for j in range(n):
            a, b = f"r{i:03d}", f"s{i:03d}"
            la = 1
            lb = 0 if j < n_disagree else 1
            rows_a.append((a, b, la, county))
            rows_b.append((a, b, lb))
            i += 1

    shared("X", 12, 8)
    shared("Y", 6, 4)
    rows_a.append(("only_a_1", "only_a_2", 1, "X"))  # absent from b -> disagreement
    pairs_a = pd.DataFrame(rows_a, columns=["a", "b", "label", "county"])
    pairs_b = pd.DataFrame(rows_b, columns=["a", "b", "label"])
    return pairs_a, pairs_b


def _disagreement_keys(pairs_a, pairs_b):
    a_keys = {(min(r.a, r.b), max(r.a, r.b)): r.label for r in pairs_a.itertuples()}
    b_keys = {(min(r.a, r.b), max(r.a, r.b)): r.label for r in pairs_b.itertuples()}
    keys = set(a_keys) | set(b_keys)
    return {k for k in keys if a_keys.get(k, "__x__") != b_keys.get(k, "__x__")}


def test_disagreement_sample_only_returns_disagreements():
    pairs_a, pairs_b = _pair_frames()
    out = disagreement_sample(pairs_a, pairs_b, strata=["county"], n=8, seed=0)
    truth = _disagreement_keys(pairs_a, pairs_b)
    got = {(r.a, r.b) for r in out.itertuples()}
    assert got <= truth
    assert len(out) == 8
    assert list(out.columns) == ["a", "b", "label_a", "label_b", "county", "weight"]


def test_disagreement_sample_stratification_and_weights():
    pairs_a, pairs_b = _pair_frames()
    out = disagreement_sample(pairs_a, pairs_b, strata=["county"], n=5, seed=3)
    # both strata represented (min-1 guarantee since n >= #strata)
    assert set(out["county"]) == {"X", "Y"}
    # per-stratum weight = N_h / n_h
    for county, n_h in (("X", 9), ("Y", 4)):
        grp = out[out["county"] == county]
        assert len(grp) >= 1
        assert grp["weight"].unique() == pytest.approx([n_h / len(grp)])
    # Horvitz-Thompson: weights sum to the disagreement-population size
    assert out["weight"].sum() == pytest.approx(13.0)


def test_disagreement_sample_full_population_when_n_large():
    pairs_a, pairs_b = _pair_frames()
    out = disagreement_sample(pairs_a, pairs_b, strata=["county"], n=1000, seed=1)
    assert len(out) == 13
    assert (out["weight"] == 1.0).all()
    # the a-only pair is included, with its b-side label missing
    only = out[out["a"] == "only_a_1"]
    assert len(only) == 1
    assert only["label_b"].isna().all()
    assert out["weight"].sum() == pytest.approx(13.0)


def test_disagreement_sample_deterministic_under_seed():
    pairs_a, pairs_b = _pair_frames()
    out1 = disagreement_sample(pairs_a, pairs_b, strata=["county"], n=6, seed=42)
    out2 = disagreement_sample(pairs_a, pairs_b, strata=["county"], n=6, seed=42)
    pd.testing.assert_frame_equal(out1, out2)
    out3 = disagreement_sample(pairs_a, pairs_b, strata=["county"], n=6, seed=43)
    assert not out1[["a", "b"]].equals(out3[["a", "b"]])


def test_disagreement_sample_unstratified_and_edge_cases():
    pairs_a, pairs_b = _pair_frames()
    out = disagreement_sample(pairs_a, pairs_b, n=4, seed=0)
    assert len(out) == 4
    assert out["weight"].sum() == pytest.approx(13.0)
    assert list(out.columns) == ["a", "b", "label_a", "label_b", "weight"]

    # no disagreements -> empty frame with the right columns
    same = pd.DataFrame({"a": ["x"], "b": ["y"], "label": [1]})
    empty = disagreement_sample(same, same.copy(), n=5, seed=0)
    assert len(empty) == 0
    assert list(empty.columns) == ["a", "b", "label_a", "label_b", "weight"]

    with pytest.raises(ValueError, match="n must be positive"):
        disagreement_sample(pairs_a, pairs_b, n=0, seed=0)
    with pytest.raises(KeyError, match="missing columns"):
        disagreement_sample(pairs_a.drop(columns=["label"]), pairs_b, n=2, seed=0)
    with pytest.raises(KeyError, match="stratum column"):
        disagreement_sample(pairs_a, pairs_b, strata=["nope"], n=2, seed=0)
    dup = pd.concat([pairs_a, pairs_a.iloc[[0]]], ignore_index=True)
    with pytest.raises(ValueError, match="duplicate unordered"):
        disagreement_sample(dup, pairs_b, n=2, seed=0)


def test_disagreement_sample_unordered_pair_matching():
    # the same pair written in opposite orders must be recognized as one pair
    pairs_a = pd.DataFrame({"a": ["u", "v"], "b": ["w", "x"], "label": [1, 1]})
    pairs_b = pd.DataFrame({"a": ["w", "x"], "b": ["u", "v"], "label": [1, 0]})
    out = disagreement_sample(pairs_a, pairs_b, n=10, seed=0)
    assert len(out) == 1  # only (v, x) disagrees; (u, w) agrees despite the flip
    assert set(out.loc[0, ["a", "b"]]) == {"v", "x"}


def test_disagreement_sample_fewer_draws_than_strata():
    rng = np.random.default_rng(0)
    pairs_a = pd.DataFrame(
        {
            "a": [f"a{i}" for i in range(6)],
            "b": [f"b{i}" for i in range(6)],
            "label": [1] * 6,
            "g": ["s1", "s1", "s1", "s2", "s2", "s3"],
        }
    )
    pairs_b = pairs_a.drop(columns=["g"]).assign(label=[0] * 6)
    out = disagreement_sample(pairs_a, pairs_b, strata=["g"], n=2, seed=int(rng.integers(100)))
    # 3 strata, 2 draws: the 2 largest strata get one draw each
    assert len(out) == 2
    assert set(out["g"]) == {"s1", "s2"}
