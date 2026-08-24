"""Tests for er_lab.score.calibrate — isotonic/Platt score->probability calibration."""

from __future__ import annotations

import numpy as np
import pytest

from er_lab.score.calibrate import fit_calibrator

RNG = np.random.default_rng(7)


def noisy_data(n: int = 400) -> tuple[np.ndarray, np.ndarray]:
    """Scores in [0, 1] where higher score means higher match probability."""
    scores = RNG.uniform(0, 1, size=n)
    labels = (RNG.uniform(0, 1, size=n) < scores**2).astype(float)
    return scores, labels


def test_isotonic_monotonic_in_score():
    scores, labels = noisy_data()
    cal = fit_calibrator(scores, labels, method="isotonic")
    grid = np.linspace(-0.2, 1.2, 200)  # includes out-of-range values
    probs = cal.transform(grid)
    assert np.all(np.diff(probs) >= 0)  # monotone non-decreasing
    assert probs.min() >= 0.0 and probs.max() <= 1.0  # clipped, never extrapolated


def test_platt_on_separable_toy():
    scores = np.concatenate([RNG.uniform(-3, -1, 50), RNG.uniform(1, 3, 50)])
    labels = np.concatenate([np.zeros(50), np.ones(50)])
    cal = fit_calibrator(scores, labels, method="platt")
    probs = cal.transform(np.array([-3.0, -1.5, 0.0, 1.5, 3.0]))
    assert np.all(np.diff(probs) > 0)  # strictly monotone (logistic)
    assert probs[0] < 0.01 and probs[-1] > 0.99  # confident at the extremes
    assert probs[1] < 0.5 < probs[3]


@pytest.mark.parametrize("method", ["isotonic", "platt"])
def test_per_stratum_fit_differs_when_strata_differ(method: str):
    # same score distribution, opposite meaning per stratum: in 'common' a 0.7
    # score is weak evidence; in 'rare' the same 0.7 is strong evidence
    n = 300
    scores = np.tile(np.linspace(0.01, 0.99, n // 2), 2)
    strata = np.array(["common"] * (n // 2) + ["rare"] * (n // 2))
    p_true = np.where(strata == "common", scores**3, scores**0.33)
    labels = (RNG.uniform(0, 1, size=n) < p_true).astype(float)
    cal = fit_calibrator(scores, labels, method=method, strata=strata)
    assert cal.stratified
    same_score = np.array([0.7, 0.7])
    both = cal.transform(same_score, strata=np.array(["common", "rare"]))
    assert both[1] > both[0] + 0.15  # same raw score, very different probability


def test_stratified_transform_requires_strata():
    scores, labels = noisy_data()
    strata = np.array(["a", "b"] * (len(scores) // 2))
    cal = fit_calibrator(scores, labels, method="isotonic", strata=strata)
    with pytest.raises(ValueError, match="fitted per-stratum"):
        cal.transform(scores)


def test_unseen_stratum_falls_back_to_pooled():
    scores, labels = noisy_data()
    strata = np.array(["a", "b"] * (len(scores) // 2))
    cal = fit_calibrator(scores, labels, method="isotonic", strata=strata)
    pooled = fit_calibrator(scores, labels, method="isotonic")
    grid = np.linspace(0, 1, 50)
    np.testing.assert_allclose(
        cal.transform(grid, strata=np.array(["zz"] * 50)), pooled.transform(grid)
    )


def test_unfittable_stratum_recorded_and_falls_back():
    scores = np.array([0.1, 0.2, 0.8, 0.9, 0.5, 0.6])
    labels = np.array([0.0, 0.0, 1.0, 1.0, 1.0, 1.0])
    strata = np.array(["ok", "ok", "ok", "ok", "pure", "pure"])  # 'pure' has one class
    cal = fit_calibrator(scores, labels, method="platt", strata=strata)
    assert cal.fallback_strata == ["pure"]
    assert "pure" not in cal.models and "ok" in cal.models


def test_determinism():
    scores, labels = noisy_data()
    grid = np.linspace(0, 1, 100)
    for method in ("isotonic", "platt"):
        a = fit_calibrator(scores, labels, method=method).transform(grid)
        b = fit_calibrator(scores, labels, method=method).transform(grid)
        np.testing.assert_array_equal(a, b)


def test_input_validation():
    scores, labels = noisy_data()
    with pytest.raises(ValueError, match="unknown method"):
        fit_calibrator(scores, labels, method="beta")
    with pytest.raises(ValueError, match="binary"):
        fit_calibrator(scores, scores, method="isotonic")
    with pytest.raises(ValueError, match="both classes"):
        fit_calibrator(scores, np.ones_like(scores), method="platt")
    with pytest.raises(ValueError, match="aligned"):
        fit_calibrator(scores, labels[:-1], method="isotonic")
    cal = fit_calibrator(scores, labels, method="isotonic", strata=np.array(["a"] * len(scores)))
    with pytest.raises(ValueError, match="entries"):
        cal.transform(scores, strata=np.array(["a", "b"]))
