"""er_lab.eval.bootstrap — resampling-unit semantics, BCa, coverage smoke."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from pytest import approx

from er_lab.eval.bootstrap import _bca_interval, bootstrap_ci, paired_delta
from er_lab.eval.metrics import bcubed, pairwise

# entity archetypes: (truth size, pred split sizes). Pred only ever splits
# within an entity, so B-cubed precision is 1 and the population F1 is an
# exact function of the archetype mix — a clean estimand for coverage tests.
ARCHETYPES = [
    (3, (2, 1)),
    (2, (2,)),
    (2, (1, 1)),
    (4, (3, 1)),
    (1, (1,)),
]


def build(arch_ids: list[int]) -> tuple[pd.Series, pd.Series]:
    recs, tr, pr = [], [], []
    for e, a in enumerate(arch_ids):
        _, split = ARCHETYPES[a]
        k = 0
        for ci, csize in enumerate(split):
            for _ in range(csize):
                recs.append(f"e{e}r{k}")
                tr.append(f"E{e}")
                pr.append(f"E{e}c{ci}")
                k += 1
    idx = pd.Index(recs)
    return pd.Series(pr, index=idx), pd.Series(tr, index=idx)


def f1w(p: pd.Series, t: pd.Series, weights: pd.Series | None = None) -> float:
    return bcubed(p, t, weights=weights)["f1"]


TRUTH = pd.Series({"a": "T1", "b": "T1", "c": "T1", "d": "T2", "e": "T2"})
PRED = pd.Series({"a": "P1", "b": "P1", "c": "P2", "d": "P2", "e": "P2"})


# ------------------------------------------------------------- basic contract


def test_point_estimate_and_reproducibility() -> None:
    pred, truth = build([0, 1, 2, 3, 4, 0, 3])
    res = bootstrap_ci(pred, truth, f1w, unit="entity", n_boot=100, seed=42)
    assert res["point"] == approx(bcubed(pred, truth)["f1"])
    assert res["ci_low"] <= res["point"] <= res["ci_high"]
    assert res["method"] == "bca" and res["n_boot"] == 100
    assert isinstance(res["boots"], np.ndarray) and len(res["boots"]) == 100
    again = bootstrap_ci(pred, truth, f1w, unit="entity", n_boot=100, seed=42)
    np.testing.assert_array_equal(res["boots"], again["boots"])
    other = bootstrap_ci(pred, truth, f1w, unit="entity", n_boot=100, seed=43)
    assert not np.array_equal(res["boots"], other["boots"])


def test_weighted_path_equals_physical_duplication_path() -> None:
    # a metric exposing `weights` (frequency-weight fast path) and a plain
    # lambda (suffixed-id duplication fallback) must produce identical boots
    pred, truth = build([0, 1, 2, 3, 4, 0, 1, 2])
    kw = {"unit": "entity", "n_boot": 60, "seed": 11, "method": "percentile"}
    r_weighted = bootstrap_ci(pred, truth, f1w, **kw)
    r_duplicated = bootstrap_ci(pred, truth, lambda p, t: bcubed(p, t)["f1"], **kw)
    np.testing.assert_allclose(r_weighted["boots"], r_duplicated["boots"], rtol=1e-12)
    assert r_weighted["ci_low"] == approx(r_duplicated["ci_low"])
    assert r_weighted["ci_high"] == approx(r_duplicated["ci_high"])


def test_block_unit_with_entity_blocks_matches_entity_unit() -> None:
    pred, truth = build([0, 2, 3, 0, 1])
    kw = {"n_boot": 80, "seed": 5, "method": "percentile"}
    r_entity = bootstrap_ci(pred, truth, f1w, unit="entity", **kw)
    r_block = bootstrap_ci(pred, truth, f1w, unit="block", block=truth, **kw)
    np.testing.assert_array_equal(r_entity["boots"], r_block["boots"])


def test_pair_unit_point_matches_pairwise_metric() -> None:
    # the pair universe preserves pairwise-decomposable metrics exactly
    res = bootstrap_ci(
        PRED,
        TRUTH,
        lambda p, t: pairwise(p, t)["precision"],
        unit="pair",
        n_boot=50,
        seed=2,
        method="percentile",
    )
    assert res["point"] == approx(pairwise(PRED, TRUTH)["precision"])
    assert res["n_units"] == 6  # 4 pred pairs ∪ 4 truth pairs, 2 shared
    boots = res["boots"]
    # a pair resample can draw zero predicted-positive pairs -> precision is
    # NaN by the documented 0/0 convention; that brittleness on tiny data is
    # part of why the naive pair unit fails MET-03
    finite = boots[~np.isnan(boots)]
    assert len(finite) > 0.8 * len(boots)
    assert np.all((finite >= 0) & (finite <= 1))


def test_argument_validation() -> None:
    with pytest.raises(ValueError, match="unknown unit"):
        bootstrap_ci(PRED, TRUTH, f1w, unit="galaxy", seed=0)
    with pytest.raises(ValueError, match="requires a block Series"):
        bootstrap_ci(PRED, TRUTH, f1w, unit="block", seed=0)
    with pytest.raises(ValueError, match="unknown method"):
        bootstrap_ci(PRED, TRUTH, f1w, seed=0, method="magic")
    with pytest.raises(TypeError, match="must return a scalar"):
        bootstrap_ci(PRED, TRUTH, lambda p, t: bcubed(p, t), n_boot=2, seed=0)
    with pytest.raises(ValueError, match="not in pred"):
        bootstrap_ci(PRED.rename(index={"a": "zz"}), TRUTH, f1w, seed=0)


# ------------------------------------------------------- statistical behavior


def test_ci_width_shrinks_like_sqrt_of_entity_count() -> None:
    # same archetype mix at E=40 vs E=160 entities: width ratio should sit
    # near sqrt(160/40) = 2
    def width(n_entities: int) -> float:
        pred, truth = build([i % 5 for i in range(n_entities)])
        r = bootstrap_ci(
            pred, truth, f1w, unit="entity", n_boot=400, seed=7, method="percentile"
        )
        return r["ci_high"] - r["ci_low"]

    ratio = width(40) / width(160)
    assert 1.5 < ratio < 2.7


def test_bca_reduces_to_percentile_when_symmetric_and_unaccelerated() -> None:
    # boots exactly symmetric about the point (half strictly below -> z0 = 0)
    # and constant jackknife values (acceleration a = 0): BCa == percentile
    point = 0.5
    deltas = np.linspace(0.01, 0.2, 50)
    boots = np.concatenate([point - deltas, point + deltas])
    jack = np.full(10, 0.42)
    lo, hi = _bca_interval(boots, point, jack, alpha=0.05)
    p_lo, p_hi = np.quantile(boots, [0.025, 0.975])
    assert lo == approx(p_lo, abs=1e-9)
    assert hi == approx(p_hi, abs=1e-9)


def test_bca_differs_from_percentile_when_skewed() -> None:
    # sanity that the correction actually does something on skewed boots
    point = 1.0
    boots = np.concatenate([np.full(80, 0.9) + np.linspace(0, 0.1, 80), np.linspace(1.0, 3.0, 20)])
    jack = np.array([0.5, 0.9, 1.0, 1.1, 1.2, 3.0])  # skewed influence
    lo, hi = _bca_interval(boots, point, jack, alpha=0.05)
    p_lo, p_hi = np.quantile(boots, [0.025, 0.975])
    assert (lo, hi) != (approx(p_lo), approx(p_hi))


def test_paired_delta_identical_systems_degenerate() -> None:
    pred, truth = build([0, 1, 2, 3, 4])
    res = paired_delta(pred, pred.copy(), truth, f1w, unit="entity", n_boot=100, seed=3)
    assert res["delta"] == 0.0
    assert res["ci_low"] == 0.0 and res["ci_high"] == 0.0
    assert res["sign_stable"] is False


def test_paired_delta_shared_draws_detect_real_difference() -> None:
    # system A = perfect clustering, system B = archetype splits: every
    # shared replicate has delta > 0, so the CI must exclude zero
    pred_b, truth = build([i % 5 for i in range(30)])
    pred_a = truth.copy()
    res = paired_delta(pred_a, pred_b, truth, f1w, unit="entity", n_boot=200, seed=9)
    assert res["delta"] == approx(1.0 - bcubed(pred_b, truth)["f1"])
    assert res["ci_low"] > 0
    assert res["sign_stable"] is True
    assert (res["boots"] > 0).all()


def test_entity_coverage_smoke() -> None:
    # 200 tiny simulations (~20 entities): >= 90% of nominal-95% entity-unit
    # BCa intervals must contain the population B-cubed F1. The population
    # value is exact: archetypes are drawn uniformly, and with within-entity
    # splits only, record-averaged P/R are ratio statistics whose population
    # value is the equal-weight archetype mixture -> bcubed on one copy of
    # each archetype.
    pop_pred, pop_truth = build([0, 1, 2, 3, 4])
    pop_f1 = bcubed(pop_pred, pop_truth)["f1"]

    n_sims, hits = 200, 0
    for i in range(n_sims):
        rng = np.random.default_rng(10_000 + i)
        pred, truth = build(list(rng.integers(0, 5, 20)))
        res = bootstrap_ci(
            pred, truth, f1w, unit="entity", n_boot=200, seed=i, method="bca"
        )
        hits += res["ci_low"] <= pop_f1 <= res["ci_high"]
    coverage = hits / n_sims
    assert coverage >= 0.90, f"entity-unit coverage {coverage:.3f} < 0.90"
