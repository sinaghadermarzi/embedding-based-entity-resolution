"""er_lab.eval.bootstrap — resampling-unit semantics, multiplier path, BCa, coverage."""

from __future__ import annotations

import itertools

import numpy as np
import pandas as pd
import pytest
from pytest import approx

from er_lab.eval.bootstrap import _bca_interval, _make_stats, bootstrap_ci, paired_delta
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


# scene archetypes with SPLIT AND MERGE errors: (entity sizes, pred clusters
# as lists of (entity_idx, n_records)). Merge scenes have pred clusters that
# span the scene's two truth entities — the regime where the old duplication
# resampling manufactured always-TP self-pairs and destroyed coverage.
SCENES = [
    ((3,), [[(0, 2)], [(0, 1)]]),
    ((2,), [[(0, 2)]]),
    ((2,), [[(0, 1)], [(0, 1)]]),
    ((4,), [[(0, 3)], [(0, 1)]]),
    ((1,), [[(0, 1)]]),
    ((2, 2), [[(0, 1), (1, 1)], [(0, 1)], [(1, 1)]]),  # partial cross-entity merge
    ((3, 2), [[(0, 3), (1, 2)]]),  # full merge of two entities
    ((1, 1), [[(0, 1), (1, 1)]]),  # two singletons merged
]


def build_scenes(scene_ids: list[int]) -> tuple[pd.Series, pd.Series]:
    recs, tr, pr = [], [], []
    for s, sid in enumerate(scene_ids):
        sizes, clusters = SCENES[sid]
        counters = [0] * len(sizes)
        for ci, cluster in enumerate(clusters):
            for ei, cnt in cluster:
                for _ in range(cnt):
                    recs.append(f"s{s}e{ei}r{counters[ei]}")
                    tr.append(f"S{s}E{ei}")
                    pr.append(f"S{s}C{ci}")
                    counters[ei] += 1
    idx = pd.Index(recs)
    return pd.Series(pr, index=idx), pd.Series(tr, index=idx)


def f1w(p: pd.Series, t: pd.Series, weights: pd.Series | None = None) -> float:
    return bcubed(p, t, weights=weights)["f1"]


TRUTH = pd.Series({"a": "T1", "b": "T1", "c": "T1", "d": "T2", "e": "T2"})
PRED = pd.Series({"a": "P1", "b": "P1", "c": "P2", "d": "P2", "e": "P2"})


# ------------------------------------------------------------- basic contract


def test_point_estimate_and_reproducibility() -> None:
    pred, truth = build([0, 1, 2, 3, 4, 0, 3])
    res = bootstrap_ci(pred, truth, "bcubed_f1", unit="entity", n_boot=100, seed=42)
    assert res["point"] == approx(bcubed(pred, truth)["f1"])
    assert res["ci_low"] <= res["point"] <= res["ci_high"]
    assert res["method"] == "bca" and res["n_boot"] == 100
    assert res["path"] == "multiplier"
    assert isinstance(res["boots"], np.ndarray) and len(res["boots"]) == 100
    again = bootstrap_ci(pred, truth, "bcubed_f1", unit="entity", n_boot=100, seed=42)
    np.testing.assert_array_equal(res["boots"], again["boots"])
    other = bootstrap_ci(pred, truth, "bcubed_f1", unit="entity", n_boot=100, seed=43)
    assert not np.array_equal(res["boots"], other["boots"])


def _reference_multiplier(
    pred: pd.Series,
    truth: pd.Series,
    unit_of: dict[str, int],
    counts: np.ndarray,
    name: str,
) -> float:
    """Brute-force frozen-contribution estimator: naive loops, no vectorization.

    Recomputes, from scratch on the index-selected observed frame, what the
    multiplier bootstrap must produce for a given unit-draw count vector:
    B-cubed as the count-weighted mean of per-record contributions, pairwise
    from frozen pair indicators with weight c_u(r)*c_u(s) across units and
    c_u(r) within a unit.
    """
    ids = list(truth.index)
    c = {i: int(counts[unit_of[i]]) for i in ids}
    if name.startswith("bcubed"):
        num_p = num_r = den = 0.0
        for r in ids:
            same_pred = [s for s in ids if pred[s] == pred[r]]
            same_truth = [s for s in ids if truth[s] == truth[r]]
            overlap = len(set(same_pred) & set(same_truth))
            den += c[r]
            num_p += c[r] * overlap / len(same_pred)
            num_r += c[r] * overlap / len(same_truth)
        p, r_ = num_p / den, num_r / den
    else:
        tp = fp = fn = 0.0
        for a, b in itertools.combinations(ids, 2):
            co_p, co_t = pred[a] == pred[b], truth[a] == truth[b]
            if not (co_p or co_t):
                continue
            w = c[a] if unit_of[a] == unit_of[b] else c[a] * c[b]
            if co_p and co_t:
                tp += w
            elif co_p:
                fp += w
            else:
                fn += w
        p = tp / (tp + fp) if tp + fp > 0 else float("nan")
        r_ = tp / (tp + fn) if tp + fn > 0 else float("nan")
    if name.endswith("precision"):
        return p
    if name.endswith("recall"):
        return r_
    if np.isnan(p) or np.isnan(r_):
        return float("nan")
    return 0.0 if p + r_ == 0 else 2 * p * r_ / (p + r_)


def test_multiplier_stat_matches_bruteforce_reference() -> None:
    # The old test here asserted weighted == duplicated — which validated the
    # BIASED estimator (a weight-w record paired with its own copies). The
    # multiplier path must instead agree with a brute-force frozen-contribution
    # reference on a world containing splits AND cross-entity merges.
    idx = pd.Index([f"r{i}" for i in range(11)])
    truth = pd.Series(["A", "A", "A", "B", "B", "C", "C", "D", "E", "E", "E"], index=idx)
    pred = pd.Series(["p1", "p1", "p2", "p2", "p3", "p3", "p3", "p4", "p5", "p6", "p6"], index=idx)
    block = pd.Series(["b1", "b1", "b1", "b1", "b2", "b2", "b2", "b3", "b3", "b3", "b3"], index=idx)

    rng = np.random.default_rng(0)
    for unit in ("entity", "record", "block"):
        if unit == "entity":
            unit_codes = pd.factorize(truth.to_numpy())[0]
        elif unit == "record":
            unit_codes = np.arange(len(idx))
        else:
            unit_codes = pd.factorize(block.to_numpy())[0]
        unit_of = {rec: int(unit_codes[j]) for j, rec in enumerate(idx)}
        for name in ("bcubed_precision", "bcubed_f1", "pairwise_precision", "pairwise_f1"):
            stats, n_units, path = _make_stats(
                [pred], truth, name, unit, block if unit == "block" else None
            )
            assert path == "multiplier"
            # point estimate (all-ones counts) reduces to the observed metric
            fam, _, fld = name.partition("_")
            observed = (bcubed if fam == "bcubed" else pairwise)(pred, truth)[fld]
            assert stats[0](np.ones(n_units, dtype=np.int64)) == approx(observed)
            for _ in range(10):
                counts = np.bincount(rng.integers(0, n_units, n_units), minlength=n_units)
                got = stats[0](counts)
                want = _reference_multiplier(pred, truth, unit_of, counts, name)
                if np.isnan(want):
                    assert np.isnan(got)
                else:
                    assert got == approx(want), (unit, name, counts)


def test_callable_metric_uses_documented_duplication_fallback() -> None:
    # callables (needed for VI/GMD/cluster_f, which have no per-unit
    # decomposition) go through the duplication fallback; the result says so
    pred, truth = build([0, 1, 2, 3])
    res = bootstrap_ci(pred, truth, f1w, unit="entity", n_boot=30, seed=1, method="percentile")
    assert res["path"] == "duplication"
    assert res["point"] == approx(bcubed(pred, truth)["f1"])
    # weights fast path and physical duplication are the same fallback estimator
    res_dup = bootstrap_ci(
        pred,
        truth,
        lambda p, t: bcubed(p, t)["f1"],
        unit="entity",
        n_boot=30,
        seed=1,
        method="percentile",
    )
    np.testing.assert_allclose(res["boots"], res_dup["boots"], rtol=1e-12)


def test_block_unit_with_entity_blocks_matches_entity_unit() -> None:
    pred, truth = build([0, 2, 3, 0, 1])
    kw = {"n_boot": 80, "seed": 5, "method": "percentile"}
    r_entity = bootstrap_ci(pred, truth, "bcubed_f1", unit="entity", **kw)
    r_block = bootstrap_ci(pred, truth, "bcubed_f1", unit="block", block=truth, **kw)
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
    assert res["path"] == "pair"
    boots = res["boots"]
    # a pair resample can draw zero predicted-positive pairs -> precision is
    # NaN by the documented 0/0 convention; that brittleness on tiny data is
    # part of why the naive pair unit fails MET-03
    finite = boots[~np.isnan(boots)]
    assert len(finite) > 0.8 * len(boots)
    assert np.all((finite >= 0) & (finite <= 1))


def test_argument_validation() -> None:
    with pytest.raises(ValueError, match="unknown unit"):
        bootstrap_ci(PRED, TRUTH, "bcubed_f1", unit="galaxy", seed=0)
    with pytest.raises(ValueError, match="requires a block Series"):
        bootstrap_ci(PRED, TRUTH, "bcubed_f1", unit="block", seed=0)
    with pytest.raises(ValueError, match="unknown method"):
        bootstrap_ci(PRED, TRUTH, "bcubed_f1", seed=0, method="magic")
    with pytest.raises(ValueError, match="unknown metric name"):
        bootstrap_ci(PRED, TRUTH, "bcubed_f2", seed=0)
    with pytest.raises(TypeError, match="must return a scalar"):
        bootstrap_ci(PRED, TRUTH, lambda p, t: bcubed(p, t), n_boot=2, seed=0)
    with pytest.raises(ValueError, match="not in pred"):
        bootstrap_ci(PRED.rename(index={"a": "zz"}), TRUTH, "bcubed_f1", seed=0)


# ------------------------------------------------------------- NaN replicates


def test_nan_replicates_dropped_and_reported() -> None:
    # pair unit on the tiny PRED/TRUTH world: exactly one replicate draws no
    # predicted-positive pairs (seeded, deterministic) -> NaN, which must be
    # excluded from the interval and reported, never reach np.quantile
    res = bootstrap_ci(
        PRED,
        TRUTH,
        lambda p, t: pairwise(p, t)["precision"],
        unit="pair",
        n_boot=50,
        seed=2,
        method="percentile",
    )
    assert res["n_nan"] == 1
    assert res["n_boot_effective"] == 49
    assert res["n_nan"] + res["n_boot_effective"] == res["n_boot"]
    assert np.isnan(res["boots"]).sum() == 1  # raw boots keep the NaN for inspection
    assert np.isfinite(res["ci_low"]) and np.isfinite(res["ci_high"])
    assert res["ci_low"] <= res["ci_high"]
    # BCa on the same draw must not crash either (NaN used to poison z0/quantiles)
    res_bca = bootstrap_ci(
        PRED,
        TRUTH,
        lambda p, t: pairwise(p, t)["precision"],
        unit="pair",
        n_boot=50,
        seed=2,
        method="bca",
    )
    assert np.isfinite(res_bca["ci_low"]) and np.isfinite(res_bca["ci_high"])


def test_nan_replicates_raise_when_excessive() -> None:
    # 1 predicted pair among 6 truth pairs: ~a third of pair draws contain no
    # predicted-positive pair -> NaN fraction > 20% must raise, not return a
    # silently-NaN or half-baked interval
    truth = pd.Series({"a": "T1", "b": "T1", "c": "T1", "d": "T1", "e": "E2", "f": "E2"})
    pred = pd.Series({"a": "P1", "b": "P1", "c": "C3", "d": "C4", "e": "C5", "f": "C6"})
    with pytest.raises(ValueError, match=r"replicates are NaN"):
        bootstrap_ci(
            pred,
            truth,
            lambda p, t: pairwise(p, t)["precision"],
            unit="pair",
            n_boot=100,
            seed=0,
            method="percentile",
        )


def test_nan_point_estimate_raises() -> None:
    # all-singleton pred: pairwise precision is NaN on the observed data itself
    truth = pd.Series({"a": "T1", "b": "T1", "c": "T2", "d": "T2"})
    pred = pd.Series({"a": "P1", "b": "P2", "c": "P3", "d": "P4"})
    with pytest.raises(ValueError, match="point estimate is NaN"):
        bootstrap_ci(pred, truth, "pairwise_precision", unit="entity", n_boot=50, seed=0)


def test_degenerate_fp_only_world_refuses_instead_of_lying() -> None:
    # The reviewer's degenerate illustration: all-singleton truth with one
    # pred link gave point precision 0.0 with a CI EXCLUDING its own point
    # under the old duplication resampling (self-pairs manufactured TPs).
    # Under the multiplier path most replicates drop the only relevant pair
    # (precision 0/0 = NaN), so the honest outcome is an informative refusal.
    truth = pd.Series({"a": "T1", "b": "T2", "c": "T3", "d": "T4"})
    pred = pd.Series({"a": "P1", "b": "P1", "c": "P2", "d": "P3"})
    assert pairwise(pred, truth)["precision"] == 0.0
    with pytest.raises(ValueError, match=r"replicates are NaN"):
        bootstrap_ci(pred, truth, "pairwise_precision", unit="entity", n_boot=100, seed=1)


def test_multiplier_degenerate_perfect_pred_ci_contains_point() -> None:
    res = bootstrap_ci(TRUTH.copy(), TRUTH, "bcubed_f1", unit="entity", n_boot=50, seed=0)
    assert res["point"] == 1.0
    assert res["ci_low"] == 1.0 and res["ci_high"] == 1.0
    assert res["path"] == "multiplier"


# ------------------------------------------------------- statistical behavior


def test_ci_width_shrinks_like_sqrt_of_entity_count() -> None:
    # same archetype mix at E=40 vs E=160 entities: width ratio should sit
    # near sqrt(160/40) = 2
    def width(n_entities: int) -> float:
        pred, truth = build([i % 5 for i in range(n_entities)])
        r = bootstrap_ci(
            pred, truth, "bcubed_f1", unit="entity", n_boot=400, seed=7, method="percentile"
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
    assert res["path"] == "duplication"


def test_paired_delta_shared_draws_detect_real_difference() -> None:
    # system A = perfect clustering, system B = archetype splits: every
    # shared replicate has delta > 0, so the CI must exclude zero
    pred_b, truth = build([i % 5 for i in range(30)])
    pred_a = truth.copy()
    res = paired_delta(pred_a, pred_b, truth, "bcubed_f1", unit="entity", n_boot=200, seed=9)
    assert res["delta"] == approx(1.0 - bcubed(pred_b, truth)["f1"])
    assert res["ci_low"] > 0
    assert res["sign_stable"] is True
    assert res["path"] == "multiplier"
    assert (res["boots"] > 0).all()


def test_entity_coverage_smoke() -> None:
    # 200 tiny simulations (~20 entities): >= 90% of nominal-95% entity-unit
    # BCa intervals must contain the population B-cubed F1. The population
    # value is exact: archetypes are drawn uniformly, and with within-entity
    # splits only, record-averaged P/R are ratio statistics whose population
    # value is the equal-weight archetype mixture -> bcubed on one copy of
    # each archetype. (Split-only worlds are the one regime where the
    # duplication fallback coincides with the multiplier estimator, so the
    # callable path keeps coverage here; see the merge-error test below for
    # the regime where only the multiplier path survives.)
    pop_pred, pop_truth = build([0, 1, 2, 3, 4])
    pop_f1 = bcubed(pop_pred, pop_truth)["f1"]

    n_sims, hits = 200, 0
    for i in range(n_sims):
        rng = np.random.default_rng(10_000 + i)
        pred, truth = build(list(rng.integers(0, 5, 20)))
        res = bootstrap_ci(pred, truth, f1w, unit="entity", n_boot=200, seed=i, method="bca")
        hits += res["ci_low"] <= pop_f1 <= res["ci_high"]
    coverage = hits / n_sims
    assert coverage >= 0.90, f"entity-unit coverage {coverage:.3f} < 0.90"


def test_multiplier_coverage_with_merge_errors() -> None:
    # The critical MET-03 regression (reviewer finding): worlds with mixed
    # split AND cross-entity merge errors, ~99 truth entities each, 100 sims.
    # The old duplication resampling measured 0.18 (percentile) / 0.08 (BCa)
    # coverage of nominal-95% B-cubed F1 CIs on this exact seeded simulation,
    # with a +0.036 upward boot-mean shift; the multiplier path measures
    # 0.90 / 0.93 with shift -0.0007. Population estimand is exact: scenes
    # are drawn uniformly and scene-local, so the ratio-statistic limit is
    # bcubed on one copy of each scene.
    pop_pred, pop_truth = build_scenes(list(range(len(SCENES))))
    pop_f1 = bcubed(pop_pred, pop_truth)["f1"]

    n_sims, n_scenes, n_boot = 100, 72, 200
    hits = {"percentile": 0, "bca": 0}
    shift = 0.0
    for i in range(n_sims):
        rng = np.random.default_rng(50_000 + i)
        pred, truth = build_scenes(list(rng.integers(0, len(SCENES), n_scenes)))
        for method in ("percentile", "bca"):
            res = bootstrap_ci(
                pred, truth, "bcubed_f1", unit="entity", n_boot=n_boot, seed=i, method=method
            )
            assert res["path"] == "multiplier"
            hits[method] += res["ci_low"] <= pop_f1 <= res["ci_high"]
            if method == "percentile":
                shift += (res["boots"].mean() - res["point"]) / n_sims
    assert hits["bca"] / n_sims >= 0.90, f"BCa coverage {hits['bca'] / n_sims:.2f} < 0.90"
    cov_pct = hits["percentile"] / n_sims
    assert cov_pct >= 0.88, f"percentile coverage {cov_pct:.2f} < 0.88"
    # the O(1) upward resampling bias of the duplication path is gone
    assert abs(shift) < 0.005, f"boot-mean shift {shift:+.4f} not ~0"
