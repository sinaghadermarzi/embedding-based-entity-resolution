"""Resampling-unit-aware bootstrap CIs for ER metrics (PLAN.md §5, MET-03).

ER records are not i.i.d.: records of one entity share their fate, so the
choice of resampling unit decides whether a bootstrap CI has nominal coverage
— the question MET-03 settles empirically with this module. Units:

- ``'entity'`` (the lab's expected default): resample TRUTH entities with
  replacement; the resampled evaluation set is the induced multiset of
  records — a record appears as many times as its entity was drawn.
- ``'record'``: resample records i.i.d. (ignores within-entity dependence).
- ``'block'``: like entity but over caller-supplied block labels (e.g.
  households), for dependence wider than the entity.
- ``'pair'``: the naive protocol — resample the multiset of *relevant* pairs
  (pairs co-clustered in pred or truth; true negatives are ~n² and excluded)
  and rebuild a two-record-per-pair evaluation universe. This reproduces
  pairwise-decomposable metrics exactly and deliberately discards
  higher-order cluster structure; for unit='pair' the point estimate is the
  metric on that pair universe, which equals metric(pred, truth) only for
  pairwise-decomposable metrics.

Weighted views: for record/entity/block units the resampled multiset is
expressed as per-record frequency weights. If the metric callable exposes a
``weights`` keyword (as every ``er_lab.eval.metrics`` partition metric does —
wrap like ``lambda p, t, weights=None: bcubed(p, t, weights=weights)["f1"]``),
weights are passed directly and no data is copied. Otherwise the fallback
physically duplicates records with suffixed ids (``id__b1``, ``id__b2``, …):
same numbers, but O(Σw) index/frame construction per replicate — fine at
evaluation scale, wasteful in tight loops.

Intervals: ``method='bca'`` implements Efron's bias-corrected-and-accelerated
interval (Efron 1987; Efron & Tibshirani 1993 ch. 14): bias correction z0
from the fraction of bootstrap replicates below the point estimate (ties
split evenly, proportion clipped to [1/(B+1), B/(B+1)]), acceleration a from
the jackknife over the *resampling unit* (leave-one-unit-out), a =
Σd³ / (6 (Σd²)^{3/2}) with d = mean(jack) − jack. When the bootstrap
distribution is symmetric about the point estimate and a ≈ 0, BCa reduces to
the percentile interval (tested). Degenerate case: if all replicates are
equal the CI collapses to that value under either method.
"""

from __future__ import annotations

import inspect
import itertools
from collections.abc import Callable

import numpy as np
import pandas as pd
from scipy.special import ndtr, ndtri

__all__ = ["bootstrap_ci", "paired_delta"]

_UNITS = ("entity", "record", "pair", "block")


# ------------------------------------------------------------------ helpers


def _aligned_series(pred: pd.Series, truth: pd.Series) -> pd.Series:
    """Validate pred against truth and return pred reindexed to truth's order."""
    for name, s in (("pred", pred), ("truth", truth)):
        if not isinstance(s, pd.Series):
            raise TypeError(f"{name} must be a pandas Series, got {type(s).__name__}")
    if len(truth) == 0:
        raise ValueError("truth is empty")
    if len(pred) != len(truth):
        raise ValueError(f"pred has {len(pred)} records but truth has {len(truth)}")
    if not truth.index.is_unique:
        raise ValueError("truth index has duplicate record ids")
    if not pred.index.is_unique:
        raise ValueError("pred index has duplicate record ids")
    if pred.index is truth.index or pred.index.equals(truth.index):
        return pred
    missing = truth.index.difference(pred.index)
    if len(missing):
        raise ValueError(f"record ids in truth but not in pred: {list(missing[:5])}")
    return pred.reindex(truth.index)


def _supports_weights(metric: Callable) -> bool:
    """A metric opts into the no-copy path by exposing a 'weights' parameter."""
    try:
        sig = inspect.signature(metric)
    except (TypeError, ValueError):  # builtins / odd callables
        return False
    return "weights" in sig.parameters


def _scalar(value: object) -> float:
    if isinstance(value, dict):
        raise TypeError(
            "metric must return a scalar; wrap dict-returning metrics, e.g. "
            "lambda p, t, weights=None: bcubed(p, t, weights=weights)['f1']"
        )
    return float(value)  # type: ignore[arg-type]


def _duplicated(
    pred: pd.Series, truth: pd.Series, w: np.ndarray
) -> tuple[pd.Series, pd.Series]:
    """Materialize integer frequency weights as physically duplicated records.

    Copies beyond the first get suffixed ids (``id__b1``, ``id__b2``, …) so
    the duplicated index stays unique. Cost: O(Σw) per call — documented
    fallback for metrics without a ``weights`` keyword.
    """
    reps = w.astype(np.int64)
    keep = np.flatnonzero(reps > 0)
    kreps = reps[keep]
    pos = np.repeat(keep, kreps)
    occ = np.arange(len(pos), dtype=np.int64)
    starts = np.cumsum(kreps) - kreps
    occ -= np.repeat(starts, kreps)
    ids = pred.index.to_numpy()[pos].astype(str)
    extra = occ > 0
    if extra.any():
        suffix = np.where(extra, "__b", "")
        num = np.where(extra, occ.astype(str), "")
        ids = np.char.add(np.char.add(ids, suffix), num)
    idx = pd.Index(ids)
    return (
        pd.Series(pred.to_numpy()[pos], index=idx),
        pd.Series(truth.to_numpy()[pos], index=idx),
    )


def _co_pair_keys(codes: np.ndarray) -> np.ndarray:
    """Unordered co-clustered position pairs, encoded as i * n + j with i < j."""
    n = len(codes)
    order = np.argsort(codes, kind="stable")
    sorted_codes = codes[order]
    bounds = np.flatnonzero(np.r_[True, sorted_codes[1:] != sorted_codes[:-1], True])
    keys: list[np.ndarray] = []
    for s, e in itertools.pairwise(bounds):
        group = np.sort(order[s:e])
        if len(group) >= 2:
            ii, jj = np.triu_indices(len(group), k=1)
            keys.append(group[ii].astype(np.int64) * n + group[jj])
    if not keys:
        return np.empty(0, dtype=np.int64)
    return np.concatenate(keys)


def _make_pair_stat(
    metric: Callable, co_pred: np.ndarray, co_truth: np.ndarray
) -> Callable[[np.ndarray], float]:
    """Statistic over pair-draw counts: rebuild the two-record-per-pair universe."""

    def stat(counts: np.ndarray) -> float:
        kidx = np.repeat(np.arange(len(counts)), counts)
        m = len(kidx)
        occ = np.arange(m, dtype=np.int64)
        pv = np.empty(2 * m, dtype=np.int64)
        tv = np.empty(2 * m, dtype=np.int64)
        pv[0::2] = 2 * occ
        pv[1::2] = np.where(co_pred[kidx], 2 * occ, 2 * occ + 1)
        tv[0::2] = 2 * occ
        tv[1::2] = np.where(co_truth[kidx], 2 * occ, 2 * occ + 1)
        idx = pd.RangeIndex(2 * m)
        return _scalar(metric(pd.Series(pv, index=idx), pd.Series(tv, index=idx)))

    return stat


def _make_weight_stat(
    metric: Callable, pred: pd.Series, truth: pd.Series, unit_codes: np.ndarray
) -> Callable[[np.ndarray], float]:
    """Statistic over unit-draw counts, via frequency weights or duplication."""
    if _supports_weights(metric):
        idx = truth.index

        def stat(counts: np.ndarray) -> float:
            w = counts[unit_codes].astype(float)
            return _scalar(metric(pred, truth, weights=pd.Series(w, index=idx)))

    else:

        def stat(counts: np.ndarray) -> float:
            w = counts[unit_codes]
            p2, t2 = _duplicated(pred, truth, w)
            return _scalar(metric(p2, t2))

    return stat


def _make_stats(
    preds: list[pd.Series],
    truth: pd.Series,
    metric: Callable,
    unit: str,
    block: pd.Series | None,
) -> tuple[list[Callable[[np.ndarray], float]], int]:
    """Build per-system statistic functions sharing one resampling structure."""
    if unit not in _UNITS:
        raise ValueError(f"unknown unit {unit!r}; expected one of {_UNITS}")
    aligned = [_aligned_series(p, truth) for p in preds]
    if truth.isna().any():
        raise ValueError("truth contains missing entity ids")

    if unit == "pair":
        t_codes = pd.factorize(truth.to_numpy())[0]
        truth_keys = _co_pair_keys(t_codes)
        pred_keys = [_co_pair_keys(pd.factorize(p.to_numpy())[0]) for p in aligned]
        all_keys = np.unique(np.concatenate([truth_keys, *pred_keys]))
        if len(all_keys) < 2:
            raise ValueError(
                "pair-unit bootstrap needs >= 2 co-clustered pairs across pred and truth"
            )
        co_truth = np.isin(all_keys, truth_keys)
        stats = [
            _make_pair_stat(metric, np.isin(all_keys, keys), co_truth) for keys in pred_keys
        ]
        return stats, len(all_keys)

    if unit == "entity":
        unit_codes = pd.factorize(truth.to_numpy())[0]
    elif unit == "record":
        unit_codes = np.arange(len(truth))
    else:  # block
        if block is None:
            raise ValueError("unit='block' requires a block Series (record_id -> block label)")
        blk = _aligned_series(block, truth)
        if blk.isna().any():
            raise ValueError("block contains missing block labels")
        unit_codes = pd.factorize(blk.to_numpy())[0]
    n_units = int(unit_codes.max()) + 1
    if n_units < 2:
        raise ValueError(f"need >= 2 resampling units, got {n_units} (unit={unit!r})")
    stats = [_make_weight_stat(metric, p, truth, unit_codes) for p in aligned]
    return stats, n_units


def _bca_interval(
    boots: np.ndarray, point: float, jack: np.ndarray, alpha: float
) -> tuple[float, float]:
    """Efron's BCa interval from bootstrap replicates and jackknife values.

    z0 from the (tie-split, clipped) fraction of replicates below the point
    estimate; acceleration a from the jackknife third-moment formula. With
    z0 = 0 and a = 0 this is exactly the percentile interval.
    """
    n = len(boots)
    prop = ((boots < point).sum() + 0.5 * (boots == point).sum()) / n
    prop = min(max(prop, 1.0 / (n + 1)), n / (n + 1.0))
    z0 = float(ndtri(prop))
    d = jack.mean() - jack
    # scale-aware zero test: a numerically-constant jackknife leaves rounding
    # residue ~eps*|jack| in d, and the a-ratio would amplify that noise to
    # O(1); genuine accelerations come from spreads many orders larger
    tol = 1e-12 * max(1.0, float(np.abs(jack).max()))
    if float(np.abs(d).max()) < tol:
        a = 0.0
    else:
        denom = float((d**2).sum()) ** 1.5
        a = float((d**3).sum()) / (6.0 * denom)

    def level(z_alpha: float) -> float:
        z = z0 + z_alpha
        d_ = 1.0 - a * z
        if d_ <= 0:  # acceleration blow-up: saturate at the distribution edge
            return 1.0 if z > 0 else 0.0
        return float(ndtr(z0 + z / d_))

    lo = float(np.quantile(boots, level(float(ndtri(alpha / 2.0)))))
    hi = float(np.quantile(boots, level(float(ndtri(1.0 - alpha / 2.0)))))
    return lo, hi


def _boot(
    stat: Callable[[np.ndarray], float],
    n_units: int,
    *,
    n_boot: int,
    seed: int,
    method: str,
    alpha: float,
) -> dict:
    if method not in ("bca", "percentile"):
        raise ValueError(f"unknown method {method!r}; expected 'bca' or 'percentile'")
    if n_boot < 2:
        raise ValueError("n_boot must be >= 2")
    if not (0 < alpha < 1):
        raise ValueError("alpha must be in (0, 1)")
    rng = np.random.default_rng(seed)
    ones = np.ones(n_units, dtype=np.int64)
    point = stat(ones)
    boots = np.empty(n_boot)
    for b in range(n_boot):
        counts = np.bincount(rng.integers(0, n_units, n_units), minlength=n_units)
        boots[b] = stat(counts)
    if boots.min() == boots.max():  # degenerate: no resampling variation
        lo = hi = float(boots[0])
    elif method == "percentile":
        lo, hi = (float(q) for q in np.quantile(boots, [alpha / 2.0, 1.0 - alpha / 2.0]))
    else:
        jack = np.empty(n_units)
        for i in range(n_units):
            counts = ones.copy()
            counts[i] = 0
            jack[i] = stat(counts)
        lo, hi = _bca_interval(boots, point, jack, alpha)
    return {
        "point": float(point),
        "ci_low": lo,
        "ci_high": hi,
        "method": method,
        "n_boot": int(n_boot),
        "boots": boots,
        "n_units": int(n_units),
        "alpha": float(alpha),
    }


# -------------------------------------------------------------------- api


def bootstrap_ci(
    pred: pd.Series,
    truth: pd.Series,
    metric: Callable,
    *,
    unit: str = "entity",
    block: pd.Series | None = None,
    n_boot: int = 1000,
    seed: int,
    method: str = "bca",
    alpha: float = 0.05,
) -> dict:
    """Bootstrap CI for a scalar ER metric under a chosen resampling unit.

    ``metric`` is called as ``metric(pred, truth)`` and must return a scalar;
    if it exposes a ``weights`` keyword it receives the resampled multiset as
    per-record frequency weights (no data copied), otherwise records are
    physically duplicated with suffixed ids (documented O(Σw) cost per
    replicate). See the module docstring for unit semantics and the BCa
    definition. ``seed`` is required: every CI in this lab is reproducible.

    Returns {point, ci_low, ci_high, method, n_boot, boots} (plus n_units and
    alpha); ``boots`` is the raw replicate array so notebooks can plot the
    resampling distribution instead of trusting two numbers. Default is a
    95% interval (alpha=0.05). BCa cost: n_boot + n_units + 1 metric calls.
    """
    stats, n_units = _make_stats([pred], truth, metric, unit, block)
    return _boot(stats[0], n_units, n_boot=n_boot, seed=seed, method=method, alpha=alpha)


def paired_delta(
    pred_a: pd.Series,
    pred_b: pd.Series,
    truth: pd.Series,
    metric: Callable,
    *,
    unit: str = "entity",
    block: pd.Series | None = None,
    n_boot: int = 1000,
    seed: int,
    method: str = "bca",
    alpha: float = 0.05,
) -> dict:
    """CI for metric(A) − metric(B) using SHARED resample draws (PLAN §5).

    Both systems are evaluated on the *same* unit draws in every replicate,
    so entity-sampling noise common to both cancels from the delta — the
    paired-comparison design that makes small differences detectable. For
    unit='pair' the pair universe is the union of pairs co-clustered in
    pred_a, pred_b, or truth, so both systems see the same pair sample.

    Returns {delta, ci_low, ci_high, sign_stable} plus point/boots/method
    extras. ``sign_stable`` is True iff the CI excludes zero; two identical
    systems give delta 0 with the degenerate CI [0, 0] (sign_stable False).
    """
    stats, n_units = _make_stats([pred_a, pred_b], truth, metric, unit, block)
    stat_a, stat_b = stats

    def delta_stat(counts: np.ndarray) -> float:
        return stat_a(counts) - stat_b(counts)

    res = _boot(delta_stat, n_units, n_boot=n_boot, seed=seed, method=method, alpha=alpha)
    return {
        "delta": res["point"],
        "ci_low": res["ci_low"],
        "ci_high": res["ci_high"],
        "sign_stable": bool(res["ci_low"] > 0 or res["ci_high"] < 0),
        "method": res["method"],
        "n_boot": res["n_boot"],
        "boots": res["boots"],
        "n_units": res["n_units"],
        "alpha": res["alpha"],
    }
