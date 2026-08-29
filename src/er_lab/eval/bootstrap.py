"""Resampling-unit-aware bootstrap CIs for ER metrics (PLAN.md §5, MET-03).

ER records are not i.i.d.: records of one entity share their fate, so the
choice of resampling unit decides whether a bootstrap CI has nominal coverage
— the question MET-03 settles empirically with this module. Units:

- ``'entity'`` (the lab's expected default): resample TRUTH entities with
  replacement.
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

Two estimators for entity/record/block units — the result dict's ``path``
key says which one ran:

**Multiplier (frozen-contribution) bootstrap** — ``path='multiplier'``, the
correct estimator, used whenever ``metric`` is one of the names in
:data:`MULTIPLIER_METRICS` (``'bcubed_precision'``, ``'bcubed_recall'``,
``'bcubed_f1'``, ``'pairwise_precision'``, ``'pairwise_recall'``,
``'pairwise_f1'``). Per-unit contributions are frozen ONCE on the observed
frame and each replicate only reweights them by the unit draw counts ``c``:

- B-cubed: per-record precision/recall contributions ``prec_r``, ``rec_r``
  are computed on the observed frame; a replicate's precision/recall are the
  ``c``-weighted means (record r carries weight ``c[unit(r)]``), and F1 is
  their harmonic mean.
- pairwise: the relevant pairs of the observed frame carry frozen tp/fp/fn
  indicators; a replicate weights pair (r, s) by ``c[unit(r)] * c[unit(s)]``
  when the two records belong to different units and by ``c[unit(r)]`` when
  they share a unit (the pair rides along with its unit's draws — copies of
  a unit are distinct units and never cross-pair).

With all-ones counts both reduce exactly to the observed metric.

**Duplication fallback** — ``path='duplication'``, used when ``metric`` is a
callable (required for metrics with no per-unit decomposition: VI, GMD,
cluster F). Each replicate recomputes ``metric`` on the merged-duplication
multiset: a record drawn w times becomes ONE record of mass w (via the
metric's ``weights`` keyword when it has one, else physical duplication with
suffixed ids — the two give identical numbers).

.. warning::
   The duplication fallback is BIASED for co-membership metrics: a record of
   mass w pairs with its own copies, so W(W−1)/2 pair semantics manufactures
   always-TP self-pairs and inflates B-cubed masses asymmetrically. Because
   relevant pairs are sparse (~n, not n²), the shift is O(1) and does NOT
   vanish with n — measured at roughly +0.2 on pairwise precision and +0.04
   on B-cubed F1 under mixed split+merge errors, which destroys coverage
   (nominal-95% intervals covering <20% of the time at ~100 entities). Use
   the string metrics (multiplier path) for every headline CI; reserve the
   fallback for genuinely non-decomposable metrics and read its intervals as
   biased upward whenever pred merges records across resampling units.

NaN replicates (expected for unit='pair', where a draw with no
predicted-positive pairs yields NaN precision under the documented 0/0
convention) are dropped before quantiles or BCa; the result dict reports
``n_boot_effective`` (finite replicates) and ``n_nan``. A NaN point estimate,
or a NaN fraction above 20%, raises instead of returning a misleading CI.

Intervals: ``method='bca'`` implements Efron's bias-corrected-and-accelerated
interval (Efron 1987; Efron & Tibshirani 1993 ch. 14): bias correction z0
from the fraction of bootstrap replicates below the point estimate (ties
split evenly, proportion clipped to [1/(B+1), B/(B+1)]), acceleration a from
the jackknife over the *resampling unit* (leave-one-unit-out), a =
Σd³ / (6 (Σd²)^{3/2}) with d = mean(jack) − jack. When the bootstrap
distribution is symmetric about the point estimate and a ≈ 0, BCa reduces to
the percentile interval (tested). Degenerate case: if all finite replicates
are equal the CI collapses to that value under either method.
"""

from __future__ import annotations

import inspect
import itertools
import math
from collections.abc import Callable

import numpy as np
import pandas as pd
from scipy.special import ndtr, ndtri

from er_lab.eval.metrics import bcubed, pairwise

__all__ = ["MULTIPLIER_METRICS", "bootstrap_ci", "paired_delta"]

_UNITS = ("entity", "record", "pair", "block")

#: Metric names with an exact per-unit decomposition — these get the
#: multiplier (frozen-contribution) bootstrap for entity/record/block units.
MULTIPLIER_METRICS = (
    "bcubed_precision",
    "bcubed_recall",
    "bcubed_f1",
    "pairwise_precision",
    "pairwise_recall",
    "pairwise_f1",
)

#: For unit='pair' a string metric is evaluated on the rebuilt pair universe.
_PAIR_UNIT_WRAPPERS: dict[str, Callable] = {
    "bcubed_precision": lambda p, t: bcubed(p, t)["precision"],
    "bcubed_recall": lambda p, t: bcubed(p, t)["recall"],
    "bcubed_f1": lambda p, t: bcubed(p, t)["f1"],
    "pairwise_precision": lambda p, t: pairwise(p, t)["precision"],
    "pairwise_recall": lambda p, t: pairwise(p, t)["recall"],
    "pairwise_f1": lambda p, t: pairwise(p, t)["f1"],
}


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
    """A metric opts into the no-copy fallback by exposing a 'weights' parameter."""
    try:
        sig = inspect.signature(metric)
    except (TypeError, ValueError):  # builtins / odd callables
        return False
    return "weights" in sig.parameters


def _scalar(value: object) -> float:
    if isinstance(value, dict):
        raise TypeError(
            "metric must return a scalar; wrap dict-returning metrics, e.g. "
            "lambda p, t: bcubed(p, t)['f1'] — or pass the name 'bcubed_f1' "
            "to get the unbiased multiplier path"
        )
    return float(value)  # type: ignore[arg-type]


def _harmonic(p: float, r: float) -> float:
    """F1 from precision and recall under the metrics-module NaN conventions."""
    if math.isnan(p) or math.isnan(r):
        return float("nan")
    if p + r == 0:
        return 0.0
    return 2.0 * p * r / (p + r)


def _duplicated(pred: pd.Series, truth: pd.Series, w: np.ndarray) -> tuple[pd.Series, pd.Series]:
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
    """Duplication-fallback statistic: recompute the metric on the merged multiset.

    Biased for co-membership metrics (see the module docstring warning); the
    ``weights`` fast path and the suffixed-id duplication produce identical
    numbers, so both are ``path='duplication'``.
    """
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


def _bcubed_contributions(
    p_codes: np.ndarray, t_codes: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Frozen per-record B-cubed contributions on the observed frame."""
    nt = int(t_codes.max()) + 1
    cell = p_codes.astype(np.int64) * nt + t_codes
    _, inv, cell_counts = np.unique(cell, return_inverse=True, return_counts=True)
    n_pred = np.bincount(p_codes).astype(float)
    n_truth = np.bincount(t_codes).astype(float)
    overlap = cell_counts[inv].astype(float)  # |C_p(r) ∩ C_t(r)| per record
    return overlap / n_pred[p_codes], overlap / n_truth[t_codes]


def _make_multiplier_stat(
    name: str, pred: pd.Series, truth: pd.Series, unit_codes: np.ndarray
) -> Callable[[np.ndarray], float]:
    """Multiplier statistic: reweight frozen per-unit contributions by draw counts."""
    p_codes = pd.factorize(pred.to_numpy())[0]
    t_codes = pd.factorize(truth.to_numpy())[0]
    family, _, field = name.partition("_")

    if family == "bcubed":
        prec_r, rec_r = _bcubed_contributions(p_codes, t_codes)

        def stat(counts: np.ndarray) -> float:
            w = counts[unit_codes].astype(float)
            total = w.sum()
            if total == 0:
                return float("nan")
            p = float(w @ prec_r) / total
            r = float(w @ rec_r) / total
            if field == "precision":
                return p
            if field == "recall":
                return r
            return _harmonic(p, r)

        return stat

    # pairwise: frozen tp/fp/fn indicators over the observed relevant pairs
    n = len(t_codes)
    pred_keys = _co_pair_keys(p_codes)
    truth_keys = _co_pair_keys(t_codes)
    all_keys = np.union1d(pred_keys, truth_keys)
    co_pred = np.isin(all_keys, pred_keys)
    co_truth = np.isin(all_keys, truth_keys)
    u_r = unit_codes[(all_keys // n).astype(np.int64)]
    u_s = unit_codes[(all_keys % n).astype(np.int64)]
    same_unit = u_r == u_s
    tp_ind = (co_pred & co_truth).astype(float)
    fp_ind = (co_pred & ~co_truth).astype(float)
    fn_ind = (~co_pred & co_truth).astype(float)

    def stat(counts: np.ndarray) -> float:
        c = counts.astype(float)
        w = np.where(same_unit, c[u_r], c[u_r] * c[u_s])
        tp = float(w @ tp_ind)
        fp = float(w @ fp_ind)
        fn = float(w @ fn_ind)
        p = tp / (tp + fp) if tp + fp > 0 else float("nan")
        r = tp / (tp + fn) if tp + fn > 0 else float("nan")
        if field == "precision":
            return p
        if field == "recall":
            return r
        return _harmonic(p, r)

    return stat


def _make_stats(
    preds: list[pd.Series],
    truth: pd.Series,
    metric: Callable | str,
    unit: str,
    block: pd.Series | None,
) -> tuple[list[Callable[[np.ndarray], float]], int, str]:
    """Build per-system statistic functions sharing one resampling structure.

    Returns (stats, n_units, path) with path in {'multiplier', 'duplication',
    'pair'} — see the module docstring for the estimator each path implements.
    """
    if unit not in _UNITS:
        raise ValueError(f"unknown unit {unit!r}; expected one of {_UNITS}")
    if isinstance(metric, str) and metric not in MULTIPLIER_METRICS:
        raise ValueError(
            f"unknown metric name {metric!r}; expected one of {MULTIPLIER_METRICS} "
            "(or pass a callable for the duplication fallback)"
        )
    aligned = [_aligned_series(p, truth) for p in preds]
    if truth.isna().any():
        raise ValueError("truth contains missing entity ids")

    if unit == "pair":
        metric_fn = _PAIR_UNIT_WRAPPERS[metric] if isinstance(metric, str) else metric
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
            _make_pair_stat(metric_fn, np.isin(all_keys, keys), co_truth) for keys in pred_keys
        ]
        return stats, len(all_keys), "pair"

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
    if isinstance(metric, str):
        stats = [_make_multiplier_stat(metric, p, truth, unit_codes) for p in aligned]
        return stats, n_units, "multiplier"
    stats = [_make_weight_stat(metric, p, truth, unit_codes) for p in aligned]
    return stats, n_units, "duplication"


def _bca_interval(
    boots: np.ndarray, point: float, jack: np.ndarray, alpha: float
) -> tuple[float, float]:
    """Efron's BCa interval from finite bootstrap replicates and jackknife values.

    z0 from the (tie-split, clipped) fraction of replicates below the point
    estimate; acceleration a from the jackknife third-moment formula (NaN
    jackknife values are dropped; with fewer than two finite values a = 0).
    With z0 = 0 and a = 0 this is exactly the percentile interval.
    """
    n = len(boots)
    prop = ((boots < point).sum() + 0.5 * (boots == point).sum()) / n
    prop = min(max(prop, 1.0 / (n + 1)), n / (n + 1.0))
    z0 = float(ndtri(prop))
    jack = jack[np.isfinite(jack)]
    if len(jack) < 2:
        a = 0.0
    else:
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


_MAX_NAN_FRACTION = 0.2


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
    if math.isnan(point):
        raise ValueError(
            "the point estimate is NaN: the metric is undefined on the observed data "
            "(e.g. pairwise precision with no predicted co-clustered pairs under the "
            "documented 0/0 convention) — a bootstrap CI would be meaningless"
        )
    boots = np.empty(n_boot)
    for b in range(n_boot):
        counts = np.bincount(rng.integers(0, n_units, n_units), minlength=n_units)
        boots[b] = stat(counts)
    nan_mask = np.isnan(boots)
    n_nan = int(nan_mask.sum())
    if n_nan > _MAX_NAN_FRACTION * n_boot:
        raise ValueError(
            f"{n_nan}/{n_boot} bootstrap replicates are NaN (> {_MAX_NAN_FRACTION:.0%}): "
            "the metric is undefined on too many resamples (e.g. draws with no "
            "predicted-positive pairs under the documented 0/0 convention), so any "
            "interval from the remainder would be unreliable — use a larger evaluation "
            "set, a different resampling unit, or a metric defined on every resample"
        )
    finite = boots[~nan_mask]
    if finite.min() == finite.max():  # degenerate: no resampling variation
        lo = hi = float(finite[0])
    elif method == "percentile":
        lo, hi = (float(q) for q in np.quantile(finite, [alpha / 2.0, 1.0 - alpha / 2.0]))
    else:
        jack = np.empty(n_units)
        for i in range(n_units):
            counts = ones.copy()
            counts[i] = 0
            jack[i] = stat(counts)
        lo, hi = _bca_interval(finite, point, jack, alpha)
    return {
        "point": float(point),
        "ci_low": lo,
        "ci_high": hi,
        "method": method,
        "n_boot": int(n_boot),
        "n_boot_effective": len(finite),
        "n_nan": n_nan,
        "boots": boots,
        "n_units": int(n_units),
        "alpha": float(alpha),
    }


# -------------------------------------------------------------------- api


def bootstrap_ci(
    pred: pd.Series,
    truth: pd.Series,
    metric: Callable | str,
    *,
    unit: str = "entity",
    block: pd.Series | None = None,
    n_boot: int = 1000,
    seed: int,
    method: str = "bca",
    alpha: float = 0.05,
) -> dict:
    """Bootstrap CI for a scalar ER metric under a chosen resampling unit.

    ``metric`` is either a name from :data:`MULTIPLIER_METRICS` (recommended
    for every headline CI — for entity/record/block units it selects the
    unbiased multiplier bootstrap over frozen per-unit contributions) or a
    callable ``metric(pred, truth) -> float``, which for those units falls
    back to recomputing the metric on the merged-duplication multiset — a
    path that is BIASED UPWARD for co-membership metrics whenever pred merges
    records across resampling units (see the module docstring warning; the
    ``weights``-keyword fast path computes those same duplication numbers
    without copying). The result's ``path`` key reports which estimator ran:
    ``'multiplier'``, ``'duplication'``, or ``'pair'``. ``seed`` is required:
    every CI in this lab is reproducible.

    NaN replicates are dropped before the interval is formed
    (``n_boot_effective``/``n_nan`` report the split); a NaN point estimate
    or more than 20% NaN replicates raises a ValueError instead of returning
    a misleading interval.

    Returns {point, ci_low, ci_high, method, n_boot, n_boot_effective, n_nan,
    boots, n_units, alpha, path}; ``boots`` is the raw replicate array
    (including any NaNs) so notebooks can plot the resampling distribution
    instead of trusting two numbers. Default is a 95% interval (alpha=0.05).
    BCa cost: n_boot + n_units + 1 statistic evaluations.
    """
    stats, n_units, path = _make_stats([pred], truth, metric, unit, block)
    res = _boot(stats[0], n_units, n_boot=n_boot, seed=seed, method=method, alpha=alpha)
    res["path"] = path
    return res


def paired_delta(
    pred_a: pd.Series,
    pred_b: pd.Series,
    truth: pd.Series,
    metric: Callable | str,
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
    paired-comparison design that makes small differences detectable.
    ``metric`` follows the :func:`bootstrap_ci` contract: a
    :data:`MULTIPLIER_METRICS` name gets the unbiased multiplier path for
    both systems (recommended); a callable gets the biased duplication
    fallback (the shared draws cancel most, but not all, of that bias — the
    residual delta shift can flip ``sign_stable``). For unit='pair' the pair
    universe is the union of pairs co-clustered in pred_a, pred_b, or truth,
    so both systems see the same pair sample. A replicate where either
    system's metric is NaN is a NaN delta replicate, handled per the
    :func:`bootstrap_ci` NaN policy.

    Returns {delta, ci_low, ci_high, sign_stable} plus
    point/boots/method/path extras. ``sign_stable`` is True iff the CI
    excludes zero; two identical systems give delta 0 with the degenerate CI
    [0, 0] (sign_stable False).
    """
    stats, n_units, path = _make_stats([pred_a, pred_b], truth, metric, unit, block)
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
        "n_boot_effective": res["n_boot_effective"],
        "n_nan": res["n_nan"],
        "boots": res["boots"],
        "n_units": res["n_units"],
        "alpha": res["alpha"],
        "path": path,
    }
