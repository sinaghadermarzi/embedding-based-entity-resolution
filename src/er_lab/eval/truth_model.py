"""Truth-error modeling for pseudo-truth keys (PLAN.md §3 MET-05, §5; lit_review §0(iv), §4, §8).

Administrative keys (NCID-style) serve as pseudo-truth but carry their own errors,
in two directions:

- **duplication** (split): one real person holds several keys, so records of the
  same person carry different entity ids — a *correct* predicted link is then
  scored as a false positive;
- **overlay** (merge): several real people share one key, so records of different
  persons carry the same entity id — a *wrong* predicted link is then scored as a
  true positive, and the truth demands links that should not exist.

``corrected_precision_recall`` inverts this contamination to first order, with
conservative bands. ``disagreement_sample`` draws the stratified adjudication
sample (input to the labeling widget) whose design weights let adjudicated rates
be reweighted back to the full disagreement population.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np
import pandas as pd

__all__ = ["corrected_precision_recall", "disagreement_sample"]

#: Sentinel standing in for "this frame does not label this pair" (never equal to
#: a real label, so one-sided pairs always count as disagreements).
_ABSENT = "__absent__"

#: Sentinel a missing stratum value maps to for GROUPING only (pandas group
#: enumeration with dropna=False raises on null single-column categories, and
#: null keys must not silently vanish); the output keeps the original <NA>.
_NA_STRATUM = "__stratum_na__"


def _clip01(x: float) -> float:
    return float(min(1.0, max(0.0, x)))


def _check_rate(name: str, value: float) -> float:
    value = float(value)
    if not math.isfinite(value) or not 0.0 <= value < 1.0:
        raise ValueError(f"{name} must be a rate in [0, 1), got {value!r}")
    return value


def _check_obs(name: str, value: float) -> float:
    value = float(value)
    if math.isnan(value):
        return value  # NaN passes through (pairwise conventions allow NaN metrics)
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"{name} must be in [0, 1] or NaN, got {value!r}")
    return value


def _check_ci(name: str, ci: tuple[float, float] | None, point: float) -> tuple[float, float]:
    if ci is None:
        return (point, point)
    lo, hi = (_check_rate(f"{name}[0]", ci[0]), _check_rate(f"{name}[1]", ci[1]))
    if lo > hi:
        raise ValueError(f"{name} must satisfy low <= high, got {ci!r}")
    if not lo <= point <= hi:
        # bands are evaluated at the CI corners only, so a point rate outside
        # its own CI would yield a "conservative" band excluding the point
        # estimate — self-contradictory output; refuse the inconsistent input
        raise ValueError(f"{name}={ci!r} must contain the point rate {point!r}")
    return (lo, hi)


def corrected_precision_recall(
    obs_precision: float,
    obs_recall: float,
    *,
    truth_dup_rate: float,
    truth_overlay_rate: float,
    dup_rate_ci: tuple[float, float] | None = None,
    overlay_rate_ci: tuple[float, float] | None = None,
) -> dict[str, float]:
    """Correct observed pairwise precision/recall for truth-key duplication and overlay.

    Model (pair-level, first-order; all rates in [0, 1)). Let ``d = truth_dup_rate``
    be the probability that a truly co-referent record pair is keyed apart (one
    person, two truth keys) and ``o = truth_overlay_rate`` the probability that a
    truly non-co-referent pair in play is keyed together (two persons, one key).
    Among the system's predicted links, ``T`` are truly co-referent and ``W`` are
    not. Scored against the erroneous truth::

        TP_obs = T (1 - d) + W o      # true links keyed apart are scored FP,
        FP_obs = T d + W (1 - o)      # wrong links keyed together are scored TP

    so with true precision ``P = T / (T + W)``::

        P_obs = P (1 - d) + (1 - P) o    =>    P = (P_obs - o) / (1 - d - o)

    At ``d = o = 0`` this is the identity. Given ``d`` and ``o`` the inversion is
    exact, so the precision band reflects only the rate CIs (evaluated at the
    corners of the ``(d, o)`` rectangle; the map is coordinate-wise monotone, so
    corners attain the extremes).

    Recall: the observed denominator is the set of same-key record pairs, of which
    a fraction ``o`` are spurious (overlaid different-person pairs), while a
    fraction ``d`` of real pairs are missing entirely (keyed apart — those the
    system found were scored FP above). With ``R_gen`` the recall on genuine
    same-key pairs and ``s`` the link rate on spurious pairs::

        R_obs = (1 - o) R_gen + o s

    Point estimate: ``s = 0`` (a system has no reason to link records of different
    persons merely because their truth keys collide) and dup-hidden real pairs
    assumed as recallable as visible ones (non-informative missingness), giving
    ``recall = min(1, R_obs / (1 - o))``. The conservative band lets ``s`` range
    over [0, 1] and the recall on dup-hidden pairs over [0, 1]::

        low  = (1 - d) * max(0, (R_obs - o) / (1 - o))
        high = (1 - d) * min(1, R_obs / (1 - o)) + d

    again evaluated over the ``(d, o)`` CI corners when CIs are given, so bands
    widen with CI inputs.

    Assumptions (stated, not hidden): (1) ``d`` and ``o`` are error rates on the
    populations the algebra conditions on — ``d`` among truly co-referent pairs,
    ``o`` among the confusable pairs a system wrongly links and among same-key
    pairs; MET-05's audit and the adjudicated disagreement sample estimate exactly
    these conditional rates. (2) First-order: a pair suffers at most one truth
    error. (3) Recall's point estimate adds ``s = 0`` and non-informative
    dup-missingness; the band drops both. Point estimates are clipped to [0, 1].

    Returns dict with keys ``precision``, ``recall``, ``precision_low``,
    ``precision_high``, ``recall_low``, ``recall_high``. A NaN observed input
    yields NaN outputs for that metric.
    """
    p_obs = _check_obs("obs_precision", obs_precision)
    r_obs = _check_obs("obs_recall", obs_recall)
    d = _check_rate("truth_dup_rate", truth_dup_rate)
    o = _check_rate("truth_overlay_rate", truth_overlay_rate)
    d_lo, d_hi = _check_ci("dup_rate_ci", dup_rate_ci, d)
    o_lo, o_hi = _check_ci("overlay_rate_ci", overlay_rate_ci, o)
    if d + o >= 1.0 or d_hi + o_hi >= 1.0:
        raise ValueError(
            "truth_dup_rate + truth_overlay_rate must stay below 1 (including CI highs); "
            f"got d={d} (ci high {d_hi}), o={o} (ci high {o_hi})"
        )
    corners = [(dc, oc) for dc in (d_lo, d_hi) for oc in (o_lo, o_hi)]

    def p_true(dc: float, oc: float) -> float:
        return (p_obs - oc) / (1.0 - dc - oc)

    def r_band(dc: float, oc: float) -> tuple[float, float]:
        m_hi = min(1.0, r_obs / (1.0 - oc))
        m_lo = _clip01((r_obs - oc) / (1.0 - oc))
        return (1.0 - dc) * m_lo, (1.0 - dc) * m_hi + dc

    if math.isnan(p_obs):
        prec = prec_lo = prec_hi = float("nan")
    else:
        prec = _clip01(p_true(d, o))
        vals = [_clip01(p_true(dc, oc)) for dc, oc in corners]
        prec_lo, prec_hi = min(vals), max(vals)
    if math.isnan(r_obs):
        rec = rec_lo = rec_hi = float("nan")
    else:
        rec = _clip01(r_obs / (1.0 - o))
        bands = [r_band(dc, oc) for dc, oc in corners]
        rec_lo = _clip01(min(b[0] for b in bands))
        rec_hi = _clip01(max(b[1] for b in bands))
    return {
        "precision": prec,
        "recall": rec,
        "precision_low": prec_lo,
        "precision_high": prec_hi,
        "recall_low": rec_lo,
        "recall_high": rec_hi,
    }


def _normalize_pairs(pairs: pd.DataFrame, side: str, strata: list[str]) -> pd.DataFrame:
    required = {"a", "b", "label"}
    missing = sorted(required - set(pairs.columns))
    if missing:
        raise KeyError(f"pairs_{side} is missing columns {missing}")
    keep = ["a", "b", "label"] + [c for c in strata if c in pairs.columns]
    out = pairs[keep].copy()
    a, b = out["a"].astype(str), out["b"].astype(str)
    out["a"] = a.where(a <= b, b)  # unordered pair key: (min, max) as strings
    out["b"] = b.where(a <= b, a)
    if out.duplicated(["a", "b"]).any():
        raise ValueError(f"pairs_{side} contains duplicate unordered pairs")
    out["label"] = out["label"].astype(object)
    return out


def _allocate(counts: np.ndarray, n: int) -> np.ndarray:
    """Per-stratum sample sizes: min-1 guarantee, then proportional largest-remainder."""
    counts = np.asarray(counts, dtype=int)
    h = len(counts)
    n = min(int(n), int(counts.sum()))
    alloc = np.zeros(h, dtype=int)
    if n < h:  # fewer draws than strata: one draw each to the n largest strata
        alloc[np.argsort(-counts, kind="stable")[:n]] = 1
        return alloc
    alloc[:] = 1
    remaining = n - h
    capacity = counts - 1
    if remaining > 0:
        quota = remaining * capacity / capacity.sum()
        add = np.floor(quota).astype(int)  # floor(quota) <= capacity since remaining <= sum(cap)
        left = int(remaining - add.sum())
        order = np.argsort(-(quota - add), kind="stable")
        while left > 0:
            progressed = False
            for i in order:
                if left == 0:
                    break
                if alloc[i] + add[i] < counts[i]:
                    add[i] += 1
                    left -= 1
                    progressed = True
            if not progressed:  # pragma: no cover - unreachable: n <= total
                break
        alloc = alloc + add
    return alloc


def disagreement_sample(
    pairs_a: pd.DataFrame,
    pairs_b: pd.DataFrame,
    *,
    strata: Sequence[str] | None = None,
    n: int,
    seed: int,
) -> pd.DataFrame:
    """Stratified sample of pairs where two labelings disagree, with design weights.

    ``pairs_a`` / ``pairs_b`` are DataFrames with columns ``a``, ``b``, ``label``
    (record-id endpoints plus that source's decision — e.g. a system's links vs
    the truth key's links). Pairs are treated as unordered. A pair present in only
    one frame counts as a disagreement (its missing side is reported as ``<NA>``);
    where both frames label a pair, disagreement means ``label`` values differ
    (a missing/NaN label also counts as absent). Duplicate unordered pairs within
    one frame raise.

    ``strata`` names columns to stratify on, taken from either frame (values from
    ``pairs_a`` win where both carry the column); rows without a stratum value
    form their own ``<NA>`` stratum (reported as missing in the output — e.g.
    pairs present only in a frame that lacks the stratum column). Stratum
    identity is decided on the string representation of the values. Allocation over strata: every nonempty
    stratum gets at least one draw when ``n`` allows (``n >= #strata``), the rest
    is proportional to stratum size with largest-remainder rounding, capped at
    stratum size. Sampling is without replacement, deterministic under ``seed``.

    Each sampled row carries ``weight = N_h / n_h`` (stratum disagreement count
    over stratum sample size) — the inverse inclusion probability, so
    design-based (Horvitz-Thompson) totals over the sample estimate totals over
    the full disagreement population; when every stratum is represented the
    weights sum exactly to the number of disagreements.

    Returns columns ``a``, ``b``, ``label_a``, ``label_b``, ``*strata``,
    ``weight``.
    """
    strata = list(strata or [])
    if n <= 0:
        raise ValueError(f"n must be positive, got {n}")
    fa = _normalize_pairs(pairs_a, "a", strata)
    fb = _normalize_pairs(pairs_b, "b", strata)
    merged = fa.merge(fb, on=["a", "b"], how="outer", suffixes=("_a", "_b"))
    for col in strata:
        if f"{col}_a" in merged.columns:  # both frames carried it: coalesce, a wins
            merged[col] = merged[f"{col}_a"].combine_first(merged[f"{col}_b"])
            merged = merged.drop(columns=[f"{col}_a", f"{col}_b"])
        elif col not in merged.columns:
            raise KeyError(f"stratum column {col!r} not found in either pairs frame")

    la = merged["label_a"].astype(object).where(merged["label_a"].notna(), _ABSENT)
    lb = merged["label_b"].astype(object).where(merged["label_b"].notna(), _ABSENT)
    population = merged.loc[(la != lb).to_numpy()]
    out_columns = ["a", "b", "label_a", "label_b", *strata, "weight"]
    if population.empty:
        return pd.DataFrame(columns=out_columns)

    if strata:
        # Group on all-string shadow keys: pandas raises on a single-column
        # grouper with null categories ("Categorical categories cannot be
        # null"), and mixed-type sort would be fragile. Missing values map to
        # the _NA_STRATUM sentinel — their own stratum, per the docstring —
        # while the original columns (real <NA> included) flow to the output.
        population = population.copy()
        gcols = []
        for k, col in enumerate(strata):
            s = population[col]
            gcol = f"__group_{k}"
            population[gcol] = np.where(s.isna(), _NA_STRATUM, s.astype(str))
            gcols.append(gcol)
        groups = list(population.groupby(gcols, sort=True))
    else:
        groups = [((), population)]
    alloc = _allocate(np.array([len(g) for _, g in groups]), n)

    rng = np.random.default_rng(seed)
    taken: list[pd.DataFrame] = []
    for (_, grp), k in zip(groups, alloc):
        if k == 0:
            continue
        pos = np.sort(rng.choice(len(grp), size=int(k), replace=False))
        part = grp.iloc[pos].copy()
        part["weight"] = len(grp) / float(k)
        taken.append(part)
    out = pd.concat(taken, axis=0).reset_index(drop=True)
    return out[out_columns]
