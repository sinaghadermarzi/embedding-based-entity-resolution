"""Scaling ladders and error-vs-n curve fits with honest extrapolation (SCL-01, PLAN §3).

The SCL-01 question is whether error-vs-n fits made at <=1e6 records predict a
held-out 1e7 run *within the fit's own prediction interval* — extrapolation is
only quoted when :func:`validate_extrapolation` says the fit earned it.

Why the ladder is ENTITY-preserving (:func:`entity_ladder`): sampling *records*
i.i.d. thins every entity — an entity with five records kept at 10% retains
~0.5 of them — so the within-entity pair structure (the thing ER measures)
shrinks with the sample, and the duplicate rate per record falls as n falls.
An error-vs-n curve over record subsamples therefore confounds "more records"
with "denser duplication", and its extrapolation predicts a corpus that never
exists at scale. Sampling *entities* and keeping all their records holds the
duplication structure fixed along the ladder, so n is the only thing moving.
The ladder is also NESTED (each rung is a strict superset of the smaller
rungs, built from one seed-fixed entity permutation): rung-to-rung deltas are
then paired comparisons on shared entities, not independent redraws.

Fits (:func:`fit_scaling`) are plain 2-parameter OLS in the form's natural
space with Student-t prediction intervals — deliberately boring: SCL-01 tests
whether such a fit transfers, so the fit itself must have no tunable slack.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

__all__ = ["entity_ladder", "fit_scaling", "validate_extrapolation"]

FORMS = ("powerlaw", "loglinear")


def entity_ladder(
    df: pd.DataFrame, *, sizes: list[int], seed: int
) -> dict[int, pd.DataFrame]:
    """Nested, entity-preserving subsamples of a canonical frame at ~each target size.

    Entities (unique ``entity_id`` values) are shuffled once under *seed*; each
    rung takes the shortest prefix of that permutation whose records total at
    least the target size, keeping EVERY record of each included entity (see
    module docstring for why records are never split). Consequences the tests
    pin down: no entity straddles an inclusion boundary; the actual size
    overshoots the target by less than the last-added entity's record count;
    smaller rungs are subsets of larger ones (shared prefix).

    Returns ``{target_size: sub-frame}`` with rows in original *df* order.
    """
    if "entity_id" not in df.columns:
        raise KeyError("entity_ladder needs an 'entity_id' column (canonical frame)")
    ent = df["entity_id"]
    if ent.isna().any():
        raise ValueError(
            "entity_id contains NA — the ladder needs truth keys for every record "
            "(sources without a truth key cannot ride the scaling ladder)"
        )
    ent = ent.astype(str)
    if not sizes:
        raise ValueError("sizes is empty")
    if any((not float(s).is_integer()) or s < 1 for s in sizes):
        raise ValueError(f"sizes must be positive integers, got {sizes}")
    if max(sizes) > len(df):
        raise ValueError(f"target size {max(sizes)} exceeds the {len(df)} records available")

    ids = sorted(ent.unique())  # sort first: determinism independent of row order
    rng = np.random.default_rng(seed)
    order = [ids[i] for i in rng.permutation(len(ids))]
    counts = ent.value_counts()
    cum = np.cumsum([int(counts[e]) for e in order])

    out: dict[int, pd.DataFrame] = {}
    for size in sorted({int(s) for s in sizes}):
        stop = int(np.searchsorted(cum, size, side="left"))  # first prefix reaching size
        keep = set(order[: stop + 1])
        out[size] = df[ent.isin(keep)].copy()
    return out


def fit_scaling(x: np.ndarray, y: np.ndarray, *, form: str, alpha: float = 0.05) -> dict:
    """OLS fit of an error-vs-n curve; returns params, cov, and a PI-emitting predict.

    Forms (both linear in their fit space, so the classic OLS prediction
    interval applies exactly):

    - ``'powerlaw'``: y = a * x**b, fitted as log y = log a + b log x. The
      returned point prediction exp(mu) is the *median* of the implied
      log-normal predictive, the standard convention for power-law error fits.
    - ``'loglinear'``: y = a + b * log x (for metrics that are already on a
      bounded/linear scale).

    Returns a dict: ``form``; ``params`` {a, b}; ``cov`` (2x2, of the fit-space
    coefficients [intercept, slope] — for powerlaw the intercept is log a);
    ``param_ci`` {a: (lo, hi), b: (lo, hi)} at 1-alpha; ``n``, ``dof``,
    ``sigma2``, ``tcrit``, ``alpha``; ``predict(n) -> (point, lo, hi)`` with a
    two-sided 1-alpha *prediction* interval (new-observation, not mean-response:
    se**2 = sigma2 * (1 + x0' (X'X)^-1 x0)); plus underscore-keyed internals
    used by :func:`validate_extrapolation`.
    """
    if form not in FORMS:
        raise ValueError(f"unknown form {form!r}; expected one of {FORMS}")
    x = np.asarray(x, dtype=float).ravel()
    y = np.asarray(y, dtype=float).ravel()
    if x.shape != y.shape:
        raise ValueError(f"x and y must be aligned, got {x.shape} vs {y.shape}")
    if len(x) < 3:
        raise ValueError(f"need >= 3 points to fit and estimate residual scale, got {len(x)}")
    if not (np.isfinite(x).all() and np.isfinite(y).all()):
        raise ValueError("x and y must be finite")
    if (x <= 0).any():
        raise ValueError("x must be positive (both forms fit against log x)")
    if form == "powerlaw" and (y <= 0).any():
        raise ValueError("powerlaw needs positive y (fit is in log y)")

    u = np.log(x)
    v = np.log(y) if form == "powerlaw" else y
    design = np.column_stack([np.ones_like(u), u])
    beta, *_ = np.linalg.lstsq(design, v, rcond=None)
    resid = v - design @ beta
    dof = len(x) - 2
    sigma2 = float(resid @ resid) / dof
    xtx_inv = np.linalg.inv(design.T @ design)
    cov = sigma2 * xtx_inv
    tcrit = float(stats.t.ppf(1 - alpha / 2, dof))

    def _fit_space_prediction(n_new: Any) -> tuple[np.ndarray, np.ndarray]:
        """(mu, se_pred) in fit space at new x values — the PI building blocks."""
        n_arr = np.atleast_1d(np.asarray(n_new, dtype=float))
        if (n_arr <= 0).any():
            raise ValueError("prediction points must be positive")
        u0 = np.log(n_arr)
        x0 = np.column_stack([np.ones_like(u0), u0])
        mu = x0 @ beta
        se = np.sqrt(sigma2 * (1.0 + np.einsum("ij,jk,ik->i", x0, xtx_inv, x0)))
        return mu, se

    def _to_fit_space_y(y_new: Any) -> np.ndarray:
        y_arr = np.atleast_1d(np.asarray(y_new, dtype=float))
        if form == "powerlaw":
            if (y_arr <= 0).any():
                raise ValueError("powerlaw fit space is log y; y values must be positive")
            return np.log(y_arr)
        return y_arr

    def predict(n_new: Any) -> tuple[Any, Any, Any]:
        scalar = np.isscalar(n_new) or np.ndim(n_new) == 0
        mu, se = _fit_space_prediction(n_new)
        lo, hi = mu - tcrit * se, mu + tcrit * se
        if form == "powerlaw":
            point, lo, hi = np.exp(mu), np.exp(lo), np.exp(hi)
        else:
            point = mu
        if scalar:
            return float(point[0]), float(lo[0]), float(hi[0])
        return point, lo, hi

    se_b = float(np.sqrt(cov[1, 1]))
    se_a = float(np.sqrt(cov[0, 0]))
    b = float(beta[1])
    a_fitspace = float(beta[0])
    a = float(np.exp(a_fitspace)) if form == "powerlaw" else a_fitspace
    a_lo, a_hi = a_fitspace - tcrit * se_a, a_fitspace + tcrit * se_a
    if form == "powerlaw":
        a_lo, a_hi = float(np.exp(a_lo)), float(np.exp(a_hi))
    return {
        "form": form,
        "params": {"a": a, "b": b},
        "cov": cov,
        "param_ci": {"a": (a_lo, a_hi), "b": (b - tcrit * se_b, b + tcrit * se_b)},
        "n": len(x),
        "dof": dof,
        "sigma2": sigma2,
        "alpha": alpha,
        "tcrit": tcrit,
        "predict": predict,
        "_fit_space_prediction": _fit_space_prediction,
        "_to_fit_space_y": _to_fit_space_y,
    }


def validate_extrapolation(fit: dict, x_test: Any, y_test: Any) -> dict:
    """Did held-out truth land inside the fit's own prediction interval?

    The SCL-01 gate: an extrapolated point may be quoted only from a fit whose
    PI covered the held-out rung. Standardized residuals are computed in the
    fit's own space (log y for powerlaw), z = (y - mu) / se_pred, so
    ``within_pi`` is exactly ``|z| <= tcrit`` — the same interval ``predict``
    reports. A wrong functional form shows up as |z| far beyond tcrit at the
    extrapolation distance even when it fit the training range acceptably.

    Returns ``{'within_pi': bool (all test points inside), 'z': standardized
    residual(s), 'n_outside': int, 'tcrit': float}``; ``z`` is a float for
    scalar input, else an ndarray aligned with *x_test*.
    """
    scalar = np.isscalar(x_test) or np.ndim(x_test) == 0
    mu, se = fit["_fit_space_prediction"](x_test)
    v = fit["_to_fit_space_y"](y_test)
    if v.shape != mu.shape:
        raise ValueError(f"x_test and y_test must be aligned, got {mu.shape} vs {v.shape}")
    z = (v - mu) / se
    inside = np.abs(z) <= fit["tcrit"]
    return {
        "within_pi": bool(inside.all()),
        "z": float(z[0]) if scalar else z,
        "n_outside": int((~inside).sum()),
        "tcrit": fit["tcrit"],
    }
