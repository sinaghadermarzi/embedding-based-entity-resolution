"""Score -> match-probability calibration (CAL-01, PLAN §3).

Cosine similarities (and FS match weights) are not probabilities; decisions at
fixed entity-precision operating points need calibrated ones. CAL-01's twist
is *heterogeneity*: the score→probability mapping differs across, e.g.,
name-frequency bands — a 0.9 cosine between two "JOHN SMITH" records means
less than between two rare names. So the calibrator optionally fits one map
per stratum and refuses to pretend the strata don't exist at transform time.

Methods (sklearn under the hood, both monotone-friendly):

- ``'isotonic'`` — nonparametric monotone regression; clips out-of-range
  scores to the fitted range (no wild extrapolation),
- ``'platt'``    — logistic fit on the raw score (Platt scaling); effectively
  unregularized (C=1e6) as Platt's method prescribes.

Selection-bias caveat carried by CAL-01, not solved here: fit on pairs drawn
the same way the deployed scorer sees them (post-blocking), or the estimated
probabilities inherit the blocking selection bias.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

__all__ = ["Calibrator", "fit_calibrator"]

_METHODS = ("isotonic", "platt")


def _fit_one(scores: np.ndarray, labels: np.ndarray, method: str):
    """Fit a single score->prob model, or None when this slice is unfittable."""
    if method == "isotonic":
        from sklearn.isotonic import IsotonicRegression

        if len(np.unique(scores)) < 2 or len(np.unique(labels)) < 2:
            return None
        model = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip", increasing=True)
        model.fit(scores, labels)
        return model
    from sklearn.linear_model import LogisticRegression

    if len(np.unique(labels)) < 2:
        return None  # logistic regression needs both classes
    model = LogisticRegression(C=1e6, solver="lbfgs", max_iter=1000)
    model.fit(scores.reshape(-1, 1), labels)
    return model


def _predict_one(model: Any, scores: np.ndarray, method: str) -> np.ndarray:
    if method == "isotonic":
        return np.asarray(model.predict(scores), dtype=float)
    return np.asarray(model.predict_proba(scores.reshape(-1, 1))[:, 1], dtype=float)


@dataclass
class Calibrator:
    """A fitted score->probability map, optionally per-stratum.

    ``models`` maps stratum value -> fitted sklearn model; ``pooled`` is the
    all-data fallback used for strata unseen at fit time (or too small/pure to
    fit) — the fallback is recorded in ``fallback_strata`` so notebooks can
    report it instead of hiding it.
    """

    method: str
    pooled: Any
    models: dict[Any, Any] = field(default_factory=dict)
    stratified: bool = False
    fallback_strata: list[Any] = field(default_factory=list)

    def transform(self, scores: np.ndarray, strata: np.ndarray | None = None) -> np.ndarray:
        """Calibrated probabilities for *scores* (per-stratum when fitted that way).

        A stratified calibrator demands *strata* — silently pooling would
        reintroduce exactly the heterogeneity bias CAL-01 measures. Stratum
        values never seen at fit time fall back to the pooled model.
        """
        scores = np.asarray(scores, dtype=float)
        if not self.stratified:
            return _predict_one(self.pooled, scores, self.method)
        if strata is None:
            raise ValueError(
                "this Calibrator was fitted per-stratum; pass strata= to transform "
                "(pooling would hide the heterogeneity it was built to correct)"
            )
        strata = np.asarray(strata)
        if len(strata) != len(scores):
            raise ValueError(f"strata has {len(strata)} entries for {len(scores)} scores")
        out = np.empty(len(scores), dtype=float)
        for value in np.unique(strata):
            mask = strata == value
            model = self.models.get(value, self.pooled)
            out[mask] = _predict_one(model, scores[mask], self.method)
        return out


def fit_calibrator(
    scores: np.ndarray,
    labels: np.ndarray,
    *,
    method: str = "isotonic",
    strata: np.ndarray | None = None,
) -> Calibrator:
    """Fit a :class:`Calibrator` on scored pairs with binary match labels.

    With *strata* (one value per pair, e.g. a name-frequency band), one model
    is fitted per stratum plus a pooled fallback; a stratum whose slice cannot
    support a fit (single class, or a single distinct score for isotonic)
    uses the pooled model and is listed in ``fallback_strata``. Deterministic:
    both sklearn fits are exact optimizations with no random state.
    """
    if method not in _METHODS:
        raise ValueError(f"unknown method {method!r}: expected one of {_METHODS}")
    scores = np.asarray(scores, dtype=float)
    labels = np.asarray(labels, dtype=float)
    if scores.ndim != 1 or scores.shape != labels.shape:
        raise ValueError(f"scores {scores.shape} and labels {labels.shape} must be 1-D aligned")
    if not np.isin(np.unique(labels), (0.0, 1.0)).all():
        raise ValueError("labels must be binary (0/1)")

    pooled = _fit_one(scores, labels, method)
    if pooled is None:
        raise ValueError(
            "cannot fit calibrator: need both classes present "
            "(and >=2 distinct scores for isotonic)"
        )
    if strata is None:
        return Calibrator(method=method, pooled=pooled)

    strata = np.asarray(strata)
    if len(strata) != len(scores):
        raise ValueError(f"strata has {len(strata)} entries for {len(scores)} scores")
    models: dict[Any, Any] = {}
    fallback: list[Any] = []
    for value in np.unique(strata):
        mask = strata == value
        model = _fit_one(scores[mask], labels[mask], method)
        if model is None:
            fallback.append(value)
        else:
            models[value] = model
    return Calibrator(
        method=method,
        pooled=pooled,
        models=models,
        stratified=True,
        fallback_strata=fallback,
    )
