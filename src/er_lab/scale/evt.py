"""GPD tail modelling of non-match similarity scores (SCL-02, PLAN §3).

At 1e8-1e9 records the false-positive budget is spent in the extreme upper
tail of the *non-match* similarity distribution — a region with almost no
observations at smoke/mid tier. Extreme-value theory (Pickands-Balkema-
de Haan) says excesses over a high threshold u converge to a Generalized
Pareto Distribution GPD(xi, sigma), which licenses extrapolating tail
exceedance rates beyond the observed range — but ONLY when the GPD actually
fits. Hence the module's shape:

- :func:`fit_gpd_tail` fits excesses at several threshold quantiles (MLE,
  location pinned at 0) and computes the three standard graphical
  diagnostics as data: the mean-excess curve (linear iff GPD), xi stability
  across thresholds (flat iff the asymptotics have kicked in), and per-
  threshold QQ agreement.
- :func:`diagnostics_pass` is the PLAN's honesty gate: no smoke-tier xi is
  quoted in any notebook without this returning True. Its checks are
  deliberately crude, documented heuristics (enough exceedances, xi spread,
  QQ linearity) — the notebooks still render the curves for the eye.
- :func:`predict_fp_count` turns a fit into an expected false-positive count
  above a score threshold at an extrapolated comparison count, with a seeded
  parametric-ish bootstrap interval (resampling both the exceedance rate and
  the excess sample, refitting each replicate).

Everything is CPU/scipy; deterministic given the seed arguments.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import genpareto

__all__ = ["diagnostics_pass", "fit_gpd_tail", "predict_fp_count"]

#: diagnostics_pass heuristics (documented, deliberately crude — see gate docstring)
MIN_EXCEEDANCES = 30  # below this, MLE xi is noise
XI_SPREAD_TOL = 0.35  # max-min xi across requested thresholds; flat-ish = stable
QQ_R2_MIN = 0.95  # squared correlation of empirical vs fitted GPD quantiles


def fit_gpd_tail(
    scores: np.ndarray, *, u_quantiles: tuple[float, ...] = (0.95, 0.99, 0.995)
) -> dict:
    """Fit GPD(xi, sigma) to excesses over each threshold quantile, with diagnostics.

    For each q in *u_quantiles*: u = quantile(scores, q), excesses = scores
    above u minus u, MLE fit with location fixed at 0 (scipy ``genpareto.fit
    (floc=0)``; scipy's ``c`` is the EVT shape xi). xi > 0 is a heavy
    (Frechet-domain) tail — the dangerous case for FP budgets; xi = 0
    exponential-like; xi < 0 a finite endpoint.

    Returns::

        {'n_total', 'u_quantiles',
         'fits': {q: {'u', 'xi', 'sigma', 'n_exceed', 'p_exceed', 'excesses'}},
         'diagnostics': {
            'mean_excess_curve': DataFrame[quantile, u, mean_excess, n_exceed],
            'xi_stability':      DataFrame[quantile, u, xi, sigma, n_exceed],
            'qq': {q: {'theoretical', 'empirical', 'r2'}}}}

    The raw ``excesses`` arrays ride along so :func:`predict_fp_count` can
    bootstrap without re-touching the (possibly huge) score vector.
    """
    scores = np.asarray(scores, dtype=float).ravel()
    if len(scores) < 100:
        raise ValueError(f"need >= 100 scores for any tail fit, got {len(scores)}")
    if not np.isfinite(scores).all():
        raise ValueError("scores must be finite")
    qs = tuple(float(q) for q in u_quantiles)
    if any(not (0.5 < q < 1.0) for q in qs) or list(qs) != sorted(set(qs)):
        raise ValueError(f"u_quantiles must be strictly increasing in (0.5, 1), got {qs}")

    fits: dict[float, dict] = {}
    for q in qs:
        u = float(np.quantile(scores, q))
        exc = scores[scores > u] - u
        if len(exc) < 10:
            raise ValueError(
                f"only {len(exc)} exceedances above the q={q} threshold — too few to fit; "
                "drop the highest quantiles or bring more scores"
            )
        xi, _, sigma = genpareto.fit(exc, floc=0)
        fits[q] = {
            "u": u,
            "xi": float(xi),
            "sigma": float(sigma),
            "n_exceed": len(exc),
            "p_exceed": len(exc) / len(scores),
            "excesses": exc,
        }

    return {
        "n_total": len(scores),
        "u_quantiles": qs,
        "fits": fits,
        "diagnostics": {
            "mean_excess_curve": _mean_excess_curve(scores),
            "xi_stability": _xi_stability(scores),
            "qq": {q: _qq(f["excesses"], f["xi"], f["sigma"]) for q, f in fits.items()},
        },
    }


def diagnostics_pass(fits: dict) -> tuple[bool, list[str]]:
    """The gate: is this tail actually GPD enough to quote xi from?

    Checks (heuristics with pinned constants, in module header):

    1. every requested threshold has >= ``MIN_EXCEEDANCES`` exceedances;
    2. xi is threshold-stable: max-min over requested thresholds <=
       ``XI_SPREAD_TOL`` (a genuine GPD tail is threshold-invariant in xi;
       mixtures/bimodal score piles swing wildly as u crosses a mode);
    3. QQ agreement: r^2 >= ``QQ_R2_MIN`` at every requested threshold,
       scored on the unit-exponential scale (see :func:`_qq` for why raw
       quantile correlation cannot gate heavy tails) — catches piled-up or
       gapped excesses that a 2-parameter MLE happily 'fits' with garbage
       parameters.

    Returns ``(ok, reasons)`` — reasons is empty iff ok. The mean-excess curve
    is rendered for the notebook eye but NOT auto-gated: its linearity
    statistic is too noisy at smoke-tier sample sizes to make a fair gate.
    """
    per_u = fits["fits"]
    reasons: list[str] = []
    for q, f in per_u.items():
        if f["n_exceed"] < MIN_EXCEEDANCES:
            reasons.append(
                f"n_exceed={f['n_exceed']} < {MIN_EXCEEDANCES} at u_quantile={q}"
            )
    xis = {q: f["xi"] for q, f in per_u.items()}
    spread = max(xis.values()) - min(xis.values())
    if spread > XI_SPREAD_TOL:
        lo_q = min(xis, key=xis.get)  # type: ignore[arg-type]
        hi_q = max(xis, key=xis.get)  # type: ignore[arg-type]
        reasons.append(
            f"xi not threshold-stable: spread {spread:.2f} > {XI_SPREAD_TOL} "
            f"(xi={xis[lo_q]:.2f} at q={lo_q} vs xi={xis[hi_q]:.2f} at q={hi_q})"
        )
    for q, d in fits["diagnostics"]["qq"].items():
        if d["r2"] < QQ_R2_MIN:
            reasons.append(f"QQ r^2={d['r2']:.3f} < {QQ_R2_MIN} at u_quantile={q}")
    return (not reasons, reasons)


def predict_fp_count(
    fit: dict,
    *,
    n_comparisons: float,
    threshold: float,
    level: float = 0.95,
    n_boot: int = 200,
    seed: int = 0,
) -> dict:
    """Expected count of non-match scores above *threshold* among *n_comparisons*.

    Uses the fitted threshold with the largest u <= *threshold* (the most
    local fit whose survival formula is valid there). The GPD tail formula:
    P(S > t) = p_u * (1 + xi (t-u)/sigma)^(-1/xi)  (p_u * exp(-(t-u)/sigma)
    at xi=0; zero beyond the finite endpoint when xi < 0), and
    point = n_comparisons * P(S > t).

    Interval: seeded bootstrap of BOTH uncertainty sources — the exceedance
    rate p_u (binomial over n_total) and the GPD parameters (resample the
    excesses, refit MLE) — percentile (1-level) interval over replicate
    counts, widened if needed so the point estimate is always inside.
    Deterministic for fixed *seed*/*n_boot*.

    Returns ``{'point', 'lo', 'hi', 'u', 'u_quantile', 'xi', 'sigma',
    'n_boot_ok'}``.
    """
    per_u = fit["fits"]
    eligible = [q for q, f in per_u.items() if f["u"] <= threshold]
    if not eligible:
        us = {q: f["u"] for q, f in per_u.items()}
        raise ValueError(
            f"threshold {threshold} is below every fitted tail threshold {us} — "
            "the GPD survival formula only extrapolates upward from u"
        )
    q_sel = max(eligible, key=lambda q: per_u[q]["u"])
    f = per_u[q_sel]
    u, exc, n_total = f["u"], np.asarray(f["excesses"], dtype=float), fit["n_total"]

    point = n_comparisons * f["p_exceed"] * _gpd_sf(threshold - u, f["xi"], f["sigma"])

    rng = np.random.default_rng(seed)
    counts = []
    for _ in range(n_boot):
        m_star = int(rng.binomial(n_total, len(exc) / n_total))
        sample = rng.choice(exc, size=max(m_star, 10), replace=True)
        refit = _try_fit(sample)
        if refit is not None:
            xi_s, sigma_s = refit
            counts.append(
                n_comparisons * (m_star / n_total) * _gpd_sf(threshold - u, xi_s, sigma_s)
            )
    if counts:
        tail = (1 - level) / 2 * 100
        lo, hi = np.percentile(counts, [tail, 100 - tail])
        lo, hi = min(float(lo), point), max(float(hi), point)
    else:  # every refit failed — no honest interval to report
        lo = hi = float("nan")
    return {
        "point": float(point),
        "lo": lo,
        "hi": hi,
        "u": u,
        "u_quantile": q_sel,
        "xi": f["xi"],
        "sigma": f["sigma"],
        "n_boot_ok": len(counts),
    }


# -- internals ---------------------------------------------------------------


def _try_fit(sample: np.ndarray) -> tuple[float, float] | None:
    """One bootstrap MLE refit; None when the optimizer fails on a degenerate resample."""
    try:
        xi, _, sigma = genpareto.fit(sample, floc=0)
    except (RuntimeError, ValueError, FloatingPointError):
        return None
    return float(xi), float(sigma)


def _gpd_sf(excess: float, xi: float, sigma: float) -> float:
    """GPD survival function at a non-negative excess over the threshold."""
    if excess < 0:
        raise ValueError("excess must be >= 0 (threshold above u)")
    if abs(xi) < 1e-9:
        return float(np.exp(-excess / sigma))
    arg = 1.0 + xi * excess / sigma
    if arg <= 0:  # xi < 0: beyond the finite endpoint u - sigma/xi
        return 0.0
    return float(arg ** (-1.0 / xi))


def _mean_excess_curve(scores: np.ndarray, n_grid: int = 40) -> pd.DataFrame:
    """Mean excess E[S - u | S > u] over a quantile grid — linear iff GPD tail."""
    rows = []
    for q in np.linspace(0.80, 0.99, n_grid):
        u = float(np.quantile(scores, q))
        exc = scores[scores > u] - u
        if len(exc) == 0:
            continue
        rows.append(
            {"quantile": q, "u": u, "mean_excess": float(exc.mean()), "n_exceed": len(exc)}
        )
    return pd.DataFrame(rows)


def _xi_stability(scores: np.ndarray, n_grid: int = 12) -> pd.DataFrame:
    """MLE (xi, sigma) across a threshold grid — flat xi iff the GPD regime is reached."""
    rows = []
    for q in np.linspace(0.90, 0.995, n_grid):
        u = float(np.quantile(scores, q))
        exc = scores[scores > u] - u
        if len(exc) < 10:
            continue
        xi, _, sigma = genpareto.fit(exc, floc=0)
        rows.append(
            {"quantile": q, "u": u, "xi": float(xi), "sigma": float(sigma),
             "n_exceed": len(exc)}
        )
    return pd.DataFrame(rows)


def _qq(excesses: np.ndarray, xi: float, sigma: float) -> dict[str, Any]:
    """QQ agreement with the fitted GPD, scored on the unit-exponential scale.

    ``theoretical``/``empirical`` are the raw GPD quantile pairs (for the
    notebook's QQ plot). The r^2 gate, however, is computed after pushing the
    empirical excesses through the fitted survival function, z = -log
    sf(excess): if the fit is right these are Exp(1) order statistics
    whatever xi is. Raw-quantile correlation is useless as a gate for heavy
    tails (xi > 0 makes the top order statistic wildly variable, so honest
    fits score badly); the exponential transform stabilizes it. An excess
    beyond the fitted support (sf = 0, possible when xi < 0) is maximal
    misfit and scores r2 = 0 outright.
    """
    emp = np.sort(excesses)
    m = len(emp)
    grid = (np.arange(1, m + 1) - 0.5) / m
    theo = genpareto.ppf(grid, xi, loc=0, scale=sigma)
    z = -genpareto.logsf(emp, xi, loc=0, scale=sigma)
    if not np.isfinite(z).all():
        r2 = 0.0
    else:
        r = np.corrcoef(-np.log(1 - grid), z)[0, 1]
        r2 = float(r * r)
    return {"theoretical": theo, "empirical": emp, "r2": r2}
