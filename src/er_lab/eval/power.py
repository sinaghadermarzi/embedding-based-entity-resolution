"""Variance decomposition and power analysis for replicate planning (MET-04).

MET-04 is the binding matrix-pruning arbiter (PLAN.md §3): every training
cell runs >=3-5 seeds x noise draws, and *this* module says whether that is
enough. The workflow: run a pilot grid, decompose metric variance into
per-factor components with :func:`variance_components`, then size seed counts
for the deltas that matter with :func:`seeds_needed` / :func:`power_table`.
All formulas are normal approximations, stated in each docstring — the point
is honest planning numbers, not exact inference.
"""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd
from scipy.special import ndtr, ndtri

__all__ = ["variance_components", "seeds_needed", "power_table"]


def variance_components(
    long_df: pd.DataFrame, *, value: str = "value", factors: list[str] | None = None
) -> dict[str, float]:
    """Method-of-moments variance components for crossed random factors.

    Model (assumptions, documented): additive crossed random effects

        y = mu + a_seed + b_noise_draw + c_split + eps,

    with independent zero-mean components and NO interactions — interaction
    variance, if present, is absorbed into the residual, so components are
    conservative for the main factors. ``long_df`` has one row per run: a
    ``value`` column plus one column per factor (default: every non-value
    column, e.g. seed / noise_draw / split); factor levels are treated as
    draws from a population of levels (random effects).

    Estimation (ANOVA-style method of moments): with L_f levels of factor f
    and n_f = N / L_f observations per level, the sample variance of the
    factor's level means satisfies E[S²_f] = sigma²_f + sigma²_eps / n_f in a
    balanced crossed design (the other factors' effects are identical across
    the level means there, so they cancel as constants). The residual is the
    additive-model residual y - Σ_f mean_f(level) + (K-1)·grand_mean with
    df = N - 1 - Σ_f (L_f - 1). Hence

        sigma²_f = max(0, S²_f - sigma²_eps / n_f)      (clipped at 0),
        sigma²_eps = SS_resid / df_resid.

    Exactly unbiased under balance; under mild imbalance n_f = N / L_f is an
    approximation and other factors leak slightly into S²_f — keep pilot
    grids balanced. A factor with a single level gets NaN (inestimable).
    Returns {factor: variance, ..., 'residual': variance}.
    """
    if not isinstance(long_df, pd.DataFrame):
        raise TypeError("long_df must be a pandas DataFrame")
    if value not in long_df.columns:
        raise ValueError(f"long_df needs a {value!r} column")
    if factors is None:
        factors = [c for c in long_df.columns if c != value]
    if not factors:
        raise ValueError("no factor columns found")
    missing = [c for c in factors if c not in long_df.columns]
    if missing:
        raise ValueError(f"factor columns not in long_df: {missing}")
    df = long_df.dropna(subset=[value])
    y = pd.to_numeric(df[value], errors="raise").astype(float).to_numpy()
    if not np.isfinite(y).all():
        raise ValueError("values must be finite")
    df = df.assign(**{value: y})
    n = len(y)
    grand = y.mean()

    level_means: dict[str, pd.Series] = {}
    n_levels: dict[str, int] = {}
    fitted = np.full(n, grand)
    df_resid = n - 1
    for f in factors:
        means = df.groupby(f, sort=False)[value].mean()
        level_means[f] = means
        n_levels[f] = len(means)
        fitted = fitted + means.loc[df[f]].to_numpy() - grand
        df_resid -= len(means) - 1
    if df_resid <= 0:
        raise ValueError(
            f"not enough runs for a residual estimate: N={n}, residual df={df_resid}"
        )
    resid = y - fitted
    var_resid = float((resid**2).sum() / df_resid)

    out: dict[str, float] = {}
    for f in factors:
        levels = n_levels[f]
        if levels < 2:
            out[f] = float("nan")
            continue
        s2 = float(level_means[f].var(ddof=1))
        n_per = n / levels
        out[f] = max(0.0, s2 - var_resid / n_per)
    out["residual"] = var_resid
    return out


def seeds_needed(
    target_delta: float,
    var_seed: float,
    var_noise: float,
    *,
    alpha: float = 0.05,
    power: float = 0.8,
    two_sided: bool = True,
) -> int:
    """Replicates (seed x noise-draw pairs) needed to detect a metric delta.

    Model and formula (normal approximation, documented): per-replicate
    paired deltas Delta_r ~ iid N(delta, sigma²) with sigma² = var_seed +
    var_noise. This is deliberately conservative for the lab's shared-
    replicate design: training seeds are system-specific so seed variance
    never cancels in the pairing, and treating the noise-draw component as
    non-cancelling too gives an upper bound (shared draws can only shrink
    sigma²). A z-test on the mean of n deltas reaches the target power when

        n = ceil( (z_{1-alpha/2} + z_{power})² · sigma² / delta² )

    (one-sided: z_{1-alpha}), ignoring the negligible far-tail term of the
    two-sided power. Known anchor: delta = sigma gives n = ceil(7.849) = 8 at
    alpha=0.05, power=0.8. Floor of 2 (you cannot even estimate variance from
    one replicate); no small-sample t inflation is applied — for the n <= 5
    regime treat the result as optimistic by roughly one replicate.
    """
    if target_delta <= 0:
        raise ValueError("target_delta must be > 0")
    if var_seed < 0 or var_noise < 0:
        raise ValueError("variance components must be >= 0")
    if not (0 < alpha < 1) or not (0 < power < 1):
        raise ValueError("alpha and power must be in (0, 1)")
    sigma2 = var_seed + var_noise
    if sigma2 == 0:
        return 2
    z_a = float(ndtri(1 - alpha / 2)) if two_sided else float(ndtri(1 - alpha))
    z_b = float(ndtri(power))
    n = int(np.ceil((z_a + z_b) ** 2 * sigma2 / target_delta**2))
    return max(n, 2)


def power_table(
    deltas: Iterable[float],
    var_components: dict[str, float],
    *,
    alpha: float = 0.05,
    power: float = 0.8,
    two_sided: bool = True,
    n_levels: tuple[int, ...] = (3, 5, 10),
) -> pd.DataFrame:
    """The MET-04 power table: seeds needed and achieved power per delta.

    ``var_components`` is the dict from :func:`variance_components`; the
    replicate variance is var_components['seed'] + var_components['noise_draw']
    (a missing key counts as 0 — e.g. a pilot without a split factor; NaN
    components are rejected, re-run the pilot with >= 2 levels). One row per
    delta: ``seeds_needed`` at (alpha, power), plus ``power_at_{n}`` for each
    n in ``n_levels`` — the achieved power of the two-sided z-test,

        power(n) = Phi(delta·sqrt(n)/sigma − z_{1−alpha/2})
                 + Phi(−delta·sqrt(n)/sigma − z_{1−alpha/2}),

    under the same normal model as :func:`seeds_needed`. This table is the
    binding arbiter for pruning the experiment matrix (PLAN §3): an arm whose
    detectable delta needs more seeds than the budget affords is fractionated
    or cut by the pre-registered rule, never quietly under-replicated.
    """
    var_seed = var_components.get("seed", 0.0)
    var_noise = var_components.get("noise_draw", 0.0)
    for name, v in (("seed", var_seed), ("noise_draw", var_noise)):
        if not np.isfinite(v) or v < 0:
            raise ValueError(f"var_components[{name!r}] must be finite and >= 0, got {v}")
    sigma = float(np.sqrt(var_seed + var_noise))
    z_a = float(ndtri(1 - alpha / 2)) if two_sided else float(ndtri(1 - alpha))

    def achieved(delta: float, n: int) -> float:
        if sigma == 0:
            return 1.0
        shift = delta * np.sqrt(n) / sigma
        p = float(ndtr(shift - z_a))
        if two_sided:
            p += float(ndtr(-shift - z_a))
        return p

    rows = []
    for delta in deltas:
        row = {
            "delta": float(delta),
            "sd_replicate": sigma,
            "seeds_needed": seeds_needed(
                float(delta), var_seed, var_noise, alpha=alpha, power=power, two_sided=two_sided
            ),
        }
        for n in n_levels:
            row[f"power_at_{n}"] = achieved(float(delta), int(n))
        rows.append(row)
    return pd.DataFrame(rows)
