"""er_lab.eval.power — planted-component recovery and formula known answers."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from pytest import approx

from er_lab.eval.power import power_table, seeds_needed, variance_components


def planted_grid(
    sd_seed: float, sd_noise: float, sd_split: float, sd_eps: float, rng_seed: int = 7
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Balanced crossed grid y = mu + a_seed + b_noise + c_split + eps."""
    rng = np.random.default_rng(rng_seed)
    ns, nd, np_ = 12, 10, 8
    a = rng.normal(0, sd_seed, ns)
    b = rng.normal(0, sd_noise, nd)
    c = rng.normal(0, sd_split, np_)
    eps = rng.normal(0, sd_eps, (ns, nd, np_))
    y = 0.8 + a[:, None, None] + b[None, :, None] + c[None, None, :] + eps
    df = pd.DataFrame(
        {
            "value": y.ravel(),
            "seed": np.repeat(np.arange(ns), nd * np_),
            "noise_draw": np.tile(np.repeat(np.arange(nd), np_), ns),
            "split": np.tile(np.arange(np_), ns * nd),
        }
    )
    return df, a, b, c, eps


def test_variance_components_recovers_planted_components() -> None:
    df, a, b, c, eps = planted_grid(sd_seed=0.2, sd_noise=0.05, sd_split=0.1, sd_eps=0.03)
    vc = variance_components(df)
    assert set(vc) == {"seed", "noise_draw", "split", "residual"}
    # the estimator targets the REALIZED level-effect variances (the level
    # draws themselves are sampling noise on the population sd), so compare
    # tightly against those and loosely against the planted population values
    assert vc["seed"] == approx(a.var(ddof=1), rel=0.05)
    assert vc["noise_draw"] == approx(b.var(ddof=1), rel=0.15)
    assert vc["split"] == approx(c.var(ddof=1), rel=0.10)
    assert vc["residual"] == approx(eps.var(ddof=1), rel=0.10)
    assert 0.3 * 0.2**2 < vc["seed"] < 3.0 * 0.2**2
    assert 0.5 * 0.03**2 < vc["residual"] < 2.0 * 0.03**2


def test_variance_components_zero_component_estimated_near_zero() -> None:
    df, *_ = planted_grid(sd_seed=0.2, sd_noise=0.0, sd_split=0.1, sd_eps=0.03, rng_seed=11)
    vc = variance_components(df)
    assert 0.0 <= vc["noise_draw"] < 5e-5  # clipped MoM estimate, ~sigma_eps^2/n_per


def test_variance_components_single_level_factor_is_nan() -> None:
    df = pd.DataFrame(
        {
            "value": [0.1, 0.2, 0.3, 0.4, 0.15, 0.25, 0.35, 0.45],
            "seed": [0, 0, 1, 1, 0, 0, 1, 1],
            "noise_draw": [0, 1, 0, 1, 0, 1, 0, 1],
            "split": [0] * 8,
        }
    )
    vc = variance_components(df)
    assert np.isnan(vc["split"])
    assert np.isfinite(vc["seed"]) and np.isfinite(vc["residual"])


def test_variance_components_validation() -> None:
    with pytest.raises(ValueError, match="'value' column"):
        variance_components(pd.DataFrame({"v": [1.0], "seed": [0]}))
    with pytest.raises(ValueError, match="no factor columns"):
        variance_components(pd.DataFrame({"value": [1.0, 2.0]}))
    with pytest.raises(ValueError, match="residual"):
        # 2x2 grid with 4 obs: df_resid = 3 - 1 - 1 - ... = 1 ok; 2 obs is not
        variance_components(pd.DataFrame({"value": [1.0, 2.0], "seed": [0, 1], "split": [0, 1]}))


def test_seeds_needed_known_answers() -> None:
    # sigma^2 = 1e-4, delta = 0.01 -> delta = sigma:
    #   n = ceil((z_.975 + z_.8)^2) = ceil((1.959964 + 0.841621)^2)
    #     = ceil(7.8489) = 8   (the classic anchor)
    assert seeds_needed(0.01, 5e-5, 5e-5) == 8
    # delta = 2 sigma -> ceil(7.8489 / 4) = 2
    assert seeds_needed(0.02, 5e-5, 5e-5) == 2
    # one-sided: ceil((1.644854 + 0.841621)^2) = ceil(6.1826) = 7
    assert seeds_needed(0.01, 5e-5, 5e-5, two_sided=False) == 7
    # power .9: ceil((1.959964 + 1.281552)^2) = ceil(10.5074) = 11
    assert seeds_needed(0.01, 5e-5, 5e-5, power=0.9) == 11
    # zero variance -> floor of 2
    assert seeds_needed(0.01, 0.0, 0.0) == 2
    with pytest.raises(ValueError, match="target_delta"):
        seeds_needed(0.0, 1e-4, 0.0)
    with pytest.raises(ValueError, match=">= 0"):
        seeds_needed(0.01, -1e-4, 0.0)


def test_power_table_known_answers_and_monotonicity() -> None:
    vc = {"seed": 5e-5, "noise_draw": 5e-5, "residual": 1e-3}
    table = power_table([0.005, 0.01, 0.02], vc, n_levels=(3, 8))
    assert list(table["delta"]) == [0.005, 0.01, 0.02]
    assert table["sd_replicate"].iloc[0] == approx(0.01)
    # seeds_needed column matches the direct function and is monotone in delta
    assert list(table["seeds_needed"]) == [32, 8, 2]
    # achieved power at the recommended n=8 for delta=sigma:
    #   Phi(sqrt(8) - 1.95996) + Phi(-sqrt(8) - 1.95996) = 0.80736...
    assert table.loc[1, "power_at_8"] == approx(0.80736, abs=1e-4)
    assert (table["power_at_8"].to_numpy() >= table["power_at_3"].to_numpy() - 1e-12).all()
    # missing keys count as zero variance; NaN components are rejected
    zero = power_table([0.01], {"residual": 1.0})
    assert zero["seeds_needed"].iloc[0] == 2 and zero["power_at_3"].iloc[0] == 1.0
    with pytest.raises(ValueError, match="var_components"):
        power_table([0.01], {"seed": float("nan"), "noise_draw": 0.0})
