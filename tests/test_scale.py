"""Tests for er_lab.scale: entity ladder, scaling fits, EVT tail gate, cost model."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy.stats import genpareto

from er_lab.scale.costmodel import (
    NODE_RAM_GB,
    CostModel,
    SingleNodeCeilingError,
)
from er_lab.scale.curves import entity_ladder, fit_scaling, validate_extrapolation
from er_lab.scale.evt import diagnostics_pass, fit_gpd_tail, predict_fp_count

# ------------------------------------------------------------- entity_ladder


def corpus(n_entities: int = 200, seed: int = 5) -> pd.DataFrame:
    """Canonical-ish frame: entities with 1-6 records each."""
    rng = np.random.default_rng(seed)
    rows = []
    for e in range(n_entities):
        for r in range(int(rng.integers(1, 7))):
            rows.append({"record_id": f"e{e:04d}_r{r}", "entity_id": f"e{e:04d}"})
    return pd.DataFrame(rows)


def test_entity_ladder_preserves_whole_entities():
    df = corpus()
    full_counts = df["entity_id"].value_counts()
    ladder = entity_ladder(df, sizes=[50, 150, 400], seed=11)
    for size, sub in ladder.items():
        got = sub["entity_id"].value_counts()
        for ent, n in got.items():
            # every included entity brings ALL of its records — none straddles
            # the inclusion boundary
            assert n == full_counts[ent], (size, ent)


def test_entity_ladder_sizes_approximate_targets():
    df = corpus()
    max_entity = int(df["entity_id"].value_counts().max())
    ladder = entity_ladder(df, sizes=[50, 150, 400], seed=11)
    for size, sub in ladder.items():
        assert size <= len(sub) < size + max_entity


def test_entity_ladder_nested_and_deterministic():
    df = corpus()
    ladder = entity_ladder(df, sizes=[50, 150, 400], seed=11)
    ids = {s: set(sub["record_id"]) for s, sub in ladder.items()}
    assert ids[50] <= ids[150] <= ids[400]  # nested: shared entity-permutation prefix
    again = entity_ladder(df, sizes=[50, 150, 400], seed=11)
    for s in ladder:
        pd.testing.assert_frame_equal(ladder[s], again[s])
    different = entity_ladder(df, sizes=[150], seed=12)
    assert set(different[150]["record_id"]) != ids[150]  # seed actually matters


def test_entity_ladder_input_validation():
    df = corpus()
    with pytest.raises(ValueError, match="exceeds"):
        entity_ladder(df, sizes=[len(df) + 1], seed=0)
    with pytest.raises(KeyError, match="entity_id"):
        entity_ladder(df.drop(columns=["entity_id"]), sizes=[10], seed=0)
    with_na = df.copy()
    with_na.loc[0, "entity_id"] = None
    with pytest.raises(ValueError, match="NA"):
        entity_ladder(with_na, sizes=[10], seed=0)


# -------------------------------------------------------------- fit_scaling


def powerlaw_data(rng, a=0.9, b=-0.4, sigma=0.05, n_pts=12):
    x = np.logspace(3, 6, n_pts)
    y = a * x**b * np.exp(rng.normal(0, sigma, size=n_pts))
    return x, y


def test_fit_scaling_recovers_planted_exponent_within_ci():
    rng = np.random.default_rng(3)
    x, y = powerlaw_data(rng)
    fit = fit_scaling(x, y, form="powerlaw")
    b_lo, b_hi = fit["param_ci"]["b"]
    assert b_lo < -0.4 < b_hi
    assert fit["params"]["b"] == pytest.approx(-0.4, abs=0.05)
    assert fit["params"]["a"] == pytest.approx(0.9, rel=0.5)
    assert fit["cov"].shape == (2, 2)


def test_fit_scaling_prediction_interval_covers_at_nominal_rate():
    # small seeded simulation: fit on 12 points, predict one held-out point 10x
    # beyond the range; the 95% PI should cover ~95% of the time
    covered = 0
    n_sim = 60
    for s in range(n_sim):
        rng = np.random.default_rng(1000 + s)
        x, y = powerlaw_data(rng)
        x_hold = 1e7
        y_hold = 0.9 * x_hold**-0.4 * np.exp(rng.normal(0, 0.05))
        fit = fit_scaling(x, y, form="powerlaw")
        _, lo, hi = fit["predict"](x_hold)
        covered += lo <= y_hold <= hi
    assert 0.85 <= covered / n_sim <= 1.0


def test_fit_scaling_predict_shapes_and_order():
    rng = np.random.default_rng(0)
    x, y = powerlaw_data(rng)
    fit = fit_scaling(x, y, form="powerlaw")
    point, lo, hi = fit["predict"](1e5)
    assert isinstance(point, float) and lo < point < hi
    pts, los, his = fit["predict"](np.array([1e4, 1e5, 1e6]))
    assert pts.shape == (3,)
    assert (los < pts).all() and (pts < his).all()


def test_loglinear_form_fits_linear_in_logx():
    rng = np.random.default_rng(4)
    x = np.logspace(2, 5, 10)
    y = 0.3 - 0.02 * np.log(x) + rng.normal(0, 0.003, size=10)
    fit = fit_scaling(x, y, form="loglinear")
    assert fit["params"]["b"] == pytest.approx(-0.02, abs=0.005)
    lo, hi = fit["param_ci"]["b"]
    assert lo < -0.02 < hi


def test_validate_extrapolation_flags_wrong_form():
    # truth is a power law; a loglinear fit looks fine in-sample but its
    # extrapolation misses badly -> the gate must say so
    rng = np.random.default_rng(9)
    x = np.logspace(3, 5, 12)
    y = 0.9 * x**-0.4 * np.exp(rng.normal(0, 0.02, size=12))
    wrong = fit_scaling(x, y, form="loglinear")
    right = fit_scaling(x, y, form="powerlaw")
    x_test, y_test = 1e8, 0.9 * 1e8**-0.4
    bad = validate_extrapolation(wrong, x_test, y_test)
    good = validate_extrapolation(right, x_test, y_test)
    assert not bad["within_pi"]
    assert abs(bad["z"]) > bad["tcrit"]
    assert good["within_pi"]


def test_fit_scaling_validation():
    with pytest.raises(ValueError, match="form"):
        fit_scaling(np.array([1.0, 2, 3]), np.array([1.0, 2, 3]), form="cubic")
    with pytest.raises(ValueError, match=">= 3"):
        fit_scaling(np.array([1.0, 2]), np.array([1.0, 2]), form="powerlaw")
    with pytest.raises(ValueError, match="positive y"):
        fit_scaling(np.array([1.0, 2, 3]), np.array([1.0, -2, 3]), form="powerlaw")


# ---------------------------------------------------------------------- EVT


def test_evt_recovers_planted_xi_and_passes_diagnostics():
    rng = np.random.default_rng(2)
    scores = genpareto.rvs(0.3, scale=2.0, size=30000, random_state=rng)
    fit = fit_gpd_tail(scores)
    for q, f in fit["fits"].items():
        assert f["xi"] == pytest.approx(0.3, abs=0.15), q
        assert f["n_exceed"] >= 30
    ok, reasons = diagnostics_pass(fit)
    assert ok and reasons == []


def test_evt_diagnostics_fail_on_bimodal_scores():
    # far second mode holding ~3% of mass: the q=0.95 threshold sits below the
    # gap, so its excesses are bimodal — not remotely GPD
    rng = np.random.default_rng(0)
    scores = np.concatenate([rng.normal(0, 1, 9700), rng.normal(8, 0.2, 300)])
    fit = fit_gpd_tail(scores)
    ok, reasons = diagnostics_pass(fit)
    assert not ok
    assert reasons  # and they say why
    assert any("xi" in r or "QQ" in r for r in reasons)


def test_evt_diagnostics_structures_present():
    rng = np.random.default_rng(2)
    fit = fit_gpd_tail(genpareto.rvs(0.2, scale=1.0, size=20000, random_state=rng))
    diag = fit["diagnostics"]
    assert {"quantile", "u", "mean_excess", "n_exceed"} <= set(diag["mean_excess_curve"].columns)
    assert {"quantile", "u", "xi", "sigma"} <= set(diag["xi_stability"].columns)
    for q in fit["u_quantiles"]:
        assert 0.0 <= diag["qq"][q]["r2"] <= 1.0


def test_predict_fp_count_known_answer_on_constructed_exceedances():
    # bulk uniform below u=10.0, tail above it = EXACT GPD(xi=0.5, sigma=1)
    # quantiles at 5% of the mass -> the fit recovers the planted tail and the
    # predicted count matches the analytic survival formula
    xi_true, sigma_true, u_true = 0.5, 1.0, 10.0
    n_tail, n_bulk = 500, 9500
    p_grid = (np.arange(n_tail) + 0.5) / n_tail
    tail = u_true + genpareto.ppf(p_grid, xi_true, loc=0, scale=sigma_true)
    bulk = np.linspace(0.0, u_true - 1e-6, n_bulk)
    scores = np.concatenate([bulk, tail])
    fit = fit_gpd_tail(scores, u_quantiles=(0.95,))
    f = fit["fits"][0.95]
    assert f["xi"] == pytest.approx(xi_true, abs=0.1)
    assert f["sigma"] == pytest.approx(sigma_true, rel=0.2)

    n_comparisons, threshold = 1e9, 14.0
    pred = predict_fp_count(fit, n_comparisons=n_comparisons, threshold=threshold, n_boot=64)
    p_u = n_tail / (n_tail + n_bulk)
    sf = (1 + xi_true * (threshold - u_true) / sigma_true) ** (-1 / xi_true)
    analytic = n_comparisons * p_u * sf
    assert pred["point"] == pytest.approx(analytic, rel=0.25)
    assert pred["lo"] <= pred["point"] <= pred["hi"]
    assert pred["n_boot_ok"] > 0


def test_evt_gate_fails_on_nan_diagnostics():
    """Regression (P1): NaN diagnostics must FAIL the gate, never slip past a `<`."""
    nan = float("nan")
    fit = {
        "fits": {
            0.95: {"xi": nan, "n_exceed": 100},
            0.99: {"xi": nan, "n_exceed": 50},
        },
        "diagnostics": {"qq": {0.95: {"r2": nan}, 0.99: {"r2": nan}}},
    }
    ok, reasons = diagnostics_pass(fit)
    assert not ok
    assert reasons  # NaN everywhere used to return (True, []) — a full silent pass
    nonfinite = [r for r in reasons if "non-finite diagnostic" in r]
    assert any("xi" in r for r in nonfinite)
    assert any("r^2" in r for r in nonfinite)
    # a single NaN xi among finite ones must also fail (spread becomes NaN)
    fit["fits"][0.95]["xi"] = 0.2
    fit["diagnostics"]["qq"][0.95]["r2"] = 0.99
    ok, reasons = diagnostics_pass(fit)
    assert not ok and any("non-finite" in r for r in reasons)


def test_evt_piled_up_tail_fails_gate_via_public_path():
    """Constant-excess pile at the top threshold: corrcoef degenerates; the QQ
    r2 must come back ~0 (not NaN) and the gate must fail with a reason for
    that threshold — piled-up duplicate scores are the ER pathology the gate
    exists to catch."""
    rng = np.random.default_rng(0)
    scores = np.concatenate([rng.uniform(0, 1, 9950), np.full(50, 3.0)])
    fit = fit_gpd_tail(scores)
    ok, reasons = diagnostics_pass(fit)
    assert not ok
    assert any("0.995" in r for r in reasons)  # the piled threshold is named
    for q in fit["u_quantiles"]:
        r2 = fit["diagnostics"]["qq"][q]["r2"]
        assert np.isfinite(r2), q  # never NaN out of _qq


def test_evt_qq_constant_excesses_score_near_zero_not_nan():
    from er_lab.scale.evt import _qq

    d = _qq(np.full(60, 1.0), 0.2, 1.0)
    assert np.isfinite(d["r2"])
    assert d["r2"] < 0.01  # degenerate agreement scores as misfit


def test_predict_fp_count_deterministic_and_validated():
    rng = np.random.default_rng(6)
    fit = fit_gpd_tail(genpareto.rvs(0.2, scale=1.0, size=20000, random_state=rng))
    kw = {
        "n_comparisons": 1e8,
        "threshold": fit["fits"][0.99]["u"] * 2 + 5,
        "n_boot": 32,
        "seed": 1,
    }
    assert predict_fp_count(fit, **kw) == predict_fp_count(fit, **kw)
    with pytest.raises(ValueError, match="below every fitted"):
        predict_fp_count(fit, n_comparisons=1e8, threshold=fit["fits"][0.95]["u"] - 1)


# ----------------------------------------------------------------- costmodel


COEFFS = {
    "encode_rps": 1000.0,
    "index_build_s_per_M": 360.0,
    "bytes_per_vector": 1000.0,
    "score_pairs_ps": 1.0e6,
    "cc_edges_ps": 1.0e6,
}
PRICES = {"gpu_hour_usd": 4.0, "cpu_hour_usd": 1.0}


def test_costmodel_breakdown_known_answer():
    model = CostModel(COEFFS, PRICES, measured_max_n=10_000_000)
    out = model.breakdown(3_600_000, {"candidates_per_record": 10}).set_index("stage")
    assert out.loc["encode", "time_h"] == pytest.approx(1.0)  # 3.6e6 / 1000 rps
    assert out.loc["encode", "cost_usd"] == pytest.approx(4.0)  # gpu-priced
    assert out.loc["index_build", "time_h"] == pytest.approx(0.36)  # 3.6 * 360s
    assert out.loc["index_build", "memory_gb"] == pytest.approx(3.6)  # 3.6e6 * 1kB
    assert out.loc["score_pairs", "time_h"] == pytest.approx(0.01)  # 3.6e7 pairs @1e6/s
    assert out.loc["cluster", "cost_usd"] == pytest.approx(0.01)  # cpu-priced
    assert (out["basis"] == "MEASURED").all()


def test_costmodel_extrapolated_basis_beyond_measured_range():
    model = CostModel(COEFFS, PRICES, measured_max_n=10_000_000)
    out = model.breakdown(100_000_000, {"candidates_per_record": 10})
    assert (out["basis"] == "EXTRAPOLATED").all()


def test_costmodel_single_node_ceiling_raises():
    # 1e9 x 1536B fp32 vectors = 1536 GB > the documented 900 GB node
    coeffs = dict(COEFFS, bytes_per_vector=1536.0)
    model = CostModel(coeffs, PRICES)
    with pytest.raises(SingleNodeCeilingError, match=str(int(NODE_RAM_GB))):
        model.breakdown(1e9, {"candidates_per_record": 10})
    # ...while an int8-compressed design (384B) fits and prices fine
    ok = CostModel(dict(COEFFS, bytes_per_vector=384.0), PRICES).breakdown(
        1e9, {"candidates_per_record": 10}
    )
    assert ok["memory_gb"].max() <= NODE_RAM_GB


def test_costmodel_gpu_ceiling_and_validation():
    model = CostModel(COEFFS, PRICES)
    with pytest.raises(SingleNodeCeilingError, match="GPU"):
        # 5e8 x 1000B = 500 GB > 4 x 80 GB combined GPU RAM
        model.breakdown(5e8, {"candidates_per_record": 5, "index_on_gpu": True})
    with pytest.raises(KeyError, match="candidates_per_record"):
        model.breakdown(1e6, {})
    with pytest.raises(KeyError, match="encode_rps"):
        CostModel({}, PRICES)
    with pytest.raises(KeyError, match="gpu_hour_usd"):
        CostModel(COEFFS, {})
