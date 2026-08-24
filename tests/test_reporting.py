"""Tests for er_lab.reporting: figures render only from artifacts; immutable cards."""

from __future__ import annotations

import re

import matplotlib

matplotlib.use("Agg")  # headless backend before pyplot is touched

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from er_lab.infra.artifacts import ArtifactMissing, ArtifactRegistry
from er_lab.reporting.cards import CardImmutableError, conjecture_card, verdict_box
from er_lab.reporting.figures import (
    CAPTION_GID,
    WATERMARK_GID,
    line_with_ci,
    regime_heatmap,
    scaling_curve,
    setup_style,
    three_panel_pressure,
)

CFG = {"run": {"seed": 17, "tier": "smoke"}}
TIER = "smoke"


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")


@pytest.fixture()
def registry(tmp_path) -> ArtifactRegistry:
    return ArtifactRegistry(tmp_path / "artifacts")


def xy_frame(basis: str | None = None) -> pd.DataFrame:
    x = np.arange(1, 8, dtype=float)
    df = pd.DataFrame({"x": x, "y": 1 / x, "lo": 0.9 / x, "hi": 1.1 / x})
    if basis is not None:
        df["basis"] = basis
    return df


def caption_of(fig) -> str:
    texts = [t for t in fig.texts if t.get_gid() == CAPTION_GID]
    assert len(texts) == 1
    return texts[0].get_text()


def has_watermark(fig) -> bool:
    return any(t.get_gid() == WATERMARK_GID for t in fig.texts)


# ------------------------------------------------------------------ figures


def test_figures_refuse_missing_artifacts(registry):
    setup_style()
    with pytest.raises(ArtifactMissing):
        line_with_ci(registry, tier=TIER, artifact="nope")
    with pytest.raises(ArtifactMissing):
        regime_heatmap(registry, tier=TIER, artifact="nope")
    with pytest.raises(ArtifactMissing):
        scaling_curve(registry, tier=TIER, artifact="nope")
    with pytest.raises(ArtifactMissing):
        three_panel_pressure(
            registry, tier=TIER, property_shift="a", geometry="b", system_metric="c"
        )


def test_three_panel_refuses_when_any_one_input_missing(registry):
    registry.register("prop", xy_frame(), cfg=CFG, tier=TIER)
    registry.register("geo", xy_frame(), cfg=CFG, tier=TIER)
    # system_metric never registered -> whole triptych refuses
    with pytest.raises(ArtifactMissing):
        three_panel_pressure(
            registry, tier=TIER, property_shift="prop", geometry="geo", system_metric="sys"
        )


def test_line_with_ci_caption_stamp(registry):
    registry.register("curve_a", xy_frame(), cfg=CFG, tier=TIER)
    _, meta = registry.load("curve_a", tier=TIER)
    fig = line_with_ci(registry, tier=TIER, artifact="curve_a")
    cap = caption_of(fig)
    assert "curve_a" in cap
    assert meta["config_hash"] in cap
    assert re.search(r"\b[0-9a-f]{12}\b", cap)  # a 12-hex config hash
    assert f"tier {TIER}" in cap
    assert cap.endswith("MEASURED")
    assert not has_watermark(fig)


def test_extrapolated_series_triggers_watermark_and_dashes(registry):
    df = pd.concat([xy_frame("MEASURED").iloc[:4], xy_frame("EXTRAPOLATED").iloc[4:]])
    registry.register("scaling", df.rename(columns={"x": "n"}), cfg=CFG, tier=TIER)
    fig = scaling_curve(registry, tier=TIER, artifact="scaling")
    assert has_watermark(fig)
    assert caption_of(fig).endswith("EXTRAPOLATED")
    styles = {line.get_linestyle() for line in fig.axes[0].get_lines()}
    assert "-" in styles and "--" in styles  # measured solid, extrapolated dashed


def test_meta_level_basis_marks_whole_figure_extrapolated(registry):
    registry.register(
        "projection", xy_frame(), cfg=CFG, tier=TIER, meta={"basis": "EXTRAPOLATED"}
    )
    fig = line_with_ci(registry, tier=TIER, artifact="projection")
    assert has_watermark(fig)
    assert caption_of(fig).endswith("EXTRAPOLATED")


def test_three_panel_renders_and_stamps_all_names(registry):
    registry.register("prop", xy_frame(), cfg=CFG, tier=TIER)
    geo = pd.DataFrame(
        {"x": np.random.default_rng(0).normal(size=30),
         "y": np.random.default_rng(1).normal(size=30),
         "group": ["a", "b", "c"] * 10}
    )
    registry.register("geo", geo, cfg=CFG, tier=TIER)
    registry.register("sys", xy_frame(), cfg=CFG, tier=TIER)
    fig = three_panel_pressure(
        registry, tier=TIER, property_shift="prop", geometry="geo", system_metric="sys"
    )
    assert len(fig.axes) == 3
    cap = caption_of(fig)
    for name in ("prop", "geo", "sys"):
        assert name in cap
    assert cap.endswith("MEASURED")


def test_regime_heatmap_hatches_significant_cells(registry):
    long = pd.DataFrame(
        {
            "row": ["typo", "typo", "nick", "nick"],
            "col": ["low", "high", "low", "high"],
            "value": [0.1, -0.3, 0.02, 0.25],
            "ci_excludes_zero": [True, True, False, True],
        }
    )
    registry.register("regime", long, cfg=CFG, tier=TIER)
    fig = regime_heatmap(registry, tier=TIER, artifact="regime")
    hatched = [p for p in fig.axes[0].patches if p.get_hatch()]
    assert len(hatched) == 3  # exactly the ci_excludes_zero cells
    assert "regime" in caption_of(fig)


def test_figures_never_accept_raw_dataframes(registry):
    # the honesty rail is structural: the artifact parameter is a NAME
    with pytest.raises((TypeError, ValueError)):
        line_with_ci(registry, tier=TIER, artifact=xy_frame())


def test_wrong_tier_refused(registry):
    registry.register("curve_a", xy_frame(), cfg=CFG, tier="mid")
    with pytest.raises(Exception, match="[Tt]ier"):
        line_with_ci(registry, tier=TIER, artifact="curve_a")


# -------------------------------------------------------------------- cards


CARD = {
    "card_id": "TRN03_aug",
    "conjecture": "calibrated-noise augmentation raises typo invariance which raises entity-F",
    "pressure": "augmentation kind {none, generic, calibrated}",
    "property": "typo-invariance battery score",
    "metric": "entity-level F at fixed precision 0.995",
    "prediction": "calibrated > generic > none on both property and metric, CIs excluding zero",
}


def test_conjecture_card_registers_immutable_artifact(registry):
    md = conjecture_card(**CARD, registry=registry)
    assert "TRN03_aug" in md and CARD["prediction"] in md
    payload, meta = registry.load("card_TRN03_aug", tier=registry.newest_tier("card_TRN03_aug"))
    assert payload["conjecture"] == CARD["conjecture"]
    assert payload["sha256"] == meta["extra"]["sha256"]
    assert meta["kind"] == "card"


def test_conjecture_card_idempotent_on_identical_content(registry):
    first = conjecture_card(**CARD, registry=registry)
    again = conjecture_card(**CARD, registry=registry)  # re-executed notebook: fine
    assert first == again
    runs = list((registry.root / "card_TRN03_aug").iterdir())
    assert len(runs) == 1  # not re-registered


@pytest.mark.parametrize("field", ["conjecture", "pressure", "property", "metric", "prediction"])
def test_conjecture_card_any_changed_field_raises(registry, field):
    conjecture_card(**CARD, registry=registry)
    changed = dict(CARD, **{field: "quietly edited after the runs"})
    with pytest.raises(CardImmutableError, match=field):
        conjecture_card(**changed, registry=registry)


def test_conjecture_card_field_validation(registry):
    with pytest.raises(ValueError, match="non-empty"):
        conjecture_card(**dict(CARD, prediction=""), registry=registry)


def test_verdict_box_validates_vocabulary_and_card_existence(registry):
    with pytest.raises(ValueError, match="vocabulary is"):
        verdict_box("TRN03_aug", "PROVEN", "n/a", registry)
    with pytest.raises(ArtifactMissing, match="conjecture card"):
        verdict_box("TRN03_aug", "CONFIRMED", "some evidence", registry)
    conjecture_card(**CARD, registry=registry)
    md = verdict_box("TRN03_aug", "UNEXPLAINED", "artifact trn03_summary: both moved", registry)
    assert "UNEXPLAINED" in md and CARD["conjecture"] in md
    with pytest.raises(ValueError, match="evidence"):
        verdict_box("TRN03_aug", "CONFIRMED", "", registry)
