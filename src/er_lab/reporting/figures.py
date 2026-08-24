"""Headline figures that render ONLY from registered artifacts (PLAN §6/§7).

The never-fabricate rail, mechanized: every figure function here takes an
:class:`~er_lab.infra.artifacts.ArtifactRegistry` plus artifact *names* — never
raw dataframes — and loads every input up front, so a missing input raises
:class:`~er_lab.infra.artifacts.ArtifactMissing` (with its 'run notebook X
first' placard) before a single axis exists. A figure you can see is a figure
whose data provably came from a registered run.

Every figure auto-stamps a caption footer::

    artifact <name(s)> | cfg <12-hex config hash> | tier <tier> | MEASURED

with ``EXTRAPOLATED`` replacing ``MEASURED`` when any input's meta
(``extra.basis``) or any plotted row (a ``basis`` column) says so — and
extrapolated segments additionally render dashed, over a shaded span, under a
diagonal ``EXTRAPOLATED`` watermark. The reader can always tell a measurement
from a projection at a glance (notebook 15's visual convention).

Figure vocabulary:

- :func:`line_with_ci` — the workhorse metric-vs-x line with a CI band.
- :func:`three_panel_pressure` — the PLAN §1 mediation triptych: pressure ->
  property shift | embedding geometry | system metric, side by side.
- :func:`regime_heatmap` — HYB-01's noise-regime map; cells where the CI
  excludes zero get hatched significance shading.
- :func:`scaling_curve` — SCL-01's measured-solid / extrapolated-dashed curve.

All functions return the matplotlib Figure. Stamped texts carry stable ``gid``
values (:data:`CAPTION_GID`, :data:`WATERMARK_GID`) so tests and the honesty
audit can find them.
"""

from __future__ import annotations

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from er_lab.infra.artifacts import ArtifactRegistry

__all__ = [
    "line_with_ci",
    "regime_heatmap",
    "scaling_curve",
    "setup_style",
    "three_panel_pressure",
]

CAPTION_GID = "caption-footer"
WATERMARK_GID = "extrapolated-watermark"

MEASURED, EXTRAPOLATED = "MEASURED", "EXTRAPOLATED"


def setup_style() -> None:
    """The lab's figure style — call once per notebook, after imports."""
    matplotlib.rcParams.update(
        {
            "figure.dpi": 110,
            "figure.autolayout": True,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "legend.fontsize": 9,
            "legend.frameon": False,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
        }
    )


# -- loading + stamping (shared rails) ---------------------------------------


def _load_frame(registry: ArtifactRegistry, name: str, *, tier: str) -> tuple[pd.DataFrame, dict]:
    """Load one artifact, insisting on a DataFrame payload.

    ArtifactMissing / TierMixingError propagate untouched — the refusal IS the
    feature; nothing here catches them to 'render anyway'.
    """
    payload, meta = registry.load(name, tier=tier)
    if not isinstance(payload, pd.DataFrame):
        raise TypeError(
            f"artifact '{name}' holds {type(payload).__name__}, not a DataFrame — "
            "headline figures render tables only"
        )
    return payload, meta


def _meta_extrapolated(meta: dict) -> bool:
    return str(meta.get("extra", {}).get("basis", "")).upper() == EXTRAPOLATED


def _frame_extrapolated(df: pd.DataFrame, basis: str | None) -> bool:
    return (
        basis is not None
        and basis in df.columns
        and df[basis].astype(str).str.upper().eq(EXTRAPOLATED).any()
    )


def _stamp(
    fig: plt.Figure, names: list[str], metas: list[dict], tier: str, extrapolated: bool
) -> None:
    """The caption footer: artifact provenance no figure ships without."""
    hashes = list(dict.fromkeys(m["config_hash"] for m in metas))  # unique, order kept
    text = (
        f"artifact {', '.join(names)} | cfg {','.join(hashes)} | tier {tier} | "
        f"{EXTRAPOLATED if extrapolated else MEASURED}"
    )
    fig.text(0.01, 0.005, text, fontsize=7, color="0.35", ha="left", va="bottom", gid=CAPTION_GID)


def _watermark(fig: plt.Figure) -> None:
    fig.text(
        0.5,
        0.5,
        EXTRAPOLATED,
        fontsize=40,
        color="0.4",
        alpha=0.18,
        ha="center",
        va="center",
        rotation=30,
        zorder=5,
        gid=WATERMARK_GID,
    )


def _check_columns(df: pd.DataFrame, name: str, cols: list[str]) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise KeyError(f"artifact '{name}' is missing plot columns {missing}")


def _draw_series(
    ax: plt.Axes,
    g: pd.DataFrame,
    *,
    x: str,
    y: str,
    lo: str | None,
    hi: str | None,
    basis: str | None,
    meta_extrapolated: bool,
    label: str | None = None,
) -> bool:
    """One line: measured part solid, extrapolated part dashed over a shaded span.

    Returns whether anything extrapolated was drawn. The extrapolated segment
    starts at the last measured point so the line stays visually continuous.
    """
    g = g.sort_values(x)
    if basis is not None and basis in g.columns:
        mask = g[basis].astype(str).str.upper().eq(EXTRAPOLATED)
    else:
        mask = pd.Series(meta_extrapolated, index=g.index)
    measured, extrap = g[~mask], g[mask]
    has_band = lo is not None and hi is not None and lo in g.columns and hi in g.columns

    color = None
    if len(measured):
        (line,) = ax.plot(measured[x], measured[y], "-", label=label)
        color = line.get_color()
        if has_band:
            ax.fill_between(
                measured[x], measured[lo], measured[hi], alpha=0.2, color=color, linewidth=0
            )
    if len(extrap):
        seg = pd.concat([measured.tail(1), extrap]) if len(measured) else extrap
        (line,) = ax.plot(seg[x], seg[y], "--", color=color, label=label if color is None else None)
        color = color or line.get_color()
        if has_band:
            ax.fill_between(seg[x], seg[lo], seg[hi], alpha=0.12, color=color, linewidth=0)
        ax.axvspan(
            float(extrap[x].min()), float(extrap[x].max()), color="0.85", alpha=0.4, zorder=0
        )
    return bool(len(extrap))


def _groups(df: pd.DataFrame, hue: str | None):
    if hue is None:
        return [(None, df)]
    return [(str(k), g) for k, g in df.groupby(hue, sort=True)]


# -- the figure vocabulary ---------------------------------------------------


def line_with_ci(
    registry: ArtifactRegistry,
    *,
    tier: str,
    artifact: str,
    x: str = "x",
    y: str = "y",
    lo: str = "lo",
    hi: str = "hi",
    hue: str | None = None,
    basis: str = "basis",
    title: str | None = None,
    xlabel: str | None = None,
    ylabel: str | None = None,
    logx: bool = False,
    logy: bool = False,
    figsize: tuple[float, float] = (6.0, 4.0),
) -> plt.Figure:
    """Metric-vs-x line(s) with a CI band, one line per *hue* group.

    The artifact must hold a DataFrame with columns *x* and *y*; *lo*/*hi*
    band columns and a *basis* column are optional (see module docstring for
    what a ``basis`` of EXTRAPOLATED triggers).
    """
    df, meta = _load_frame(registry, artifact, tier=tier)
    _check_columns(df, artifact, [x, y] + ([hue] if hue else []))
    meta_ex = _meta_extrapolated(meta)
    fig, ax = plt.subplots(figsize=figsize)
    any_extrap = False
    for key, g in _groups(df, hue):
        any_extrap |= _draw_series(
            ax, g, x=x, y=y, lo=lo, hi=hi, basis=basis, meta_extrapolated=meta_ex, label=key
        )
    if logx:
        ax.set_xscale("log")
    if logy:
        ax.set_yscale("log")
    ax.set_xlabel(xlabel or x)
    ax.set_ylabel(ylabel or y)
    if title:
        ax.set_title(title)
    if hue is not None:
        ax.legend(title=hue)
    extrapolated = any_extrap or meta_ex
    if extrapolated:
        _watermark(fig)
    _stamp(fig, [artifact], [meta], tier, extrapolated)
    return fig


def three_panel_pressure(
    registry: ArtifactRegistry,
    *,
    tier: str,
    property_shift: str,
    geometry: str,
    system_metric: str,
    x: str = "x",
    y: str = "y",
    lo: str = "lo",
    hi: str = "hi",
    basis: str = "basis",
    group: str = "group",
    titles: tuple[str, str, str] = ("property shift", "embedding geometry", "system metric"),
    suptitle: str | None = None,
    figsize: tuple[float, float] = (12.0, 4.0),
) -> plt.Figure:
    """The mediation triptych: pressure -> property | geometry | system metric.

    PLAN §1 commitment 1 made visual: every training-design claim shows its
    full causal chain side by side — the property the pressure was supposed to
    move (left, line+CI over the pressure dial), what the embedding space
    looks like (middle, 2-d scatter, colored by *group* when present), and the
    system metric that is supposed to follow (right, line+CI). All three
    artifacts are loaded before any drawing — one missing link refuses the
    whole triptych, because a partial chain is exactly the dishonest figure.

    Column contract: *property_shift* and *system_metric* frames need
    *x*/*y* (+ optional *lo*/*hi*, *basis*); the *geometry* frame needs
    *x*/*y* point coordinates (+ optional *group*, *basis* — an EXTRAPOLATED
    row in ANY of the three frames marks the whole triptych EXTRAPOLATED).
    """
    df_prop, meta_prop = _load_frame(registry, property_shift, tier=tier)
    df_geo, meta_geo = _load_frame(registry, geometry, tier=tier)
    df_sys, meta_sys = _load_frame(registry, system_metric, tier=tier)
    _check_columns(df_prop, property_shift, [x, y])
    _check_columns(df_geo, geometry, [x, y])
    _check_columns(df_sys, system_metric, [x, y])
    metas = [meta_prop, meta_geo, meta_sys]
    names = [property_shift, geometry, system_metric]

    fig, axes = plt.subplots(1, 3, figsize=figsize)
    any_extrap = _draw_series(
        axes[0],
        df_prop,
        x=x,
        y=y,
        lo=lo,
        hi=hi,
        basis=basis,
        meta_extrapolated=_meta_extrapolated(meta_prop),
    )
    for key, g in _groups(df_geo, group if group in df_geo.columns else None):
        axes[1].scatter(g[x], g[y], s=8, alpha=0.6, label=key)
    if group in df_geo.columns:
        axes[1].legend(title=group, markerscale=1.5)
    any_extrap |= _draw_series(
        axes[2],
        df_sys,
        x=x,
        y=y,
        lo=lo,
        hi=hi,
        basis=basis,
        meta_extrapolated=_meta_extrapolated(meta_sys),
    )
    for ax, panel_title in zip(axes, titles):
        ax.set_title(panel_title)
        ax.set_xlabel(x)
    axes[0].set_ylabel(y)
    if suptitle:
        fig.suptitle(suptitle)

    extrapolated = (
        any_extrap
        # every loaded frame's basis rows count — including the geometry
        # scatter, which has no dashed-line vocabulary of its own
        or any(_frame_extrapolated(df, basis) for df in (df_prop, df_geo, df_sys))
        or any(_meta_extrapolated(m) for m in metas)
    )
    if extrapolated:
        _watermark(fig)
    _stamp(fig, names, metas, tier, extrapolated)
    return fig


def regime_heatmap(
    registry: ArtifactRegistry,
    *,
    tier: str,
    artifact: str,
    row: str = "row",
    col: str = "col",
    value: str = "value",
    sig: str | None = "ci_excludes_zero",
    basis: str | None = "basis",
    cmap: str = "RdBu_r",
    title: str | None = None,
    figsize: tuple[float, float] = (6.5, 5.0),
) -> plt.Figure:
    """HYB-01's noise-regime map: a diverging matrix with hatched significance.

    The artifact holds a LONG-form DataFrame — one row per (*row*, *col*) cell
    with the effect in *value* and a boolean *sig* column ('does the CI
    exclude zero'). Cells where *sig* is True get ``///`` hatching: hatched =
    an effect the protocol lets you believe; unhatched color = noise-level.
    Color scale is symmetric about zero (it is an effect map). Pass
    ``sig=None`` only for exploratory maps that carry no CIs at all. A *basis*
    column with any EXTRAPOLATED row (or meta basis) stamps + watermarks the
    whole map EXTRAPOLATED — a heatmap has no dashed-line vocabulary to mark
    single cells.
    """
    df, meta = _load_frame(registry, artifact, tier=tier)
    needed = [row, col, value] + ([sig] if sig is not None else [])
    _check_columns(df, artifact, needed)
    matrix = df.pivot(index=row, columns=col, values=value).sort_index(axis=0).sort_index(axis=1)
    abs_values = np.abs(matrix.to_numpy())
    vmax = float(np.nanmax(abs_values)) if np.isfinite(abs_values).any() else float("nan")
    if not np.isfinite(vmax) or vmax == 0:
        vmax = 1.0

    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(matrix.to_numpy(), cmap=cmap, vmin=-vmax, vmax=vmax, aspect="auto")
    ax.set_xticks(range(matrix.shape[1]), [str(c) for c in matrix.columns], rotation=45, ha="right")
    ax.set_yticks(range(matrix.shape[0]), [str(r) for r in matrix.index])
    ax.set_xlabel(col)
    ax.set_ylabel(row)
    fig.colorbar(im, ax=ax, label=value)
    if sig is not None:
        sig_matrix = df.pivot(index=row, columns=col, values=sig).reindex(
            index=matrix.index, columns=matrix.columns
        )
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                if bool(sig_matrix.iloc[i, j]):
                    ax.add_patch(
                        plt.Rectangle(
                            (j - 0.5, i - 0.5),
                            1,
                            1,
                            fill=False,
                            hatch="///",
                            edgecolor="0.2",
                            linewidth=0,
                        )
                    )
    if title:
        ax.set_title(title)
    extrapolated = _meta_extrapolated(meta) or _frame_extrapolated(df, basis)
    if extrapolated:
        _watermark(fig)
    _stamp(fig, [artifact], [meta], tier, extrapolated)
    return fig


def scaling_curve(
    registry: ArtifactRegistry,
    *,
    tier: str,
    artifact: str,
    x: str = "n",
    y: str = "y",
    lo: str = "lo",
    hi: str = "hi",
    basis: str = "basis",
    hue: str | None = None,
    title: str | None = None,
    ylabel: str | None = None,
    loglog: bool = True,
    figsize: tuple[float, float] = (6.5, 4.5),
) -> plt.Figure:
    """SCL-01's error-vs-n curve: measured solid, extrapolated dashed + watermarked.

    The artifact frame carries one row per ladder rung: *x* (record count),
    *y* (+ optional *lo*/*hi* prediction band), and a *basis* column labeling
    each rung MEASURED or EXTRAPOLATED — typically written by the notebook
    from :func:`er_lab.scale.curves.fit_scaling` predictions past the ladder,
    only after :func:`~er_lab.scale.curves.validate_extrapolation` passed.
    """
    df, meta = _load_frame(registry, artifact, tier=tier)
    _check_columns(df, artifact, [x, y] + ([hue] if hue else []))
    meta_ex = _meta_extrapolated(meta)
    fig, ax = plt.subplots(figsize=figsize)
    any_extrap = False
    for key, g in _groups(df, hue):
        any_extrap |= _draw_series(
            ax, g, x=x, y=y, lo=lo, hi=hi, basis=basis, meta_extrapolated=meta_ex, label=key
        )
    if loglog:
        ax.set_xscale("log")
        ax.set_yscale("log")
    ax.set_xlabel("records (n)")
    ax.set_ylabel(ylabel or y)
    if title:
        ax.set_title(title)
    if hue is not None:
        ax.legend(title=hue)
    extrapolated = any_extrap or meta_ex
    if extrapolated:
        _watermark(fig)
    _stamp(fig, [artifact], [meta], tier, extrapolated)
    return fig
