"""Identifiability accounting: the in-principle unresolvable fraction (PLAN.md §3 MET-06, §5).

Some records cannot be resolved by ANY method on a given field set: twins sharing
name/DOB/address, distinct people whose recorded fields collide exactly (Johndrow,
Lum & Dunson, Biometrika 2018 prove accurate ER is impossible unless entities are
few or well separated — lit_review §4). Every headline figure reports this
fraction, and metrics may additionally be *conditioned* on the resolvable subset —
never silently, always alongside the unconditioned number.

A record is *unresolvable on ``fields``* when its exact tuple of values on those
fields is also carried by a record of a DIFFERENT entity. This is deliberately
conservative-by-exactness: near-collisions (one character off) do not count, and
missing values compare equal to missing values (two records that both lack a DOB
present identical information to any matcher restricted to these fields).
"""

from __future__ import annotations

from collections.abc import Sequence

import pandas as pd
from pandas.api.types import is_bool_dtype

__all__ = ["conditioned", "unresolvable_mask", "unresolvable_report"]


def _validate_frame(df: pd.DataFrame, fields: Sequence[str]) -> list[str]:
    fields = list(fields)
    if not fields:
        raise ValueError("fields must name at least one column")
    missing = [c for c in ["record_id", "entity_id", *fields] if c not in df.columns]
    if missing:
        raise KeyError(f"columns missing from frame: {missing}")
    if df["entity_id"].isna().any():
        raise ValueError(
            "entity_id has missing values; unresolvable accounting needs full truth "
            "(filter to truth-labeled records first)"
        )
    if df["record_id"].duplicated().any():
        raise ValueError("record_id values must be unique")
    return fields


def unresolvable_mask(df_canonical: pd.DataFrame, *, fields: Sequence[str]) -> pd.Series:
    """Boolean Series (indexed by record_id): True where the record is unresolvable.

    A record is flagged when its exact value tuple on ``fields`` is shared with at
    least one record of a *different* entity — in-principle unresolvable on those
    fields (MET-06). Missing values compare equal to missing values (see module
    docstring). Records sharing a tuple only within one entity (ordinary
    duplicates) are NOT flagged. Requires ``record_id`` (unique) and ``entity_id``
    (non-missing) columns.
    """
    fields = _validate_frame(df_canonical, fields)
    n_entities = df_canonical.groupby(fields, dropna=False)["entity_id"].transform("nunique")
    return pd.Series(
        (n_entities > 1).to_numpy(),
        index=pd.Index(df_canonical["record_id"].astype(str), name="record_id"),
        name="unresolvable",
        dtype=bool,
    )


def unresolvable_report(
    df: pd.DataFrame,
    fields: Sequence[str],
    *,
    by: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Unresolvable counts and rates, overall and per group.

    Returns a DataFrame with columns ``[*by, "n", "n_unresolvable", "rate"]``.
    The first row is the overall figure (its ``by`` columns are ``<NA>``);
    subsequent rows are one per group value combination (missing group values form
    their own group), sorted by group key. ``rate = n_unresolvable / n``.
    Unresolvability is computed on the FULL frame first, then tabulated per group
    — a record's collision partners are searched corpus-wide, not group-wide.
    """
    by = list(by or [])
    missing = [c for c in by if c not in df.columns]
    if missing:
        raise KeyError(f"'by' columns missing from frame: {missing}")
    mask = unresolvable_mask(df, fields=fields)  # aligned to df row order
    flags = mask.to_numpy()

    rows: list[dict] = [
        {
            **{c: pd.NA for c in by},
            "n": len(df),
            "n_unresolvable": int(flags.sum()),
            "rate": float(flags.sum() / len(df)) if len(df) else float("nan"),
        }
    ]
    if by:
        tmp = df[by].copy()
        tmp["_unresolvable"] = flags
        grouped = tmp.groupby(by, dropna=False, sort=True)["_unresolvable"].agg(["size", "sum"])
        for key, rec in grouped.iterrows():
            key_t = key if isinstance(key, tuple) else (key,)
            rows.append(
                {
                    **dict(zip(by, key_t)),
                    "n": int(rec["size"]),
                    "n_unresolvable": int(rec["sum"]),
                    "rate": float(rec["sum"] / rec["size"]),
                }
            )
    return pd.DataFrame(rows, columns=[*by, "n", "n_unresolvable", "rate"])


def conditioned(
    pred: pd.Series,
    truth: pd.Series,
    mask: pd.Series,
) -> tuple[pd.Series, pd.Series]:
    """Drop unresolvable records from a (pred, truth) clustering pair.

    ``pred`` and ``truth`` are cluster-label Series indexed by record_id and must
    cover the same records (raises on mismatch, per the metric contract);
    ``mask`` is a boolean Series indexed by record_id (from ``unresolvable_mask``)
    that must cover every evaluated record (a superset is fine). Returns the
    subset pair with mask-True records removed, order preserved. Metrics computed
    on this pair are *conditioned on resolvability* — report them alongside, never
    instead of, the unconditioned figures (PLAN §5).
    """
    if pred.index.has_duplicates or truth.index.has_duplicates:
        raise ValueError("pred/truth indexes must not contain duplicate record ids")
    if set(pred.index) != set(truth.index):
        raise ValueError("pred and truth must be indexed by the same record ids")
    if not is_bool_dtype(mask):
        raise TypeError(f"mask must be boolean, got dtype {mask.dtype}")
    if mask.index.has_duplicates:
        raise ValueError("mask index must not contain duplicate record ids")
    uncovered = pred.index.difference(mask.index)
    if len(uncovered):
        raise KeyError(
            f"mask does not cover {len(uncovered)} evaluated record ids "
            f"(e.g. {list(uncovered[:5])})"
        )
    keep = pred.index[~mask.reindex(pred.index).to_numpy()]
    return pred.loc[keep], truth.loc[keep]
