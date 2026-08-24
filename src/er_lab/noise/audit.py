"""Noise taxonomy over same-entity pair diffs (PLAN.md §3 NSE-01/NSE-02; notebook 04).

Classifies field-level differences between two versions of the same entity —
``er_lab.data.nc.align_pair`` output: a key column plus ``<field>_a``/
``<field>_b`` — into a fixed category taxonomy, then summarizes prevalence
with Wilson 95% intervals. These measured distributions calibrate the
generator (NSE-01) and score generator fidelity (NSE-02); to keep that
scoring honest, this module is deliberately independent of the generator
(no import of the channel implementations).

Categories, in decision order per (pair, field):

- ``identical``: verbatim-equal, or missing on both sides
- ``missing_gain`` / ``missing_loss``: missing (NA or blank) on exactly one side
- ``swap``: the value pair is exchanged with another name field of the same record
- ``format_drift``: same alphanumeric skeleton — only punctuation/spacing/case differ
- ``nickname``: given_name pair related by the nickname lexicon
- ``typo``: case-insensitive Levenshtein distance <= 2
- ``wholesale``: everything else
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence

import jellyfish
import numpy as np
import pandas as pd
from scipy.stats import norm

CATEGORIES = (
    "identical",
    "missing_gain",
    "missing_loss",
    "typo",
    "nickname",
    "swap",
    "format_drift",
    "wholesale",
)

#: Fields eligible for the ``swap`` category (values exchanged within a record).
SWAP_FIELDS = ("given_name", "middle_name", "family_name")

_TYPO_MAX_EDITS = 2
_Z95 = float(norm.ppf(0.975))  # Wilson 95%


def _is_missing(s: pd.Series) -> np.ndarray:
    """NA or whitespace-only counts as missing (NC snapshots use '' for NULL)."""
    return (s.isna() | (s.str.strip() == "").fillna(False)).to_numpy(dtype=bool)


def _eq(a: pd.Series, b: pd.Series) -> np.ndarray:
    """Verbatim equality; NA on either side compares False."""
    return (a == b).fillna(False).to_numpy(dtype=bool)


def _skeleton(s: pd.Series) -> pd.Series:
    """Lower-cased with every non-alphanumeric character removed."""
    return s.str.lower().str.replace(r"[\W_]+", "", regex=True)


def _classify_field(
    field: str,
    sides: dict[str, tuple[pd.Series, pd.Series]],
    lexicon: Mapping[str, set[str]] | None,
    swap_fields: Sequence[str],
) -> np.ndarray:
    a, b = sides[field]
    miss_a, miss_b = _is_missing(a), _is_missing(b)
    present = ~miss_a & ~miss_b

    cat = np.full(len(a), "wholesale", dtype=object)
    cat[(miss_a & miss_b) | (present & _eq(a, b))] = "identical"
    cat[miss_a & ~miss_b] = "missing_gain"
    cat[~miss_a & miss_b] = "missing_loss"
    rest = present & ~_eq(a, b)

    if field in swap_fields:
        swap = np.zeros(len(a), dtype=bool)
        for other in swap_fields:
            if other == field or other not in sides:
                continue
            oa, ob = sides[other]
            swap |= rest & _eq(a, ob) & _eq(oa, b) & ~_is_missing(oa) & ~_is_missing(ob)
        cat[swap] = "swap"
        rest &= ~swap

    drift = rest & _eq(_skeleton(a), _skeleton(b))
    cat[drift] = "format_drift"
    rest &= ~drift

    lower_a = a.str.lower().tolist()
    lower_b = b.str.lower().tolist()
    if field == "given_name" and lexicon is not None:
        nick = np.fromiter(
            (
                bool(r)
                and (
                    y.strip() in lexicon.get(x.strip(), ())
                    or x.strip() in lexicon.get(y.strip(), ())
                )
                for r, x, y in zip(rest, lower_a, lower_b)
            ),
            dtype=bool,
            count=len(a),
        )
        cat[nick] = "nickname"
        rest &= ~nick

    typo = np.fromiter(
        (
            bool(r) and jellyfish.levenshtein_distance(x, y) <= _TYPO_MAX_EDITS
            for r, x, y in zip(rest, lower_a, lower_b)
        ),
        dtype=bool,
        count=len(a),
    )
    cat[typo] = "typo"
    return cat  # anything still uncaught keeps 'wholesale'


def classify_pair_diffs(
    aligned_df: pd.DataFrame,
    *,
    fields: Sequence[str],
    lexicon: Mapping[str, set[str]] | None = None,
    key: str = "ncid",
    swap_fields: Sequence[str] = SWAP_FIELDS,
) -> pd.DataFrame:
    """Classify each (pair, field) diff -> long DataFrame [key, field, category].

    ``aligned_df`` is ``er_lab.data.nc.align_pair`` output: a ``key`` column
    plus ``<field>_a``/``<field>_b`` for every field. ``lexicon`` maps
    lower-cased canonical given names to sets of lower-cased variants; without
    it the nickname category never fires. Swap detection compares each name
    field against the other ``swap_fields`` whose columns are present.

    Row order is part of the contract: one block per field in ``fields``
    order, each block in ``aligned_df`` row order.
    """
    if not fields:
        raise ValueError("fields must be non-empty")
    if key not in aligned_df.columns:
        raise KeyError(f"aligned frame has no key column {key!r}")
    wanted = list(dict.fromkeys([*fields, *(f for f in swap_fields if f not in fields)]))
    sides: dict[str, tuple[pd.Series, pd.Series]] = {}
    for f in wanted:
        if f"{f}_a" in aligned_df.columns and f"{f}_b" in aligned_df.columns:
            sides[f] = (
                aligned_df[f"{f}_a"].astype("string"),
                aligned_df[f"{f}_b"].astype("string"),
            )
        elif f in fields:
            raise KeyError(f"aligned frame lacks columns {f}_a/{f}_b for field {f!r}")

    keys = aligned_df[key].astype("string").tolist()
    blocks = [
        pd.DataFrame(
            {"key": keys, "field": f, "category": _classify_field(f, sides, lexicon, swap_fields)}
        )
        for f in fields
    ]
    long_df = pd.concat(blocks, ignore_index=True)
    return long_df.astype({"key": "string", "field": "string", "category": "string"})


def _wilson(k: np.ndarray, n: np.ndarray, z: float = _Z95) -> tuple[np.ndarray, np.ndarray]:
    """Wilson score interval for k successes in n trials (vectorized)."""
    p = k / n
    denom = 1.0 + z * z / n
    center = (p + z * z / (2.0 * n)) / denom
    half = z * np.sqrt(p * (1.0 - p) / n + z * z / (4.0 * n * n)) / denom
    return np.clip(center - half, 0.0, 1.0), np.clip(center + half, 0.0, 1.0)


def prevalence(long_df: pd.DataFrame, *, by: list[str] | None = None) -> pd.DataFrame:
    """Category prevalence per field (and optional group columns) with Wilson 95% CIs.

    Returns [field, category, *by, n, rate, ci_low, ci_high]: ``n`` is the
    number of pairs in that category, ``rate`` = n / (pairs for that field
    within the same ``by`` cell). Only observed categories appear, so grouped
    ``n`` sums to the ungrouped ``n`` per (field, category).
    """
    by = list(by) if by else []
    missing = [c for c in ("field", "category", *by) if c not in long_df.columns]
    if missing:
        raise KeyError(f"long frame lacks columns: {missing}")
    denom_cols = ["field", *by]
    totals = long_df.groupby(denom_cols, dropna=False, observed=True).size().rename("total")
    out = (
        long_df.groupby([*denom_cols, "category"], dropna=False, observed=True)
        .size()
        .rename("n")
        .reset_index()
        .merge(totals.reset_index(), on=denom_cols, how="left")
    )
    k = out["n"].to_numpy(dtype=float)
    n = out["total"].to_numpy(dtype=float)
    out["rate"] = k / n
    out["ci_low"], out["ci_high"] = _wilson(k, n)
    out = out[["field", "category", *by, "n", "rate", "ci_low", "ci_high"]]
    return out.sort_values(["field", "category", *by], kind="stable", ignore_index=True)


def audit_report(
    aligned_df: pd.DataFrame,
    fields: Sequence[str],
    *,
    lexicon: Mapping[str, set[str]] | None = None,
    by: Iterable[str] | None = None,
    key: str = "ncid",
    swap_fields: Sequence[str] = SWAP_FIELDS,
) -> pd.DataFrame:
    """Classify then summarize: the prevalence table for an aligned snapshot pair.

    ``by`` names ``aligned_df`` columns (e.g. ``'race_code_a'``) attached to
    every pair as grouping columns for a per-group breakdown — the input to
    ``ExposureModel.fit_from_prevalence``.
    """
    long_df = classify_pair_diffs(
        aligned_df, fields=fields, lexicon=lexicon, key=key, swap_fields=swap_fields
    )
    by = list(by) if by else []
    missing = [c for c in by if c not in aligned_df.columns]
    if missing:
        raise KeyError(f"aligned frame lacks by columns: {missing}")
    for col in by:  # classify emits len(fields) blocks, each in aligned row order
        long_df[col] = np.tile(aligned_df[col].astype("string").to_numpy(), len(fields))
    return prevalence(long_df, by=by or None)
