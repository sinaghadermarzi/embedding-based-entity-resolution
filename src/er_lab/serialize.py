"""Record -> text serialization for embedding models (TRN-05; notebook 08).

How a person record becomes the string an encoder sees is an experimental
factor, not a convenience: TRN-05 compares four schemes and two missing-field
treatments, and probes field-order sensitivity at eval time. This module is the
single place that mapping happens, so every arm of the matrix serializes through
the same code path.

Schemes (exact formats, so tests can assert known answers):

- ``colval``   -> ``[COL] role [VAL] value [COL] role [VAL] value ...``
                  (the Ditto-style marked format; single spaces between tokens)
- ``template`` -> ``role: value | role: value | ...``
- ``json``     -> compact JSON object, keys in field order, no whitespace
                  (``{"given_name":"ANA","zip":"27510"}``)
- ``bare``     -> values only, space-joined (no role names at all)

Missing treatment: a cell is *missing* when it is NA (``pd.NA``/``None``/NaN)
or the empty string — the same definition ``er_lab.data.schema`` and
``er_lab.noise.channels`` use. ``missing='token'`` substitutes the literal
``[MISSING]`` token as the value (the field stays visible); ``missing='drop'``
omits the field entirely.

Field order defaults to ``text_roles`` order (the declared-schema yaml order);
``field_order`` must be a permutation of ``text_roles`` — the TRN-05 order
probe shuffles the same fields, it never adds or removes any.
"""

from __future__ import annotations

import json
from collections.abc import Mapping

import pandas as pd

__all__ = ["MISSING_TOKEN", "SCHEMES", "serialize_frame", "serialize_record"]

MISSING_TOKEN = "[MISSING]"
SCHEMES = ("colval", "template", "json", "bare")


def _validated_fields(
    text_roles: list[str], field_order: list[str] | None, scheme: str, missing: str
) -> list[str]:
    """Validate scheme/missing/field_order once; return the field list to serialize."""
    if scheme not in SCHEMES:
        raise ValueError(f"unknown scheme {scheme!r}: expected one of {SCHEMES}")
    if missing not in ("token", "drop"):
        raise ValueError(f"unknown missing treatment {missing!r}: expected 'token' or 'drop'")
    if field_order is None:
        return list(text_roles)
    if sorted(field_order) != sorted(text_roles):
        raise ValueError(
            f"field_order must be a permutation of text_roles: got {field_order} "
            f"vs text_roles {text_roles}"
        )
    return list(field_order)


def _render(items: list[tuple[str, str]], scheme: str) -> str:
    if scheme == "colval":
        return " ".join(f"[COL] {role} [VAL] {value}" for role, value in items)
    if scheme == "template":
        return " | ".join(f"{role}: {value}" for role, value in items)
    if scheme == "json":
        return json.dumps(dict(items), separators=(",", ":"), ensure_ascii=False)
    return " ".join(value for _, value in items)  # bare


def serialize_record(
    row: Mapping,
    *,
    text_roles: list[str],
    scheme: str = "colval",
    missing: str = "token",
    field_order: list[str] | None = None,
) -> str:
    """Serialize one record (a mapping of role -> cell value) to its text form.

    A role absent from ``row`` is treated exactly like an NA cell. Values are
    rendered verbatim via ``str`` — no cleaning, casing, or trimming (upstream
    standardization is PRS-01's factor, never a serialization side effect).
    """
    fields = _validated_fields(text_roles, field_order, scheme, missing)
    items: list[tuple[str, str]] = []
    for role in fields:
        value = row.get(role)
        if value is None or pd.isna(value) or value == "":
            if missing == "drop":
                continue
            items.append((role, MISSING_TOKEN))
        else:
            items.append((role, str(value)))
    return _render(items, scheme)


def serialize_frame(
    df: pd.DataFrame,
    *,
    text_roles: list[str],
    scheme: str = "colval",
    missing: str = "token",
    field_order: list[str] | None = None,
) -> pd.Series:
    """Serialize every row of a canonical frame; returns Series[str] aligned to df.index.

    Same contract as :func:`serialize_record`, vectorized over the frame. A
    ``text_roles`` entry with no matching column serializes as missing in every
    row (some corpora simply lack a role).
    """
    fields = _validated_fields(text_roles, field_order, scheme, missing)
    present = [c for c in fields if c in df.columns]
    records = df[present].to_dict("records") if present else [{} for _ in range(len(df))]
    texts = [
        serialize_record(
            rec, text_roles=text_roles, scheme=scheme, missing=missing, field_order=field_order
        )
        for rec in records
    ]
    return pd.Series(texts, index=df.index, dtype="string")
