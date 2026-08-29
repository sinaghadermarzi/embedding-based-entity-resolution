"""Multi-pass deterministic matchkey blocking — the rules arm of BAS-02 (PLAN §3).

Settled large-scale practice (lit review: Census BigMatch ~10 passes, ONS 29
matchkeys) is an OR-union of several cheap deterministic "passes": each pass
derives a compound key per record (e.g. soundex of the family name + birth
year) and pairs up all records that share a key. This module reproduces that
design at lab scale with one honest twist: an optional per-record candidate
*budget* so matchkeys can be compared to BM25 (:mod:`er_lab.blocking.sparse`)
and ANN (:mod:`er_lab.blocking.ann`) at a *matched* budget k — the BAS-02
protocol requirement.

Key grammar (a pass is a list of these; the pass key is their tuple):

- ``"role"``            — the verbatim column value,
- ``"soundex(role)"``   — jellyfish soundex of the stripped value,
- ``"initial(role)"``   — first non-space character, uppercased,
- ``"year(role)"``      — first 4-digit run in the value (birth year from a dob).

A record whose key component is missing (NA, empty, or whitespace-only — NC
pads blanks with spaces) simply does not participate in that pass; values
themselves are never cleaned or modified (PRS-01 owns standardization).
"""

from __future__ import annotations

import re

import jellyfish
import pandas as pd

__all__ = ["candidates", "default_passes"]

_FUNC_RE = re.compile(r"^(?P<func>[a-z_]+)\((?P<col>[A-Za-z0-9_]+)\)$")
_YEAR_RE = re.compile(r"(\d{4})")


def _soundex(value: str) -> str | None:
    code = jellyfish.soundex(value.strip())
    return code or None


def _initial(value: str) -> str | None:
    stripped = value.strip()
    return stripped[0].upper() if stripped else None


def _year(value: str) -> str | None:
    match = _YEAR_RE.search(value)
    return match.group(1) if match else None


_KEY_FUNCS = {"soundex": _soundex, "initial": _initial, "year": _year}


def _key_series(df: pd.DataFrame, spec: str) -> pd.Series:
    """Evaluate one key spec into a 'string' Series (NA = record sits this pass out)."""
    match = _FUNC_RE.match(spec)
    if match:
        func_name, col = match.group("func"), match.group("col")
        if func_name not in _KEY_FUNCS:
            raise ValueError(
                f"unknown matchkey function {func_name!r} in {spec!r}; "
                f"expected one of {sorted(_KEY_FUNCS)}"
            )
        func = _KEY_FUNCS[func_name]
    else:
        col, func = spec, None
    if col not in df.columns:
        raise KeyError(f"matchkey spec {spec!r}: column {col!r} not in frame")
    values = df[col].astype("string")
    out = []
    for v in values:
        if pd.isna(v) or not str(v).strip():  # blank/padded fields carry no key
            out.append(None)
            continue
        out.append(func(str(v)) if func else str(v))
    return pd.Series(out, index=df.index, dtype="string")


def default_passes(df: pd.DataFrame) -> list[list[str]]:
    """The default OR-union passes, chosen from the role columns actually present.

    1. soundex(family_name) + a birth-ish field (year(dob) > birth_year > age;
       initial(given_name) when no birth field exists, to keep the block sizes sane),
    2. zip + initial(given_name),
    3. phone (exact).

    Passes whose columns are absent are skipped; an empty result raises.
    """
    passes: list[list[str]] = []
    fam = "family_name" if "family_name" in df.columns else (
        "full_name" if "full_name" in df.columns else None
    )
    if fam is not None:
        if "dob" in df.columns:
            passes.append([f"soundex({fam})", "year(dob)"])
        elif "birth_year" in df.columns:
            passes.append([f"soundex({fam})", "birth_year"])
        elif "age" in df.columns:
            passes.append([f"soundex({fam})", "age"])
        elif "given_name" in df.columns:
            passes.append([f"soundex({fam})", "initial(given_name)"])
        else:
            passes.append([f"soundex({fam})"])
    if "zip" in df.columns and "given_name" in df.columns:
        passes.append(["zip", "initial(given_name)"])
    if "phone" in df.columns:
        passes.append(["phone"])
    if not passes:
        raise ValueError(
            "default_passes: no usable blocking columns found "
            "(need family_name/full_name, zip+given_name, or phone) — pass explicit passes="
        )
    return passes


def candidates(
    df: pd.DataFrame,
    *,
    passes: list[list[str]] | None = None,
    budget_k: int | None = None,
) -> pd.DataFrame:
    """OR-union multi-pass matchkey candidates as ``DataFrame[a, b]``, a<b, deduped.

    Within each block, pairs are generated exhaustively when ``budget_k`` is
    None; with a budget, each record pairs only with its next ``budget_k``
    neighbors in sorted-id order (sorted-neighborhood-within-block), which
    bounds a block of size s to s*k pairs instead of s². After the pass union,
    a global per-record cap of ``budget_k`` is enforced in generation order
    (pass order, then block key, then id order) so the *candidate budget per
    record* is comparable with sparse/ann blocking at the same k.

    ``.attrs['stats']`` records per-pass block counts, pairs generated, pairs
    newly contributed to the union, and the number dropped by the budget —
    the raw material for pair-completeness/reduction-ratio reporting (PLAN §5).
    """
    if "record_id" not in df.columns:
        raise KeyError("candidates: frame has no 'record_id' column")
    rid = df["record_id"].astype("string")
    if rid.isna().any() or not rid.is_unique:
        raise ValueError("candidates: record_id must be unique and non-null")
    if budget_k is not None and budget_k < 1:
        raise ValueError(f"budget_k must be >= 1 or None, got {budget_k}")
    if passes is None:
        passes = default_passes(df)

    seen: set[tuple[str, str]] = set()
    ordered: list[tuple[str, str]] = []  # union in deterministic generation order
    pass_stats: list[dict] = []
    for keys in passes:
        key_cols = [_key_series(df, spec) for spec in keys]
        frame = pd.DataFrame({f"k{i}": s for i, s in enumerate(key_cols)})
        frame["rid"] = rid.to_numpy()
        frame = frame.dropna(subset=[f"k{i}" for i in range(len(keys))])
        generated = new = 0
        n_blocks = max_block = 0
        grouped = frame.groupby([f"k{i}" for i in range(len(keys))], sort=True)
        for _, block in grouped:
            ids = sorted(block["rid"].tolist())
            if len(ids) < 2:
                continue
            n_blocks += 1
            max_block = max(max_block, len(ids))
            window = len(ids) - 1 if budget_k is None else min(budget_k, len(ids) - 1)
            for i, a in enumerate(ids):
                for j in range(i + 1, min(i + 1 + window, len(ids))):
                    pair = (a, ids[j])  # sorted ids => already a<b
                    generated += 1
                    if pair not in seen:
                        seen.add(pair)
                        ordered.append(pair)
                        new += 1
        pass_stats.append(
            {
                "keys": list(keys),
                "n_blocks": n_blocks,
                "max_block": max_block,
                "pairs_generated": generated,
                "pairs_new": new,
            }
        )

    dropped = 0
    if budget_k is not None:
        counts: dict[str, int] = {}
        kept = []
        for a, b in ordered:
            if counts.get(a, 0) >= budget_k or counts.get(b, 0) >= budget_k:
                dropped += 1
                continue
            counts[a] = counts.get(a, 0) + 1
            counts[b] = counts.get(b, 0) + 1
            kept.append((a, b))
        ordered = kept

    out = pd.DataFrame(sorted(ordered), columns=["a", "b"], dtype="string")
    out.attrs["stats"] = {
        "method": "matchkeys",
        "passes": pass_stats,
        "budget_k": budget_k,
        "pairs_dropped_by_budget": dropped,
        "n_pairs": len(out),
        "n_records": len(df),
    }
    return out
