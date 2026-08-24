"""Corpus generation: duplicates through noise channels + household confusables.

``generate_corpus`` is the calibrated dirt machine's assembly line (PLAN §4,
notebook 05): every base record spawns a Zipf-distributed number of duplicates,
each duplicate passes through each requested channel at its (exposure-adjusted)
rate, and every mutation lands in the ops log — the ground truth for NSE-02
fidelity scoring and the PLAN §5 mediation analysis.

``household_confusables`` synthesizes the co-resident trap negatives (distinct
people sharing family_name + address: siblings, Jr/Sr pairs, twins) that generic
generators like pseudopeople cannot express (lit_review §3, §12).

Determinism: all randomness flows from ``numpy.random.default_rng(seed)`` with a
fixed draw order — (1) cluster sizes, (2) keep-original mask, (3) channel
applications in ``channel_rates`` insertion order, (4) the final shuffle. Same
seed + same inputs = identical corpus and ops log.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from er_lab.noise.channels import CHANNELS, Channel, match_case, ops_frame, parse_dob

if TYPE_CHECKING:  # pragma: no cover - exposure is an optional collaborator
    from er_lab.noise.exposure import ExposureModel

__all__ = ["generate_corpus", "household_confusables"]

_DEFAULT_DUP_PARAMS = {"a": 2.5, "max": 20}


def _resolve_channels(
    channel_rates: Mapping[str, float], channels: Mapping[str, Channel] | None
) -> dict[str, Channel]:
    resolved: dict[str, Channel] = {}
    for name in channel_rates:
        if channels is not None and name in channels:
            resolved[name] = channels[name]
        elif name in CHANNELS:
            resolved[name] = CHANNELS[name]()
        else:
            raise KeyError(f"unknown channel '{name}'; known: {sorted(CHANNELS)}")
    return resolved


def _resolve_entity_ids(df: pd.DataFrame) -> pd.DataFrame:
    """entity_id := existing value, or the record's own record_id when missing."""
    out = df.copy()
    if "record_id" not in out.columns:
        raise ValueError("base_df must have a 'record_id' column")
    rid = out["record_id"].astype("string")
    if "entity_id" in out.columns:
        out["entity_id"] = out["entity_id"].astype("string").fillna(rid)
    else:
        out["entity_id"] = rid
    return out


def generate_corpus(
    base_df: pd.DataFrame,
    *,
    channel_rates: Mapping[str, float],
    exposure: ExposureModel | None = None,
    channels: Mapping[str, Channel] | None = None,
    dup_dist: str = "zipf",
    dup_params: Mapping[str, float] | None = None,
    keep_original_rate: float = 1.0,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Spawn noisy duplicates of ``base_df`` and return (corpus, ops_log).

    Duplicate-count distribution (``dup_dist='zipf'``, the only supported one):
    each base record's *cluster size* is ``s = min(Z, max)`` with ``Z ~ Zipf(a)``
    (``P[Z=k] ∝ k**-a`` for k >= 1, numpy ``Generator.zipf``), so the record
    spawns ``k = s - 1 >= 0`` duplicates. With the default a=2.5,
    P[s=1] = 1/zeta(2.5) ≈ 0.75 — most entities stay singletons — and no entity
    exceeds ``max`` records (originals included).

    Duplicates get ``record_id = f'{base_record_id}#dup{i}'`` (i = 1..k) and
    ``entity_id`` = the base record's entity_id when present, else the base
    record_id (originals get the same resolution, so clusters share one id).
    Each duplicate then passes through each channel in ``channel_rates``
    insertion order at that channel's rate — turned into per-record rates by
    ``exposure.per_record_rates(dup_df, rate)`` when an exposure model is given
    (group-correlated exposure), else uniform.

    The corpus is originals (each kept with probability ``keep_original_rate``)
    + duplicates, deterministically shuffled; the ops log concatenates all
    channel logs in application order (its record_ids are duplicate ids).

    ``channels`` may inject configured channel instances (e.g. ``nickname`` with
    a fetched lexicon); missing names are built from the ``CHANNELS`` registry.
    """
    if dup_dist != "zipf":
        raise ValueError(f"unsupported dup_dist '{dup_dist}'; only 'zipf' is implemented")
    params = dict(_DEFAULT_DUP_PARAMS)
    params.update(dup_params or {})
    a, k_max = float(params["a"]), int(params["max"])
    if a <= 1.0 or k_max < 1:
        raise ValueError(f"invalid dup_params: a={a} (need >1), max={k_max} (need >=1)")

    rng = np.random.default_rng(seed)
    base = _resolve_entity_ids(base_df)
    n = len(base)

    cluster_size = np.minimum(rng.zipf(a, size=n), k_max) if n else np.array([], dtype=int)
    keep = rng.random(n) < float(keep_original_rate)

    dup_counts = cluster_size - 1
    reps = np.repeat(np.arange(n), dup_counts)
    dup = base.iloc[reps].reset_index(drop=True)
    if len(dup):
        within = np.arange(len(dup)) - np.repeat(np.cumsum(dup_counts) - dup_counts, dup_counts)
        dup["record_id"] = pd.Series(
            [f"{rid}#dup{i + 1}" for rid, i in zip(dup["record_id"], within)], dtype="string"
        )

    resolved = _resolve_channels(channel_rates, channels)
    ops_logs: list[pd.DataFrame] = []
    for name, rate in channel_rates.items():
        if exposure is not None:
            rates = exposure.per_record_rates(dup, float(rate))
        else:
            rates = pd.Series(float(rate), index=dup.index)
        dup, ops = resolved[name].apply(dup, rng, rates)
        ops_logs.append(ops)

    corpus = pd.concat([base.iloc[np.flatnonzero(keep)], dup], ignore_index=True)
    perm = rng.permutation(len(corpus))
    corpus = corpus.iloc[perm].reset_index(drop=True)
    corpus = corpus.astype({c: "string" for c in corpus.columns})

    ops_log = (
        pd.concat(ops_logs, ignore_index=True).astype("string") if ops_logs else ops_frame([])
    )
    return corpus, ops_log


def _shift_year(value: str, years: int) -> str | None:
    """Shift a dob string's year, preserving its format; None when unparseable."""
    dt, fmt = parse_dob(value)
    if dt is None or fmt is None:
        return None
    try:
        shifted = dt.replace(year=dt.year + years)
    except ValueError:  # Feb 29 in a non-leap target year
        shifted = dt.replace(year=dt.year + years, day=28)
    return shifted.strftime(fmt)


def _different_given(pool: list[str], current: str, rng: np.random.Generator) -> str | None:
    if not pool:
        return None
    idx = int(rng.integers(len(pool)))
    if pool[idx].lower() == current.lower():
        idx = (idx + 1) % len(pool)
    if pool[idx].lower() == current.lower():
        return None
    return pool[idx]


def household_confusables(
    base_df: pd.DataFrame, *, rate: float, seed: int
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Synthesize co-resident trap negatives: NEW DISTINCT entities sharing
    family_name + address with a base record (the household/hub pathology,
    lit_review §3). Returns (confusables_df, ops_log).

    Each base record with given_name, family_name, and an address independently
    spawns one confusable with probability ``rate``. Kinds: 'sibling' (different
    given_name, birth field shifted ±1..10 years), 'jr_sr' (same given_name,
    complementary Jr/Sr suffix, birth shifted 20..35 years the right way), and
    'twin' (different given_name, identical birth field). Confusables copy every
    other field verbatim, get ``record_id = f'{base_record_id}#hh1'`` and a FRESH
    ``entity_id`` equal to that record_id — never the base entity. The ops-style
    log (channel='household') records a '__kind__' row plus one row per field
    changed relative to the base record.
    """
    rng = np.random.default_rng(seed)
    df = base_df
    if "record_id" not in df.columns:
        raise ValueError("base_df must have a 'record_id' column")

    addr_field = "street_address" if "street_address" in df.columns else "street"
    kinds = ["sibling", "twin"] + (["jr_sr"] if "name_suffix" in df.columns else [])
    kind_p = [0.5, 0.25, 0.25] if len(kinds) == 3 else [2 / 3, 1 / 3]
    birth_field = "dob" if "dob" in df.columns else ("birth_year" if "birth_year" in df.columns else None)

    pool = (
        sorted({str(v) for v in df["given_name"].dropna() if str(v) != ""})
        if "given_name" in df.columns
        else []
    )

    def value(pos: int, fld: str) -> str | None:
        if fld not in df.columns:
            return None
        v = df[fld].iloc[pos]
        if pd.isna(v) or v == "":
            return None
        return str(v)

    selected = np.flatnonzero(rng.random(len(df)) < np.clip(float(rate), 0.0, 1.0))
    rows: list[pd.Series] = []
    ops: list[tuple] = []
    for pos in selected:
        pos = int(pos)
        given, family = value(pos, "given_name"), value(pos, "family_name")
        if given is None or family is None or value(pos, addr_field) is None:
            continue
        kind = str(rng.choice(kinds, p=kind_p))
        new = df.iloc[pos].copy()
        rid = f"{df['record_id'].iloc[pos]}#hh1"
        new["record_id"] = rid
        new["entity_id"] = rid
        edits: list[tuple[str, object, object]] = []

        if kind in ("sibling", "twin"):
            other = _different_given(pool, given, rng)
            if other is None:
                continue
            edits.append(("given_name", given, other))
            if kind == "sibling" and birth_field is not None:
                b = value(pos, birth_field)
                delta = int(rng.integers(1, 11)) * (-1 if rng.random() < 0.5 else 1)
                if b is not None:
                    shifted = (
                        _shift_year(b, delta)
                        if birth_field == "dob"
                        else (str(int(b) + delta) if b.strip().isdigit() else None)
                    )
                    if shifted is not None and shifted != b:
                        edits.append((birth_field, b, shifted))
        else:  # jr_sr
            cur_suffix = value(pos, "name_suffix")
            if cur_suffix is not None and cur_suffix.upper().strip(".") == "JR":
                new_suffix, direction = "sr", -1  # base is the junior; new is older
            elif cur_suffix is not None and cur_suffix.upper().strip(".") == "SR":
                new_suffix, direction = "jr", 1
            else:
                new_suffix = str(rng.choice(["jr", "sr"]))
                direction = 1 if new_suffix == "jr" else -1
            cased = match_case(family, new_suffix)
            if not _eq_na(df["name_suffix"].iloc[pos], cased):
                edits.append(("name_suffix", df["name_suffix"].iloc[pos], cased))
            if birth_field is not None:
                b = value(pos, birth_field)
                delta = direction * int(rng.integers(20, 36))
                if b is not None:
                    shifted = (
                        _shift_year(b, delta)
                        if birth_field == "dob"
                        else (str(int(b) + delta) if b.strip().isdigit() else None)
                    )
                    if shifted is not None and shifted != b:
                        edits.append((birth_field, b, shifted))

        ops.append((rid, "household", "__kind__", pd.NA, kind))
        for fld, before, after in edits:
            new[fld] = after
            ops.append((rid, "household", fld, before, after))
        rows.append(new)

    if rows:
        hh = pd.DataFrame(rows).reset_index(drop=True)
    else:
        hh = pd.DataFrame(columns=df.columns)
    hh = hh.astype({c: "string" for c in hh.columns})
    return hh, ops_frame(ops)


def _eq_na(a: object, b: object) -> bool:
    a_na, b_na = bool(pd.isna(a)), bool(pd.isna(b))
    if a_na or b_na:
        return a_na and b_na
    return a == b
