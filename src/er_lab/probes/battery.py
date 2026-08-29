"""CheckList-style invariance battery for record embeddings (PLAN.md notebook 08; lit_review §0 claim 6).

A *ProbeSet* is a DataFrame with columns ``slice``, ``kind``, ``text_a``,
``text_b``. ``should_hold`` slices pair a record's serialization with a
same-person variant the encoder should keep close (typos at dialed intensity,
nickname swaps, field-order permutations, format drift). ``must_not_hold``
slices pair it with a different-person confusable it must keep apart (same
record one birth year off, Jr vs Sr, twin/household: different given name, same
family/DOB/address).

The battery only MEASURES — per-slice cosine statistics and separation AUC.
Verdicts (does the invariance hold? is the confusable separated?) are drawn in
the notebooks against pre-registered conjecture cards, never here.

``build_probe_set`` perturbs via ``er_lab.noise.channels`` (imported lazily —
that module is built and calibrated separately); ``battery_report`` needs only
an ``encode`` callable and is import-independent of the noise machinery.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence

import numpy as np
import pandas as pd

__all__ = [
    "MUST_NOT_HOLD",
    "PROBE_COLUMNS",
    "SHOULD_HOLD",
    "battery_report",
    "build_probe_set",
]

PROBE_COLUMNS = ["slice", "kind", "text_a", "text_b"]
SHOULD_HOLD = "should_hold"
MUST_NOT_HOLD = "must_not_hold"

#: canonical-frame metadata columns excluded from the serialized row dict.
_META = ("record_id", "entity_id", "source")


def _channels():
    """Lazy import of er_lab.noise.channels (module under separate construction)."""
    try:
        from er_lab.noise import channels as ch
    except Exception as exc:
        raise ImportError(
            "build_probe_set builds its perturbation slices from er_lab.noise.channels, "
            f"which failed to import ({exc!r}). Install its dependencies (gecko-syndata) "
            "or fix that module before building probe sets; battery_report does not "
            "need it."
        ) from exc
    return ch


def _sample_rows(df: pd.DataFrame, n: int, rng: np.random.Generator) -> pd.DataFrame:
    if len(df) <= n:
        return df
    pos = np.sort(rng.choice(len(df), size=n, replace=False))
    return df.iloc[pos]


def _row_dict(row: pd.Series) -> dict:
    return {k: row[k] for k in row.index if k not in _META}


def _present(value: object) -> bool:
    return value is not None and not pd.isna(value)


def build_probe_set(
    df_canonical: pd.DataFrame,
    *,
    serialize: Callable[[dict], str],
    seed: int,
    lexicon: Mapping[str, set[str]] | None = None,
    dial_rates: Sequence[float] = (0.02, 0.05, 0.1, 0.2),
    n_per_slice: int = 200,
) -> pd.DataFrame:
    """Build the probe set from a canonical frame (see module docstring).

    ``serialize`` maps a row dict (role -> verbatim value, metadata columns
    removed, missing values passed through as-is) to the encoder's input text —
    the SAME serializer the system under test uses, so probes measure the
    deployed representation. Deterministic under ``seed``. Up to ``n_per_slice``
    records are sampled per slice (independently, so slices need not overlap).

    Slices built (a slice is silently empty when no sampled record is applicable
    or when the perturbation cannot change the text — e.g. ``field_order_permutation``
    is empty under an order-invariant serializer; that emptiness is itself
    information for the notebook):

    - ``typo@{rate}`` (should_hold), one per dial rate: each record receives
      ``max(1, round(rate * len(text)))`` sequential single-field typo-channel
      edits, approximating a per-character corruption rate on the serialized text.
    - ``nickname_swap`` (should_hold): given-name lexicon swap
      (``lexicon`` or the channel's built-in default; English-centric, PLAN §4).
    - ``field_order_permutation`` (should_hold): same values, dict key order
      shuffled before serialization.
    - ``format_drift`` (should_hold): dob/address/phone re-formatted, content
      unchanged (``field_format_drift`` channel).
    - ``different_birth_year`` (must_not_hold): dob and/or birth_year shifted by
      ±1–3 years — an otherwise-identical different person.
    - ``suffix_jr_sr`` (must_not_hold): the record as Jr vs the record as Sr
      (father/son confusable).
    - ``twin_household`` (must_not_hold): given_name replaced by a different
      corpus given name — twin/sibling at the same address, same family, same DOB.

    Pairs whose two texts are identical are dropped (a no-op perturbation is not
    a probe; for must_not_hold pairs it would be degenerate).
    """
    ch = _channels()
    if "record_id" not in df_canonical.columns:
        raise KeyError("df_canonical must carry a 'record_id' column (canonical frame contract)")
    if n_per_slice <= 0:
        raise ValueError(f"n_per_slice must be positive, got {n_per_slice}")
    rng = np.random.default_rng(seed)
    rows: list[tuple[str, str, str, str]] = []

    def add(slice_name: str, kind: str, pairs: list[tuple[str, str]]) -> None:
        for ta, tb in pairs:
            if ta != tb:
                rows.append((slice_name, kind, ta, tb))

    def ser(row: pd.Series) -> str:
        return serialize(_row_dict(row))

    # --- should_hold: typo dial -------------------------------------------
    typo_channel = ch.typo()
    for rate in dial_rates:
        sampled = _sample_rows(df_canonical, n_per_slice, rng)
        texts = [ser(r) for _, r in sampled.iterrows()]
        times = [max(1, round(float(rate) * len(t))) for t in texts]
        pairs: list[tuple[str, str]] = []
        for t_val in sorted(set(times)):
            pos = [i for i, tv in enumerate(times) if tv == t_val]
            cur = sampled.iloc[pos]
            for _ in range(t_val):
                cur, _ops = typo_channel.apply(cur, rng, 1.0)
            for i, (_, prow) in zip(pos, cur.iterrows()):
                pairs.append((texts[i], ser(prow)))
        add(f"typo@{rate:g}", SHOULD_HOLD, pairs)

    # --- should_hold: nickname swap ---------------------------------------
    sampled = _sample_rows(df_canonical, n_per_slice, rng)
    swapped, _ops = ch.nickname(lexicon).apply(sampled, rng, 1.0)
    add(
        "nickname_swap",
        SHOULD_HOLD,
        [(ser(r0), ser(r1)) for (_, r0), (_, r1) in zip(sampled.iterrows(), swapped.iterrows())],
    )

    # --- should_hold: field-order permutation ------------------------------
    sampled = _sample_rows(df_canonical, n_per_slice, rng)
    pairs = []
    for _, r in sampled.iterrows():
        d = _row_dict(r)
        keys = list(d)
        perm = [keys[i] for i in rng.permutation(len(keys))]
        pairs.append((serialize(d), serialize({k: d[k] for k in perm})))
    add("field_order_permutation", SHOULD_HOLD, pairs)

    # --- should_hold: format drift -----------------------------------------
    sampled = _sample_rows(df_canonical, n_per_slice, rng)
    drifted, _ops = ch.field_format_drift().apply(sampled, rng, 1.0)
    add(
        "format_drift",
        SHOULD_HOLD,
        [(ser(r0), ser(r1)) for (_, r0), (_, r1) in zip(sampled.iterrows(), drifted.iterrows())],
    )

    # --- must_not_hold: different birth year -------------------------------
    sampled = _sample_rows(df_canonical, n_per_slice, rng)
    pairs = []
    for _, r in sampled.iterrows():
        d = _row_dict(r)
        d2 = dict(d)
        offset = int(rng.choice([-3, -2, -1, 1, 2, 3]))
        changed = False
        if _present(d.get("dob")):
            dt, fmt = ch.parse_dob(str(d["dob"]))
            if dt is not None:
                try:
                    shifted = dt.replace(year=dt.year + offset)
                except ValueError:  # Feb 29 to a non-leap year
                    shifted = dt.replace(year=dt.year + offset, day=28)
                d2["dob"] = shifted.strftime(fmt)
                changed = True
        if _present(d.get("birth_year")) and str(d["birth_year"]).strip().isdigit():
            d2["birth_year"] = str(int(str(d["birth_year"]).strip()) + offset)
            changed = True
        if changed:
            pairs.append((serialize(d), serialize(d2)))
    add("different_birth_year", MUST_NOT_HOLD, pairs)

    # --- must_not_hold: suffix Jr vs Sr ------------------------------------
    sampled = _sample_rows(df_canonical, n_per_slice, rng)
    pairs = []
    for _, r in sampled.iterrows():
        d = _row_dict(r)
        template = str(d["family_name"]) if _present(d.get("family_name")) else "Xx"
        da, db = dict(d), dict(d)
        da["name_suffix"] = ch.match_case(template, "jr")
        db["name_suffix"] = ch.match_case(template, "sr")
        pairs.append((serialize(da), serialize(db)))
    add("suffix_jr_sr", MUST_NOT_HOLD, pairs)

    # --- must_not_hold: twin/household confusable --------------------------
    pairs = []
    if "given_name" in df_canonical.columns:
        pool = sorted({str(v) for v in df_canonical["given_name"].dropna().unique()})
        applicable = df_canonical[df_canonical["given_name"].notna()]
        if len(pool) >= 2 and len(applicable):
            sampled = _sample_rows(applicable, n_per_slice, rng)
            for _, r in sampled.iterrows():
                d = _row_dict(r)
                g = str(d["given_name"])
                alts = [p for p in pool if p.lower() != g.lower()]
                if not alts:
                    continue
                d2 = dict(d)
                d2["given_name"] = ch.match_case(g, alts[int(rng.integers(len(alts)))].lower())
                pairs.append((serialize(d), serialize(d2)))
    add("twin_household", MUST_NOT_HOLD, pairs)

    return pd.DataFrame(rows, columns=PROBE_COLUMNS)


def battery_report(
    encode: Callable[[list[str]], np.ndarray],
    probes: pd.DataFrame,
    *,
    true_pairs: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Measure a probe set under an encoder: per-slice cosine statistics.

    ``encode`` maps a list of texts to a 2-d array of embeddings (one row per
    text, any dimension; rows are L2-normalized here, so cosine similarity is the
    measured quantity). Unique texts are encoded exactly once.

    Returns a DataFrame indexed by ``slice`` with columns:

    - ``kind``: the slice's kind (a slice mixing kinds raises);
    - ``n``: number of probe pairs measured;
    - ``mean_cos`` / ``p05_cos`` / ``p95_cos``: mean and 5th/95th percentile of
      pair cosine similarity within the slice;
    - ``auc_vs_true``: for ``must_not_hold`` slices when ``true_pairs``
      (DataFrame[text_a, text_b] of genuine same-person pairs) is given — the
      probability that a randomly drawn true pair scores a HIGHER cosine than a
      randomly drawn slice pair (Mann-Whitney AUC, ties count 1/2). 1.0 means
      perfect separation of confusables from true matches, 0.5 means none;
      NaN for should_hold slices or when ``true_pairs`` is absent.

    The battery only measures; verdicts are drawn in the notebooks against
    pre-registered conjecture cards (PLAN §5).
    """
    missing = [c for c in PROBE_COLUMNS if c not in probes.columns]
    if missing:
        raise KeyError(f"probes is missing columns {missing}")
    texts: set[str] = set(probes["text_a"]) | set(probes["text_b"])
    if true_pairs is not None:
        tp_missing = [c for c in ("text_a", "text_b") if c not in true_pairs.columns]
        if tp_missing:
            raise KeyError(f"true_pairs is missing columns {tp_missing}")
        texts |= set(true_pairs["text_a"]) | set(true_pairs["text_b"])
    ordered = sorted(texts)
    if ordered:
        emb = np.asarray(encode(list(ordered)), dtype=np.float64)
        if emb.ndim != 2 or emb.shape[0] != len(ordered):
            raise ValueError(
                f"encode must return one embedding row per text: got shape {emb.shape} "
                f"for {len(ordered)} texts"
            )
        emb = emb / np.maximum(np.linalg.norm(emb, axis=1, keepdims=True), 1e-12)
    else:
        emb = np.zeros((0, 1))
    position = {t: i for i, t in enumerate(ordered)}

    def cosines(col_a: pd.Series, col_b: pd.Series) -> np.ndarray:
        ia = col_a.map(position).to_numpy(dtype=int)
        ib = col_b.map(position).to_numpy(dtype=int)
        return np.einsum("ij,ij->i", emb[ia], emb[ib])

    true_sims: np.ndarray | None = None
    if true_pairs is not None and len(true_pairs):
        true_sims = cosines(true_pairs["text_a"], true_pairs["text_b"])

    records: list[dict] = []
    for slice_name, grp in probes.groupby("slice", sort=False):
        kinds = grp["kind"].unique()
        if len(kinds) != 1:
            raise ValueError(f"slice {slice_name!r} mixes kinds {sorted(kinds)}")
        kind = str(kinds[0])
        if kind not in (SHOULD_HOLD, MUST_NOT_HOLD):
            raise ValueError(f"slice {slice_name!r} has unknown kind {kind!r}")
        sims = cosines(grp["text_a"], grp["text_b"])
        auc = float("nan")
        if kind == MUST_NOT_HOLD and true_sims is not None and len(sims):
            from sklearn.metrics import roc_auc_score

            labels = np.concatenate([np.ones(len(true_sims)), np.zeros(len(sims))])
            auc = float(roc_auc_score(labels, np.concatenate([true_sims, sims])))
        records.append(
            {
                "slice": slice_name,
                "kind": kind,
                "n": len(sims),
                "mean_cos": float(np.mean(sims)) if len(sims) else float("nan"),
                "p05_cos": float(np.percentile(sims, 5)) if len(sims) else float("nan"),
                "p95_cos": float(np.percentile(sims, 95)) if len(sims) else float("nan"),
                "auc_vs_true": auc,
            }
        )
    out = pd.DataFrame(
        records, columns=["slice", "kind", "n", "mean_cos", "p05_cos", "p95_cos", "auc_vs_true"]
    )
    return out.set_index("slice")
