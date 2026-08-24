"""Noise channels for the calibrated dirt machine (PLAN.md §4; notebook 05; NSE-02/03).

Unlike loaders (which stay verbatim), these transforms ARE the product: each channel
perturbs canonical person-record frames (``er_lab.data.schema`` role columns, all
pandas 'string' dtype) and logs every cell it touches. The ops log is ground truth
for the mediation analysis (PLAN §5) and generator-fidelity scoring (NSE-02).

Contract
--------
A ``Channel`` has a ``name`` and ``apply(df, rng, rate_per_record) -> (df_new, ops)``.
``rate_per_record`` is a per-record-per-channel probability (scalar, or a Series
positionally aligned to ``df`` — group-correlated exposure multiplies a base rate
into it, see ``er_lab.noise.exposure``). ``ops`` has columns
record_id/channel/field/before/after with one row per changed cell; unchanged cells
never appear. All randomness flows from the passed ``numpy.random.Generator``:
same seed + same frame = identical output. The input frame is never mutated.

Gecko (gecko-syndata v0.6.4) wrapping
-------------------------------------
Gecko mutators are ``(list[Series], p) -> list[Series]`` closures produced by factory
functions with the rng bound at construction. We bind the caller's rng at apply time
and do row selection ourselves (gecko's scalar ``p`` cannot express per-record
exposure-adjusted rates), then call each mutator with ``p=1.0`` on the selected
subset. Wrapped gecko functions:

- ``gecko.mutator.with_cldr_keymap_file`` — keyboard-neighbor typos ('typo').
  Gecko ships no keymap data, so a minimal en-US QWERTY CLDR keymap is embedded here.
- ``gecko.mutator.with_insert`` / ``with_delete`` / ``with_transpose`` /
  ``with_substitute`` — random character edit ops, mixed into 'typo'.
- ``gecko.mutator.with_replacement_table`` (``inline=True``) — OCR confusions
  ('ocr'), GeCO-style embedded confusion table.
- ``gecko.mutator.with_phonetic_replacement_table`` — phonetic respellings
  ('phonetic_spelling'), GeCO-style embedded rule table.

Not wrapped, and why: ``with_missing_value`` writes a sentinel string while
field_dropout needs true ``pd.NA`` plus per-record field choice; ``with_permute``
cannot log both swapped cells; ``with_datetime_offset`` changes the date *value*
while field_format_drift changes only its *format*; ``with_group`` /
``mutate_data_frame`` select rows internally (incompatible with per-record rates)
and ``with_group`` crashes sub-mutators on empty selections.
"""

from __future__ import annotations

import string
import tempfile
import warnings
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable, Mapping, Protocol, Sequence

import numpy as np
import pandas as pd
from gecko import GeckoWarning
from gecko import mutator as gmut

__all__ = [
    "Channel",
    "CHANNELS",
    "OPS_COLUMNS",
    "ops_frame",
    "match_case",
    "parse_dob",
    "typo",
    "ocr",
    "phonetic_spelling",
    "nickname",
    "name_order_swap",
    "field_dropout",
    "field_format_drift",
    "suffix_confusion",
    "hub_value",
    "raw_unparse",
    "fetch_lexicon",
    "load_lexicon",
]

OPS_COLUMNS = ["record_id", "channel", "field", "before", "after"]

GeckoMutator = Callable[[list[pd.Series], float], list[pd.Series]]
MutatorFactory = Callable[[np.random.Generator], GeckoMutator]
Edit = tuple[str, object, object]  # (field, before, after)


class Channel(Protocol):
    """A noise channel: a named, logged, rate-driven frame transform."""

    name: str

    def apply(
        self,
        df: pd.DataFrame,
        rng: np.random.Generator,
        rate_per_record: pd.Series | float,
    ) -> tuple[pd.DataFrame, pd.DataFrame]: ...


# ---------------------------------------------------------------------------
# shared helpers
# ---------------------------------------------------------------------------


def ops_frame(rows: Sequence[tuple]) -> pd.DataFrame:
    """Build an ops-log frame (record_id, channel, field, before, after; string dtype)."""
    return pd.DataFrame(list(rows), columns=OPS_COLUMNS).astype("string")


def match_case(template: str, value: str) -> str:
    """Re-case ``value`` to the case pattern of ``template`` (WILLIAM->BILL, William->Bill)."""
    if template.isupper():
        return value.upper()
    if template.islower():
        return value.lower()
    if template[:1].isupper():
        return value[:1].upper() + value[1:].lower()
    return value.lower()


#: dob formats recognized for format drift / hub values, tried in order.
DOB_FORMATS: tuple[str, ...] = ("%Y-%m-%d", "%m/%d/%Y", "%Y%m%d", "%m-%d-%Y", "%d %b %Y")


def parse_dob(value: str) -> tuple[datetime | None, str | None]:
    """Parse a dob string against ``DOB_FORMATS``; return (datetime, format) or (None, None)."""
    for fmt in DOB_FORMATS:
        try:
            return datetime.strptime(value, fmt), fmt
        except ValueError:
            continue
    return None, None


def _rates_array(rate_per_record: pd.Series | float, n: int) -> np.ndarray:
    if np.isscalar(rate_per_record):
        arr = np.full(n, float(rate_per_record))  # type: ignore[arg-type]
    else:
        arr = np.asarray(pd.Series(rate_per_record), dtype=float)
    if arr.shape[0] != n:
        raise ValueError(f"rate_per_record has length {arr.shape[0]}, frame has {n} rows")
    return np.clip(arr, 0.0, 1.0)


def _get(out: pd.DataFrame, pos: int, fld: str) -> str | None:
    """Value of cell (pos, fld) as str, or None when the column/value is absent/empty."""
    if fld not in out.columns:
        return None
    v = out[fld].iloc[pos]
    if pd.isna(v) or v == "":
        return None
    return str(v)


def _cell_eq(a: object, b: object) -> bool:
    a_na, b_na = bool(pd.isna(a)), bool(pd.isna(b))
    if a_na or b_na:
        return a_na and b_na
    return a == b


# ---------------------------------------------------------------------------
# per-record channels (one python-level edit function per selected record)
# ---------------------------------------------------------------------------

EditFn = Callable[[pd.DataFrame, int, np.random.Generator], list[Edit]]


@dataclass(frozen=True)
class PerRecordChannel:
    """Channel driven by an edit function applied to each rate-selected record.

    ``edit_fn(out, pos, rng)`` returns the list of (field, before, after) edits for
    the record at positional index ``pos`` — an empty list when the channel is not
    applicable to that record (the record then contributes no ops). ``prepare_fn``
    may add columns the channel writes into (e.g. full_name for raw_unparse).
    """

    name: str
    edit_fn: EditFn
    prepare_fn: Callable[[pd.DataFrame], pd.DataFrame] | None = None

    def apply(
        self,
        df: pd.DataFrame,
        rng: np.random.Generator,
        rate_per_record: pd.Series | float,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        out = df.copy()
        if self.prepare_fn is not None:
            out = self.prepare_fn(out)
        rates = _rates_array(rate_per_record, len(df))
        selected = np.flatnonzero(rng.random(len(df)) < rates)
        ops: list[tuple] = []
        for pos in selected:
            pos = int(pos)
            rid = str(out["record_id"].iloc[pos])
            for fld, before, after in self.edit_fn(out, pos, rng):
                out.iloc[pos, out.columns.get_loc(fld)] = after
                ops.append((rid, self.name, fld, before, after))
        return out, ops_frame(ops)


# ---------------------------------------------------------------------------
# gecko-backed channels
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class GeckoMutatorChannel:
    """Channel that wraps one or more gecko mutators (see module docstring).

    Row selection happens here (per-record rates); each selected record has one
    applicable field chosen uniformly at random, then one weighted mutator applied
    to it with gecko ``p=1.0``. A gecko mutator may leave a value unchanged (e.g.
    no keymap candidate); such records produce no ops row.
    """

    name: str
    fields: tuple[str, ...]
    mutators: tuple[tuple[float, MutatorFactory], ...]

    def apply(
        self,
        df: pd.DataFrame,
        rng: np.random.Generator,
        rate_per_record: pd.Series | float,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        out = df.copy()
        rates = _rates_array(rate_per_record, len(df))
        selected = np.flatnonzero(rng.random(len(df)) < rates)
        fields = [f for f in self.fields if f in out.columns]
        ops: list[tuple] = []
        if selected.size == 0 or not fields:
            return out, ops_frame(ops)

        # one applicable field per selected record, chosen uniformly
        chosen: dict[str, list[int]] = {}
        for pos in selected:
            pos = int(pos)
            applicable = [f for f in fields if _get(out, pos, f) is not None]
            if not applicable:
                continue
            fld = applicable[int(rng.integers(len(applicable)))]
            chosen.setdefault(fld, []).append(pos)

        weights = np.array([w for w, _ in self.mutators], dtype=float)
        weights = weights / weights.sum()
        built = [make(rng) for _, make in self.mutators]

        for fld, pos_list in chosen.items():
            positions = np.array(pos_list)
            col = out.columns.get_loc(fld)
            assign = rng.choice(len(built), size=positions.size, p=weights)
            for mut_idx, mut in enumerate(built):
                grp = positions[assign == mut_idx]
                if grp.size == 0:
                    continue
                srs = out.iloc[grp, col].astype(str)
                srs.index = pd.RangeIndex(grp.size)  # unique index for gecko internals
                try:
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore", GeckoWarning)
                        res = mut([srs], 1.0)[0]
                except ZeroDivisionError:  # gecko keymap: no candidate row in group
                    continue
                for j in range(grp.size):
                    before, after = srs.iloc[j], str(res.iloc[j])
                    if after != before:
                        p = int(grp[j])
                        out.iloc[p, col] = after
                        ops.append((str(out["record_id"].iloc[p]), self.name, fld, before, after))
        return out, ops_frame(ops)


# --- embedded en-US QWERTY CLDR keymap (gecko ships no keymap data) ---------

_QWERTY_CLDR_XML = """<?xml version="1.0" encoding="UTF-8"?>
<keyboard locale="en-t-k0-er-lab-qwerty">
  <keyMap>
    <map iso="E00" to="`"/><map iso="E01" to="1"/><map iso="E02" to="2"/>
    <map iso="E03" to="3"/><map iso="E04" to="4"/><map iso="E05" to="5"/>
    <map iso="E06" to="6"/><map iso="E07" to="7"/><map iso="E08" to="8"/>
    <map iso="E09" to="9"/><map iso="E10" to="0"/><map iso="E11" to="-"/>
    <map iso="E12" to="="/>
    <map iso="D01" to="q"/><map iso="D02" to="w"/><map iso="D03" to="e"/>
    <map iso="D04" to="r"/><map iso="D05" to="t"/><map iso="D06" to="y"/>
    <map iso="D07" to="u"/><map iso="D08" to="i"/><map iso="D09" to="o"/>
    <map iso="D10" to="p"/><map iso="D11" to="["/><map iso="D12" to="]"/>
    <map iso="C01" to="a"/><map iso="C02" to="s"/><map iso="C03" to="d"/>
    <map iso="C04" to="f"/><map iso="C05" to="g"/><map iso="C06" to="h"/>
    <map iso="C07" to="j"/><map iso="C08" to="k"/><map iso="C09" to="l"/>
    <map iso="C10" to=";"/><map iso="C11" to="'"/>
    <map iso="B01" to="z"/><map iso="B02" to="x"/><map iso="B03" to="c"/>
    <map iso="B04" to="v"/><map iso="B05" to="b"/><map iso="B06" to="n"/>
    <map iso="B07" to="m"/><map iso="B08" to=","/><map iso="B09" to="."/>
    <map iso="B10" to="/"/>
  </keyMap>
  <keyMap modifiers="shift">
    <map iso="E00" to="~"/><map iso="E01" to="!"/><map iso="E02" to="@"/>
    <map iso="E03" to="#"/><map iso="E04" to="$"/><map iso="E05" to="%"/>
    <map iso="E06" to="^"/><map iso="E07" to="&amp;"/><map iso="E08" to="*"/>
    <map iso="E09" to="("/><map iso="E10" to=")"/><map iso="E11" to="_"/>
    <map iso="E12" to="+"/>
    <map iso="D01" to="Q"/><map iso="D02" to="W"/><map iso="D03" to="E"/>
    <map iso="D04" to="R"/><map iso="D05" to="T"/><map iso="D06" to="Y"/>
    <map iso="D07" to="U"/><map iso="D08" to="I"/><map iso="D09" to="O"/>
    <map iso="D10" to="P"/><map iso="D11" to="{"/><map iso="D12" to="}"/>
    <map iso="C01" to="A"/><map iso="C02" to="S"/><map iso="C03" to="D"/>
    <map iso="C04" to="F"/><map iso="C05" to="G"/><map iso="C06" to="H"/>
    <map iso="C07" to="J"/><map iso="C08" to="K"/><map iso="C09" to="L"/>
    <map iso="C10" to=":"/><map iso="C11" to="&quot;"/>
    <map iso="B01" to="Z"/><map iso="B02" to="X"/><map iso="B03" to="C"/>
    <map iso="B04" to="V"/><map iso="B05" to="B"/><map iso="B06" to="N"/>
    <map iso="B07" to="M"/><map iso="B08" to="&lt;"/><map iso="B09" to="."/>
    <map iso="B10" to="?"/>
  </keyMap>
</keyboard>
"""

_KEYMAP_CACHE: Path | None = None


def _qwerty_keymap_path() -> Path:
    """Write the embedded QWERTY CLDR keymap to a cached temp file, return its path."""
    global _KEYMAP_CACHE
    if _KEYMAP_CACHE is None or not _KEYMAP_CACHE.exists():
        fd, name = tempfile.mkstemp(prefix="er_lab_qwerty_", suffix=".xml")
        path = Path(name)
        with open(fd, "w", encoding="utf-8") as f:
            f.write(_QWERTY_CLDR_XML)
        _KEYMAP_CACHE = path
    return _KEYMAP_CACHE


# --- embedded OCR confusion table (GeCO-lineage; both directions explicit) --

_OCR_PAIRS: tuple[tuple[str, str], ...] = (
    ("0", "O"), ("O", "0"), ("1", "I"), ("I", "1"), ("1", "l"), ("l", "1"),
    ("5", "S"), ("S", "5"), ("8", "B"), ("B", "8"), ("2", "Z"), ("Z", "2"),
    ("6", "G"), ("G", "6"), ("U", "V"), ("V", "U"), ("u", "v"), ("v", "u"),
    ("D", "O"), ("m", "rn"), ("rn", "m"), ("cl", "d"), ("d", "cl"),
    ("w", "vv"), ("vv", "w"), ("g", "q"), ("q", "g"),
)

# --- embedded phonetic rules (GeCO-lineage; flags: ^ start, $ end, _ middle,
#     "" = anywhere). Case variants generated below. --------------------------

_PHONETIC_RULES: tuple[tuple[str, str, str], ...] = (
    ("ph", "f", ""), ("f", "ph", "$"), ("gh", "f", "$"),
    ("ck", "k", ""), ("k", "ck", "$"), ("c", "k", "^"), ("k", "c", "^"),
    ("sch", "sh", "^"), ("sh", "sch", "^"),
    ("z", "s", ""), ("s", "z", "_"),
    ("ee", "ea", ""), ("ea", "ee", ""),
    ("ie", "y", "$"), ("y", "ie", "$"),
    ("ou", "ow", ""), ("ow", "ou", ""),
    ("x", "ks", ""), ("qu", "kw", "^"),
    ("kn", "n", "^"), ("wr", "r", "^"),
    ("mb", "m", "$"), ("dt", "t", "$"),
)


def _phonetic_table() -> pd.DataFrame:
    rows: list[tuple[str, str, str]] = []
    for pat, rep, flags in _PHONETIC_RULES:
        variants = {(pat, rep), (pat.upper(), rep.upper()), (pat.capitalize(), rep.capitalize())}
        for p_, r_ in sorted(variants):
            rows.append((p_, r_, flags))
    return pd.DataFrame(rows, columns=["pattern", "replacement", "flags"])


_TYPO_FIELDS = ("given_name", "middle_name", "family_name", "street", "city")
_OCR_FIELDS = (
    "given_name", "middle_name", "family_name", "street", "city",
    "house_number", "zip", "phone",
)
_PHONETIC_FIELDS = ("given_name", "middle_name", "family_name", "street", "city")
_DROPOUT_FIELDS = (
    "middle_name", "name_suffix", "dob", "birth_year", "phone", "email",
    "unit", "race", "ethnicity", "sex",
)


def typo(fields: Sequence[str] = _TYPO_FIELDS) -> Channel:
    """Keyboard-neighbor typos (gecko CLDR keymap) mixed with random edit ops.

    Mixture per selected record: 60% keyboard neighbor, 10% each of insert /
    delete / transpose / substitute (gecko ``with_insert``/``with_delete``/
    ``with_transpose``/``with_substitute``).
    """

    def kb(rng: np.random.Generator) -> GeckoMutator:
        return gmut.with_cldr_keymap_file(
            _qwerty_keymap_path(), charset=string.ascii_letters, rng=rng
        )

    return GeckoMutatorChannel(
        name="typo",
        fields=tuple(fields),
        mutators=(
            (0.6, kb),
            (0.1, lambda rng: gmut.with_insert(rng=rng)),
            (0.1, lambda rng: gmut.with_delete(rng=rng)),
            (0.1, lambda rng: gmut.with_transpose(rng=rng)),
            (0.1, lambda rng: gmut.with_substitute(rng=rng)),
        ),
    )


def ocr(fields: Sequence[str] = _OCR_FIELDS) -> Channel:
    """OCR character confusions via gecko ``with_replacement_table`` (inline)."""
    table = pd.DataFrame(list(_OCR_PAIRS), columns=["source", "target"])

    def make(rng: np.random.Generator) -> GeckoMutator:
        return gmut.with_replacement_table(
            table, source_column="source", target_column="target", inline=True, rng=rng
        )

    return GeckoMutatorChannel(name="ocr", fields=tuple(fields), mutators=((1.0, make),))


def phonetic_spelling(fields: Sequence[str] = _PHONETIC_FIELDS) -> Channel:
    """Phonetic respellings via gecko ``with_phonetic_replacement_table``."""
    table = _phonetic_table()

    def make(rng: np.random.Generator) -> GeckoMutator:
        return gmut.with_phonetic_replacement_table(
            table,
            source_column="pattern",
            target_column="replacement",
            flags_column="flags",
            rng=rng,
        )

    return GeckoMutatorChannel(
        name="phonetic_spelling", fields=tuple(fields), mutators=((1.0, make),)
    )


# ---------------------------------------------------------------------------
# nickname lexicon
# ---------------------------------------------------------------------------

NICKNAMES_URL = "https://raw.githubusercontent.com/carltonnorthern/nicknames/master/names.csv"

#: Small built-in fallback so the channel works without a fetched lexicon.
#: Real runs should pass ``load_lexicon(data_root)`` (carltonnorthern; English-
#: centric with documented provenance bias — PLAN §4).
_DEFAULT_LEXICON: dict[str, set[str]] = {
    "william": {"bill", "will", "billy", "liam"},
    "robert": {"bob", "rob", "bobby"},
    "elizabeth": {"liz", "beth", "betsy", "eliza"},
    "margaret": {"peggy", "meg", "maggie"},
    "james": {"jim", "jimmy", "jamie"},
    "john": {"jack", "johnny"},
    "richard": {"dick", "rick", "richie"},
    "katherine": {"kate", "katie", "kathy"},
    "michael": {"mike", "mikey"},
    "joseph": {"joe", "joey"},
    "thomas": {"tom", "tommy"},
    "charles": {"chuck", "charlie"},
    "patricia": {"pat", "patty", "tricia"},
    "edward": {"ed", "ted", "ned", "eddie"},
    "anthony": {"tony"},
    "christopher": {"chris", "topher"},
    "daniel": {"dan", "danny"},
    "matthew": {"matt"},
    "nicholas": {"nick"},
    "susan": {"sue", "susie"},
    "theodore": {"ted", "theo"},
    "alexander": {"alex", "sandy"},
    "benjamin": {"ben", "benny"},
    "samuel": {"sam", "sammy"},
    "andrew": {"andy", "drew"},
    "steven": {"steve"},
    "kenneth": {"ken", "kenny"},
    "donald": {"don", "donnie"},
}


def fetch_lexicon(data_root: str | Path, url: str = NICKNAMES_URL, timeout: float = 30.0) -> Path:
    """Download the carltonnorthern nickname lexicon into ``data_root/lexicons/names.csv``.

    Network call (skipped if the file already exists). Returns the csv path.
    """
    import requests

    dest = Path(data_root) / "lexicons" / "names.csv"
    if dest.exists():
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    resp = requests.get(url, timeout=timeout)
    resp.raise_for_status()
    dest.write_bytes(resp.content)
    return dest


def load_lexicon(data_root: str | Path) -> dict[str, set[str]]:
    """Load the fetched lexicon: dict[canonical_lower, set of variants_lower].

    File format (carltonnorthern names.csv): one canonical name per line followed
    by its comma-separated nicknames, no header.
    """
    path = Path(data_root) / "lexicons" / "names.csv"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found; run fetch_lexicon(data_root) first")
    lexicon: dict[str, set[str]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        parts = [p.strip().lower() for p in line.split(",") if p.strip()]
        if len(parts) >= 2:
            lexicon.setdefault(parts[0], set()).update(parts[1:])
    return lexicon


def _nickname_candidates(lexicon: Mapping[str, set[str]]) -> dict[str, tuple[str, ...]]:
    """Symmetric swap candidates: canonical -> variants, variant -> canonical + siblings."""
    cand: dict[str, set[str]] = {}
    for canonical, variants in sorted(lexicon.items()):
        vs = {v for v in variants if v != canonical}
        if not vs:
            continue
        cand.setdefault(canonical, set()).update(vs)
        for v in sorted(vs):
            cand.setdefault(v, set()).update(({canonical} | vs) - {v})
    return {k: tuple(sorted(v)) for k, v in sorted(cand.items()) if v}


def nickname(lexicon: Mapping[str, set[str]] | None = None) -> Channel:
    """Lexicon-based given-name swap, case-pattern preserving (WILLIAM->BILL)."""
    candidates = _nickname_candidates(_DEFAULT_LEXICON if lexicon is None else lexicon)

    def edit(out: pd.DataFrame, pos: int, rng: np.random.Generator) -> list[Edit]:
        value = _get(out, pos, "given_name")
        if value is None:
            return []
        options = candidates.get(value.lower())
        if not options:
            return []
        pick = options[int(rng.integers(len(options)))]
        new = match_case(value, pick)
        if new == value:
            return []
        return [("given_name", value, new)]

    return PerRecordChannel(name="nickname", edit_fn=edit)


# ---------------------------------------------------------------------------
# remaining per-record channels
# ---------------------------------------------------------------------------


def name_order_swap() -> Channel:
    """Swap given_name and family_name (both cells logged)."""

    def edit(out: pd.DataFrame, pos: int, rng: np.random.Generator) -> list[Edit]:
        g, f = _get(out, pos, "given_name"), _get(out, pos, "family_name")
        if g is None or f is None or g == f:
            return []
        return [("given_name", g, f), ("family_name", f, g)]

    return PerRecordChannel(name="name_order_swap", edit_fn=edit)


def field_dropout(fields: Sequence[str] = _DROPOUT_FIELDS) -> Channel:
    """Set one populated field (chosen uniformly among ``fields``) to pd.NA."""
    fields = tuple(fields)

    def edit(out: pd.DataFrame, pos: int, rng: np.random.Generator) -> list[Edit]:
        present = [f for f in fields if _get(out, pos, f) is not None]
        if not present:
            return []
        fld = present[int(rng.integers(len(present)))]
        return [(fld, _get(out, pos, fld), pd.NA)]

    return PerRecordChannel(name="field_dropout", edit_fn=edit)


_ADDR_ABBREV: tuple[tuple[str, str], ...] = (
    ("STREET", "ST"), ("AVENUE", "AVE"), ("ROAD", "RD"), ("DRIVE", "DR"),
    ("LANE", "LN"), ("BOULEVARD", "BLVD"), ("COURT", "CT"), ("CIRCLE", "CIR"),
    ("PLACE", "PL"), ("HIGHWAY", "HWY"), ("TRAIL", "TRL"), ("PARKWAY", "PKWY"),
    ("NORTH", "N"), ("SOUTH", "S"), ("EAST", "E"), ("WEST", "W"),
    ("APARTMENT", "APT"), ("SUITE", "STE"),
)
_ADDR_WORDMAP: dict[str, str] = {}
for _long, _short in _ADDR_ABBREV:
    _ADDR_WORDMAP[_long] = _short
    _ADDR_WORDMAP[_short] = _long


def _phone_variants(value: str) -> list[str]:
    digits = "".join(ch for ch in value if ch.isdigit())
    if len(digits) == 10:
        a, b, c = digits[:3], digits[3:6], digits[6:]
        cands = [digits, f"({a}) {b}-{c}", f"{a}-{b}-{c}", f"{a}.{b}.{c}"]
    elif len(digits) == 7:
        b, c = digits[:3], digits[3:]
        cands = [digits, f"{b}-{c}", f"{b}.{c}"]
    elif digits:
        cands = [digits]
    else:
        return []
    return [c for c in cands if c != value]


def field_format_drift() -> Channel:
    """Re-format one field without changing its content.

    dob: re-emit in a different ``DOB_FORMATS`` member; street/street_address:
    swap one word with its abbreviation table counterpart (case-pattern kept);
    phone: re-punctuate the same digits.
    """

    def edit(out: pd.DataFrame, pos: int, rng: np.random.Generator) -> list[Edit]:
        options: list[str] = []
        dob = _get(out, pos, "dob")
        if dob is not None and parse_dob(dob)[0] is not None:
            options.append("dob")
        for fld in ("street_address", "street"):
            v = _get(out, pos, fld)
            if v is not None and any(w.upper() in _ADDR_WORDMAP for w in v.split(" ")):
                options.append(fld)
        phone = _get(out, pos, "phone")
        if phone is not None and _phone_variants(phone):
            options.append("phone")
        if not options:
            return []
        fld = options[int(rng.integers(len(options)))]

        if fld == "dob":
            dt, fmt = parse_dob(dob)  # type: ignore[arg-type]
            alts = [f for f in DOB_FORMATS if f != fmt]
            new = dt.strftime(alts[int(rng.integers(len(alts)))])  # type: ignore[union-attr]
            return [("dob", dob, new)] if new != dob else []
        if fld == "phone":
            variants = _phone_variants(phone)  # type: ignore[arg-type]
            return [("phone", phone, variants[int(rng.integers(len(variants)))])]
        value = _get(out, pos, fld)
        words = value.split(" ")  # type: ignore[union-attr]
        hits = [i for i, w in enumerate(words) if w.upper() in _ADDR_WORDMAP]
        i = hits[int(rng.integers(len(hits)))]
        words[i] = match_case(words[i], _ADDR_WORDMAP[words[i].upper()].lower())
        new = " ".join(words)
        return [(fld, value, new)] if new != value else []

    return PerRecordChannel(name="field_format_drift", edit_fn=edit)


_SUFFIX_CONFUSION: dict[str, tuple[str, ...]] = {
    "JR": ("SR", "II"),
    "SR": ("JR",),
    "II": ("JR", "III"),
    "III": ("II", "IV"),
    "IV": ("III",),
}


def suffix_confusion() -> Channel:
    """Add, drop, or confuse a name suffix (Jr/Sr/II...), case-matched."""

    def edit(out: pd.DataFrame, pos: int, rng: np.random.Generator) -> list[Edit]:
        if "name_suffix" not in out.columns:
            return []
        raw = out["name_suffix"].iloc[pos]
        cur = _get(out, pos, "name_suffix")
        if cur is None:  # add a suffix
            pick = ("jr", "sr", "ii")[int(rng.choice(3, p=[0.5, 0.25, 0.25]))]
            template = _get(out, pos, "family_name") or "Xx"
            return [("name_suffix", raw, match_case(template, pick))]
        if rng.random() < 0.5:  # drop
            return [("name_suffix", cur, pd.NA)]
        options = _SUFFIX_CONFUSION.get(cur.upper().strip("."))
        if not options:
            return [("name_suffix", cur, pd.NA)]
        new = match_case(cur, options[int(rng.integers(len(options)))].lower())
        if new == cur:
            return []
        return [("name_suffix", cur, new)]

    return PerRecordChannel(name="suffix_confusion", edit_fn=edit)


#: hub placeholder values (chain-merge pathology injectors; lit_review §3).
HUB_PHONE = "0000000"
HUB_ADDRESS = "GENERAL DELIVERY"
HUB_BIRTH_YEAR = "1900"


def hub_value() -> Channel:
    """Inject placeholder hub values: default dob (1900-01-01 in the record's own
    format), phone 0000000, address GENERAL DELIVERY, birth_year 1900.

    Applies to missing values too (hub values commonly stand in for unknowns);
    one target field is chosen uniformly among those present in the frame.
    """

    def edit(out: pd.DataFrame, pos: int, rng: np.random.Generator) -> list[Edit]:
        options = [f for f in ("dob", "birth_year", "phone") if f in out.columns]
        if "street_address" in out.columns:
            options.append("street_address")
        elif "street" in out.columns:
            options.append("street")
        if not options:
            return []
        fld = options[int(rng.integers(len(options)))]
        raw = out[fld].iloc[pos]
        cur = _get(out, pos, fld)
        if fld == "dob":
            fmt = parse_dob(cur)[1] if cur is not None else None
            hub = datetime(1900, 1, 1).strftime(fmt or "%Y-%m-%d")
        elif fld == "birth_year":
            hub = HUB_BIRTH_YEAR
        elif fld == "phone":
            hub = HUB_PHONE
        else:
            hub = HUB_ADDRESS
        if _cell_eq(raw, hub):
            return []
        return [(fld, raw, hub)]

    return PerRecordChannel(name="hub_value", edit_fn=edit)


def raw_unparse() -> Channel:
    """Compose parsed roles into raw single-string forms and blank the parts.

    The pre-standardization regime for PRS-01: name parts become
    ``'FAMILY, GIVEN M SUFFIX'`` in ``full_name`` (middle reduced to initial);
    house_number/street/unit become a space-joined ``street_address``. Every
    consumed part is set to pd.NA and logged.
    """

    def prepare(out: pd.DataFrame) -> pd.DataFrame:
        if (
            "given_name" in out.columns
            and "family_name" in out.columns
            and "full_name" not in out.columns
        ):
            out["full_name"] = pd.Series(pd.NA, index=out.index, dtype="string")
        if "street" in out.columns and "street_address" not in out.columns:
            out["street_address"] = pd.Series(pd.NA, index=out.index, dtype="string")
        return out

    def edit(out: pd.DataFrame, pos: int, rng: np.random.Generator) -> list[Edit]:
        edits: list[Edit] = []
        g, f = _get(out, pos, "given_name"), _get(out, pos, "family_name")
        if g is not None and f is not None and "full_name" in out.columns:
            m, s = _get(out, pos, "middle_name"), _get(out, pos, "name_suffix")
            full = f"{f}, {g}" + (f" {m[0]}" if m else "") + (f" {s}" if s else "")
            cur_full = out["full_name"].iloc[pos]
            if not _cell_eq(cur_full, full):
                edits.append(("full_name", cur_full, full))
            edits.append(("given_name", g, pd.NA))
            edits.append(("family_name", f, pd.NA))
            if m is not None:
                edits.append(("middle_name", m, pd.NA))
            if s is not None:
                edits.append(("name_suffix", s, pd.NA))
        st = _get(out, pos, "street")
        if st is not None and "street_address" in out.columns:
            hn, un = _get(out, pos, "house_number"), _get(out, pos, "unit")
            addr = " ".join(p for p in (hn, st, un) if p is not None)
            cur_addr = out["street_address"].iloc[pos]
            if not _cell_eq(cur_addr, addr):
                edits.append(("street_address", cur_addr, addr))
            if hn is not None:
                edits.append(("house_number", hn, pd.NA))
            edits.append(("street", st, pd.NA))
            if un is not None:
                edits.append(("unit", un, pd.NA))
        return edits

    return PerRecordChannel(name="raw_unparse", edit_fn=edit, prepare_fn=prepare)


#: Channel registry: name -> factory (callable with sensible defaults).
CHANNELS: dict[str, Callable[..., Channel]] = {
    "typo": typo,
    "ocr": ocr,
    "phonetic_spelling": phonetic_spelling,
    "nickname": nickname,
    "name_order_swap": name_order_swap,
    "field_dropout": field_dropout,
    "field_format_drift": field_format_drift,
    "suffix_confusion": suffix_confusion,
    "hub_value": hub_value,
    "raw_unparse": raw_unparse,
}
