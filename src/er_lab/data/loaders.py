"""Dataset loaders: fetch-if-absent raw files -> canonical role frames (PLAN.md §4).

Every loader returns ``(df_canonical, DeclaredSchema)`` and prints a one-line
provenance/terms note on first call per process (DATA_GOVERNANCE.md, rule 1).
Loaders deliver field values verbatim as strings — no cleaning, normalization, or
standardization here: upstream standardization is an experimental factor (PRS-01).
"""

from __future__ import annotations

import hashlib
import re
from pathlib import Path

import pandas as pd
import requests

from er_lab.data.schema import DeclaredSchema

REPO_ROOT = Path(__file__).resolve().parents[3]
SCHEMA_DIR = REPO_ROOT / "configs" / "schemas"

SPLINK_RAW_BASE = (
    "https://raw.githubusercontent.com/moj-analytical-services/splink_datasets/master/data"
)
ONC_CONTENTS_URL = "https://api.github.com/repos/onc-healthit/patient-matching/contents/"
ONC_RAW_BASE = "https://raw.githubusercontent.com/onc-healthit/patient-matching/master"
_ONC_FILE_RE = re.compile(
    r"^ONC Patient Matching Algorithm Challenge Test Dataset\.(?P<segment>[^.]+)\.csv$"
)

_seen_notes: set[str] = set()


def _provenance_note(source: str, text: str) -> None:
    """Print a one-line provenance/terms note — once per source per process."""
    if source in _seen_notes:
        return
    _seen_notes.add(source)
    print(f"[er_lab.data] {source}: {text}")


def _download(url: str, dest: Path, *, timeout: int = 300) -> Path:
    """Stream ``url`` to ``dest`` if absent (atomic via a .part temp file)."""
    if dest.exists():
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_name(dest.name + ".part")
    with requests.get(url, stream=True, timeout=timeout) as resp:
        resp.raise_for_status()
        with open(tmp, "wb") as fh:
            fh.writelines(resp.iter_content(1 << 20))
    tmp.replace(dest)
    return dest


def _read_csv_verbatim(path: Path) -> pd.DataFrame:
    # dtype=str + keep_default_na=False: '', 'NA', 'NULL', ... arrive verbatim, never NaN.
    return pd.read_csv(path, dtype=str, keep_default_na=False)


def _schema(name: str, schema_path: str | Path | None) -> DeclaredSchema:
    path = Path(schema_path) if schema_path is not None else SCHEMA_DIR / f"{name}.yaml"
    if not path.exists():
        raise FileNotFoundError(f"declared-schema yaml for '{name}' not found: {path}")
    return DeclaredSchema.from_yaml(path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        while chunk := fh.read(1 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def _checksum_gate(path: Path) -> None:
    """Record a ``<file>.sha256`` sidecar on first load; verify against it afterwards."""
    sidecar = path.with_name(path.name + ".sha256")
    digest = _sha256(path)
    if not sidecar.exists():
        sidecar.write_text(digest + "\n")
        return
    recorded = sidecar.read_text().strip()
    if digest != recorded:
        raise ValueError(
            f"sha256 mismatch for {path}: recorded {recorded} (in {sidecar.name}), "
            f"got {digest}. The file changed since first load — if that is intentional, "
            f"delete the sidecar and reload."
        )


def _gated_files(raw_dir: Path, patterns: tuple[str, ...], missing_msg: str) -> list[Path]:
    """Checksum-gated local files under ``raw_dir``; FileNotFoundError with instructions if none."""
    files = sorted(p for pat in patterns for p in raw_dir.glob(pat) if p.is_file())
    if not files:
        raise FileNotFoundError(missing_msg)
    for path in files:
        _checksum_gate(path)
    return files


# --------------------------------------------------------------------------- splink_datasets

_SPLINK_TERMS = (
    "splink_datasets (moj-analytical-services): openly downloadable; fake_1000 is synthetic, "
    "historical_50k is Wikidata-derived with injected errors; fetched at run time for "
    "quickstart and CI fixtures (DATA_GOVERNANCE.md)."
)


def load_fake_1000(
    data_root: str | Path, *, schema_path: str | Path | None = None
) -> tuple[pd.DataFrame, DeclaredSchema]:
    """The 1000-row synthetic splink quickstart corpus ('cluster' is the truth key)."""
    _provenance_note("splink_datasets", _SPLINK_TERMS)
    dest = Path(data_root) / "splink_datasets" / "fake_1000.csv"
    _download(f"{SPLINK_RAW_BASE}/fake_1000.csv", dest)
    schema = _schema("fake_1000", schema_path)
    return schema.to_canonical(_read_csv_verbatim(dest)), schema


def load_historical_50k(
    data_root: str | Path, *, schema_path: str | Path | None = None
) -> tuple[pd.DataFrame, DeclaredSchema]:
    """The ~50k Wikidata-derived historical-persons corpus ('cluster' is the truth key)."""
    _provenance_note("splink_datasets", _SPLINK_TERMS)
    dest = Path(data_root) / "splink_datasets" / "historical_figures_with_errors_50k.parquet"
    _download(f"{SPLINK_RAW_BASE}/historical_figures_with_errors_50k.parquet", dest)
    schema = _schema("historical_50k", schema_path)
    return schema.to_canonical(pd.read_parquet(dest)), schema


# --------------------------------------------------------------------------- ONC patient matching

_ONC_TERMS = (
    "ONC patient-matching challenge (github.com/onc-healthit/patient-matching): no upstream "
    "LICENSE — research-use only, never redistributed, within-lab labels only; DOB is an "
    "Excel-style serial integer kept verbatim (DATA_GOVERNANCE.md)."
)


def _onc_entries() -> list[dict]:
    """List the repo's files via the GitHub contents API."""
    resp = requests.get(
        ONC_CONTENTS_URL, headers={"Accept": "application/vnd.github+json"}, timeout=60
    )
    resp.raise_for_status()
    entries = resp.json()
    if isinstance(entries, dict):  # an API error payload (rate limit, moved repo, ...)
        # not a Python type bug, so RuntimeError, not TypeError:
        raise RuntimeError(  # noqa: TRY004
            f"GitHub contents API error: {entries.get('message', entries)!r}"
        )
    return entries


def _onc_select(entries: list[dict], letters: list[str] | None) -> list[dict]:
    """Pick the challenge CSVs (8 alphabetical segments + 'Null', the null-last-name file).

    ``letters`` are segment labels matched case-insensitively against the label between
    the last two dots of the filename (e.g. 'A-C', 'J-mid L', 'Null'); None means all.
    """
    by_segment: dict[str, dict] = {}
    order: list[str] = []
    for entry in entries:
        match = _ONC_FILE_RE.match(entry.get("name", ""))
        if match:
            segment = match.group("segment")
            by_segment[segment.lower()] = entry
            order.append(segment)
    if letters is None:
        return [by_segment[segment.lower()] for segment in order]
    selected = []
    for wanted in letters:
        entry = by_segment.get(str(wanted).lower())
        if entry is None:
            raise ValueError(f"unknown ONC segment {wanted!r}; available segments: {order}")
        selected.append(entry)
    return selected


def load_onc(
    data_root: str | Path,
    *,
    letters: list[str] | None = None,
    schema_path: str | Path | None = None,
) -> tuple[pd.DataFrame, DeclaredSchema]:
    """ONC challenge records; ``letters`` selects last-name segments (see ``_onc_select``).

    DOB arrives as Excel-style serial integers and is delivered verbatim as a string —
    the noise is the object of study, do not convert downstream of here either.
    """
    _provenance_note("onc", _ONC_TERMS)
    selected = _onc_select(_onc_entries(), letters)
    frames = []
    for entry in selected:
        dest = Path(data_root) / "onc" / entry["name"]
        url = entry.get("download_url") or f"{ONC_RAW_BASE}/{requests.utils.quote(entry['name'])}"
        _download(url, dest)
        frames.append(_read_csv_verbatim(dest))
    df = pd.concat(frames, ignore_index=True)
    schema = _schema("onc", schema_path)
    return schema.to_canonical(df), schema


# --------------------------------------------------------------------------- checksum-gated local

_BPID_TERMS = (
    "BPID (Zenodo record 13932202, Apache-2.0): 1M synthetic PII profiles + 10k labeled "
    "pairs; user-side download, checksum-gated; never committed (DATA_GOVERNANCE.md)."
)
_OHIO_TERMS = (
    "Ohio voter file (ohiosos.gov): public records, state-law use restrictions apply; "
    "user-side download, checksum-gated; aggregates only in published artifacts "
    "(DATA_GOVERNANCE.md)."
)


def _bpid_missing_msg(raw_dir: Path) -> str:
    return (
        f"BPID data not found: no csv/parquet files under {raw_dir}.\n"
        "zenodo.org is blocked from the build container, so this download is user-side:\n"
        "  1. Download the BPID archive (Apache-2.0) from https://zenodo.org/records/13932202\n"
        f"  2. Extract its files into {raw_dir}/\n"
        "The loader records a .sha256 sidecar per file on first load and verifies it on\n"
        "later loads. See DATA_GOVERNANCE.md — the repo ships scripts and checksums, never data."
    )


def _ohio_missing_msg(raw_dir: Path) -> str:
    return (
        f"Ohio voter file not found: no data files under {raw_dir}.\n"
        "ohiosos.gov is blocked from the build container, so this download is user-side:\n"
        "  1. Download the statewide voter files from the Ohio Secretary of State portal:\n"
        "     https://www6.ohiosos.gov/ords/f?p=VOTERFTP:STWD\n"
        f"  2. Place the extracted files (.txt/.csv, gzip ok) into {raw_dir}/\n"
        "The loader records a .sha256 sidecar per file on first load and verifies it on\n"
        "later loads. See DATA_GOVERNANCE.md — the repo ships scripts and checksums, never data."
    )


def load_bpid(
    data_root: str | Path, *, schema_path: str | Path | None = None
) -> tuple[pd.DataFrame, DeclaredSchema]:
    """BPID profiles from ``data_root/raw/bpid/`` (largest file = the 1M-profile table)."""
    _provenance_note("bpid", _BPID_TERMS)
    raw_dir = Path(data_root) / "raw" / "bpid"
    files = _gated_files(raw_dir, ("*.csv", "*.parquet"), _bpid_missing_msg(raw_dir))
    records = max(files, key=lambda p: p.stat().st_size)
    df = (
        pd.read_parquet(records)
        if records.suffix == ".parquet"
        else _read_csv_verbatim(records)
    )
    schema = _schema("bpid", schema_path)
    return schema.to_canonical(df), schema


def load_ohio(
    data_root: str | Path, *, schema_path: str | Path | None = None
) -> tuple[pd.DataFrame, DeclaredSchema]:
    """Ohio statewide voter file from ``data_root/raw/ohio/`` (row-partitioned files concatenated)."""
    _provenance_note("ohio", _OHIO_TERMS)
    raw_dir = Path(data_root) / "raw" / "ohio"
    files = _gated_files(
        raw_dir, ("*.csv", "*.txt", "*.csv.gz", "*.txt.gz"), _ohio_missing_msg(raw_dir)
    )
    df = pd.concat([_read_csv_verbatim(path) for path in files], ignore_index=True)
    schema = _schema("ohio", schema_path)
    return schema.to_canonical(df), schema
