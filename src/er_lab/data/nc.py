"""NC voter-registration snapshots: list, download, stream-parse, and align (PLAN.md §4).

The NCSBE publishes ~72 statewide snapshots (2005–2026) as zip files on the
``dl.ncsbe.gov`` S3 bucket. Each zip holds one tab-delimited UTF-16 LE text file
(~4 GB decompressed, header row first) keyed statewide by ``ncid``. This module
never loads a snapshot whole: parsing streams 50k-row chunks into parquet, and
``align_pair`` joins two parsed snapshots on ncid in DuckDB — the same-person
temporal diffs that feed the NSE-01 noise audit. Values are delivered verbatim
as strings (upstream standardization is an experimental factor, PRS-01).
"""

from __future__ import annotations

import csv
import io
import re
import zipfile
from pathlib import Path
from xml.etree import ElementTree

import duckdb
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import requests

from er_lab.data.loaders import _provenance_note

BUCKET_URL = "https://s3.amazonaws.com/dl.ncsbe.gov"
SNAPSHOT_PREFIX = "data/Snapshots/"

_S3_NS = "{http://s3.amazonaws.com/doc/2006-03-01/}"
_SNAPSHOT_KEY_RE = re.compile(r"VR_Snapshot_(\d{8})\.zip$")
_CHUNK_ROWS = 50_000
_STREAM_BYTES = 1 << 20
_PROGRESS_EVERY = 64 << 20

_NC_TERMS = (
    "NC voter registration snapshots (dl.ncsbe.gov): public records under N.C.G.S. 132-1 / "
    "163-82.10; the statutes govern permitted uses; records never redistributed — published "
    "artifacts carry aggregates only (DATA_GOVERNANCE.md)."
)


def _parse_listing(xml_text: str) -> tuple[list[tuple[str, str, int]], str | None]:
    """One ListObjectsV2 page -> ([(YYYYMMDD, url, size_bytes), ...], continuation token)."""
    root = ElementTree.fromstring(xml_text)
    entries: list[tuple[str, str, int]] = []
    for contents in root.iter(f"{_S3_NS}Contents"):
        key = contents.findtext(f"{_S3_NS}Key") or ""
        match = _SNAPSHOT_KEY_RE.search(key)
        if not match:  # layout docs, PDFs, folder keys, ...
            continue
        size = int(contents.findtext(f"{_S3_NS}Size") or 0)
        entries.append((match.group(1), f"{BUCKET_URL}/{key}", size))
    token = None
    if (root.findtext(f"{_S3_NS}IsTruncated") or "").lower() == "true":
        token = root.findtext(f"{_S3_NS}NextContinuationToken")
    return entries, token


def list_snapshots() -> list[tuple[str, str, int]]:
    """All statewide snapshots on the bucket, as (date YYYYMMDD, url, size_bytes), sorted."""
    _provenance_note("nc_voter", _NC_TERMS)
    entries: list[tuple[str, str, int]] = []
    token: str | None = None
    while True:
        params = {"list-type": "2", "prefix": SNAPSHOT_PREFIX}
        if token is not None:
            params["continuation-token"] = token
        resp = requests.get(BUCKET_URL, params=params, timeout=60)
        resp.raise_for_status()
        page, token = _parse_listing(resp.text)
        entries.extend(page)
        if token is None:
            return sorted(entries)


def snapshot_url(date: str) -> str:
    if not re.fullmatch(r"\d{8}", date):
        raise ValueError(f"date must be YYYYMMDD, got {date!r}")
    return f"{BUCKET_URL}/{SNAPSHOT_PREFIX}VR_Snapshot_{date}.zip"


def download_snapshot(date: str, dest_dir: str | Path, *, progress: bool = True) -> Path:
    """Download one snapshot zip: skips when complete, resumes via Range when supported."""
    _provenance_note("nc_voter", _NC_TERMS)
    url = snapshot_url(date)
    dest = Path(dest_dir) / f"VR_Snapshot_{date}.zip"
    dest.parent.mkdir(parents=True, exist_ok=True)

    head = requests.head(url, timeout=60)
    head.raise_for_status()
    total = int(head.headers["Content-Length"])
    resumable = head.headers.get("Accept-Ranges") == "bytes"

    done = dest.stat().st_size if dest.exists() else 0
    if done == total:
        return dest
    headers, mode = {}, "wb"
    if 0 < done < total and resumable:
        headers, mode = {"Range": f"bytes={done}-"}, "ab"
    else:
        done = 0

    with requests.get(url, headers=headers, stream=True, timeout=60) as resp:
        resp.raise_for_status()
        if mode == "ab" and resp.status_code != 206:  # server ignored the range after all
            mode, done = "wb", 0
        next_mark = 0
        with open(dest, mode) as fh:
            for chunk in resp.iter_content(_STREAM_BYTES):
                fh.write(chunk)
                done += len(chunk)
                if progress and done >= next_mark:
                    print(f"\r  VR_Snapshot_{date}.zip: {done >> 20} / {total >> 20} MiB",
                          end="", flush=True)
                    next_mark = done + _PROGRESS_EVERY
        if progress:
            print(f"\r  VR_Snapshot_{date}.zip: {done >> 20} / {total >> 20} MiB")

    size = dest.stat().st_size
    if size != total:
        raise OSError(f"incomplete download of {url}: {size} of {total} bytes")
    return dest


def _county_matches(cell: str, wanted: set[int]) -> bool:
    try:
        return int(cell) in wanted  # int() tolerates the files' space padding
    except ValueError:  # empty / masked values
        return False


def _chunk_table(rows: list[list[str]], schema: pa.Schema) -> pa.Table:
    arrays = [pa.array([row[j] for row in rows], type=pa.string()) for j in range(len(schema))]
    return pa.Table.from_arrays(arrays, schema=schema)


def parse_snapshot(
    zip_path: str | Path,
    out_parquet: str | Path,
    *,
    counties: list[int] | None = None,
    columns: list[str] | None = None,
) -> Path:
    """Stream the zip's inner UTF-16 LE TSV into parquet, all values verbatim strings.

    The header comes from the file's first row. ``counties`` filters on the
    ``county_id`` column (integer comparison, so '92' matches ' 92'); ``columns``
    selects a subset, written in the order given. 50k-row chunks — the ~4 GB
    member is never held in memory.
    """
    _provenance_note("nc_voter", _NC_TERMS)
    out_parquet = Path(out_parquet)
    out_parquet.parent.mkdir(parents=True, exist_ok=True)
    wanted = {int(c) for c in counties} if counties is not None else None

    with zipfile.ZipFile(zip_path) as zf:
        member = max((i for i in zf.infolist() if not i.is_dir()), key=lambda i: i.file_size)
        with zf.open(member) as raw:
            # The files are UTF-16 LE; honor (and strip) a BOM when one is present.
            bom = raw.peek(2)[:2]
            encoding = "utf-16" if bom in (b"\xff\xfe", b"\xfe\xff") else "utf-16-le"
            text = io.TextIOWrapper(raw, encoding=encoding, newline="")
            reader = csv.reader(text, delimiter="\t")
            header = next(reader)

            out_cols = list(header) if columns is None else list(columns)
            missing = [c for c in out_cols if c not in header]
            if missing:
                raise ValueError(f"columns not in snapshot header {header}: {missing}")
            out_idx = [header.index(c) for c in out_cols]
            county_idx: int | None = None
            if wanted is not None:
                if "county_id" not in header:
                    raise ValueError("cannot filter counties: snapshot has no 'county_id' column")
                county_idx = header.index("county_id")

            schema = pa.schema([pa.field(c, pa.string()) for c in out_cols])
            rows: list[list[str]] = []
            wrote = False
            with pq.ParquetWriter(out_parquet, schema) as writer:
                for row in reader:
                    if not row:  # trailing blank line
                        continue
                    if len(row) < len(header):
                        row = row + [""] * (len(header) - len(row))
                    if county_idx is not None and not _county_matches(row[county_idx], wanted):
                        continue
                    rows.append([row[i] for i in out_idx])
                    if len(rows) >= _CHUNK_ROWS:
                        writer.write_table(_chunk_table(rows, schema))
                        rows, wrote = [], True
                if rows or not wrote:
                    writer.write_table(_chunk_table(rows, schema))
    return out_parquet


def align_pair(parquet_a: str | Path, parquet_b: str | Path) -> pd.DataFrame:
    """Inner-join two parsed snapshots on ncid: (ncid, <cols>_a, <cols>_b), ncid-sorted.

    Rows are the records present in both snapshots — the same-person temporal
    pairs whose field diffs the NSE-01 noise audit measures.
    """
    cols_a = pq.read_schema(parquet_a).names
    cols_b = pq.read_schema(parquet_b).names
    for label, cols in (("parquet_a", cols_a), ("parquet_b", cols_b)):
        if "ncid" not in cols:
            raise ValueError(f"{label} has no 'ncid' column; align_pair joins on ncid")

    select = ", ".join(
        ['a."ncid" AS ncid']
        + [f'a."{c}" AS "{c}_a"' for c in cols_a if c != "ncid"]
        + [f'b."{c}" AS "{c}_b"' for c in cols_b if c != "ncid"]
    )
    path_a, path_b = (str(Path(p)).replace("'", "''") for p in (parquet_a, parquet_b))
    query = (
        f"SELECT {select} FROM read_parquet('{path_a}') a "
        f"JOIN read_parquet('{path_b}') b ON a.ncid = b.ncid ORDER BY ncid"
    )
    con = duckdb.connect()
    try:
        return con.execute(query).df()
    finally:
        con.close()
