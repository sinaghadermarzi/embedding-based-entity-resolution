"""er_lab.data.nc: listing-XML parsing, download resume, streaming parse, ncid alignment."""

from __future__ import annotations

import http.server
import re
import threading
import zipfile
from pathlib import Path
from typing import ClassVar

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import requests

from er_lab.data import nc

# ------------------------------------------------------------------ S3 listing XML (canned)

LISTING_TRUNCATED = """<?xml version="1.0" encoding="UTF-8"?>
<ListBucketResult xmlns="http://s3.amazonaws.com/doc/2006-03-01/">
  <Name>dl.ncsbe.gov</Name><Prefix>data/Snapshots/</Prefix>
  <NextContinuationToken>tok/abc+123=</NextContinuationToken>
  <KeyCount>4</KeyCount><MaxKeys>4</MaxKeys><IsTruncated>true</IsTruncated>
  <Contents><Key>data/Snapshots/End-of-Year VR Snapshots/2016VRbyCongDist.pdf</Key>
    <Size>167696</Size></Contents>
  <Contents><Key>data/Snapshots/VR_Snapshot_20051125.zip</Key><Size>599981040</Size></Contents>
  <Contents><Key>data/Snapshots/layout_VR_Snapshot.txt</Key><Size>5074</Size></Contents>
  <Contents><Key>data/Snapshots/VR_Snapshot_20260303.zip</Key><Size>1300286001</Size></Contents>
</ListBucketResult>"""

LISTING_FINAL = """<?xml version="1.0" encoding="UTF-8"?>
<ListBucketResult xmlns="http://s3.amazonaws.com/doc/2006-03-01/">
  <Name>dl.ncsbe.gov</Name><Prefix>data/Snapshots/</Prefix>
  <KeyCount>1</KeyCount><MaxKeys>1000</MaxKeys><IsTruncated>false</IsTruncated>
  <Contents><Key>data/Snapshots/VR_Snapshot_20100101.zip</Key><Size>682893824</Size></Contents>
</ListBucketResult>"""


def test_parse_listing_filters_and_extracts():
    entries, token = nc._parse_listing(LISTING_TRUNCATED)
    assert entries == [
        (
            "20051125",
            "https://s3.amazonaws.com/dl.ncsbe.gov/data/Snapshots/VR_Snapshot_20051125.zip",
            599981040,
        ),
        (
            "20260303",
            "https://s3.amazonaws.com/dl.ncsbe.gov/data/Snapshots/VR_Snapshot_20260303.zip",
            1300286001,
        ),
    ]
    assert token == "tok/abc+123="


def test_parse_listing_final_page_has_no_token():
    entries, token = nc._parse_listing(LISTING_FINAL)
    assert [e[0] for e in entries] == ["20100101"]
    assert token is None


# ------------------------------------------------------------------ download skip/resume (local)


class _RangeHandler(http.server.BaseHTTPRequestHandler):
    """Serves one payload with HEAD + Range support, recording each request."""

    payload: ClassVar[bytes] = b""
    requests_seen: ClassVar[list[tuple[str, str | None]]] = []

    def do_HEAD(self):
        self.send_response(200)
        self.send_header("Content-Length", str(len(self.payload)))
        self.send_header("Accept-Ranges", "bytes")
        self.end_headers()

    def do_GET(self):
        range_header = self.headers.get("Range")
        type(self).requests_seen.append(("GET", range_header))
        if range_header:
            start = int(range_header.removeprefix("bytes=").split("-")[0])
            body, status = self.payload[start:], 206
        else:
            body, status = self.payload, 200
        self.send_response(status)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args):  # keep test output quiet
        pass


def test_download_snapshot_skips_complete_and_resumes_partial(tmp_path, monkeypatch):
    _RangeHandler.payload = bytes(range(256)) * 1024  # 256 KiB, position-dependent bytes
    _RangeHandler.requests_seen = []
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _RangeHandler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        monkeypatch.setattr(nc, "BUCKET_URL", f"http://127.0.0.1:{server.server_port}")
        dest = nc.download_snapshot("20200101", tmp_path, progress=False)
        assert dest.name == "VR_Snapshot_20200101.zip"
        assert dest.read_bytes() == _RangeHandler.payload

        before = dest.stat().st_mtime_ns
        nc.download_snapshot("20200101", tmp_path, progress=False)
        assert dest.stat().st_mtime_ns == before  # size matched -> no re-download

        with open(dest, "r+b") as fh:
            fh.truncate(1000)
        out = nc.download_snapshot("20200101", tmp_path, progress=False)
        assert out.read_bytes() == _RangeHandler.payload  # resumed bytes land correctly
        assert _RangeHandler.requests_seen[-1] == ("GET", "bytes=1000-")
    finally:
        server.shutdown()


def test_snapshot_url_rejects_non_dates():
    with pytest.raises(ValueError, match="YYYYMMDD"):
        nc.snapshot_url("2020-01-01")


# ------------------------------------------------------------------ synthetic snapshot fixture

HEADER = ["snapshot_dt", "county_id", "voter_reg_num", "ncid", "last_name", "first_name", "zip_code"]

ROWS_A = [
    ["2020-01-01", "  1", "9001", "AA1", "SMITH  ", "MARY ", "27510"],
    ["2020-01-01", "  1", "9002", "AA2", "O'NEAL", "JOHN", "27511"],
    ["2020-01-01", " 92", "9003", "AA3", "LEE", "ANA", "27601"],
]
ROWS_B = [
    ["2021-01-01", "  1", "9002", "AA2", "ONEAL", "JOHN", "27511"],
    ["2021-01-01", " 92", "9003", "AA3", "LEE", "ANA", "27601"],
    ["2021-01-01", " 92", "9004", "AA4", "KIM", "SOO", "27605"],
]


def _snapshot_zip(
    tmp_path: Path, stem: str, rows: list[list[str]], *, encoding: str = "utf-16"
) -> Path:
    """A miniature VR snapshot: tab-delimited UTF-16 LE (BOM) text inside a zip."""
    text = "\r\n".join("\t".join(r) for r in [HEADER, *rows]) + "\r\n"
    path = tmp_path / f"{stem}.zip"
    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr(f"{stem}.txt", text.encode(encoding))
    return path


def test_parse_snapshot_verbatim(tmp_path):
    zip_path = _snapshot_zip(tmp_path, "VR_Snapshot_20200101", ROWS_A)
    out = nc.parse_snapshot(zip_path, tmp_path / "a.parquet")
    table = pq.read_table(out)
    assert table.column_names == HEADER
    assert all(t == pa.string() for t in table.schema.types)
    df = table.to_pandas()
    assert len(df) == 3
    # verbatim: padding, apostrophes, and leading spaces survive untouched
    assert df["last_name"].tolist() == ["SMITH  ", "O'NEAL", "LEE"]
    assert df["first_name"][0] == "MARY "
    assert df["county_id"].tolist() == ["  1", "  1", " 92"]


def test_parse_snapshot_utf16le_without_bom(tmp_path):
    zip_path = _snapshot_zip(tmp_path, "VR_Snapshot_20200101", ROWS_A, encoding="utf-16-le")
    df = pq.read_table(nc.parse_snapshot(zip_path, tmp_path / "nobom.parquet")).to_pandas()
    assert list(df.columns) == HEADER  # no BOM garbage in the first header cell
    assert df["ncid"].tolist() == ["AA1", "AA2", "AA3"]


def test_parse_snapshot_county_and_column_filters(tmp_path):
    zip_path = _snapshot_zip(tmp_path, "VR_Snapshot_20200101", ROWS_A)
    out = nc.parse_snapshot(
        zip_path,
        tmp_path / "county1.parquet",
        counties=[1],  # integer compare: 1 matches the file's space-padded '  1'
        columns=["ncid", "first_name", "county_id"],
    )
    df = pq.read_table(out).to_pandas()
    assert list(df.columns) == ["ncid", "first_name", "county_id"]  # requested order
    assert df["ncid"].tolist() == ["AA1", "AA2"]
    assert df["county_id"].tolist() == ["  1", "  1"]


def test_parse_snapshot_chunked_writes(tmp_path, monkeypatch):
    monkeypatch.setattr(nc, "_CHUNK_ROWS", 2)
    zip_path = _snapshot_zip(tmp_path, "VR_Snapshot_20200101", ROWS_A)
    out = nc.parse_snapshot(zip_path, tmp_path / "chunked.parquet")
    pf = pq.ParquetFile(out)
    assert pf.metadata.num_row_groups == 2  # 3 rows at chunk size 2
    assert pf.metadata.num_rows == 3


def test_parse_snapshot_unknown_column_raises(tmp_path):
    zip_path = _snapshot_zip(tmp_path, "VR_Snapshot_20200101", ROWS_A)
    with pytest.raises(ValueError, match="not_a_column"):
        nc.parse_snapshot(zip_path, tmp_path / "x.parquet", columns=["ncid", "not_a_column"])


def test_align_pair(tmp_path):
    a = nc.parse_snapshot(
        _snapshot_zip(tmp_path, "VR_Snapshot_20200101", ROWS_A), tmp_path / "a.parquet"
    )
    b = nc.parse_snapshot(
        _snapshot_zip(tmp_path, "VR_Snapshot_20210101", ROWS_B), tmp_path / "b.parquet"
    )
    df = nc.align_pair(a, b)
    other = [c for c in HEADER if c != "ncid"]
    assert list(df.columns) == ["ncid"] + [f"{c}_a" for c in other] + [f"{c}_b" for c in other]
    assert df["ncid"].tolist() == ["AA2", "AA3"]  # intersection only, ncid-sorted
    # the same-person temporal diff NSE-01 audits:
    row = df[df["ncid"] == "AA2"].iloc[0]
    assert row["last_name_a"] == "O'NEAL"
    assert row["last_name_b"] == "ONEAL"


def test_align_pair_requires_ncid(tmp_path):
    good = tmp_path / "good.parquet"
    bad = tmp_path / "bad.parquet"
    pd.DataFrame({"ncid": ["AA1"], "x": ["1"]}).to_parquet(good)
    pd.DataFrame({"y": ["2"]}).to_parquet(bad)
    with pytest.raises(ValueError, match="ncid"):
        nc.align_pair(good, bad)


# ------------------------------------------------------------------ live bucket (network)


@pytest.mark.network
def test_list_snapshots_live():
    snapshots = nc.list_snapshots()
    assert len(snapshots) >= 70
    assert snapshots == sorted(snapshots)
    for date, url, size in snapshots:
        assert re.fullmatch(r"\d{8}", date)
        assert 2005 <= int(date[:4]) <= 2035
        assert 1 <= int(date[4:6]) <= 12
        assert url == nc.snapshot_url(date)
        assert size > 1_000_000


@pytest.mark.network
def test_snapshot_url_serves_zip_ranges():
    _date, url, _size = nc.list_snapshots()[0]
    resp = requests.get(url, headers={"Range": "bytes=0-3"}, timeout=60)
    assert resp.status_code == 206  # resumable downloads depend on Range support
    assert resp.content == b"PK\x03\x04"  # a real zip lives at the constructed URL
