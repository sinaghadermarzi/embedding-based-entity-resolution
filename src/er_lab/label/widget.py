"""In-notebook pair adjudication: queue, JSONL persistence, inter-rater stats (PLAN §5/§10).

The MET-05 truth audit and the target-tier sign-off both require a human to
label ~200-500 stratified candidate pairs. This module keeps that honest and
reproducible:

- :class:`PairQueue` presents pairs in a DETERMINISTIC order (canonical a<b
  orientation, then sorted by (a, b), duplicates dropped) so two annotators
  looking at "the queue" see the same sequence regardless of how the pairs
  frame was assembled, and a re-run resumes identically.
- Labels persist as append-only JSONL (one object per decision, with
  annotator and UTC timestamp) — never mutated in place, so disagreement and
  changes-of-mind stay visible; :func:`read_labels` returns the full history
  and :func:`latest_labels` the per-(pair, annotator) last word.
- :func:`cohens_kappa` is the agreement statistic the MET-05 gate quotes.
- :class:`AdjudicationWidget` is a thin ipywidgets front-end over the queue;
  it owns no state beyond a cursor — every decision goes straight to JSONL.

Everything except the widget itself is plain pandas/stdlib and fully testable
headless; the widget is smoke-tested construct-only.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

__all__ = [
    "AdjudicationWidget",
    "PairQueue",
    "cohens_kappa",
    "latest_labels",
    "read_labels",
]

#: the closed label vocabulary — 'unsure' is a first-class outcome (MET-06:
#: some pairs are genuinely unresolvable and must be countable, not skipped)
LABELS = ("match", "nonmatch", "unsure")

PAIR_SEP = "||"  # pair_id = f"{a}||{b}" with a < b


class PairQueue:
    """Deterministic adjudication queue over candidate pairs.

    *pairs_df* needs columns ``a`` and ``b`` (record ids); extra columns (e.g.
    ``prob``, stratum tags) ride along on each item. *records_df* is the
    canonical frame the ids point into — either indexed by record id or
    carrying a ``record_id`` column; ids on both sides are normalized to str
    at construction, so integer record ids work throughout. *path*, when
    given, is the JSONL file :meth:`record_label` appends to.

    Iterating yields one dict per pair: ``pair_id``, ``a``, ``b``,
    ``record_a`` / ``record_b`` (the two records as plain dicts), plus any
    extra pair columns.
    """

    def __init__(
        self,
        pairs_df: pd.DataFrame,
        records_df: pd.DataFrame,
        *,
        path: str | Path | None = None,
    ):
        missing = {"a", "b"} - set(pairs_df.columns)
        if missing:
            raise KeyError(f"pairs_df missing columns {sorted(missing)}")
        records = records_df
        if "record_id" in records.columns:
            records = records.set_index("record_id")
        # normalize ids to str ONCE so validation and .loc lookups agree on type
        # (set_axis: the caller's frame keeps its original index untouched)
        records = records.set_axis(records.index.astype(str), axis=0)
        if not records.index.is_unique:
            raise ValueError("records_df has duplicate record ids")
        self.records = records

        pairs = pairs_df.copy()
        pairs["a"] = pairs["a"].astype(str)
        pairs["b"] = pairs["b"].astype(str)
        if (pairs["a"] == pairs["b"]).any():
            raise ValueError("pairs_df contains self-pairs (a == b)")
        swap = pairs["a"] > pairs["b"]  # canonical orientation: a < b
        pairs.loc[swap, ["a", "b"]] = pairs.loc[swap, ["b", "a"]].to_numpy()
        pairs = pairs.drop_duplicates(subset=["a", "b"])
        pairs = pairs.sort_values(["a", "b"]).reset_index(drop=True)
        known = set(records.index)
        unknown = (set(pairs["a"]) | set(pairs["b"])) - known
        if unknown:
            raise ValueError(f"pairs reference ids absent from records: {sorted(unknown)[:5]}")
        self.pairs = pairs
        self.path = None if path is None else Path(path)

    def __len__(self) -> int:
        return len(self.pairs)

    def __iter__(self):
        for _, row in self.pairs.iterrows():
            yield self.item(row["a"], row["b"])

    def item(self, a: str, b: str) -> dict:
        """The adjudication payload for one pair (also what iteration yields)."""
        row = self.pairs[(self.pairs["a"] == a) & (self.pairs["b"] == b)]
        if row.empty:
            raise KeyError(f"pair ({a}, {b}) is not in the queue")
        extra = row.iloc[0].drop(["a", "b"]).to_dict()
        return {
            "pair_id": f"{a}{PAIR_SEP}{b}",
            "a": a,
            "b": b,
            "record_a": self.records.loc[a].to_dict(),
            "record_b": self.records.loc[b].to_dict(),
            **extra,
        }

    def pair_ids(self) -> list[str]:
        """All pair ids in queue order."""
        return [f"{a}{PAIR_SEP}{b}" for a, b in zip(self.pairs["a"], self.pairs["b"])]

    def record_label(self, pair_id: str, label: str, annotator: str) -> dict:
        """Append one decision to the JSONL file and return the written row.

        Validates the pair is actually in this queue and the label is in the
        closed vocabulary ``LABELS``. Append-only: relabeling writes a new
        line; :func:`latest_labels` resolves to the last word per
        (pair, annotator).
        """
        if self.path is None:
            raise ValueError("this queue has no path — construct PairQueue(..., path=...)")
        if label not in LABELS:
            raise ValueError(f"unknown label {label!r}; expected one of {LABELS}")
        if not annotator or not isinstance(annotator, str):
            raise ValueError("annotator must be a non-empty string")
        if pair_id not in set(self.pair_ids()):
            raise KeyError(f"pair_id {pair_id!r} is not in this queue")
        row = {
            "pair_id": pair_id,
            "label": label,
            "annotator": annotator,
            "at": datetime.now(UTC).isoformat(),
        }
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a") as fh:
            fh.write(json.dumps(row) + "\n")
        return row

    def pending(self, annotator: str) -> list[str]:
        """Pair ids *annotator* has not labeled yet (queue order) — the resume point."""
        done: set[str] = set()
        if self.path is not None and self.path.exists():
            labels = read_labels(self.path)
            done = set(labels.loc[labels["annotator"] == annotator, "pair_id"])
        return [pid for pid in self.pair_ids() if pid not in done]


def read_labels(path: str | Path) -> pd.DataFrame:
    """The full JSONL decision history as a DataFrame (file order preserved)."""
    rows = []
    with Path(path).open() as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return pd.DataFrame(rows, columns=["pair_id", "label", "annotator", "at"])


def latest_labels(labels: pd.DataFrame) -> pd.DataFrame:
    """Last decision per (pair_id, annotator) — changes of mind resolved, history kept."""
    return labels.drop_duplicates(subset=["pair_id", "annotator"], keep="last")


def cohens_kappa(labels_a: pd.Series, labels_b: pd.Series) -> float:
    """Cohen's kappa between two annotators' labels over the same pairs.

    Series are aligned on their (pair-id) index — only pairs both annotators
    labeled count. kappa = (p_o - p_e) / (1 - p_e) with p_o the observed
    agreement rate and p_e the chance agreement from the two marginal label
    distributions. Degenerate case p_e = 1 (both annotators constant on the
    same label): returns 1.0 — agreement is perfect even though chance
    'explains' it; the sample is simply uninformative and the caller should
    stratify harder.
    """
    common = labels_a.index.intersection(labels_b.index)
    if len(common) == 0:
        raise ValueError("no common pairs between the two annotators")
    a = labels_a.loc[common].astype(str)
    b = labels_b.loc[common].astype(str)
    if a.isna().any() or b.isna().any():
        raise ValueError("labels must be non-null on all common pairs")
    p_o = float((a.to_numpy() == b.to_numpy()).mean())
    cats = sorted(set(a) | set(b))
    p_e = sum(float((a == c).mean()) * float((b == c).mean()) for c in cats)
    if p_e >= 1.0:
        return 1.0
    return (p_o - p_e) / (1.0 - p_e)


class AdjudicationWidget:
    """ipywidgets front-end over a :class:`PairQueue` — construct, then display.

    Layout: the two records side by side as an HTML table, one button per
    label in ``LABELS``, and a progress caption. Clicking a button persists
    via ``queue.record_label`` and advances to the annotator's next pending
    pair, so a closed-and-reopened notebook resumes where it left off.

    ipywidgets is imported lazily here: the queue/kappa machinery must work
    headless (tests, CI) without a widget frontend.
    """

    def __init__(self, queue: PairQueue, *, annotator: str = "anon"):
        import ipywidgets as w  # lazy: only the UI needs it

        if queue.path is None:
            raise ValueError("AdjudicationWidget needs a queue with a JSONL path to persist to")
        self.queue = queue
        self.annotator = annotator
        self._pending = queue.pending(annotator)

        self._pair_html = w.HTML()
        self._progress = w.HTML()
        self._buttons = [
            w.Button(description=lbl, layout=w.Layout(width="110px")) for lbl in LABELS
        ]
        for btn in self._buttons:
            btn.on_click(self._on_click)
        self.box = w.VBox([self._progress, self._pair_html, w.HBox(self._buttons)])
        self._render()

    # -- behavior ------------------------------------------------------------

    def _on_click(self, btn) -> None:
        if not self._pending:
            return
        self.queue.record_label(self._pending[0], btn.description, self.annotator)
        self._pending = self._pending[1:]
        self._render()

    def _render(self) -> None:
        total = len(self.queue)
        done = total - len(self._pending)
        self._progress.value = f"<b>{done}/{total}</b> labeled (annotator: {self.annotator})"
        if not self._pending:
            self._pair_html.value = "<i>Queue complete.</i>"
            return
        a, b = self._pending[0].split(PAIR_SEP, 1)
        item = self.queue.item(a, b)
        ra, rb = item["record_a"], item["record_b"]
        fields = sorted(set(ra) | set(rb))
        rows = "".join(
            f"<tr><td><b>{f}</b></td><td>{_esc(ra.get(f))}</td><td>{_esc(rb.get(f))}</td></tr>"
            for f in fields
        )
        self._pair_html.value = (
            f"<table><tr><th></th><th>{_esc(a)}</th><th>{_esc(b)}</th></tr>{rows}</table>"
        )

    def _ipython_display_(self) -> None:
        from IPython.display import display

        display(self.box)


def _esc(value) -> str:
    """Minimal HTML escaping for record values shown in the widget."""
    if value is None or value is pd.NA or (isinstance(value, float) and pd.isna(value)):
        return "<i>&mdash;</i>"
    return str(value).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
