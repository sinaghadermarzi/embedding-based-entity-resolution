"""CI honesty audit over the executed notebook series — the promotion of NB17 §8.

This is the standalone form of notebook 17's §8 in-series audit (see
``notebooks/src/17_verdict_at_1e7.py``), with its semantics kept exactly: walk
every executed ``notebooks/*.ipynb`` on disk and count the lab's honesty rails —

- **figures**: ``image/png`` outputs per notebook, with per-cell PNG-vs-stamp
  accounting — a cell's PNGs count as *stamped* only up to its number of
  ``er_lab.reporting.figures`` calls (which stamp the artifact-provenance
  caption unconditionally), so one raw-plt figure hiding next to a stamped one
  in the same cell is flagged rather than riding along; any excess PNG is a
  raw-figure rail breach.
- **RUN-IN-TARGET**: matches of the ``# [RUN-IN-TARGET <label>]`` placard
  pattern in code-cell sources, labels extracted (source-based, so this count
  is identical whether a notebook's outputs are present or not).
- **cards / verdicts**: rendered conjecture-card markdown outputs, and verdict
  boxes parsed from rendered markdown and cross-checked against the registered
  immutable card set (``artifacts/card_*``) — a verdict without a registered
  card is post-hoc storytelling and fails the audit.
- **errors**: ``output_type == 'error'`` cells — must be zero everywhere.

Exit code 0 iff every notebook passes every rail (no error outputs, no
unstamped figure, no cardless verdict); 1 otherwise.

Semantics note vs the registered ``honesty_audit`` artifact: NB17's §8 cell
runs *while NB17 itself executes*, so its own row (``self_scan``) reflects the
on-disk file BEFORE that execution — zero outputs on a fresh build (the run
whose series totals BUILD_STATE quotes: 69/69 figures, 36 cards, 36 verdict
boxes). This tool runs after the fact and scans the post-execution files, so
its ``17_*`` row carries that notebook's real rendered outputs; RUN-IN-TARGET
counts are source-based and agree between the two views.

Registered-card cross-check: the card set is read directly from the run
directories (``artifacts/card_*/<run>/`` with a complete ``meta.json`` — the
registry's own completeness rule; ``er_lab.infra.artifacts`` writes
``meta.json`` last, so a run without one is an aborted write). We read the
directories rather than import er_lab because the registry exposes no public
enumeration API (only name-addressed ``load``/``exists``), and a CI audit
should keep running even when the package under audit does not import.

Usage:
    uv run python tools/honesty_audit.py [--root PATH] [--json]
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

# The scanning conventions, verbatim from NB17 §8 (notebooks/src/17_verdict_at_1e7.py).
# tools/gen_run_in_target.py imports RIT_RE from here — the manifest and the audit
# must count RUN-IN-TARGET matches identically.
FIG_CALL_RE = re.compile(
    r"figures\s*\.\s*(plot_artifact|line_with_ci|scaling_curve|regime_heatmap|"
    r"three_panel_pressure)\s*\("
)
RIT_RE = re.compile(r"#\s*\[RUN-IN-TARGET\s+([^\]]+)\]")
CARD_MD_RE = re.compile(r"### Conjecture card `([^`]+)`")
VERD_RE = re.compile(r"> ### VERDICT: (CONFIRMED|REFUTED|UNEXPLAINED) — card `([^`]+)`")

COLUMNS = (
    "notebook",
    "n_code_cells",
    "n_figures",
    "n_figures_stamped",
    "n_figures_unstamped",
    "n_rit_cells",
    "rit_labels",
    "n_card_renders",
    "n_verdict_boxes",
    "n_unregistered_verdicts",
    "n_error_outputs",
    "rails_pass",
)
_SUMMED = tuple(c for c in COLUMNS if c.startswith("n_"))


def registered_card_ids(artifacts_root: Path) -> set[str]:
    """Card ids with at least one *complete* registered run under ``card_<id>/``.

    Mirrors ``ArtifactRegistry._runs`` completeness: a run directory counts only
    if it holds a ``meta.json`` (written last, atomically — its absence marks an
    aborted write).
    """
    ids: set[str] = set()
    for card_dir in sorted(artifacts_root.glob("card_*")):
        if not card_dir.is_dir():
            continue
        complete = any(
            run.is_dir() and (run / "meta.json").is_file() for run in card_dir.iterdir()
        )
        if complete:
            ids.add(card_dir.name.removeprefix("card_"))
    return ids


def audit_notebook(path: Path, registered_cards: set[str] | None) -> dict:
    """One audit row for an executed notebook — NB17 §8 semantics, exactly.

    ``registered_cards=None`` means the card cross-check is skipped (fresh clone:
    ``artifacts/`` is gitignored, so no card registry exists to check against) —
    verdict boxes are still counted but never flagged as unregistered.
    """
    nb_json = json.loads(path.read_text())
    n_code = n_png = n_stamped = n_rit = n_cards = n_verd = n_err = n_unreg = 0
    rit_labels: list[str] = []
    for cell in nb_json["cells"]:
        if cell["cell_type"] != "code":
            continue
        n_code += 1
        src = "".join(cell["source"])
        # Per-cell PNG-vs-stamp accounting: a cell's PNGs count as stamped only up
        # to its number of figures.* calls, so one raw-plt figure hiding next to a
        # stamped one in the same cell is flagged rather than riding along.
        n_fig_calls = len(FIG_CALL_RE.findall(src))
        cell_pngs = 0
        for mlab in RIT_RE.finditer(src):
            n_rit += 1
            rit_labels.append(mlab.group(1).strip())
        for out in cell.get("outputs", []):
            if out.get("output_type") == "error":
                n_err += 1
            data = out.get("data", {})
            if "image/png" in data:
                cell_pngs += 1
            md = "".join(data.get("text/markdown", []))
            n_cards += len(CARD_MD_RE.findall(md))
            for mv in VERD_RE.finditer(md):
                n_verd += 1
                if registered_cards is not None:
                    n_unreg += int(mv.group(2) not in registered_cards)
        n_png += cell_pngs
        n_stamped += min(cell_pngs, n_fig_calls)
    return {
        "notebook": path.name,
        "n_code_cells": n_code,
        "n_figures": n_png,
        "n_figures_stamped": n_stamped,
        "n_figures_unstamped": n_png - n_stamped,
        "n_rit_cells": n_rit,
        "rit_labels": ",".join(rit_labels),
        "n_card_renders": n_cards,
        "n_verdict_boxes": n_verd,
        "n_unregistered_verdicts": n_unreg,
        "n_error_outputs": n_err,
        "rails_pass": bool(n_err == 0 and n_png == n_stamped and n_unreg == 0),
    }


def audit_series(root: Path) -> tuple[list[dict], dict]:
    """Audit every ``notebooks/*.ipynb`` under *root*; return (rows, totals).

    When the artifact store holds no complete card run (a fresh clone —
    ``artifacts/`` is gitignored), the card cross-check is skipped rather than
    flagging every committed verdict box as unregistered; ``totals`` carries
    ``card_check_skipped`` so callers can say so loudly.
    """
    cards = registered_card_ids(root / "artifacts")
    rows = [
        audit_notebook(path, cards or None)
        for path in sorted((root / "notebooks").glob("*.ipynb"))
    ]
    totals = {col: sum(r[col] for r in rows) for col in _SUMMED}
    totals["n_notebooks"] = len(rows)
    totals["n_registered_cards"] = len(cards)
    totals["card_check_skipped"] = not cards
    return rows, totals


def _print_table(rows: list[dict]) -> None:
    headers = {
        "notebook": "notebook",
        "n_code_cells": "code",
        "n_figures": "figs",
        "n_figures_stamped": "stamped",
        "n_figures_unstamped": "unstamped",
        "n_rit_cells": "rit",
        "rit_labels": "rit_labels",
        "n_card_renders": "cards",
        "n_verdict_boxes": "verdicts",
        "n_unregistered_verdicts": "unreg",
        "n_error_outputs": "errors",
        "rails_pass": "pass",
    }
    display = [
        {c: ("PASS" if r[c] else "FAIL") if c == "rails_pass" else str(r[c]) for c in COLUMNS}
        for r in rows
    ]
    widths = {
        c: max(len(headers[c]), *(len(d[c]) for d in display)) if display else len(headers[c])
        for c in COLUMNS
    }
    left = {"notebook", "rit_labels"}

    def fmt(values: dict[str, str]) -> str:
        return "  ".join(
            values[c].ljust(widths[c]) if c in left else values[c].rjust(widths[c])
            for c in COLUMNS
        )

    print(fmt(headers))
    print("  ".join("-" * widths[c] for c in COLUMNS))
    for d in display:
        print(fmt(d))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="repo root holding notebooks/ and artifacts/ (default: this repo)",
    )
    parser.add_argument(
        "--json", action="store_true", help="emit machine-readable JSON instead of the table"
    )
    args = parser.parse_args(argv)

    root = args.root.resolve()
    if not (root / "notebooks").is_dir():
        print(f"honesty_audit: no notebooks/ directory under {root}", file=sys.stderr)
        return 1
    rows, totals = audit_series(root)
    if not rows:
        print(f"honesty_audit: no *.ipynb files under {root / 'notebooks'}", file=sys.stderr)
        return 1
    ok = all(r["rails_pass"] for r in rows)

    if args.json:
        print(json.dumps({"root": str(root), "rows": rows, "totals": totals, "pass": ok}))
        return 0 if ok else 1

    _print_table(rows)
    if totals["card_check_skipped"]:
        print(
            "card cross-check SKIPPED: no complete card run under "
            f"{root / 'artifacts'} (artifacts/ is gitignored, so a fresh clone has "
            "no card registry) — run the series first "
            "(uv run python -m er_lab.run nb=all run.tier=smoke) to audit "
            "verdict-vs-card registration; all other rails were audited"
        )
    print(
        f"series totals: {totals['n_figures']} figures "
        f"({totals['n_figures_stamped']} provenance-stamped, "
        f"{totals['n_figures_unstamped']} NOT), {totals['n_rit_cells']} "
        f"RUN-IN-TARGET placards, {totals['n_card_renders']} card renders, "
        f"{totals['n_verdict_boxes']} verdict boxes "
        f"({totals['n_unregistered_verdicts']} without a registered card), "
        f"{totals['n_error_outputs']} error outputs "
        f"across {totals['n_notebooks']} notebooks "
        f"({totals['n_registered_cards']} registered cards)"
    )
    fails = [r for r in rows if not r["rails_pass"]]
    if fails:
        print("RAIL FAILURES FLAGGED (not fixed here — the audit reports, the owner fixes):")
        for r in fails:
            why = []
            if r["n_error_outputs"]:
                why.append(f"{r['n_error_outputs']} error output(s)")
            if r["n_figures_unstamped"]:
                why.append(f"{r['n_figures_unstamped']} unstamped figure(s)")
            if r["n_unregistered_verdicts"]:
                why.append(f"{r['n_unregistered_verdicts']} verdict(s) without a card")
            print(f"  {r['notebook']}: " + "; ".join(why))
    else:
        print("every notebook passes every audited rail")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
