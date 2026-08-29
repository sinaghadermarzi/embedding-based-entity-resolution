"""Hashed run registry — figures render only from artifacts registered here.

Layout: ``<root>/<name>/<stamp>_<tier>/`` holding the payload file
(``payload.parquet`` for DataFrames, ``payload.json`` for dict/list) plus a
``meta.json`` sidecar. ``meta.json`` is written last, so a run directory
without one is an aborted write and is ignored.

Honesty rails (PLAN §2/§7): ``load`` serves the *newest* run of a name and
refuses to cross tiers — if the newest run belongs to a different tier than
requested, that is a :class:`TierMixingError`, never a silent fallback to an
older run. A missing artifact raises :class:`ArtifactMissing` with a placard
naming the producing notebook.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd
from omegaconf import OmegaConf

from er_lab.config import config_hash

TIERS = ("smoke", "mid", "target", "analytical")

_META = "meta.json"
_NAME_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_\-]*")


class ArtifactMissing(FileNotFoundError):
    """A required artifact is not registered yet — its producing notebook has not run."""


class TierMixingError(RuntimeError):
    """The newest run of an artifact belongs to a different tier than requested."""


def missing_placard(name: str, tier: str, root: Path | str) -> str:
    """The 'not yet run' placard for a missing artifact, naming its producing notebook."""
    producer = _producing_notebook(name)
    lines = [f"NOT YET RUN — artifact '{name}' (tier='{tier}') is not registered under '{root}'."]
    if producer is not None:
        num = producer.split("_", 1)[0]
        lines += [
            f"It is produced by notebooks/{producer}.",
            f"Run it first:  uv run python -m er_lab.run nb={num} run.tier={tier}",
        ]
    else:
        lines.append(
            "No notebook in er_lab.infra.runner.NOTEBOOK_DAG declares it — "
            "add the producer there before depending on it."
        )
    return "\n".join(lines)


class ArtifactRegistry:
    """Append-only registry of run outputs under a single root directory."""

    def __init__(self, root: str | Path):
        self.root = Path(root)

    @classmethod
    def from_env(cls, default_root: str = "artifacts") -> ArtifactRegistry:
        """Registry at $ER_LAB_ARTIFACTS (injected by the notebook runner), else *default_root*."""
        import os

        return cls(os.environ.get("ER_LAB_ARTIFACTS", default_root))

    def register(
        self,
        name: str,
        payload: Any,
        *,
        cfg: Any,
        tier: str,
        kind: str = "table",
        meta: dict | None = None,
    ) -> Path:
        """Write *payload* as a new run of *name* and return the run directory.

        DataFrames go to parquet; dicts/lists to JSON. The ``meta.json``
        sidecar records config_hash, tier, kind, git rev (when available),
        an ISO-8601 UTC timestamp, and ``run.seed`` / ``run.seeds`` from *cfg*;
        *meta* is stored verbatim under the ``extra`` key.
        """
        check_tier(tier)
        _check_name(name)
        if not OmegaConf.is_config(cfg):
            cfg = OmegaConf.create(cfg)
        run_dir = self._new_run_dir(name, tier)
        payload_file = _write_payload(run_dir, payload)
        sidecar = {
            "name": name,
            "kind": kind,
            "tier": tier,
            "config_hash": config_hash(cfg),
            "created_at": datetime.now(UTC).isoformat(),
            "git_rev": _git_rev(),
            "seed": OmegaConf.select(cfg, "run.seed", default=None),
            "seeds": _as_list(OmegaConf.select(cfg, "run.seeds", default=None)),
            "payload": payload_file,
            "extra": meta or {},
        }
        # meta.json last and atomically: its presence marks the run directory
        # as complete, so a crash mid-write must not leave a truncated sidecar.
        tmp = run_dir / (_META + ".tmp")
        tmp.write_text(json.dumps(sidecar, indent=2))
        os.replace(tmp, run_dir / _META)
        return run_dir

    def load(self, name: str, *, tier: str) -> tuple[Any, dict]:
        """Return ``(payload, meta)`` of the *newest* run of *name*, which must be of *tier*.

        Only the newest run counts — an older right-tier run behind a newer
        wrong-tier one raises :class:`TierMixingError`. :meth:`exists` mirrors
        exactly these semantics.
        """
        run_dir, meta = self._newest_run(name, tier)
        return _read_payload(run_dir / meta["payload"]), meta

    def exists(self, name: str, *, tier: str) -> bool:
        """True iff the *newest* complete run of *name* is of *tier* — mirrors :meth:`load`.

        Like ``load``, this looks only at the newest run, so an exists-gated
        rebuild resolves a tier mix instead of skipping the rebuild and dying
        at load time. Use :meth:`newest_tier` to inspect what is actually there.
        """
        check_tier(tier)
        return self.newest_tier(name) == tier

    def newest_tier(self, name: str) -> str | None:
        """Tier of the newest complete run of *name*, or None if there are no runs."""
        runs = self._runs(name)
        return _read_meta(runs[-1])["tier"] if runs else None

    # -- internals ---------------------------------------------------------

    def _newest_run(self, name: str, tier: str) -> tuple[Path, dict]:
        """Newest run dir + meta for *name*, enforcing the no-tier-mixing rule."""
        check_tier(tier)
        runs = self._runs(name)
        if not runs:
            raise ArtifactMissing(missing_placard(name, tier, self.root))
        newest = runs[-1]
        meta = _read_meta(newest)
        if meta["tier"] != tier:
            raise TierMixingError(
                f"Tier mixing: artifact '{name}' was requested at tier='{tier}' but its newest "
                f"registered run ({newest.name}) is tier='{meta['tier']}'. Artifacts from "
                f"different tiers must never mix — re-run the producing notebook at "
                f"tier='{tier}', or point at a different artifacts root."
            )
        return newest, meta

    def _runs(self, name: str) -> list[Path]:
        """Complete run directories of *name*, oldest first (dir names sort by timestamp)."""
        _check_name(name)
        base = self.root / name
        if not base.is_dir():
            return []
        return sorted(p for p in base.iterdir() if p.is_dir() and (p / _META).is_file())

    def _new_run_dir(self, name: str, tier: str) -> Path:
        ts = time.time_ns()
        while True:  # bump the nanosecond field on (rare) collisions
            run_dir = self.root / name / _stamp(ts, tier)
            if not run_dir.exists():
                run_dir.mkdir(parents=True)
                return run_dir
            ts += 1


def _stamp(ts_ns: int, tier: str) -> str:
    """Fixed-width UTC stamp + tier suffix; lexicographic order == chronological order."""
    sec = datetime.fromtimestamp(ts_ns // 1_000_000_000, tz=UTC)
    return f"{sec:%Y%m%dT%H%M%S}.{ts_ns % 1_000_000_000:09d}Z_{tier}"


def check_tier(tier: str) -> None:
    """Raise ValueError unless *tier* is one of the known ``TIERS``."""
    if tier not in TIERS:
        raise ValueError(f"unknown tier {tier!r}; expected one of {TIERS}")


def _check_name(name: str) -> None:
    if not isinstance(name, str) or not _NAME_RE.fullmatch(name):
        raise ValueError(
            f"invalid artifact name {name!r}: must match {_NAME_RE.pattern!r} "
            "(names are used as registry path components)"
        )


def _write_payload(run_dir: Path, payload: Any) -> str:
    if isinstance(payload, pd.DataFrame):
        payload.to_parquet(run_dir / "payload.parquet")
        return "payload.parquet"
    if isinstance(payload, (dict, list)):
        (run_dir / "payload.json").write_text(json.dumps(payload, indent=2))
        return "payload.json"
    raise TypeError(
        f"unsupported payload type {type(payload).__name__}: "
        "pass a pandas DataFrame (-> parquet) or a dict/list (-> json)"
    )


def _read_payload(path: Path) -> Any:
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    return json.loads(path.read_text())


def _read_meta(run_dir: Path) -> dict:
    try:
        return json.loads((run_dir / _META).read_text())
    except json.JSONDecodeError as err:
        raise json.JSONDecodeError(
            f"corrupt meta.json in run directory {run_dir}: {err.msg}", err.doc, err.pos
        ) from err


def _as_list(value: Any) -> list | None:
    return None if value is None else list(value)


def _producing_notebook(name: str) -> str | None:
    # Lazy import: runner imports this module at its top level.
    try:
        from er_lab.infra.runner import producer_of
    except ImportError:
        return None
    return producer_of(name)


def _git_rev() -> str | None:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).parent,
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    return out.stdout.strip() if out.returncode == 0 else None
