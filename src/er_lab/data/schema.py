"""Declared schemas: column -> role mappings for each corpus (PLAN.md §2, notebook 01).

A ``DeclaredSchema`` maps a source's real column names onto a fixed vocabulary of
person-record roles. ``to_canonical`` re-labels and re-types only — it must NOT
clean, normalize, or standardize values. Upstream standardization is an
experimental factor (PRS-01), so every field is delivered verbatim as a string.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd
import yaml

ROLES = frozenset(
    {
        "given_name",
        "middle_name",
        "family_name",
        "name_suffix",
        "full_name",
        "dob",
        "birth_year",
        "age",
        "sex",
        "race",
        "ethnicity",
        "house_number",
        "street",
        "unit",
        "city",
        "state",
        "zip",
        "street_address",
        "phone",
        "email",
        "ssn_like",
        "county",
        "snapshot_date",
    }
)

#: Roles that are provenance metadata rather than person content — excluded from
#: text serialization (``text_roles``) but still carried in the canonical frame.
NON_TEXT_ROLES = frozenset({"snapshot_date"})

#: ``record_id`` sentinel meaning "this source has no row id; use the row position".
ROW_SENTINEL = "__row__"


def _cols(spec: str | list[str]) -> list[str]:
    return [spec] if isinstance(spec, str) else list(spec)


def _join_verbatim(parts: list[pd.Series]) -> pd.Series:
    """Join aligned string columns with single spaces, dropping empty/missing parts.

    Parts are joined in the given order; a row with no non-empty part becomes NA.
    Parts themselves are NOT trimmed or otherwise altered. Deliberately, only NA
    and ``''`` count as "missing": whitespace-only parts are kept VERBATIM. NC
    snapshots pad blank fields with spaces (e.g. ``half_code=' '``), so an NC
    street joins to something like ``'  WARD ST  '`` — that padding is part of
    the noise under study (PRS-01 owns any standardization upstream), and this
    join must not quietly clean it away.
    """
    values: list[str | None] = []
    for row in zip(*(p.tolist() for p in parts)):
        kept = [v for v in row if not pd.isna(v) and v != ""]
        values.append(" ".join(kept) if kept else None)
    return pd.Series(values, index=parts[0].index, dtype="string")


@dataclass
class DeclaredSchema:
    """A source's column->role declaration (loaded from ``configs/schemas/*.yaml``)."""

    name: str
    #: source column, ROW_SENTINEL to synthesize from row position, or a LIST of
    #: source columns joined with ':' into a compound key — for sources whose row id
    #: is only unique within a partition (NC: voter_reg_num per county, so
    #: [county_id, voter_reg_num] -> 'county:regnum' is the statewide-unique id).
    record_id: str | list[str]
    entity_id: str | None  # truth-key column; None when the source has no truth key
    roles: dict[str, str | list[str]]  # role -> source column(s); list order = join order
    extra_keep: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        unknown = set(self.roles) - ROLES
        if unknown:
            raise ValueError(f"schema '{self.name}': unknown roles {sorted(unknown)}")
        duplicates = {c for c in self.extra_keep if self.extra_keep.count(c) > 1}
        if duplicates:
            raise ValueError(
                f"schema '{self.name}': duplicate extra_keep columns {sorted(duplicates)}"
            )
        reserved = {"record_id", "entity_id", "source"} | set(self.roles)
        collisions = set(self.extra_keep) & reserved
        if collisions:
            raise ValueError(
                f"schema '{self.name}': extra_keep columns {sorted(collisions)} collide "
                "with canonical output columns (record_id/entity_id/source or a role name)"
            )

    @classmethod
    def from_yaml(cls, path: str | Path) -> DeclaredSchema:
        raw = yaml.safe_load(Path(path).read_text())
        return cls(
            name=raw["name"],
            record_id=raw["record_id"],
            entity_id=raw.get("entity_id"),
            roles=dict(raw.get("roles") or {}),
            extra_keep=list(raw.get("extra_keep") or []),
        )

    def has(self, role: str) -> bool:
        return role in self.roles

    def text_roles(self) -> list[str]:
        """Text-bearing roles present, in yaml order — the serialization field order."""
        return [role for role in self.roles if role not in NON_TEXT_ROLES]

    def to_canonical(self, df: pd.DataFrame) -> pd.DataFrame:
        """Relabel ``df`` into the canonical role frame — verbatim, no cleaning.

        Columns: each role present (multi-column roles joined per ``_join_verbatim``),
        then 'record_id', 'entity_id' (NA when the schema has no truth key), 'source'
        (= schema name) — all pandas 'string' dtype — then ``extra_keep`` unchanged.
        """
        needed = [c for spec in self.roles.values() for c in _cols(spec)]
        if self.record_id != ROW_SENTINEL:
            needed.extend(_cols(self.record_id))
        if self.entity_id is not None:
            needed.append(self.entity_id)
        needed.extend(self.extra_keep)
        missing = [c for c in needed if c not in df.columns]
        if missing:
            raise KeyError(f"schema '{self.name}': columns missing from frame: {missing}")

        out = pd.DataFrame(index=df.index)
        for role, spec in self.roles.items():
            cols = _cols(spec)
            if len(cols) == 1:
                out[role] = df[cols[0]].astype("string")
            else:
                out[role] = _join_verbatim([df[c].astype("string") for c in cols])
        if self.record_id == ROW_SENTINEL:
            out["record_id"] = pd.Series(
                [str(i) for i in range(len(df))], index=df.index, dtype="string"
            )
        elif isinstance(self.record_id, list):
            # Compound key: parts are ids, not person content, so — unlike role joins —
            # they are stripped (NC files space-pad short fields) and ':'-joined; a
            # missing part propagates to NA so a broken key can never silently collide.
            parts = [df[c].astype("string").str.strip() for c in self.record_id]
            joined = parts[0]
            for part in parts[1:]:
                joined = joined + ":" + part
            out["record_id"] = joined
        else:
            out["record_id"] = df[self.record_id].astype("string")
        if self.entity_id is None:
            out["entity_id"] = pd.Series(pd.NA, index=df.index, dtype="string")
        else:
            out["entity_id"] = df[self.entity_id].astype("string")
        out["source"] = pd.Series(self.name, index=df.index, dtype="string")
        for col in self.extra_keep:
            out[col] = df[col]
        return out
