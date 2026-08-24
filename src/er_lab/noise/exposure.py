"""Group-correlated noise exposure (PLAN.md §3 NSE-01/FAIR-01; lit_review §9, §12).

The fairness literature's central mechanism is pre-algorithmic: identifier
quality differs by group, so noise exposure is group-correlated. Generic
generators (pseudopeople's column noise) apply rates independently of record
attributes and structurally cannot express this; here a channel's base
per-record rate is multiplied by a fitted per-group multiplier instead.

An ``ExposureModel`` is a table of group values -> multiplier. Multipliers are
fitted from measured per-group prevalence (NSE-01 same-NCID diffs) as each
group's rate relative to the overall rate, so ``base_rate`` keeps its meaning
as the corpus-level expectation. Records whose group is unknown, missing, or
absent from the table get multiplier 1.0.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

_MULT = "multiplier"


@dataclass(frozen=True)
class ExposureModel:
    """Per-group rate multipliers layered on a channel's base per-record rate."""

    group_cols: list[str]
    multipliers: pd.DataFrame  # columns: *group_cols ('string' dtype), 'multiplier' (float)

    @classmethod
    def uniform(cls) -> ExposureModel:
        """No group structure: every record keeps the base rate unchanged."""
        return cls(group_cols=[], multipliers=pd.DataFrame({_MULT: pd.Series(dtype=float)}))

    @classmethod
    def from_table(cls, df: pd.DataFrame) -> ExposureModel:
        """Build from a table whose columns are group columns plus 'multiplier'.

        Group values are cast to pandas 'string' dtype (so int-coded groups match
        the string-typed canonical frames); keys must be non-missing and unique;
        multipliers must be finite and >= 0.
        """
        if _MULT not in df.columns:
            raise ValueError("multiplier table needs a 'multiplier' column")
        group_cols = [c for c in df.columns if c != _MULT]
        if not group_cols:
            raise ValueError("multiplier table needs at least one group column")
        keys = df[group_cols].astype("string").reset_index(drop=True)
        if keys.isna().any().any():
            raise ValueError("multiplier table group values must be non-missing")
        if keys.duplicated().any():
            raise ValueError("multiplier table has duplicate group keys")
        mult = pd.to_numeric(df[_MULT], errors="raise").astype(float).to_numpy()
        if not np.isfinite(mult).all() or (mult < 0).any():
            raise ValueError("multipliers must be finite and >= 0")
        return cls(group_cols=group_cols, multipliers=keys.assign(**{_MULT: mult}))

    @classmethod
    def fit_from_prevalence(cls, prevalence_by_group: pd.DataFrame) -> ExposureModel:
        """Fit multipliers as each group's measured rate relative to the overall rate.

        ``prevalence_by_group``: group columns + 'rate' (+ optional 'n' record
        counts). The overall rate is the 'n'-weighted mean of group rates when
        'n' is present, else the unweighted mean. Rows with a missing rate are
        ignored; rows with missing group values contribute to the overall rate
        but are dropped from the table (such records default to multiplier 1.0).
        """
        df = prevalence_by_group
        if "rate" not in df.columns:
            raise ValueError("prevalence table needs a 'rate' column")
        group_cols = [c for c in df.columns if c not in ("rate", "n")]
        if not group_cols:
            raise ValueError("prevalence table needs at least one group column")
        df = df.dropna(subset=["rate"])
        rates = pd.to_numeric(df["rate"], errors="raise").astype(float).to_numpy()
        if "n" in df.columns:
            weights = pd.to_numeric(df["n"], errors="raise").astype(float).to_numpy()
            overall = float((rates * weights).sum() / weights.sum())
        else:
            overall = float(rates.mean())
        if not overall > 0:
            raise ValueError("overall prevalence is zero; multipliers are undefined")
        table = df[group_cols].astype("string").assign(**{_MULT: rates / overall})
        table = table[~table[group_cols].isna().any(axis=1)]
        return cls.from_table(table)

    def per_record_rates(self, df: pd.DataFrame, base_rate: float) -> pd.Series:
        """Per-record probability aligned to ``df``: base_rate x group multiplier.

        Unknown/NA groups get multiplier 1.0. The product is clipped to [0, 1]
        because the result is a per-record-per-channel probability.
        """
        if not 0.0 <= base_rate <= 1.0:
            raise ValueError(f"base_rate must be a probability in [0, 1], got {base_rate}")
        if not self.group_cols:
            return pd.Series(base_rate, index=df.index, dtype=float, name="rate")
        missing = [c for c in self.group_cols if c not in df.columns]
        if missing:
            raise KeyError(f"group columns missing from frame: {missing}")
        keys = df[self.group_cols].astype("string").reset_index(drop=True)
        merged = keys.merge(self.multipliers, on=self.group_cols, how="left", sort=False)
        if len(merged) != len(df):  # unreachable: from_table forbids duplicate keys
            raise RuntimeError("multiplier join changed the row count")
        mult = merged[_MULT].astype(float).to_numpy()
        unknown = np.isnan(mult) | keys.isna().any(axis=1).to_numpy(dtype=bool)
        rates = np.clip(base_rate * np.where(unknown, 1.0, mult), 0.0, 1.0)
        return pd.Series(rates, index=df.index, dtype=float, name="rate")
