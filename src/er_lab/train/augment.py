"""Train-time augmentation via the lab's noise channels (TRN-03; notebook 10).

TRN-03's factor is augmentation {none, generic typo, calibrated channels}: does
corrupting training batches with the *measured* noise mix (NSE-01 prevalences)
buy more than a generic typo model — and is the induced invariance property the
mediator? This module is only the train-time adapter: the channels themselves
live in ``er_lab.noise.channels`` (imported lazily, so the train package does
not pay the gecko import unless augmentation is actually on).

An augmenter is a plain ``callable(df_batch) -> df_batch`` over canonical
frames (a ``record_id`` column is required — channels log per record). It holds
one seeded ``numpy`` Generator created at build time: successive calls draw
successive noise (batches are corrupted differently), while the whole sequence
is reproducible from ``seed``. Ops logs are discarded here — training wants the
corrupted frame; the mediation analysis reads ops from the generator pipeline,
not from train-time augmentation.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping

import numpy as np
import pandas as pd

__all__ = ["AUGMENT_KINDS", "make_augmenter"]

AUGMENT_KINDS = ("none", "generic_typo", "calibrated")

#: The 'generic_typo' arm's fixed recipe: keyboard/edit typos only, at a flat
#: per-record rate — deliberately uncalibrated (that is the point of the arm).
_GENERIC_TYPO_RATES = {"typo": 0.5}


def make_augmenter(
    kind: str,
    *,
    channel_rates: dict | None = None,
    lexicon: Mapping[str, set[str]] | None = None,
    seed: int = 0,
) -> Callable[[pd.DataFrame], pd.DataFrame]:
    """Build a train-time augmenter: ``callable(df_batch) -> df_batch``.

    kind='none'         -> identity (returns the batch untouched).
    kind='generic_typo' -> the 'typo' channel at a flat default rate
                           (``channel_rates`` may override, e.g. ``{'typo': 0.3}``).
    kind='calibrated'   -> ``channel_rates`` required: channel name -> per-record
                           rate, the NSE-01-measured prevalences. Channels apply
                           in dict order; ``lexicon`` feeds the nickname channel.
    """
    if kind not in AUGMENT_KINDS:
        raise ValueError(f"unknown augment kind {kind!r}: expected one of {AUGMENT_KINDS}")
    if kind == "none":
        return lambda df_batch: df_batch
    if kind == "calibrated" and not channel_rates:
        raise ValueError(
            "augment kind 'calibrated' requires channel_rates — the NSE-01 "
            "measured per-channel prevalences (notebook 04 artifact)"
        )

    from er_lab.noise.channels import CHANNELS  # lazy: gecko import only when needed

    if kind == "generic_typo":
        rates = dict(channel_rates) if channel_rates else dict(_GENERIC_TYPO_RATES)
    else:  # calibrated
        rates = dict(channel_rates)
    unknown = set(rates) - set(CHANNELS)
    if unknown:
        raise ValueError(f"unknown noise channels {sorted(unknown)}: have {sorted(CHANNELS)}")

    channels = [
        (CHANNELS[name](lexicon) if name == "nickname" else CHANNELS[name](), rate)
        for name, rate in rates.items()
    ]
    rng = np.random.default_rng(seed)

    def augment(df_batch: pd.DataFrame) -> pd.DataFrame:
        out = df_batch
        for channel, rate in channels:
            out, _ops = channel.apply(out, rng, rate)
        return out

    return augment
