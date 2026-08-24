"""The one training loop every TRN arm runs through (notebooks 09-11).

One code path, factor-parameterized: loss family (TRN-01), negative miner
(TRN-02), augmentation (TRN-03), encoder regime (TRN-04, via the ``encoder``
argument), serialization scheme (TRN-05, via cfg). Budget-matching across arms
is *step-count accounting* (PLAN §3.1): ``steps`` is the exact number of
optimizer steps executed — never epochs, never wall-clock — so "matched budget"
is checkable arithmetic.

How the factors combine per step:

1. A batch of positive pairs is drawn from the within-entity pair pool
   (:func:`er_lab.train.mining.build_pairs_inbatch`).
2. With a mining miner ('bm25' / 'ann' / 'ann_filtered'), batches are
   *hard-composed*: a seed pair is drawn, then the batch is filled with pairs
   whose anchors are the seed anchor's mined neighbors — so in-batch negatives
   are hard for every loss family, without changing any loss signature.
   'ann' mines with the encoder's *current* embeddings (initially untrained;
   ``eval_every`` > 0 re-mines every ``eval_every`` steps as the encoder
   improves). FN-contamination is measured on the raw mined table
   (:func:`fn_contamination`) *before* 'ann_filtered' applies
   :func:`cluster_aware_filter` — the measurement sees the trap, the filter is
   the mitigation. The filter proxy defaults to the corpus ``entity_id``
   (synthetic corpora), i.e. the oracle-filter arm; pass ``filter_proxy`` to
   run a deployable (non-truth) proxy arm instead.
3. Both sides of each pair are augmented (:func:`make_augmenter`), serialized
   (``er_lab.serialize``, scheme/missing from the typed ``serialize.scheme`` /
   ``serialize.missing`` cfg keys), and embedded differentiably.
4. Losses needing explicit negatives (triplet, cosent) draw one mined negative
   per anchor (falling back to the next pair's b-side in-batch — which can
   itself be a false negative in a dedup-dense batch; that is the measured
   phenomenon, not a bug).

Device and precision come from ``er_lab.infra.device`` — never hardcoded. The
learning rate is constant (no scheduler — thin); the ``lr`` history column
exists so schedule-carrying variants keep the same history schema.

Multi-GPU (``cfg.infra.multi_gpu=true``): single-node data parallelism via
``accelerate``, installed through the optional ``multi-gpu`` extra
(``uv sync --extra multi-gpu``; also in the dev group so the DDP path is
CI-tested 2-process on CPU — ``accelerate launch`` supports multi-process
CPU). Under DDP, gradient synchronization only runs through the wrapper's
``forward``, so the loop tokenizes OUTSIDE the model and calls the wrapped
module itself for training embeddings, and uses ``accelerator.unwrap_model``
for the no-grad ``encode()`` during re-mining. There is deliberately no
multi-machine assumption anywhere: nothing in the lab may require more than
the single node (PLAN §2).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from omegaconf import DictConfig, OmegaConf

from er_lab.config import set_all_seeds
from er_lab.data.schema import NON_TEXT_ROLES, ROLES
from er_lab.infra.device import resolve_device, resolve_precision
from er_lab.serialize import serialize_frame
from er_lab.train.augment import make_augmenter
from er_lab.train.losses import cosent, infonce, supcon, triplet
from er_lab.train.mining import (
    ann_hard_negatives,
    bm25_hard_negatives,
    build_pairs_inbatch,
    cluster_aware_filter,
    fn_contamination,
)

__all__ = ["HISTORY_COLUMNS", "LOSSES", "MINERS", "train_encoder"]

LOSSES = ("infonce", "supcon", "triplet", "cosent")
MINERS = ("inbatch", "bm25", "ann", "ann_filtered")
HISTORY_COLUMNS = ["step", "loss", "fn_contamination_rate", "lr"]


def _compute_setup(cfg: DictConfig):
    """Resolve (accelerator, device, dtype) honoring the multi-GPU opt-in.

    Single-node only: a launched ``Accelerator()`` spans at most this
    machine's processes (multi-GPU on the node, multi-process CPU in CI).
    """
    if bool(cfg.infra.get("multi_gpu", False)):
        try:
            from accelerate import Accelerator  # lazy: optional 'multi-gpu' extra
        except ImportError as exc:
            raise RuntimeError(
                "infra.multi_gpu=true requires accelerate — install the optional "
                "extra: uv sync --extra multi-gpu"
            ) from exc
        mixed = "no"
        if torch.cuda.is_available():  # CUDA-less accelerate (CPU DDP) stays fp32
            dtype_probe = resolve_precision(cfg, torch.device("cuda"))
            mixed = {torch.float16: "fp16", torch.bfloat16: "bf16"}.get(dtype_probe, "no")
        accelerator = Accelerator(mixed_precision=mixed)
        # accelerate owns precision handling; report its device, keep fp32 locally
        return accelerator, accelerator.device, torch.float32
    device = resolve_device(cfg)
    return None, device, resolve_precision(cfg, device)


def train_encoder(
    encoder,
    corpus_df: pd.DataFrame,
    cfg: DictConfig,
    *,
    loss_name: str,
    miner_name: str,
    augment_kind: str,
    steps: int,
    eval_every: int = 0,
    filter_proxy: pd.Series | None = None,
    seed: int,
):
    """Train ``encoder`` on ``corpus_df`` for exactly ``steps`` optimizer steps.

    Returns ``(encoder, history)`` where history is a DataFrame with columns
    ``step`` (1..steps), ``loss`` (this step's training loss),
    ``fn_contamination_rate`` (latest measured raw-mined contamination; NaN for
    'inbatch' or truthless corpora), and ``lr``. Deterministic under ``seed``:
    same encoder init + same arguments => identical final weights.

    ``filter_proxy`` (record_id -> proxy cluster id) is what the
    'ann_filtered' miner hands to :func:`cluster_aware_filter`; the default
    ``None`` uses the corpus ``entity_id`` truth — the documented oracle
    upper-bound arm. Pass a deployable proxy (e.g. a phonetic key or a prior
    model's clustering) for the non-oracle mitigation arms.

    cfg keys read: the typed ``train.batch_size``, ``train.lr``,
    ``train.temperature``, ``train.margin``, ``train.miner_k``,
    ``serialize.scheme``/``serialize.missing``, the optional extension key
    ``train.channel_rates`` (calibrated augmentation), plus the ``infra``
    block via ``er_lab.infra.device``.
    """
    if steps < 1:
        raise ValueError(f"steps must be >= 1, got {steps}")
    if loss_name not in LOSSES:
        raise ValueError(f"unknown loss {loss_name!r}: expected one of {LOSSES}")
    if miner_name not in MINERS:
        raise ValueError(f"unknown miner {miner_name!r}: expected one of {MINERS}")
    if corpus_df["record_id"].duplicated().any():
        raise ValueError("corpus_df.record_id must be unique")

    set_all_seeds(seed)
    rng = np.random.default_rng(seed)
    accelerator, device, dtype = _compute_setup(cfg)
    batch = int(cfg.train.batch_size)
    lr = float(cfg.train.lr)
    temperature = float(cfg.train.temperature)
    margin = float(cfg.train.margin)
    mine_k = int(cfg.train.miner_k)
    scheme = str(cfg.serialize.scheme)
    missing = str(cfg.serialize.missing)

    text_roles = [c for c in corpus_df.columns if c in ROLES and c not in NON_TEXT_ROLES]
    if not text_roles:
        raise ValueError("corpus_df has no text-bearing role columns to serialize")

    def _texts(df: pd.DataFrame) -> list[str]:
        return serialize_frame(df, text_roles=text_roles, scheme=scheme, missing=missing).tolist()

    recs = corpus_df.set_index(corpus_df["record_id"].astype(str))
    truth = corpus_df.set_index(corpus_df["record_id"].astype(str))["entity_id"]
    pool = build_pairs_inbatch(corpus_df, n_pairs=max(8 * batch, 256), seed=seed)
    pair_rows_by_anchor: dict[str, list[int]] = {}
    for row_i, aid in enumerate(pool["a_id"]):
        pair_rows_by_anchor.setdefault(str(aid), []).append(row_i)

    channel_rates = OmegaConf.select(cfg, "train.channel_rates")
    if channel_rates is not None:
        channel_rates = OmegaConf.to_container(channel_rates, resolve=True)
    augmenter = make_augmenter(augment_kind, channel_rates=channel_rates, seed=seed)

    encoder.to(device)

    def _unwrapped():
        """The bare module — under accelerate, the DDP wrapper hides custom methods."""
        return encoder if accelerator is None else accelerator.unwrap_model(encoder)

    def _mine() -> tuple[dict[str, list[str]] | None, float]:
        """(neighbors by anchor_id, raw contamination); (None, NaN) for inbatch."""
        if miner_name == "inbatch":
            return None, float("nan")
        if miner_name == "bm25":
            mined = bm25_hard_negatives(corpus_df, _texts(corpus_df), k=mine_k, seed=seed)
        else:  # ann / ann_filtered: mine with the encoder's current embeddings
            emb = _unwrapped().encode(_texts(corpus_df), batch_size=batch, device=device)
            mined = ann_hard_negatives(corpus_df, emb, k=mine_k, seed=seed)
        rate = fn_contamination(mined, truth)  # measured BEFORE any mitigation
        if miner_name == "ann_filtered":
            mined = cluster_aware_filter(mined, truth if filter_proxy is None else filter_proxy)
        neighbors = mined.groupby("anchor_id", sort=False)["neg_id"].apply(list).to_dict()
        return {str(k): [str(v) for v in vs] for k, vs in neighbors.items()}, rate

    neighbors, contamination = _mine()

    def _batch_rows() -> list[int]:
        """Pool row indices for one batch (hard-composed when a mined table exists)."""
        if neighbors is None:
            return [int(i) for i in rng.integers(0, len(pool), size=batch)]
        picked = [int(rng.integers(len(pool)))]
        for neg in neighbors.get(str(pool["a_id"].iloc[picked[0]]), []):
            for row_i in pair_rows_by_anchor.get(neg, []):
                if len(picked) < batch:
                    picked.append(row_i)
        while len(picked) < batch:
            picked.append(int(rng.integers(len(pool))))
        return picked[:batch]

    def _neg_ids(a_ids: list[str], b_ids: list[str]) -> list[str]:
        """One negative record per anchor: mined when available, in-batch roll otherwise."""
        negs = []
        for i, aid in enumerate(a_ids):
            cands = neighbors.get(aid) if neighbors else None
            if cands:
                negs.append(cands[int(rng.integers(len(cands)))])
            else:
                negs.append(b_ids[(i + 1) % len(b_ids)])
        return negs

    def _embed(ids: list[str]) -> torch.Tensor:
        frame = augmenter(recs.loc[ids].reset_index(drop=True))
        texts = _texts(frame)
        if accelerator is None:
            return encoder.embed_batch(texts, device=device)
        # DDP gradient sync only runs through the wrapper's forward(): tokenize
        # outside (on the bare module) and call the wrapped module itself.
        tokens = _unwrapped().tokenize(texts, device=device)
        return encoder(*tokens) if isinstance(tokens, tuple) else encoder(tokens)

    optimizer = torch.optim.AdamW(encoder.parameters(), lr=lr)
    if accelerator is not None:
        encoder, optimizer = accelerator.prepare(encoder, optimizer)
    encoder.train()

    history: list[tuple[int, float, float, float]] = []
    for step in range(1, steps + 1):
        if eval_every > 0 and step > 1 and (step - 1) % eval_every == 0:
            neighbors, contamination = _mine()  # re-mine/re-measure cadence
            encoder.train()  # encode() inside _mine flips to eval

        rows = pool.iloc[_batch_rows()]
        a_ids = [str(v) for v in rows["a_id"]]
        b_ids = [str(v) for v in rows["b_id"]]
        autocast = torch.autocast(
            device_type=device.type,
            dtype=dtype,
            enabled=dtype != torch.float32 and accelerator is None,
        )
        with autocast:
            emb_a, emb_b = _embed(a_ids), _embed(b_ids)
            if loss_name == "infonce":
                loss = infonce(emb_a, emb_b, temperature=temperature)
            elif loss_name == "supcon":
                codes = torch.as_tensor(pd.factorize(truth.loc[a_ids])[0], device=emb_a.device)
                loss = supcon(
                    torch.cat([emb_a, emb_b]),
                    torch.cat([codes, codes]),
                    temperature=temperature,
                )
            elif loss_name == "triplet":
                emb_n = _embed(_neg_ids(a_ids, b_ids))
                loss = triplet(emb_a, emb_b, emb_n, margin=margin)
            else:  # cosent
                emb_n = _embed(_neg_ids(a_ids, b_ids))
                labels = torch.cat(
                    [torch.ones(batch, dtype=torch.long), torch.zeros(batch, dtype=torch.long)]
                ).to(emb_a.device)
                loss = cosent(
                    torch.cat([emb_a, emb_a]), torch.cat([emb_b, emb_n]), labels, tau=temperature
                )

        optimizer.zero_grad()
        if accelerator is not None:
            accelerator.backward(loss)
        else:
            loss.backward()
        optimizer.step()  # the budget unit: exactly `steps` of these run
        history.append((step, float(loss.detach().float().cpu()), contamination, lr))

    encoder.eval()
    # hand back the bare module: callers use encode()/embed_batch(), which a
    # DDP wrapper does not expose (its job — synced training — is done)
    return _unwrapped(), pd.DataFrame(history, columns=HISTORY_COLUMNS)
