"""The two encoder regimes: from-scratch char/byte transformer vs pretrained subword (TRN-04).

Both regimes expose the same two entry points so every experiment cell is
regime-agnostic:

- ``encode(texts, *, batch_size, device, dtype) -> np.ndarray`` — inference:
  float32, L2-normalized rows, one row per input text.
- ``embed_batch(texts, *, device) -> torch.Tensor`` — the differentiable path
  the training loop backpropagates through (normalized, batch-first).

Device and dtype always come from ``er_lab.infra.device`` resolution upstream;
nothing here hardcodes a backend. ``build_encoder(cfg)`` is the single factory
the notebooks use (``model.kind`` in ``{'scratch_char', 'pretrained'}``).

Pretrained weights and the offline smoke container: huggingface.co is blocked
from the build sandbox (notes/COMPAT.md go/no-go tree), so ``PretrainedEncoder``
raises a RuntimeError that names the fallback — pretrained cells are
``# [RUN-IN-TARGET mac]`` (mac/node have normal internet), or the user points
``paths.hf_local`` at a local copy loaded with ``local_files_only=True``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import DictConfig
from torch import nn

__all__ = [
    "BOS_ID",
    "DEFAULT_PRETRAINED",
    "PAD_ID",
    "VOCAB_SIZE",
    "CharByteEncoder",
    "PretrainedEncoder",
    "build_encoder",
]

#: Byte-level vocabulary: raw bytes 0..255 plus two specials.
PAD_ID = 256  # padding (masked out of attention and pooling)
BOS_ID = 257  # prepended to every sequence, so even '' has one real position
VOCAB_SIZE = 258

DEFAULT_PRETRAINED = "sentence-transformers/all-MiniLM-L6-v2"


def _autocast(device: torch.device, dtype: torch.dtype):
    """Autocast context for the requested compute dtype (no-op for fp32)."""
    return torch.autocast(device_type=device.type, dtype=dtype, enabled=dtype != torch.float32)


class CharByteEncoder(nn.Module):
    """Small from-scratch byte-level transformer encoder (the smoke-tier default).

    Tokenization is UTF-8 bytes — no vocabulary to learn or leak, and a 1-char
    typo perturbs exactly one or two input ids, which is the property TRN-04
    puts under test against subword tokenizers. Architecture: byte + learned
    position embeddings -> ``layers`` post-norm transformer blocks -> masked
    mean-pool over real positions -> linear head to ``dim`` -> L2 normalize.

    ``d_model`` (the trunk width) defaults to ``dim`` (the output width =
    ``cfg.model.dim``); both are configurable for matched-parameter budgets.
    Dropout defaults to 0.0 so eval-mode outputs are exactly deterministic.
    """

    def __init__(
        self,
        *,
        dim: int = 256,
        layers: int = 2,
        heads: int = 4,
        max_len: int = 192,
        d_model: int | None = None,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        d_model = dim if d_model is None else d_model
        self.dim, self.max_len = dim, max_len
        self.byte_embed = nn.Embedding(VOCAB_SIZE, d_model, padding_idx=PAD_ID)
        self.pos_embed = nn.Embedding(max_len, d_model)
        block = nn.TransformerEncoderLayer(
            d_model, heads, dim_feedforward=2 * d_model, dropout=dropout, batch_first=True
        )
        # enable_nested_tensor=False: the nested-tensor fast path re-packs padded
        # batches and can change numerics with batch composition; we need
        # batch-size-invariant embeddings (tested).
        self.trunk = nn.TransformerEncoder(block, num_layers=layers, enable_nested_tensor=False)
        self.head = nn.Linear(d_model, dim)

    def tokenize(
        self, texts: list[str], *, device: torch.device | str | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Texts -> (ids, mask), both (B, L); mask is True at real positions.

        Each text becomes ``[BOS] + utf8_bytes[: max_len - 1]``, padded with
        PAD to the batch max length (truncation, never an error, for long rows).
        """
        seqs = [[BOS_ID, *text.encode("utf-8")[: self.max_len - 1]] for text in texts]
        length = max(len(s) for s in seqs)
        ids = torch.full((len(seqs), length), PAD_ID, dtype=torch.long)
        mask = torch.zeros((len(seqs), length), dtype=torch.bool)
        for i, seq in enumerate(seqs):
            ids[i, : len(seq)] = torch.tensor(seq, dtype=torch.long)
            mask[i, : len(seq)] = True
        if device is not None:
            ids, mask = ids.to(device), mask.to(device)
        return ids, mask

    def forward(self, ids: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """(B, L) ids + mask -> (B, dim) L2-normalized embeddings (differentiable)."""
        positions = torch.arange(ids.shape[1], device=ids.device)
        x = self.byte_embed(ids) + self.pos_embed(positions)
        h = self.trunk(x, src_key_padding_mask=~mask)
        denom = mask.sum(dim=1, keepdim=True).clamp(min=1)
        pooled = (h * mask.unsqueeze(-1)).sum(dim=1) / denom
        return F.normalize(self.head(pooled), dim=-1)

    def embed_batch(
        self, texts: list[str], *, device: torch.device | str | None = None
    ) -> torch.Tensor:
        """Differentiable encode of one batch — the training loop's forward path."""
        ids, mask = self.tokenize(texts, device=device)
        return self(ids, mask)

    @torch.no_grad()
    def encode(
        self,
        texts: list[str],
        *,
        batch_size: int = 64,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> np.ndarray:
        """Inference encode: (len(texts), dim) float32, rows L2-normalized.

        ``device``/``dtype`` come from ``infra.device`` resolution (defaults:
        the module's current device, fp32). Non-fp32 dtypes run under autocast;
        the output is always cast back to float32 and re-normalized.
        """
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        device = (
            next(self.parameters()).device if device is None else torch.device(device)
        )
        dtype = torch.float32 if dtype is None else dtype
        was_training = self.training
        self.eval().to(device)
        chunks = []
        for start in range(0, len(texts), batch_size):
            ids, mask = self.tokenize(texts[start : start + batch_size], device=device)
            with _autocast(device, dtype):
                emb = self(ids, mask)
            chunks.append(emb.float().cpu().numpy())
        if was_training:
            self.train()
        out = np.concatenate(chunks, axis=0).astype(np.float32)
        return out / np.linalg.norm(out, axis=1, keepdims=True)


class PretrainedEncoder(nn.Module):
    """sentence-transformers wrapper — the pretrained-subword regime of TRN-04.

    Weight acquisition follows the notes/COMPAT.md go/no-go tree: with
    ``hf_local`` set, weights load strictly offline (``local_files_only=True``)
    from either ``<hf_local>/<name>`` (a plain model directory) or ``hf_local``
    used as an HF cache folder; without it, the normal sentence-transformers
    path runs (HF hub or its default cache — the mac/node situation). Any
    failure raises one RuntimeError naming the RUN-IN-TARGET fallback, so an
    offline smoke container can never half-load a pretrained arm.
    """

    def __init__(self, name: str = DEFAULT_PRETRAINED, *, hf_local: str | None = None) -> None:
        super().__init__()
        from sentence_transformers import SentenceTransformer  # heavy import, deferred

        self.name = name
        try:
            if hf_local is not None:
                local_dir = Path(hf_local) / name
                if local_dir.is_dir():
                    self.model = SentenceTransformer(str(local_dir), local_files_only=True)
                else:
                    self.model = SentenceTransformer(
                        name, cache_folder=str(hf_local), local_files_only=True
                    )
            else:
                self.model = SentenceTransformer(name)
        except Exception as exc:
            raise RuntimeError(
                f"pretrained encoder weights for {name!r} are unavailable here "
                f"(hf_local={hf_local!r}). huggingface.co is blocked from the build "
                "sandbox, so per the go/no-go tree in notes/COMPAT.md this cell is "
                "# [RUN-IN-TARGET mac] — run it on the mac/node (normal internet), or "
                "set paths.hf_local to a directory holding the weights (loaded with "
                "local_files_only=True). The smoke-tier default is the from-scratch "
                "encoder: model.kind=scratch_char."
            ) from exc
        self.dim = int(self.model.get_sentence_embedding_dimension())

    def embed_batch(
        self, texts: list[str], *, device: torch.device | str | None = None
    ) -> torch.Tensor:
        """Differentiable encode of one batch (normalized sentence embeddings)."""
        if device is not None:
            self.model.to(torch.device(device))
        features = self.model.tokenize(list(texts))
        if device is not None:
            features = {
                k: v.to(device) if isinstance(v, torch.Tensor) else v
                for k, v in features.items()
            }
        return F.normalize(self.model(features)["sentence_embedding"], dim=-1)

    @torch.no_grad()
    def encode(
        self,
        texts: list[str],
        *,
        batch_size: int = 64,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> np.ndarray:
        """Inference encode: (len(texts), dim) float32, rows L2-normalized.

        Non-fp32 ``dtype`` runs the underlying model under autocast, matching
        :meth:`CharByteEncoder.encode` semantics; output is float32 either way.
        """
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        device = (
            next(self.parameters()).device if device is None else torch.device(device)
        )
        dtype = torch.float32 if dtype is None else dtype
        with _autocast(device, dtype):
            out = self.model.encode(
                list(texts),
                batch_size=batch_size,
                device=str(device),
                convert_to_numpy=True,
                normalize_embeddings=True,
                show_progress_bar=False,
            )
        out = np.asarray(out, dtype=np.float32)
        return out / np.linalg.norm(out, axis=1, keepdims=True)


def build_encoder(cfg: DictConfig):
    """Factory on ``cfg.model.kind``: 'scratch_char' or 'pretrained'.

    scratch_char reads dim/max_len (typed keys) plus optional extension keys
    ``model.layers`` / ``model.heads`` / ``model.d_model`` / ``model.dropout``;
    pretrained reads ``model.name`` (default all-MiniLM-L6-v2) and
    ``paths.hf_local``.
    """
    kind = str(cfg.model.kind)
    if kind == "scratch_char":
        return CharByteEncoder(
            dim=int(cfg.model.dim),
            layers=int(cfg.model.get("layers", 2)),
            heads=int(cfg.model.get("heads", 4)),
            max_len=int(cfg.model.max_len),
            d_model=cfg.model.get("d_model"),
            dropout=float(cfg.model.get("dropout", 0.0)),
        )
    if kind == "pretrained":
        return PretrainedEncoder(
            cfg.model.name or DEFAULT_PRETRAINED, hf_local=cfg.paths.hf_local
        )
    raise ValueError(f"unknown model.kind {kind!r}: expected 'scratch_char' or 'pretrained'")
