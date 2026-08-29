"""Tests for er_lab.models.encoders (the two TRN-04 encoder regimes).

CharByteEncoder tests cover encode() MECHANICS only — shape, normalization,
determinism, batch invariance, byte-level tokenization. Semantic robustness
(typo moves an embedding less than a different name) is a TRAINED property;
an untrained pooling stack need not have it, so asserting it here would be a
flaky lie. The trained property is what the probes battery measures.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from er_lab.config import load_config, set_all_seeds
from er_lab.models.encoders import (
    BOS_ID,
    PAD_ID,
    VOCAB_SIZE,
    CharByteEncoder,
    build_encoder,
)

TINY = {"dim": 32, "layers": 1, "heads": 2, "max_len": 64}


def tiny_encoder(seed: int = 0) -> CharByteEncoder:
    set_all_seeds(seed)
    return CharByteEncoder(**TINY)


TEXTS = ["alice smith", "bob jones 1959-01-01", "", "a" * 500]


def test_output_shape_dtype_and_l2_norm():
    enc = tiny_encoder()
    out = enc.encode(TEXTS, batch_size=2, device="cpu")
    assert out.shape == (4, 32)
    assert out.dtype == np.float32
    assert np.allclose(np.linalg.norm(out, axis=1), 1.0, atol=1e-5)
    assert np.isfinite(out).all()  # incl. the empty string (BOS guarantees one position)


def test_empty_input_list():
    assert tiny_encoder().encode([]).shape == (0, 32)


def test_determinism_same_call_and_same_seed():
    enc = tiny_encoder(seed=3)
    a = enc.encode(TEXTS, batch_size=2)
    b = enc.encode(TEXTS, batch_size=2)
    assert np.array_equal(a, b)  # same weights, same batching -> bitwise equal
    c = tiny_encoder(seed=3).encode(TEXTS, batch_size=2)
    assert np.array_equal(a, c)  # seeded construction reproduces the weights


def test_batch_invariance():
    """A text alone embeds (almost) identically to the same text inside a padded batch."""
    enc = tiny_encoder()
    solo = enc.encode(["alice smith"])
    batched = enc.encode(TEXTS, batch_size=4)  # padded to the 500-char row's length
    assert np.abs(solo[0] - batched[0]).max() < 1e-5


def test_long_text_truncates_not_crashes():
    enc = tiny_encoder()
    out = enc.encode(["x" * 10_000])
    assert out.shape == (1, 32)
    # truncated to max_len: identical to the already-truncated prefix
    prefix = enc.encode(["x" * (TINY["max_len"] - 1)])
    assert np.array_equal(out, prefix)


def test_tokenize_byte_level_mechanics():
    enc = tiny_encoder()
    ids, mask = enc.tokenize(["ab", "a"])
    assert ids.shape == (2, 3)  # BOS + 2 bytes, padded
    assert ids[0].tolist() == [BOS_ID, ord("a"), ord("b")]
    assert ids[1].tolist() == [BOS_ID, ord("a"), PAD_ID]
    assert mask.tolist() == [[True, True, True], [True, True, False]]
    assert VOCAB_SIZE == 258  # 256 bytes + PAD + BOS


def test_non_ascii_goes_through_utf8_bytes():
    enc = tiny_encoder()
    ids, _ = enc.tokenize(["é"])  # 2 utf-8 bytes
    assert ids.shape == (1, 3)
    assert all(0 <= i < 256 for i in ids[0, 1:].tolist())


def test_embed_batch_is_differentiable():
    enc = tiny_encoder()
    emb = enc.embed_batch(["alice smith", "bob jones"], device="cpu")
    assert emb.requires_grad
    emb.sum().backward()  # reaches the byte embedding
    assert enc.byte_embed.weight.grad is not None


def test_build_encoder_scratch_char_uses_cfg_dims():
    cfg = load_config(dotlist=["model.kind=scratch_char", "model.dim=16", "model.max_len=32"])
    enc = build_encoder(cfg)
    assert isinstance(enc, CharByteEncoder)
    assert enc.encode(["x"]).shape == (1, 16)
    assert enc.max_len == 32


def test_build_encoder_unknown_kind_raises():
    cfg = load_config(dotlist=["model.kind=parrot"])
    with pytest.raises(ValueError, match="model.kind"):
        build_encoder(cfg)


def test_pretrained_unavailable_raises_run_in_target_error(monkeypatch, tmp_path):
    """Offline + no hf_local copy => one RuntimeError naming the fallback tree.

    HF_HUB_OFFLINE=1 mirrors this container's reality (huggingface.co blocked)
    while guaranteeing the test never waits on a network timeout.
    """
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("HF_HOME", str(tmp_path))  # empty cache: weights truly absent
    cfg = load_config(dotlist=["model.kind=pretrained"])
    with pytest.raises(RuntimeError) as excinfo:
        build_encoder(cfg)
    msg = str(excinfo.value)
    assert "RUN-IN-TARGET" in msg
    assert "COMPAT" in msg
    assert "hf_local" in msg
    assert "scratch_char" in msg  # the smoke-tier fallback is named


def test_pretrained_hf_local_without_weights_raises_same_error(monkeypatch, tmp_path):
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    cfg = load_config(dotlist=["model.kind=pretrained", f"paths.hf_local={tmp_path}"])
    with pytest.raises(RuntimeError, match="RUN-IN-TARGET"):
        build_encoder(cfg)


def test_encode_respects_dtype_argument():
    enc = tiny_encoder()
    out = enc.encode(["alice smith"], dtype=torch.bfloat16)
    assert out.dtype == np.float32  # always cast back
    assert np.allclose(np.linalg.norm(out, axis=1), 1.0, atol=1e-5)
