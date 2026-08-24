"""Tests for er_lab.train.losses (the TRN-01 loss families)."""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from er_lab.train.losses import cosent, infonce, supcon, triplet


def _fixed_pairs(n: int = 2, dim: int = 8, seed: int = 0):
    g = torch.Generator().manual_seed(seed)
    a = F.normalize(torch.randn(n, dim, generator=g), dim=-1)
    b = F.normalize(torch.randn(n, dim, generator=g), dim=-1)
    return a, b


# --- infonce ----------------------------------------------------------------


def test_infonce_known_answer_computed_by_hand():
    """Two pairs -> 4x4 similarity matrix; NT-Xent recomputed with plain math."""
    a, b = _fixed_pairs(n=2)
    t = 0.1
    got = float(infonce(a, b, temperature=t))

    z = torch.cat([a, b]).numpy()
    sim = z @ z.T / t
    partner = {0: 2, 1: 3, 2: 0, 3: 1}  # z[i] pairs with z[i+2 mod 4]
    total = 0.0
    for i in range(4):
        denom = sum(math.exp(sim[i, k]) for k in range(4) if k != i)
        total += -math.log(math.exp(sim[i, partner[i]]) / denom)
    assert got == pytest.approx(total / 4, abs=1e-5)


def test_infonce_perfect_separation_is_near_zero():
    # orthogonal entities, identical views: partner sim 1, negatives sim 0
    a = torch.eye(4)
    loss = float(infonce(a, a.clone(), temperature=0.02))
    assert loss < 1e-4


def test_infonce_shape_mismatch_raises():
    with pytest.raises(ValueError, match="aligned"):
        infonce(torch.randn(3, 8), torch.randn(2, 8), temperature=0.1)


def test_infonce_differentiable():
    a, b = _fixed_pairs()
    a = a.clone().requires_grad_(True)
    infonce(a, b, temperature=0.1).backward()
    assert a.grad is not None and torch.isfinite(a.grad).all()


# --- supcon -----------------------------------------------------------------


def test_supcon_reduces_to_infonce_for_singleton_positive_case():
    """Each label appearing exactly twice (one positive per anchor) == NT-Xent."""
    a, b = _fixed_pairs(n=4, seed=3)
    t = 0.07
    labels = torch.tensor([0, 1, 2, 3, 0, 1, 2, 3])
    got = supcon(torch.cat([a, b]), labels, temperature=t)
    want = infonce(a, b, temperature=t)
    assert torch.allclose(got, want, atol=1e-6)


def test_supcon_multi_positive_groups_run():
    emb = F.normalize(torch.randn(6, 8, generator=torch.Generator().manual_seed(1)), dim=-1)
    loss = supcon(emb, torch.tensor([0, 0, 0, 1, 1, 2]), temperature=0.1)
    assert torch.isfinite(loss)  # anchor 5 has no positive and is excluded, not NaN


def test_supcon_no_positives_raises():
    emb = torch.randn(3, 8)
    with pytest.raises(ValueError, match="positive"):
        supcon(emb, torch.tensor([0, 1, 2]), temperature=0.1)


def test_supcon_label_count_mismatch_raises():
    with pytest.raises(ValueError, match="labels"):
        supcon(torch.randn(3, 8), torch.tensor([0, 0]), temperature=0.1)


# --- triplet ----------------------------------------------------------------


def test_triplet_margin_boundary():
    """s(a,p)=1, s(a,n)=0: loss = relu(margin - 1), zero exactly at margin<=1."""
    a = torch.tensor([[1.0, 0.0]])
    p = torch.tensor([[1.0, 0.0]])
    n = torch.tensor([[0.0, 1.0]])
    assert float(triplet(a, p, n, margin=1.0)) == 0.0  # exactly at the boundary
    assert float(triplet(a, p, n, margin=0.5)) == 0.0  # inside
    assert float(triplet(a, p, n, margin=1.5)) == pytest.approx(0.5, abs=1e-6)  # violated


def test_triplet_means_over_batch():
    a = torch.tensor([[1.0, 0.0], [1.0, 0.0]])
    p = torch.tensor([[1.0, 0.0], [0.0, 1.0]])  # second triplet violated by 1+margin
    n = torch.tensor([[0.0, 1.0], [1.0, 0.0]])
    # row 1: relu(0.5 - 1 + 0) = 0; row 2: relu(0.5 - 0 + 1) = 1.5 -> mean 0.75
    assert float(triplet(a, p, n, margin=0.5)) == pytest.approx(0.75, abs=1e-6)


def test_triplet_normalizes_inputs():
    a = torch.tensor([[10.0, 0.0]])  # unnormalized on purpose
    p = torch.tensor([[2.0, 0.0]])
    n = torch.tensor([[0.0, 3.0]])
    assert float(triplet(a, p, n, margin=1.0)) == 0.0


# --- cosent -----------------------------------------------------------------


def test_cosent_sign_behavior():
    """Loss ~0 when positives out-rank negatives; large when the order flips."""
    a = torch.tensor([[1.0, 0.0], [1.0, 0.0]])
    b = torch.tensor([[1.0, 0.0], [0.0, 1.0]])  # pair sims: 1.0 and 0.0
    good = float(cosent(a, b, torch.tensor([1, 0]), tau=0.05))
    bad = float(cosent(a, b, torch.tensor([0, 1]), tau=0.05))
    assert good == pytest.approx(0.0, abs=1e-4)  # log(1 + e^-20)
    assert bad == pytest.approx(20.0, abs=1e-3)  # log(1 + e^20) ~ (1-0)/tau
    assert bad > good


def test_cosent_single_class_is_exactly_zero():
    a, b = _fixed_pairs(n=3)
    assert float(cosent(a, b, torch.tensor([1, 1, 1]), tau=0.05)) == 0.0
    assert float(cosent(a, b, torch.tensor([0, 0, 0]), tau=0.05)) == 0.0


@pytest.mark.parametrize("labels", [[1, 1, 1], [0, 0, 0]])
def test_cosent_single_class_backward_is_noop_not_error(labels):
    """The zero must stay graph-connected: an all-positive minibatch in a real
    training loop should be a harmless no-op step, not a backward() error."""
    a, b = _fixed_pairs(n=3)
    a = a.clone().requires_grad_(True)
    b = b.clone().requires_grad_(True)
    loss = cosent(a, b, torch.tensor(labels), tau=0.05)
    assert float(loss.detach()) == 0.0
    assert loss.requires_grad
    loss.backward()  # used to raise 'element 0 ... does not require grad'
    assert a.grad is not None and torch.all(a.grad == 0)
    assert b.grad is not None and torch.all(b.grad == 0)


def test_cosent_monotone_in_violation():
    a = torch.tensor([[1.0, 0.0], [1.0, 0.0]])
    pos_b = torch.tensor([[1.0, 0.0]])
    small_neg = torch.cat([pos_b, torch.tensor([[0.0, 1.0]])])  # neg sim 0
    big_neg = torch.cat([pos_b, F.normalize(torch.tensor([[1.0, 1.0]]), dim=-1)])  # neg sim .707
    labels = torch.tensor([1, 0])
    assert float(cosent(a, big_neg, labels, tau=0.05)) > float(
        cosent(a, small_neg, labels, tau=0.05)
    )


def test_all_losses_are_scalar_tensors():
    a, b = _fixed_pairs(n=4, seed=5)
    labels = torch.tensor([1, 0, 1, 0])
    pair_labels = torch.tensor([0, 1, 2, 3, 0, 1, 2, 3])
    for value in (
        infonce(a, b, temperature=0.1),
        supcon(torch.cat([a, b]), pair_labels, temperature=0.1),
        triplet(a, b, torch.roll(b, 1, dims=0), margin=0.2),
        cosent(a, b, labels, tau=0.05),
    ):
        assert value.ndim == 0 and np.isfinite(float(value))
