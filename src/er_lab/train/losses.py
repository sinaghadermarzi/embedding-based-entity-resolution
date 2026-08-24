"""The four contrastive loss families compared in TRN-01 (notebook 09).

All functions are torch, batch-first, and defensively L2-normalize their
embedding inputs (a no-op for already-normalized encoders), so every loss sees
cosine similarities. Nothing here is novel — these are the standard published
formulations, written down exactly so the matched-budget comparison is a
comparison of losses, not of implementation quirks.

Formulas (z = normalized embeddings, s(i,j) = z_i . z_j):

- ``infonce``  — SimCLR-form NT-Xent over the concatenated 2N batch:
  L = (1/2N) sum_i -log[ exp(s(i, j(i))/t) / sum_{k != i} exp(s(i,k)/t) ],
  where j(i) is i's aligned partner. The denominator runs over ALL other 2N-1
  items (both views), which makes ``supcon`` with pair labels reduce to it
  exactly (tested).
- ``supcon``   — Khosla et al. 2020, the L_out form: each anchor averages
  -log softmax over its positive set P(i) = {p != i : label_p = label_i};
  anchors with no positive are excluded from the mean.
- ``triplet``  — cosine-similarity margin: mean relu(margin - s(a,p) + s(a,n)).
- ``cosent``   — Su 2022: L = log(1 + sum_{i in pos, j in neg}
  exp((s_j - s_i)/tau)) over pair cosines s with binary labels; every
  (positive, negative) ordering violation contributes one term.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

__all__ = ["cosent", "infonce", "supcon", "triplet"]


def infonce(emb_a: torch.Tensor, emb_b: torch.Tensor, *, temperature: float = 0.05) -> torch.Tensor:
    """In-batch NT-Xent over aligned positive pairs (emb_a[i] <-> emb_b[i]).

    Every non-partner item in the concatenated 2N batch is a negative, so
    same-entity records that collide in one batch silently become false
    negatives — the TRN-02 contamination phenomenon this lab measures rather
    than assumes away.
    """
    if emb_a.shape != emb_b.shape:
        raise ValueError(f"aligned pair batches required: {emb_a.shape} vs {emb_b.shape}")
    n = emb_a.shape[0]
    z = F.normalize(torch.cat([emb_a, emb_b], dim=0), dim=-1)
    sim = (z @ z.T) / temperature
    self_mask = torch.eye(2 * n, dtype=torch.bool, device=z.device)
    sim = sim.masked_fill(self_mask, float("-inf"))
    partners = torch.cat(
        [torch.arange(n, 2 * n, device=z.device), torch.arange(n, device=z.device)]
    )
    return F.cross_entropy(sim, partners)


def supcon(emb: torch.Tensor, labels: torch.Tensor, *, temperature: float = 0.05) -> torch.Tensor:
    """Supervised contrastive loss over one batch of labeled embeddings.

    ``labels`` are integer group ids (same id = positive). With each label
    appearing exactly twice (the singleton-positive case) this equals
    :func:`infonce` on the two halves. Raises if no anchor has any positive —
    a degenerate batch is a bug upstream, not a zero loss.
    """
    labels = torch.as_tensor(labels, device=emb.device).reshape(-1)
    n = emb.shape[0]
    if labels.shape[0] != n:
        raise ValueError(f"{n} embeddings but {labels.shape[0]} labels")
    z = F.normalize(emb, dim=-1)
    sim = (z @ z.T) / temperature
    self_mask = torch.eye(n, dtype=torch.bool, device=z.device)
    pos_mask = (labels.unsqueeze(0) == labels.unsqueeze(1)) & ~self_mask
    n_pos = pos_mask.sum(dim=1)
    if not (n_pos > 0).any():
        raise ValueError("supcon: no anchor in the batch has a positive")
    log_prob = sim - sim.masked_fill(self_mask, float("-inf")).logsumexp(dim=1, keepdim=True)
    valid = n_pos > 0
    per_anchor = -(log_prob * pos_mask).sum(dim=1)[valid] / n_pos[valid]
    return per_anchor.mean()


def triplet(
    anchor: torch.Tensor, pos: torch.Tensor, neg: torch.Tensor, *, margin: float = 0.2
) -> torch.Tensor:
    """Cosine triplet loss: mean(relu(margin - s(a,p) + s(a,n))), rows aligned.

    Zero exactly when every anchor's positive beats its negative by at least
    ``margin`` in cosine similarity (the boundary case is tested).
    """
    a = F.normalize(anchor, dim=-1)
    s_ap = (a * F.normalize(pos, dim=-1)).sum(dim=-1)
    s_an = (a * F.normalize(neg, dim=-1)).sum(dim=-1)
    return F.relu(margin - s_ap + s_an).mean()


def cosent(
    emb_a: torch.Tensor, emb_b: torch.Tensor, labels: torch.Tensor, *, tau: float = 0.05
) -> torch.Tensor:
    """CoSENT loss over labeled pairs (emb_a[i], emb_b[i], labels[i] in {0,1}).

    Purely rank-based: it pushes every positive pair's cosine above every
    negative pair's, with no absolute similarity target — which is why CAL-01
    still has to calibrate scores afterwards. Batches with only one class have
    no ordering constraint and return exactly 0 (log 1) — a zero that stays
    connected to the input graph, so ``backward()`` is a harmless no-op step
    rather than a 'does not require grad' error.
    """
    labels = torch.as_tensor(labels, device=emb_a.device).reshape(-1)
    sims = (F.normalize(emb_a, dim=-1) * F.normalize(emb_b, dim=-1)).sum(dim=-1)
    s_pos, s_neg = sims[labels == 1], sims[labels == 0]
    if s_pos.numel() == 0 or s_neg.numel() == 0:
        return sims.sum() * 0.0  # exact 0, graph-connected (zero gradients, not an error)
    zero = torch.zeros(1, dtype=sims.dtype, device=sims.device)
    terms = ((s_neg.unsqueeze(0) - s_pos.unsqueeze(1)) / tau).reshape(-1)
    return torch.logsumexp(torch.cat([zero, terms]), dim=0)
