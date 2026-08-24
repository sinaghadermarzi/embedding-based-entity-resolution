import warnings

import numpy as np
import pytest

from er_lab.probes.geometry import HUBNESS_MAX_ROWS, alignment, hubness, rankme, uniformity

# ---------------------------------------------------------------------------
# alignment
# ---------------------------------------------------------------------------


def test_alignment_zero_for_identical_positives():
    rng = np.random.default_rng(0)
    x = rng.standard_normal((50, 8))
    assert alignment(x, x) == pytest.approx(0.0, abs=1e-12)


def test_alignment_known_values_and_scale_invariance():
    e1 = np.array([[1.0, 0.0]])
    e2 = np.array([[0.0, 1.0]])
    assert alignment(e1, e2) == pytest.approx(2.0)  # orthogonal unit vectors
    assert alignment(e1, -e1) == pytest.approx(4.0)  # antipodal: the maximum
    # normalization: scaling either side changes nothing
    assert alignment(3.0 * e1, 0.1 * e2) == pytest.approx(2.0)


def test_alignment_validation():
    with pytest.raises(ValueError, match="share a shape"):
        alignment(np.ones((3, 2)), np.ones((4, 2)))
    with pytest.raises(ValueError, match="2-d"):
        alignment(np.ones(3), np.ones(3))


# ---------------------------------------------------------------------------
# uniformity
# ---------------------------------------------------------------------------


def test_uniformity_collapsed_is_zero_and_spread_is_lower():
    collapsed = np.ones((10, 4))
    assert uniformity(collapsed) == pytest.approx(0.0, abs=1e-12)
    # 8 evenly spread directions on the circle
    angles = np.linspace(0.0, 2.0 * np.pi, 8, endpoint=False)
    spread = np.stack([np.cos(angles), np.sin(angles)], axis=1)
    assert uniformity(spread) < uniformity(collapsed) - 0.5


def test_uniformity_monotone_in_spread():
    rng = np.random.default_rng(1)
    base = rng.standard_normal((100, 16))
    tight = base * 0.01 + np.ones(16)  # small jitter around one direction
    assert uniformity(base) < uniformity(tight)


def test_uniformity_validation():
    with pytest.raises(ValueError, match="at least 2 rows"):
        uniformity(np.ones((1, 4)))
    with pytest.raises(ValueError, match="t must be positive"):
        uniformity(np.ones((3, 4)), t=0.0)


# ---------------------------------------------------------------------------
# rankme
# ---------------------------------------------------------------------------


def test_rankme_near_full_for_gaussian():
    rng = np.random.default_rng(2)
    x = rng.standard_normal((512, 16))
    r = rankme(x)
    assert 14.0 <= r <= 16.0 + 1e-9  # ~min(n, d) for a flat spectrum


def test_rankme_near_one_for_rank_one():
    rng = np.random.default_rng(3)
    x = np.outer(rng.standard_normal(512), rng.standard_normal(16))
    assert rankme(x) == pytest.approx(1.0, abs=1e-3)


def test_rankme_bounded_by_min_dim_and_rejects_zero():
    rng = np.random.default_rng(4)
    x = rng.standard_normal((40, 6))
    assert 1.0 <= rankme(x) <= 6.0 + 1e-9
    with pytest.raises(ValueError, match="all-zero"):
        rankme(np.zeros((5, 3)))


# ---------------------------------------------------------------------------
# hubness
# ---------------------------------------------------------------------------


def _uniform_points(n: int = 200, d: int = 3, seed: int = 5) -> np.ndarray:
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((n, d))
    return x / np.linalg.norm(x, axis=1, keepdims=True)


def _hub_points(n: int = 200, d: int = 16, seed: int = 6) -> np.ndarray:
    """Planted hubs: 3 points exactly at the common direction mu, the rest
    scattered around it — the mu-points sit in nearly every k-NN list."""
    rng = np.random.default_rng(seed)
    mu = np.ones(d) / np.sqrt(d)
    x = mu + 0.2 * rng.standard_normal((n - 3, d))
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    return np.vstack([np.tile(mu, (3, 1)), x])


def test_hubness_hub_geometry_more_skewed_than_uniform():
    uni = hubness(_uniform_points(), k=10)
    hub = hubness(_hub_points(), k=10)
    assert hub["skewness"] > uni["skewness"] + 1.0
    assert hub["max_k_occurrence"] > 3 * uni["max_k_occurrence"]
    assert hub["max_k_occurrence"] > 60  # planted hubs sit in a large share of lists


def test_hubness_counts_are_conserved():
    x = _uniform_points(n=60, d=4, seed=7)
    out = hubness(x, k=5)
    assert set(out) == {"skewness", "max_k_occurrence"}
    # every row contributes exactly k occurrences, so max >= mean = k
    assert out["max_k_occurrence"] >= 5


def test_hubness_constant_counts_is_zero_skew_without_warning():
    # a corpus of same-size exact-duplicate groups: with k=2 every row's two
    # nearest neighbors are its duplicates, so N_k == 2 for all rows — the
    # most hub-free outcome possible must be skewness 0.0, not scipy's
    # NaN plus a catastrophic-cancellation RuntimeWarning
    rng = np.random.default_rng(0)
    base = rng.standard_normal((5, 8))
    x = np.tile(base, (3, 1))
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # any scipy precision-loss warning fails
        out = hubness(x, k=2)
    assert out["skewness"] == 0.0
    assert out["max_k_occurrence"] == 2


def test_hubness_validation():
    with pytest.raises(ValueError, match="capped"):
        hubness(np.zeros((HUBNESS_MAX_ROWS + 1, 2)), k=10)
    with pytest.raises(ValueError, match="k must satisfy"):
        hubness(np.random.default_rng(0).standard_normal((5, 2)), k=5)


def test_torch_tensors_accepted_if_available():
    torch = pytest.importorskip("torch")
    x = torch.randn(20, 4, generator=torch.Generator().manual_seed(0))
    assert alignment(x, x) == pytest.approx(0.0, abs=1e-12)
    assert rankme(x) > 1.0
