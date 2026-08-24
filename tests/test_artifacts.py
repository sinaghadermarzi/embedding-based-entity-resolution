"""Tests for er_lab.infra.artifacts (register/load round-trips, placards, tier rails)."""

from __future__ import annotations

import hashlib
import re
import sys
import types
from datetime import datetime

import pandas as pd
import pytest
from omegaconf import OmegaConf


def _ensure_config_module() -> None:
    """Install a contract-shaped ``er_lab.config`` stand-in iff the real one is absent.

    Creates no files: once the real module lands, the import succeeds and the
    stand-in is never installed, so these tests keep exercising the real thing.
    """
    try:
        import er_lab.config

        return
    except ImportError:
        pass
    import er_lab

    stub = types.ModuleType("er_lab.config")
    stub.__doc__ = "Test stand-in for the er_lab.config contract (real module not built yet)."

    def config_hash(cfg) -> str:
        return hashlib.sha256(OmegaConf.to_yaml(cfg, resolve=True).encode()).hexdigest()[:12]

    def load_config(yaml_path=None, dotlist=None):
        cfg = OmegaConf.create(
            {
                "run": {"tier": "smoke", "seed": 17, "seeds": [17, 23, 29], "name": "dev"},
                "paths": {"data_root": "data", "artifacts_root": "artifacts", "hf_local": None},
            }
        )
        if dotlist:
            cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(list(dotlist)))
        return cfg

    def set_all_seeds(seed: int) -> None:
        pass

    stub.config_hash = config_hash
    stub.load_config = load_config
    stub.set_all_seeds = set_all_seeds
    sys.modules["er_lab.config"] = stub
    er_lab.config = stub


_ensure_config_module()

from er_lab.infra.artifacts import (
    ArtifactMissing,
    ArtifactRegistry,
    TierMixingError,
)


@pytest.fixture
def cfg():
    return OmegaConf.create(
        {"run": {"tier": "smoke", "seed": 17, "seeds": [17, 23, 29], "name": "dev"}}
    )


@pytest.fixture
def registry(tmp_path):
    return ArtifactRegistry(tmp_path / "artifacts")


def test_dataframe_round_trip_with_sidecar_meta(registry, cfg):
    df = pd.DataFrame({"rec_id": [1, 2, 3], "surname": ["ashe", "byrd", "cole"]})
    run_dir = registry.register("bas01_fs_baseline", df, cfg=cfg, tier="smoke")
    assert run_dir.is_dir()
    assert (run_dir / "payload.parquet").is_file()
    assert (run_dir / "meta.json").is_file()

    payload, meta = registry.load("bas01_fs_baseline", tier="smoke")
    pd.testing.assert_frame_equal(payload, df)
    assert meta["name"] == "bas01_fs_baseline"
    assert meta["tier"] == "smoke"
    assert meta["kind"] == "table"
    assert re.fullmatch(r"[0-9a-f]{12}", meta["config_hash"])
    datetime.fromisoformat(meta["created_at"])  # valid ISO timestamp
    assert meta["seed"] == 17
    assert meta["seeds"] == [17, 23, 29]
    assert "git_rev" in meta  # present; a hash in a checkout, None outside one


def test_dict_round_trip_with_extra_meta(registry, cfg):
    payload_in = {"unit": "entity", "coverage": {"nominal": 0.95, "observed": 0.93}}
    registry.register(
        "met03_bootstrap_coverage",
        payload_in,
        cfg=cfg,
        tier="smoke",
        kind="table",
        meta={"note": "coverage sweep"},
    )
    payload, meta = registry.load("met03_bootstrap_coverage", tier="smoke")
    assert payload == payload_in
    assert meta["extra"] == {"note": "coverage sweep"}
    assert meta["kind"] == "table"


def test_list_payload_round_trip(registry, cfg):
    payload_in = [{"seed": 17, "f1": 0.81}, {"seed": 23, "f1": 0.79}]
    registry.register("trn01_loss_matrix", payload_in, cfg=cfg, tier="smoke")
    payload, _ = registry.load("trn01_loss_matrix", tier="smoke")
    assert payload == payload_in


def test_load_serves_the_newest_run(registry, cfg):
    registry.register("met07_splits", {"v": 1}, cfg=cfg, tier="smoke")
    registry.register("met07_splits", {"v": 2}, cfg=cfg, tier="smoke")
    payload, _ = registry.load("met07_splits", tier="smoke")
    assert payload == {"v": 2}


def test_missing_artifact_placard_names_the_producing_notebook(registry):
    with pytest.raises(ArtifactMissing) as exc:
        registry.load("calibrated_corpus", tier="smoke")
    message = str(exc.value)
    assert "not yet run" in message.lower()
    assert "05_calibrated_dirt_machine.ipynb" in message
    assert "calibrated_corpus" in message


def test_exists_is_tier_scoped(registry, cfg):
    assert not registry.exists("met07_splits", tier="smoke")
    registry.register("met07_splits", {"v": 1}, cfg=cfg, tier="smoke")
    assert registry.exists("met07_splits", tier="smoke")
    assert not registry.exists("met07_splits", tier="target")


def test_tier_mixing_rejected(registry, cfg):
    registry.register("met07_splits", {"v": "smoke"}, cfg=cfg, tier="smoke")
    with pytest.raises(TierMixingError):
        registry.load("met07_splits", tier="target")


def test_newer_tier_shadows_older_tier(registry, cfg):
    registry.register("met07_splits", {"v": "smoke"}, cfg=cfg, tier="smoke")
    registry.register("met07_splits", {"v": "target"}, cfg=cfg, tier="target")
    payload, meta = registry.load("met07_splits", tier="target")
    assert payload == {"v": "target"}
    assert meta["tier"] == "target"
    # The newest run is target-tier, so a smoke load must refuse — never
    # silently fall back to the stale smoke run.
    with pytest.raises(TierMixingError):
        registry.load("met07_splits", tier="smoke")


def test_unsupported_payload_type_rejected(registry, cfg):
    with pytest.raises(TypeError):
        registry.register("bad", object(), cfg=cfg, tier="smoke")


def test_unknown_tier_rejected(registry, cfg):
    with pytest.raises(ValueError):
        registry.register("x", {"a": 1}, cfg=cfg, tier="smok")
    with pytest.raises(ValueError):
        registry.exists("x", tier="node")
