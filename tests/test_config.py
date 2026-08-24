"""Tests for er_lab.config: layered loading, hashing, seeding."""

import random
import re

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf
from omegaconf.errors import ConfigKeyError, ValidationError

from er_lab.config import (
    DEFAULT_YAML,
    LabConfig,
    config_hash,
    load_config,
    load_config_from_env,
    set_all_seeds,
)


def test_default_load_matches_contract():
    cfg = load_config()
    assert cfg.run.tier == "smoke"
    assert cfg.run.seed == 17
    assert list(cfg.run.seeds) == [17, 23, 29]
    assert cfg.run.name == "dev"
    assert cfg.model.kind == "scratch_char"
    assert cfg.model.name is None
    assert cfg.model.dim == 256
    assert cfg.model.layers == 4
    assert cfg.model.heads == 4
    assert cfg.model.max_len == 192
    assert cfg.train.batch_size == 64
    assert cfg.train.epochs == 2
    assert cfg.train.steps == 200
    assert cfg.train.lr == pytest.approx(3.0e-4)
    assert cfg.train.loss == "infonce"
    assert cfg.train.miner == "inbatch"
    assert cfg.train.miner_k == 10
    assert cfg.train.augment == "none"
    assert cfg.train.temperature == pytest.approx(0.05)
    assert cfg.train.margin == pytest.approx(0.2)
    assert cfg.serialize.scheme == "colval"
    assert cfg.serialize.missing == "token"
    assert cfg.infra.device == "auto"
    assert cfg.infra.precision == "auto"
    assert cfg.infra.num_workers == 2
    assert cfg.infra.multi_gpu is False
    assert cfg.data.dataset == "historical_50k"
    assert cfg.data.schema == "configs/schemas/historical_50k.yaml"
    assert cfg.data.n_records is None
    assert cfg.paths.data_root == "data"
    assert cfg.paths.artifacts_root == "artifacts"
    assert cfg.paths.hf_local is None


def test_default_yaml_matches_dataclass_defaults():
    """configs/default.yaml hand-duplicates LabConfig; drift would win the merge silently."""
    assert OmegaConf.to_container(OmegaConf.structured(LabConfig)) == OmegaConf.to_container(
        OmegaConf.load(DEFAULT_YAML)
    )


def test_load_config_from_env_applies_runner_injected_env(monkeypatch):
    monkeypatch.setenv("ER_LAB_DOTLIST", '["train.batch_size=128"]')
    monkeypatch.setenv("ER_LAB_TIER", "target")
    monkeypatch.setenv("ER_LAB_ARTIFACTS", "/somewhere/artifacts")
    cfg = load_config_from_env()
    assert cfg.train.batch_size == 128
    assert cfg.run.tier == "target"
    assert cfg.paths.artifacts_root == "/somewhere/artifacts"


def test_load_config_from_env_is_plain_load_config_outside_a_kernel(monkeypatch):
    for var in ("ER_LAB_DOTLIST", "ER_LAB_TIER", "ER_LAB_ARTIFACTS"):
        monkeypatch.delenv(var, raising=False)
    assert OmegaConf.to_container(load_config_from_env()) == OmegaConf.to_container(load_config())


def test_dotlist_override_changes_value_and_hash():
    base = load_config()
    over = load_config(dotlist=["train.batch_size=128", "infra.precision=bf16"])
    assert over.train.batch_size == 128
    assert over.infra.precision == "bf16"
    assert config_hash(base) != config_hash(over)


def test_yaml_layer_applied_and_dotlist_wins(tmp_path):
    run_yaml = tmp_path / "run.yaml"
    run_yaml.write_text("train:\n  batch_size: 32\n  epochs: 5\n")
    cfg = load_config(yaml_path=str(run_yaml), dotlist=["train.batch_size=256"])
    assert cfg.train.epochs == 5  # yaml layer merged over defaults
    assert cfg.train.batch_size == 256  # dotlist beats yaml


def test_dotlist_extension_key_allowed():
    cfg = load_config(dotlist=["a.b=c"])
    assert cfg.a.b == "c"


def test_dotlist_drives_typed_model_train_serialize_knobs():
    """The PLAN's dotlist contract: model/train/serialize knobs are typed keys."""
    cfg = load_config(
        dotlist=[
            "model.layers=1",
            "model.heads=2",
            "train.steps=50",
            "train.miner_k=3",
            "train.temperature=0.1",
            "train.margin=0.5",
            "serialize.scheme=json",
            "serialize.missing=drop",
        ]
    )
    assert cfg.model.layers == 1 and cfg.model.heads == 2
    assert cfg.train.steps == 50 and cfg.train.miner_k == 3
    assert cfg.train.temperature == pytest.approx(0.1)
    assert cfg.train.margin == pytest.approx(0.5)
    assert cfg.serialize.scheme == "json" and cfg.serialize.missing == "drop"


def test_typed_field_rejects_bad_value():
    with pytest.raises(ValidationError):
        load_config(dotlist=["train.batch_size=notanint"])


def test_typed_section_rejects_typoed_key():
    """Typo protection survives the knob additions: unknown keys under a typed
    node still fail loudly (root-level extension keys stay allowed)."""
    with pytest.raises(ConfigKeyError):
        load_config(dotlist=["model.layerz=3"])
    with pytest.raises(ConfigKeyError):
        load_config(dotlist=["serialize.schema=colval"])


def test_hash_deterministic_and_12_hex():
    h1 = config_hash(load_config())
    h2 = config_hash(load_config())
    assert h1 == h2
    assert re.fullmatch(r"[0-9a-f]{12}", h1)


def test_hash_independent_of_key_order():
    a = OmegaConf.create({"x": {"b": 1, "a": 2}, "y": 3})
    b = OmegaConf.create({"y": 3, "x": {"a": 2, "b": 1}})
    assert config_hash(a) == config_hash(b)
    # ...but list order is meaning-bearing and must change the hash
    assert config_hash(OmegaConf.create({"s": [1, 2]})) != config_hash(
        OmegaConf.create({"s": [2, 1]})
    )


def test_set_all_seeds_reproducible():
    set_all_seeds(123)
    t1 = torch.randn(4, 4)
    n1 = np.random.rand(3)
    r1 = random.random()
    set_all_seeds(123)
    assert torch.equal(t1, torch.randn(4, 4))
    assert np.array_equal(n1, np.random.rand(3))
    assert r1 == random.random()
