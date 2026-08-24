"""Tests for er_lab.config: layered loading, hashing, seeding."""

import random
import re

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf
from omegaconf.errors import ValidationError

from er_lab.config import config_hash, load_config, set_all_seeds


def test_default_load_matches_contract():
    cfg = load_config()
    assert cfg.run.tier == "smoke"
    assert cfg.run.seed == 17
    assert list(cfg.run.seeds) == [17, 23, 29]
    assert cfg.run.name == "dev"
    assert cfg.model.kind == "scratch_char"
    assert cfg.model.name is None
    assert cfg.model.dim == 256
    assert cfg.model.max_len == 192
    assert cfg.train.batch_size == 64
    assert cfg.train.epochs == 2
    assert cfg.train.lr == pytest.approx(3.0e-4)
    assert cfg.train.loss == "infonce"
    assert cfg.train.miner == "inbatch"
    assert cfg.train.augment == "none"
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


def test_typed_field_rejects_bad_value():
    with pytest.raises(ValidationError):
        load_config(dotlist=["train.batch_size=notanint"])


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
    assert config_hash(OmegaConf.create({"s": [1, 2]})) != config_hash(OmegaConf.create({"s": [2, 1]}))


def test_set_all_seeds_reproducible():
    set_all_seeds(123)
    t1 = torch.randn(4, 4)
    n1 = np.random.rand(3)
    r1 = random.random()
    set_all_seeds(123)
    assert torch.equal(t1, torch.randn(4, 4))
    assert np.array_equal(n1, np.random.rand(3))
    assert r1 == random.random()
