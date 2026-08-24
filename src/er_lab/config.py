"""Config loading (OmegaConf structured schema + yaml + dotlist), config hashing, seeding.

Every run is described by one DictConfig built as: typed defaults (``LabConfig``)
<- ``configs/default.yaml`` <- optional run yaml <- dotlist overrides. Known keys
are type-checked on merge; unknown keys are allowed (thin lab, not a framework).
"""

from __future__ import annotations

import hashlib
import json
import os
import random
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
import yaml
from omegaconf import DictConfig, OmegaConf

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_YAML = REPO_ROOT / "configs" / "default.yaml"


@dataclass
class RunConfig:
    tier: str = "smoke"  # smoke | mid | target | analytical
    seed: int = 17
    seeds: list[int] = field(default_factory=lambda: [17, 23, 29])
    name: str = "dev"


@dataclass
class ModelConfig:
    kind: str = "scratch_char"
    name: str | None = None
    dim: int = 256
    max_len: int = 192


@dataclass
class TrainConfig:
    batch_size: int = 64
    epochs: int = 2
    lr: float = 3.0e-4
    loss: str = "infonce"
    miner: str = "inbatch"
    augment: str = "none"


@dataclass
class InfraConfig:
    device: str = "auto"  # auto | cpu | mps | cuda
    precision: str = "auto"  # auto | fp32 | fp16 | bf16
    num_workers: int = 2
    multi_gpu: bool = False


@dataclass
class DataConfig:
    dataset: str = "historical_50k"
    schema: str = "configs/schemas/historical_50k.yaml"
    n_records: int | None = None


@dataclass
class PathsConfig:
    data_root: str = "data"
    artifacts_root: str = "artifacts"
    hf_local: str | None = None


@dataclass
class LabConfig:
    run: RunConfig = field(default_factory=RunConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    infra: InfraConfig = field(default_factory=InfraConfig)
    data: DataConfig = field(default_factory=DataConfig)
    paths: PathsConfig = field(default_factory=PathsConfig)


def load_config(yaml_path: str | None = None, dotlist: list[str] | None = None) -> DictConfig:
    """Build the run config: LabConfig defaults <- configs/default.yaml <- yaml_path <- dotlist."""
    cfg = OmegaConf.structured(LabConfig)
    OmegaConf.set_struct(cfg, False)  # allow extension keys (e.g. ad-hoc notebook knobs)
    layers = [OmegaConf.load(DEFAULT_YAML)]
    if yaml_path is not None:
        layers.append(OmegaConf.load(yaml_path))
    if dotlist:
        layers.append(OmegaConf.from_dotlist(list(dotlist)))
    merged = OmegaConf.merge(cfg, *layers)
    OmegaConf.set_struct(merged, False)
    assert isinstance(merged, DictConfig)
    return merged


def load_config_from_env() -> DictConfig:
    """Config for a runner-launched kernel — the standard first call of every notebook.

    Layers the defaults with the CLI dotlist the notebook runner forwarded as
    JSON in ``ER_LAB_DOTLIST``, then pins ``run.tier`` / ``paths.artifacts_root``
    from ``ER_LAB_TIER`` / ``ER_LAB_ARTIFACTS`` when the runner injected them.
    Outside a runner kernel (no env vars set) this is just ``load_config()``.
    """
    cfg = load_config(dotlist=json.loads(os.environ.get("ER_LAB_DOTLIST", "[]")))
    tier = os.environ.get("ER_LAB_TIER")
    if tier is not None:
        cfg.run.tier = tier
    artifacts_root = os.environ.get("ER_LAB_ARTIFACTS")
    if artifacts_root is not None:
        cfg.paths.artifacts_root = artifacts_root
    return cfg


def config_hash(cfg: DictConfig) -> str:
    """12-hex sha256 of the canonical resolved yaml — key-order independent."""
    if not isinstance(cfg, DictConfig):
        cfg = OmegaConf.create(cfg)
    container = OmegaConf.to_container(cfg, resolve=True)
    canonical = yaml.safe_dump(container, sort_keys=True)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:12]


def set_all_seeds(seed: int) -> None:
    """Seed random, numpy, and torch (CPU + all accelerators) for reproducible runs."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)  # also seeds CUDA/MPS generators when present
