"""Tests for er_lab.infra.device: device resolution, precision policy, platform snapshot."""

import pytest
import torch

from er_lab.config import load_config
from er_lab.infra.device import describe_platform, resolve_device, resolve_precision

cpu_only = pytest.mark.skipif(
    torch.cuda.is_available() or torch.backends.mps.is_available(),
    reason="test assumes a CPU-only box",
)


def _cfg(**infra: str) -> object:
    return load_config(dotlist=[f"infra.{k}={v}" for k, v in infra.items()])


@cpu_only
def test_auto_resolves_to_cpu_on_cpu_only_box():
    assert resolve_device(_cfg(device="auto")) == torch.device("cpu")


def test_explicit_cpu():
    assert resolve_device(_cfg(device="cpu")) == torch.device("cpu")


@cpu_only
def test_explicit_cuda_without_cuda_raises_clear_error():
    with pytest.raises(RuntimeError, match="infra.device=cuda but CUDA is not available"):
        resolve_device(_cfg(device="cuda"))


@cpu_only
def test_explicit_mps_without_mps_raises_clear_error():
    with pytest.raises(RuntimeError, match="infra.device=mps but MPS is not available"):
        resolve_device(_cfg(device="mps"))


def test_unknown_device_raises():
    with pytest.raises(ValueError, match="Unknown infra.device"):
        resolve_device(_cfg(device="tpu"))


# The precision policy table. torch.device objects are plain descriptors, so the
# cuda/mps rows are exercised even on this CPU-only box.
@pytest.mark.parametrize(
    ("precision", "device_type", "expected"),
    [
        ("auto", "cpu", torch.float32),
        ("auto", "mps", torch.float32),  # fp16 on mps is explicit opt-in
        ("auto", "cuda", torch.bfloat16),
        ("fp32", "cuda", torch.float32),
        ("fp16", "mps", torch.float16),
        ("fp16", "cpu", torch.float16),
        ("bf16", "cpu", torch.bfloat16),
        ("bf16", "cuda", torch.bfloat16),
    ],
)
def test_precision_policy_table(precision, device_type, expected):
    cfg = _cfg(precision=precision)
    assert resolve_precision(cfg, torch.device(device_type)) is expected


def test_unknown_precision_raises():
    with pytest.raises(ValueError, match="Unknown infra.precision"):
        resolve_precision(_cfg(precision="int8"), torch.device("cpu"))


def test_describe_platform_keys_and_sanity():
    info = describe_platform()
    assert set(info) == {
        "os",
        "python",
        "torch",
        "cuda_available",
        "mps_available",
        "cpu_count",
        "ram_gb",
    }
    assert info["torch"] == torch.__version__
    assert isinstance(info["cuda_available"], bool)
    assert isinstance(info["mps_available"], bool)
    assert info["cpu_count"] >= 1
    assert info["ram_gb"] is None or info["ram_gb"] > 0
