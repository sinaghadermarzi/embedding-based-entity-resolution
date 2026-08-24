"""Tests for er_lab.infra.runner (DAG, gating, headless execution) and er_lab.run parsing."""

from __future__ import annotations

import hashlib
import os
import sys
import textwrap
import types

import nbformat
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

import er_lab.run as er_run
from er_lab.infra import runner
from er_lab.infra.artifacts import ArtifactMissing, ArtifactRegistry


def _write_notebook(path, sources) -> None:
    nb = nbformat.v4.new_notebook(cells=[nbformat.v4.new_code_cell(src) for src in sources])
    nbformat.write(nb, str(path))


# ---------------------------------------------------------------- NOTEBOOK_DAG


def test_dag_covers_the_series_in_order():
    keys = list(runner.NOTEBOOK_DAG)
    assert len(keys) == 20  # notebooks 00-18 plus the appendix
    assert [k.split("_", 1)[0] for k in keys] == [f"{i:02d}" for i in range(19)] + ["A"]
    assert all(k.endswith(".ipynb") for k in keys)


def test_dag_is_topologically_consistent():
    produced: set[str] = set()
    for nb_name, spec in runner.NOTEBOOK_DAG.items():
        assert set(spec) == {"requires", "produces"}, nb_name
        assert spec["produces"], f"{nb_name} produces nothing"
        not_yet = [a for a in spec["requires"] if a not in produced]
        assert not not_yet, (
            f"{nb_name} requires {not_yet} before any earlier notebook produces them"
        )
        for artifact in spec["produces"]:
            assert artifact not in produced, f"{artifact} declared by two notebooks"
            produced.add(artifact)


def test_producer_of():
    assert runner.producer_of("calibrated_corpus") == "05_calibrated_dirt_machine.ipynb"
    assert runner.producer_of("no_such_artifact") is None


def test_resolve_notebook():
    assert runner.resolve_notebook("05") == "05_calibrated_dirt_machine.ipynb"
    assert runner.resolve_notebook("5") == "05_calibrated_dirt_machine.ipynb"
    assert runner.resolve_notebook("05_calibrated_dirt_machine.ipynb") == (
        "05_calibrated_dirt_machine.ipynb"
    )
    assert runner.resolve_notebook("notebooks/17_verdict_at_1e7.ipynb") == "17_verdict_at_1e7.ipynb"
    assert runner.resolve_notebook("A") == "A_appendix_exploratory_arms.ipynb"
    with pytest.raises(KeyError):
        runner.resolve_notebook("99")


# ---------------------------------------------------------------- run_notebook


def test_run_notebook_hard_fails_before_execution(tmp_path, monkeypatch):
    marker = tmp_path / "executed_marker.txt"
    nb_path = tmp_path / "91_gated_probe.ipynb"
    _write_notebook(
        nb_path,
        [f"open({str(marker)!r}, 'w').write('ran')", "print('second cell')"],
    )
    monkeypatch.setitem(
        runner.NOTEBOOK_DAG,
        "91_gated_probe.ipynb",
        {"requires": ["calibrated_corpus"], "produces": []},
    )
    with pytest.raises(ArtifactMissing) as exc:
        runner.run_notebook(nb_path, "smoke", tmp_path / "artifacts")
    message = str(exc.value)
    assert "not yet run" in message.lower()
    assert "05_calibrated_dirt_machine.ipynb" in message
    assert not marker.exists()  # the kernel never started


def test_run_notebook_injects_env_and_registers_artifact(tmp_path, monkeypatch):
    monkeypatch.delenv("ER_LAB_TIER", raising=False)
    monkeypatch.delenv("ER_LAB_ARTIFACTS", raising=False)

    # The stand-in mirrors _ensure_config_module for the *kernel* process; once
    # the real er_lab.config lands, the try-import wins and it is inert.
    stub_cell = textwrap.dedent(
        """
        import hashlib, sys, types
        try:
            import er_lab.config
        except ImportError:
            from omegaconf import OmegaConf
            stub = types.ModuleType("er_lab.config")
            stub.config_hash = lambda cfg: hashlib.sha256(
                OmegaConf.to_yaml(cfg, resolve=True).encode()).hexdigest()[:12]
            sys.modules["er_lab.config"] = stub
        """
    )
    register_cell = textwrap.dedent(
        """
        import os
        from omegaconf import OmegaConf
        from er_lab.infra.artifacts import ArtifactRegistry
        tier = os.environ["ER_LAB_TIER"]
        root = os.environ["ER_LAB_ARTIFACTS"]
        cfg = OmegaConf.create({"run": {"seed": 17, "seeds": [17]}})
        ArtifactRegistry(root).register(
            "env_probe", {"tier": tier, "root": root}, cfg=cfg, tier=tier)
        """
    )
    nb_path = tmp_path / "90_env_probe.ipynb"
    _write_notebook(nb_path, [stub_cell, register_cell])
    root = tmp_path / "artifacts"

    runner.run_notebook(nb_path, "smoke", root)

    assert "ER_LAB_TIER" not in os.environ  # env restored after the run
    assert "ER_LAB_ARTIFACTS" not in os.environ
    payload, meta = ArtifactRegistry(root).load("env_probe", tier="smoke")
    assert payload == {"tier": "smoke", "root": str(root.resolve())}
    assert meta["tier"] == "smoke"
    assert meta["seed"] == 17
    executed = nbformat.read(str(nb_path), as_version=4)  # written back with outputs
    assert all(cell.execution_count is not None for cell in executed.cells)


def test_run_all_fails_on_missing_notebook_file(tmp_path):
    with pytest.raises(FileNotFoundError) as exc:
        runner.run_all("smoke", upto="00", notebooks_dir=tmp_path, artifacts_root=tmp_path / "a")
    assert "00_hardest_easy_problem.ipynb" in str(exc.value)


# ---------------------------------------------------------------- er_lab.run


def test_parse_argv_splits_nb_from_dotlist():
    nb, dotlist = er_run.parse_argv(["nb=05", "run.tier=smoke", "a.b=c"])
    assert nb == "05"
    assert dotlist == ["run.tier=smoke", "a.b=c"]


def test_parse_argv_without_nb():
    nb, dotlist = er_run.parse_argv(["run.tier=smoke"])
    assert nb is None
    assert dotlist == ["run.tier=smoke"]


def test_parse_argv_keeps_values_with_equals_signs():
    nb, dotlist = er_run.parse_argv(["nb=all", "run.name=lr=probe"])
    assert nb == "all"
    assert dotlist == ["run.name=lr=probe"]


def test_parse_argv_rejects_bare_tokens():
    with pytest.raises(SystemExit):
        er_run.parse_argv(["notebook05"])
    with pytest.raises(SystemExit):
        er_run.parse_argv(["=smoke"])
