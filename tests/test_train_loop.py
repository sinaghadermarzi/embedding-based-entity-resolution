"""Tests for er_lab.train.loop (and the er_lab.train.augment adapter it drives).

All CPU (infra.device=cpu via config, never hardcoded past the cfg), tiny
encoder, ~200 synthetic records — the whole file stays well under a minute.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import torch
from omegaconf import OmegaConf

from er_lab.config import load_config, set_all_seeds
from er_lab.models.encoders import build_encoder
from er_lab.train.augment import make_augmenter
from er_lab.train.loop import HISTORY_COLUMNS, train_encoder

SEED = 7
PROBE = "[COL] given_name [VAL] WILLIAM [COL] family_name [VAL] SMITH"

GIVEN = [
    "WILLIAM",
    "ROBERT",
    "ELIZABETH",
    "MARGARET",
    "JAMES",
    "KATHERINE",
    "MICHAEL",
    "PATRICIA",
    "JOHN",
    "MARY",
]
FAMILY = [
    "SMITH",
    "GARCIA",
    "JOHNSON",
    "MCDONALD",
    "OBRIEN",
    "LEE",
    "MARTINEZ",
    "BROWN",
    "DAVIS",
    "WILSON",
]


def make_corpus(n_ent: int = 100) -> pd.DataFrame:
    """~200 records: n_ent entities x 2 records, the duplicate lightly varied."""
    rows = []
    for e in range(n_ent):
        g, f = GIVEN[e % 10], FAMILY[(e // 10) % 10]
        for c in range(2):
            rows.append(
                {
                    "record_id": f"r{e:03d}_{c}",
                    "entity_id": f"e{e:03d}",
                    "given_name": g if c == 0 else g[:-1] + "N",
                    "family_name": f,
                    "city": "DURHAM",
                    "zip": f"27{e:03d}",
                }
            )
    return pd.DataFrame(rows, dtype="string")


def tiny_cfg(batch: int = 16) -> OmegaConf:
    # every knob is a typed dotlist key — the PLAN's dotlist-driven contract
    return load_config(
        dotlist=[
            "infra.device=cpu",
            f"train.batch_size={batch}",
            "train.lr=1e-3",
            "train.miner_k=5",
            "model.dim=32",
            "model.max_len=96",
            "model.layers=1",
            "model.heads=2",
        ]
    )


def run(corpus, cfg, **kw):
    defaults = {
        "loss_name": "infonce",
        "miner_name": "inbatch",
        "augment_kind": "none",
        "seed": SEED,
    }
    defaults.update(kw)
    set_all_seeds(SEED)
    encoder = build_encoder(cfg)
    return train_encoder(encoder, corpus, cfg, **defaults)


@pytest.fixture(scope="module")
def main_run():
    """One 60-step run, shared across assertions; optimizer.step calls counted."""
    calls = {"n": 0}
    real_adamw = torch.optim.AdamW

    class CountingAdamW(real_adamw):
        def step(self, *args, **kwargs):
            calls["n"] += 1
            return super().step(*args, **kwargs)

    mp = pytest.MonkeyPatch()
    mp.setattr(torch.optim, "AdamW", CountingAdamW)
    try:
        encoder, history = run(make_corpus(), tiny_cfg(), steps=60)
    finally:
        mp.undo()
    probe = encoder.encode([PROBE])
    return history, probe, calls["n"]


def test_budget_is_exact_optimizer_step_count(main_run):
    history, _probe, n_step_calls = main_run
    assert n_step_calls == 60  # the budget unit, counted at the optimizer
    assert len(history) == 60
    assert history["step"].tolist() == list(range(1, 61))


def test_history_schema(main_run):
    history, _probe, _n = main_run
    assert (
        list(history.columns) == HISTORY_COLUMNS == ["step", "loss", "fn_contamination_rate", "lr"]
    )
    assert np.isfinite(history["loss"]).all()
    assert (history["lr"] == 1e-3).all()
    assert history["fn_contamination_rate"].isna().all()  # inbatch miner: nothing mined


def test_smoothed_loss_strictly_decreases(main_run):
    history, _probe, _n = main_run
    smoothed = history["loss"].rolling(10).mean().dropna()
    assert smoothed.iloc[-1] < smoothed.iloc[0]
    assert history["loss"].tail(10).mean() < 0.8 * history["loss"].head(10).mean()


def test_determinism_two_runs_identical_probe_embedding(main_run):
    history_a, probe_a, _n = main_run
    encoder_b, history_b = run(make_corpus(), tiny_cfg(), steps=60)
    probe_b = encoder_b.encode([PROBE])
    assert np.array_equal(probe_a, probe_b)
    assert history_a["loss"].tolist() == history_b["loss"].tolist()


def test_generic_typo_augmentation_exercises_noise_channels():
    pytest.importorskip("gecko")  # noise.channels wraps gecko mutators
    _encoder, history = run(make_corpus(), tiny_cfg(batch=8), augment_kind="generic_typo", steps=6)
    assert len(history) == 6
    assert np.isfinite(history["loss"]).all()


def test_bm25_miner_records_fn_contamination():
    _encoder, history = run(make_corpus(), tiny_cfg(batch=8), miner_name="bm25", steps=7)
    assert history["step"].tolist() == list(range(1, 8))  # exact budget, non-round count
    rate = history["fn_contamination_rate"]
    assert rate.notna().all()
    assert ((rate >= 0) & (rate <= 1)).all()
    assert rate.iloc[0] > 0  # near-identical duplicates get mined: the trap is visible


@pytest.mark.parametrize("loss_name", ["supcon", "triplet", "cosent"])
def test_other_loss_families_run(loss_name):
    _encoder, history = run(make_corpus(30), tiny_cfg(batch=8), loss_name=loss_name, steps=3)
    assert np.isfinite(history["loss"]).all()


def test_ann_filtered_with_remine_cadence():
    _encoder, history = run(
        make_corpus(30), tiny_cfg(batch=8), miner_name="ann_filtered", steps=6, eval_every=3
    )
    # contamination measured on the RAW mined table even though training uses the filtered one
    assert history["fn_contamination_rate"].notna().all()
    assert history["fn_contamination_rate"].iloc[0] >= 0


def test_ann_filtered_uses_supplied_non_truth_proxy(monkeypatch):
    """The deployable-proxy arm: filter_proxy reaches cluster_aware_filter; the
    default (None) stays the documented entity_id oracle arm."""
    import er_lab.train.loop as loop_mod

    captured: list[pd.Series] = []
    real_filter = loop_mod.cluster_aware_filter

    def spying_filter(mined, truth_proxy):
        captured.append(truth_proxy)
        return real_filter(mined, truth_proxy)

    monkeypatch.setattr(loop_mod, "cluster_aware_filter", spying_filter)

    corpus = make_corpus(30)
    # coarse non-truth proxy: first letter of the family name (record_id -> key)
    proxy = corpus.set_index(corpus["record_id"].astype(str))["family_name"].str[0]
    assert proxy.nunique() < corpus["entity_id"].nunique()  # genuinely not the truth
    run(corpus, tiny_cfg(batch=8), miner_name="ann_filtered", steps=2, filter_proxy=proxy)
    assert len(captured) == 1
    assert captured[0] is proxy  # the supplied proxy, not entity_id

    captured.clear()
    run(corpus, tiny_cfg(batch=8), miner_name="ann_filtered", steps=2)
    assert (
        captured[0] == corpus.set_index(corpus["record_id"].astype(str))["entity_id"]
    ).all()  # oracle default


def test_cfg_driven_serialization_scheme():
    cfg = tiny_cfg(batch=8)
    cfg.serialize.scheme = "template"  # typed SerializeConfig node
    cfg.serialize.missing = "drop"
    _encoder, history = run(make_corpus(30), cfg, steps=2)
    assert len(history) == 2


def test_validation_errors():
    corpus, cfg = make_corpus(20), tiny_cfg(batch=8)
    with pytest.raises(ValueError, match="loss"):
        run(corpus, cfg, loss_name="hinge", steps=1)
    with pytest.raises(ValueError, match="miner"):
        run(corpus, cfg, miner_name="oracle", steps=1)
    with pytest.raises(ValueError, match="steps"):
        run(corpus, cfg, steps=0)
    dup = pd.concat([corpus, corpus.iloc[[0]]], ignore_index=True)
    with pytest.raises(ValueError, match="unique"):
        run(dup, cfg, steps=1)
    singletons = corpus.drop_duplicates("entity_id")
    with pytest.raises(ValueError, match="positive pairs"):
        run(singletons, cfg, steps=1)


def test_multi_gpu_without_accelerate_raises_clear_error():
    """The missing-extra error path (only reachable when accelerate is absent)."""
    try:
        import accelerate  # noqa: F401

        pytest.skip("accelerate installed: the missing-extra error cannot occur")
    except ImportError:
        pass
    cfg = tiny_cfg()
    cfg.infra.multi_gpu = True
    with pytest.raises(RuntimeError, match="multi-gpu"):
        run(make_corpus(20), cfg, steps=1)


_ACCEL_SCRIPT = """
import sys

sys.path.insert(0, {tests_dir!r})
import numpy as np
from accelerate.state import PartialState
from test_train_loop import PROBE, make_corpus, tiny_cfg

from er_lab.config import set_all_seeds
from er_lab.models.encoders import build_encoder
from er_lab.train.loop import train_encoder

state = PartialState(cpu=True)  # bare PartialState does not parse ACCELERATE_USE_CPU
assert state.num_processes == 2, f"expected 2 DDP processes, got {{state.num_processes}}"
assert str(state.distributed_type) == "DistributedType.MULTI_CPU", state.distributed_type

cfg = tiny_cfg(batch=8)
cfg.infra.multi_gpu = True
set_all_seeds(7)
encoder = build_encoder(cfg)
encoder, history = train_encoder(
    encoder, make_corpus(30), cfg,
    loss_name="infonce", miner_name="ann", augment_kind="none",
    steps=3, eval_every=2, seed=7,
)
assert len(history) == 3
assert np.isfinite(history["loss"]).all()
# the returned module is unwrapped: encode() must exist and run per-process
probe = encoder.encode([PROBE])
np.save({out_dir!r} + f"/probe_{{state.process_index}}.npy", probe)
"""


@pytest.mark.slow
def test_multi_gpu_ddp_path_two_process_cpu(tmp_path):
    """The accelerate branch under a REAL 2-process `accelerate launch` on CPU:
    embeddings must route through the DDP wrapper's forward (custom methods
    like embed_batch do not exist on the wrapper and would AttributeError),
    re-mining must unwrap for encode(), and the synced ranks must agree."""
    import os
    import subprocess
    import sys

    pytest.importorskip("accelerate")
    script = tmp_path / "ddp_probe.py"
    script.write_text(
        _ACCEL_SCRIPT.format(tests_dir=os.path.dirname(__file__), out_dir=str(tmp_path))
    )
    port = 29510 + os.getpid() % 400  # avoid clashes with concurrent launches
    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "accelerate.commands.launch",
            # --multi_gpu + ACCELERATE_USE_CPU is accelerate's supported spelling
            # of true multi-PROCESS CPU (gloo DDP); --cpu alone stays 1-process
            "--multi_gpu",
            "--num_processes",
            "2",
            "--num_machines",
            "1",
            "--main_process_ip",
            "127.0.0.1",
            "--main_process_port",
            str(port),
            str(script),
        ],
        env={**os.environ, "ACCELERATE_USE_CPU": "true"},
        capture_output=True,
        text=True,
        timeout=240,
        check=False,  # the assert below reports stdout/stderr on failure
    )
    assert proc.returncode == 0, f"stdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
    probes = [np.load(tmp_path / f"probe_{rank}.npy") for rank in (0, 1)]
    assert np.allclose(probes[0], probes[1], atol=1e-6)  # DDP-synced weights agree


# --- the augment adapter ----------------------------------------------------


def test_augment_none_is_identity():
    corpus = make_corpus(5)
    augment = make_augmenter("none", seed=0)
    assert augment(corpus) is corpus


def test_augment_unknown_kind_raises():
    with pytest.raises(ValueError, match="augment kind"):
        make_augmenter("heavy_metal", seed=0)


def test_augment_calibrated_requires_rates():
    # validated before the lazy noise-channel import: no gecko needed for the error
    with pytest.raises(ValueError, match="channel_rates"):
        make_augmenter("calibrated", seed=0)


def test_augment_generic_typo_perturbs_and_is_seed_deterministic():
    pytest.importorskip("gecko")
    corpus = make_corpus(20)
    out_a = make_augmenter("generic_typo", channel_rates={"typo": 1.0}, seed=5)(corpus)
    out_b = make_augmenter("generic_typo", channel_rates={"typo": 1.0}, seed=5)(corpus)
    pd.testing.assert_frame_equal(out_a, out_b)
    changed = (out_a["given_name"] != corpus["given_name"]) | (
        out_a["family_name"] != corpus["family_name"]
    )
    assert changed.any()  # rate 1.0 must actually touch records
    assert (out_a["record_id"] == corpus["record_id"]).all()  # ids never corrupted


def test_augment_unknown_channel_raises():
    pytest.importorskip("gecko")
    with pytest.raises(ValueError, match="unknown noise channels"):
        make_augmenter("calibrated", channel_rates={"gremlins": 0.1}, seed=0)
