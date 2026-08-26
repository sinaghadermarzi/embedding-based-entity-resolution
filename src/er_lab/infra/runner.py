"""Headless notebook DAG runner.

``NOTEBOOK_DAG`` declares, in series order, which registered artifacts each
notebook requires and produces (PLAN §6). ``run_notebook`` executes one
notebook via nbclient with ``ER_LAB_TIER`` / ``ER_LAB_ARTIFACTS`` /
``ER_LAB_DOTLIST`` (the CLI overrides, as JSON — consumed by
``er_lab.config.load_config_from_env``) injected into the kernel environment —
and hard-fails *before* starting the kernel, with the 'not yet run' placard,
if a required upstream artifact is missing for that tier. One code path across
tiers: the tier only ever changes via config/env, never via forked notebook
logic.
"""

from __future__ import annotations

import json
import os
from contextlib import contextmanager
from pathlib import Path

import nbformat
from nbclient import NotebookClient

from er_lab.config import REPO_ROOT
from er_lab.infra.artifacts import ArtifactMissing, ArtifactRegistry, check_tier

NOTEBOOKS_DIR = REPO_ROOT / "notebooks"

# Series order is execution order (dicts preserve insertion order).
NOTEBOOK_DAG: dict[str, dict[str, list[str]]] = {
    "00_hardest_easy_problem.ipynb": {
        "requires": [],
        "produces": ["smoke_check", "quickstart_dedup"],
    },
    "01_corpora_and_schemas.ipynb": {
        "requires": [],
        "produces": [
            "corpus_registry",
            "declared_schemas",
            "nc_snapshot_listing",
            "nc_align_summary",
        ],
    },
    "02_how_to_tell_who_won.ipynb": {
        "requires": ["corpus_registry"],
        "produces": [
            "met01_metric_rankings",
            "met02_operating_point_map",
            "met03_bootstrap_coverage",
        ],
    },
    "03_is_the_truth_true.ipynb": {
        "requires": ["corpus_registry"],
        "produces": ["met05_ncid_audit", "met06_identifiability", "met05_adjudication_plan"],
    },
    "04_auditing_real_noise.ipynb": {
        "requires": ["corpus_registry"],
        "produces": ["nse01_channel_prevalence", "nse01_exposure_model"],
    },
    "05_calibrated_dirt_machine.ipynb": {
        "requires": ["nse01_channel_prevalence"],
        "produces": ["calibrated_corpus", "nse02_fidelity_scores", "met07_splits"],
    },
    "06_baseline_to_beat.ipynb": {
        "requires": ["calibrated_corpus", "met07_splits"],
        "produces": ["bas01_fs_baseline"],
    },
    "07_chain_merge_catastrophe.ipynb": {
        "requires": ["calibrated_corpus", "bas01_fs_baseline"],
        "produces": ["chain_merge_demo"],
    },
    "08_person_as_a_vector.ipynb": {
        "requires": ["calibrated_corpus", "met07_splits"],
        "produces": [
            "trn05_serialization_matrix",
            "invariance_battery",
            "embedding_geometry_panel",
        ],
    },
    "09_losses_and_negatives.ipynb": {
        "requires": ["calibrated_corpus", "met07_splits"],
        "produces": ["met04_power_table", "trn01_loss_matrix", "trn02_miner_matrix"],
    },
    "10_typos_tokenizer_or_augmentation.ipynb": {
        "requires": ["calibrated_corpus", "met07_splits", "met04_power_table"],
        "produces": ["trn03_augmentation_matrix", "trn04_encoder_head_to_head"],
    },
    "11_nicknames_missing_circular.ipynb": {
        "requires": ["calibrated_corpus", "met07_splits"],
        "produces": ["trn06_nickname_matrix", "label_provenance_audit"],
    },
    "12_candidates_in_ten_million.ipynb": {
        "requires": ["calibrated_corpus", "trn01_loss_matrix"],
        "produces": ["bas02_blocking_frontier"],
    },
    "13_from_pairs_to_people.ipynb": {
        "requires": ["calibrated_corpus", "bas02_blocking_frontier"],
        "produces": ["cal01_calibration_map", "clu01_clustering_scores"],
    },
    "14_where_rules_earn_their_place.ipynb": {
        "requires": ["bas01_fs_baseline", "cal01_calibration_map", "clu01_clustering_scores"],
        "produces": ["hyb01_rule_value_map", "nse03_ranking_invariance", "prs01_parsing_factorial"],
    },
    "15_last_two_orders_of_magnitude.ipynb": {
        "requires": ["bas02_blocking_frontier", "clu01_clustering_scores"],
        "produces": [
            "scl01_scaling_fits",
            "scl02_tail_fits",
            "eff01_compression_matrix",
            "scl03_cost_model",
            "scl04_ambiguity_budget",
        ],
    },
    "16_unequal_noise_unequal_errors.ipynb": {
        "requires": ["calibrated_corpus", "nse01_exposure_model", "cal01_calibration_map"],
        "produces": ["fair01_disparity_panel"],
    },
    "17_verdict_at_1e7.ipynb": {
        "requires": [
            "bas01_fs_baseline",
            "hyb01_rule_value_map",
            "clu01_clustering_scores",
            "scl01_scaling_fits",
        ],
        "produces": ["verdict_1e7", "honesty_audit"],
    },
    "18_your_data_your_schema.ipynb": {
        "requires": ["declared_schemas", "calibrated_corpus"],
        "produces": ["adp01_adaptation_report", "adp02_linkage_results"],
    },
    "A_appendix_exploratory_arms.ipynb": {
        "requires": ["calibrated_corpus"],
        "produces": ["x01_reranker_value", "x02_id_churn", "eff02_static_distill"],
    },
}


def producer_of(artifact: str) -> str | None:
    """Notebook filename that produces *artifact*, or None if no notebook declares it."""
    for nb_name, spec in NOTEBOOK_DAG.items():
        if artifact in spec["produces"]:
            return nb_name
    return None


def resolve_notebook(spec: str) -> str:
    """Resolve a notebook number/name/path ('5', '05', '05_...', path) to its DAG key."""
    name = Path(str(spec)).name
    if name in NOTEBOOK_DAG:
        return name
    stem = name.removesuffix(".ipynb")
    if stem.isdigit():
        stem = stem.zfill(2)
    matches = [
        k for k in NOTEBOOK_DAG if k.removesuffix(".ipynb") == stem or k.startswith(stem + "_")
    ]
    if len(matches) == 1:
        return matches[0]
    known = ", ".join(NOTEBOOK_DAG)
    raise KeyError(f"cannot resolve notebook {spec!r} to one of: {known}")


def run_notebook(
    nb_path: str | Path,
    tier: str,
    artifacts_root: str | Path,
    dotlist: list[str] | None = None,
    *,
    cell_timeout: int = 1800,
) -> None:
    """Execute *nb_path* headlessly at *tier*, gated on its upstream artifacts.

    *dotlist* (the CLI config overrides) is forwarded to the kernel as JSON in
    ``ER_LAB_DOTLIST``; notebooks apply it via ``load_config_from_env()``.
    Notebooks not listed in ``NOTEBOOK_DAG`` (scratch/test notebooks) run
    ungated. The executed notebook is written back in place even when a cell
    fails, so partial outputs and the traceback are preserved for debugging.
    """
    check_tier(tier)
    nb_path = _anchor(nb_path)
    root = _anchor(artifacts_root).resolve()
    if not nb_path.is_file():
        raise FileNotFoundError(f"notebook {nb_path} does not exist (not built yet?)")
    _check_requires(NOTEBOOK_DAG.get(nb_path.name, {}).get("requires", []), tier, root)

    nb = nbformat.read(str(nb_path), as_version=4)
    with _injected_env(
        ER_LAB_TIER=tier,
        ER_LAB_ARTIFACTS=str(root),
        ER_LAB_DOTLIST=json.dumps(list(dotlist or [])),
    ):
        client = NotebookClient(
            nb,
            timeout=cell_timeout,
            kernel_name="python3",
            resources={"metadata": {"path": str(nb_path.parent)}},
        )
        try:
            client.execute()
        finally:
            nbformat.write(nb, str(nb_path))


def run_all(
    tier: str,
    upto: str | None = None,
    *,
    notebooks_dir: str | Path = NOTEBOOKS_DIR,
    artifacts_root: str | Path = "artifacts",
    dotlist: list[str] | None = None,
) -> None:
    """Run the series in DAG order, through *upto* (number or name) inclusive."""
    check_tier(tier)
    stop = None if upto is None else resolve_notebook(upto)
    notebooks_dir = _anchor(notebooks_dir)
    for nb_name in NOTEBOOK_DAG:
        run_notebook(notebooks_dir / nb_name, tier, artifacts_root, dotlist=dotlist)
        if nb_name == stop:
            break


def _anchor(path: str | Path) -> Path:
    """Absolute paths pass through; relative ones anchor at the repo root, not the CWD."""
    path = Path(path)
    return path if path.is_absolute() else REPO_ROOT / path


def _check_requires(requires: list[str], tier: str, root: Path) -> None:
    """Raise before any kernel starts if an upstream artifact is missing or tier-mixed."""
    registry = ArtifactRegistry(root)
    placards = []
    for name in requires:
        try:
            registry._newest_run(name, tier)  # raises TierMixingError eagerly too
        except ArtifactMissing as err:
            placards.append(str(err))
    if placards:
        raise ArtifactMissing("\n\n".join(placards))


@contextmanager
def _injected_env(**env: str):
    """Temporarily set env vars; the nbclient kernel inherits the parent environment."""
    saved = {k: os.environ.get(k) for k in env}
    os.environ.update(env)
    try:
        yield
    finally:
        for key, old in saved.items():
            if old is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = old
