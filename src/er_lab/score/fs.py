"""Fellegi–Sunter scoring via Splink 4 (DuckDB) — the BAS-01 baseline adapter.

The lab's working hypothesis is judged against "a genuinely well-tuned FS
baseline, measured ourselves" (PLAN §3, BAS-01). This adapter keeps the tuning
that matters and nothing else:

- Jaro–Winkler-binned comparators on name fields (Splink's ``NameComparison``:
  exact / JW>=0.92 / JW>=0.88 / JW>=0.70 / else),
- term-frequency adjustment on given/family (and full) name exact matches, so
  a shared "SMITH" counts for less than a shared "ALJUNDI",
- m/u trained by expectation maximisation (u seeded by deterministic random
  sampling, the prior by deterministic rules), OR-combined blocking rules.

Splink-4 API notes (verified against the installed splink 4.0.16):
``Linker`` + ``SettingsCreator`` + ``linker.training.*`` for fitting, and
``linker.inference.predict()`` for rule-blocked scoring. Scoring an *explicit*
candidate-pair list (pairs mode — how blocking arms hand their candidates to
FS) has no public one-call API, so :func:`fs_scores` does what Splink itself
does internally for exactly this job (see ``Linker.inference.
_score_missing_cluster_edges``): register the pairs as a DuckDB table, emit it
as the ``__splink__blocked_id_pairs`` junction table, then reuse Splink's own
comparison-vector and match-weight SQL. That scores exactly the requested
pairs — no blocking-rule detour, no cartesian filter.
"""

from __future__ import annotations

import uuid

import pandas as pd

from er_lab.data.schema import DeclaredSchema

__all__ = ["fit_fs", "fs_scores"]

#: Fixed seed for Splink's u-estimation random sampling — fit_fs has no seed
#: parameter (the FS baseline is one deterministic fit, not a seeded arm).
_U_SAMPLING_SEED = 17

#: Assumed recall of the deterministic prior-estimation rules (Splink docs'
#: conventional starting point; the prior only shifts probabilities globally,
#: never the rank order BAS-01 compares on).
_PRIOR_RECALL = 0.9


def _usable_roles(df: pd.DataFrame, schema: DeclaredSchema) -> list[str]:
    """Roles present in both schema and frame with >=2 distinct non-null values.

    Constant or empty columns are skipped: they carry no pairwise signal and
    push EM toward degenerate, never-observed comparison levels.
    """
    usable = []
    for role in schema.text_roles():
        if role not in df.columns:
            continue
        non_null = df[role].dropna()
        if len(non_null) >= 2 and non_null.nunique() >= 2:
            usable.append(role)
    return usable


def _comparisons(roles: list[str]) -> list:
    """The curated role -> Splink comparison map (see module docstring)."""
    import splink.comparison_library as cl

    have = set(roles)
    comps = []
    if "given_name" in have:
        comps.append(cl.NameComparison("given_name"))  # JW-binned + TF on exact
    if "family_name" in have:
        comps.append(cl.NameComparison("family_name"))
    if "full_name" in have and not {"given_name", "family_name"} <= have:
        comps.append(cl.NameComparison("full_name"))
    if "middle_name" in have:
        comps.append(cl.JaroWinklerAtThresholds("middle_name", [0.92, 0.88]))
    for role, comp in (
        ("dob", lambda: cl.LevenshteinAtThresholds("dob", [1, 2])),
        ("birth_year", lambda: cl.ExactMatch("birth_year")),
        ("age", lambda: cl.ExactMatch("age")),
        ("sex", lambda: cl.ExactMatch("sex")),
        ("city", lambda: cl.JaroWinklerAtThresholds("city", [0.92])),
        ("state", lambda: cl.ExactMatch("state")),
        ("zip", lambda: cl.LevenshteinAtThresholds("zip", 1)),
        ("phone", lambda: cl.LevenshteinAtThresholds("phone", 1)),
        ("email", lambda: cl.LevenshteinAtThresholds("email", 2)),
        ("street", lambda: cl.JaroWinklerAtThresholds("street", [0.92, 0.88])),
        ("street_address", lambda: cl.JaroWinklerAtThresholds("street_address", [0.92])),
        ("house_number", lambda: cl.ExactMatch("house_number")),
        ("name_suffix", lambda: cl.ExactMatch("name_suffix")),
    ):
        if role in have:
            comps.append(comp())
    if not comps:
        raise ValueError("fit_fs: no usable role columns to build FS comparisons from")
    return comps


def _rule_roles(roles: list[str]) -> tuple[list[str], str | None, str | None]:
    """(prediction-blocking roles, name-side EM key or None, non-name EM key or None)."""
    have = set(roles)
    name_key = None
    if {"given_name", "family_name"} <= have:
        name_key = "given_name,family_name"
    elif "family_name" in have:
        name_key = "family_name"
    elif "full_name" in have:
        name_key = "full_name"
    other_key = next((r for r in ("dob", "zip", "phone", "birth_year") if r in have), None)
    predict = []
    for r in ("family_name", "full_name", "dob", "zip", "phone"):
        if r in have:
            predict.append(r)
    return predict, name_key, other_key


def _em_summary(session) -> dict:
    """Convergence record for one EM session (iterations + final parameter change)."""
    from splink.internals.expectation_maximisation import (
        _max_change_in_parameters_comparison_levels,
    )

    history = session._core_model_settings_history
    iterations = len(history) - 1
    tol = session.training_settings.em_convergence
    final_change = (
        _max_change_in_parameters_comparison_levels(history)["max_abs_change_value"]
        if iterations >= 1
        else float("inf")
    )
    return {
        "blocking_rule": session._blocking_rule_for_training.blocking_rule_sql,
        "iterations": iterations,
        "max_iterations": session.training_settings.max_iterations,
        "em_convergence": tol,
        "final_change": final_change,
        "converged": final_change < tol,
    }


def fit_fs(
    df: pd.DataFrame,
    *,
    schema: DeclaredSchema,
    blocking_rules: list[str] | None = None,
    max_pairs: float = 1e6,
):
    """Fit the tuned FS baseline on a canonical role frame; returns the Splink linker.

    *df* is a canonical frame (:meth:`DeclaredSchema.to_canonical` output:
    role columns + record_id). Only record_id and usable role columns are
    handed to Splink — entity_id/source never enter the model (truth must not
    leak into the baseline). *blocking_rules* entries are either bare role
    names (turned into ``block_on(role)``) or raw Splink SQL (anything
    containing ``l.``); None derives OR-union defaults from the schema.
    *max_pairs* is Splink's u-sampling budget — raise it for target-tier fits.

    Training recipe (all deterministic): prior from an AND-rule over the two
    strongest keys at assumed recall 0.9; u by seeded random sampling; two EM
    sessions with complementary blocking (block on the non-name key to train
    the name m's, block on names to train the rest). Convergence facts are
    attached as ``linker._er_lab_training['em_sessions']`` — BAS-01 asserts on
    them instead of trusting log lines.
    """
    from splink import DuckDBAPI, Linker, SettingsCreator, block_on

    if "record_id" not in df.columns:
        raise KeyError("fit_fs: frame has no 'record_id' column — pass a canonical role frame")
    roles = _usable_roles(df, schema)
    comparisons = _comparisons(roles)
    predict_roles, name_key, other_key = _rule_roles(roles)

    if blocking_rules is not None:
        predict_rules = [
            r if "l." in r else block_on(*r.split(",")) for r in blocking_rules
        ]
    else:
        if not predict_roles:
            raise ValueError(
                f"fit_fs: schema '{schema.name}' has no default-blockable roles; "
                "pass blocking_rules= explicitly"
            )
        predict_rules = [block_on(r) for r in predict_roles]

    settings = SettingsCreator(
        link_type="dedupe_only",
        unique_id_column_name="record_id",
        comparisons=comparisons,
        blocking_rules_to_generate_predictions=predict_rules,
        retain_intermediate_calculation_columns=False,
    )
    model_cols = ["record_id"] + roles
    linker = Linker(df[model_cols], settings, db_api=DuckDBAPI())

    prior_keys = [k for k in (name_key, other_key) if k is not None]
    if prior_keys:
        prior_rule = block_on(*[c for k in prior_keys for c in k.split(",")])
        linker.training.estimate_probability_two_random_records_match(
            [prior_rule], recall=_PRIOR_RECALL
        )
    linker.training.estimate_u_using_random_sampling(
        max_pairs=max_pairs, seed=_U_SAMPLING_SEED
    )

    em_sessions = []
    # Block on the non-name key to train name m's, and vice versa: EM cannot
    # estimate parameters for comparisons that appear in its own blocking rule.
    for key in (other_key, name_key):
        if key is None:
            continue
        session = linker.training.estimate_parameters_using_expectation_maximisation(
            block_on(*key.split(","))
        )
        em_sessions.append(_em_summary(session))

    linker._er_lab_training = {"em_sessions": em_sessions, "roles": roles}
    return linker


def _pairs_mode_scores(linker, pairs: pd.DataFrame) -> pd.DataFrame:
    """Score exactly *pairs* — Splink's own junction-table mechanism, see module doc."""
    from splink.internals.comparison_vector_values import (
        compute_comparison_vector_values_from_id_pairs_sqls,
    )
    from splink.internals.pipeline import CTEPipeline
    from splink.internals.predict import (
        predict_from_comparison_vectors_sqls_using_settings,
    )
    from splink.internals.vertically_concatenate import compute_df_concat_with_tf

    sdf_pairs = linker.table_management.register_table(
        pairs, f"__er_lab_pairs_{uuid.uuid4().hex[:8]}", overwrite=True
    )
    try:
        settings_obj = linker._settings_obj
        pipeline = CTEPipeline()
        nodes_with_tf = compute_df_concat_with_tf(linker, pipeline)
        pipeline = CTEPipeline([nodes_with_tf])
        # The junction table Splink's comparison-vector SQL joins against:
        # one row per requested pair, keyed exactly like blocked id pairs.
        pipeline.enqueue_sql(
            f"select '0' as match_key, p.a as join_key_l, p.b as join_key_r "
            f"from {sdf_pairs.physical_name} p",
            "__splink__blocked_id_pairs",
        )
        pipeline.enqueue_list_of_sqls(
            compute_comparison_vector_values_from_id_pairs_sqls(
                settings_obj._columns_to_select_for_blocking,
                settings_obj._columns_to_select_for_comparison_vector_values,
                input_tablename_l="__splink__df_concat_with_tf",
                input_tablename_r="__splink__df_concat_with_tf",
                source_dataset_input_column=(
                    settings_obj.column_info_settings.source_dataset_input_column
                ),
                unique_id_input_column=settings_obj.column_info_settings.unique_id_input_column,
            )
        )
        pipeline.enqueue_list_of_sqls(
            predict_from_comparison_vectors_sqls_using_settings(
                settings_obj, None, None, sql_infinity_expression=linker._infinity_expression
            )
        )
        predictions = linker._db_api.sql_pipeline_to_splink_dataframe(pipeline)
        try:
            return predictions.as_pandas_dataframe()
        finally:
            predictions.drop_table_from_database_and_remove_from_cache()
    finally:
        # force_non_splink_table: the pairs table is ours, not a Splink-created one
        sdf_pairs.drop_table_from_database_and_remove_from_cache(force_non_splink_table=True)


def fs_scores(linker, *, pairs: pd.DataFrame | None = None) -> pd.DataFrame:
    """Match probabilities as ``DataFrame[a, b, prob]``, a<b, sorted by (a, b).

    With ``pairs=None``, scores the pairs generated by the linker's own
    blocking rules (``linker.inference.predict()``). With a candidate table
    ``pairs`` (columns a, b — e.g. a blocking arm's output), scores *exactly*
    those pairs: input pairs are canonicalized (a<b) and deduped, self-pairs
    raise, and a pair naming a record_id absent from the fitted frame raises
    rather than being silently dropped.
    """
    uid_name = linker._settings_obj.column_info_settings.unique_id_column_name
    col_l, col_r = f"{uid_name}_l", f"{uid_name}_r"

    if pairs is None:
        predictions = linker.inference.predict()
        try:
            raw = predictions.as_pandas_dataframe()
        finally:
            predictions.drop_table_from_database_and_remove_from_cache()
    else:
        if not {"a", "b"} <= set(pairs.columns):
            raise KeyError("fs_scores: pairs must have columns 'a' and 'b'")
        req = pd.DataFrame(
            {"a": pairs["a"].astype(str).to_numpy(), "b": pairs["b"].astype(str).to_numpy()}
        )
        if (req["a"] == req["b"]).any():
            raise ValueError("fs_scores: pairs contains self-pairs (a == b)")
        swap = req["a"] > req["b"]
        req.loc[swap, ["a", "b"]] = req.loc[swap, ["b", "a"]].to_numpy()
        req = req.drop_duplicates(ignore_index=True)
        raw = _pairs_mode_scores(linker, req)
        if len(raw) != len(req):
            scored = set(zip(raw[col_l].astype(str), raw[col_r].astype(str)))
            missing = [p for p in zip(req["a"], req["b"]) if p not in scored]
            raise ValueError(
                f"fs_scores: {len(missing)} requested pair(s) reference record_ids "
                f"absent from the fitted frame, e.g. {missing[:3]}"
            )

    out = pd.DataFrame(
        {
            "a": raw[col_l].astype("string"),
            "b": raw[col_r].astype("string"),
            "prob": raw["match_probability"].astype(float),
        }
    )
    swap = out["a"] > out["b"]
    out.loc[swap, ["a", "b"]] = out.loc[swap, ["b", "a"]].to_numpy()
    return out.sort_values(["a", "b"], ignore_index=True)
