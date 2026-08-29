# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # 14. Where Rules Earn Their Place
#
# **The question.** The lab's working hypothesis (PLAN §1) keeps deterministic rules
# "wherever they demonstrably earn their place" — this is the notebook where they must
# demonstrate it. Three rules with three different jobs face a calibrated whole-record
# embedding on the same candidate graph: a **hard override** (exact name+birth key forces a
# link), a **guard** (a birth-year gap blocks one), and a **feature** (an agreement pattern
# floors the match probability). Each is priced by its *marginal* entity-level value next to
# the embedding — not by whether it "sounds safe" — and then the whole verdict is stress-
# tested: does the answer survive a change of noise model (HYB-01, NSE-03), and does it
# survive removing the upstream parser (PRS-01, exploratory)?
#
# **What this notebook settles.** (1) **HYB-01**: the NB13 pipeline is re-derived identically
# (recipe from `bas02_blocking_frontier`, retrain stated; global isotonic calibrator refit and
# validated against the registered `cal01_calibration_map`); on the entity-disjoint EVAL
# half's union candidate graph, {FS stand-in, embedding-only, each single-rule hybrid,
# all-rules} are compared at the precision@0.99 operating point with paired entity-bootstrap
# deltas on shared resamples — then the flagship **noise-regime map**: the eval slice
# re-corrupted at axis-specific mixes (typo/nickname/dropout/drift-heavy, rates = the measured
# NB04-mapped mix scaled per axis, scaling labeled) × {FS stand-in, embedding, hybrid-all},
# paired entity-F deltas per cell, rendered as a `regime_heatmap` with CI-excludes-zero
# hatching. Registered: `hyb01_rule_value_map` (+ the auxiliary `hyb01_regime_map` the
# heatmap renders from). (2) **NSE-03**: the FS-vs-embedding-vs-hybrid *ranking* across three
# noise models — the calibrated corpus slice, a budget-matched generic-uniform slice (NB05's
# generic baseline params), and REAL NC temporal-drift pairs (same-NCID = positive, sampled
# cross-NCID within county = negative; **pair-level AUC only — no entity structure there**).
# Registered: `nse03_ranking_invariance`. (3) **PRS-01 (EXPLORATORY)**: a 2×2
# {raw single-string via `noise.channels.raw_unparse`, parsed fields} × {FS stand-in,
# embedding} — does standardization level flip the embeddings-vs-rules ranking? Registered:
# `prs01_parsing_factorial`.
#
# **Scope honesty, up front.** One trained encoder, one seed, one corpus draw, one noise
# regeneration per regime: every verdict below says "demonstration" and quotes the MET-04
# replicate bar. Eval discipline is `met07_splits` entity-disjoint throughout (PLAN §5): the
# encoder and calibrator see only the train half; every rule constant (gap, floor, key) is
# pre-registered in the cards, never tuned on the eval half. The NSE-03 drift regime touches
# **real NC person records**: values are casefolded for scoring (stated where it happens),
# every displayed value is **masked** per `DATA_GOVERNANCE.md` (initials+length for names,
# `<present>` for phone-like fields; city/zip stay coarse), and only aggregates are
# registered — no NC person-level value lands in any artifact.

# %%
import itertools
import time

import jellyfish
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from IPython.display import display
from scipy.stats import rankdata

from er_lab.blocking import ann, matchkeys
from er_lab.cluster.schemes import transitive_closure
from er_lab.config import REPO_ROOT, config_hash, load_config_from_env, set_all_seeds
from er_lab.eval.bootstrap import bootstrap_ci, paired_delta
from er_lab.eval.metrics import blocking_metrics
from er_lab.eval.operating_points import find_threshold_for_precision
from er_lab.infra.artifacts import ArtifactRegistry
from er_lab.infra.device import describe_platform
from er_lab.models.encoders import build_encoder
from er_lab.noise.channels import (
    PerRecordChannel,
    field_dropout,
    get_cell,
    hub_value,
    load_lexicon,
    name_order_swap,
    nickname,
    raw_unparse,
    typo,
)
from er_lab.noise.generate import generate_corpus
from er_lab.reporting import figures
from er_lab.reporting.cards import conjecture_card, verdict_box
from er_lab.score.calibrate import fit_calibrator
from er_lab.serialize import serialize_frame
from er_lab.train.loop import train_encoder

cfg = load_config_from_env()
registry = ArtifactRegistry.from_env()
figures.setup_style()
set_all_seeds(cfg.run.seed)
NB_T0 = time.time()
SECTION_TIMES: list[tuple[str, float]] = []


def tick(label: str, t0: float) -> None:
    """Record and print one section's wall-clock — the smoke budget is measured, not vibes."""
    secs = time.time() - t0
    SECTION_TIMES.append((label, secs))
    print(f"[section timing] {label}: {secs:.0f}s")


# %%
# Tier banner — where and under what config this run happened.
print(f"tier        = {cfg.run.tier}")
print(f"config hash = {config_hash(cfg)}")
for key, val in describe_platform().items():
    print(f"  {key:>14}: {val}")

# %% [markdown]
# ## Tier constants
#
# Encoder shape and step budget are the same tier table as notebooks 09/12/13, so the NB12
# recipe transfers verbatim (asserted against the registry below, not assumed). Protocol
# constants (the precision targets, the rule grammar's gap/floor/key) are identical at every
# tier and pre-registered in the cards; analysis budgets (grids, bootstrap replicates, regime
# slice size, NC pair sample, AUC caps) scale with tier — every subsample is stated where it
# happens. Smoke budget for the whole notebook: **<= ~30 min uncontended on the 4-CPU
# container** (a concurrent notebook re-execution can inflate wall-clock 2-4x; per-section
# timings print at the end).

# %%
TIER = str(cfg.run.tier)
SIZES = {"smoke": 3_000, "mid": None, "target": None, "analytical": 3_000}  # train records
STEPS_BY_TIER = {"smoke": 250, "mid": 2000, "target": 4000, "analytical": 250}
MODEL_SHAPE = {
    "smoke": {"dim": 128, "layers": 2, "heads": 4, "max_len": 128},
    "mid": {"dim": 256, "layers": 4, "heads": 4, "max_len": 192},
    "target": {"dim": 384, "layers": 6, "heads": 6, "max_len": 192},
    "analytical": {"dim": 128, "layers": 2, "heads": 4, "max_len": 128},
}[TIER]
STEPS = STEPS_BY_TIER[TIER]
ENC_BATCH = 256  # inference encode batch
K_CAND = 10  # dense candidate budget (bas02's noise-slice k, same as NB13 — matched, not tuned)
GRID = {"smoke": 60, "mid": 150, "target": 200, "analytical": 60}[TIER]
N_BOOT = {"smoke": 500, "mid": 1000, "target": 2000, "analytical": 500}[TIER]
REG_GRID = {"smoke": 40, "mid": 100, "target": 150, "analytical": 40}[TIER]  # per regime cell
REGIME_BASE = {"smoke": 3_500, "mid": 30_000, "target": 100_000, "analytical": 3_500}[TIER]
NC_POS = {"smoke": 4_000, "mid": 30_000, "target": 100_000, "analytical": 4_000}[TIER]
AUC_CAP = {"smoke": 40_000, "mid": 200_000, "target": 500_000, "analytical": 40_000}[TIER]
N_BOOT_AUC = {"smoke": 300, "mid": 600, "target": 1000, "analytical": 300}[TIER]
# protocol constants (PLAN §5 + the cards below — identical at every tier, never tuned):
PREC_TARGETS = (0.99, 0.995)  # fixed B-cubed entity-precision operating points
PREC_PRIMARY = 0.99  # the operating point every rule is priced at (task scope; card)
GUARD_GAP = 2  # guard: birth-year gap > 2 blocks a link
BY_TOL = 1  # birth-year equality tolerance where dob is absent (NC: year := snapshot - age)
P_FLOOR = 0.95  # feature: exact family+birth-key floors the calibrated probability here
CAL_FIT_SHARE = 0.7  # NB13's calibration fit/eval entity share, re-derived identically
MAP_GRID = np.round(np.linspace(-1.0, 1.0, 81), 4)  # cal01's registered cosine grid
AXIS_DOSE = 0.30  # per-duplicate axis dose for the regime map (BEYOND-AUDITED, labeled)
PRIMARY_SEED = int(cfg.run.seed)
REGEN_SEED = PRIMARY_SEED + 401  # one regeneration seed shared by every regime column

cfg.model.kind = "scratch_char"
cfg.model.dim = MODEL_SHAPE["dim"]
cfg.model.layers = MODEL_SHAPE["layers"]
cfg.model.heads = MODEL_SHAPE["heads"]
cfg.model.max_len = MODEL_SHAPE["max_len"]
cfg.train.steps = STEPS
BATCH = int(cfg.train.batch_size)
print(f"tier={TIER}: encoder {MODEL_SHAPE}, steps={STEPS}, batch={BATCH}, train-slice budget "
      f"{SIZES[TIER]}; k={K_CAND}; grids {GRID}/{REG_GRID}, n_boot {N_BOOT}/{N_BOOT_AUC} (auc), "
      f"regime base {REGIME_BASE} records, NC positives {NC_POS}, AUC pair cap {AUC_CAP}")
print(f"protocol: precision {PREC_TARGETS} (rules priced at {PREC_PRIMARY}); rule constants "
      f"gap>{GUARD_GAP}y, floor {P_FLOOR}, birth-year tol ±{BY_TOL} (dob-absent corpora); "
      f"axis dose {AXIS_DOSE}/duplicate (beyond-audited, labeled). Cost-grid operating points "
      "are inherited from bas01/clu01 and not re-swept here (budget; NB17 re-unites them).")

# %% [markdown]
# ## 1. The arena: everything loads through the registry
#
# The DAG-declared upstreams (`bas01_fs_baseline`, `cal01_calibration_map`,
# `clu01_clustering_scores`) plus the artifacts the re-derivation needs: `calibrated_corpus` +
# `met07_splits` (the arena and its discipline), `bas02_blocking_frontier` (the encoder recipe
# NB12→13 locked), `bas01_scored_pairs` (validates the FS stand-in), `met04_power_table` (the
# replicate bar), `calibrated_channel_rates` (the measured NB04-mapped mix the regime map
# scales), and `corpus_registry` (the NC aligned-pair path the drift regime reads).
#
# `TEXT_ROLES` inherits notebook 08's **measured decision**: `full_name` is EXCLUDED — it
# roughly doubles serialized length (forcing truncation at the smoke byte window) and NB05
# leaves it stale under nickname/typo edits, so keeping it would leak the clean name past the
# injected noise. (PRS-01 below is exactly the arm that puts `full_name` back — as the *only*
# name field, in the raw-string regime.)

# %%
t_sec = time.time()
corpus, corpus_meta = registry.load("calibrated_corpus", tier=cfg.run.tier)
splits, _splits_meta = registry.load("met07_splits", tier=cfg.run.tier)
bas02, bas02_meta = registry.load("bas02_blocking_frontier", tier=cfg.run.tier)
bas01, bas01_meta = registry.load("bas01_fs_baseline", tier=cfg.run.tier)
bas01_pairs, bas01_pairs_meta = registry.load("bas01_scored_pairs", tier=cfg.run.tier)
cal01, cal01_meta = registry.load("cal01_calibration_map", tier=cfg.run.tier)
clu01, clu01_meta = registry.load("clu01_clustering_scores", tier=cfg.run.tier)
met04, met04_meta = registry.load("met04_power_table", tier=cfg.run.tier)
rates_tbl, rates_meta = registry.load("calibrated_channel_rates", tier=cfg.run.tier)
creg, creg_meta = registry.load("corpus_registry", tier=cfg.run.tier)

CORPUS_PROVENANCE = (
    f"calibrated_corpus run {corpus_meta['created_at']} (cfg {corpus_meta['config_hash']}, "
    f"tier {corpus_meta['tier']}, seed {corpus_meta['extra']['seed']})"
)
RECIPE = dict(bas02_meta["extra"]["encoder_recipe"])
print(f"corpus: {CORPUS_PROVENANCE}")
print(f"  {len(corpus):,} records / {corpus['entity_id'].nunique():,} entities; "
      f"source={corpus['source'].unique().tolist()} (public historical records; the only NC "
      "content in the SYNTHETIC arena is aggregate channel rates in the corpus meta)")
print(f"recipe source: bas02_blocking_frontier run {bas02_meta['created_at']} "
      f"(cfg {bas02_meta['config_hash']}, tier {bas02_meta['tier']})")
print(f"  NB12 recipe: loss={RECIPE['loss']}, regime={RECIPE['regime']}, "
      f"miner={RECIPE['miner']}, augment={RECIPE['augment']}, steps={RECIPE['steps']} x "
      f"batch {RECIPE['batch']}, model={RECIPE['model']} ({RECIPE['n_params']:,} params)")

if str(bas02_meta["tier"]) == TIER:
    assert int(RECIPE["steps"]) == STEPS and int(RECIPE["batch"]) == BATCH, (
        f"recipe budget {RECIPE['steps']}x{RECIPE['batch']} != tier table {STEPS}x{BATCH}"
    )
    assert dict(RECIPE["model"]) == MODEL_SHAPE, (
        f"recipe shape {RECIPE['model']} != tier table {MODEL_SHAPE}"
    )
cfg.model.kind = "scratch_char" if RECIPE["regime"] == "scratch_char" else "pretrained"

# %%
# NB08's measured field-set decision, inherited (see the section note above).
TEXT_ROLES = ["given_name", "family_name", "dob", "city", "zip", "sex"]
assert TEXT_ROLES == list(RECIPE["text_roles"]), (
    f"TEXT_ROLES {TEXT_ROLES} != the registered recipe's {RECIPE['text_roles']}"
)
KEEP_COLS = ["record_id", "entity_id"] + TEXT_ROLES
print(f"TEXT_ROLES (full_name excluded, NB08 decision) = {TEXT_ROLES}")

SCHEME = "entity_disjoint"
train_ids = set(splits["schemes"][SCHEME]["train"])
eval_ids = set(splits["schemes"][SCHEME]["eval"])
straddle = splits["metadata"]["checks"][SCHEME]["entities_straddling"]
assert straddle == 0, "entity-disjoint split must have zero straddling entities (PLAN §5)"
arena = corpus.reset_index(drop=True)
N_ARENA = len(arena)
train_full = corpus[corpus["record_id"].isin(train_ids)].reset_index(drop=True)
eval_full = corpus[corpus["record_id"].isin(eval_ids)].reset_index(drop=True)
TRUTH_ALL = arena.set_index(arena["record_id"].astype(str))["entity_id"]
ARENA_POS = pd.Series(np.arange(N_ARENA), index=arena["record_id"].astype(str))
print(f"split {SCHEME}: {len(train_full):,} train / {len(eval_full):,} eval records "
      f"({train_full['entity_id'].nunique():,} / {eval_full['entity_id'].nunique():,} "
      "entities, 0 straddling). Encoder + calibrator see the train half only; every rule "
      "constant is pre-registered — nothing in this notebook is tuned on the eval half.")

SER_SCHEME, SER_MISSING = str(cfg.serialize.scheme), str(cfg.serialize.missing)
assert {"scheme": SER_SCHEME, "missing": SER_MISSING} == dict(RECIPE["serialization"]), (
    "serialization drifted from the registered recipe"
)
ARENA_TEXTS = serialize_frame(
    arena, text_roles=TEXT_ROLES, scheme=SER_SCHEME, missing=SER_MISSING
).tolist()
print(f"serialization: scheme={SER_SCHEME} missing={SER_MISSING} (recipe-locked)")

# %%
# The MET-04 caution bar (same read as notebooks 12/13: residual-inclusive from the meta when
# present, else computed and said so) — plus what the two DAG upstreams hand this notebook.
SD_REPLICATE = float(met04_meta["extra"]["sd_replicate"])
_comps = met04[met04["kind"] == "variance_component"].set_index("component")["variance"]
if "detect_bar_residual_inclusive" in met04_meta["extra"]:
    BAR_FULL = float(met04_meta["extra"]["detect_bar_residual_inclusive"])
    BAR_SRC = "read from met04_power_table meta"
else:
    BAR_FULL = float(2.0 * np.sqrt(2.0) * np.sqrt(_comps.sum()))
    BAR_SRC = ("computed here as 2*sqrt(2)*sqrt(seed+noise_draw+residual) — the registered "
               "meta carries only the residual-EXCLUSIVE sd_replicate")
print(f"MET-04 bars (B3F1 units): sd_replicate={SD_REPLICATE:.4f}, residual-inclusive "
      f"single-seed bar {BAR_FULL:.4f} ({BAR_SRC}) — a rule delta beyond the bar is more "
      "than replicate noise; within it, a single-seed run cannot adjudicate.")

_c99 = clu01[(clu01["row_type"] == "op") & (clu01["protocol"] == f"precision@{PREC_PRIMARY}")]
_c_spread = float(_c99["f1"].max() - _c99["f1"].min())
print(f"clu01 ledger read: scheme F1 spread at precision@{PREC_PRIMARY} is {_c_spread:.4f} "
      f"({'<' if _c_spread < BAR_FULL else '>='} the bar) -> scheme choice is second-order "
      "on this corpus; NB14 arbitrates behind TRANSITIVE CLOSURE at protocol thresholds "
      "(the protocol, not the scheme, keeps closure out of the fire — NB13's finding).")
_b99 = bas01[(bas01["system"] == "fs_tuned") & (bas01["protocol"] == f"precision@{PREC_PRIMARY}")]
FS_TUNED_F1 = float(_b99["f1"].iloc[0])
FS_TUNED_R = float(_b99["recall"].iloc[0])
print(f"bas01 context: the TUNED Splink FS baseline scored F1 {FS_TUNED_F1:.4f} "
      f"(recall {FS_TUNED_R:.4f}) at precision@{PREC_PRIMARY} on its matchkey graph — the "
      "stand-in below is validated against bas01's scores, and this number stays on screen "
      "as the honest reference whenever the stand-in looks weak.")
tick("§1 arena + upstreams + bars", t_sec)

# %% [markdown]
# ## 2. Three conjectures, registered before anything runs
#
# Cards up front (PLAN §6): all three are registered now — before the encoder retrains,
# before a single rule fires, before any NC pair is read. Decision rules live inside the
# predictions; the verdict boxes at the end score exactly them.

# %%
t_sec = time.time()
_ = conjecture_card(
    card_id="HYB-01",
    conjecture=(
        "Next to a calibrated whole-record embedding, a deterministic rule earns its place "
        "only where it carries information the embedding provably lacks: the birth-year "
        "GUARD (a structural hard negative the encoder can only infer softly) buys "
        "entity-level quality at the fixed-precision operating point, while the exact-key "
        "OVERRIDE is redundant (the embedding already puts exact-key pairs at the top of "
        "its ranking) and the agreement-FEATURE floor sits between the two — and the map of "
        "who-beats-whom shifts with the noise regime while the hybrid's edge over the "
        "embedding alone does not turn negative anywhere."
    ),
    pressure=(
        "a three-rule grammar applied to FIXED calibrated embedding probabilities on the "
        "entity-disjoint EVAL half's union candidate graph (matchkey passes UNION dense ANN "
        "top-10 — one shared pair universe, so blocking credit is not being re-adjudicated "
        "here; BAS-02 owned that): (a) hard override — casefolded exact "
        "given_name+family_name+birth key forces prob 1 (birth key = full dob string where "
        "dob exists, else birth year within ±1); (b) guard — birth-year gap > 2 forces "
        "prob 0; (c) feature — casefolded exact family_name+birth key floors prob at 0.95; "
        "precedence feature < override < guard (the guard vetoes). Systems: {JW-FS "
        "stand-in, embedding-only, emb+override, emb+guard, emb+feature, hybrid-all}. Then "
        "the regime map: the eval base slice re-corrupted at axis-specific mixes "
        "(typo-heavy / nickname-heavy / dropout-heavy / drift-heavy; rates = the measured "
        "NB04-mapped channel mix with the axis channels scaled to a common 0.30 "
        "per-duplicate dose — BEYOND-AUDITED scaling, labeled, proportions within the axis "
        "preserved; all other channels stay at measured rates; one shared regeneration "
        "seed so the duplicate skeleton matches across columns) x systems {FS stand-in, "
        "embedding, hybrid-all}"
    ),
    property=(
        "rule-fire composition on the union graph, measured before any operating point: "
        "per rule, how many pairs fire, what share of fired pairs are true same-entity "
        "pairs, and — for the override — what share of its fires the embedding-only system "
        "already links at its own precision-0.99 threshold (the redundancy the conjecture "
        "asserts); for the guard, the share of its fires that are true pairs (its "
        "collateral damage, e.g. hub-value 1900-01-01 dobs colliding with real birth years)"
    ),
    metric=(
        "B-cubed entity precision/recall/F1 at the precision@0.99 operating point "
        "(transitive closure per the clu01 ledger; loosest threshold attaining the target, "
        "PLAN §5 highest-attainable fallback flagged loudly) with entity-unit BCa 95% CIs, "
        "PAIRED entity-bootstrap deltas vs embedding-only on shared resamples, and per "
        "regime-map cell the paired delta entity-F1 for {embedding - FS stand-in, "
        "hybrid-all - FS stand-in, hybrid-all - embedding} with ci-excludes-zero "
        "significance shading — registered in hyb01_rule_value_map and hyb01_regime_map"
    ),
    prediction=(
        "P1 (the guard earns its place): at precision@0.99 the guard hybrid's paired "
        "delta B3F1 vs embedding-only is positive with its 95% entity-BCa CI excluding "
        "zero. P2 (the override does not): the override hybrid's paired delta CI includes "
        "zero. P3 (subordinate, cannot flip the outcome): in the regime map, hybrid-all "
        "minus embedding has a nonnegative point estimate in every regime column and is "
        "nowhere negative with a sign-stable CI. Decision rule, pre-registered: CONFIRMED "
        "iff P1 and P2 both hold; REFUTED iff the guard's delta CI excludes zero on the "
        "NEGATIVE side (the guard measurably hurts — e.g. hub-dob collateral) OR the "
        "override's delta CI excludes zero on the positive side (it adds what I claimed "
        "the embedding already has); UNEXPLAINED otherwise — in particular a guard CI "
        "covering zero. Single seed, one corpus draw, one regeneration per regime: a "
        "DEMONSTRATION; HYB-01's definitive home is tier=mid (PLAN §3) with replicate "
        "counts from met04_power_table."
    ),
    registry=registry,
)

# %%
_ = conjecture_card(
    card_id="NSE-03",
    conjecture=(
        "The FS-vs-embedding-vs-hybrid ranking is a property of the systems, not of the "
        "noise model: the same ordering holds on the calibrated-noise corpus, on a "
        "budget-matched generic-uniform-noise corpus, and on real NC temporal-drift pairs "
        "— so conclusions bought on the synthetic corpus transfer to real drift."
    ),
    pressure=(
        "eval noise model x system: noise models {calibrated channels at the NB05 measured "
        "rates (the eval half's union candidate graph, as generated), generic uniform typo "
        "at the NB05-matched total edit budget over given/family/city/zip/dob "
        "(regenerated from the same eval base slice with the same regeneration seed), REAL "
        "NC drift pairs from the corpus_registry aligned snapshots (same-NCID aligned pair "
        "= positive; sampled cross-NCID pairs within the same county = negative, half "
        "uniform and half same-family-name hard negatives, composition registered)}; "
        "systems {JW-FS stand-in, calibrated embedding, hybrid-all with the HYB-01 "
        "grammar (on NC the birth key falls back to snapshot_year - age within ±1, "
        "pre-registered above; dob-keyed exact-date matching cannot fire there)}"
    ),
    property=(
        "pair-level ROC AUC per system per regime — the one metric all three regimes "
        "support. THE NC REGIME IS PAIR-LEVEL ONLY: aligned snapshot pairs carry no entity "
        "structure, so no entity metric, no clustering, and no entity bootstrap exists "
        "there — stated loudly wherever those numbers appear; NC values are casefolded "
        "before scoring (one documented normalization, applied to every system equally) "
        "and masked in every display"
    ),
    metric=(
        "the regime x system AUC table with pair-bootstrap 95% CIs and SHARED-DRAW "
        "adjacent-pair AUC deltas (same pair resamples for all systems within a regime), "
        "the per-regime ranking, and — for the two synthetic regimes only — supporting "
        "entity-F rows at precision@0.99; registered in nse03_ranking_invariance"
    ),
    prediction=(
        "P1: the AUC ordering of {FS stand-in, embedding, hybrid-all} is identical in all "
        "three regimes. P2: within each regime, every adjacent pair in that ordering has a "
        "shared-draw AUC delta whose 95% CI excludes zero. Decision rule, pre-registered: "
        "REFUTED iff any adjacent pair is sign-stable in one regime and sign-stable with "
        "the OPPOSITE sign in another (a real reversal); CONFIRMED iff P1 and P2 both "
        "hold; UNEXPLAINED otherwise (same ordering, fragile deltas). Stated limitation: "
        "pair-resample CIs ignore entity clustering in the two synthetic regimes (the NC "
        "regime is genuinely pair-level); single seed, one draw per regime — a "
        "DEMONSTRATION; NSE-03's definitive home is tier=mid (PLAN §3)."
    ),
    registry=registry,
)

# %%
_ = conjecture_card(
    card_id="PRS-01",
    conjecture=(
        "EXPLORATORY (PLAN §3, cuttable): upstream parsing is worth more to a field-wise "
        "rule system than to a whole-record embedding — collapsing person records to raw "
        "single strings degrades the JW-FS stand-in more than the embedding, and the "
        "embeddings-vs-rules ranking does not flip with standardization level."
    ),
    pressure=(
        "standardization level x system, a 2x2 on the SAME union candidate pairs: "
        "standardization {parsed role fields (the corpus as-is), raw single-string via "
        "noise.channels.raw_unparse applied to the eval half at rate 1.0 (name parts "
        "composed into 'FAMILY, GIVEN M' in full_name and blanked; dob/city/zip/sex "
        "survive — the raw penalty lands on names)}; system {JW-FS stand-in (parsed: "
        "given/family/city fuzzy + dob/zip exact — NB11's validated field set; raw: "
        "full_name/city fuzzy + dob/zip exact), embedding (parsed: the recipe TEXT_ROLES; "
        "raw: full_name replacing the two name parts in the serialization — an eval-time "
        "field-set deviation the raw regime forces, stated)}"
    ),
    property=(
        "pair-level ROC AUC in each of the four cells, on one shared subsample of the "
        "union candidate pairs (record identities survive raw_unparse, so the same pairs "
        "are scored in all four cells)"
    ),
    metric=(
        "the 2x2 AUC table with pair-bootstrap 95% CIs, the parsed->raw AUC drop per "
        "system, and the shared-draw difference-in-differences (FS drop minus embedding "
        "drop); registered in prs01_parsing_factorial"
    ),
    prediction=(
        "P1 (no flip): sign(AUC_embedding - AUC_fs) is the same in the parsed and raw "
        "arms. P2 (parsing is the rule system's subsidy): the FS stand-in's parsed->raw "
        "AUC drop exceeds the embedding's, with the shared-draw CI of the drop difference "
        "excluding zero. Decision rule, pre-registered: REFUTED iff the ranking flips "
        "with both arms' system deltas sign-stable; CONFIRMED iff P1 and P2 both hold; "
        "UNEXPLAINED otherwise. EXPLORATORY: single seed, pair-resample CIs (no entity "
        "unit), one corpus draw — a demonstration, never an adoption gate."
    ),
    registry=registry,
)
tick("§2 cards registered", t_sec)

# %% [markdown]
# ## 3. The encoder, retrained to the registered recipe — budget stated
#
# Identical to notebooks 12/13 by construction: the same entity-complete train slice (same
# subsample seed), the same recipe read from the registry, the same seed into
# `train.loop.train_encoder` — deterministic under seed, so this *is* the NB12/NB13 encoder,
# re-derived rather than re-invented. Budget: exactly the recipe's optimizer-step count at
# the recipe's batch size, asserted from the training history, wall-clock measured.


# %%
def entity_complete_subsample(
    frame: pd.DataFrame, n_records: int | None, rng: np.random.Generator
) -> pd.DataFrame:
    """Random whole entities until ~n_records — an entity is never split across the cut.

    (Same notebook-local helper as notebooks 05/09/12/13 — a shared home in er_lab.data
    would serve; reported as a package gap.)
    """
    if n_records is None or n_records >= len(frame):
        return frame.reset_index(drop=True)
    ents = frame["entity_id"].unique()
    sizes = frame["entity_id"].value_counts()
    picked: list[str] = []
    total = 0
    for i in rng.permutation(len(ents)):
        picked.append(ents[i])
        total += int(sizes[ents[i]])
        if total >= n_records:
            break
    return frame[frame["entity_id"].isin(set(picked))].reset_index(drop=True)


t_sec = time.time()
train_slice = entity_complete_subsample(
    train_full, SIZES[TIER], np.random.default_rng(PRIMARY_SEED)
)[KEEP_COLS]
assert len(train_slice) == int(RECIPE["train_slice"]["records"]) or str(
    bas02_meta["tier"]
) != TIER, "train slice diverged from the registered recipe's — not an identical retrain"
print(f"train slice: {len(train_slice):,} records / {train_slice['entity_id'].nunique():,} "
      f"entities (entity-complete, same construction+seed as NB12/NB13)")

set_all_seeds(PRIMARY_SEED)
encoder = build_encoder(cfg)
N_PARAMS = int(sum(p.numel() for p in encoder.parameters()))
_t0 = time.time()
encoder, hist = train_encoder(
    encoder, train_slice, cfg, loss_name=str(RECIPE["loss"]), miner_name=str(RECIPE["miner"]),
    augment_kind=str(RECIPE["augment"]), steps=STEPS, seed=PRIMARY_SEED,
)
TRAIN_SECS = time.time() - _t0
assert len(hist) == STEPS and int(hist["step"].iloc[-1]) == STEPS, (
    f"budget violation: expected exactly {STEPS} optimizer steps, history shows {len(hist)}"
)
print(f"trained: {RECIPE['loss']}/{RECIPE['miner']}/{RECIPE['augment']}, {STEPS} steps x "
      f"batch {BATCH} (asserted from history), {N_PARAMS:,} params -> {TRAIN_SECS:.0f}s; "
      f"final train loss {float(hist['loss'].tail(20).mean()):.4f}")
tick("§3a recipe retrain", t_sec)

# %%
t_sec = time.time()
_t0 = time.time()
EMB = encoder.encode(ARENA_TEXTS, batch_size=ENC_BATCH)
ENCODE_SECS = time.time() - _t0
ENC_RATE = N_ARENA / ENCODE_SECS
print(f"encoded {N_ARENA:,} records -> ({EMB.shape[0]:,}, {EMB.shape[1]}) in "
      f"{ENCODE_SECS:.0f}s = {ENC_RATE:,.0f} rec/s on this container (rows L2-normalized, "
      "so a pair's dot product IS its cosine)")
tick("§3b encode", t_sec)


# %%
def pair_cosines(a_ids: pd.Series, b_ids: pd.Series) -> np.ndarray:
    """Cosine per pair from the arena embedding matrix (positional lookup)."""
    ia = ARENA_POS.loc[a_ids.astype(str)].to_numpy()
    ib = ARENA_POS.loc[b_ids.astype(str)].to_numpy()
    return np.einsum("ij,ij->i", EMB[ia], EMB[ib]).astype(float)


def flags_to_float(df: pd.DataFrame) -> pd.DataFrame:
    """Boolean flag columns -> float before parquet (concat leaves bool+NaN object cols)."""
    out = df.copy()
    for col in ("attained", "sign_stable", "exceeds_bar", "ci_excludes_zero",
                "fallback_involved"):
        if col in out.columns:
            out[col] = out[col].astype(float)
    return out


# %% [markdown]
# ## 4. The calibrator, re-derived — and validated against the registered map
#
# NB13's CAL-01 construction, repeated verbatim for the one map NB14 consumes (the **global
# isotonic** — the calibrator `clu01` clustered on): the same train-half fit pairs (matchkey
# candidates + dense top-k + uniform negatives, same seeds), the same entity-disjoint
# calibration split within the train half, the same isotonic fit. Then the honesty check: the
# re-derived map is evaluated on `cal01_calibration_map`'s registered cosine grid and compared
# to the registered curve — re-derivation must reproduce the registry, or say loudly that it
# could not.

# %%
t_sec = time.time()
mk_cand = matchkeys.candidates(train_full, passes=matchkeys.default_passes(train_full))
tr_pos = ARENA_POS.loc[train_full["record_id"].astype(str)].to_numpy()
ann_cand = ann.candidates(train_full, EMB[tr_pos], k=K_CAND, index="flat")

_rng_neg = np.random.default_rng(PRIMARY_SEED + 7)
_n_neg = len(mk_cand) + len(ann_cand)
_tr_ids = train_full["record_id"].astype(str).to_numpy()
_ia = _rng_neg.integers(0, len(_tr_ids), size=int(1.2 * _n_neg))
_ib = _rng_neg.integers(0, len(_tr_ids), size=int(1.2 * _n_neg))
_keep = _ia != _ib
rand_pairs = pd.DataFrame({
    "a": np.minimum(_tr_ids[_ia[_keep]], _tr_ids[_ib[_keep]]),
    "b": np.maximum(_tr_ids[_ia[_keep]], _tr_ids[_ib[_keep]]),
}).drop_duplicates().head(_n_neg)

fitpairs = pd.concat(
    [mk_cand.assign(source="matchkeys"), ann_cand.assign(source=f"ann_flat_k{K_CAND}"),
     rand_pairs.assign(source="uniform_random")],
    ignore_index=True,
)
fitpairs["a"], fitpairs["b"] = fitpairs["a"].astype(str), fitpairs["b"].astype(str)
fitpairs = fitpairs.drop_duplicates(subset=["a", "b"], keep="first").reset_index(drop=True)
fitpairs["label"] = (
    TRUTH_ALL.loc[fitpairs["a"]].to_numpy() == TRUTH_ALL.loc[fitpairs["b"]].to_numpy()
).astype(float)
fitpairs["cos"] = pair_cosines(fitpairs["a"], fitpairs["b"])

_tr_ents = pd.Index(train_full["entity_id"].unique())
_rng_split = np.random.default_rng(PRIMARY_SEED + 13)
_fit_ents = set(_tr_ents[_rng_split.permutation(len(_tr_ents))[: int(CAL_FIT_SHARE
                                                                     * len(_tr_ents))]])
_ea = TRUTH_ALL.loc[fitpairs["a"]].to_numpy()
_eb = TRUTH_ALL.loc[fitpairs["b"]].to_numpy()
_in_fit = np.isin(_ea, list(_fit_ents)) & np.isin(_eb, list(_fit_ents))
calfit = fitpairs[_in_fit].reset_index(drop=True)
cal_iso_g = fit_calibrator(calfit["cos"].to_numpy(), calfit["label"].to_numpy(),
                           method="isotonic")

_reg_map = cal01[(cal01["row_type"] == "map") & (cal01["fit"] == "isotonic_global")]
_reg_map = _reg_map.sort_values("score")
_derived = cal_iso_g.transform(_reg_map["score"].to_numpy())
MAP_DIVERGENCE = float(np.max(np.abs(_derived - _reg_map["prob"].to_numpy())))
ECE_ISO_REG = float(cal01[(cal01["row_type"] == "ece") & (cal01["fit"] == "isotonic_global")
                          & (cal01["stratum"] == "all")]["ece"].iloc[0])
print(f"calibration fit pairs: {len(fitpairs):,} "
      f"({dict(fitpairs['source'].value_counts())}); positive rate "
      f"{fitpairs['label'].mean():.3f}; global isotonic fitted on {len(calfit):,} "
      f"entity-disjoint fit-split pairs (share {CAL_FIT_SHARE} of train entities)")
print(f"re-derivation check vs the registered cal01 map ({len(_reg_map)} grid points): "
      f"max |prob difference| = {MAP_DIVERGENCE:.6f}"
      + ("" if MAP_DIVERGENCE <= 0.02 else "  <-- LOUD WARNING: the re-derived map "
         "diverges from the registry beyond 0.02 — treat every calibrated number below "
         "as this run's own, not CAL-01's"))
print(f"registered held-out ECE of this map (cal01, isotonic_global/all): {ECE_ISO_REG:.4f} "
      "— the reliability evidence lives in NB13; NB14 consumes the map, not the claim")
tick("§4 calibrator re-derivation", t_sec)

# %% [markdown]
# ## 5. The FS reference — a validated stand-in, and why
#
# The regime map re-scores five re-corrupted corpora and the PRS factorial two more; a full
# Splink refit per arm is the expensive option (bas01 measured its fit; scoring needs a refit
# per frame in pairs mode). The cheap option is **notebook 11's JW-FS stand-in** — mean
# Jaro-Winkler over `given_name`/`family_name`/`city` plus exact-match indicators on
# `dob`/`zip`, averaged over fields present on BOTH sides — which NB11 already used as its
# FS-shaped teacher. **Chosen: the stand-in (cheaper), validated here** against
# `bas01_scored_pairs` on shared eval-half pairs before it is trusted, exactly NB11's rail.
# The tuned-FS operating point from `bas01` stays quoted alongside as the honest ceiling.


# %%
def jw_standin(recs_a: pd.DataFrame, recs_b: pd.DataFrame,
               fuzzy: tuple = ("given_name", "family_name", "city"),
               exact: tuple = ("dob", "zip")) -> np.ndarray:
    """NB11's FS stand-in: mean JW(fuzzy) + equality(exact) over fields present in BOTH.

    recs_a/recs_b are row-aligned record frames (one row per pair side).
    """
    n = len(recs_a)
    scores = np.zeros(n)
    counts = np.zeros(n)
    for fld in tuple(fuzzy) + tuple(exact):
        if fld not in recs_a.columns or fld not in recs_b.columns:
            continue
        va = recs_a[fld].astype("string").to_numpy()
        vb = recs_b[fld].astype("string").to_numpy()
        ok = ~(pd.isna(va) | pd.isna(vb))
        sim = np.zeros(n)
        if fld in fuzzy:
            sim[ok] = [jellyfish.jaro_winkler_similarity(str(x), str(y))
                       for x, y in zip(va[ok], vb[ok])]
        else:
            sim[ok] = (va[ok] == vb[ok]).astype(float)
        scores += np.where(ok, sim, 0.0)
        counts += ok.astype(float)
    return np.where(counts > 0, scores / np.maximum(counts, 1), 0.0)


t_sec = time.time()
EV_RECS = eval_full.set_index(eval_full["record_id"].astype(str))
_val = bas01_pairs.iloc[np.sort(np.random.default_rng(PRIMARY_SEED + 51).choice(
    len(bas01_pairs), size=min(20_000, len(bas01_pairs)), replace=False))]
_jw_val = jw_standin(EV_RECS.loc[_val["a"].astype(str)].reset_index(drop=True),
                     EV_RECS.loc[_val["b"].astype(str)].reset_index(drop=True))
_rho = float(pd.Series(_jw_val).corr(pd.Series(_val["prob"].to_numpy()), method="spearman"))
print(f"stand-in validation on {len(_val):,} shared eval-half pairs from bas01_scored_pairs: "
      f"spearman rho={_rho:.3f} vs the tuned Splink FS probabilities -> an imperfect but "
      "FS-shaped proxy (NB11's precedent); every 'FS' below means THIS stand-in, with "
      f"bas01's tuned F1 {FS_TUNED_F1:.4f}@{PREC_PRIMARY} quoted as the real FS reference")
tick("§5 FS stand-in validation", t_sec)

# %% [markdown]
# ## 6. The union candidate graph, and the rule grammar's fire pattern
#
# One shared pair universe for every system: **matchkey candidates ∪ dense ANN top-k** on the
# eval half. The union matters — matchkey passes carry exact-key pairs the dense graph
# under-ranks, the dense graph carries fuzzy pairs no key finds; scoring both sets for every
# system means NB14 adjudicates *scores and rules*, never blocking (BAS-02 owned that).
# The grammar is then measured **before** any operating point: how often each rule fires and
# what its fires are worth — including the guard's collateral (hub `1900-01-01` dobs against
# real birth years are exactly where a dob guard back-fires).


# %%
def _norm_str(s: pd.Series) -> pd.Series:
    out = s.astype("string").str.strip().str.casefold()
    return out.mask(out == "")


def birth_year_col(frame: pd.DataFrame) -> pd.Series:
    """Birth cohort per record: year(dob) where dob parses, else snapshot_year - age."""
    by = pd.Series(np.nan, index=frame.index, dtype=float)
    if "dob" in frame.columns:
        y = frame["dob"].astype("string").str.extract(r"^(\d{4})")[0]
        by = pd.to_numeric(y, errors="coerce").astype(float)
    if "age" in frame.columns and "snapshot_year" in frame.columns:
        alt = frame["snapshot_year"].astype(float) - pd.to_numeric(
            frame["age"], errors="coerce").astype(float)
        by = by.fillna(alt)
    return by


def rule_fires(recs_a: pd.DataFrame, recs_b: pd.DataFrame) -> dict[str, np.ndarray]:
    """Boolean fire masks for the three pre-registered rules on row-aligned pair sides.

    Rules normalize by casefold+strip — deterministic rules ARE standardizers, and that
    normalization is part of each rule's pre-registered definition (the corpus values
    themselves are never rewritten; PRS-01 owns standardization as a factor).
    """
    n = len(recs_a)

    def eq(col: str) -> np.ndarray:
        if col not in recs_a.columns or col not in recs_b.columns:
            return np.zeros(n, dtype=bool)
        va, vb = _norm_str(recs_a[col]).to_numpy(), _norm_str(recs_b[col]).to_numpy()
        ok = ~(pd.isna(va) | pd.isna(vb))
        out = np.zeros(n, dtype=bool)
        out[ok] = va[ok] == vb[ok]
        return out

    ya = birth_year_col(recs_a).to_numpy()
    yb = birth_year_col(recs_b).to_numpy()
    gap = np.abs(ya - yb)
    if "dob" in recs_a.columns and "dob" in recs_b.columns:
        key_eq = eq("dob")  # full-date string equality where dob exists
    else:
        key_eq = np.isfinite(gap) & (gap <= BY_TOL)  # dob-absent fallback (card: NC path)
    return {
        "override": eq("given_name") & eq("family_name") & key_eq,
        "guard": np.isfinite(gap) & (gap > GUARD_GAP),
        "feature": eq("family_name") & key_eq,
    }


def apply_rules(prob: np.ndarray, fires: dict[str, np.ndarray], *, use_override: bool,
                use_guard: bool, use_feature: bool) -> np.ndarray:
    """The hybrid grammar: feature floor, then override, then the guard's veto."""
    s = prob.copy()
    if use_feature:
        s = np.where(fires["feature"], np.maximum(s, P_FLOOR), s)
    if use_override:
        s = np.where(fires["override"], 1.0, s)
    if use_guard:
        s = np.where(fires["guard"], 0.0, s)
    return s


# The AUC representation, declared once: the isotonic map is only WEAKLY monotone
# (piecewise-constant), and AUC is invariant only under STRICTLY monotone maps — pushing
# cosines through it collapses distinct cosines onto shared steps, and that tie collapse
# measurably shifts AUC. So every embedding-based AUC arm in this notebook (NSE-03, PRS-01)
# scores the RAW cosine (rank-native, tie-free); the hybrid AUC arm applies the same
# pre-registered grammar rank-natively on the cosine scale. Entity operating points keep
# the calibrated probabilities — that is the system CLU-01 adjudicated, and the prob-scale
# floor is part of its grammar.
_floor_grid = np.linspace(-1.0, 1.0, 4001)
_floor_probs = cal_iso_g.transform(_floor_grid)
FLOOR_COS = (float(_floor_grid[np.argmax(_floor_probs >= P_FLOOR)])
             if bool((_floor_probs >= P_FLOOR).any()) else 1.0)


def apply_rules_ranknative(cos: np.ndarray, fires: dict[str, np.ndarray]) -> np.ndarray:
    """The same grammar on the RAW cosine scale, for AUC arms only: the feature floors at
    the prob-floor's cosine preimage (FLOOR_COS), the override sits strictly above the
    cosine range, the guard strictly below it (precedence unchanged: the guard vetoes)."""
    s = cos.copy()
    s = np.where(fires["feature"], np.maximum(s, FLOOR_COS), s)
    s = np.where(fires["override"], 2.0, s)
    s = np.where(fires["guard"], -2.0, s)
    return s


# %%
t_sec = time.time()
ev_pos = ARENA_POS.loc[eval_full["record_id"].astype(str)].to_numpy()
mk_ev = matchkeys.candidates(eval_full, passes=matchkeys.default_passes(eval_full))
ann_ev = ann.candidates(eval_full, EMB[ev_pos], k=K_CAND, index="flat")
upairs = pd.concat([mk_ev.assign(src="matchkeys"), ann_ev.assign(src="ann")],
                   ignore_index=True)
upairs["a"], upairs["b"] = upairs["a"].astype(str), upairs["b"].astype(str)
upairs = upairs.drop_duplicates(subset=["a", "b"], keep="first").reset_index(drop=True)
truth_ev = eval_full.set_index(eval_full["record_id"].astype(str))["entity_id"]
records_ev = pd.Index(truth_ev.index)
N_EV, N_ENT_EV = len(records_ev), int(truth_ev.nunique())

UP_TRUE = (truth_ev.loc[upairs["a"]].to_numpy() == truth_ev.loc[upairs["b"]].to_numpy())
bm_u = blocking_metrics(upairs[["a", "b"]], truth_ev, n_records=N_EV)
A_EV = EV_RECS.loc[upairs["a"]].reset_index(drop=True)
B_EV = EV_RECS.loc[upairs["b"]].reset_index(drop=True)
UP_COS = pair_cosines(upairs["a"], upairs["b"])
UP_PROB = cal_iso_g.transform(UP_COS)
UP_JW = jw_standin(A_EV, B_EV)
FIRES = rule_fires(A_EV, B_EV)

print(f"union graph on the {SCHEME} eval half ({N_EV:,} records / {N_ENT_EV:,} entities): "
      f"{len(mk_ev):,} matchkey + {len(ann_ev):,} ann_flat k={K_CAND} -> {len(upairs):,} "
      f"unique pairs; pair completeness {bm_u['pair_completeness']:.4f} of "
      f"{int(bm_u['n_true_pairs']):,} true pairs (matchkey-only was "
      f"{float(bas01_meta['extra']['blocking']['metrics']['pair_completeness']):.4f}, "
      f"dense-only was {float(clu01_meta['extra']['candidate_graph']['pair_completeness']):.4f}"
      " — the union is the fairest arena either system gets at this budget)")
N_ISO_STEPS = int(pd.Series(UP_PROB).nunique())
print(f"edge precision at t=0: {UP_TRUE.mean():.4f}; calibrated prob range "
      f"{UP_PROB.min():.3f}..{UP_PROB.max():.3f} ({N_ISO_STEPS} distinct isotonic steps "
      f"over {len(upairs):,} pairs — the map is weakly monotone / many-to-one; prob-floor "
      f"{P_FLOOR} cosine preimage {FLOOR_COS:.4f})")
for rule in ("override", "guard", "feature"):
    f = FIRES[rule]
    if f.sum():
        print(f"  rule '{rule}': fires on {int(f.sum()):,} pairs "
              f"({f.mean():.2%} of graph); {UP_TRUE[f].mean():.4f} of fires are TRUE pairs"
              + (f"; guard collateral: {int((f & UP_TRUE).sum()):,} true pairs blocked"
                 if rule == "guard" else ""))
    else:
        print(f"  rule '{rule}': fires on 0 pairs")
_hub_dob = (A_EV["dob"].astype("string").str.startswith("1900-01-01").fillna(False)
            | B_EV["dob"].astype("string").str.startswith("1900-01-01").fillna(False))
print(f"  hub-dob presence on the graph: {int(_hub_dob.sum()):,} pairs touch a 1900-01-01 "
      f"dob; of those, the guard fires on {float(FIRES['guard'][_hub_dob].mean()):.2%} "
      "(the pre-registered collateral mechanism, measured)")
tick("§6 union graph + rule fires", t_sec)

# %% [markdown]
# ## 7. Six systems at the protocol operating point
#
# Every system is a score vector over the same pairs; every system is thresholded by the
# same protocol (loosest threshold attaining B-cubed entity precision 0.99 under transitive
# closure, loud fallback if unattainable); every CI is an entity-unit BCa bootstrap; every
# rule's marginal value is a PAIRED delta vs embedding-only on shared entity resamples. The
# override-redundancy number the card's property clause demands is computed here, after the
# embedding's own threshold is known.

# %%
t_sec = time.time()
SYSTEMS = {
    "fs_standin": UP_JW,
    "embedding": UP_PROB,
    "emb+override": apply_rules(UP_PROB, FIRES, use_override=True, use_guard=False,
                                use_feature=False),
    "emb+guard": apply_rules(UP_PROB, FIRES, use_override=False, use_guard=True,
                             use_feature=False),
    "emb+feature": apply_rules(UP_PROB, FIRES, use_override=False, use_guard=False,
                               use_feature=True),
    "hybrid_all": apply_rules(UP_PROB, FIRES, use_override=True, use_guard=True,
                              use_feature=True),
}
SYS_ROLE = {"fs_standin": "reference", "embedding": "none", "emb+override": "override",
            "emb+guard": "guard", "emb+feature": "feature", "hybrid_all": "all"}
SYS_RULE = {"emb+override": "exact_key_override", "emb+guard": "dob_year_guard",
            "emb+feature": "family_dob_floor", "hybrid_all": "all_rules"}

PRED_CACHE: dict[tuple[str, float], pd.Series] = {}
PRED_SECS: dict[tuple[str, float], float] = {}
_SFRAMES = {name: pd.DataFrame({"a": upairs["a"], "b": upairs["b"], "score": s, "prob": s})
            for name, s in SYSTEMS.items()}


def pred_at(system: str, thr: float) -> pd.Series:
    key = (system, float(thr))
    if key not in PRED_CACHE:
        t0 = time.time()
        PRED_CACHE[key] = transitive_closure(_SFRAMES[system], threshold=float(thr),
                                             records=records_ev)
        PRED_SECS[key] = time.time() - t0
    return PRED_CACHE[key]


CI_METRICS = (("precision", "bcubed_precision"), ("recall", "bcubed_recall"),
              ("f1", "bcubed_f1"))
op_rows: list[dict] = []
T_AT: dict[tuple[str, float], float] = {}
for system in SYSTEMS:
    for target in PREC_TARGETS:
        res = find_threshold_for_precision(
            _SFRAMES[system][["a", "b", "score"]], lambda t, s=system: pred_at(s, t),
            truth_ev, target=target, grid=GRID)
        T_AT[(system, target)] = float(res["threshold"])
        row = {"row_type": "op", "system": system, "role": SYS_ROLE[system],
               "rule": SYS_RULE.get(system, ""), "protocol": f"precision@{target}",
               "threshold": float(res["threshold"]), "attained": bool(res["attained"]),
               "fallback": res["fallback"] or "", "basis": "MEASURED"}
        pred = pred_at(system, res["threshold"])
        for short, metric in CI_METRICS:
            ci = bootstrap_ci(pred, truth_ev, metric, unit="entity", n_boot=N_BOOT,
                              seed=PRIMARY_SEED)
            row[short] = ci["point"]
            row[f"{short}_lo"], row[f"{short}_hi"] = ci["ci_low"], ci["ci_high"]
        op_rows.append(row)
        flag = "" if res["attained"] else (
            f"  <-- precision {target} UNATTAINABLE (max "
            f"{res['attained_precision']:.4f}) — PLAN §5 rail: loud, never silent")
        print(f"[{system}] precision@{target}: t={res['threshold']:.4f} "
              f"P={res['attained_precision']:.4f} R={res['recall_at']:.4f}{flag}")
ops = pd.DataFrame(op_rows)
display(ops[ops["protocol"] == f"precision@{PREC_PRIMARY}"]
        [["system", "role", "threshold", "attained", "precision", "recall", "f1",
          "f1_lo", "f1_hi"]].round(4))
tick("§7a operating points + CIs", t_sec)

# %%
# The card's property clause: is the override redundant BY the embedding's own ranking?
t_sec = time.time()
_t99_emb = T_AT[("embedding", PREC_PRIMARY)]
_ov = FIRES["override"]
OV_ALREADY = float((SYSTEMS["embedding"][_ov] >= _t99_emb).mean()) if _ov.sum() else np.nan
print(f"override redundancy: {OV_ALREADY:.2%} of the {int(_ov.sum()):,} override fires are "
      f"ALREADY at/above the embedding-only precision@{PREC_PRIMARY} threshold "
      f"({_t99_emb:.4f}) — the share of the override's work the embedding does anyway")
print(f"feature-floor inertness check: the pre-registered floor {P_FLOOR} sits "
      f"{'BELOW' if P_FLOOR < _t99_emb else 'above'} the embedding's own "
      f"precision@{PREC_PRIMARY} threshold {_t99_emb:.4f} — a floor below the operating "
      "threshold cannot flip any link decision there, so emb+feature can only differ from "
      "embedding-only through threshold re-selection (read its delta with that in mind)")

delta_rows: list[dict] = []
_emb_pred = pred_at("embedding", T_AT[("embedding", PREC_PRIMARY)])
for system in ("fs_standin", "emb+override", "emb+guard", "emb+feature", "hybrid_all"):
    res = paired_delta(pred_at(system, T_AT[(system, PREC_PRIMARY)]), _emb_pred, truth_ev,
                       "bcubed_f1", unit="entity", n_boot=N_BOOT, seed=PRIMARY_SEED)
    res_r = paired_delta(pred_at(system, T_AT[(system, PREC_PRIMARY)]), _emb_pred, truth_ev,
                         "bcubed_recall", unit="entity", n_boot=N_BOOT, seed=PRIMARY_SEED)
    delta_rows.append({
        "row_type": "delta", "system": system, "role": SYS_ROLE[system],
        "rule": SYS_RULE.get(system, ""), "protocol": f"precision@{PREC_PRIMARY}",
        "vs": "embedding", "delta_f1": res["delta"], "delta_lo": res["ci_low"],
        "delta_hi": res["ci_high"], "sign_stable": bool(res["sign_stable"]),
        "exceeds_bar": bool(abs(res["delta"]) >= BAR_FULL),
        "delta_recall": res_r["delta"], "delta_recall_lo": res_r["ci_low"],
        "delta_recall_hi": res_r["ci_high"], "basis": "MEASURED",
    })
deltas = pd.DataFrame(delta_rows)
print(f"\npaired deltas vs embedding-only at precision@{PREC_PRIMARY} "
      f"(shared entity resamples, {N_BOOT} replicates; MET-04 bar {BAR_FULL:.4f}):")
display(deltas[["system", "role", "delta_f1", "delta_lo", "delta_hi", "sign_stable",
                "exceeds_bar", "delta_recall"]].round(4))
tick("§7b paired rule deltas", t_sec)

# %% [markdown]
# ## 8. The noise-regime map — the flagship
#
# Which of these conclusions is a property of THIS dirt? The eval half's base records (native
# variants, no generated duplicates) are re-corrupted under **axis-specific mixes**: the
# measured NB04-mapped channel mix with one axis's channels scaled to a common per-duplicate
# dose, everything else held at measured rates, one shared regeneration seed so every regime
# column shares the same duplicate skeleton. The measured rates are 1e-4..1e-2 — the axis
# dose is **deliberately beyond the audited range** (scale factors printed and labeled: the
# audit bounds entry-error rates only from below, and the map's question is *ranking under
# stress*, not realism). Custom drift channels (pool redraw, name format drift) are NB05's
# notebook-local implementations, replicated here — the same package gap NB05 reported.


# %%
# NB05's custom channels, notebook-local (package gap, noted in the close): pool redraw for
# move/name_change, skeleton-preserving name format drift. full_name sync is omitted —
# full_name is not serialized (TEXT_ROLES) and regime slices are scored, never re-audited.
def pool_redraw_channel(name: str, fields: tuple, pool: list) -> PerRecordChannel:
    """Wholesale redraw of ``fields`` (jointly) from a pool of real value tuples."""
    fields = tuple(fields)
    pool = list(pool)

    def edit(out: pd.DataFrame, pos: int, rng: np.random.Generator) -> list:
        cur = tuple(get_cell(out, pos, f) for f in fields)
        if all(v is None for v in cur) or not pool:
            return []
        pick = None
        for _ in range(8):
            cand = pool[int(rng.integers(len(pool)))]
            if cand != cur:
                pick = cand
                break
        if pick is None:
            return []
        return [(f, out[f].iloc[pos], new) for f, old, new in zip(fields, cur, pick)
                if old != new]

    return PerRecordChannel(name=name, edit_fn=edit)


def name_format_drift_channel(field: str) -> PerRecordChannel:
    """Case/punctuation drift preserving the alphanumeric skeleton (NB04's format_drift)."""

    def edit(out: pd.DataFrame, pos: int, rng: np.random.Generator) -> list:
        v = get_cell(out, pos, field)
        if v is None:
            return []
        variants = []
        if v.upper() != v:
            variants.append(v.upper())
        cap = v[:1].upper() + v[1:]
        if cap not in (v, v.upper()):
            variants.append(cap)
        if " " in v:
            variants.append(v.replace(" ", "-"))
        if "-" in v:
            variants.append(v.replace("-", " "))
        if "'" in v:
            variants.append(v.replace("'", ""))
        if not variants:
            return []
        new = variants[int(rng.integers(len(variants)))]
        return [(field, v, new)] if new != v else []

    return PerRecordChannel(name="name_format_drift", edit_fn=edit)


def value_pool(df: pd.DataFrame, fields: tuple) -> list:
    """Sorted unique tuples of ``fields`` over rows where every field is present."""
    mask = np.ones(len(df), dtype=bool)
    for f in fields:
        s = df[f].astype("string")
        mask &= (s.notna() & (s.str.strip() != "")).to_numpy()
    sub = df.loc[mask, list(fields)].astype(str)
    return sorted(set(map(tuple, sub.itertuples(index=False, name=None))))


# %%
t_sec = time.time()
eval_base = eval_full[~eval_full["record_id"].str.contains("#dup")].reset_index(drop=True)
reg_base = entity_complete_subsample(
    eval_base, REGIME_BASE, np.random.default_rng(REGEN_SEED))[KEEP_COLS]
print(f"regime base: {len(reg_base):,} of {len(eval_base):,} eval-half base records "
      f"({reg_base['entity_id'].nunique():,} whole entities; entity-complete subsample, "
      f"seed {REGEN_SEED}, STATED: the regime map runs on this slice, not the full half)")

try:
    _lex_raw = load_lexicon(str(REPO_ROOT / "data"))
    LEXICON = {c: {v for v in vs if v not in ("has_nickname", "relationship")}
               for c, vs in _lex_raw.items() if c != "name1"}
    LEXICON = {c: v for c, v in LEXICON.items() if v}
    _lex_note = (f"carltonnorthern lexicon: {len(LEXICON)} canonical names, "
                 f"{sum(len(v) for v in LEXICON.values())} links (NB05's sanitize pass kept)")
except FileNotFoundError:
    LEXICON = None
    _lex_note = "lexicon file missing -> nickname() built-in fallback (LOUD: tiny lexicon)"
print(_lex_note + " — English-centric provenance bias carried on every nickname claim")

pool_cityzip = value_pool(reg_base, ("city", "zip"))
pool_zip = value_pool(reg_base, ("zip",))
pool_family = value_pool(reg_base, ("family_name",))
pool_given = value_pool(reg_base, ("given_name",))
REG_CHANNELS = {
    "move": pool_redraw_channel("move", ("city", "zip"), pool_cityzip),
    "move_zip_only": pool_redraw_channel("move", ("zip",), pool_zip),
    "name_change_family": pool_redraw_channel("name_change", ("family_name",), pool_family),
    "name_change_given": pool_redraw_channel("name_change", ("given_name",), pool_given),
    "name_format_drift_family": name_format_drift_channel("family_name"),
    "name_format_drift_given": name_format_drift_channel("given_name"),
    "typo_family": typo(fields=("family_name",)),
    "typo_given": typo(fields=("given_name",)),
    "nickname": nickname(lexicon=LEXICON),
    "name_order_swap": name_order_swap(),
    "dropout_given": field_dropout(fields=("given_name",)),
    "hub_value": hub_value(),
}
CH_RATE = {k: float(v) for k, v in corpus_meta["extra"]["channel_rates"].items()}
AXES = {
    "typo-heavy": ("typo_given", "typo_family"),
    "nickname-heavy": ("nickname",),
    "dropout-heavy": ("dropout_given",),
    "drift-heavy": ("move", "move_zip_only", "name_change_family", "name_change_given",
                    "name_format_drift_family", "name_format_drift_given"),
}
REGIME_MIXES: dict[str, dict[str, float]] = {"calibrated": dict(CH_RATE)}
print(f"\naxis mixes (axis channels scaled to a common {AXIS_DOSE} per-duplicate dose, "
      "within-axis proportions preserved, non-axis channels at measured rates):")
for axis, members in AXES.items():
    base_rate = sum(CH_RATE[m] for m in members)
    factor = AXIS_DOSE / base_rate
    mix = dict(CH_RATE)
    for m in members:
        mix[m] = CH_RATE[m] * factor
    REGIME_MIXES[axis] = mix
    print(f"  {axis:>15}: measured axis total {base_rate:.6f} x {factor:,.0f} "
          f"-> {AXIS_DOSE}  [BEYOND-AUDITED scaling — the NB04 audit bounds these rates "
          "from below; this is a stress dose, not a prevalence claim]")
print("  calibrated column: the measured rates verbatim "
      f"(provenance: {corpus_meta['extra']['channel_rate_provenance'][:80]}...)")
tick("§8a regime machinery", t_sec)

# %%
t_sec = time.time()
HEAT_SYSTEMS = ("fs_standin", "embedding", "hybrid_all")
HEAT_COMPARISONS = (("embedding - FS", "embedding", "fs_standin"),
                    ("hybrid - FS", "hybrid_all", "fs_standin"),
                    ("hybrid - embedding", "hybrid_all", "embedding"))
cell_rows: list[dict] = []
regime_op_rows: list[dict] = []
REGIME_SECS: dict[str, dict[str, float]] = {}
REGIME_STORE: dict[str, dict] = {}  # scores + labels per regime, reused by NSE-03

for regime, mix in REGIME_MIXES.items():
    secs: dict[str, float] = {}
    _t0 = time.time()
    with np.errstate(divide="ignore"):  # benign gecko keymap divide (NB05's note)
        slice_r, ops_r = generate_corpus(
            reg_base, channel_rates=mix, channels=REG_CHANNELS,
            keep_original_rate=1.0, seed=REGEN_SEED,
        )
    secs["generate"] = time.time() - _t0
    truth_r = slice_r.set_index(slice_r["record_id"].astype(str))["entity_id"]
    records_r = pd.Index(truth_r.index)
    _t0 = time.time()
    texts_r = serialize_frame(slice_r, text_roles=TEXT_ROLES, scheme=SER_SCHEME,
                              missing=SER_MISSING).tolist()
    emb_r = encoder.encode(texts_r, batch_size=ENC_BATCH)
    secs["encode"] = time.time() - _t0
    _t0 = time.time()
    mk_r = matchkeys.candidates(slice_r, passes=matchkeys.default_passes(slice_r))
    ann_r = ann.candidates(slice_r, emb_r, k=K_CAND, index="flat")
    pr = pd.concat([mk_r, ann_r], ignore_index=True)
    pr["a"], pr["b"] = pr["a"].astype(str), pr["b"].astype(str)
    pr = pr.drop_duplicates(subset=["a", "b"]).reset_index(drop=True)
    secs["candidates"] = time.time() - _t0
    pos_r = pd.Series(np.arange(len(slice_r)), index=slice_r["record_id"].astype(str))
    recs_r = slice_r.set_index(slice_r["record_id"].astype(str))
    ar = recs_r.loc[pr["a"]].reset_index(drop=True)
    br = recs_r.loc[pr["b"]].reset_index(drop=True)
    cos_r = np.einsum("ij,ij->i", emb_r[pos_r.loc[pr["a"]].to_numpy()],
                      emb_r[pos_r.loc[pr["b"]].to_numpy()]).astype(float)
    fires_r = rule_fires(ar, br)
    scores_r = {
        "fs_standin": jw_standin(ar, br),
        "embedding": cal_iso_g.transform(cos_r),
    }
    scores_r["hybrid_all"] = apply_rules(scores_r["embedding"], fires_r, use_override=True,
                                         use_guard=True, use_feature=True)
    true_r = (truth_r.loc[pr["a"]].to_numpy() == truth_r.loc[pr["b"]].to_numpy())
    REGIME_STORE[regime] = {"scores": scores_r, "labels": true_r, "n_records": len(slice_r),
                            "n_pairs": len(pr), "ops": ops_r}

    _t0 = time.time()
    preds_r: dict[str, pd.Series] = {}
    attained_r: dict[str, bool] = {}
    for system, sc in scores_r.items():
        sf = pd.DataFrame({"a": pr["a"], "b": pr["b"], "score": sc, "prob": sc})
        cache_r: dict[float, pd.Series] = {}

        def clus(t: float, frame=sf, cache=cache_r, recs=records_r) -> pd.Series:
            t = float(t)
            if t not in cache:
                cache[t] = transitive_closure(frame, threshold=t, records=recs)
            return cache[t]

        res = find_threshold_for_precision(sf[["a", "b", "score"]], clus, truth_r,
                                           target=PREC_PRIMARY, grid=REG_GRID)
        preds_r[system] = clus(res["threshold"])
        attained_r[system] = bool(res["attained"])
        ci = bootstrap_ci(preds_r[system], truth_r, "bcubed_f1", unit="entity",
                          n_boot=N_BOOT, seed=PRIMARY_SEED)
        regime_op_rows.append({
            "row_type": "regime_op", "regime": regime, "system": system,
            "threshold": float(res["threshold"]), "attained": bool(res["attained"]),
            "fallback": res["fallback"] or "", "precision": res["attained_precision"],
            "recall": res["recall_at"], "f1": ci["point"], "f1_lo": ci["ci_low"],
            "f1_hi": ci["ci_high"], "n_records": len(slice_r), "n_pairs": len(pr),
            "basis": "MEASURED",
        })
    for label, sys_a, sys_b in HEAT_COMPARISONS:
        d = paired_delta(preds_r[sys_a], preds_r[sys_b], truth_r, "bcubed_f1",
                         unit="entity", n_boot=N_BOOT, seed=PRIMARY_SEED)
        cell_rows.append({
            "row": label, "col": regime, "value": d["delta"], "lo": d["ci_low"],
            "hi": d["ci_high"], "ci_excludes_zero": bool(d["sign_stable"]),
            "exceeds_bar": bool(abs(d["delta"]) >= BAR_FULL),
            "fallback_involved": bool(not attained_r[sys_a] or not attained_r[sys_b]),
            "basis": "MEASURED",
        })
    secs["ops_and_deltas"] = time.time() - _t0
    REGIME_SECS[regime] = secs
    _fb = [r for r in regime_op_rows if r["regime"] == regime and not r["attained"]]
    print(f"[{regime}] {len(slice_r):,} records, {len(ops_r):,} channel ops, {len(pr):,} "
          f"pairs; secs {({k: round(v, 1) for k, v in secs.items()})}"
          + (f"; FALLBACK systems (precision {PREC_PRIMARY} unattainable): "
             f"{[r['system'] for r in _fb]}" if _fb else ""))

regime_ops = pd.DataFrame(regime_op_rows)
regime_cells = pd.DataFrame(cell_rows)
display(regime_ops[["regime", "system", "attained", "precision", "recall", "f1", "f1_lo",
                    "f1_hi"]].round(4))
display(regime_cells[["row", "col", "value", "lo", "hi", "ci_excludes_zero"]].round(4))
tick("§8b regime map computed", t_sec)

# %%
t_sec = time.time()
hyb01 = pd.concat([ops, deltas,
                   regime_ops.assign(protocol=f"precision@{PREC_PRIMARY}"),
                   regime_cells.assign(row_type="regime_delta")], ignore_index=True)
_fallback_cells = regime_ops.loc[~regime_ops["attained"].astype(bool),
                                 ["regime", "system"]]
registry.register(
    "hyb01_rule_value_map", flags_to_float(hyb01), cfg=cfg, tier=cfg.run.tier,
    meta={
        "card": "HYB-01",
        "row_types": {
            "op": "system x fixed-precision operating point on the eval-half union graph: "
                  "threshold, B-cubed point + entity-BCa 95% CI, attained/fallback rail",
            "delta": "PAIRED entity-bootstrap delta (shared resamples) of each rule system "
                     "vs embedding-only at precision@0.99; exceeds_bar = |delta| >= the "
                     "MET-04 residual-inclusive bar",
            "regime_op": "system x re-corrupted regime at precision@0.99 (regime slice; "
                         "fallback flag = PLAN §5 rail)",
            "regime_delta": "the regime map cells: paired delta entity-F1 per "
                            "(comparison row, regime col), ci_excludes_zero = hatching; "
                            "fallback_involved = 1.0 where either compared system sat at "
                            "a PLAN §5 fallback operating point in that regime (the ○ "
                            "marker on the heatmap)",
        },
        "grammar": {
            "override": "casefolded exact given+family+birth key -> prob 1 (birth key: "
                        "full dob string where dob exists, else birth year within "
                        f"±{BY_TOL})",
            "guard": f"birth-year gap > {GUARD_GAP} -> prob 0 (precedence: the guard "
                     "vetoes)",
            "feature": f"casefolded exact family+birth key -> prob floored at {P_FLOOR}",
            "constants_pre_registered": "card HYB-01 (never tuned on the eval half)",
            "override_redundancy_share": OV_ALREADY,
        },
        "arena": {
            "union_graph": f"matchkeys ∪ ann_flat k={K_CAND} on the {SCHEME} eval half "
                           f"({N_EV} records / {N_ENT_EV} entities), "
                           f"{len(upairs)} pairs, PC {float(bm_u['pair_completeness']):.4f}",
            "scores": "JW-FS stand-in (validated vs bas01_scored_pairs, spearman "
                      f"{_rho:.3f}) | calibrated embedding (re-derived NB13 pipeline; "
                      f"map divergence vs registered cal01: {MAP_DIVERGENCE:.6f})",
            "clustering": "transitive closure (clu01 ledger: scheme second-order at fixed "
                          "precision)",
            "fs_note": f"tuned Splink FS (bas01) scored F1 {FS_TUNED_F1:.4f}@"
                       f"{PREC_PRIMARY} on its matchkey graph — the stand-in is the "
                       "regime-portable proxy, not the ceiling",
        },
        "regime_map": {
            "base": f"{len(reg_base)} eval-half base records (entity-complete subsample, "
                    f"seed {REGEN_SEED}); one regeneration seed across columns (shared "
                    "duplicate skeleton)",
            "axis_dose": AXIS_DOSE,
            "axis_channels": {k: list(v) for k, v in AXES.items()},
            "scaling_label": "BEYOND-AUDITED: measured NB04-mapped rates scaled per axis "
                             "to the common dose; a stress test of ranking, not a "
                             "prevalence claim",
            "fallback_cells": _fallback_cells.to_dict("records"),
        },
        "protocol": {"precision_targets": list(PREC_TARGETS), "primary": PREC_PRIMARY,
                     "grid": GRID, "regime_grid": REG_GRID,
                     "ci": {"unit": "entity", "method": "bca", "n_boot": N_BOOT,
                            "seed": PRIMARY_SEED},
                     "cost_grid_note": "cost-grid points inherited from bas01/clu01, not "
                                       "re-swept here (NB17 re-unites them)"},
        "met04": {"sd_replicate": SD_REPLICATE, "residual_inclusive_bar": BAR_FULL,
                  "bar_source": BAR_SRC, "units": "B3F1"},
        "single_seed_caveat": "one encoder, one seed, one corpus draw, one regeneration "
                              "per regime — a DEMONSTRATION; HYB-01's definitive home is "
                              "tier=mid (PLAN §3)",
        "corpus_provenance": CORPUS_PROVENANCE,
    },
)
registry.register(
    "hyb01_regime_map", flags_to_float(regime_cells), cfg=cfg, tier=cfg.run.tier,
    meta={
        "card": "HYB-01",
        "note": "auxiliary long-form view of hyb01_rule_value_map's regime_delta rows — "
                "the (row, col, value, ci_excludes_zero) contract "
                "reporting.figures.regime_heatmap renders, plus a fallback_involved "
                "column (1.0 = either compared system sat at a PLAN §5 fallback "
                "operating point in that regime — rendered as the ○ marker so the "
                "unmatched operating point is visible ON the figure); value = paired "
                f"delta entity-B3F1 at precision@{PREC_PRIMARY} (shared entity resamples)",
        "fallback_cells": _fallback_cells.to_dict("records"),
        "single_seed_caveat": "single seed, one regeneration per regime — a DEMONSTRATION",
    },
)
print(f"registered hyb01_rule_value_map: {len(ops)} op + {len(deltas)} delta + "
      f"{len(regime_ops)} regime_op + {len(regime_cells)} regime_delta rows; "
      f"hyb01_regime_map: {len(regime_cells)} cells")
tick("§8c register", t_sec)


# %%
def draw_rule_ledger(ax, df, meta):
    d = df[(df["row_type"] == "op") & (df["protocol"] == f"precision@{PREC_PRIMARY}")]
    order = [s for s in SYSTEMS if s in set(d["system"])]
    xs = {s: i for i, s in enumerate(order)}
    x = d["system"].map(xs).to_numpy(dtype=float)
    yerr = np.vstack([np.clip(d["f1"] - d["f1_lo"], 0, None),
                      np.clip(d["f1_hi"] - d["f1"], 0, None)])
    ax.errorbar(x, d["f1"], yerr=yerr, fmt="o", capsize=3, markersize=6, color="C0")
    for xi, (_, r) in zip(x, d.iterrows()):
        if not r["attained"]:
            ax.annotate("fallback", (xi, r["f1"]), textcoords="offset points",
                        xytext=(0, -14), ha="center", fontsize=6.5, color="C3")
    y0 = float(d["f1"].min())
    ax.plot([len(order) - 0.6] * 2, [y0, y0 + BAR_FULL], color="0.25", linewidth=2.2)
    ax.annotate(f"MET-04 residual-\ninclusive bar {BAR_FULL:.3f}",
                (len(order) - 0.55, y0 + BAR_FULL / 2), fontsize=7.5, color="0.25",
                va="center")
    ax.axhline(FS_TUNED_F1, linestyle=":", color="C3", linewidth=1.0)
    ax.text(0.02, FS_TUNED_F1, "tuned Splink FS (bas01, matchkey graph)", color="C3",
            fontsize=7.5, va="bottom", transform=ax.get_yaxis_transform())
    ax.set_xticks(range(len(order)), order, rotation=12)
    ax.set_ylabel(f"B-cubed F1 at precision@{PREC_PRIMARY} (entity-BCa 95% CI)")


fig = figures.plot_artifact(
    registry, tier=cfg.run.tier, artifact="hyb01_rule_value_map", draw=draw_rule_ledger,
    title="What each rule is worth next to the embedding, at the protocol operating point",
    figsize=(7.2, 4.6),
)

# %%
# The heatmap itself must show where a comparison involves a fallback operating point —
# a skimmer reads THIS figure, not the meta — so it renders via plot_artifact with the
# regime_heatmap contract (diverging scale, /// = CI excludes zero) plus a ○ marker on
# every cell whose comparison involves a PLAN §5 fallback (precision unattainable).
_fb_ops = regime_ops[~regime_ops["attained"].astype(bool)]
_fb_note = ("" if _fb_ops.empty else
            "○ = comparison involves a fallback operating point ("
            + "; ".join(f"{s} at max-attainable {g['precision'].min():.3f}"
                        + ("" if len(g) == 1 or g["precision"].min() == g["precision"].max()
                           else f"–{g['precision'].max():.3f}")
                        for s, g in _fb_ops.groupby("system")) + ")")


def draw_regime_map(ax, df, meta):
    matrix = (df.pivot(index="row", columns="col", values="value")
              .sort_index(axis=0).sort_index(axis=1))
    vals = matrix.to_numpy(dtype=float)
    vmax = float(np.nanmax(np.abs(vals))) if np.isfinite(vals).any() else 1.0
    im = ax.imshow(vals, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
    ax.set_xticks(range(matrix.shape[1]), [str(c) for c in matrix.columns],
                  rotation=45, ha="right")
    ax.set_yticks(range(matrix.shape[0]), [str(r) for r in matrix.index])
    ax.set_xlabel("regime")
    ax.set_ylabel("comparison")
    ax.figure.colorbar(im, ax=ax, label=f"paired Δ B³F1 at precision@{PREC_PRIMARY}")
    sig = df.pivot(index="row", columns="col", values="ci_excludes_zero").reindex(
        index=matrix.index, columns=matrix.columns)
    fb = df.pivot(index="row", columns="col", values="fallback_involved").reindex(
        index=matrix.index, columns=matrix.columns)
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            if pd.notna(sig.iloc[i, j]) and float(sig.iloc[i, j]) > 0:
                ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False,
                                           hatch="///", edgecolor="0.2", linewidth=0))
            if pd.notna(fb.iloc[i, j]) and float(fb.iloc[i, j]) > 0:
                ax.plot(j + 0.36, i - 0.33, marker="o", markersize=5.5, color="0.1",
                        markerfacecolor="white", markeredgewidth=1.1)


fig = figures.plot_artifact(
    registry, tier=cfg.run.tier, artifact="hyb01_regime_map", draw=draw_regime_map,
    title=f"The noise-regime map: paired ΔB³F1 at precision@{PREC_PRIMARY} "
          "(hatched = 95% CI excludes zero)"
          + (f"\n{_fb_note}" if _fb_note else ""),
    figsize=(7.6, 4.9),
)

# %% [markdown]
# ### Verdict — scoring the HYB-01 card against the registered numbers

# %%
_d = deltas.set_index("system")
_g = _d.loc["emb+guard"]
_o = _d.loc["emb+override"]
p1 = bool(_g["delta_f1"] > 0 and _g["sign_stable"])
p2 = bool(not _o["sign_stable"])
_hyb_emb = regime_cells[regime_cells["row"] == "hybrid - embedding"]
p3 = bool((_hyb_emb["value"] >= 0).all()
          and not ((_hyb_emb["value"] < 0) & _hyb_emb["ci_excludes_zero"]).any())
refuted = bool((_g["sign_stable"] and _g["delta_f1"] < 0)
               or (_o["sign_stable"] and _o["delta_f1"] > 0))
print(f"P1 (guard earns): delta_f1 {_g['delta_f1']:+.4f} "
      f"[{_g['delta_lo']:+.4f}, {_g['delta_hi']:+.4f}], sign_stable={_g['sign_stable']} "
      f"-> {p1}")
print(f"P2 (override redundant): delta_f1 {_o['delta_f1']:+.4f} "
      f"[{_o['delta_lo']:+.4f}, {_o['delta_hi']:+.4f}], sign_stable={_o['sign_stable']} "
      f"-> {p2}  (redundancy share {OV_ALREADY:.2%})")
print(f"P3 (subordinate): hybrid-vs-embedding across {len(_hyb_emb)} regimes, min delta "
      f"{_hyb_emb['value'].min():+.4f}, negative-and-sign-stable cells "
      f"{int(((_hyb_emb['value'] < 0) & _hyb_emb['ci_excludes_zero']).sum())} -> {p3}")
hyb01_outcome = "REFUTED" if refuted else ("CONFIRMED" if (p1 and p2) else "UNEXPLAINED")
print(f"-> outcome: {hyb01_outcome}")

_ = verdict_box(
    "HYB-01",
    outcome=hyb01_outcome,
    evidence=(
        f"hyb01_rule_value_map + hyb01_regime_map (tier {TIER}; union graph {len(upairs):,} "
        f"pairs, PC {float(bm_u['pair_completeness']):.4f}; {N_BOOT} shared entity "
        f"resamples). P1 {p1}: the birth-year guard's paired ΔB³F1 vs embedding-only at "
        f"precision@{PREC_PRIMARY} is {_g['delta_f1']:+.4f} "
        f"[{_g['delta_lo']:+.4f}, {_g['delta_hi']:+.4f}] "
        f"(guard fired on {int(FIRES['guard'].sum()):,} pairs, "
        f"{float(UP_TRUE[FIRES['guard']].mean() if FIRES['guard'].sum() else np.nan):.4f} "
        "of them true — the hub-dob collateral mechanism measured in §6). "
        f"P2 {p2}: the exact-key override's Δ is {_o['delta_f1']:+.4f} "
        f"[{_o['delta_lo']:+.4f}, {_o['delta_hi']:+.4f}] with "
        f"{OV_ALREADY:.0%} of its fires already above the embedding's own threshold. "
        f"P3 {p3}: in the regime map, hybrid-minus-embedding is nonnegative in "
        f"{int((_hyb_emb['value'] >= 0).sum())}/{len(_hyb_emb)} columns and nowhere "
        "negative-and-sign-stable. The MET-04 residual-inclusive bar is "
        f"{BAR_FULL:.4f} B³F1 ({BAR_SRC}): rule deltas inside the bar are within "
        "single-seed replicate noise even when their paired CI excludes zero — the paired "
        "design detects small effects, the bar says how far they generalize across "
        "retrains. SINGLE-SEED DEMONSTRATION: one encoder, one corpus draw, one "
        "regeneration per regime; the definitive HYB-01 is the tier=mid multi-seed run "
        "(placard below)."
    ),
    registry=registry,
)

# %% [markdown]
# ## 9. NSE-03 — does the ranking survive a change of noise model?
#
# Three noise models, one question: is "who beats whom" a property of the systems or of the
# dirt? Regime 1 is the calibrated eval-half graph (§7). Regime 2 regenerates the same base
# slice with **NB05's generic baseline**: one uniform typo channel over five fields at the
# NB05-matched total edit budget (read from `calibrated_channel_rates` meta — the classic
# generic generator). Regime 3 is **real**: NC same-NCID aligned pairs across two snapshots
# as positives, sampled cross-NCID within-county pairs as negatives. **The NC regime is
# pair-level only** — aligned pairs carry no entity structure, so no entity metric exists
# there; the shared currency across all three regimes is pair-level AUC. One declared
# representation for every AUC arm (§6): the **raw cosine** for the embedding and the
# grammar applied rank-natively for the hybrid — the isotonic map is only weakly monotone
# (piecewise-constant, measured step count in §6), so pushing scores through it collapses
# ties and measurably shifts AUC; the map still matters wherever probabilities are needed,
# and the entity_op rows keep using it. NC values are casefolded before scoring (one
# documented normalization, identical for every system — the corpus the encoder trained on
# is lowercase); every display is masked.


# %%
def fast_auc(scores: np.ndarray, labels: np.ndarray) -> float:
    """Mann-Whitney AUC via ranks (ties handled by average ranks)."""
    r = rankdata(scores)
    n_pos = int(labels.sum())
    n_neg = len(labels) - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    return float((r[labels].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def auc_table(score_dict: dict[str, np.ndarray], labels: np.ndarray, *, cap: int,
              n_boot: int, seed: int) -> tuple[dict, dict, dict]:
    """AUC point + pair-bootstrap CI per system, SHARED-DRAW pairwise deltas, raw boots.

    Pair-resample bootstrap (unit=pair, stated limitation for entity-structured regimes);
    all systems share every resample draw, so within-regime deltas — and any function of
    several systems' replicate AUCs (e.g. PRS-01's difference-in-differences) — are paired.
    """
    rng = np.random.default_rng(seed)
    n = len(labels)
    idx_full = np.arange(n)
    if n > cap:
        idx_full = np.sort(rng.choice(n, size=cap, replace=False))
    lab = labels[idx_full]
    subs = {k: v[idx_full] for k, v in score_dict.items()}
    boots = {k: np.empty(n_boot) for k in subs}
    m = len(idx_full)
    for i in range(n_boot):
        draw = rng.integers(0, m, size=m)
        yl = lab[draw]
        for k, v in subs.items():
            boots[k][i] = fast_auc(v[draw], yl)
    out = {k: {"auc": fast_auc(v, lab),
               "lo": float(np.nanquantile(boots[k], 0.025)),
               "hi": float(np.nanquantile(boots[k], 0.975)),
               "n_pairs": int(m), "pos_rate": float(lab.mean())}
           for k, v in subs.items()}
    dd = {}
    names = list(subs)
    for i, ka in enumerate(names):
        for kb in names[i + 1:]:
            d = boots[ka] - boots[kb]
            lo, hi = float(np.nanquantile(d, 0.025)), float(np.nanquantile(d, 0.975))
            dd[(ka, kb)] = {"delta": out[ka]["auc"] - out[kb]["auc"], "lo": lo, "hi": hi,
                            "sign_stable": bool(lo > 0 or hi < 0)}
    return out, dd, boots


# %%
t_sec = time.time()
GENERIC_BUDGET = float(rates_meta["extra"]["generic_typo_budget"])
with np.errstate(divide="ignore"):
    slice_gen, ops_gen = generate_corpus(
        reg_base, channel_rates={"typo_uniform": GENERIC_BUDGET},
        channels={"typo_uniform": typo(fields=("given_name", "family_name", "city", "zip",
                                               "dob"))},
        keep_original_rate=1.0, seed=REGEN_SEED,
    )
truth_g = slice_gen.set_index(slice_gen["record_id"].astype(str))["entity_id"]
records_g = pd.Index(truth_g.index)
texts_g = serialize_frame(slice_gen, text_roles=TEXT_ROLES, scheme=SER_SCHEME,
                          missing=SER_MISSING).tolist()
emb_g = encoder.encode(texts_g, batch_size=ENC_BATCH)
mk_g = matchkeys.candidates(slice_gen, passes=matchkeys.default_passes(slice_gen))
ann_g = ann.candidates(slice_gen, emb_g, k=K_CAND, index="flat")
pg = pd.concat([mk_g, ann_g], ignore_index=True)
pg["a"], pg["b"] = pg["a"].astype(str), pg["b"].astype(str)
pg = pg.drop_duplicates(subset=["a", "b"]).reset_index(drop=True)
pos_g = pd.Series(np.arange(len(slice_gen)), index=slice_gen["record_id"].astype(str))
recs_g = slice_gen.set_index(slice_gen["record_id"].astype(str))
ag = recs_g.loc[pg["a"]].reset_index(drop=True)
bg = recs_g.loc[pg["b"]].reset_index(drop=True)
cos_g = np.einsum("ij,ij->i", emb_g[pos_g.loc[pg["a"]].to_numpy()],
                  emb_g[pos_g.loc[pg["b"]].to_numpy()]).astype(float)
fires_g = rule_fires(ag, bg)
gen_scores = {"fs_standin": jw_standin(ag, bg), "embedding": cal_iso_g.transform(cos_g)}
gen_scores["hybrid_all"] = apply_rules(gen_scores["embedding"], fires_g, use_override=True,
                                       use_guard=True, use_feature=True)
gen_true = (truth_g.loc[pg["a"]].to_numpy() == truth_g.loc[pg["b"]].to_numpy())
print(f"generic regime: uniform typo over 5 fields at the NB05-matched budget "
      f"{GENERIC_BUDGET:.6f} (calibrated_channel_rates meta), same base slice + "
      f"regeneration seed as the regime map -> {len(slice_gen):,} records, "
      f"{len(ops_gen):,} ops, {len(pg):,} union pairs")

# supporting entity-F rows for the generic regime (same protocol as the regime map)
gen_entity_rows: list[dict] = []
gen_preds: dict[str, pd.Series] = {}
for system, sc in gen_scores.items():
    sf = pd.DataFrame({"a": pg["a"], "b": pg["b"], "score": sc, "prob": sc})
    cache_g: dict[float, pd.Series] = {}

    def clus_g(t: float, frame=sf, cache=cache_g) -> pd.Series:
        t = float(t)
        if t not in cache:
            cache[t] = transitive_closure(frame, threshold=t, records=records_g)
        return cache[t]

    res = find_threshold_for_precision(sf[["a", "b", "score"]], clus_g, truth_g,
                                       target=PREC_PRIMARY, grid=REG_GRID)
    gen_preds[system] = clus_g(res["threshold"])
    ci = bootstrap_ci(gen_preds[system], truth_g, "bcubed_f1", unit="entity",
                      n_boot=N_BOOT, seed=PRIMARY_SEED)
    gen_entity_rows.append({"row_type": "entity_op", "regime": "generic-uniform",
                            "system": system, "attained": bool(res["attained"]),
                            "fallback": res["fallback"] or "",
                            "precision": res["attained_precision"],
                            "recall": res["recall_at"], "f1": ci["point"],
                            "f1_lo": ci["ci_low"], "f1_hi": ci["ci_high"],
                            "level": "entity", "basis": "MEASURED"})
print(f"generic-regime entity support (precision@{PREC_PRIMARY}): "
      + "; ".join(f"{r['system']} F1 {r['f1']:.4f}"
                  + ("" if r["attained"] else " [FALLBACK]") for r in gen_entity_rows))
tick("§9a generic regime", t_sec)

# %% [markdown]
# ### The real thing: NC temporal-drift pairs (pair-level, masked)
#
# Positives are same-NCID aligned pairs across the two registered snapshots (the drift the
# NB04 audit measured); negatives are sampled cross-NCID pairs within the same county — half
# uniform, half **same-family-name hard negatives** (a uniform-only negative set makes every
# system look perfect; the composition is registered). SAID LOUDLY ONCE MORE: this regime is
# **pair-level only** — no entity structure, no clustering, no entity bootstrap. The
# hybrid's birth key falls back to `snapshot_year - age` within ±1 (pre-registered in the
# card); the exact-date override path cannot fire here (NC has no full DOB — MET-06's
# identifiability point, felt again).

# %%
t_sec = time.time()
NC_FIELDS = ["given_name", "family_name", "city", "zip", "sex", "age"]
_al_cols = ["ncid", "snapshot_date_a", "snapshot_date_b"] + [
    f"{f}_{s}" for f in NC_FIELDS for s in ("a", "b")]
aligned = pd.read_parquet(REPO_ROOT / creg["nc_snapshots"]["aligned_path"],
                          columns=_al_cols)
_rng_nc = np.random.default_rng(PRIMARY_SEED + 71)
n_pos = min(NC_POS, len(aligned))
pos_idx = np.sort(_rng_nc.choice(len(aligned), size=n_pos, replace=False))

# uniform cross-NCID negatives (a-side vs b-side of DIFFERENT rows)
_iu_a = _rng_nc.integers(0, len(aligned), size=int(1.3 * n_pos))
_iu_b = _rng_nc.integers(0, len(aligned), size=int(1.3 * n_pos))
_ncid = aligned["ncid"].to_numpy()
_keep_u = _ncid[_iu_a] != _ncid[_iu_b]
neg_u = np.stack([_iu_a[_keep_u], _iu_b[_keep_u]], axis=1)[:n_pos]

# same-family hard negatives: for a sampled a-row, a b-row sharing casefolded family_name
_famb = aligned["family_name_b"].astype("string").str.strip().str.casefold()
_fam_groups = pd.Series(np.arange(len(aligned))).groupby(_famb.to_numpy()).indices
_fama = aligned["family_name_a"].astype("string").str.strip().str.casefold().to_numpy()
neg_f_rows: list[tuple[int, int]] = []
for ia in _rng_nc.integers(0, len(aligned), size=int(4 * n_pos)):
    fam = _fama[ia]
    if pd.isna(fam) or fam not in _fam_groups:
        continue
    grp = _fam_groups[fam]
    ib = int(grp[_rng_nc.integers(len(grp))])
    if _ncid[ia] != _ncid[ib]:
        neg_f_rows.append((int(ia), ib))
    if len(neg_f_rows) >= n_pos:
        break
neg_f = (np.asarray(neg_f_rows, dtype=int) if neg_f_rows
         else np.empty((0, 2), dtype=int))

ia_all = np.concatenate([pos_idx, neg_u[:, 0], neg_f[:, 0]])
ib_all = np.concatenate([pos_idx, neg_u[:, 1], neg_f[:, 1]])
nc_label = np.concatenate([np.ones(n_pos, bool), np.zeros(len(neg_u), bool),
                           np.zeros(len(neg_f), bool)])


def nc_side(rows: pd.DataFrame, side: str) -> pd.DataFrame:
    """One pair side as a canonical role frame; values casefolded for scoring (stated)."""
    out = pd.DataFrame(index=pd.RangeIndex(len(rows)))
    for f in ("given_name", "family_name", "city", "zip", "sex"):
        out[f] = _norm_str(rows[f"{f}_{side}"].reset_index(drop=True))
    out["age"] = pd.to_numeric(rows[f"age_{side}"].reset_index(drop=True), errors="coerce")
    out["snapshot_year"] = pd.to_numeric(
        rows[f"snapshot_date_{side}"].reset_index(drop=True).astype("string").str[:4],
        errors="coerce")
    return out


nc_a = nc_side(aligned.iloc[ia_all], "a")
nc_b = nc_side(aligned.iloc[ib_all], "b")
print(f"NC drift pairs: {n_pos:,} positives (same NCID, "
      f"{aligned['snapshot_date_a'].iloc[0][:4]} vs {aligned['snapshot_date_b'].iloc[0][:4]}"
      f") + {len(neg_u):,} uniform + {len(neg_f):,} same-family cross-NCID negatives, all "
      f"within county {creg['nc_snapshots']['counties']}; PAIR-LEVEL ONLY (no entity "
      "structure); values casefolded for scoring; subsampling stated: "
      f"{n_pos:,} of {len(aligned):,} aligned pairs")


def mask_value(field: str, v: object) -> str:
    """DATA_GOVERNANCE display masking: initials+length for names; coarse fields verbatim."""
    if pd.isna(v) or not str(v).strip():
        return "-"
    s = str(v).strip()
    if field in ("given_name", "family_name"):
        return f"{s[0]}.({len(s)})"
    return s  # city / zip / sex / age: coarse, NB03/NB04 convention


_show = [0, 1, n_pos + 2]  # two positives, one uniform negative
_mask_rows = {}
for i in _show:
    _mask_rows[f"pair{i}({'pos' if nc_label[i] else 'neg'})"] = {
        f"{f}_{s}": mask_value(f, (nc_a if s == 'a' else nc_b)[f].iloc[i])
        for f in ("given_name", "family_name", "city", "zip") for s in ("a", "b")}
print("\nmasked examples (initials+length only — real NC person fields never display raw):")
display(pd.DataFrame(_mask_rows))
tick("§9b NC pair construction", t_sec)

# %%
t_sec = time.time()
_t0 = time.time()
nc_texts = serialize_frame(pd.concat([nc_a, nc_b], ignore_index=True),
                           text_roles=TEXT_ROLES, scheme=SER_SCHEME,
                           missing=SER_MISSING).tolist()
nc_emb = encoder.encode(nc_texts, batch_size=ENC_BATCH)
NC_ENC_SECS = time.time() - _t0
nc_cos = np.einsum("ij,ij->i", nc_emb[: len(nc_a)], nc_emb[len(nc_a):]).astype(float)
nc_fires = rule_fires(nc_a, nc_b)  # dob absent -> birth key = snapshot_year - age (±1)
nc_scores = {
    "fs_standin": jw_standin(nc_a, nc_b),  # dob absent -> given/family/city fuzzy + zip
    "embedding": nc_cos,  # RAW cosine — the declared AUC representation (§6)
    "hybrid_all": apply_rules_ranknative(nc_cos, nc_fires),
}
print(f"encoded {len(nc_texts):,} NC pair sides in {NC_ENC_SECS:.0f}s; dob is [MISSING] in "
      "every serialization (NC has no full DOB). Measured tie-collapse caveat: the "
      f"isotonic map is only WEAKLY monotone ({N_ISO_STEPS} distinct steps on the union "
      "graph) and AUC is invariant only under STRICTLY monotone maps — pushing cosines "
      "through it collapses ties and measurably shifts AUC, so every embedding-based AUC "
      "arm here scores the RAW cosine (rank-native, tie-free); the map would in any case "
      "transfer UNVALIDATED to this distribution wherever probabilities were needed")
for rule in ("override", "guard", "feature"):
    print(f"  NC rule '{rule}': fires on {int(nc_fires[rule].sum()):,} pairs "
          f"({nc_fires[rule].mean():.2%}); true-share "
          f"{float(nc_label[nc_fires[rule]].mean()) if nc_fires[rule].sum() else np.nan:.4f}")

# Every AUC arm on the declared rank-native representation (§6): raw cosine for the
# embedding, the grammar applied rank-natively for the hybrid, the JW score for FS.
# (The prob-scale SYSTEMS/gen_scores vectors keep driving the entity operating points —
# that is the system the protocol clusters; only the AUC table changes representation.)
REGIME_AUC_INPUTS = {
    "calibrated": ({"fs_standin": UP_JW, "embedding": UP_COS,
                    "hybrid_all": apply_rules_ranknative(UP_COS, FIRES)},
                   UP_TRUE, "pair (union graph)"),
    "generic-uniform": ({"fs_standin": gen_scores["fs_standin"], "embedding": cos_g,
                         "hybrid_all": apply_rules_ranknative(cos_g, fires_g)},
                        gen_true, "pair (union graph)"),
    "nc-drift": (nc_scores, nc_label, "pair (aligned +/- sampled negatives)"),
}
auc_rows: list[dict] = []
auc_delta_rows: list[dict] = []
RANKINGS: dict[str, tuple] = {}
for regime, (score_dict, labels, universe) in REGIME_AUC_INPUTS.items():
    pts, dds, _ = auc_table(score_dict, np.asarray(labels, bool), cap=AUC_CAP,
                            n_boot=N_BOOT_AUC, seed=PRIMARY_SEED + 77)
    order = tuple(sorted(pts, key=lambda k: -pts[k]["auc"]))
    RANKINGS[regime] = order
    for k, v in pts.items():
        auc_rows.append({"row_type": "auc", "regime": regime, "system": k, "auc": v["auc"],
                         "auc_lo": v["lo"], "auc_hi": v["hi"],
                         "rank": order.index(k) + 1, "n_pairs": v["n_pairs"],
                         "pos_rate": v["pos_rate"], "level": "pair",
                         "universe": universe, "basis": "MEASURED"})
    for (ka, kb), v in dds.items():
        auc_delta_rows.append({"row_type": "auc_delta", "regime": regime, "system": ka,
                               "system_b": kb, "delta": v["delta"], "delta_lo": v["lo"],
                               "delta_hi": v["hi"], "sign_stable": v["sign_stable"],
                               "level": "pair", "basis": "MEASURED"})
auc_df = pd.DataFrame(auc_rows)
print(f"\npair-level AUC by regime ({N_BOOT_AUC} shared pair-resamples, cap {AUC_CAP:,}; "
      "pair unit — entity clustering ignored in the two synthetic regimes, stated; every "
      "embedding-based arm scored on the RAW cosine, rank-native — see the §6 note):")
display(auc_df[["regime", "system", "auc", "auc_lo", "auc_hi", "rank", "n_pairs",
                "pos_rate"]].round(4))
for regime, order in RANKINGS.items():
    print(f"  {regime:>16}: " + " > ".join(order))
_ops_p = ops[(ops["protocol"] == f"precision@{PREC_PRIMARY}") & ops["system"].isin(HEAT_SYSTEMS)]
_ent_cal = " > ".join(_ops_p.sort_values("f1", ascending=False)["system"])
_ent_gen = " > ".join(pd.DataFrame(gen_entity_rows).sort_values("f1", ascending=False)["system"])
print("metric-choice caution (MET-01/02's reversal, live): the pair-AUC ranking above is "
      "NOT the protocol's entity ranking — at the fixed-precision operating point the "
      f"same systems rank {_ent_cal} (calibrated) and {_ent_gen} (generic). NSE-03 is "
      "scored on the pair-level ranking because the NC regime supports nothing else; the "
      "entity_op rows carry the primary-protocol view where it exists.")
tick("§9c NC scoring + AUC table", t_sec)

# %%
t_sec = time.time()
# entity-op support rows for the calibrated regime come from §7's full-eval-half table
_cal_ops = ops[(ops["protocol"] == f"precision@{PREC_PRIMARY}")
               & ops["system"].isin(HEAT_SYSTEMS)].set_index("system")
cal_entity_rows = [{"row_type": "entity_op", "regime": "calibrated", "system": s,
                    "attained": bool(r["attained"]), "fallback": r["fallback"],
                    "precision": r["precision"], "recall": r["recall"], "f1": r["f1"],
                    "f1_lo": r["f1_lo"], "f1_hi": r["f1_hi"], "level": "entity",
                    "basis": "MEASURED"}
                   for s, r in _cal_ops.iterrows()]
nse03 = pd.concat([auc_df, pd.DataFrame(auc_delta_rows),
                   pd.DataFrame(cal_entity_rows + gen_entity_rows)], ignore_index=True)
registry.register(
    "nse03_ranking_invariance", flags_to_float(nse03), cfg=cfg, tier=cfg.run.tier,
    meta={
        "card": "NSE-03",
        "row_types": {
            "auc": "regime x system pair-level ROC AUC with pair-bootstrap 95% CI and the "
                   "within-regime rank",
            "auc_delta": "SHARED-DRAW pairwise AUC deltas within each regime "
                         "(same pair resamples for every system)",
            "entity_op": "supporting entity-F at precision@0.99 — SYNTHETIC REGIMES ONLY "
                         "(the NC regime has no entity structure)",
        },
        "regimes": {
            "calibrated": "eval-half union graph, NB05 measured channel rates (the §7 "
                          "arena)",
            "generic-uniform": f"same base slice + regen seed, one uniform typo channel "
                               f"over 5 fields at the NB05-matched budget "
                               f"{GENERIC_BUDGET:.6f}",
            "nc-drift": "REAL NC pairs — PAIR-LEVEL ONLY, no entity structure: same-NCID "
                        "aligned pairs across 2 snapshots = positives; sampled cross-NCID "
                        "within-county negatives, half uniform / half same-family-name "
                        "hard negatives (composition in the rows); values casefolded for "
                        "scoring; ONLY AGGREGATES REGISTERED — no NC person-level value "
                        "in this artifact; displays masked per DATA_GOVERNANCE.md",
        },
        "systems_note": "hybrid_all uses the HYB-01 grammar; on NC the birth key falls "
                        "back to snapshot_year - age (±1) as pre-registered — the "
                        "exact-date override path cannot fire without full DOB (MET-06)",
        "auc_protocol": {"n_boot": N_BOOT_AUC, "cap": AUC_CAP, "unit": "pair",
                         "limitation": "pair resamples ignore entity clustering in the "
                                       "synthetic regimes (stated in the card)"},
        "auc_representation": "every embedding-based AUC arm scores the RAW cosine "
                              "(embedding) or the grammar applied rank-natively on the "
                              "raw cosine (hybrid_all: override above the cosine range, "
                              "guard below it, feature floored at the prob-floor's "
                              f"cosine preimage {FLOOR_COS:.4f}); the isotonic map is "
                              f"only WEAKLY monotone ({N_ISO_STEPS} distinct steps over "
                              f"the {len(upairs)} union pairs) and AUC is invariant only "
                              "under strictly monotone maps — the tie collapse "
                              "measurably shifts AUC, so one tie-free representation is "
                              "declared for every arm; entity_op rows keep the "
                              "calibrated-probability systems the protocol clusters",
        "single_seed_caveat": "one encoder, one seed, one draw per regime — a "
                              "DEMONSTRATION; NSE-03's definitive home is tier=mid",
        "nc_provenance": f"corpus_registry aligned pairs "
                         f"({creg['nc_snapshots']['align_stats']['pairs']:,} available; "
                         f"{n_pos:,} positives sampled, seed {PRIMARY_SEED + 71})",
        "corpus_provenance": CORPUS_PROVENANCE,
    },
)
print(f"registered nse03_ranking_invariance: {len(auc_df)} auc + {len(auc_delta_rows)} "
      f"delta + {len(cal_entity_rows) + len(gen_entity_rows)} entity_op rows")


def draw_ranking(ax, df, meta):
    d = df[df["row_type"] == "auc"]
    regimes = ["calibrated", "generic-uniform", "nc-drift"]
    xs = {r: i for i, r in enumerate(regimes)}
    for system, mark in (("fs_standin", "o"), ("embedding", "s"), ("hybrid_all", "D")):
        g = d[d["system"] == system].set_index("regime").reindex(regimes)
        x = np.array([xs[r] for r in regimes], dtype=float)
        yerr = np.vstack([np.clip(g["auc"] - g["auc_lo"], 0, None),
                          np.clip(g["auc_hi"] - g["auc"], 0, None)])
        ax.errorbar(x, g["auc"], yerr=yerr, marker=mark, capsize=3, markersize=6,
                    linewidth=1.4, label=system)
        for xi, (_, r) in zip(x, g.iterrows()):
            ax.annotate(f"#{int(r['rank'])}", (xi, r["auc"]), textcoords="offset points",
                        xytext=(8, -3), fontsize=7.5, color="0.35")
    ax.set_xticks(range(len(regimes)),
                  [r + ("\n(PAIR-LEVEL, real NC)" if r == "nc-drift" else "")
                   for r in regimes])
    ax.set_ylabel("pair-level ROC AUC (pair-bootstrap 95% CI)")
    ax.legend(loc="lower left", fontsize=8)


fig = figures.plot_artifact(
    registry, tier=cfg.run.tier, artifact="nse03_ranking_invariance", draw=draw_ranking,
    title="NSE-03: the same three systems under three noise models",
    figsize=(7.2, 4.6),
)

# %% [markdown]
# ### Verdict — scoring the NSE-03 card against the registered numbers

# %%
_orders = list(RANKINGS.values())
p1 = bool(all(o == _orders[0] for o in _orders))
_adj_ok: dict[str, bool] = {}
_dd_df = pd.DataFrame(auc_delta_rows)
for regime, order in RANKINGS.items():
    ok = True
    for ka, kb in itertools.pairwise(order):
        m = _dd_df[(_dd_df["regime"] == regime)
                   & (((_dd_df["system"] == ka) & (_dd_df["system_b"] == kb))
                      | ((_dd_df["system"] == kb) & (_dd_df["system_b"] == ka)))]
        ok &= bool(m["sign_stable"].iloc[0])
    _adj_ok[regime] = ok
p2 = bool(all(_adj_ok.values()))
_rev = False
for i, (_, ra) in enumerate(_dd_df.iterrows()):
    for _, rb in _dd_df.iloc[i + 1:].iterrows():
        if (ra["system"] == rb["system"] and ra["system_b"] == rb["system_b"]
                and ra["regime"] != rb["regime"] and ra["sign_stable"]
                and rb["sign_stable"] and np.sign(ra["delta"]) != np.sign(rb["delta"])):
            _rev = True
print(f"P1 (identical ordering): {[' > '.join(o) for o in RANKINGS.values()]} -> {p1}")
print(f"P2 (adjacent deltas sign-stable per regime): {_adj_ok} -> {p2}")
print(f"REFUTED clause (sign-stable reversal across regimes): {_rev}")
nse03_outcome = "REFUTED" if _rev else ("CONFIRMED" if (p1 and p2) else "UNEXPLAINED")
print(f"-> outcome: {nse03_outcome}")

_ = verdict_box(
    "NSE-03",
    outcome=nse03_outcome,
    evidence=(
        f"nse03_ranking_invariance (tier {TIER}). Orderings by pair-AUC: "
        + "; ".join(f"{r}: {' > '.join(o)}" for r, o in RANKINGS.items())
        + f". P1 {p1}; P2 {p2} (adjacent shared-draw deltas, {N_BOOT_AUC} pair resamples, "
        f"per-regime: {_adj_ok}); sign-stable cross-regime reversal: {_rev}. The NC regime "
        f"is PAIR-LEVEL ONLY ({n_pos:,} same-NCID positives + {len(neg_u) + len(neg_f):,} "
        "sampled cross-NCID negatives, half same-family hard negatives; no entity "
        "structure, values casefolded, aggregates only). Metric-choice caution: at the "
        f"entity fixed-precision operating point the synthetic-regime ranking is "
        f"{_ent_cal} (calibrated) / {_ent_gen} (generic) — MET-01's pair-vs-entity "
        "reversal, live in the entity_op rows; the invariance scored here is pair-level "
        "by necessity, not by preference. Representation, declared: every embedding-based "
        "AUC arm scores the RAW cosine (rank-native, tie-free) — the isotonic map is only "
        f"WEAKLY monotone ({N_ISO_STEPS} distinct steps on the union graph), AUC is "
        "invariant only under STRICTLY monotone maps, and the tie collapse measurably "
        "shifts AUC; the map would in any case transfer to NC unvalidated wherever "
        "probabilities were needed. Stated limitations: pair-resample CIs ignore entity "
        "clustering in the two synthetic regimes; the FS stand-in proxies the tuned FS "
        f"(spearman {_rho:.3f} vs bas01). SINGLE-SEED DEMONSTRATION: one encoder, one "
        "draw per regime — the definitive NSE-03 is tier=mid (PLAN §3)."
    ),
    registry=registry,
)
tick("§9d NSE-03 register + verdict", t_sec)

# %% [markdown]
# ## 10. PRS-01 (EXPLORATORY) — does the parser subsidize the rules?
#
# The 2×2, labeled exploratory: standardization level {parsed fields, raw single-string} ×
# system {FS stand-in, embedding}. `raw_unparse` (the package's pre-standardization channel)
# is applied to the eval half at rate 1.0: name parts compose into `'FAMILY, GIVEN M'` and
# blank out; dob/city/zip/sex survive, so the raw penalty lands exactly on the name signal.
# Same pairs in all four cells; pair-level AUC — and **both embedding cells score the raw
# cosine** (the §6 declared representation), so the parsed→raw drop measures the parsing
# change alone, never a transform change.

# %%
t_sec = time.time()
_rng_raw = np.random.default_rng(PRIMARY_SEED + 501)
raw_eval, raw_ops = raw_unparse().apply(eval_full, _rng_raw, 1.0)
_n_unparsed = int((raw_eval["given_name"].isna() & ~eval_full["given_name"].isna()).sum())
RAW_ROLES = ["full_name", "dob", "city", "zip", "sex"]
print(f"raw_unparse at rate 1.0: {_n_unparsed:,}/{len(eval_full):,} records lost their "
      f"parsed name fields into full_name ({len(raw_ops):,} logged cell edits; records "
      "missing given or family name keep their original fields — reported, not hidden)")
print(f"raw serialization roles (eval-time deviation the raw regime forces): {RAW_ROLES}")
_kept_native = int(len(eval_full) - _n_unparsed)
print(f"stale-full_name caveat (NB08's leak, resurfacing): {_kept_native:,} records "
      f"({_kept_native / len(eval_full):.1%}) could not be re-composed (a name part "
      "missing) and keep their NATIVE full_name — for generated duplicates that value is "
      "the clean pre-noise composition, so the raw embedding arm sees leaked clean names "
      "on exactly those records; any raw-arm gain must be read with this leak in mind")

RAW_RECS = raw_eval.set_index(raw_eval["record_id"].astype(str))
ar_raw = RAW_RECS.loc[upairs["a"]].reset_index(drop=True)
br_raw = RAW_RECS.loc[upairs["b"]].reset_index(drop=True)
_t0 = time.time()
raw_texts = serialize_frame(raw_eval, text_roles=RAW_ROLES, scheme=SER_SCHEME,
                            missing=SER_MISSING).tolist()
raw_emb = encoder.encode(raw_texts, batch_size=ENC_BATCH)
_pos_raw = pd.Series(np.arange(len(raw_eval)), index=raw_eval["record_id"].astype(str))
raw_cos = np.einsum("ij,ij->i", raw_emb[_pos_raw.loc[upairs["a"]].to_numpy()],
                    raw_emb[_pos_raw.loc[upairs["b"]].to_numpy()]).astype(float)
print(f"encoded the raw eval half in {time.time() - _t0:.0f}s")

prs_scores = {
    ("parsed", "fs_standin"): UP_JW,
    ("parsed", "embedding"): UP_COS,  # RAW cosine — the §6 declared AUC representation
    ("raw", "fs_standin"): jw_standin(ar_raw, br_raw, fuzzy=("full_name", "city"),
                                      exact=("dob", "zip")),
    ("raw", "embedding"): raw_cos,  # RAW cosine — same representation as the parsed arm
}
_flat = {f"{arm}|{system}": v for (arm, system), v in prs_scores.items()}
prs_pts, prs_dds, prs_boots = auc_table(_flat, np.asarray(UP_TRUE, bool), cap=AUC_CAP,
                                        n_boot=N_BOOT_AUC, seed=PRIMARY_SEED + 88)
prs_rows = [{"row_type": "auc", "parsing": k.split("|")[0], "system": k.split("|")[1],
             "auc": v["auc"], "auc_lo": v["lo"], "auc_hi": v["hi"],
             "n_pairs": v["n_pairs"], "level": "pair", "basis": "MEASURED"}
            for k, v in prs_pts.items()]
prs_df = pd.DataFrame(prs_rows)
display(prs_df[["parsing", "system", "auc", "auc_lo", "auc_hi"]].round(4))

_fs_drop = prs_dds[("parsed|fs_standin", "raw|fs_standin")]
_emb_drop = prs_dds[("parsed|embedding", "raw|embedding")]
# difference-in-differences on the SAME shared draws (per-replicate, all four systems)
_dnd_boot = ((prs_boots["parsed|fs_standin"] - prs_boots["raw|fs_standin"])
             - (prs_boots["parsed|embedding"] - prs_boots["raw|embedding"]))
_dd_point = _fs_drop["delta"] - _emb_drop["delta"]
_dnd_boot_lo = float(np.nanquantile(_dnd_boot, 0.025))
_dnd_boot_hi = float(np.nanquantile(_dnd_boot, 0.975))
print(f"\nparsed->raw AUC drop: fs_standin {_fs_drop['delta']:+.4f} "
      f"[{_fs_drop['lo']:+.4f}, {_fs_drop['hi']:+.4f}] vs embedding "
      f"{_emb_drop['delta']:+.4f} [{_emb_drop['lo']:+.4f}, {_emb_drop['hi']:+.4f}]")
print(f"difference-in-differences (fs drop - emb drop): {_dd_point:+.4f}, shared-draw 95% "
      f"interval [{_dnd_boot_lo:+.4f}, {_dnd_boot_hi:+.4f}] "
      f"({N_BOOT_AUC} pair resamples shared across all four cells)")

# %%
# Robustness on the LEAK-FREE subset (exploratory, no card change): the stale-full_name
# caveat above names the un-recomposable records that keep a clean native full_name in the
# raw arm. Restricting the SAME shared subsample to pairs where BOTH sides were
# re-composed removes that leak entirely — the four AUCs re-scored there say whether
# P1/P2 stand on their own.
_recomposed = (raw_eval["given_name"].isna().to_numpy()
               & ~eval_full["given_name"].isna().to_numpy())
_recomposed_ids = set(raw_eval.loc[_recomposed, "record_id"].astype(str))
_rng_lf = np.random.default_rng(PRIMARY_SEED + 88)  # auc_table's subsample draw, replayed
_idx_lf = (np.sort(_rng_lf.choice(len(upairs), size=AUC_CAP, replace=False))
           if len(upairs) > AUC_CAP else np.arange(len(upairs)))
_lf = (upairs["a"].iloc[_idx_lf].isin(_recomposed_ids).to_numpy()
       & upairs["b"].iloc[_idx_lf].isin(_recomposed_ids).to_numpy())
_lab_lf = np.asarray(UP_TRUE, bool)[_idx_lf][_lf]
LF_AUCS = {k: fast_auc(v[_idx_lf][_lf], _lab_lf) for k, v in _flat.items()}
LF_FS_DROP = LF_AUCS["parsed|fs_standin"] - LF_AUCS["raw|fs_standin"]
LF_EMB_DROP = LF_AUCS["parsed|embedding"] - LF_AUCS["raw|embedding"]
LF_DND = LF_FS_DROP - LF_EMB_DROP
LF_P1 = bool(np.sign(LF_AUCS["parsed|embedding"] - LF_AUCS["parsed|fs_standin"])
             == np.sign(LF_AUCS["raw|embedding"] - LF_AUCS["raw|fs_standin"]))
LF_P2_DIR = bool(LF_DND > 0)
print(f"leak-free robustness (point estimates, exploratory): {int(_lf.sum()):,} of "
      f"{len(_idx_lf):,} subsampled pairs have NO un-recomposable record on either side; "
      f"there: parsed fs {LF_AUCS['parsed|fs_standin']:.4f} / emb "
      f"{LF_AUCS['parsed|embedding']:.4f}; raw fs {LF_AUCS['raw|fs_standin']:.4f} / emb "
      f"{LF_AUCS['raw|embedding']:.4f} -> fs drop {LF_FS_DROP:+.4f} vs emb drop "
      f"{LF_EMB_DROP:+.4f}, DnD {LF_DND:+.4f} — "
      + ("P1's sign and P2's direction SURVIVE without the stale-full_name leak"
         if (LF_P1 and LF_P2_DIR) else
         f"P1 {LF_P1} / P2-direction {LF_P2_DIR} on the leak-free subset — the verdict "
         "leans on the leak; read it with that weight"))
tick("§10a PRS factorial computed", t_sec)

# %%
t_sec = time.time()
prs_delta_rows = [
    {"row_type": "drop", "system": "fs_standin", "delta": _fs_drop["delta"],
     "delta_lo": _fs_drop["lo"], "delta_hi": _fs_drop["hi"],
     "sign_stable": _fs_drop["sign_stable"], "basis": "MEASURED"},
    {"row_type": "drop", "system": "embedding", "delta": _emb_drop["delta"],
     "delta_lo": _emb_drop["lo"], "delta_hi": _emb_drop["hi"],
     "sign_stable": _emb_drop["sign_stable"], "basis": "MEASURED"},
    {"row_type": "dnd", "system": "fs_minus_emb_drop", "delta": _dd_point,
     "delta_lo": _dnd_boot_lo, "delta_hi": _dnd_boot_hi,
     "sign_stable": bool(_dnd_boot_lo > 0 or _dnd_boot_hi < 0), "basis": "MEASURED"},
]
prs01 = pd.concat([prs_df, pd.DataFrame(prs_delta_rows)], ignore_index=True)
registry.register(
    "prs01_parsing_factorial", flags_to_float(prs01), cfg=cfg, tier=cfg.run.tier,
    meta={
        "card": "PRS-01",
        "exploratory": "EXPLORATORY arm (PLAN §3): cheap, labeled, cuttable — never an "
                       "adoption gate",
        "row_types": {
            "auc": "parsing arm x system pair-level AUC on one shared union-pair "
                   "subsample (pair-bootstrap 95% CI)",
            "drop": "parsed->raw AUC drop per system (shared draws)",
            "dnd": "difference-in-differences, fs drop minus embedding drop; shared-draw "
                   "per-replicate 95% interval",
        },
        "design": {
            "raw_arm": "noise.channels.raw_unparse at rate 1.0 on the eval half "
                       f"({_n_unparsed} records unparsed; names -> 'FAMILY, GIVEN M' in "
                       "full_name, parts blanked; dob/city/zip/sex survive)",
            "fs_fields": {"parsed": "given/family/city fuzzy + dob/zip exact (NB11's "
                                    "validated set)",
                          "raw": "full_name/city fuzzy + dob/zip exact"},
            "embedding_fields": {"parsed": TEXT_ROLES, "raw": RAW_ROLES},
            "pair_universe": f"{min(len(upairs), AUC_CAP)} of {len(upairs)} union pairs "
                             "(same subsample in all four cells)",
        },
        "auc_representation": "BOTH embedding cells score the RAW cosine (the declared "
                              "rank-native AUC representation — see nse03 meta): the "
                              "parsed->raw drop measures the parsing change alone, never "
                              "a transform change",
        "leakfree_subset": {
            "n_pairs": int(_lf.sum()), "of_subsample": len(_idx_lf),
            "aucs": {k: float(v) for k, v in LF_AUCS.items()},
            "fs_drop": float(LF_FS_DROP), "emb_drop": float(LF_EMB_DROP),
            "dnd": float(LF_DND), "p1_sign_survives": LF_P1,
            "p2_direction_survives": LF_P2_DIR,
            "note": "pairs where NEITHER side is an un-recomposable record (no stale "
                    "native full_name leak); point estimates — an exploratory robustness "
                    "check, not a card clause",
        },
        "single_seed_caveat": "one encoder, one seed, pair-resample CIs — an EXPLORATORY "
                              "DEMONSTRATION",
        "corpus_provenance": CORPUS_PROVENANCE,
    },
)
print(f"registered prs01_parsing_factorial: {len(prs_df)} auc + {len(prs_delta_rows)} "
      "delta rows")


def draw_prs(ax, df, meta):
    d = df[df["row_type"] == "auc"]
    xs = {"parsed": 0.0, "raw": 1.0}
    for system, mark in (("fs_standin", "o"), ("embedding", "s")):
        g = d[d["system"] == system].set_index("parsing").reindex(["parsed", "raw"])
        x = np.array([xs[p] for p in g.index], dtype=float)
        yerr = np.vstack([np.clip(g["auc"] - g["auc_lo"], 0, None),
                          np.clip(g["auc_hi"] - g["auc"], 0, None)])
        ax.errorbar(x, g["auc"], yerr=yerr, marker=mark, capsize=3, markersize=6,
                    linewidth=1.4, label=system)
    ax.set_xticks([0, 1], ["parsed fields", "raw single-string\n(raw_unparse)"])
    ax.set_xlim(-0.35, 1.35)
    ax.set_ylabel("pair-level ROC AUC (pair-bootstrap 95% CI)")
    ax.legend(loc="lower left", fontsize=8)


fig = figures.plot_artifact(
    registry, tier=cfg.run.tier, artifact="prs01_parsing_factorial", draw=draw_prs,
    title="PRS-01 (exploratory): standardization level x system — does the ranking flip?",
    figsize=(6.4, 4.4),
)

# %% [markdown]
# ### Verdict — scoring the PRS-01 card against the registered numbers

# %%
_sign_parsed = np.sign(prs_pts["parsed|embedding"]["auc"] - prs_pts["parsed|fs_standin"]["auc"])
_sign_raw = np.sign(prs_pts["raw|embedding"]["auc"] - prs_pts["raw|fs_standin"]["auc"])
p1 = bool(_sign_parsed == _sign_raw)
_dnd_stable = bool(_dnd_boot_lo > 0 or _dnd_boot_hi < 0)
p2 = bool(_dd_point > 0 and _dnd_stable)
_parsed_sys = prs_dds[("parsed|fs_standin", "parsed|embedding")]
_raw_sys = prs_dds[("raw|fs_standin", "raw|embedding")]
refuted = bool((not p1) and _parsed_sys["sign_stable"] and _raw_sys["sign_stable"])
print(f"P1 (no flip): sign(emb - fs) parsed {_sign_parsed:+.0f} vs raw {_sign_raw:+.0f} "
      f"-> {p1}")
print(f"P2 (fs drop > emb drop, DnD interval excludes 0): DnD {_dd_point:+.4f} "
      f"[{_dnd_boot_lo:+.4f}, {_dnd_boot_hi:+.4f}] -> {p2}")
prs01_outcome = "REFUTED" if refuted else ("CONFIRMED" if (p1 and p2) else "UNEXPLAINED")
print(f"-> outcome: {prs01_outcome}")

_ = verdict_box(
    "PRS-01",
    outcome=prs01_outcome,
    evidence=(
        f"prs01_parsing_factorial (tier {TIER}; EXPLORATORY). AUCs (both embedding cells "
        "on the RAW cosine — one declared representation, so the drop is the parsing "
        f"change alone): parsed fs {prs_pts['parsed|fs_standin']['auc']:.4f} / emb "
        f"{prs_pts['parsed|embedding']['auc']:.4f}; raw fs "
        f"{prs_pts['raw|fs_standin']['auc']:.4f} / emb "
        f"{prs_pts['raw|embedding']['auc']:.4f}. P1 {p1} (ranking sign parsed vs raw); "
        f"P2 {p2}: fs parsed->raw drop {_fs_drop['delta']:+.4f} vs embedding "
        f"{_emb_drop['delta']:+.4f}, shared-draw DnD {_dd_point:+.4f} "
        f"[{_dnd_boot_lo:+.4f}, {_dnd_boot_hi:+.4f}]. Leak-free robustness (point "
        f"estimates, {int(_lf.sum()):,} pairs with no un-recomposable record on either "
        f"side): fs drop {LF_FS_DROP:+.4f} vs emb drop {LF_EMB_DROP:+.4f}, DnD "
        f"{LF_DND:+.4f} — P1's sign {'survives' if LF_P1 else 'does NOT survive'} and "
        f"P2's direction {'survives' if LF_P2_DIR else 'does NOT survive'} without the "
        "stale-full_name leak. Caveats carried: pair-level AUC only, pair-resample CIs, "
        "the raw embedding arm re-serializes with full_name (an eval-time field-set "
        f"deviation the regime forces, stated in the card), {_kept_native:,} "
        f"un-recomposable records ({_kept_native / len(eval_full):.1%}) keep a stale "
        "NATIVE full_name — a clean-name leak into the raw embedding arm (NB08's leak "
        "mechanism, quantified above and bounded by the leak-free subset) — and the "
        "encoder never trained on raw-form strings; a raw-trained encoder on a leak-free "
        "composition is the mid-tier follow-up this exploratory arm motivates, not a "
        "claim it makes. SINGLE-SEED EXPLORATORY DEMONSTRATION — never an adoption gate "
        "(PLAN §3)."
    ),
    registry=registry,
)
tick("§10b PRS register + verdict", t_sec)

# %% [markdown]
# ## 11. Pricing the definitive runs

# %%
# [RUN-IN-TARGET mac] definitive HYB-01/NSE-03: THESE same cells at tier=mid on the mac —
# ~1e6-record corpus, met04-sized replicates (seeds x noise draws x regime regenerations),
# the same grammar, the same registry names. The regime map is only adjudicable there: at
# smoke each cell is ONE regeneration of a ~6k-record slice and the MET-04 bar dwarfs most
# rule deltas. Estimates below use THIS run's measured coefficients (4-CPU container; a
# concurrent workload can inflate them 2-4x — planning numbers, re-measured on arrival).
if TIER in ("mid", "target"):
    print(f"tier={TIER}: the regime map above IS the definitive run at this scale; "
          "replicate counts per met04_power_table at this tier.")
else:
    _gen_s = float(np.mean([s["generate"] for s in REGIME_SECS.values()]))
    _cell_s = float(np.mean([s["ops_and_deltas"] for s in REGIME_SECS.values()])) / 3
    _n_mid = 1e6
    _scale_rec = _n_mid / float(np.mean([REGIME_STORE[r]["n_records"]
                                         for r in REGIME_MIXES]))
    _seeds = len(met04_meta["extra"]["pilot_design"]["seeds"])
    _draws = len(met04_meta["extra"]["pilot_design"]["noise_draw_seeds"])
    _cells = len(REGIME_MIXES) * 3
    _mid_h = (_seeds * _draws * (len(REGIME_MIXES) * (_gen_s + _cell_s * 3) * _scale_rec
                                 + _n_mid / ENC_RATE * len(REGIME_MIXES))) / 3600
    print(f"[RUN-IN-TARGET mac] HYB-01 regime map at tier=mid (~{_n_mid:.0e} records), "
          f"priced from THIS run's coefficients:")
    print(f"  per regime here: generate {_gen_s:.1f}s, encode "
          f"{np.mean([s['encode'] for s in REGIME_SECS.values()]):.1f}s "
          f"({ENC_RATE:,.0f} rec/s), sweeps+CIs+paired deltas {_cell_s * 3:.1f}s "
          f"for 3 systems (grid {REG_GRID}, {N_BOOT} boots)")
    print(f"  mid, {_seeds} seeds x {_draws} noise draws (met04 pilot design) x "
          f"{len(REGIME_MIXES)} regimes x 3 systems = {_seeds * _draws * _cells} cells: "
          f"~{_mid_h:.1f} container-hours if coefficients scale ~linearly in records; "
          "MPS-side encode and a numpy union-find (package note) cut this well under a "
          "mac day — clustering sweeps, not encoding, dominate at 1e6")
    print(f"  [RUN-IN-TARGET node] the 1e7 arbitration NB17 consumes: encode "
          f"~{1e7 / 20_000 / 60:.0f} min on one A100 (PLAN §8 anchor); the regime map at "
          f"1e7 multiplies the mid cost by ~10 — run the 3-system map at ONE definitive "
          "mix per axis and let SCL-01's ladder own the n-scaling question")

# %% [markdown]
# ## 12. What earned its place — and what did not, at smoke scope
#
# The closing ledger is printed from the registered artifacts (nothing asserted that a cell
# did not compute). Read it with the standing caveat: single seed, single corpus draw, one
# regeneration per regime — smoke-scope demonstrations that fix the *machinery* (grammar,
# paired-delta protocol, regime map, invariance table) for the mid-tier adjudication.

# %%
print("THE CLOSING LEDGER (smoke scope, single seed — demonstrations, not adjudications)")
print("=" * 78)
_d99 = deltas.set_index("system")
for system in ("emb+override", "emb+guard", "emb+feature", "hybrid_all"):
    r = _d99.loc[system]
    verdict = ("EARNS ITS PLACE (sign-stable, but read vs the "
               f"{BAR_FULL:.3f} bar)" if (r["sign_stable"] and r["delta_f1"] > 0)
               else "HURTS (sign-stable negative)" if (r["sign_stable"]
                                                      and r["delta_f1"] < 0)
               else "ADDS NOTHING DETECTABLE at this scope")
    print(f"  {system:>13}: ΔB³F1 {r['delta_f1']:+.4f} "
          f"[{r['delta_lo']:+.4f}, {r['delta_hi']:+.4f}] vs embedding-only -> {verdict}")
print(f"  ranking invariance: {nse03_outcome} across calibrated / generic / real NC drift "
      f"({' > '.join(RANKINGS['calibrated'])} on the calibrated slice)")
print(f"  cross-regime nuance: on REAL NC drift the guard's true-pair collateral is "
      f"{float(nc_label[nc_fires['guard']].mean()):.4f} (vs "
      f"{float(UP_TRUE[FIRES['guard']].mean()):.4f} on the synthetic corpus, where the "
      "native dob errors exceed the gap) — consistent with the guard's harm being "
      "corpus-native dob noise rather than a property of guards; tier=mid adjudicates")
print(f"  parsing factorial (EXPLORATORY): {prs01_outcome} — fs drop "
      f"{_fs_drop['delta']:+.4f} vs emb drop {_emb_drop['delta']:+.4f} parsed->raw")
print(f"  cards: HYB-01 {hyb01_outcome} | NSE-03 {nse03_outcome} | PRS-01 {prs01_outcome}")

# %% [markdown]
# **What the rest of the series inherits.**
#
# - **Notebook 17 consumes `hyb01_rule_value_map` by DAG contract**: the rule-by-rule paired
#   deltas and the regime map are the hybrid half of the 1e7 verdict; the grammar and its
#   pre-registered constants transfer unchanged, and the fallback cells in the meta tell 17
#   which operating points were smoke-unattainable.
# - **`nse03_ranking_invariance` is the transfer license** (or its refusal): whether
#   synthetic-corpus conclusions may be quoted against real drift, with the pair-level
#   caveat stamped into the rows.
# - **`prs01_parsing_factorial` hands notebook 18 the standardization question**: ADP-01's
#   schema adaptation decides how much parsing to buy; the exploratory 2×2 is the first
#   price signal.
#
# **Package gaps noted this notebook** (worked around in-notebook, per series convention):
# `entity_complete_subsample` still notebook-local (05/09/12/13/14 copies); NB05's
# pool-redraw / name-format-drift channels re-implemented here (a shared
# `noise.channels` home would serve both); no pair-level AUC helper with a
# cluster-aware (entity-unit) bootstrap in `er_lab.eval`; `regime_heatmap` orders rows and
# columns alphabetically (labels here chosen to sort correctly — an explicit order parameter
# would be cleaner); python union-find remains the target-tier closure bottleneck (NB13's
# note stands).
#
# **Artifacts registered** (exact names): `hyb01_rule_value_map`, `hyb01_regime_map`
# (auxiliary), `nse03_ranking_invariance`, `prs01_parsing_factorial` — plus the immutable
# cards `card_HYB-01`, `card_NSE-03`, `card_PRS-01`.

# %%
timing = pd.DataFrame(SECTION_TIMES, columns=["section", "secs"])
display(timing.assign(secs=timing["secs"].round(0)))
total = time.time() - NB_T0
print(f"notebook wall-clock: {total:.0f}s ({total / 60:.1f} min) — smoke budget <= ~30 min "
      "uncontended"
      + ("" if total <= 2100 else "  <-- over that budget at this run (see section table; "
                                  "a concurrent notebook re-execution contends for these "
                                  "4 CPUs)"))
