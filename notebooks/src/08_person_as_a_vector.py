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
# # 08. A Person as a Vector
#
# **The question.** What *is* a good person-embedding — and how would we know? This is the
# first notebook of the TRAINING-PRESSURES arc, and it earns the arc its vocabulary. The lab's
# working hypothesis (PLAN §1) commits to a discipline: "a good embedding for ER" is not a
# leaderboard number but a **set of testable properties** — invariances that should hold
# (typos, nicknames, format drift, field order), separations that must not collapse (twins,
# Jr/Sr, one-birth-year-off doppelgängers), and geometry that supports retrieval (alignment,
# uniformity, effective rank, hubness). Every training design decision is a pre-registered
# conjecture that a *pressure* moves a *property* which moves a *system metric* — and the
# mediation is checked, never asserted.
#
# **What this notebook settles.** Three firsts. (1) The first trained encoders of the lab —
# TRN-05's smoke slice: is the way a record becomes a *string* (serialization scheme ×
# missing-field treatment) a real training pressure at matched step budget? (2) The debut of
# the **invariance battery**: a CheckList-style report card measured on a trained encoder AND
# its untrained identically-initialized twin — the contrast is the story, because training is
# what reorganizes string similarity into person similarity. (3) The **geometry panel**
# (alignment / uniformity / RankMe / hubness) and the arc's signature visual, the
# pressure→property→metric triptych that notebooks 09–11 will re-render for every pressure.
#
# Everything downstream is protocol from notebooks 02–07: entity-disjoint splits (MET-07),
# entity-level metrics with entity-unit BCa CIs (MET-01/03), fixed-precision operating points
# with the loud fallback rail (MET-02, PLAN §5), and the calibrated corpus (notebook 05).

# %%
import hashlib
import time

import numpy as np
import pandas as pd
from IPython.display import display
from omegaconf import OmegaConf

from er_lab.blocking import ann
from er_lab.cluster.schemes import transitive_closure
from er_lab.config import REPO_ROOT, config_hash, load_config_from_env, set_all_seeds
from er_lab.eval.bootstrap import bootstrap_ci, paired_delta
from er_lab.eval.metrics import bcubed, blocking_metrics
from er_lab.eval.operating_points import find_threshold_for_precision
from er_lab.infra.artifacts import ArtifactRegistry
from er_lab.infra.device import describe_platform
from er_lab.models.encoders import build_encoder
from er_lab.noise.channels import fetch_lexicon, load_lexicon
from er_lab.probes.battery import MUST_NOT_HOLD, battery_report, build_probe_set
from er_lab.probes.geometry import alignment, hubness, rankme, uniformity
from er_lab.reporting import figures
from er_lab.reporting.cards import conjecture_card, verdict_box
from er_lab.serialize import MISSING_TOKEN, SCHEMES, serialize_frame, serialize_record
from er_lab.train.loop import train_encoder

cfg = load_config_from_env()
registry = ArtifactRegistry.from_env()
figures.setup_style()
set_all_seeds(cfg.run.seed)
NB_T0 = time.time()
SECTION_SECS: dict[str, float] = {}

# %%
# Tier banner — where and under what config this run happened.
print(f"tier        = {cfg.run.tier}")
print(f"config hash = {config_hash(cfg)}")
for key, val in describe_platform().items():
    print(f"  {key:>14}: {val}")

# %% [markdown]
# ## Tier constants
#
# One knob block, tier-scaled; the code path never forks on tier (PLAN §2). The encoder is
# the lab's **from-scratch char/byte transformer** — the smoke-tier default per the
# notes/COMPAT.md go/no-go tree (huggingface.co is blocked here; the pretrained-subword
# regime appears below as a `# [RUN-IN-TARGET mac]` cell that runs for real wherever weights
# are reachable). Budget-matching is PLAN §3.1's step-count accounting: every arm gets an
# **identical optimizer-step count**, asserted from the returned training history — never
# epochs, never wall-clock. Smoke budget for this whole notebook: **<= ~30 min on the 4-CPU
# container** (section timings measured and printed at the end; the container's wall-clock
# is contention-noisy, so the per-arm coefficients are printed too).

# %%
TIER = str(cfg.run.tier)
STEPS = {"smoke": 250, "mid": 2000, "target": 4000, "analytical": 250}[TIER]
N_TRAIN = {"smoke": 6_000, "mid": None, "target": None, "analytical": 6_000}[TIER]  # None=all
N_EVAL = {"smoke": 6_000, "mid": None, "target": None, "analytical": 6_000}[TIER]
N_BOOT = {"smoke": 400, "mid": 1000, "target": 2000, "analytical": 400}[TIER]
GRID = {"smoke": 60, "mid": 150, "target": 200, "analytical": 60}[TIER]
PROBE_N = {"smoke": 120, "mid": 200, "target": 200, "analytical": 120}[TIER]
GEOM_N = {"smoke": 2000, "mid": 4000, "target": 8000, "analytical": 2000}[TIER]
N_TRUE_PAIRS = {"smoke": 1200, "mid": 3000, "target": 5000, "analytical": 1200}[TIER]
BATCH = 64
K_CANDS = 10  # ANN candidate budget per record (flat index: the exact-search ceiling)
PREC_TARGET = 0.99  # PLAN §5 primary fixed entity-precision operating point
DIAL_RATES = (0.02, 0.05, 0.1, 0.2)  # battery typo dial (per-char corruption rates)
PRIMARY_SEED = int(cfg.run.seed)

# The TRN-05 factor sets (PLAN §3.1: 4 schemes x 2 missing treatments = 8 cells). The smoke
# slice trains all four schemes at missing=token plus the colval x drop cell; the full
# factorial x seeds runs at the definitive tier (placard below, with measured coefficients).
FULL_FACTORIAL = [(s, m) for s in SCHEMES for m in ("token", "drop")]
SMOKE_SLICE = [(s, "token") for s in SCHEMES] + [("colval", "drop")]
ARMS = FULL_FACTORIAL if TIER in ("mid", "target") else SMOKE_SLICE
SEEDS = [int(s) for s in cfg.run.seeds] if TIER in ("mid", "target") else [PRIMARY_SEED]

# The tiny budget-matched encoder, set through the typed config keys (dotlist-style, exactly
# what `uv run python -m er_lab.run nb=08 model.dim=128 ...` would do).
cfg.merge_with_dotlist(
    [
        "model.kind=scratch_char",
        "model.dim=128",
        "model.layers=2",
        "model.heads=4",
        "model.max_len=128",
        f"train.steps={STEPS}",
        f"train.batch_size={BATCH}",
        "train.loss=infonce",
        "train.miner=inbatch",
        "train.augment=none",
    ]
)
print(f"tier={TIER}: arms={len(ARMS)} x seeds={SEEDS} at {STEPS} optimizer steps each, "
      f"batch {BATCH}; encoder dim={cfg.model.dim} layers={cfg.model.layers} "
      f"heads={cfg.model.heads} max_len={cfg.model.max_len}")
print(f"eval protocol: ANN flat k={K_CANDS}, precision@{PREC_TARGET} (loud fallback), "
      f"entity-unit BCa n_boot={N_BOOT}, grid={GRID}")

# %% [markdown]
# ## 1. The arena: whose records, whose boundaries?
#
# The calibrated corpus (notebook 05) and the MET-07 splits, loaded through the registry at
# this run's tier. Training uses the **entity-disjoint train half**, evaluation the
# **entity-disjoint eval half** — no entity contributes records to both sides (PLAN §5), so
# nothing an encoder memorizes about a person can flatter that person's evaluation. At smoke
# both halves are subsampled **entity-complete** (an entity is never split across the cut —
# splitting one would silently delete true pairs and corrupt every entity-level metric).
#
# Governance note (as in notebook 06): name/place strings display verbatim because the corpus
# is `historical_50k` (public Wikidata historical figures, error-injected). No NC
# person-level value appears anywhere in this notebook.

# %%
t_sec = time.time()
corpus, corpus_meta = registry.load("calibrated_corpus", tier=cfg.run.tier)
splits, splits_meta = registry.load("met07_splits", tier=cfg.run.tier)
CORPUS_PROVENANCE = (
    f"calibrated_corpus run {corpus_meta['created_at']} (cfg {corpus_meta['config_hash']}, "
    f"tier {corpus_meta['tier']}, seed {corpus_meta['extra']['seed']})"
)
print(f"corpus: {CORPUS_PROVENANCE}")
print(f"  {len(corpus):,} records, {corpus['entity_id'].nunique():,} entities")

SCHEME_SPLIT = "entity_disjoint"
train_ids = set(splits["schemes"][SCHEME_SPLIT]["train"])
eval_ids = set(splits["schemes"][SCHEME_SPLIT]["eval"])
straddle = splits["metadata"]["checks"][SCHEME_SPLIT]["entities_straddling"]
train_full = corpus[corpus["record_id"].isin(train_ids)].reset_index(drop=True)
eval_full = corpus[corpus["record_id"].isin(eval_ids)].reset_index(drop=True)
print(f"split scheme: {SCHEME_SPLIT} (split seed {splits['metadata']['seed']}, "
      f"entities straddling the boundary: {straddle})")


def entity_complete_subsample(
    frame: pd.DataFrame, n_records: int | None, rng: np.random.Generator
) -> pd.DataFrame:
    """Random whole entities until ~n_records — never splits an entity (notebook 05's rule)."""
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


# TEXT_ROLES: the text-bearing role columns every arm serializes, in a fixed order shared by
# training and evaluation. full_name is EXCLUDED, a measured decision: (a) it is a verbatim
# join of the name parts and roughly doubles serialized length, which at this notebook's
# max_len=128 budget would push every scheme into truncation; (b) notebook 05 documents that
# nickname/typo edits leave full_name stale ('full_name_sync' in the corpus meta), so keeping
# it would let a serialization partially UNDO the injected noise — leakage of the clean name
# through a side channel. The field SET is held identical across all arms (the factor under
# test is how fields are rendered, never which fields exist).
TEXT_ROLES = ["given_name", "family_name", "dob", "city", "zip", "sex"]
KEEP_COLS = ["record_id", "entity_id"] + TEXT_ROLES
print(f"\nfull_name excluded from TEXT_ROLES — corpus meta says: "
      f"{corpus_meta['extra']['full_name_sync']!r}")

rng_split = np.random.default_rng(PRIMARY_SEED)
train_sub = entity_complete_subsample(train_full, N_TRAIN, rng_split)[KEEP_COLS]
ev_sub = entity_complete_subsample(eval_full, N_EVAL, rng_split)[KEEP_COLS]
truth_ev = ev_sub.set_index("record_id")["entity_id"]
records_ev = pd.Index(truth_ev.index)
print(f"train: {len(train_sub):,} records / {train_sub['entity_id'].nunique():,} entities "
      f"(entity-complete subsample of {len(train_full):,})")
print(f"eval : {len(ev_sub):,} records / {truth_ev.nunique():,} entities "
      f"(entity-complete subsample of {len(eval_full):,})")
missing_any = ev_sub[TEXT_ROLES].isna().any(axis=1).mean()
print(f"eval rows with >= 1 missing text role: {missing_any:.1%} "
      f"(the {MISSING_TOKEN}-vs-drop factor has real mass to act on)")
SECTION_SECS["1 arena"] = time.time() - t_sec

# %% [markdown]
# ## 2. What does the encoder actually see?
#
# Before any conjecture: the object under study, verbatim. `er_lab.serialize` is the single
# code path that turns a record into the encoder's input string — four schemes × two
# missing-field treatments. Below, two real eval-split records (one complete, one with
# missing fields) under all eight combinations. Read them the way a byte-level encoder does:
# `colval` spends ~13 bytes of scaffolding per field to *name* what each value is; `template`
# and `json` name fields more cheaply; `bare` spends nothing and leaves the model to guess
# whether `sy10 8as` is a zip or a street. With `missing='token'` an empty field stays
# visible as `[MISSING]`; with `'drop'` the field silently vanishes — and under `bare`, the
# fields after it *shift left* into its place.

# %%
t_sec = time.time()
complete_mask = ev_sub[TEXT_ROLES].notna().all(axis=1)
missing_mask = ev_sub[TEXT_ROLES].isna().sum(axis=1) >= 2
row_complete = ev_sub[complete_mask].iloc[0]
row_missing = ev_sub[missing_mask].iloc[0]

for label, row in (("COMPLETE record", row_complete), ("RECORD WITH MISSING FIELDS", row_missing)):
    print(f"=== {label}: record_id={row['record_id']} ===")
    print("   raw cells:", {r: (None if pd.isna(row[r]) else row[r]) for r in TEXT_ROLES})
    for missing_treatment in ("token", "drop"):
        for scheme in SCHEMES:
            text = serialize_record(
                row, text_roles=TEXT_ROLES, scheme=scheme, missing=missing_treatment
            )
            print(f"   {scheme:>8s}/{missing_treatment:<5s} ({len(text.encode()):3d}B): {text}")
    print()

# %% [markdown]
# ### The context-budget census — and a confound flagged before it can bite
#
# The tokenizer truncates every input at `max_len - 1 = 127` UTF-8 bytes (one slot goes to
# BOS). Serialization schemes differ enormously in how much of that budget they spend on
# scaffolding, so at this notebook's tiny smoke budget the schemes do not just *format*
# differently — they *fit* differently. The census below measures it. This is a real smoke
# confound for TRN-05 and it is pre-flagged in the conjecture card rather than discovered
# after the results: at `max_len=128` every `colval` string truncates, losing its trailing
# fields, while `bare` fits whole. The definitive factorial runs at the lab default
# `max_len=192` where the marked schemes fit.

# %%
census_rows = []
for scheme in SCHEMES:
    for missing_treatment in ("token", "drop"):
        blens = np.array([
            len(t.encode())
            for t in serialize_frame(
                ev_sub, text_roles=TEXT_ROLES, scheme=scheme, missing=missing_treatment
            )
        ])
        census_rows.append({
            "scheme": scheme, "missing": missing_treatment,
            "bytes_mean": float(blens.mean()), "bytes_p95": float(np.percentile(blens, 95)),
            "frac_truncated@127B": float((blens > 127).mean()),
            "bytes_lost_mean": float(np.maximum(blens - 127, 0).mean()),
        })
census = pd.DataFrame(census_rows)
display(census.round(3))

trunc_demo = serialize_record(row_complete, text_roles=TEXT_ROLES, scheme="colval",
                              missing="token")
print("what the encoder ACTUALLY sees of the colval/token complete record "
      f"(first 127 of {len(trunc_demo.encode())} bytes):")
print("   " + trunc_demo.encode()[:127].decode(errors="replace"))
print("   ...lost: " + trunc_demo.encode()[127:].decode(errors="replace"))
COLVAL_TRUNC_FRAC = float(
    census.set_index(["scheme", "missing"]).loc[("colval", "token"), "frac_truncated@127B"]
)
COLVAL_BYTES_LOST = float(
    census.set_index(["scheme", "missing"]).loc[("colval", "token"), "bytes_lost_mean"]
)
print(f"\nmeasured: colval/token truncates on {COLVAL_TRUNC_FRAC:.0%} of eval rows "
      f"(mean {COLVAL_BYTES_LOST:.0f} bytes lost — typically the zip/sex tail); "
      "bare/template fit almost everywhere. Carried into the TRN-05 card as a "
      "pre-flagged smoke risk.")
SECTION_SECS["2 serialization walkthrough"] = time.time() - t_sec

# %% [markdown]
# ## 3. The conjectures, before any encoder exists
#
# Two cards, both rendered and immutably registered BEFORE a single optimizer step runs
# (PLAN §5). TRN-05 is the serialization pressure; NB08-BATTERY pre-registers what the
# battery/geometry debut should show about training itself. One discipline note both cards
# repeat: a pressure moving its property AND its metric is the **adoption-gate** shape — it
# is *not* mediation, which needs indirect-effect estimation across arms × seeds × noise
# draws (PLAN §5). Nothing in a single-seed smoke run can establish mediation, and the
# verdicts below say so.

# %%
_ = conjecture_card(
    card_id="TRN-05",
    conjecture=(
        "How a person record becomes the encoder's input string is a real training "
        "pressure, not plumbing: at matched optimizer-step budget, schemes that mark field "
        "identity (colval, template, json) beat bare value concatenation, and a visible "
        "[MISSING] token beats silently dropping empty fields."
    ),
    pressure=(
        "serialization scheme {colval, template, json, bare} x missing-field treatment "
        "{token, drop} — the PLAN §3.1 locked 8-cell factorial; this notebook's smoke slice "
        "trains 5 budget-matched cells (all four schemes at missing=token, plus colval x "
        "drop), scratch char/byte encoder, with loss=infonce, miner=inbatch, augment=none "
        "and the field set held fixed across arms"
    ),
    property=(
        "true-pair alignment and the invariance-battery slice profile (typo dial, "
        "field-order permutation, missing-field visibility) plus the geometry panel "
        "(uniformity, RankMe, hubness), measured on entity-disjoint eval embeddings"
    ),
    metric=(
        "B-cubed F1 (entity-unit BCa 95% CI) at the fixed 0.99 entity-precision operating "
        "point — loud highest-attainable fallback per PLAN §5, never a silent protocol "
        "switch — over ANN flat k=10 cosine candidates clustered by transitive closure, on "
        "the entity-disjoint eval split; registered in trn05_serialization_matrix"
    ),
    prediction=(
        "P1: colval/token beats bare/token on B-cubed F1 at the operating point, with a "
        "shared-draw paired entity-bootstrap delta excluding zero. P2: within colval, token "
        "beats drop on the same test. P3 (descriptive): every field-marked scheme at "
        "missing=token ranks above bare/token. Pre-flagged smoke risk: at max_len=128 every "
        "colval string truncates (measured in this notebook's census), so the marked-scheme "
        "advantage may be capped or reversed at smoke by the context budget alone; the "
        "binding test is the multi-seed 8-cell factorial at the definitive tier with the "
        "lab-default max_len=192. Single-seed smoke outcomes are demonstrations, and "
        "mediation through the named properties is expected to remain UNEXPLAINED here."
    ),
    registry=registry,
)

# %%
_ = conjecture_card(
    card_id="NB08-BATTERY",
    conjecture=(
        "'A good person-embedding' is a measurable property set, and training is what buys "
        "it: contrastive training reorganizes similarity so that same-person variants stay "
        "close WHILE different-person confusables separate — a contrast an untrained "
        "same-architecture encoder cannot show, because a randomly-initialized smooth "
        "encoder ranks nearly-identical strings close regardless of who they denote."
    ),
    pressure=(
        "training itself, at fixed architecture and serialization: the winning TRN-05 "
        "arm's encoder after its full optimizer-step budget vs its own "
        "identically-initialized untrained twin (0 steps), same eval split, same probe set"
    ),
    property=(
        "the invariance battery (must_not_hold separation AUC vs true pairs on "
        "twin_household / different_birth_year / suffix_jr_sr; should_hold cosine under "
        "the typo dial) and geometry (uniformity, true-pair alignment) on entity-disjoint "
        "eval embeddings"
    ),
    metric=(
        "the shared system-metric recipe (B-cubed F1 at the 0.99 entity-precision point "
        "with loud fallback, ANN flat k=10, transitive closure, entity-unit BCa CI), "
        "trained vs untrained"
    ),
    prediction=(
        "P1: the untrained twin shows no usable confusable separation — every non-empty "
        "must_not_hold AUC at or below 0.55, because raw string geometry ranks "
        "near-identical different-person pairs above genuinely varying same-person pairs. "
        "P2: training pushes uniformity strictly more negative (the space spreads) and "
        "raises the AUC of every non-empty must_not_hold slice. P3: the system metric "
        "follows — trained beats untrained on B-cubed F1 with a shared-draw paired delta "
        "excluding zero. RankMe and hubness are reported as exploratory descriptors with "
        "no pre-registered direction. Property and metric moving together is the "
        "adoption-gate shape, NOT mediation (PLAN §5); with one seed the mediation status "
        "is expected UNEXPLAINED."
    ),
    registry=registry,
)

# %% [markdown]
# ## 4. Five encoders, one budget
#
# Every arm runs the one training loop (`er_lab.train.loop.train_encoder`) with everything
# held fixed except `serialize.scheme` / `serialize.missing`: same corpus, same InfoNCE loss,
# same in-batch miner, no augmentation, same learning rate, same batch size — and, by
# construction, the **same initial weights** (the encoder factory is seeded identically per
# arm; the init hash is asserted below, so "the arms differ only in serialization" is
# checkable, not asserted). Budget-matching is the identical optimizer-step count, asserted
# from each returned history (PLAN §3.1).

# %%
t_sec = time.time()


def arm_config(scheme: str, missing_treatment: str, extra: tuple[str, ...] = ()):
    """Per-arm config: the shared cfg plus this arm's serialization dotlist."""
    return OmegaConf.merge(
        cfg,
        OmegaConf.from_dotlist(
            [f"serialize.scheme={scheme}", f"serialize.missing={missing_treatment}", *extra]
        ),
    )


def param_hash(module) -> str:
    """12-hex digest of all parameters in state_dict order — init-identity checks."""
    h = hashlib.sha256()
    for tensor in module.state_dict().values():
        h.update(tensor.detach().cpu().numpy().tobytes())
    return h.hexdigest()[:12]


encoders: dict[tuple[str, str, int], object] = {}
histories: list[pd.DataFrame] = []
train_secs: dict[tuple[str, str, int], float] = {}
init_hashes: set[str] = set()

for scheme, missing_treatment in ARMS:
    for seed in SEEDS:
        a_cfg = arm_config(scheme, missing_treatment)
        set_all_seeds(seed)  # identical init across arms at the same seed
        enc = build_encoder(a_cfg)
        init_hashes.add(param_hash(enc))
        t0 = time.time()
        enc, hist = train_encoder(
            enc, train_sub, a_cfg,
            loss_name="infonce", miner_name="inbatch", augment_kind="none",
            steps=STEPS, seed=seed,
        )
        dt = time.time() - t0
        key = (scheme, missing_treatment, seed)
        encoders[key] = enc
        train_secs[key] = dt
        histories.append(hist.assign(scheme=scheme, missing=missing_treatment, seed=seed))
        print(f"[{scheme:>8s}/{missing_treatment:<5s} seed={seed}] {len(hist)} steps in "
              f"{dt:5.1f}s ({dt / len(hist):.3f}s/step) "
              f"loss {hist['loss'].iloc[0]:.3f} -> {hist['loss'].tail(20).mean():.3f}")

N_PARAMS = sum(p.numel() for p in next(iter(encoders.values())).parameters())
SECTION_SECS["4 training"] = time.time() - t_sec
print(f"\nencoder parameter count: {N_PARAMS:,}")

# %%
# The budget-match assertion — from the returned histories, not from intent.
hist_all = pd.concat(histories, ignore_index=True)
step_counts = hist_all.groupby(["scheme", "missing", "seed"])["step"].agg(["size", "max"])
assert (step_counts["size"] == STEPS).all() and (step_counts["max"] == STEPS).all(), step_counts
assert len(set(SEEDS)) > 1 or len(init_hashes) == 1, (
    f"arms at one seed must share one init, got {init_hashes}"
)
print(f"budget check PASSED: every arm executed exactly {STEPS} optimizer steps "
      f"(asserted from history rows: {sorted(step_counts['size'].unique())})")
print(f"init check PASSED: {len(init_hashes)} distinct init hash(es) across "
      f"{len(ARMS)} arms x {len(SEEDS)} seed(s): {sorted(init_hashes)}")

registry.register(
    "trn05_train_histories", hist_all, cfg=cfg, tier=cfg.run.tier,
    meta={
        "note": "per-step training loss for every TRN-05 arm; identical step budget "
                "asserted (PLAN §3.1 step-count accounting)",
        "steps": STEPS, "batch_size": BATCH, "seeds": SEEDS,
        "init_hashes": sorted(init_hashes), "n_params": int(N_PARAMS),
        "train_records": len(train_sub), "corpus_provenance": CORPUS_PROVENANCE,
    },
)


def draw_losses(ax, df, meta):
    for (scheme, missing_treatment), g in df[df["seed"] == df["seed"].iloc[0]].groupby(
        ["scheme", "missing"]
    ):
        smooth = g.sort_values("step")["loss"].rolling(15, min_periods=1).mean()
        ax.plot(g.sort_values("step")["step"], smooth, linewidth=1.2,
                label=f"{scheme}/{missing_treatment}")
    ax.set_xlabel(f"optimizer step (identical budget: {int(df['step'].max())} steps/arm)")
    ax.set_ylabel("InfoNCE loss (rolling mean, 15 steps)")
    ax.legend()


fig = figures.plot_artifact(
    registry, tier=cfg.run.tier, artifact="trn05_train_histories", draw=draw_losses,
    title="One budget, five serializations - every arm gets the same optimizer steps",
    figsize=(6.8, 4.2),
)

# %% [markdown]
# ## 5. Who won — under the lab's protocol, not a leaderboard's
#
# The **shared system-metric recipe**, used for every encoder in this notebook and reused
# verbatim by notebooks 09–11: serialize the entity-disjoint eval split with the arm's own
# scheme → encode → ANN candidates at a matched budget (`flat` index, k=10 — exact search,
# so no ANN-recall noise enters the comparison) → cosine scores → the fixed
# `precision@0.99` operating point via the exhaustive sweep (loud highest-attainable
# fallback when a smoke-sized eval set can't reach it — PLAN §5's rail, never a silent
# protocol switch) → transitive closure → B-cubed with an **entity-unit BCa 95% CI**.
# One honest asymmetry vs notebook 06: here each arm *retrieves its own candidates* — the
# serialization changes the embedding space and therefore the candidate set, so the arms are
# compared as whole retrieve+score systems at matched candidate budget (BAS-02's rule), and
# each arm's ANN pair-completeness (its recall ceiling) is recorded in the matrix.

# %%
t_sec = time.time()


def system_point(encoder, a_cfg, *, seed: int) -> tuple[dict, pd.Series, np.ndarray]:
    """The shared recipe: encoder + serialization cfg -> operating-point row + prediction.

    Returns (row, pred, emb): row carries the B-cubed point + entity-unit BCa CI at the
    precision@PREC_TARGET operating point (fallback flagged loudly), pred the clustering
    at that threshold (for shared-draw paired deltas between systems), emb the eval
    embeddings (reused by the geometry panel — encoded exactly once per encoder).
    """
    scheme = str(a_cfg.serialize.scheme)
    missing_treatment = str(a_cfg.serialize.missing)
    t0 = time.time()
    texts = serialize_frame(
        ev_sub, text_roles=TEXT_ROLES, scheme=scheme, missing=missing_treatment
    ).tolist()
    emb = encoder.encode(texts, batch_size=BATCH)
    encode_secs = time.time() - t0

    cand = ann.candidates(ev_sub, emb, k=K_CANDS, index="flat")
    bstats = blocking_metrics(cand, truth_ev, n_records=len(ev_sub))
    pos = {rid: i for i, rid in enumerate(ev_sub["record_id"])}
    ia = cand["a"].map(pos).to_numpy(dtype=int)
    ib = cand["b"].map(pos).to_numpy(dtype=int)
    cos = np.einsum("ij,ij->i", emb[ia], emb[ib])
    scored = pd.DataFrame(
        {"a": cand["a"], "b": cand["b"], "score": np.clip((cos + 1.0) / 2.0, 0.0, 1.0)}
    )

    pred_cache: dict[float, pd.Series] = {}

    def clusterer(thr: float) -> pd.Series:
        if thr not in pred_cache:
            pred_cache[thr] = transitive_closure(
                scored.rename(columns={"score": "prob"}), threshold=float(thr),
                records=records_ev,
            )
        return pred_cache[thr]

    op = find_threshold_for_precision(
        scored, clusterer, truth_ev, target=PREC_TARGET, grid=GRID
    )
    if not op["attained"]:
        print(f"    !! precision@{PREC_TARGET} UNATTAINABLE for {scheme}/{missing_treatment}: "
              f"fallback='{op['fallback']}' at attained P={op['attained_precision']:.4f} "
              "(PLAN §5 rail: reported loudly, protocol never switched)")
    pred = clusterer(op["threshold"])
    b3 = bcubed(pred, truth_ev)
    ci = bootstrap_ci(
        pred, truth_ev, "bcubed_f1", unit="entity", n_boot=N_BOOT, seed=PRIMARY_SEED
    )
    row = {
        "scheme": scheme, "missing": missing_treatment, "seed": seed, "steps": STEPS,
        "bcubed_f1": b3["f1"], "f1_lo": ci["ci_low"], "f1_hi": ci["ci_high"],
        "bcubed_precision": b3["precision"], "bcubed_recall": b3["recall"],
        "threshold_prob": op["threshold"], "threshold_cos": 2.0 * op["threshold"] - 1.0,
        "attained": float(op["attained"]), "fallback": op["fallback"] or "",
        "attained_precision": op["attained_precision"],
        "n_candidate_pairs": len(cand),
        "pair_completeness": float(bstats["pair_completeness"]),
        "encode_secs": encode_secs,
    }
    return row, pred, emb


matrix_rows: list[dict] = []
preds: dict[tuple[str, str, int], pd.Series] = {}
arm_embeddings: dict[tuple[str, str, int], np.ndarray] = {}
for (scheme, missing_treatment, seed), enc in encoders.items():
    a_cfg = arm_config(scheme, missing_treatment)
    row, pred, emb = system_point(enc, a_cfg, seed=seed)
    row["train_secs"] = train_secs[(scheme, missing_treatment, seed)]
    matrix_rows.append(row)
    preds[(scheme, missing_treatment, seed)] = pred
    arm_embeddings[(scheme, missing_treatment, seed)] = emb
    flag = "" if row["attained"] else f"  FALLBACK({row['attained_precision']:.3f})"
    print(f"[{scheme:>8s}/{missing_treatment:<5s} seed={seed}] "
          f"F={row['bcubed_f1']:.4f} [{row['f1_lo']:.4f},{row['f1_hi']:.4f}] "
          f"P={row['bcubed_precision']:.4f} R={row['bcubed_recall']:.4f} "
          f"t_cos={row['threshold_cos']:.3f} PC={row['pair_completeness']:.3f}{flag}")

matrix = pd.DataFrame(matrix_rows)
SECTION_SECS["5 system metric"] = time.time() - t_sec

# %%
registry.register(
    "trn05_serialization_matrix", matrix, cfg=cfg, tier=cfg.run.tier,
    meta={
        "protocol": {
            "recipe": "serialize eval per arm -> encode -> ann flat k=10 -> cosine -> "
                      f"precision@{PREC_TARGET} operating point (loud fallback) -> "
                      "transitive closure -> bcubed + entity-unit BCa CI on bcubed_f1",
            "grid": GRID, "n_boot": N_BOOT, "k": K_CANDS,
            "candidate_sets": "per-arm (the embedding owns retrieval; matched k budget)",
        },
        "budget_match": {"steps": STEPS, "batch_size": BATCH,
                         "asserted_from_history": True, "init_hashes": sorted(init_hashes)},
        "arms": [f"{s}/{m}" for s, m in ARMS], "seeds": SEEDS,
        "smoke_slice_note": "4 schemes x token + colval x drop at smoke; the full 8-cell "
                            "factorial x seeds is the definitive-tier run (placard)",
        "split": {"scheme": SCHEME_SPLIT, "seed": splits["metadata"]["seed"],
                  "n_eval_records": len(ev_sub),
                  "n_eval_entities": int(truth_ev.nunique()),
                  "n_train_records": len(train_sub)},
        "truncation_census": census.to_dict("records"),
        "text_roles": TEXT_ROLES,
        "corpus_provenance": CORPUS_PROVENANCE,
    },
)
cell_view = (
    matrix.groupby(["scheme", "missing"], sort=False)
    .agg(f1=("bcubed_f1", "mean"), f1_lo=("f1_lo", "mean"), f1_hi=("f1_hi", "mean"),
         recall=("bcubed_recall", "mean"), attained=("attained", "min"),
         pair_completeness=("pair_completeness", "mean"), seeds=("seed", "size"))
    .sort_values("f1", ascending=False)
)
print(f"registered trn05_serialization_matrix ({len(matrix)} arm rows)")
display(cell_view.round(4))

WIN_SCHEME, WIN_MISSING = cell_view.sort_values(
    ["attained", "f1"], ascending=False
).index[0]
WIN_KEY = (WIN_SCHEME, WIN_MISSING, PRIMARY_SEED)
print(f"winning arm at this tier: {WIN_SCHEME}/{WIN_MISSING} "
      f"(prefer attained targets, then F1) — carried into the battery and geometry sections")


# %%
def draw_matrix(ax, df, meta):
    g = (
        df.groupby(["scheme", "missing"], sort=False)
        .agg(f1=("bcubed_f1", "mean"), lo=("f1_lo", "mean"), hi=("f1_hi", "mean"),
             attained=("attained", "min"))
        .reset_index()
        .sort_values("f1")
    )
    y = np.arange(len(g))
    colors = g["missing"].map({"token": "C0", "drop": "C3"})
    xerr = np.vstack([
        np.clip(g["f1"] - g["lo"], 0, None), np.clip(g["hi"] - g["f1"], 0, None)
    ])
    for yi, (_, r), c in zip(y, g.iterrows(), colors):
        filled = bool(r["attained"])
        ax.errorbar(r["f1"], yi, xerr=xerr[:, [yi]], fmt="o", capsize=3, markersize=7,
                    color=c, markerfacecolor=c if filled else "white")
    ax.set_yticks(y, [f"{s}/{m}" for s, m in zip(g["scheme"], g["missing"])])
    ax.set_xlabel(f"B-cubed F1 at precision@{PREC_TARGET} "
                  "(open marker = target unattained, highest-attainable fallback)")
    for c, lab in (("C0", "missing=token"), ("C3", "missing=drop")):
        ax.errorbar([], [], fmt="o", color=c, label=lab)
    ax.legend(loc="lower right")


fig = figures.plot_artifact(
    registry, tier=cfg.run.tier, artifact="trn05_serialization_matrix", draw=draw_matrix,
    title="TRN-05 smoke slice - serialization arms at one matched budget",
    figsize=(6.8, 4.0),
)

# %%
# [RUN-IN-TARGET node] definitive TRN-05: the full 8-cell factorial x run.seeds at the
# definitive tier (PLAN §3 puts TRN-05 at mid; the multi-seed factorial sweep belongs to the
# node per PLAN §8). Same cells, same code path — ARMS/SEEDS switch on tier above.
if TIER in ("mid", "target"):
    print(f"tier={TIER}: the matrix above IS the definitive {len(ARMS)}-cell x "
          f"{len(SEEDS)}-seed TRN-05 run for this tier.")
else:
    per_arm_train = float(np.mean(list(train_secs.values())))
    per_arm_eval = float(matrix["encode_secs"].mean())
    steps_ratio = 2000 / STEPS  # the mid/target step budget over this run's
    eval_ratio = len(eval_full) / len(ev_sub)
    est_min = (
        8 * len(list(cfg.run.seeds))
        * (per_arm_train * steps_ratio + per_arm_eval * eval_ratio)
    ) / 60
    print("[RUN-IN-TARGET node] definitive TRN-05 = 8 cells x seeds "
          f"{[int(s) for s in cfg.run.seeds]} at tier=mid (steps=2000, max_len=192, full "
          "splits), on the node via these same cells (ARMS/SEEDS switch on tier). "
          f"Measured smoke coefficients: {per_arm_train:.0f}s train/arm at {STEPS} steps "
          f"({per_arm_train / STEPS:.2f}s/step on this 4-CPU container), "
          f"{per_arm_eval:.0f}s eval-encode/arm at {len(ev_sub):,} records. Naive "
          f"CPU-equivalent estimate: 8 x {len(list(cfg.run.seeds))} x "
          f"({per_arm_train:.0f}s x {steps_ratio:.0f} + {per_arm_eval:.0f}s x "
          f"{eval_ratio:.1f}) ~= {est_min:.0f} min; the node's GPUs cut the training term "
          "by orders of magnitude (PLAN §8), so this is an upper bound re-measured on "
          "first node run.")

# %%
# [RUN-IN-TARGET mac] pretrained-subword spot-check (TRN-04's regime factor grazing TRN-05):
# does the serialization ranking transfer to a pretrained subword encoder? Real code, gated
# on weight availability per notes/COMPAT.md — huggingface.co is blocked from the smoke
# container, so this branch runs wherever weights are reachable (mac/node have normal
# internet) or when paths.hf_local points at a local copy (loaded offline).
PRETRAINED_AVAILABLE = cfg.paths.hf_local is not None or TIER in ("mid", "target")
if PRETRAINED_AVAILABLE:
    pret_rows = []
    for scheme, missing_treatment in (("colval", "token"), ("bare", "token")):
        p_cfg = arm_config(scheme, missing_treatment, extra=("model.kind=pretrained",))
        set_all_seeds(PRIMARY_SEED)
        enc_p = build_encoder(p_cfg)  # raises with the COMPAT fallback text if unreachable
        enc_p, hist_p = train_encoder(
            enc_p, train_sub, p_cfg,
            loss_name="infonce", miner_name="inbatch", augment_kind="none",
            steps=STEPS, seed=PRIMARY_SEED,
        )
        assert len(hist_p) == STEPS  # same budget-match rail as the scratch arms
        row_p, _pred_p, _emb_p = system_point(enc_p, p_cfg, seed=PRIMARY_SEED)
        pret_rows.append({**row_p, "model_kind": "pretrained"})
        print(f"[pretrained {scheme}/{missing_treatment}] F={row_p['bcubed_f1']:.4f} "
              f"[{row_p['f1_lo']:.4f},{row_p['f1_hi']:.4f}]")
    registry.register(
        "trn05_pretrained_spotcheck", pd.DataFrame(pret_rows), cfg=cfg, tier=cfg.run.tier,
        meta={"note": "pretrained-subword regime spot-check on two TRN-05 cells; the "
                      "regime factor itself is TRN-04 (notebook 10)",
              "steps": STEPS, "recipe": "same shared system-metric recipe"},
    )
else:
    minilm_params = 22_700_000  # all-MiniLM-L6-v2, for the coefficient ratio only
    print("[RUN-IN-TARGET mac] pretrained-subword spot-check skipped: no weights reachable "
          f"at tier={TIER} (huggingface.co blocked, paths.hf_local unset — COMPAT.md "
          "go/no-go tree). On the mac this cell trains all-MiniLM-L6-v2 on 2 cells at the "
          f"same {STEPS}-step budget. Measured coefficient: scratch arm "
          f"({N_PARAMS:,} params) trains in {np.mean(list(train_secs.values())):.0f}s "
          f"here; MiniLM is ~{minilm_params / N_PARAMS:.0f}x the parameters, so a CPU run "
          "would be ~that factor slower — on MPS expect minutes/arm (PLAN §8 planning "
          "ratio, +-2x; re-measured on first mac run).")

# %% [markdown]
# ### Verdict — scoring TRN-05's smoke slice against the card
#
# The two pre-registered contrasts get **shared-draw paired entity-bootstrap deltas**
# (`eval.bootstrap.paired_delta`): both systems are evaluated on the same entity resamples,
# so entity-sampling noise common to both cancels — the design that makes small deltas
# readable at all on a smoke-sized split. Verdict mapping, fixed before looking: both deltas
# sign-stable in the predicted direction → CONFIRMED *as a single-seed demonstration*; any
# sign-stable reversal → REFUTED at smoke; anything else → UNEXPLAINED (one seed cannot
# settle it).

# %%
d_colval_bare = paired_delta(
    preds[("colval", "token", PRIMARY_SEED)], preds[("bare", "token", PRIMARY_SEED)],
    truth_ev, "bcubed_f1", unit="entity", n_boot=N_BOOT, seed=PRIMARY_SEED,
)
d_token_drop = paired_delta(
    preds[("colval", "token", PRIMARY_SEED)], preds[("colval", "drop", PRIMARY_SEED)],
    truth_ev, "bcubed_f1", unit="entity", n_boot=N_BOOT, seed=PRIMARY_SEED,
)
token_cells = cell_view.loc[[(s, "token") for s in SCHEMES]]
bare_f1 = float(token_cells.loc[("bare", "token"), "f1"])
p3_rank = bool(all(
    float(token_cells.loc[(s, "token"), "f1"]) > bare_f1
    for s in ("colval", "template", "json")
))

print(f"P1 colval/token - bare/token   : delta={d_colval_bare['delta']:+.4f} "
      f"[{d_colval_bare['ci_low']:+.4f}, {d_colval_bare['ci_high']:+.4f}] "
      f"sign_stable={d_colval_bare['sign_stable']}")
print(f"P2 colval/token - colval/drop  : delta={d_token_drop['delta']:+.4f} "
      f"[{d_token_drop['ci_low']:+.4f}, {d_token_drop['ci_high']:+.4f}] "
      f"sign_stable={d_token_drop['sign_stable']}")
print(f"P3 all field-marked token arms above bare/token: {p3_rank}")

p1_conf = bool(d_colval_bare["sign_stable"] and d_colval_bare["delta"] > 0)
p1_refu = bool(d_colval_bare["sign_stable"] and d_colval_bare["delta"] < 0)
p2_conf = bool(d_token_drop["sign_stable"] and d_token_drop["delta"] > 0)
p2_refu = bool(d_token_drop["sign_stable"] and d_token_drop["delta"] < 0)
if p1_conf and p2_conf:
    trn05_outcome = "CONFIRMED"
elif p1_refu or p2_refu:
    trn05_outcome = "REFUTED"
else:
    trn05_outcome = "UNEXPLAINED"

fallback_arms = matrix.loc[matrix["attained"] == 0.0, ["scheme", "missing"]]
fallback_note = (
    "none" if fallback_arms.empty
    else ", ".join(f"{s}/{m}" for s, m in fallback_arms.itertuples(index=False))
)
_ = verdict_box(
    "TRN-05",
    outcome=trn05_outcome,
    evidence=(
        f"trn05_serialization_matrix (tier {TIER}, {len(ARMS)} arms x {len(SEEDS)} "
        f"seed(s), {STEPS} steps each — asserted; entity-disjoint eval {len(ev_sub):,} "
        f"records / {truth_ev.nunique():,} entities). P1 shared-draw paired delta "
        f"colval/token - bare/token = {d_colval_bare['delta']:+.4f} "
        f"[{d_colval_bare['ci_low']:+.4f}, {d_colval_bare['ci_high']:+.4f}] "
        f"(sign_stable={d_colval_bare['sign_stable']}); P2 colval/token - colval/drop = "
        f"{d_token_drop['delta']:+.4f} [{d_token_drop['ci_low']:+.4f}, "
        f"{d_token_drop['ci_high']:+.4f}] (sign_stable={d_token_drop['sign_stable']}); "
        f"P3 rank check = {p3_rank}. Winner this tier: {WIN_SCHEME}/{WIN_MISSING} "
        f"(B-cubed F {float(cell_view.loc[(WIN_SCHEME, WIN_MISSING), 'f1']):.4f}). "
        f"Precision@{PREC_TARGET} fallback fired for: {fallback_note}. Pre-flagged "
        f"context-budget confound was real: colval truncates on "
        f"{COLVAL_TRUNC_FRAC:.0%} of eval rows at max_len=128 (census in the matrix "
        f"meta), so smoke under/over-states the marked-scheme effect. SCOPE: single-seed "
        f"smoke DEMONSTRATION — the definitive verdict needs the multi-seed 8-cell "
        f"factorial at mid/node (placard above). Mediation through the named properties: "
        f"NOT ESTABLISHED here — property and metric shifts are two marginal effects "
        f"(adoption-gate evidence at best), and one seed supports no indirect-effect "
        f"estimate (PLAN §5)."
    ),
    registry=registry,
)

# %% [markdown]
# ## 6. The invariance battery: a report card for "good"
#
# The battery (`er_lab.probes.battery`) is the lab's operationalization of "a good embedding
# for ER". From real eval-split records it builds probe pairs in two families:
# **should_hold** slices pair a record's serialization with a same-person variant the encoder
# must keep close — typos at four dialed intensities, nickname swaps (carltonnorthern
# lexicon), field-order permutations, format drift. **must_not_hold** slices pair it with a
# different-person confusable it must keep apart — same record one-to-three birth-years off,
# Jr vs Sr, and the twin/household trap (different given name, same family/DOB/place). The
# battery *measures only* — per-slice cosine statistics and, for confusables, the
# Mann-Whitney AUC of separation against genuine same-person pairs; verdicts live here in
# the notebook, against the pre-registered card.
#
# Two deliberate wrinkles, on the record: (a) the probe serializer uses the **winning arm's
# scheme** but honors the probe's field order, so the `field_order_permutation` slice is a
# live eval-time probe (exactly TRN-05's locked design: order sensitivity is probed at eval,
# never trained); (b) `suffix_jr_sr` injects a `name_suffix` role the corpus never carries —
# the battery may ask questions the training distribution never posed, and a low score there
# is information, not a bug. The nickname lexicon needs notebook 04/05's sanitization
# workaround for the upstream `names.csv` format drift (`load_lexicon` mis-parses it) — a
# package gap carried on the record, fixed in-notebook.

# %%
t_sec = time.time()
lexicon_csv = fetch_lexicon(data_root=str(REPO_ROOT / "data"))
lexicon_raw = load_lexicon(str(REPO_ROOT / "data"))
drifted = "name1" in lexicon_raw or any("has_nickname" in v for v in lexicon_raw.values())
lexicon = {
    canon: {v for v in variants if v not in ("has_nickname", "relationship")}
    for canon, variants in lexicon_raw.items()
    if canon != "name1"
}
lexicon = {c: v for c, v in lexicon.items() if v}
print(f"lexicon: {lexicon_csv} — upstream format drift detected: {drifted}; sanitized to "
      f"{len(lexicon)} canonical names (English-centric; provenance bias carried, PLAN §4)")

PROBE_ROLE_SET = set(TEXT_ROLES) | {"name_suffix"}


def probe_serialize(d: dict) -> str:
    """The winning arm's serialization, honoring the probe's own field order.

    Keeps the field-order slice live (serialize_record itself is order-fixed by
    text_roles) and admits the battery's injected name_suffix role.
    """
    roles = [k for k in d if k in PROBE_ROLE_SET]
    return serialize_record(d, text_roles=roles, scheme=WIN_SCHEME, missing=WIN_MISSING)


probes = build_probe_set(
    ev_sub[["record_id"] + TEXT_ROLES],
    serialize=probe_serialize,
    seed=PRIMARY_SEED,
    lexicon=lexicon,
    dial_rates=DIAL_RATES,
    n_per_slice=PROBE_N,
)
print(f"probe set: {len(probes):,} pairs across {probes['slice'].nunique()} slices "
      f"(<= {PROBE_N}/slice, serialized as {WIN_SCHEME}/{WIN_MISSING})")
display(probes.groupby(["kind", "slice"], sort=False).size().rename("n_pairs").to_frame())

# genuine same-person pairs (two records of one entity), for the separation AUC
rng_pairs = np.random.default_rng(PRIMARY_SEED)
pair_pos: list[tuple[int, int]] = []
for _eid, g in ev_sub.groupby("entity_id", sort=False):
    idx = g.index.to_numpy()
    if len(idx) >= 2:
        take = min(3, len(idx) - 1)
        for j in range(take):
            a, b = rng_pairs.choice(idx, size=2, replace=False)
            if a != b:
                pair_pos.append((int(a), int(b)))
rng_pairs.shuffle(pair_pos)
pair_pos = pair_pos[:N_TRUE_PAIRS]
true_pairs = pd.DataFrame({
    "text_a": [probe_serialize(dict(ev_sub.loc[a, TEXT_ROLES])) for a, _ in pair_pos],
    "text_b": [probe_serialize(dict(ev_sub.loc[b, TEXT_ROLES])) for _, b in pair_pos],
})
print(f"true-pair reference set: {len(true_pairs):,} within-entity serialization pairs")

# %% [markdown]
# ### Trained vs untrained: the contrast IS the story
#
# The untrained twin is not a strawman — it is the *same architecture at the same
# initialization* the winning arm started from (asserted by init hash below), encoding the
# same texts. Whatever separates the two report cards is what those optimizer steps bought.

# %%
set_all_seeds(PRIMARY_SEED)
untrained = build_encoder(arm_config(WIN_SCHEME, WIN_MISSING))
if len(SEEDS) == 1:
    assert param_hash(untrained) in init_hashes, "untrained twin must share the arms' init"
    print(f"untrained twin init hash {param_hash(untrained)} == the trained arms' init "
          "(the contrast below is purely the training)")

winner_enc = encoders[WIN_KEY]
ENCODER_LABELS = {"trained": f"trn05:{WIN_SCHEME}/{WIN_MISSING}", "untrained": "untrained"}
reports = []
for label, enc in (("trained", winner_enc), ("untrained", untrained)):
    rep = battery_report(
        lambda texts, e=enc: e.encode(texts, batch_size=BATCH), probes,
        true_pairs=true_pairs,
    )
    reports.append(rep.reset_index().assign(encoder=ENCODER_LABELS[label]))
battery = pd.concat(reports, ignore_index=True)

registry.register(
    "invariance_battery", battery, cfg=cfg, tier=cfg.run.tier,
    meta={
        "note": "CheckList-style battery: per-slice cosine stats + must_not_hold "
                "separation AUC vs true pairs, for the winning trained encoder and its "
                "identically-initialized untrained twin",
        "winner_arm": f"{WIN_SCHEME}/{WIN_MISSING}", "steps": STEPS,
        "probe_serializer": "winning scheme, probe field order honored "
                            "(field-order slice live); name_suffix admitted for the "
                            "Jr/Sr slice",
        "n_true_pairs": len(true_pairs), "n_per_slice": PROBE_N,
        "dial_rates": list(DIAL_RATES), "lexicon_workaround": bool(drifted),
        "split": {"scheme": SCHEME_SPLIT, "n_eval_records": len(ev_sub)},
        "corpus_provenance": CORPUS_PROVENANCE,
    },
)
show = battery.pivot_table(
    index=["kind", "slice"], columns="encoder",
    values=["mean_cos", "auc_vs_true"], sort=False,
)
print("registered invariance_battery — the report card:")
display(show.round(3))
SECTION_SECS["6 battery"] = time.time() - t_sec


# %%
def draw_battery(ax, df, meta):
    order = (
        df[df["encoder"] == df["encoder"].iloc[0]]
        .sort_values(["kind", "slice"], ascending=[False, True])["slice"]
        .tolist()
    )
    y = np.arange(len(order))
    enc_names = sorted(df["encoder"].unique())
    offsets = dict(zip(enc_names, (-0.2, +0.2)))
    colors = dict(zip(enc_names, ("C0", "0.6")))
    kinds = df.drop_duplicates("slice").set_index("slice")["kind"]
    for name in enc_names:
        g = df[df["encoder"] == name].set_index("slice").reindex(order)
        hatches = ["///" if kinds[s] == MUST_NOT_HOLD else "" for s in order]
        bars = ax.barh(y + offsets[name], g["mean_cos"], height=0.38, color=colors[name],
                       label=name)
        for bar, hatch in zip(bars, hatches):
            bar.set_hatch(hatch)
        if name != "untrained":
            for yi, s in zip(y, order):
                auc = g.loc[s, "auc_vs_true"]
                if np.isfinite(auc):
                    ax.annotate(f"AUC {auc:.2f}", (float(g.loc[s, 'mean_cos']), yi - 0.2),
                                xytext=(4, -3), textcoords="offset points", fontsize=7)
    first_mnh = next(i for i, s in enumerate(order) if kinds[s] == MUST_NOT_HOLD)
    ax.axhspan(first_mnh - 0.5, len(order) - 0.5, color="C3", alpha=0.06, zorder=0)
    ax.set_yticks(y, order)
    for tick, s in zip(ax.get_yticklabels(), order):
        if kinds[s] == MUST_NOT_HOLD:
            tick.set_color("C3")
    ax.invert_yaxis()
    ax.set_xlabel("mean pair cosine (hatched red band = must_not_hold: HIGH is bad there)")
    ax.legend(loc="lower right")


fig = figures.plot_artifact(
    registry, tier=cfg.run.tier, artifact="invariance_battery", draw=draw_battery,
    title="The ER-CheckList report card - trained vs its untrained twin",
    figsize=(7.6, 5.4),
)

# %% [markdown]
# Read the card the way the conjecture said to. The untrained twin's bars sit high
# *everywhere* — should-hold and must-not-hold alike (every slice above 0.99 in the table):
# a randomly-initialized smooth encoder scores near-identical strings as near-identical
# vectors, whoever they denote, so its "invariance" is vacuous — and its confusable AUCs
# sit *below* 0.5, because genuinely varying same-person records score LOWER than
# one-field-off impostors. That inversion is notebook 00's "hardest easy problem" restated
# in embedding space: string similarity is not person similarity.
#
# Training reorganizes the space — but *selectively*, and the red band names the
# selectivity. `twin_household` (different given name, same family/place/DOB — the trap
# notebook 05 planted) drops well below the typo slices and its AUC climbs far above the
# twin's: the training pairs relentlessly contrast given names within and across entities,
# so that axis got carved. `different_birth_year` barely moves — only the dob digits
# differ, and a few hundred steps of name-dominated positives taught the encoder to read a
# small dob edit like a typo rather than an identity change, so the one-birth-year-off
# doppelgänger still outscores genuine variants. And `suffix_jr_sr` does not move at all:
# the corpus carries no suffixes, so nothing ever pushed Jr away from Sr. The battery can
# ask questions the training distribution never posed — and its per-slice verdicts point at
# the pressures to come: augmentation/tokenization for field-level edits (notebook 10),
# nickname and suffix supervision (notebook 11).

# %% [markdown]
# ## 7. The geometry panel: four numbers for a space
#
# Slice-level behavior tells you *what* the encoder does to specific perturbations; the
# geometry panel tells you what the whole space looks like while it does it
# (`er_lab.probes.geometry`, the lab's mediator vocabulary — PLAN §5):
#
# - **alignment** (Wang–Isola): mean squared distance between true-pair embeddings — but
#   read it WITH uniformity: a collapsed space aligns everything for free;
# - **uniformity**: log mean Gaussian potential — how much of the hypersphere the corpus
#   actually uses (0 = collapse; more negative = more spread);
# - **RankMe**: effective rank of the embedding matrix — how many directions carry signal;
# - **hubness**: k-occurrence skewness — whether "wolf records" haunt everyone's neighbor
#   lists (SCL-02's scale question, first measured here).
#
# All computed on the same eval embeddings the system metric used, for every trained arm
# plus the untrained twin.

# %%
t_sec = time.time()
rng_geo = np.random.default_rng(PRIMARY_SEED)
geo_pos = np.sort(rng_geo.choice(len(ev_sub), size=min(GEOM_N, len(ev_sub)), replace=False))
ia = np.array([a for a, _ in pair_pos], dtype=int)
ib = np.array([b for _, b in pair_pos], dtype=int)

untrained_row, untrained_pred, untrained_emb = system_point(
    untrained, arm_config(WIN_SCHEME, WIN_MISSING), seed=PRIMARY_SEED
)

geo_rows = []
geometry_inputs = {ENCODER_LABELS["untrained"]: (untrained_emb, untrained_row, 0)}
for (scheme, missing_treatment, seed), emb in arm_embeddings.items():
    if seed != PRIMARY_SEED:
        continue  # the panel is one row per encoder; multi-seed geometry is the node run's
    row = next(
        r for r in matrix_rows
        if (r["scheme"], r["missing"], r["seed"]) == (scheme, missing_treatment, seed)
    )
    geometry_inputs[f"trn05:{scheme}/{missing_treatment}"] = (emb, row, STEPS)

for enc_label, (emb, row, steps_done) in geometry_inputs.items():
    # hubness is exact brute-force and row-capped: measured on the geometry sample, the
    # same subset uniformity/rankme use (full-corpus hubness at scale is SCL-02's job)
    hub = hubness(emb[geo_pos], k=K_CANDS)
    geo_rows.append({
        "encoder": enc_label, "scheme": row["scheme"], "missing": row["missing"],
        "steps": steps_done, "trained": steps_done > 0,
        "alignment_true_pairs": alignment(emb[ia], emb[ib]),
        "true_pair_mean_cos": float(np.einsum("ij,ij->i", emb[ia], emb[ib]).mean()),
        "uniformity": uniformity(emb[geo_pos]),
        "rankme": rankme(emb[geo_pos]),
        "hubness_skewness": hub["skewness"],
        "hubness_max_k_occurrence": hub["max_k_occurrence"],
        "bcubed_f1": row["bcubed_f1"], "f1_lo": row["f1_lo"], "f1_hi": row["f1_hi"],
        "attained": row["attained"], "fallback": row["fallback"],
    })
geometry_panel = pd.DataFrame(geo_rows)
registry.register(
    "embedding_geometry_panel", geometry_panel, cfg=cfg, tier=cfg.run.tier,
    meta={
        "note": "alignment/uniformity/rankme/hubness per encoder on entity-disjoint eval "
                "embeddings, + true-pair alignment inputs and the system metric; one row "
                "per trained TRN-05 arm plus the untrained twin",
        "geometry_sample": len(geo_pos), "n_true_pairs": len(ia),
        "hubness_k": K_CANDS, "winner_arm": f"{WIN_SCHEME}/{WIN_MISSING}",
        "definitions": "alignment: Wang-Isola mean ||a-b||^2 on normalized true pairs "
                       "(lower = tighter); uniformity: log mean Gaussian potential "
                       "(0 = collapse, more negative = more spread); rankme: effective "
                       "rank; hubness: k-occurrence skewness",
        "corpus_provenance": CORPUS_PROVENANCE,
    },
)
print("registered embedding_geometry_panel:")
display(
    geometry_panel.set_index("encoder")[
        ["steps", "alignment_true_pairs", "true_pair_mean_cos", "uniformity", "rankme",
         "hubness_skewness", "bcubed_f1"]
    ].round(4)
)
SECTION_SECS["7 geometry"] = time.time() - t_sec

# %% [markdown]
# ### Verdict — scoring NB08-BATTERY against the registered numbers
#
# Same discipline as TRN-05: predictions P1–P3 checked against `invariance_battery` and
# `embedding_geometry_panel`, the P3 system-metric contrast as a shared-draw paired delta,
# and the adoption-gate / mediation distinction stated rather than blurred.

# %%
b_ix = battery.set_index(["encoder", "slice"])
mnh_slices = sorted(
    battery.loc[(battery["kind"] == MUST_NOT_HOLD) & (battery["n"] > 0), "slice"].unique()
)
tr_label, un_label = ENCODER_LABELS["trained"], ENCODER_LABELS["untrained"]
auc_tr = {s: float(b_ix.loc[(tr_label, s), "auc_vs_true"]) for s in mnh_slices}
auc_un = {s: float(b_ix.loc[(un_label, s), "auc_vs_true"]) for s in mnh_slices}
p1_batt = bool(all(v <= 0.55 for v in auc_un.values()))

g_ix = geometry_panel.set_index("encoder")
unif_tr = float(g_ix.loc[tr_label, "uniformity"])
unif_un = float(g_ix.loc[un_label, "uniformity"])
p2_batt = bool(unif_tr < unif_un and all(auc_tr[s] > auc_un[s] for s in mnh_slices))

d_train = paired_delta(
    preds[WIN_KEY], untrained_pred, truth_ev, "bcubed_f1",
    unit="entity", n_boot=N_BOOT, seed=PRIMARY_SEED,
)
p3_batt = bool(d_train["sign_stable"] and d_train["delta"] > 0)

print(f"P1 untrained AUC <= 0.55 on every must_not_hold slice: {p1_batt}  ({auc_un})")
print(f"P2 uniformity {unif_tr:.3f} (trained) < {unif_un:.3f} (untrained) AND every "
      f"must_not_hold AUC raised: {p2_batt}  (trained: {auc_tr})")
print(f"P3 paired delta trained - untrained bcubed_f1 = {d_train['delta']:+.4f} "
      f"[{d_train['ci_low']:+.4f}, {d_train['ci_high']:+.4f}] "
      f"sign_stable={d_train['sign_stable']}: {p3_batt}")

if p1_batt and p2_batt and p3_batt:
    batt_outcome = "CONFIRMED"
elif (not p2_batt and all(auc_tr[s] <= auc_un[s] for s in mnh_slices)) or (
    d_train["sign_stable"] and d_train["delta"] < 0
):
    batt_outcome = "REFUTED"
else:
    batt_outcome = "UNEXPLAINED"

_ = verdict_box(
    "NB08-BATTERY",
    outcome=batt_outcome,
    evidence=(
        f"invariance_battery + embedding_geometry_panel (tier {TIER}, winner "
        f"{WIN_SCHEME}/{WIN_MISSING} vs its identically-initialized untrained twin, "
        f"{len(probes):,} probe pairs, {len(true_pairs):,} true pairs). P1={p1_batt}: "
        f"untrained must_not_hold AUCs "
        f"{ {s: round(v, 3) for s, v in auc_un.items()} }. P2={p2_batt}: uniformity "
        f"{unif_un:.3f} -> {unif_tr:.3f}; trained AUCs "
        f"{ {s: round(v, 3) for s, v in auc_tr.items()} }. P3={p3_batt}: shared-draw "
        f"paired B-cubed F1 delta {d_train['delta']:+.4f} [{d_train['ci_low']:+.4f}, "
        f"{d_train['ci_high']:+.4f}]. RankMe {float(g_ix.loc[un_label, 'rankme']):.1f} -> "
        f"{float(g_ix.loc[tr_label, 'rankme']):.1f} and hubness skew "
        f"{float(g_ix.loc[un_label, 'hubness_skewness']):.2f} -> "
        f"{float(g_ix.loc[tr_label, 'hubness_skewness']):.2f} reported as exploratory "
        f"descriptors (no pre-registered direction). SCOPE: property AND metric moved "
        f"together, which satisfies the adoption-gate SHAPE only — it is not mediation, "
        f"and with one seed and one noise draw the mediation status is UNEXPLAINED by "
        f"construction (PLAN §5); the dose-response and indirect-effect machinery arrives "
        f"with the multi-seed runs of notebooks 09-11."
    ),
    registry=registry,
)

# %% [markdown]
# ## 8. The signature visual: pressure → property → geometry → metric
#
# This triptych is the arc's recurring exhibit, rendered by
# `reporting.figures.three_panel_pressure` from three registered panel frames. Read it left
# to right as the lab's causal grammar: **left** — the pressure dial on x (here: the typo
# dial), and on y how much training moved the property under pressure (the margin between
# same-person typo variants and the twin-household confusable, trained minus untrained — a
# margin that shrinks and finally inverts as the dial rises: invariance is a budgeted
# quantity, not a switch);
# **middle** — where each encoder lives on the alignment–uniformity plane (Wang–Isola's
# map: bottom-left = aligned *and* spread, the contrastive sweet spot; the untrained twin
# sits apart, aligned-by-collapse); **right** — the system metric over the same pressure
# (0 → trained steps), with its CI band. Notebooks 09–11 re-render exactly this triptych
# with their own dials — loss family (TRN-01), miner and duplication rate (TRN-02),
# augmentation dose (TRN-03), regime (TRN-04) — so that every pressure's claim is read the
# same way: left panel moving without the right is a property that doesn't matter; right
# without left is a win the conjecture doesn't explain; both moving is admission to the
# adoption gate — and only the cross-arm mediation analysis may call it *mediation*.

# %%
b_marg = battery.set_index(["encoder", "slice"])["mean_cos"]
prop_rows = []
for rate in DIAL_RATES:
    s_typo = f"typo@{rate:g}"
    keys = [(e, s) for e in (tr_label, un_label) for s in (s_typo, "twin_household")]
    if not all(k in b_marg.index for k in keys):
        continue
    m_tr = float(b_marg.loc[(tr_label, s_typo)]) - float(b_marg.loc[(tr_label, "twin_household")])
    m_un = float(b_marg.loc[(un_label, s_typo)]) - float(b_marg.loc[(un_label, "twin_household")])
    prop_rows.append({"x": rate, "y": m_tr - m_un})
registry.register(
    "nb08_triptych_property", pd.DataFrame(prop_rows, columns=["x", "y"]),
    cfg=cfg, tier=cfg.run.tier,
    meta={"panel": "property shift", "x": "typo dial rate (per-char corruption)",
          "y": "margin(typo mean_cos - twin_household mean_cos), trained minus untrained",
          "derived_from": "invariance_battery"},
)
registry.register(
    "nb08_triptych_geometry",
    geometry_panel.rename(
        columns={"alignment_true_pairs": "x", "uniformity": "y", "encoder": "group"}
    )[["x", "y", "group"]],
    cfg=cfg, tier=cfg.run.tier,
    meta={"panel": "embedding geometry", "x": "alignment (true pairs; lower = tighter)",
          "y": "uniformity (more negative = more spread)",
          "derived_from": "embedding_geometry_panel"},
)
registry.register(
    "nb08_triptych_system",
    geometry_panel.loc[
        geometry_panel["encoder"].isin([un_label, tr_label]),
        ["steps", "bcubed_f1", "f1_lo", "f1_hi"],
    ].rename(columns={"steps": "x", "bcubed_f1": "y", "f1_lo": "lo", "f1_hi": "hi"}),
    cfg=cfg, tier=cfg.run.tier,
    meta={"panel": "system metric", "x": "optimizer steps (0 = untrained twin)",
          "y": f"B-cubed F1 at precision@{PREC_TARGET} (entity-unit BCa 95% CI)",
          "derived_from": "embedding_geometry_panel"},
)

fig = figures.three_panel_pressure(
    registry, tier=cfg.run.tier,
    property_shift="nb08_triptych_property",
    geometry="nb08_triptych_geometry",
    system_metric="nb08_triptych_system",
    titles=(
        "property: typo-vs-confusable margin,\ntrained - untrained (x = typo dial)",
        "geometry: alignment vs uniformity\n(one point per encoder)",
        "system: B-cubed F1 @ P0.99\n(x = optimizer steps)",
    ),
    suptitle="NB08 pressure triptych - the template notebooks 09-11 re-render per pressure",
    figsize=(12.5, 4.4),
)

# %% [markdown]
# ## What this notebook demands downstream
#
# - **The battery + geometry panel are now the lab's definition of "good embedding".**
#   From here on, no training pressure gets to claim victory on a system metric alone:
#   TRN-01/02 (notebook 09), TRN-03/04 (notebook 10) and TRN-06 (notebook 11) each name, in
#   their cards, which battery slices and which geometry numbers their pressure should move
#   — and render this notebook's triptych over their own dial.
# - **`trn05_serialization_matrix`** fixes the serialization default the later pressure
#   notebooks train under (winner at this tier: printed above, with its fallback flags and
#   the truncation census that bounds what smoke can say). The full 8-cell factorial ×
#   seeds at the definitive tier — with `max_len=192`, retiring the census confound — is
#   the placarded node run; field-order sensitivity stays an eval-time battery slice.
# - **`invariance_battery`** debuts the report card consumed as an acceptance test in
#   notebook 18 (adapting the pipeline to a new schema means re-running the battery, not
#   trusting vibes), and its nickname/suffix slices set up TRN-06's question directly.
# - **`embedding_geometry_panel`** starts the mediator record: hubness feeds SCL-02's
#   wolf-record analysis, uniformity/alignment anchor the TRN-01/03 mediation chains.
#
# ## What we now know
#
# - **Serialization is a real dial at matched budget** — the matrix separates arms trained
#   identically in every other respect, and the paired-delta verdict above says exactly how
#   far one smoke seed lets us trust the ordering (with the fixed-precision fallback rail's
#   status recorded arm by arm either way). The context-budget census turned a would-be
#   silent confound into a measured, pre-flagged limitation of the smoke tier.
# - **Training is what turns string geometry into person geometry — and the battery shows
#   exactly how far.** The untrained twin — same architecture, same initialization, same
#   inputs — scores confusables as high as variants (separation AUCs below chance); a few
#   hundred optimizer steps spread the space (the uniformity shift), separate the household
#   trap, and move the system metric. But the report card also names what this one tiny
#   contrastive run did NOT buy — dob-only doppelgängers and never-seen suffixes stay
#   unseparated — which is precisely why the NB08-BATTERY verdict lands where it lands, and
#   precisely the unfinished business notebooks 10 and 11 take up. "A person as a vector"
#   is earned per-property, not granted wholesale.
# - **The property battery, not any single number, is the lab's working answer to "what is
#   a good person-embedding"** — invariances that hold, separations that don't collapse,
#   geometry that supports retrieval, all measured on entity-disjoint eval data. The
#   pressures arc now begins: notebook 09 turns the loss/negatives dial with MET-04's
#   variance machinery, and every claim it makes will be read off this notebook's triptych.
#
# **Artifacts registered** (exact names): `trn05_serialization_matrix`,
# `invariance_battery`, `embedding_geometry_panel` — plus supporting
# `trn05_train_histories`, the triptych panel frames `nb08_triptych_property` /
# `nb08_triptych_geometry` / `nb08_triptych_system`, and the immutable cards `card_TRN-05`
# and `card_NB08-BATTERY`.

# %%
print("section timings (measured):")
for name, secs in SECTION_SECS.items():
    print(f"  {name:<32s} {secs:7.1f}s")
print(f"notebook wall-clock: {time.time() - NB_T0:.0f}s")
