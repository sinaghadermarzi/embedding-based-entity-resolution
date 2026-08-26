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
# # 10. Pressures II: Typos — Tokenizer or Augmentation?
#
# **The question.** A person record's worst everyday enemy is the typo — and there are two
# places to buy robustness against it. In the **tokenizer**: a byte-level encoder sees a
# one-character typo as a one-or-two-id perturbation of its input, so "MARGARET" and
# "MARGARFT" are near-identical sequences *by construction*. Or in the **training data**:
# corrupt the training pairs so the loss itself teaches "these two spellings are the same
# person" — and then the follow-up question is *which* corruption, a generic typo dose or
# the corpus's own **measured** noise mix (notebook 04's NSE-01 prevalences, baked into
# notebook 05's calibrated corpus). This notebook runs both halves of PLAN §3's answer:
# **TRN-03** (augmentation {none, generic typo, calibrated channels}, the protected middle
# of the pruning order) and **TRN-04** (from-scratch char/byte vs pretrained-subword, the
# encoder-regime factor those augmentation arms will be crossed with at mid/target).
#
# **What this notebook settles.** (1) TRN-03 smoke slice: three budget-matched arms of the
# scratch-char encoder differing ONLY in train-time augmentation, each scored by the
# shared system-metric recipe, the typo-dial battery, and the geometry panel — then read
# through notebook 09's **MET-04 lens**: this notebook loads `met04_power_table` by DAG
# contract and every verdict below quotes its measured replicate sigma, because a
# single-seed margin inside that sigma is noise and saying so is the finding. An
# **eval-side dial** (the eval slice re-corrupted at 0.5×/1×/2× the measured rates) probes
# dose-response cheaply — explicitly *not* the PLAN §3.1 train-side augmentation-rate
# dial, which is the node run's mediation instrument. (2) TRN-04 smoke slice: the
# head-to-head's scratch side, live; the pretrained-subword side is a real
# `# [RUN-IN-TARGET mac]` code cell (huggingface.co is blocked from this container,
# `notes/COMPAT.md`), so the registered head-to-head carries the scratch rows and an
# honest absence — never a fabricated pretrained row.
#
# **Scope honesty, up front.** One seed per arm at smoke: every matrix row is a
# DEMONSTRATION of machinery; the definitive multi-seed matrices are tier=mid/node and
# every verdict says so. Mediation (augmentation → invariance → F1) is not claimable from
# one seed: co-movement is reported as SUGGESTIVE at most, mediation UNEXPLAINED (PLAN §5
# keeps the adoption gate and the mediation claim strictly apart). No entity appears on
# both sides of any train/eval boundary (entity-disjoint `met07_splits`, PLAN §5).

# %%
import copy
import time

import numpy as np
import pandas as pd
from IPython.display import display
from omegaconf import OmegaConf
from sklearn.metrics import roc_auc_score

from er_lab.blocking import ann
from er_lab.cluster.schemes import transitive_closure
from er_lab.config import REPO_ROOT, config_hash, load_config_from_env, set_all_seeds
from er_lab.data.schema import NON_TEXT_ROLES, ROLES
from er_lab.eval.bootstrap import bootstrap_ci
from er_lab.eval.operating_points import find_threshold_for_precision
from er_lab.infra.artifacts import ArtifactRegistry
from er_lab.infra.device import describe_platform
from er_lab.models.encoders import build_encoder
from er_lab.noise.channels import load_lexicon
from er_lab.probes.battery import battery_report, build_probe_set
from er_lab.probes.geometry import alignment, hubness, rankme, uniformity
from er_lab.reporting import figures
from er_lab.reporting.cards import conjecture_card, verdict_box
from er_lab.serialize import serialize_frame, serialize_record
from er_lab.train import augment as augment_mod
from er_lab.train.augment import AUGMENT_KINDS, make_augmenter
from er_lab.train.loop import train_encoder
from er_lab.train.mining import build_pairs_inbatch

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
# ## Tier constants — and the budget rule, restated once
#
# **Budget-matching is step-count accounting** (PLAN §3.1): every compared arm runs the
# *exact same number of optimizer steps*, asserted from its returned training history —
# never epochs, never wall-clock. The dials below scale sizes with tier; the protocol
# (precision-0.99 operating point, entity-unit BCa CIs, entity-disjoint splits) is
# identical at every tier, and the encoder shape matches notebook 09's smoke shape (dim
# 128, 2 layers, 4 heads, 128-byte window) so this notebook's arms are comparable with
# that one's. Smoke budget for the whole notebook: **<= ~30 min on the 4-CPU container**
# (three trainings plus an eval-dial sweep; measured per section, printed at the end).

# %%
TIER = str(cfg.run.tier)
STEPS_BY_TIER = {"smoke": 300, "mid": 2000, "target": 4000, "analytical": 300}
SIZES = {"smoke": 3_000, "mid": None, "target": None, "analytical": 3_000}  # train records
EVAL_N = {"smoke": 5_000, "mid": None, "target": None, "analytical": 5_000}  # eval records
GRID = {"smoke": 40, "mid": 150, "target": 200, "analytical": 40}[TIER]
N_BOOT = {"smoke": 400, "mid": 1000, "target": 2000, "analytical": 400}[TIER]
PROBE_N = {"smoke": 120, "mid": 400, "target": 800, "analytical": 120}[TIER]
TRUE_PAIR_N = {"smoke": 300, "mid": 1000, "target": 2000, "analytical": 300}[TIER]
GEO_N = {"smoke": 1_500, "mid": 4_000, "target": 8_000, "analytical": 1_500}[TIER]
MODEL_SHAPE = {
    "smoke": {"dim": 128, "layers": 2, "heads": 4, "max_len": 128},
    "mid": {"dim": 256, "layers": 4, "heads": 4, "max_len": 192},
    "target": {"dim": 384, "layers": 6, "heads": 6, "max_len": 192},
    "analytical": {"dim": 128, "layers": 2, "heads": 4, "max_len": 128},
}[TIER]
STEPS = STEPS_BY_TIER[TIER]
PREC_TARGET = 0.99  # PLAN §5 primary fixed entity-precision operating point
CAND_K = 10  # eval candidate budget: ann.candidates(k=10, index='flat')
ENC_BATCH = 256  # inference encode batch
PRIMARY_SEED = int(cfg.run.seed)
DIAL_RATES = (0.02, 0.05, 0.1, 0.2)  # battery typo dial (per-char corruption), as NB08
EVAL_NOISE_MULTS = (0.5, 1.0, 2.0)  # the prescribed eval-side dial, × measured rates
STRESS_MULT = 10.0  # one exploratory stress point (measured rates are floors; see §5)
AUG_ARMS = ("none", "generic_typo", "calibrated")  # TRN-03's locked factor levels
LOSS_LOCKED = "infonce"  # everything but augmentation locked at pre-registered defaults
assert set(AUG_ARMS) == set(AUGMENT_KINDS), "TRN-03 arms must be the augment module's vocabulary"

# Budget-matched encoder + step budget through the typed cfg keys, so the config hash on
# every registered artifact records exactly what trained.
cfg.model.kind = "scratch_char"
cfg.model.dim = MODEL_SHAPE["dim"]
cfg.model.layers = MODEL_SHAPE["layers"]
cfg.model.heads = MODEL_SHAPE["heads"]
cfg.model.max_len = MODEL_SHAPE["max_len"]
cfg.train.steps = STEPS
BATCH = int(cfg.train.batch_size)
print(f"tier={TIER}: steps={STEPS} batch={BATCH} model=(dim {cfg.model.dim}, "
      f"layers {cfg.model.layers}, heads {cfg.model.heads}, max_len {cfg.model.max_len}); "
      f"train-slice budget {SIZES[TIER]}, eval-slice budget {EVAL_N[TIER]}, grid={GRID}, "
      f"n_boot={N_BOOT}")
print(f"TRN-03 arms: {AUG_ARMS} x scratch-char (pretrained regime: RUN-IN-TARGET); "
      f"battery dial {DIAL_RATES}; eval-side dial x{EVAL_NOISE_MULTS} "
      f"(+ exploratory stress x{STRESS_MULT:g}); smoke budget <= ~30 min")

# %% [markdown]
# ## 1. The arena, the shared recipe — and the MET-04 lens this notebook must wear
#
# Both data inputs load through the registry at this run's tier: the calibrated corpus
# (notebook 05) and the MET-07 splits. Training uses the **entity-disjoint train half**
# subsampled *entity-complete* (whole clusters, never split); evaluation uses a fixed
# entity-complete subsample of the **eval half**, identical across every arm (paired
# comparison). The third required input is notebook 09's `met04_power_table` — the
# binding arbiter (PLAN §3). From it we take `sd_replicate` (the measured run-to-run
# sigma of B³F1 under seed × noise-draw at this exact smoke recipe) and the pre-computed
# **single-seed detectability bar** 2·√2·sd: a one-replicate-per-arm margin smaller than
# the bar is indistinguishable from replicate noise, and every verdict below applies
# exactly that rule rather than inventing its own.
#
# **The shared system-metric recipe** — the same helper notebook 09 defined, verbatim in
# behavior: serialize the eval slice per `cfg.serialize` → encode →
# `ann.candidates(k=10, index='flat')` → cosine scores mapped to [0, 1] →
# `find_threshold_for_precision` at the locked 0.99 entity-precision point (PLAN §5
# **loud fallback**, never a silent protocol switch) → transitive closure → B-cubed with
# **entity-unit BCa CIs**. One extension for this notebook: the helper accepts alternate
# eval *texts* over the same records, which is how the eval-side dial re-scores each
# encoder on re-corrupted serializations without touching the protocol.

# %%
t_sec = time.time()
corpus, corpus_meta = registry.load("calibrated_corpus", tier=cfg.run.tier)
splits, splits_meta = registry.load("met07_splits", tier=cfg.run.tier)
met04, met04_meta = registry.load("met04_power_table", tier=cfg.run.tier)
CORPUS_PROVENANCE = (
    f"calibrated_corpus run {corpus_meta['created_at']} (cfg {corpus_meta['config_hash']}, "
    f"tier {corpus_meta['tier']}, seed {corpus_meta['extra']['seed']})"
)
print(f"corpus: {CORPUS_PROVENANCE}")
print(f"  {len(corpus):,} records / {corpus['entity_id'].nunique():,} entities")

SD_REPLICATE = float(met04_meta["extra"]["sd_replicate"])
DETECT_BAR = float(met04_meta["extra"]["detect_bar_single_seed"])
_power = met04[met04["kind"] == "power"]
SEEDS_NEEDED = {float(r["delta"]): int(r["seeds_needed"]) for _, r in _power.iterrows()}
print(f"\nMET-04 lens (met04_power_table run {met04_meta['created_at']}): "
      f"sd_replicate={SD_REPLICATE:.4f}, single-seed bar 2*sqrt(2)*sd={DETECT_BAR:.4f}; "
      f"seeds needed at delta 0.01/0.02/0.05 = "
      f"{SEEDS_NEEDED[0.01]}/{SEEDS_NEEDED[0.02]}/{SEEDS_NEEDED[0.05]}")
print(f"  scope flag carried verbatim: {met04_meta['extra']['scope']}")

SCHEME = "entity_disjoint"
train_ids = set(splits["schemes"][SCHEME]["train"])
eval_ids = set(splits["schemes"][SCHEME]["eval"])
train_full = corpus[corpus["record_id"].isin(train_ids)].reset_index(drop=True)
eval_full = corpus[corpus["record_id"].isin(eval_ids)].reset_index(drop=True)
straddle = splits["metadata"]["checks"][SCHEME]["entities_straddling"]
print(f"\nsplit scheme: {SCHEME} (seed {splits['metadata']['seed']}, "
      f"entities straddling the boundary: {straddle})")
assert straddle == 0, "entity-disjoint split must have zero straddling entities (PLAN §5)"


def entity_complete_subsample(
    frame: pd.DataFrame, n_records: int | None, rng: np.random.Generator
) -> pd.DataFrame:
    """Random whole entities until ~n_records — an entity is never split across the cut.

    (Same helper as notebooks 05/09; still notebook-local — a shared home in
    er_lab.data would serve, carried again as a package gap.)
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


train_slice = entity_complete_subsample(
    train_full, SIZES[TIER], np.random.default_rng(PRIMARY_SEED)
)
ev_slice = entity_complete_subsample(
    eval_full, EVAL_N[TIER], np.random.default_rng(PRIMARY_SEED + 1)
)
print(f"train slice: {len(train_slice):,} records / {train_slice['entity_id'].nunique():,} "
      f"entities; eval slice: {len(ev_slice):,} records / "
      f"{ev_slice['entity_id'].nunique():,} entities — fixed across ALL arms (paired)")
tick("§1 arena + MET-04 lens", t_sec)

# %%
t_sec = time.time()
TEXT_ROLES = [c for c in corpus.columns if c in ROLES and c not in NON_TEXT_ROLES]
SER_SCHEME, SER_MISSING = str(cfg.serialize.scheme), str(cfg.serialize.missing)


def texts_of(frame: pd.DataFrame) -> list[str]:
    return serialize_frame(
        frame, text_roles=TEXT_ROLES, scheme=SER_SCHEME, missing=SER_MISSING
    ).tolist()


EVAL_TEXTS = texts_of(ev_slice)
TRUTH_EV = ev_slice.set_index("record_id")["entity_id"]
RECORDS_EV = pd.Index(TRUTH_EV.index)
EVAL_POS = {rid: i for i, rid in enumerate(ev_slice["record_id"].astype(str))}
lens = np.array([len(t.encode("utf-8")) for t in EVAL_TEXTS])
share_trunc = float((lens > cfg.model.max_len - 1).mean())
print(f"serialization: scheme={SER_SCHEME} missing={SER_MISSING}, roles={TEXT_ROLES}")
print(f"eval text length: median {int(np.median(lens))} bytes; {share_trunc:.0%} exceed the "
      f"encoder's max_len={cfg.model.max_len} byte window and are TRUNCATED — the same "
      "measured smoke constraint notebook 09 carried; the mid window (192) covers the row.")


def fresh_encoder(seed: int):
    """Deterministic encoder init: same seed => identical starting weights across arms."""
    set_all_seeds(seed)
    return build_encoder(cfg)


def train_arm(*, augment: str, seed: int, cfg_arm=None, encoder=None):
    """One budget-matched training run; asserts the step budget from the history.

    TRN-03's factor is ONLY the augmentation kind: loss/miner locked at the
    pre-registered defaults (infonce, in-batch), same train slice, same init seed.
    The np.errstate guard silences gecko's benign 0-candidate divide warning
    (same workaround as notebooks 05/09) during augmented batches.
    """
    arm_cfg = cfg if cfg_arm is None else cfg_arm
    enc = fresh_encoder(seed) if encoder is None else encoder
    t0 = time.time()
    with np.errstate(divide="ignore"):
        enc, hist = train_encoder(
            enc, train_slice, arm_cfg, loss_name=LOSS_LOCKED, miner_name="inbatch",
            augment_kind=augment, steps=STEPS, seed=seed,
        )
    secs = time.time() - t0
    assert len(hist) == STEPS and int(hist["step"].iloc[-1]) == STEPS, (
        f"budget violation: expected exactly {STEPS} optimizer steps, history shows {len(hist)}"
    )
    return enc, hist, secs


def system_metric(encoder, *, label: str, texts: list[str] | None = None,
                  with_ci: bool = True) -> dict:
    """THE shared recipe (§1 narrative): every encoder answers the same question.

    ``texts`` (default: the eval slice's own serializations) lets the eval-side
    dial score the SAME records under re-corrupted serializations — protocol
    otherwise identical, including the loud precision fallback.
    """
    t0 = time.time()
    use_texts = EVAL_TEXTS if texts is None else texts
    emb = encoder.encode(use_texts, batch_size=ENC_BATCH)
    cand = ann.candidates(ev_slice, emb, k=CAND_K, index="flat")
    ia = cand["a"].astype(str).map(EVAL_POS).to_numpy()
    ib = cand["b"].astype(str).map(EVAL_POS).to_numpy()
    cos = np.einsum("ij,ij->i", emb[ia], emb[ib])
    scored = pd.DataFrame({"a": cand["a"], "b": cand["b"],
                           "prob": np.clip((cos + 1.0) / 2.0, 0.0, 1.0)})
    cache: dict[float, pd.Series] = {}

    def clusterer(t: float) -> pd.Series:
        if t not in cache:
            cache[t] = transitive_closure(scored, threshold=float(t), records=RECORDS_EV)
        return cache[t]

    res = find_threshold_for_precision(
        scored.rename(columns={"prob": "score"}), clusterer, TRUTH_EV,
        target=PREC_TARGET, grid=GRID,
    )
    if not res["attained"]:
        print(f"  !! [{label}] precision {PREC_TARGET} UNATTAINABLE on this eval slice — "
              f"reporting the highest-attainable point P={res['attained_precision']:.4f} "
              f"(fallback='{res['fallback']}'; PLAN §5 rail: loud, never silent)")
    row = {
        "threshold": res["threshold"], "attained": bool(res["attained"]),
        "attained_precision": res["attained_precision"], "recall_at": res["recall_at"],
        "f1": res["f1_at"], "n_candidate_pairs": len(scored),
    }
    if with_ci:
        pred = clusterer(res["threshold"])
        ci = bootstrap_ci(pred, TRUTH_EV, "bcubed_f1", unit="entity", n_boot=N_BOOT,
                          seed=PRIMARY_SEED)
        row["f1_lo"], row["f1_hi"] = ci["ci_low"], ci["ci_high"]
    row["eval_secs"] = time.time() - t0
    return row


tick("§1b serialization + shared recipe", t_sec)

# %% [markdown]
# ### The property instruments — and one new number: the dial AUC
#
# One probe set (notebook 08's battery vocabulary, typo dial at the same four rates) and
# one geometry panel (alignment / uniformity / RankMe / hubness) are built ONCE on the
# eval slice and measured under every trained encoder — property deltas between arms are
# about the arms. New here is the **typo-dial AUC**, the retrieval-shaped reading of the
# typo slices this notebook's cards pre-register: at each dial rate, the probability that
# a typo-corrupted copy of a record still scores a higher cosine with its original than a
# random *different-person* (impostor) pair does — Mann-Whitney AUC of typo-pair cosines
# vs impostor-pair cosines. 1.0 means typos never cost a retrieval margin; 0.5 means a
# typo'd duplicate is indistinguishable from a stranger. This is deliberately different
# from the battery's `auc_vs_true` (which separates *confusables* from true pairs); both
# are reported. Impostor pairs are fixed across arms, like every other instrument.

# %%
t_sec = time.time()
try:
    _lex_raw = load_lexicon(str(REPO_ROOT / "data"))
    LEXICON = {
        c: {v for v in vs if v not in ("has_nickname", "relationship")}
        for c, vs in _lex_raw.items() if c != "name1"
    }
    LEXICON = {c: v for c, v in LEXICON.items() if v} or None
    lex_note = f"carltonnorthern lexicon, sanitized (NB04 workaround): {len(LEXICON or {})} names"
except FileNotFoundError:
    LEXICON = None
    lex_note = "lexicon file absent -> nickname channel built-in mini-lexicon"
print(lex_note)


def probe_serialize(row: dict) -> str:
    return serialize_record(row, text_roles=TEXT_ROLES, scheme=SER_SCHEME, missing=SER_MISSING)


PROBES = build_probe_set(
    ev_slice, serialize=probe_serialize, seed=PRIMARY_SEED, lexicon=LEXICON,
    dial_rates=DIAL_RATES, n_per_slice=PROBE_N,
)
TYPO_SLICES = {r: PROBES[PROBES["slice"] == f"typo@{r:g}"] for r in DIAL_RATES}
print("probe slices built:")
display(PROBES.groupby(["kind", "slice"]).size().rename("n_pairs").to_frame())

_tp = build_pairs_inbatch(ev_slice, n_pairs=TRUE_PAIR_N, seed=PRIMARY_SEED)
_etext = pd.Series(EVAL_TEXTS, index=ev_slice["record_id"].astype(str))
TRUE_PAIRS_TXT = pd.DataFrame({
    "text_a": _etext.loc[_tp["a_id"]].to_numpy(),
    "text_b": _etext.loc[_tp["b_id"]].to_numpy(),
})

# impostor pairs for the dial AUC: random cross-entity record pairs, fixed once
_rng_imp = np.random.default_rng(PRIMARY_SEED + 11)
_ents = ev_slice["entity_id"].to_numpy()
_imp_a: list[int] = []
_imp_b: list[int] = []
while len(_imp_a) < TRUE_PAIR_N:
    i, j = (int(v) for v in _rng_imp.integers(0, len(ev_slice), size=2))
    if _ents[i] != _ents[j]:
        _imp_a.append(i)
        _imp_b.append(j)
IMP_TXT_A = [EVAL_TEXTS[i] for i in _imp_a]
IMP_TXT_B = [EVAL_TEXTS[i] for i in _imp_b]

_rng_geo = np.random.default_rng(PRIMARY_SEED + 7)
_geo_pos = np.sort(_rng_geo.choice(len(ev_slice), size=min(GEO_N, len(ev_slice)),
                                   replace=False))
GEO_TEXTS = [EVAL_TEXTS[i] for i in _geo_pos]
_rid_geo = ev_slice["record_id"].astype(str).iloc[_geo_pos]
GEO_KIND = np.where(
    _rid_geo.str.contains("#dup"), "generated_duplicate",
    np.where(_rid_geo.str.contains("#hh1"), "household_confusable", "base_record"),
)
print(f"true-pair reference: {len(TRUE_PAIRS_TXT)} same-entity pairs; impostor reference: "
      f"{len(IMP_TXT_A)} cross-entity pairs; geometry sample: {len(GEO_TEXTS)} records "
      f"({pd.Series(GEO_KIND).value_counts().to_dict()})")


def property_panel(encoder) -> tuple[pd.DataFrame, dict, np.ndarray]:
    """Battery report + geometry numbers + geometry-sample embeddings, one encoder."""

    def encode_fn(ts):
        return encoder.encode(list(ts), batch_size=ENC_BATCH)

    rep = battery_report(encode_fn, PROBES, true_pairs=TRUE_PAIRS_TXT)
    emb_a = encode_fn(TRUE_PAIRS_TXT["text_a"].tolist())
    emb_b = encode_fn(TRUE_PAIRS_TXT["text_b"].tolist())
    geo_emb = encode_fn(GEO_TEXTS)
    hub = hubness(geo_emb, k=10)
    geo = {
        "align": alignment(emb_a, emb_b),
        "uniform": uniformity(geo_emb),
        "rankme": rankme(geo_emb),
        "hub_skew": hub["skewness"],
    }
    return rep, geo, geo_emb


def dial_curve(encoder) -> pd.DataFrame:
    """Typo-dial AUC per rate: typo'd same-record pairs vs fixed impostor pairs.

    All unique texts encode once per encoder; cosines come from the encoder's
    own L2-normalized embeddings. AUC is Mann-Whitney (ties 1/2) with typo
    pairs as positives and impostor pairs as negatives.
    """
    texts = set(IMP_TXT_A) | set(IMP_TXT_B)
    for grp in TYPO_SLICES.values():
        texts |= set(grp["text_a"]) | set(grp["text_b"])
    ordered = sorted(texts)
    emb = encoder.encode(ordered, batch_size=ENC_BATCH)
    pos = {t: i for i, t in enumerate(ordered)}

    def cos(col_a, col_b) -> np.ndarray:
        ia = np.fromiter((pos[t] for t in col_a), dtype=int, count=len(col_a))
        ib = np.fromiter((pos[t] for t in col_b), dtype=int, count=len(col_b))
        return np.einsum("ij,ij->i", emb[ia], emb[ib])

    imp = cos(IMP_TXT_A, IMP_TXT_B)
    rows = []
    for rate in DIAL_RATES:
        grp = TYPO_SLICES[rate]
        sim = cos(grp["text_a"].tolist(), grp["text_b"].tolist())
        labels = np.concatenate([np.ones(len(sim)), np.zeros(len(imp))])
        auc = float(roc_auc_score(labels, np.concatenate([sim, imp])))
        rows.append({"rate": float(rate), "auc": auc,
                     "mean_cos": float(np.mean(sim)), "n_pairs": len(sim)})
    return pd.DataFrame(rows)


def pca2(emb: np.ndarray) -> np.ndarray:
    """First two principal components — the triptych's geometry scatter coordinates."""
    x = np.asarray(emb, dtype=np.float64)
    x = x - x.mean(axis=0)
    _u, _s, vt = np.linalg.svd(x, full_matrices=False)
    return x @ vt[:2].T


tick("§1c property instruments", t_sec)

# %% [markdown]
# ## 2. Two conjectures, before any encoder exists
#
# Both cards render (and register, immutably) *now* — before a single training step —
# per PLAN §5. Each pre-registers its decision rule, so the honest "it's noise" outcome
# is a scored outcome, not a shrug; and per PLAN §5's adoption-gate vs mediation
# distinction, nothing below ever presents two marginal effects as mediation — that
# analysis needs arms × seeds × noise draws that only the node grid has. TRN-03's card
# also carries a scope caveat the calibrated corpus itself forces: NSE-01's typo rates
# are *drift-visible lower bounds* (an entry error stable across both audit snapshots is
# invisible by construction — the corpus meta says so), so the calibrated arm's typo dose
# is tiny at smoke, and the card says what that means for the smoke verdict up front.

# %%
_ = conjecture_card(
    card_id="TRN-03",
    conjecture=(
        "Train-time augmentation is a real typo-robustness lever, and what noise it "
        "injects matters: corrupting training pairs with the corpus's measured channel "
        "mix (calibrated) teaches the invariance the eval noise actually demands, a "
        "generic flat typo dose teaches a related but mis-weighted invariance, and no "
        "augmentation leaves invariance to whatever duplicates the corpus happens to "
        "show the loss."
    ),
    pressure=(
        "augmentation kind {none, generic_typo, calibrated} with everything else locked "
        "at the pre-registered defaults: infonce loss, in-batch negatives, colval "
        "serialization, identical scratch-char encoder and init seed, identical "
        "optimizer-step count asserted from each returned history. calibrated = "
        "make_augmenter('calibrated', channel_rates projected verbatim from the "
        "calibrated corpus meta onto the registry channels, provenance and exclusions "
        "printed before training); generic_typo = the augment module's flat default "
        "typo dose. The pretrained-subword regime completes the 6-cell factorial at "
        "mid/target (PLAN section 3.1)."
    ),
    property=(
        "typo-dial invariance, read two ways: the battery's should-hold typo@rate "
        "cosines, and the dial AUC (probability a typo-corrupted same-record pair "
        "outscores a random cross-entity impostor pair) at dial rates "
        "0.02/0.05/0.1/0.2 — with alignment/uniformity/RankMe/hubness as collapse "
        "guards"
    ),
    metric=(
        "B-cubed F1 with entity-unit BCa 95% CI on the entity-disjoint eval slice at "
        "the precision-0.99 operating point (loud highest-attainable fallback per PLAN "
        "section 5), one shared recipe for every arm; plus an eval-side robustness "
        "dial: the same eval records re-corrupted at 0.5x/1x/2x the measured channel "
        "rates and re-scored by the same recipe"
    ),
    prediction=(
        "calibrated > generic_typo > none on BOTH the typo-dial battery property (dial "
        "AUC at rate 0.2) AND the system metric, at matched budget; and the calibrated "
        "arm's system metric decays least across the eval-side dial. Decision rule, "
        "pre-registered: the system-metric ordering is scored only where the "
        "calibrated-over-none margin exceeds the MET-04 single-seed detectability bar "
        "2*sqrt(2)*sd_replicate from the registered met04_power_table; CONFIRMED needs "
        "the predicted ordering on both channels plus that margin; REFUTED needs a "
        "significant reversal (none over calibrated beyond the bar); anything else is "
        "UNEXPLAINED. Scope honesty: the calibrated typo rates are NSE-01 drift-visible "
        "LOWER BOUNDS (corpus meta), so the calibrated dose at smoke is tiny and a "
        "noise-bound smoke outcome is itself the expected honest result — the "
        "definitive test is the multi-seed mid/node grid. Mediation (augmentation -> "
        "invariance -> F1) is not claimable from one seed: property-metric co-movement "
        "is reported as SUGGESTIVE at most, mediation UNEXPLAINED; the product-of-paths "
        "a*b analysis with entity-bootstrap CIs over the train-side augmentation-rate "
        "dial is the node run."
    ),
    registry=registry,
)

# %%
_ = conjecture_card(
    card_id="TRN-04",
    conjecture=(
        "Typo robustness can be bought in the tokenizer instead of the training data: a "
        "from-scratch byte-level encoder, for which a one-character typo perturbs one "
        "or two input ids, degrades more gracefully under the typo dial than a "
        "pretrained subword encoder, whose tokenizer can shatter a typo'd name into "
        "unrelated pieces — while the pretrained model's semantic head start shows up "
        "as higher clean-eval F1 at matched step budget."
    ),
    pressure=(
        "encoder regime {scratch char/byte, pretrained subword (all-MiniLM-L6-v2)} at "
        "each regime's own winning TRN-03 augmentation recipe, identical optimizer-step "
        "budget asserted from the histories. Budget accounting stated, not hidden: this "
        "is STEP-matched; parameter counts differ by construction and are printed with "
        "the rows (PLAN section 3.1's matched-parameter reading is the mac/node run's "
        "job, where both regimes are live)."
    ),
    property=(
        "the typo-dial AUC curve's decay from rate 0.02 to 0.2 (flatter = "
        "tokenizer-robust), plus the battery's should-hold cosines and the geometry "
        "collapse guards"
    ),
    metric=(
        "B-cubed F1 with entity-unit BCa 95% CI at the precision-0.99 operating point "
        "on the entity-disjoint eval slice — the same shared recipe as every other arm "
        "in the series"
    ),
    prediction=(
        "(a) pretrained-subword posts higher clean B-cubed F1 at matched step budget; "
        "(b) scratch-char shows the flatter typo-dial AUC decay; (c) calibrated "
        "augmentation narrows the regimes' robustness gap — augmentation can buy back "
        "for the subword model what the byte tokenizer gets for free. Decision rule: "
        "scoring any of a/b/c needs BOTH regimes live. huggingface.co is blocked from "
        "the smoke container (notes/COMPAT.md), so at smoke only the scratch side runs "
        "and the verdict is PARTIAL by construction — recorded as UNEXPLAINED, with the "
        "pretrained rows ABSENT from the registered artifact (never fabricated). The "
        "head-to-head is decided by the RUN-IN-TARGET mac cell, with system-metric "
        "margins read against the MET-04 bar at that tier."
    ),
    registry=registry,
)

# %% [markdown]
# ## 3. TRN-03 — three doses of noise, one budget
#
# ### 3a. What each arm actually injects
#
# The **calibrated** arm's rates come verbatim from the calibrated corpus meta (NSE-01
# provenance chain intact), projected onto the augmenter's channel registry exactly as
# notebook 09's regeneration did: the temporal-drift channels (`move*`, `name_change_*`,
# `name_format_drift_*`) are notebook-05 pool-redraw machinery with no train-time
# augmenter equivalent, and stay out — a documented exclusion, not a silent one. Registry
# channels apply their default field sets (the same `make_augmenter` path the training
# loop itself calls; the loop cannot pass the fetched nickname lexicon, so the nickname
# channel falls back to its built-in mini-lexicon — a package gap carried on the record).
# The **generic** arm is the augment module's own flat default: a typo dose orders of
# magnitude above the measured floor, deliberately uncalibrated — that contrast IS the
# arm. The **none** arm is the identity.

# %%
t_sec = time.time()
CAL_RATES = {k: float(v) for k, v in corpus_meta["extra"]["channel_rates"].items()}
AUG_RATES = {
    "typo": CAL_RATES.get("typo_given", 0.0) + CAL_RATES.get("typo_family", 0.0),
    "nickname": CAL_RATES.get("nickname", 0.0),
    "name_order_swap": CAL_RATES.get("name_order_swap", 0.0),
    "field_dropout": CAL_RATES.get("dropout_given", 0.0),
    "hub_value": CAL_RATES.get("hub_value", 0.0),
}
AUG_RATES = {k: v for k, v in AUG_RATES.items() if v > 0.0}
EXCLUDED = sorted(
    k for k in CAL_RATES
    if k.startswith(("move", "name_change", "name_format_drift"))
)
GENERIC_RATES = dict(augment_mod._GENERIC_TYPO_RATES)  # the module's documented default
print("calibrated augmentation rates (registry-channel projection, provenance = corpus meta):")
for k, v in AUG_RATES.items():
    print(f"  {k:>16}: {v:.6f}  (per record, per batch pass)")
print(f"excluded drift channels (no train-time augmenter equivalent): {EXCLUDED}")
print(f"generic_typo arm rate: {GENERIC_RATES} — "
      f"{GENERIC_RATES['typo'] / AUG_RATES['typo']:,.0f}x the measured typo floor "
      "(the measured rate is itself a lower bound; corpus meta caveat)")

cfg_calibrated = copy.deepcopy(cfg)
# the typed TrainConfig node is struct-locked; force_add is the documented extension-key path
OmegaConf.update(cfg_calibrated, "train.channel_rates", dict(AUG_RATES), force_add=True)

ARM_STATE: dict[str, dict] = {}
trn03_rows: list[dict] = []


def run_aug_arm(aug: str, *, regime: str = "scratch_char", cfg_arm=None, encoder=None) -> dict:
    """Train + fully instrument one TRN-03/TRN-04 arm; returns the matrix row."""
    if cfg_arm is None:
        cfg_arm = cfg_calibrated if aug == "calibrated" else cfg
    enc_a, hist_a, secs = train_arm(augment=aug, seed=PRIMARY_SEED, cfg_arm=cfg_arm,
                                    encoder=encoder)
    sysrow = system_metric(enc_a, label=f"trn03/{regime}/{aug}")
    rep, geo, geo_emb = property_panel(enc_a)
    dial = dial_curve(enc_a)

    def slice_val(name: str, col: str) -> float:
        return float(rep.loc[name, col]) if name in rep.index else float("nan")

    row = {
        "arm": aug, "regime": regime, "loss": LOSS_LOCKED, "miner": "inbatch",
        "seed": PRIMARY_SEED, "steps": len(hist_a), "batch": BATCH, "train_secs": secs,
        "final_train_loss": float(hist_a["loss"].tail(20).mean()),
        "n_params": int(sum(p.numel() for p in enc_a.parameters())),
        **{k: sysrow[k] for k in ("f1", "f1_lo", "f1_hi", "recall_at",
                                  "attained_precision", "attained", "threshold")},
        **geo,
        "typo05_cos": slice_val("typo@0.05", "mean_cos"),
        "typo20_cos": slice_val("typo@0.2", "mean_cos"),
        "nickname_cos": slice_val("nickname_swap", "mean_cos"),
        "dob_auc": slice_val("different_birth_year", "auc_vs_true"),
        "twin_auc": slice_val("twin_household", "auc_vs_true"),
    }
    for _, d in dial.iterrows():
        row[f"dial_auc@{d['rate']:g}"] = float(d["auc"])
        row[f"dial_cos@{d['rate']:g}"] = float(d["mean_cos"])
    ARM_STATE[f"{regime}/{aug}"] = {
        "encoder": enc_a, "history": hist_a, "battery": rep, "geo_emb": geo_emb,
        "dial": dial, "sysrow": sysrow,
    }
    print(f"  [{regime}/{aug}] B3F1={row['f1']:.4f} [{row['f1_lo']:.4f},{row['f1_hi']:.4f}] "
          f"(P {row['attained_precision']:.4f}, R {row['recall_at']:.4f}) | "
          f"dialAUC@0.2={row['dial_auc@0.2']:.3f} typo20_cos={row['typo20_cos']:.3f} | "
          f"align={row['align']:.3f} rankme={row['rankme']:.1f} | train {secs:.0f}s")
    return row


tick("§3a augmentation recipes", t_sec)

# %% [markdown]
# ### 3b. The three arms
#
# One arm per cell (per-cell-timeout hygiene, as notebook 09). All three start from the
# *identical* initialization (same seed → same weights), so every downstream delta is the
# augmentation and nothing else. Watch the per-arm print: the dial AUC and the battery
# cosines are the property half of the card's chain, the B³F1 the metric half.

# %%
t_sec = time.time()
trn03_rows.append(run_aug_arm("none"))

# %%
trn03_rows.append(run_aug_arm("generic_typo"))

# %%
trn03_rows.append(run_aug_arm("calibrated"))
step_counts = {r["arm"]: r["steps"] for r in trn03_rows}
assert len(set(step_counts.values())) == 1, f"budget mismatch across arms: {step_counts}"
print(f"\nbudget-matched: every arm ran exactly {trn03_rows[0]['steps']} optimizer steps "
      "(asserted from each returned history)")
tick("§3b TRN-03 arms", t_sec)

# %%
t_sec = time.time()
trn03 = pd.DataFrame(trn03_rows)
registry.register(
    "trn03_augmentation_matrix", trn03, cfg=cfg, tier=cfg.run.tier,
    meta={
        "design": "TRN-03 smoke slice: augmentation {none, generic_typo, calibrated} x "
                  "scratch_char regime x 1 seed, budget-matched at exactly "
                  f"{int(trn03['steps'].iloc[0])} optimizer steps (asserted); loss/miner "
                  "locked at pre-registered defaults (infonce, in-batch); identical init",
        "augmentation": {
            "calibrated_rates": AUG_RATES,
            "calibrated_provenance": corpus_meta["extra"]["channel_rate_provenance"],
            "excluded_drift_channels": EXCLUDED,
            "generic_rates": GENERIC_RATES,
            "nickname_lexicon": "train-loop augmenter cannot receive the fetched "
                                "lexicon (package gap) -> channel built-in mini-lexicon",
            "rate_caveat": "measured typo rates are NSE-01 drift-visible LOWER BOUNDS "
                           "(corpus meta caveat carried verbatim)",
        },
        "protocol": {"operating_point": f"precision@{PREC_TARGET} (loud fallback)",
                     "ci": {"unit": "entity", "method": "bca", "n_boot": N_BOOT},
                     "candidates": f"ann.candidates(k={CAND_K}, index='flat')",
                     "grid": GRID},
        "dial_auc": "typo'd same-record pairs vs fixed cross-entity impostor pairs, "
                    f"Mann-Whitney AUC at dial rates {list(DIAL_RATES)} "
                    f"({TRUE_PAIR_N} impostor pairs, probes <= {PROBE_N}/slice)",
        "met04_sd_replicate": SD_REPLICATE, "met04_detect_bar": DETECT_BAR,
        "single_seed_caveat": "one seed per cell — DEMONSTRATION rows; the definitive "
                              "multi-seed 6-cell grid is tier=mid/node (placard in this "
                              "notebook)",
        "split": {"scheme": SCHEME, "eval_records": len(ev_slice),
                  "eval_entities": int(ev_slice["entity_id"].nunique())},
        "corpus_provenance": CORPUS_PROVENANCE,
        "truncation": {"max_len": int(cfg.model.max_len),
                       "share_eval_texts_truncated": share_trunc},
    },
)
print(f"registered trn03_augmentation_matrix: {len(trn03)} rows")
show_cols = ["arm", "f1", "f1_lo", "f1_hi", "recall_at", "attained_precision",
             "dial_auc@0.02", "dial_auc@0.2", "typo20_cos", "nickname_cos", "dob_auc",
             "twin_auc", "align", "uniform", "rankme", "hub_skew", "final_train_loss",
             "train_secs"]
display(trn03[show_cols].round(4))

dial_long = pd.concat(
    [ARM_STATE[f"scratch_char/{a}"]["dial"].assign(arm=a) for a in AUG_ARMS],
    ignore_index=True,
)
registry.register(
    "trn03_typo_dial", dial_long, cfg=cfg, tier=cfg.run.tier,
    meta={"note": "typo-dial AUC curves per TRN-03 arm (typo'd pair vs impostor pair "
                  "separation at each per-char dial rate); supports "
                  "trn03_augmentation_matrix — same probes/impostors for every arm",
          "single_seed_caveat": "1 seed; property curves have no registered replicate "
                                "sigma at smoke — read descriptively"},
)
print(f"registered trn03_typo_dial ({len(dial_long)} rows)")
tick("§3c matrix registration", t_sec)

# %%
fig = figures.line_with_ci(
    registry, tier=cfg.run.tier, artifact="trn03_typo_dial",
    x="rate", y="auc", hue="arm",
    title="The property under pressure: typo-dial AUC per augmentation arm (1 seed)",
    xlabel="typo dial rate (per-char corruption of the serialized record)",
    ylabel="AUC: typo'd pair vs impostor pair",
    figsize=(6.6, 4.3),
)

# %% [markdown]
# ### 3d. The chain in one figure
#
# The lab's signature triptych: **pressure → property → metric**, side by side, all three
# panels from registered artifacts. Left: the property the pressure was supposed to move
# (dial AUC at the heaviest rate, per arm). Middle: what the winning arm's embedding
# space looks like (PCA-2d of the shared geometry sample, colored by record kind).
# Right: the system metric with its entity-BCa band per arm. Reading rule, pre-committed:
# co-movement of left and right panels at one seed is SUGGESTIVE only — the triptych
# *shows* a chain, it does not *test* mediation (that is the node grid's a·b analysis).

# %%
t_sec = time.time()
WIN_ARM = str(trn03.sort_values("f1", ascending=False)["arm"].iloc[0])
print(f"single-seed best TRN-03 arm by B3F1: {WIN_ARM} (a choice read through the MET-04 "
      "bar in the verdict below — inside the bar it is a coin-flip, and says so)")
prop_frame = trn03[["arm", "dial_auc@0.2"]].rename(columns={"arm": "x", "dial_auc@0.2": "y"})
geo_xy = pca2(ARM_STATE[f"scratch_char/{WIN_ARM}"]["geo_emb"])
geo_frame = pd.DataFrame({"x": geo_xy[:, 0], "y": geo_xy[:, 1], "group": GEO_KIND})
sys_frame = trn03[["arm", "f1", "f1_lo", "f1_hi"]].rename(
    columns={"arm": "x", "f1": "y", "f1_lo": "lo", "f1_hi": "hi"})
registry.register("trn03_panel_property", prop_frame, cfg=cfg, tier=cfg.run.tier,
                  meta={"note": "TRN-03 triptych left: dial AUC@0.2 per augmentation arm"})
registry.register("trn03_panel_geometry", geo_frame, cfg=cfg, tier=cfg.run.tier,
                  meta={"note": f"TRN-03 triptych middle: PCA-2d of the {WIN_ARM} arm's "
                                "eval-sample embeddings, colored by record kind"})
registry.register("trn03_panel_system", sys_frame, cfg=cfg, tier=cfg.run.tier,
                  meta={"note": "TRN-03 triptych right: B3F1 + entity BCa CI per arm"})
fig = figures.three_panel_pressure(
    registry, tier=cfg.run.tier,
    property_shift="trn03_panel_property", geometry="trn03_panel_geometry",
    system_metric="trn03_panel_system",
    titles=("typo-dial AUC@0.2 by augmentation",
            f"embedding space: {WIN_ARM} arm (PCA-2d)",
            "B³F1 @ 0.99-precision point (BCa 95%)"),
    suptitle="TRN-03 chain, single seed: augmentation → typo invariance → system metric",
)
tick("§3d triptych", t_sec)

# %% [markdown]
# ## 4. The eval-side dial — dose-response on the cheap
#
# The card's dose-response instrument, smoke edition. PLAN §3.1's dial is *train-side*
# (augmentation rate within the winning cell — three more trainings per level per seed,
# the node run's mediation input). What smoke can afford is the **eval-side dial**: hold
# each trained encoder fixed, re-corrupt the *same* eval records at 0.5×/1×/2× the
# measured channel rates (fresh corruption on top of the corpus's own realistic noise —
# so ×1 means "one extra measured dose", not "the corpus's total noise"), and re-run the
# shared recipe on the re-serialized texts. Every arm sees the identical corrupted
# variants (one draw per intensity, shared), so the lines are paired. One exploratory
# **stress point at ×10** is added *because the measured rates are lower bounds* (corpus
# meta): at ×2 the extra dose touches ~2% of records and an honest near-null is the
# likely reading; ×10 shows whether the machinery detects degradation at all. It is
# flagged `stress=True` in the artifact and read as exploratory, never as the card's
# prescribed dial.

# %%
t_sec = time.time()
EVAL_VARIANTS: dict[float, list[str]] = {0.0: EVAL_TEXTS}
for mult in (*EVAL_NOISE_MULTS, STRESS_MULT):
    rates = {k: min(1.0, v * mult) for k, v in AUG_RATES.items()}
    corrupt = make_augmenter("calibrated", channel_rates=rates,
                             seed=int(90_000 + 10 * mult))
    with np.errstate(divide="ignore"):  # gecko keymap: benign 0-candidate divide
        frame = corrupt(ev_slice.copy())
    assert list(frame["record_id"]) == list(ev_slice["record_id"]), "dial must not reorder"
    EVAL_VARIANTS[mult] = texts_of(frame)
    changed = sum(a != b for a, b in zip(EVAL_VARIANTS[mult], EVAL_TEXTS))
    print(f"  x{mult:g}: {changed:,}/{len(EVAL_TEXTS):,} eval texts changed "
          f"({changed / len(EVAL_TEXTS):.2%}) by the added dose")

dial_sys_rows: list[dict] = []
for arm in AUG_ARMS:
    enc_d = ARM_STATE[f"scratch_char/{arm}"]["encoder"]
    for mult, txts in EVAL_VARIANTS.items():
        if mult == 0.0:
            s = ARM_STATE[f"scratch_char/{arm}"]["sysrow"]  # the headline eval, reused
        else:
            s = system_metric(enc_d, label=f"dial x{mult:g}/{arm}", texts=txts)
        dial_sys_rows.append({
            "arm": arm, "mult": float(mult), "stress": bool(mult >= STRESS_MULT),
            "f1": s["f1"], "f1_lo": s["f1_lo"], "f1_hi": s["f1_hi"],
            "recall_at": s["recall_at"], "attained_precision": s["attained_precision"],
            "attained": s["attained"], "threshold": s["threshold"],
            "eval_secs": s["eval_secs"],
        })
dial_sys = pd.DataFrame(dial_sys_rows)
registry.register(
    "trn03_eval_noise_dial", dial_sys, cfg=cfg, tier=cfg.run.tier,
    meta={
        "note": "EVAL-SIDE dial (documented as such): each trained TRN-03 arm re-scored "
                "by the shared recipe on the same eval records re-corrupted at "
                f"{list(EVAL_NOISE_MULTS)}x the measured channel rates (one shared "
                f"corruption draw per intensity, paired across arms); mult=0 is the "
                "headline uncorrupted eval; the x"
                f"{STRESS_MULT:g} row is an EXPLORATORY stress point (stress=True) "
                "motivated by the rates being lower bounds",
        "not_the_plan_dial": "PLAN section 3.1's dose-response dial is TRAIN-side "
                             "(augmentation rate within the winning cell) — that is the "
                             "node run's mediation input, placarded below",
        "added_rates": {f"x{m:g}": {k: min(1.0, v * m) for k, v in AUG_RATES.items()}
                        for m in (*EVAL_NOISE_MULTS, STRESS_MULT)},
        "met04_detect_bar": DETECT_BAR,
        "single_seed_caveat": "1 seed per arm; CI bands are eval-resampling only",
    },
)
print(f"\nregistered trn03_eval_noise_dial ({len(dial_sys)} rows)")
display(dial_sys.pivot(index="mult", columns="arm", values="f1").round(4))
tick("§4 eval-side dial", t_sec)

# %%
fig = figures.line_with_ci(
    registry, tier=cfg.run.tier, artifact="trn03_eval_noise_dial",
    x="mult", y="f1", lo="f1_lo", hi="f1_hi", hue="arm",
    title=f"Eval-side dial: added corruption dose vs B³F1 (x{STRESS_MULT:g} = stress point)",
    xlabel="added eval corruption (multiple of measured channel rates)",
    ylabel=f"B³F1 @ precision-{PREC_TARGET} point (entity BCa 95%)",
    figsize=(6.8, 4.4),
)

# %%
# [RUN-IN-TARGET node] definitive TRN-03: the locked 6-cell grid (3 augmentation levels x
# 2 encoder regimes) x replicates from met04_power_table, PLUS the train-side
# dose-response dial (calibrated augmentation rate at e.g. 0.5x/1x/2x, trained per level)
# that feeds the a*b mediation analysis. The estimate uses THIS run's measured
# coefficients. TRN-03 sits second in PLAN §3's protected-arm order (TRN-02 > TRN-03
# chain > TRN-04) — it survives most budget cuts.
if TIER in ("mid", "target"):
    print(f"tier={TIER}: rows above are definitive-tier cells for the regimes that ran; "
          "replicate counts come from met04_power_table at this tier.")
else:
    n_star = SEEDS_NEEDED[0.02]
    per_arm = float(trn03["train_secs"].mean())
    per_eval = float(dial_sys["eval_secs"].mean())
    runs = 6 * n_star + 3 * n_star  # 6 factorial cells + 3 train-side dial levels
    est_h = runs * (per_arm * (STEPS_BY_TIER["target"] / STEPS) + per_eval) / 3600
    print(f"[RUN-IN-TARGET node] definitive TRN-03 = 6 cells x n*={n_star} replicates "
          f"(met04_power_table @ delta 0.02) + train-side dial 3 levels x n*={n_star} "
          f"= {runs} runs. Measured here: {per_arm:.0f}s/arm at {STEPS} steps, "
          f"{per_eval:.0f}s/eval -> ~{est_h:.1f} container-hours naive at "
          f"{STEPS_BY_TIER['target']} steps; the node's A100s are the intended home "
          "(PLAN §8 books the full TRN factorial program at ~130-170 A100-h "
          "pre-pruning, which the power table prunes).")

# %% [markdown]
# ### 4b. TRN-03 verdict — through the MET-04 lens

# %%
by03 = trn03.set_index("arm")
f1_none = float(by03.loc["none", "f1"])
f1_gen = float(by03.loc["generic_typo", "f1"])
f1_cal = float(by03.loc["calibrated", "f1"])
p20 = {a: float(by03.loc[a, "dial_auc@0.2"]) for a in AUG_ARMS}
order_sys = f1_cal > f1_gen > f1_none
order_prop = p20["calibrated"] > p20["generic_typo"] > p20["none"]
margin_cal_none = f1_cal - f1_none
sig_gate = margin_cal_none > DETECT_BAR
reversal = (f1_none - f1_cal) > DETECT_BAR
if order_sys and order_prop and sig_gate:
    trn03_outcome = "CONFIRMED"
elif reversal:
    trn03_outcome = "REFUTED"
else:
    trn03_outcome = "UNEXPLAINED"

co_move = float(pd.Series([p20[a] for a in AUG_ARMS]).corr(
    pd.Series([float(by03.loc[a, "f1"]) for a in AUG_ARMS]), method="spearman"))
drops = dial_sys[~dial_sys["stress"]].pivot(index="mult", columns="arm", values="f1")
drop2 = (drops.loc[0.0] - drops.loc[2.0]).to_dict()
stress_f1 = dial_sys[dial_sys["stress"]].set_index("arm")["f1"].to_dict()
drop_stress = {a: float(by03.loc[a, "f1"]) - stress_f1[a] for a in AUG_ARMS}
cal_flattest_2x = min(drop2, key=drop2.get) == "calibrated"

print(f"system ranking: {by03['f1'].sort_values(ascending=False).round(4).to_dict()}")
print(f"property (dialAUC@0.2): { {a: round(v, 3) for a, v in p20.items()} }")
print(f"predicted order calibrated>generic>none — system: {order_sys}, property: {order_prop}")
print(f"calibrated-over-none margin {margin_cal_none:+.4f} vs MET-04 bar {DETECT_BAR:.4f} "
      f"-> {'EXCEEDS' if sig_gate else 'INSIDE the bar (noise-bound)'}")
print(f"property-metric co-movement (Spearman over 3 arms): {co_move:+.2f} — SUGGESTIVE "
      "at most; 3 points, 1 seed")
print(f"eval dial drop 0->2x: { {a: round(v, 4) for a, v in drop2.items()} } "
      f"(calibrated flattest: {cal_flattest_2x}); stress 0->{STRESS_MULT:g}x drop: "
      f"{ {a: round(v, 4) for a, v in drop_stress.items()} }")

_ = verdict_box(
    "TRN-03",
    outcome=trn03_outcome,
    evidence=(
        f"trn03_augmentation_matrix + trn03_typo_dial + trn03_eval_noise_dial (tier "
        f"{TIER}, scratch-char regime, 1 seed, {int(trn03['steps'].iloc[0])} matched "
        f"steps, identical init). System: none {f1_none:.4f}, generic {f1_gen:.4f}, "
        f"calibrated {f1_cal:.4f}; predicted ordering holds on system: {order_sys}, on "
        f"property (dialAUC@0.2 {p20['none']:.3f}/{p20['generic_typo']:.3f}/"
        f"{p20['calibrated']:.3f}): {order_prop}. Calibrated-over-none margin "
        f"{margin_cal_none:+.4f} vs the pre-registered MET-04 bar {DETECT_BAR:.4f} "
        f"(sd_replicate {SD_REPLICATE:.4f}) — "
        + ("the margin clears the bar, so the smoke slice supports scoring the "
           "ordering. " if sig_gate else
           "the margin sits INSIDE the bar: by the card's own decision rule the "
           "system-metric ordering is noise-bound at one seed — the expected honest "
           "outcome the card itself predicted, given the calibrated dose is built from "
           "drift-visible LOWER-BOUND rates. ")
        + f"Eval-side dial (documented as eval-side, not the PLAN train-side dial): "
        f"F1 drop from x0 to x2 added dose = none {drop2['none']:+.4f} / generic "
        f"{drop2['generic_typo']:+.4f} / calibrated {drop2['calibrated']:+.4f} "
        f"(calibrated flattest: {cal_flattest_2x}); exploratory x{STRESS_MULT:g} stress "
        f"drops none {drop_stress['none']:+.4f} / generic "
        f"{drop_stress['generic_typo']:+.4f} / calibrated "
        f"{drop_stress['calibrated']:+.4f}. Property-metric co-movement (Spearman over "
        f"the 3 arms) {co_move:+.2f}: SUGGESTIVE at most. Mediation augmentation -> "
        f"invariance -> F1: UNEXPLAINED — one seed cannot establish it, the two "
        f"marginal effects are reported separately (PLAN §5, adoption gate vs "
        f"mediation), and the a*b product-of-paths analysis over the train-side "
        f"augmentation-rate dial is the node run (placard above; 6 cells + dial x "
        f"n*={SEEDS_NEEDED[0.02]} replicates). The definitive multi-seed matrix is "
        f"tier=mid/node; this row set is a demonstration."
    ),
    registry=registry,
)

# %% [markdown]
# ## 5. TRN-04 — the head-to-head: buy robustness in the tokenizer, or the data?
#
# ### 5a. The scratch side, live — from TRN-03's winning arm
#
# TRN-04's smoke half costs nothing new: the scratch-char contender is TRN-03's winning
# arm, reused with all its instruments (stated, not retrained — identical configuration
# by construction). The pretrained-subword contender needs weights this container cannot
# fetch (huggingface.co blocked, `notes/COMPAT.md` go/no-go tree), so its cell below is
# *real gated code*: on the mac (or with `paths.hf_local` set) it trains all three
# augmentation recipes on the pretrained regime at the same step budget, picks that
# regime's own winner, and re-registers the head-to-head with both regimes live. Here it
# prints the COMPAT placard — and the registered artifact records the absence honestly
# instead of fabricating a row.

# %%
t_sec = time.time()
HEAD_COLS = ["regime", "arm", "f1", "f1_lo", "f1_hi", "recall_at", "attained_precision",
             "attained", "threshold", "steps", "batch", "train_secs", "n_params",
             "align", "uniform", "rankme", "hub_skew", "typo05_cos", "typo20_cos",
             "nickname_cos", "dob_auc", "twin_auc"] + \
    [f"dial_auc@{r:g}" for r in DIAL_RATES] + [f"dial_cos@{r:g}" for r in DIAL_RATES]
head_rows: list[dict] = [
    {**{c: r[c] for c in HEAD_COLS}, "recipe": r["arm"], "reused_from_trn03": True}
    for r in trn03_rows
]
scratch_params = int(trn03["n_params"].iloc[0])
print(f"scratch-char rows adopted from TRN-03 (reuse stated): {len(head_rows)} arms, "
      f"{scratch_params:,} parameters each, {STEPS} steps each")
print(f"scratch regime's winning recipe (by single-seed F1): {WIN_ARM}")
tick("§5a scratch side", t_sec)

# %%
# [RUN-IN-TARGET mac] the pretrained-subword side of TRN-04. huggingface.co is blocked
# from the smoke container (notes/COMPAT.md go/no-go tree), so this cell runs where
# weights can load: the mac/node, or here if paths.hf_local points at offline weights.
# Same arm loop, same instruments, same step budget — real code gated on availability.
t_sec = time.time()
PRETRAINED_READY = cfg.paths.hf_local is not None or TIER in ("mid", "target")
if PRETRAINED_READY:
    for aug_kind in AUG_ARMS:
        cfg_pre = copy.deepcopy(cfg)
        cfg_pre.model.kind = "pretrained"
        if aug_kind == "calibrated":
            OmegaConf.update(cfg_pre, "train.channel_rates", dict(AUG_RATES),
                             force_add=True)
        set_all_seeds(PRIMARY_SEED)
        enc_pre = build_encoder(cfg_pre)  # raises with the COMPAT placard if weights absent
        row = run_aug_arm(aug_kind, regime="pretrained", cfg_arm=cfg_pre, encoder=enc_pre)
        head_rows.append({**{c: row[c] for c in HEAD_COLS},
                          "recipe": aug_kind, "reused_from_trn03": False})
else:
    print("[RUN-IN-TARGET mac] pretrained-subword arms skipped here: no paths.hf_local "
          f"and tier={TIER} has no huggingface egress (notes/COMPAT.md). Rerun THESE "
          "cells at tier=mid on the mac (or set paths.hf_local) to fill the pretrained "
          "rows; the artifact registered below records their ABSENCE — no fabricated "
          f"rows. Measured here: a scratch arm costs {float(trn03['train_secs'].mean()):.0f}s "
          f"at {STEPS} steps with {scratch_params:,} params; all-MiniLM-L6-v2 is orders "
          "of magnitude larger (its parameter count prints from this same loop when the "
          "mac cell runs) — budget accordingly.")
tick("§5b pretrained gate", t_sec)

# %%
t_sec = time.time()
trn04 = pd.DataFrame(head_rows)
assert trn04["steps"].nunique() == 1, "TRN-04 regimes must share one step budget"
trn04["winner"] = False
for regime, grp in trn04.groupby("regime"):
    trn04.loc[grp["f1"].idxmax(), "winner"] = True
LIVE_REGIMES = sorted(trn04["regime"].unique())
pretrained_live = "pretrained" in LIVE_REGIMES
registry.register(
    "trn04_encoder_head_to_head", trn04, cfg=cfg, tier=cfg.run.tier,
    meta={
        "design": "TRN-04 head-to-head: encoder regime {scratch_char, pretrained} x the "
                  "TRN-03 augmentation recipes, 1 seed, budget-matched at exactly "
                  f"{int(trn04['steps'].iloc[0])} optimizer steps (asserted); each "
                  "regime's winner flagged — the head-to-head is winner vs winner "
                  "(PLAN §3.1: each regime at its own winning training recipe)",
        "budget_accounting": "STEP-matched (the lab's checkable unit); parameter counts "
                             "per row in n_params — the regimes differ by construction "
                             "and the matched-parameter reading is the mac/node run's",
        "live_regimes": LIVE_REGIMES,
        "pretrained_rows": (
            "present" if pretrained_live else
            "ABSENT at smoke — NOT fabricated. huggingface.co is blocked from this "
            "container (notes/COMPAT.md); the # [RUN-IN-TARGET mac] cell in this "
            "notebook trains the three pretrained-subword arms at the same step budget "
            "and re-registers this artifact with both regimes live"
        ),
        "scratch_reuse": "scratch rows adopt TRN-03's trained arms verbatim (identical "
                         "configuration, stated; reused_from_trn03 column)",
        "protocol": {"operating_point": f"precision@{PREC_TARGET} (loud fallback)",
                     "ci": {"unit": "entity", "method": "bca", "n_boot": N_BOOT},
                     "candidates": f"ann.candidates(k={CAND_K}, index='flat')"},
        "met04_sd_replicate": SD_REPLICATE, "met04_detect_bar": DETECT_BAR,
        "single_seed_caveat": "one seed per cell — demonstration rows; the definitive "
                              "matched-budget head-to-head is tier=mid/node",
        "corpus_provenance": CORPUS_PROVENANCE,
    },
)
print(f"registered trn04_encoder_head_to_head: {len(trn04)} rows, regimes {LIVE_REGIMES} "
      f"(pretrained rows: {'live' if pretrained_live else 'absent, placarded in meta'})")
display(trn04[["regime", "recipe", "winner", "f1", "f1_lo", "f1_hi", "dial_auc@0.02",
               "dial_auc@0.2", "n_params", "steps", "reused_from_trn03"]].round(4))
tick("§5c head-to-head registration", t_sec)


# %%
def draw_head_to_head(ax, df, meta):
    rates = sorted(float(c.split("@")[1]) for c in df.columns if c.startswith("dial_auc@"))
    for _, r in df[df["winner"]].sort_values("regime").iterrows():
        y = [float(r[f"dial_auc@{rt:g}"]) for rt in rates]
        ax.plot(rates, y, marker="o",
                label=(f"{r['regime']} ({r['recipe']}): "
                       f"F1 {r['f1']:.3f} [{r['f1_lo']:.3f},{r['f1_hi']:.3f}]"))
    if "pretrained" not in set(df["regime"]):
        ax.text(0.98, 0.05,
                "pretrained regime ABSENT at smoke\n(# [RUN-IN-TARGET mac] — see meta)",
                transform=ax.transAxes, ha="right", va="bottom", fontsize=8, color="0.35")
    ax.set_ylim(0.45, 1.02)
    ax.set_xlabel("typo dial rate (per-char)")
    ax.set_ylabel("dial AUC: typo'd pair vs impostor")
    ax.legend(loc="lower left", fontsize=8, title="regime winner (1 seed)")


fig = figures.plot_artifact(
    registry, tier=cfg.run.tier, artifact="trn04_encoder_head_to_head",
    draw=draw_head_to_head,
    title="TRN-04 head-to-head, live rows only: typo-dial decay per regime winner",
    figsize=(6.8, 4.4),
)

# %% [markdown]
# ### 5d. TRN-04 verdict — PARTIAL at smoke, and that is the point
#
# The tier rails exist precisely so this box can be honest: one regime ran, so the
# head-to-head's predictions (a)/(b)/(c) are *unscoreable* here — not confirmed, not
# refuted, and not padded with a fabricated pretrained row. The scratch side's numbers
# stand ready for the comparison the mac cell completes.

# %%
win04 = trn04[trn04["winner"] & (trn04["regime"] == "scratch_char")].iloc[0]
scratch_decay = float(win04["dial_auc@0.02"] - win04["dial_auc@0.2"])
print(f"scratch winner: {win04['recipe']} — F1 {win04['f1']:.4f} "
      f"[{win04['f1_lo']:.4f},{win04['f1_hi']:.4f}], dialAUC 0.02->0.2: "
      f"{win04['dial_auc@0.02']:.3f} -> {win04['dial_auc@0.2']:.3f} "
      f"(decay {scratch_decay:.3f}), {int(win04['n_params']):,} params")
if pretrained_live:
    win_pre = trn04[trn04["winner"] & (trn04["regime"] == "pretrained")].iloc[0]
    pre_decay = float(win_pre["dial_auc@0.02"] - win_pre["dial_auc@0.2"])
    delta_f1 = float(win_pre["f1"] - win04["f1"])
    scored = (f"BOTH regimes live: pretrained winner {win_pre['recipe']} F1 "
              f"{win_pre['f1']:.4f} [{win_pre['f1_lo']:.4f},{win_pre['f1_hi']:.4f}] "
              f"({int(win_pre['n_params']):,} params), delta vs scratch "
              f"{delta_f1:+.4f} vs bar {DETECT_BAR:.4f} "
              f"({'clears' if abs(delta_f1) > DETECT_BAR else 'inside'}); dial decay "
              f"scratch {scratch_decay:.3f} vs pretrained {pre_decay:.3f} (prediction "
              f"(b) flatter-scratch: {scratch_decay < pre_decay}). Still 1 seed — the "
              "multi-seed grid at this tier is the definitive scorer. ")
else:
    scored = ("PARTIAL AT SMOKE — one regime live. Predictions (a)/(b)/(c) all require "
              "the pretrained-subword side, whose weights this container cannot fetch "
              "(huggingface.co blocked, notes/COMPAT.md); its rows are ABSENT from "
              "trn04_encoder_head_to_head by design, never fabricated, and the "
              "# [RUN-IN-TARGET mac] cell above is the real code path that fills them. ")
_ = verdict_box(
    "TRN-04",
    outcome="UNEXPLAINED",
    evidence=(
        f"trn04_encoder_head_to_head (tier {TIER}, regimes live: {LIVE_REGIMES}, "
        f"{int(trn04['steps'].iloc[0])} matched steps, budget asserted). " + scored +
        f"Scratch side on record: winning recipe {win04['recipe']}, B3F1 "
        f"{win04['f1']:.4f} [{win04['f1_lo']:.4f},{win04['f1_hi']:.4f}] at the "
        f"precision-{PREC_TARGET} point, typo-dial AUC {win04['dial_auc@0.02']:.3f} -> "
        f"{win04['dial_auc@0.2']:.3f} across the dial (decay {scratch_decay:.3f}), "
        f"{int(win04['n_params']):,} parameters, {int(win04['steps'])} steps. Budget "
        f"accounting: step-matched; parameter counts differ by regime and are printed, "
        f"per the card. Read through the MET-04 lens (bar {DETECT_BAR:.4f}) whenever "
        f"both regimes are present. This PARTIAL outcome is the honesty rail working — "
        f"the comparison completes on the mac, then multi-seed on the node."
    ),
    registry=registry,
)

# %% [markdown]
# ## 6. What the answer means at 1e9 — and what downstream inherits
#
# **The cost asymmetry the 1e9 design has to price** (qualitative here; SCL-03 in
# notebook 15 owns the priced model — no numbers invented in prose). The two places to
# buy typo robustness age differently at scale:
#
# - **Tokenizer route (byte/char).** The robustness is a *construction-time property of
#   the input encoding*: no vocabulary to build, version, or migrate when the corpus
#   grows or a new source lands; a typo'd byte sequence is near its clean twin for free,
#   in every deployment, forever. Its price is paid elsewhere — the scratch model starts
#   from zero semantics, which is exactly what TRN-04's pretrained side is there to
#   measure against.
# - **Augmentation route (calibrated).** The invariance is only as good as the
#   *calibration*, and calibration is a per-deployment measurement: notebook 04's audit
#   contract requires a stable key and at least two temporal versions of *your* data
#   before NSE-01-style rates exist at all, the measured rates are lower bounds (this
#   notebook carried that caveat into its own verdict), and rates go stale as sources
#   drift — a recurring cost at 1e8–1e9, not a one-time one. Generic augmentation waives
#   the calibration bill but pays in dose mis-weighting (the generic:calibrated typo
#   ratio printed in §3a is the measured size of that gap at smoke).
#
# The two routes are complements, not substitutes — which is why TRN-03's factor set
# crosses augmentation *with* encoder regime, and why the mac/node runs of this notebook
# (both regimes × all three doses, multi-seed) are the arbiter rather than any prose.
#
# **Downstream contract.**
# - `trn03_augmentation_matrix` (+ `trn03_typo_dial`, `trn03_eval_noise_dial`, triptych
#   panels) carries the augmentation evidence; the node factorial sizes replicates from
#   `met04_power_table` (`seeds_needed`), per the §3 placard.
# - `trn04_encoder_head_to_head` carries the regime evidence with the pretrained absence
#   recorded in meta — notebook 12 and the cross-pressure study consume the regime
#   factor; the mac cell re-registers with both regimes live.
# - Verdicts here quoted notebook 09's bar rather than inventing one — notebooks 11+
#   should keep doing the same.
#
# **Artifacts registered** (exact names): `trn03_augmentation_matrix`,
# `trn04_encoder_head_to_head` — plus supporting `trn03_typo_dial`,
# `trn03_eval_noise_dial`, `trn03_panel_property`, `trn03_panel_geometry`,
# `trn03_panel_system`, and immutable cards `card_TRN-03`, `card_TRN-04`.

# %%
timing = pd.DataFrame(SECTION_TIMES, columns=["section", "secs"])
display(timing.assign(secs=timing["secs"].round(0)))
total = time.time() - NB_T0
print(f"notebook wall-clock: {total:.0f}s ({total / 60:.1f} min) — smoke budget <= ~30 min"
      + ("" if total <= 2100 else "  <-- OVER BUDGET at this run; see section table"))
