# %% [markdown]
# # 00. The Hardest Easy Problem
#
# **The question.** Deduplicating person records looks easy — humans do it at a glance, and a
# dozen lines of string matching gets you something that *looks* right. So why do census
# bureaus, health services, and voter-roll custodians run entire departments around it, and why
# does every national-scale system end up a carefully-guarded hybrid of rules, probabilities,
# and clerical review? This notebook shows the problem before the lab argues about solutions:
# real record groups where one person's strings barely overlap, and different people's records
# barely differ — then runs a complete, honest, five-minute dedup end-to-end so you can see
# both what cheap methods buy and exactly where they stop.
#
# **What this notebook settles.** Nothing statistical — deliberately. It is the hook and the
# calibration chapter: (1) it demonstrates, on `historical_50k`, why person dedup at scale is
# hard; (2) it ships a measured end-to-end quickstart (blocking → scoring → clustering →
# entity-level metrics with a real bootstrap CI), registered as `quickstart_dedup`; (3) it
# measures THIS machine's throughput coefficients and registers them as `smoke_check` — the
# only source any wall-clock claim in this repo is allowed to cite; and (4) it teaches you how
# to read the rest of the series (tiers, placards, conjecture cards, artifact-stamped figures).

# %%
import time

import jellyfish
import numpy as np
import pandas as pd
from IPython.display import display

from er_lab.blocking import matchkeys, sparse
from er_lab.cluster.schemes import transitive_closure
from er_lab.config import config_hash, load_config_from_env, set_all_seeds
from er_lab.data.loaders import load_historical_50k
from er_lab.eval.bootstrap import MULTIPLIER_METRICS, bootstrap_ci
from er_lab.eval.metrics import blocking_metrics
from er_lab.infra.artifacts import ArtifactRegistry
from er_lab.infra.device import describe_platform, resolve_device, resolve_precision
from er_lab.models.encoders import build_encoder
from er_lab.reporting import cards, figures
from er_lab.serialize import serialize_frame

cfg = load_config_from_env()
registry = ArtifactRegistry.from_env()
figures.setup_style()
set_all_seeds(cfg.run.seed)
rng = np.random.default_rng(cfg.run.seed)

# %%
# Tier banner: every notebook states what it ran as, on what, under which config.
print(f"tier        : {cfg.run.tier}")
print(f"config hash : {config_hash(cfg)}")
for key, val in describe_platform().items():
    print(f"{key:>13s} : {val}")

# %% [markdown]
# Tier-conditional sizes live here, at the top, so a reader can see exactly what scales and
# nothing else changes between tiers — one code path, different budgets (PLAN §2).

# %%
TIER = str(cfg.run.tier)
SIZES = {"smoke": 8_000, "mid": None, "target": None}  # records; None = the full corpus
ENC_TEXTS = {"smoke": 256, "mid": 2_048, "target": 8_192}  # encoder micro-benchmark batch
N_BOOT = {"smoke": 500, "mid": 1_000, "target": 2_000}  # bootstrap replicates
BM25_K = 5  # neighbors per record in the BM25 micro-benchmark
JW_THRESHOLD = 0.90  # quickstart link threshold — fixed a priori, never tuned (see §2)

n_target = SIZES.get(TIER, SIZES["smoke"])
n_enc = ENC_TEXTS.get(TIER, ENC_TEXTS["smoke"])
n_boot = N_BOOT.get(TIER, N_BOOT["smoke"])

# %% [markdown]
# ## 1. The hook: what do these records actually look like?
#
# `historical_50k` is a corpus of ~50k records describing ~5k real historical figures
# (Wikidata-derived, with injected errors — the splink teaching dataset). Each *cluster* of
# records describes one person; the `entity_id` column is the ground-truth key. It is tiny by
# this lab's standards, and it already contains every pathology that makes person ER hard.
# We load it verbatim — no cleaning, ever, at load time: the noise is the object of study.

# %%
full, schema = load_historical_50k(cfg.paths.data_root)
print(f"{len(full):,} records, {full['entity_id'].nunique():,} true entities")
print(f"text roles (declared schema order): {schema.text_roles()}")
display(full.head(4)[["record_id", "full_name", "given_name", "family_name", "dob", "city"]])

# %% [markdown]
# ### Same person, wildly different strings
#
# First pathology: one entity whose records barely share a token. We *compute* the worst
# offenders rather than cherry-picking by hand: for every entity we take its distinct
# `full_name` variants and score the largest token-set (Jaccard) distance between any two of
# them — 1.0 means the entity contains two name variants with **no tokens in common at all**.
# The selection below is a deterministic query against this run's data.

# %%
HOOK_COLS = ["full_name", "given_name", "family_name", "dob", "city", "zip", "entity_id"]


def name_spread(names: list[str]) -> float:
    """Largest pairwise token-set Jaccard distance among an entity's name variants."""
    toks = [frozenset(n.split()) for n in names]
    worst = 0.0
    for i in range(len(toks)):
        for j in range(i + 1, len(toks)):
            union = toks[i] | toks[j]
            worst = max(worst, 1.0 - len(toks[i] & toks[j]) / len(union))
    return worst


variants = full.dropna(subset=["full_name"]).groupby("entity_id")["full_name"].unique()
spread_tbl = pd.DataFrame(
    {
        "spread": variants.map(lambda v: name_spread(list(v))),
        "n_name_variants": variants.map(len),
        "n_records": full.groupby("entity_id").size(),
    }
).query("n_records >= 8")
spread_tbl = spread_tbl.reset_index().sort_values(
    ["spread", "n_name_variants", "n_records", "entity_id"],
    ascending=[False, False, False, True],
)
display(spread_tbl.head(3))

# %%
for eid in spread_tbl["entity_id"].head(2):
    print(f"--- all records of entity {eid} (one real person) ---")
    display(full.loc[full["entity_id"] == eid, HOOK_COLS].head(10))

# %% [markdown]
# Look at what a matcher is being asked to survive: nickname substitution, initials, surname
# replacement, titles absorbed into the name, digits corrupted inside the date of birth,
# fields simply gone. Every one of these is a *documented* real-world error channel (the lab
# measures their real prevalence in notebook 04); here they pile up inside a single identity.
#
# ### Different people, near-identical strings
#
# Second pathology — the mirror image. We compute the multi-token name (excluding bare
# title fragments like "sir … baronet") that is shared by the *most distinct true entities*,
# and show one record from each of the people who carry it.

# %%
named = full.dropna(subset=["full_name", "given_name", "family_name"])
plain = named[
    ~named["full_name"].str.contains(r"\b(?:sir|baronet|bt\.?|1st|2nd|3rd)\b", regex=True)
]
multi = plain[plain["full_name"].str.split().str.len() >= 2]
share = (
    multi.groupby("full_name")["entity_id"]
    .nunique()
    .rename("n_entities")
    .reset_index()
    .sort_values(["n_entities", "full_name"], ascending=[False, True])
)
top_name = share.iloc[0]["full_name"]
print(f"most entity-shared plain name: '{top_name}' " f"({share.iloc[0]['n_entities']} people)")
display(
    full.loc[full["full_name"] == top_name, HOOK_COLS].drop_duplicates("entity_id").head(6)
)

# %% [markdown]
# Same name, different people — sometimes in nearby places, sometimes with equally-vague
# birth dates. Nothing in these strings says "distinct"; only the surrounding evidence does.
#
# ### Hub values: the strings that glue everything together
#
# Third pathology: a handful of name strings are shared across *dozens* of entities. These
# are the "hub values" of the chain-merge literature — any method that trusts exact string
# equality inherits every hub as a ready-made false-merge highway.

# %%
hubs = (
    full.dropna(subset=["full_name"])
    .groupby("full_name")["entity_id"]
    .nunique()
    .rename("n_distinct_entities")
    .sort_values(ascending=False)
    .head(10)
)
display(hubs.to_frame())

# %% [markdown]
# ### Why this becomes brutal at 10⁹
#
# At 50k records these pathologies are curiosities; at national scale they are the whole
# problem. Three compounding mechanisms (lit review §1, §3, §5): **pair mass** — candidate
# pairs grow with n², so at 10⁹ records a blocker must discard all but ~one in a billion
# pairs while somehow keeping the true ones, and every rare error mode gets ~10⁹ chances per
# error rate to occur; **chain merges** — pairwise decisions are stitched into entities by
# transitive closure, so a single false link between two hub-heavy clusters fuses two real
# people irreversibly, and replicated production experience shows mega-components glued
# together exactly this way (shared households, placeholder values, frequent names);
# **name frequency** — person names are heavy-tailed, so "identical name string" stops being
# evidence precisely where the collisions concentrate (we just watched several distinct
# people share a name inside a corpus of *famous* historical figures; a national corpus
# holds tens of thousands of collisions per common name). Cheap signals saturate, and the
# question stops being "can you match strings" and becomes "can you budget errors" — which
# is why this lab is built around measurement and statistics rather than around a clever
# matcher.

# %% [markdown]
# ## 2. The five-minute end-to-end
#
# Can we dedup this corpus in one screenful of honest code — and what does "good" even mean?
# We run the naive-but-complete recipe every practitioner writes first:
#
# 1. **Block** with deterministic matchkeys (OR-union of cheap compound keys), because all
#    pairs is n² and already unaffordable at 50k.
# 2. **Score** each candidate pair with the mean Jaro-Winkler similarity over the
#    `given_name` / `family_name` / `dob` fields. This is a deliberately cheap strawman —
#    it is **NOT** the lab's Fellegi-Sunter baseline. The baseline that must actually be
#    beaten is a *well-tuned* Splink FS model (TF-adjusted, comparison-binned), built and
#    measured in notebook 06 (BAS-01). Nothing in this notebook is evidence about FS.
# 3. **Cluster** by transitive closure at a threshold fixed a priori at 0.90 — we never
#    tune a threshold to flatter a method (MET-02, notebook 02, demonstrates how per-method
#    best-F1 comparisons lie).
# 4. **Evaluate at the entity level** with a real bootstrap CI, resampled by entity —
#    records of one person share their fate, so record-level resampling would fake
#    independence (MET-03 demonstrates the coverage failure).
#
# The subsample is *entity-complete*: we draw whole entities until the record budget is
# reached, never splitting a person across the cut — a split entity would corrupt recall
# accounting. At smoke tier that is ~8k records; at mid/target the full corpus.

# %%
def entity_complete_subsample(
    frame: pd.DataFrame, n_records: int | None, rng: np.random.Generator
) -> pd.DataFrame:
    """Random whole entities until ~n_records — an entity is never split across the cut."""
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


sub = entity_complete_subsample(full, n_target, rng)
truth = sub.set_index("record_id")["entity_id"]
print(f"quickstart corpus: {len(sub):,} records, {truth.nunique():,} entities (tier={TIER})")

# %% [markdown]
# Before running anything we pre-register what we expect — this is the series' core habit.
# The card below is immutable: it is hashed and stored the first time this cell ever runs,
# and any later edit to its text is a protocol violation the registry itself will refuse.

# %%
CARD_ID = "NB00-quickstart"
PRED_PREC_MIN, PRED_REC_MAX, PRED_PC_MAX = 0.95, 0.70, 0.70
_card_md = cards.conjecture_card(
    card_id=CARD_ID,
    conjecture=(
        "On historical_50k, the naive quickstart (matchkey blocking, mean Jaro-Winkler "
        "field similarity, transitive closure) solves only the easy half of person dedup: "
        "entity-level precision stays high while entity-level recall is left far behind, "
        "capped by the pairs blocking never proposed."
    ),
    pressure=(
        "No learned pressure — the dial is the pipeline recipe itself: matchkeys OR-union "
        "blocking, mean Jaro-Winkler over given_name/family_name/dob, transitive closure "
        f"at threshold {JW_THRESHOLD:.2f} fixed a priori (never tuned per MET-02)."
    ),
    property=(
        "Not an embedding property in this notebook; the measured mediator is blocking "
        "pair-completeness, which upper-bounds all downstream recall — a pair never "
        "proposed can never be linked."
    ),
    metric=(
        "Entity-level B-cubed precision and recall on the tier-sized historical_50k "
        "subsample, with entity-unit BCa 95% bootstrap CIs."
    ),
    prediction=(
        f"B-cubed precision >= {PRED_PREC_MIN:.2f} and B-cubed recall <= {PRED_REC_MAX:.2f}; "
        f"blocking pair-completeness < {PRED_PC_MAX:.2f}, naming blocking as the binding "
        "constraint; precision's CI lower bound stays above recall's CI upper bound."
    ),
    registry=registry,
)

# %% [markdown]
# **Step 1 — blocking.** `matchkeys.candidates` derives cheap compound keys per record
# (soundex of the surname + birth year; postcode + first initial) and pairs records sharing
# a key, OR-union over passes — the design every national-scale system uses (~10 passes at
# the US Census, 29 matchkeys at ONS). It exists because comparing everything to everything
# is quadratic: even our subsample would otherwise mean tens of millions of comparisons.

# %%
t_pipeline = time.perf_counter()
t0 = time.perf_counter()
cand = matchkeys.candidates(sub)
t_block = time.perf_counter() - t0
print(f"{len(cand):,} candidate pairs from {len(sub):,} records in {t_block:.2f}s")
display(pd.DataFrame(cand.attrs["stats"]["passes"]))

# %% [markdown]
# How much did blocking throw away, and how much truth did it keep? Reduction ratio is the
# discarded fraction of all possible pairs; pair completeness is the fraction of true
# same-entity pairs that survived. The second number is the ceiling on everything downstream.

# %%
bstats = blocking_metrics(cand, truth, n_records=len(sub))
display(pd.DataFrame([bstats]))

# %% [markdown]
# **Step 2 — scoring.** Mean Jaro-Winkler similarity over `given_name`, `family_name`, and
# `dob`, averaged over the fields present in both records (a pair with no comparable field
# scores 0). Jaro-Winkler is the classical typo-tolerant name comparator; averaging it
# uniformly over three fields is exactly the kind of ad-hoc weighting Fellegi-Sunter
# replaces with estimated evidence weights — which is why this scorer is a strawman and
# notebook 06's tuned Splink model is the real baseline.

# %%
SCORE_FIELDS = ["given_name", "family_name", "dob"]


def jw_column(x: pd.Series, y: pd.Series) -> np.ndarray:
    """Jaro-Winkler per aligned row; NaN where either side is missing."""
    out = np.full(len(x), np.nan)
    for i, (u, v) in enumerate(zip(x, y)):
        if not (pd.isna(u) or pd.isna(v)):
            out[i] = jellyfish.jaro_winkler_similarity(str(u), str(v))
    return out


t0 = time.perf_counter()
lut = sub.set_index("record_id")[SCORE_FIELDS]
left = lut.loc[cand["a"]].reset_index(drop=True)
right = lut.loc[cand["b"]].reset_index(drop=True)
sim_mat = np.column_stack([jw_column(left[f], right[f]) for f in SCORE_FIELDS])
n_valid = (~np.isnan(sim_mat)).sum(axis=1)
sims = np.where(n_valid > 0, np.nansum(sim_mat, axis=1) / np.maximum(n_valid, 1), 0.0)
scored = pd.DataFrame({"a": cand["a"], "b": cand["b"], "prob": sims})
t_score = time.perf_counter() - t0
print(f"scored {len(scored):,} pairs in {t_score:.2f}s")
display(scored["prob"].describe().to_frame().T)

# %% [markdown]
# **Step 3 — clustering.** Transitive closure: link every pair at or above the threshold and
# take connected components. This is what naive pipelines do, and it is the chain-merge
# catastrophe's enabling mechanism — one bad edge fuses two people irreversibly. Notebook 07
# stages that catastrophe deliberately; here we just note the largest predicted cluster
# against the largest true entity.

# %%
t0 = time.perf_counter()
pred = transitive_closure(scored, threshold=JW_THRESHOLD, records=pd.Index(truth.index))
t_cluster = time.perf_counter() - t0
print(f"{pred.nunique():,} predicted clusters in {t_cluster:.2f}s")
print(f"largest predicted cluster: {int(pred.value_counts().iloc[0])} records")
print(f"largest true entity      : {int(truth.value_counts().iloc[0])} records")

# %% [markdown]
# **Step 4 — entity-level metrics, with a real CI.** B-cubed scores each *record* on how
# pure and complete its predicted cluster is, so one mega-merge costs what it should;
# pairwise P/R is reported alongside because most of the literature speaks it. The CI
# resamples *entities* (BCa, bias-corrected): records of one person are not independent
# evidence, and pretending otherwise is how evaluations under-report their own noise.

# %%
t0 = time.perf_counter()
ci_rows = []
for metric_name in MULTIPLIER_METRICS:
    res = bootstrap_ci(
        pred, truth, metric_name, unit="entity", n_boot=n_boot, seed=cfg.run.seed
    )
    ci_rows.append(
        {
            "metric": metric_name,
            "value": res["point"],
            "ci_low": res["ci_low"],
            "ci_high": res["ci_high"],
        }
    )
t_eval = time.perf_counter() - t0
runtime_s = time.perf_counter() - t_pipeline
print(f"bootstrap ({n_boot} replicates x {len(MULTIPLIER_METRICS)} metrics): {t_eval:.2f}s")
print(f"end-to-end pipeline wall-clock: {runtime_s:.2f}s")

# %% [markdown]
# Everything the quickstart measured becomes one registered artifact — `quickstart_dedup`,
# one row per metric — and the figure below renders *from the registry*, not from variables
# in memory. That indirection is the lab's honesty rail for figures (see §4).

# %%
qd = pd.DataFrame(
    ci_rows
    + [
        {"metric": "pair_completeness", "value": bstats["pair_completeness"]},
        {"metric": "reduction_ratio", "value": bstats["reduction_ratio"]},
        {"metric": "n_records", "value": float(len(sub))},
        {"metric": "n_entities", "value": float(truth.nunique())},
        {"metric": "n_candidate_pairs", "value": float(len(cand))},
        {"metric": "n_pred_clusters", "value": float(pred.nunique())},
        {"metric": "runtime_s", "value": runtime_s},
    ]
)
registry.register(
    "quickstart_dedup",
    qd,
    cfg=cfg,
    tier=cfg.run.tier,
    meta={
        "corpus": "historical_50k",
        "scorer": "mean Jaro-Winkler over given_name/family_name/dob — NOT the FS baseline "
        "(that is notebook 06 / BAS-01)",
        "blocking": "matchkeys.default_passes",
        "clustering": f"transitive_closure @ {JW_THRESHOLD:.2f} (fixed a priori)",
        "bootstrap": {"unit": "entity", "method": "bca", "n_boot": n_boot},
    },
)
display(qd)

# %%
POINT_ONLY = ["pair_completeness", "reduction_ratio"]


def draw_quickstart(ax, frame: pd.DataFrame, meta: dict) -> None:
    banded = frame[frame["ci_low"].notna()].reset_index(drop=True)
    points = frame[frame["metric"].isin(POINT_ONLY)].reset_index(drop=True)
    y_b = np.arange(len(banded))
    y_p = np.arange(len(banded), len(banded) + len(points))
    xerr = np.vstack(
        [
            np.clip(banded["value"] - banded["ci_low"], 0, None),
            np.clip(banded["ci_high"] - banded["value"], 0, None),
        ]
    )
    ax.errorbar(
        banded["value"], y_b, xerr=xerr, fmt="o", capsize=3, label="95% BCa CI (entity boot)"
    )
    ax.scatter(
        points["value"], y_p, marker="D", facecolors="none", color="C1", label="point (no CI)"
    )
    ax.set_yticks(np.concatenate([y_b, y_p]))
    ax.set_yticklabels(list(banded["metric"]) + list(points["metric"]))
    ax.set_xlim(0, 1.02)
    ax.set_xlabel("score")
    ax.invert_yaxis()
    ax.legend(loc="lower left")


_fig = figures.plot_artifact(
    registry,
    tier=cfg.run.tier,
    artifact="quickstart_dedup",
    draw=draw_quickstart,
    title="Quickstart dedup on historical_50k — cheap pipeline, honest error bars",
    figsize=(7.0, 4.2),
)

# %% [markdown]
# The verdict below is computed from the registered numbers, never hand-written — the three
# allowed outcomes are CONFIRMED / REFUTED / UNEXPLAINED, and UNEXPLAINED is reserved for
# "the headline held but not through the pre-named mechanism".

# %%
qd_ix = qd.set_index("metric")
prec, rec = qd_ix.loc["bcubed_precision"], qd_ix.loc["bcubed_recall"]
pc = float(qd_ix.loc["pair_completeness", "value"])
prec_ok = prec["value"] >= PRED_PREC_MIN
rec_ok = rec["value"] <= PRED_REC_MAX
order_ok = prec["ci_low"] > rec["ci_high"]
pc_ok = pc < PRED_PC_MAX
if prec_ok and rec_ok and order_ok and pc_ok:
    outcome = "CONFIRMED"
elif prec_ok and rec_ok and order_ok:
    outcome = "UNEXPLAINED"
else:
    outcome = "REFUTED"
_verdict_md = cards.verdict_box(
    CARD_ID,
    outcome=outcome,
    evidence=(
        f"B-cubed precision {prec['value']:.3f} [{prec['ci_low']:.3f}, {prec['ci_high']:.3f}] "
        f"vs recall {rec['value']:.3f} [{rec['ci_low']:.3f}, {rec['ci_high']:.3f}] "
        f"(entity-unit BCa, n_boot={n_boot}); blocking pair-completeness {pc:.3f} on "
        f"{len(sub):,} records / {truth.nunique():,} entities at threshold "
        f"{JW_THRESHOLD:.2f}. Artifact: quickstart_dedup (tier {TIER})."
    ),
    registry=registry,
)

# %% [markdown]
# Read the figure, not the vibes. The cheap pipeline is genuinely excellent at *not* merging
# different people — and it silently abandons a large share of each person's records,
# with the blocking stage's pair-completeness as the visible ceiling. "Easy" was precision.
# Recall — under noise, at scale, without a labeled answer key — is the hard part, and it is
# where the rest of this series lives.

# %% [markdown]
# ## 3. `smoke_check`: measuring THIS machine
#
# The lab's honesty rails forbid quoted wall-clock numbers: any runtime claim in the README
# or a later notebook must re-render from measured coefficients, and this section is where
# those coefficients come from. We time four primitive operations on this machine, at this
# tier, on real data from this run — and we project *nothing*. The registered `smoke_check`
# artifact is the citation target; PLAN §8's planning estimates are hypotheses these
# measurements exist to replace.

# %%
t0 = time.perf_counter()
texts = serialize_frame(sub, text_roles=schema.text_roles())
t_serialize = time.perf_counter() - t0
serialize_rows_per_s = len(sub) / t_serialize
print(f"serialize_frame: {len(sub):,} rows in {t_serialize:.2f}s")
print(f"example serialization:\n  {texts.iloc[0]}")

# %% [markdown]
# The encoder coefficient uses the lab's from-scratch byte-level transformer with UNTRAINED
# weights — this measures compute cost per record, and says nothing about quality (training
# starts in notebook 08). A tiny warm-up batch runs first so thread-pool spin-up is not
# billed to the coefficient. Device and precision come from config resolution, so the same
# cell measures CPU here and MPS/CUDA on the mac/node.

# %%
device = resolve_device(cfg)
dtype = resolve_precision(cfg, device)
encoder = build_encoder(cfg)
enc_params = int(sum(p.numel() for p in encoder.parameters()))
enc_texts = texts.iloc[:n_enc].tolist()
encoder.encode(enc_texts[:8], batch_size=8, device=device, dtype=dtype)  # warm-up
t0 = time.perf_counter()
emb = encoder.encode(enc_texts, batch_size=int(cfg.train.batch_size), device=device, dtype=dtype)
t_encode = time.perf_counter() - t0
encode_texts_per_s = len(enc_texts) / t_encode
print(
    f"CharByteEncoder (untrained, {enc_params:,} params, dim {emb.shape[1]}, "
    f"device {device.type}): {len(enc_texts)} texts in {t_encode:.2f}s"
)

# %%
t0 = time.perf_counter()
bm25_pairs = sparse.candidates(sub, texts, k=BM25_K)
t_bm25 = time.perf_counter() - t0
bm25_records_per_s = len(sub) / t_bm25
print(f"BM25 build+query (k={BM25_K}): {len(sub):,} records in {t_bm25:.2f}s")

t0 = time.perf_counter()
mk_pairs = matchkeys.candidates(sub)
t_matchkeys = time.perf_counter() - t0
matchkeys_pairs_per_s = len(mk_pairs) / t_matchkeys
print(f"matchkeys: {len(mk_pairs):,} pairs in {t_matchkeys:.2f}s")

# %%
smoke_check = {
    "platform": describe_platform(),
    "tier": TIER,
    "device": device.type,
    "measured": {
        "serialize_rows_per_s": float(serialize_rows_per_s),
        "serialize_n_rows": len(sub),
        "charbyte_encode_texts_per_s": float(encode_texts_per_s),
        "charbyte_encode_n_texts": len(enc_texts),
        "charbyte_encode_batch_size": int(cfg.train.batch_size),
        "charbyte_params": enc_params,
        "charbyte_dim": int(emb.shape[1]),
        "charbyte_untrained": True,
        "bm25_build_query_records_per_s": float(bm25_records_per_s),
        "bm25_n_records": len(sub),
        "bm25_k": int(BM25_K),
        "bm25_n_pairs": len(bm25_pairs),
        "matchkeys_pairs_per_s": float(matchkeys_pairs_per_s),
        "matchkeys_n_pairs": len(mk_pairs),
        "matchkeys_n_records": len(sub),
        "quickstart_runtime_s": float(runtime_s),
    },
}
registry.register(
    "smoke_check",
    smoke_check,
    cfg=cfg,
    tier=cfg.run.tier,
    meta={"note": "measured throughput coefficients; the only citable wall-clock source"},
)
coeff_table = pd.DataFrame(
    [
        ("serialize_frame", serialize_rows_per_s, "rows/s", len(sub)),
        ("CharByteEncoder.encode (untrained)", encode_texts_per_s, "texts/s", len(enc_texts)),
        (f"BM25 build+query (k={BM25_K})", bm25_records_per_s, "records/s", len(sub)),
        ("matchkeys blocking", matchkeys_pairs_per_s, "pairs/s", len(mk_pairs)),
        ("quickstart end-to-end", runtime_s, "s wall-clock", len(sub)),
    ],
    columns=["coefficient", "value", "unit", "n_measured"],
)
display(coeff_table)
print(
    f"Measured on this machine, this run (tier={TIER}, device={device.type}). These are\n"
    "coefficients, not projections: later notebooks and the README cite the registered\n"
    "smoke_check artifact instead of quoting wall-clock folklore. Nothing on this table\n"
    "says anything about 1e9 — the cost model that does (SCL-03, notebook 15) is built\n"
    "from measured coefficients like these and is labeled EXTRAPOLATED wherever it leaves\n"
    "measurement behind."
)

# %% [markdown]
# ## 4. How to read this series
#
# **Tiers.** Every notebook runs at `smoke` (a 4-CPU container, minutes, subsampled data),
# `mid` (the lab's M2 Max, ~1e6 records), or `target` (one 4xA100 node, ~1e7) — the tier
# only ever arrives through config (`run.tier`), never through forked notebook logic, so the
# cell you read is the cell that runs at every scale. Sizes and budgets sit in
# tier-conditional constants at the top of each notebook (like `SIZES` above); `analytical`
# marks work that is pure computation over registered measurements. Artifacts from
# different tiers never mix: the registry refuses to serve a smoke artifact to a target run.
#
# **RUN-IN-TARGET placards.** A cell whose real run only makes sense on the mac or the node
# carries a `# [RUN-IN-TARGET mac]` (or `node`) comment and, at smoke, prints a placard
# saying what it computes and where, instead of pretending. This notebook has none — the
# quickstart's entire point is running end-to-end anywhere — but from notebook 06 onward you
# will meet cells that print, e.g.:
#
# ```
# [RUN-IN-TARGET mac] this cell fine-tunes the pretrained encoder at tier=mid;
# at smoke it is skipped.
# ```
#
# Same cell, same code path — only the tier changes.
#
# **Conjecture cards.** Every claim-bearing computation is preceded by a pre-registered
# conjecture (pressure → measurable property → system metric, with a directional
# prediction), rendered and hash-frozen *before* the results exist, and closed by a verdict
# box that must honestly read CONFIRMED, REFUTED, or UNEXPLAINED off the computed numbers.
# Cards are immutable — re-rendering an edited card raises an error by design — so a verdict
# can never quietly rewrite what was predicted. UNEXPLAINED is a first-class outcome: an
# effect can be real while its claimed mechanism is not.
#
# **Artifact-stamped figures.** No figure in this series renders from an in-memory frame.
# Data is first registered (name, tier, config hash, git revision, timestamp), and figures
# load it back through `er_lab.reporting.figures`, which stamps every plot with its artifact
# name, config hash, tier, and a MEASURED / EXTRAPOLATED flag — extrapolated segments render
# dashed, shaded, and watermarked. If a figure exists, its data provably came from a
# registered run of a named notebook; if the upstream notebook has not run, you get a
# "NOT YET RUN" placard instead of a picture.

# %% [markdown]
# ## What we now know / What this changes downstream
#
# **What we now know.** On real person records, the failure modes come in mirrored pairs:
# one person's strings can share almost nothing while two people's records are
# near-identical, and a few hub name-values are shared across dozens of true entities. A
# complete cheap pipeline (matchkeys → mean Jaro-Winkler → transitive closure) runs on this
# corpus in well under the five-minute budget and is precise but recall-starved, with
# blocking pair-completeness the measured ceiling — exact numbers, with entity-bootstrap
# CIs, in `quickstart_dedup` and stamped on the figure above. This machine's measured
# throughput coefficients (serialization, untrained byte-encoder encode, BM25 build+query,
# matchkey pair generation) are registered in `smoke_check`.
#
# **What this changes downstream.** `smoke_check` becomes the only legitimate source for
# wall-clock and cost statements (the README quickstart claim re-renders from it; SCL-03's
# 1e9 cost model starts from coefficients measured like these). `quickstart_dedup` is the
# strawman reference point later notebooks improve on — and the metrics discipline it
# previews (entity-level scores, entity-unit bootstrap, fixed a-priori operating points) is
# built properly in notebook 02 before any model is trained. The scorer used here is
# explicitly not the baseline; notebook 06 (BAS-01) builds the Fellegi-Sunter model that
# embeddings must actually beat.
#
# **Artifacts registered by this notebook:** `smoke_check`, `quickstart_dedup`.
